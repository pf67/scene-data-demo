"""Occlusion-aware dominant-view facade projection, CC0 PBR weathering baker & Blender material linker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:
    cv2 = None  # type: ignore
    Image = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

from bridge_client import run_blender_script  # noqa: E402


TILE_WORLD_METERS: dict[str, float] = {
    "plaster_white": 4.4,
    "plaster_warm": 4.4,
    "plaster_grey": 4.4,
    "red_brick": 3.0,
    "concrete_roof": 4.8,
    "corrugated_silver": 1.15,
    "corrugated_blue": 1.15,
    "terracotta_tile": 1.85,
    "slate_roof": 1.85,
}

BUILDING_SWATCH_KEYS = tuple(TILE_WORLD_METERS.keys())


def sample_tiled_swatch(
    swatch_img: np.ndarray,
    u_meters: np.ndarray,
    v_meters: np.ndarray,
    tile_meters: float,
) -> np.ndarray:
    """Bilinearly sample a tileable swatch texture using exact metric coordinates."""
    sh, sw = swatch_img.shape[:2]
    px = np.mod(u_meters / tile_meters * sw, sw).astype(np.float32)
    py = np.mod(v_meters / tile_meters * sh, sh).astype(np.float32)
    return cv2.remap(swatch_img, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def apply_uv_gutter_padding(atlas_img: np.ndarray, mask: np.ndarray, rect_px: list[int], pad_px: int = 5) -> None:
    """Dilate valid polygon interior pixels into surrounding UV gutters inside rect_px."""
    rx0, ry0, rx1, ry1 = rect_px
    ah, aw = atlas_img.shape[:2]
    px0 = max(0, rx0 - pad_px)
    py0 = max(0, ry0 - pad_px)
    px1 = min(aw, rx1 + pad_px)
    py1 = min(ah, ry1 + pad_px)
    sub = atlas_img[py0:py1, px0:px1]
    sub_mask = mask[py0:py1, px0:px1]
    if sub_mask.all() or not sub_mask.any():
        return
    cur = sub.copy()
    valid = sub_mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(pad_px):
        dil = cv2.dilate(cur, kernel)
        new_valid = cv2.dilate(valid.astype(np.uint8), kernel) > 0
        fill = new_valid & (~valid)
        cur[fill] = dil[fill]
        valid = new_valid
    sub[:] = cur


def _weathering_multiplier(u_m: np.ndarray, v_m: np.ndarray, v_len_m: float, seed: int) -> np.ndarray:
    """Compute subtle top-down rain-drip and bottom-up warm dust/damp RGB multiplier."""
    rh, rw = u_m.shape
    phase = (seed * 1.37) % 10.0
    # Gentle vertical rain-drip streaks beneath roof parapet / eave (<= 11% darkening)
    streaks = (
        0.5
        + 0.25 * np.sin(u_m * 3.1 + phase)
        + 0.15 * np.cos(u_m * 7.4 + phase * 1.7)
        + 0.10 * np.sin(u_m * 15.2 - phase)
    )
    top_drip = 1.0 - 0.11 * streaks * np.exp(-np.maximum(v_m, 0.0) / 0.75)
    # Bottom plinth splash: gentle 8% darkening + slightly warmer ochre hue bias near ground
    dist_ground = np.maximum(v_len_m - v_m, 0.0)
    bot_profile = (0.65 + 0.35 * np.sin(u_m * 2.4 + phase)) * np.exp(-dist_ground / 0.45)
    bot_luma = 1.0 - 0.08 * bot_profile
    warm_bias = np.stack(
        [1.0 + 0.04 * bot_profile, 1.0 + 0.01 * bot_profile, 1.0 - 0.04 * bot_profile],
        axis=-1,
    )
    h_u = np.linspace(-1.0, 1.0, rw, dtype=np.float32)[None, :]
    corner_fade = 1.0 - 0.03 * (h_u ** 4)
    mul_rgb = (top_drip * bot_luma * corner_fade)[:, :, None] * warm_bias
    return np.clip(mul_rgb, 0.82, 1.04).astype(np.float32)


def _apply_wall_weathering(comp: np.ndarray, u_m: np.ndarray, v_m: np.ndarray, v_len_m: float, seed: int) -> np.ndarray:
    """Apply subtle RGB wall weathering plus warm earthy plinth dust splash without muddy charcoal darkening."""
    mul_rgb = _weathering_multiplier(u_m, v_m, v_len_m, seed=seed)
    dist_ground = np.maximum(v_len_m - v_m, 0.0)
    phase = (seed * 1.37) % 10.0
    dust_w = (0.16 * (0.65 + 0.35 * np.cos(u_m * 2.7 + phase)) * np.exp(-dist_ground / 0.42))[:, :, None]
    dust_rgb = np.array([178.0, 162.0, 138.0], dtype=np.float32)[None, None, :]
    out = comp * mul_rgb
    return out * (1.0 - dust_w) + dust_rgb * dust_w


def composite_novel_facade(
    fallback_alb: np.ndarray,
    obs_rgb: np.ndarray,
    obs_alpha: np.ndarray,
    face_mask: np.ndarray,
    interior: np.ndarray,
    tg: np.ndarray,
    ao_up: np.ndarray,
    u_m: np.ndarray,
    v_m: np.ndarray,
    v_len_m: float,
    seed: int,
) -> tuple[np.ndarray, bool]:
    """Composite synthetic novel-view wall observation with outlier rejection & 2K PBR grain preservation."""
    luma_o = obs_rgb.mean(axis=2)
    green_ex_o = 2.0 * obs_rgb[:, :, 1] - obs_rgb[:, :, 0] - obs_rgb[:, :, 2]
    slate_roof_spill = (obs_rgb[:, :, 2] - obs_rgb[:, :, 0] > 6.0) & (luma_o < 135.0)
    color_jump = np.abs(obs_rgb - fallback_alb).mean(axis=2) > 58.0
    wall_ok = (luma_o >= 64.0) & (luma_o <= 238.0) & (green_ex_o <= 12.0) & (~slate_roof_spill) & (~color_jump)

    if not interior.any() or float(wall_ok[interior].mean()) < 0.45:
        fallback = _apply_wall_weathering(fallback_alb, u_m, v_m, v_len_m, seed=seed) * (0.60 + 0.40 * ao_up[:, :, None])
        return fallback, False

    fb_low = cv2.GaussianBlur(fallback_alb, (21, 21), 6.0)
    hf_ratio = np.clip(fallback_alb / np.maximum(fb_low, 16.0), 0.78, 1.25)
    obs_safe = np.where(wall_ok[:, :, None], obs_rgb, fb_low)
    obs_low = cv2.GaussianBlur(obs_safe, (21, 21), 6.0)

    eroded = cv2.erode(face_mask, np.ones((7, 7), np.uint8))
    edge_soft = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, (9, 9), 2.5)
    v_trim = np.clip((tg - 0.015) / 0.045, 0.0, 1.0) * np.clip((0.975 - tg) / 0.045, 0.0, 1.0)
    ok_soft = cv2.GaussianBlur(wall_ok.astype(np.float32), (11, 11), 3.0)
    novel_w = np.clip(obs_alpha * ok_soft * (0.25 + 0.75 * edge_soft) * v_trim.astype(np.float32), 0.0, 0.56)[:, :, None]

    base_low = novel_w * obs_low + (1.0 - novel_w) * fb_low
    tinted = np.clip(base_low * hf_ratio, 18.0, 244.0)
    comp = _apply_wall_weathering(tinted, u_m, v_m, v_len_m, seed=seed) * (0.64 + 0.36 * ao_up[:, :, None])
    return comp, True


def bake_atlases(
    *,
    spec_path: Path,
    uv_map_path: Path,
    probes_path: Path,
    out_dir: Path,
    blend_path: Path,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    uv_map = json.loads(uv_map_path.read_text(encoding="utf-8"))
    probes = np.load(probes_path)
    vis_all = probes["vis"]
    ao_all = probes["ao"]

    atlas_size = int(uv_map["atlas_size"])
    num_atlases = int(uv_map["atlases"])
    faces = uv_map["faces"]
    cameras = spec["cameras"]

    cam_images: list[np.ndarray] = []
    cam_r: list[np.ndarray] = []
    cam_t: list[np.ndarray] = []
    cam_eye: list[np.ndarray] = []
    cam_k: list[tuple[float, float, float, float, int, int]] = []
    cam_pri: list[float] = []
    cam_wm: list[list[float]] = []
    cam_is_ref: list[bool] = []
    cam_is_clip: list[bool] = []
    cam_allow_facade: list[bool] = []

    ref_cam_idx = 0
    for idx, c in enumerate(cameras):
        bgr = cv2.imread(c["image_path"])
        if bgr is None:
            raise FileNotFoundError(f"Camera image missing: {c['image_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * np.asarray(c["gain_rgb"], dtype=np.float32)
        np.clip(rgb, 0.0, 255.0, out=rgb)
        cam_images.append(rgb)
        r = np.asarray(c["r_cw"], dtype=np.float64)
        t = np.asarray(c["t_cw"], dtype=np.float64)
        cam_r.append(r)
        cam_t.append(t)
        cam_eye.append(-r.T @ t)
        cam_k.append((float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"]), int(c["width"]), int(c["height"])))
        cam_pri.append(float(c["priority"]))
        cam_wm.append(c["watermark_box"])
        src = str(c["source_clip"])
        is_ref_src = src == "ref"
        is_extra_src = src.startswith("extra_")
        cam_is_ref.append(is_ref_src)
        cam_is_clip.append((not is_ref_src) and (not is_extra_src))
        cam_allow_facade.append(bool(c.get("allow_facade_bake", False)))
        if is_ref_src:
            ref_cam_idx = idx
    has_allow_facade = any(cam_allow_facade)

    swatches: dict[str, dict[str, np.ndarray]] = {}
    for sname in BUILDING_SWATCH_KEYS:
        spaths = spec["swatches"].get(sname)
        if not spaths:
            continue
        swatches[sname] = {
            "albedo": np.asarray(Image.open(spaths["albedo"]).convert("RGB"), dtype=np.float32),
            "normal": np.asarray(Image.open(spaths["normal"]).convert("RGB"), dtype=np.float32),
            "roughness": np.asarray(Image.open(spaths["roughness"]).convert("L"), dtype=np.float32),
        }

    albedos = [np.full((atlas_size, atlas_size, 3), 156, dtype=np.uint8) for _ in range(num_atlases)]
    normals = [np.full((atlas_size, atlas_size, 3), (128, 128, 255), dtype=np.uint8) for _ in range(num_atlases)]
    roughs = [np.full((atlas_size, atlas_size), 216, dtype=np.uint8) for _ in range(num_atlases)]
    masks = [np.zeros((atlas_size, atlas_size), dtype=bool) for _ in range(num_atlases)]

    total_obs_weight = 0.0
    total_faces = 0
    photo_facade_faces = 0

    for fi, face in enumerate(faces):
        aidx = int(face["atlas_idx"])
        rx0, ry0, rx1, ry1 = face["rect_px"]
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw <= 0 or rh <= 0:
            continue

        origin = np.asarray(face["origin_3d"], dtype=np.float64)
        u_span = np.asarray(face["u_span_3d"], dtype=np.float64)
        v_span = np.asarray(face["v_span_3d"], dtype=np.float64)
        nrm = np.asarray(face["normal_world"], dtype=np.float64)
        u_len_m = float(np.linalg.norm(u_span))
        v_len_m = float(np.linalg.norm(v_span))

        st_poly = np.asarray(face["st_poly"], dtype=np.float64)
        poly_px = np.round(st_poly * np.array([rw, rh])).astype(np.int32)
        face_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(face_mask, [poly_px], 255)
        if not face_mask.any():
            face_mask.fill(255)
        interior = face_mask > 0

        if face["role"] == "bottom":
            albedos[aidx][ry0:ry1, rx0:rx1] = (114, 110, 102)
            roughs[aidx][ry0:ry1, rx0:rx1] = 238
            masks[aidx][ry0:ry1, rx0:rx1] = True
            apply_uv_gutter_padding(albedos[aidx], masks[aidx], face["rect_px"], pad_px=4)
            continue

        xs = (np.arange(rw, dtype=np.float64) + 0.5) / rw
        ys = (np.arange(rh, dtype=np.float64) + 0.5) / rh
        sg, tg = np.meshgrid(xs, ys)
        world_xyz = origin + sg[:, :, None] * u_span + tg[:, :, None] * v_span

        sw_name = face["swatch"]
        sw_set = swatches.get(sw_name, swatches["plaster_white"])
        tile_m = TILE_WORLD_METERS.get(sw_name, 4.4)
        u_m = (sg * u_len_m).astype(np.float32)
        v_m = (tg * v_len_m).astype(np.float32)
        sw_alb = sample_tiled_swatch(sw_set["albedo"], u_m, v_m, tile_m)
        sw_nrm = sample_tiled_swatch(sw_set["normal"], u_m, v_m, tile_m)
        sw_rgh = sample_tiled_swatch(sw_set["roughness"], u_m, v_m, tile_m)

        is_roof = face["role"] == "roof"
        is_core_face = bool(face.get("is_core", True))
        bake_mode = face.get("bake_mode", "pbr_procedural")

        obs_tint = np.asarray(face["observed_rgb"], dtype=np.float32)
        sw_mean = np.maximum(sw_alb.mean(axis=(0, 1)), 1.0)
        if is_core_face and is_roof:
            tint_ratio = np.clip(obs_tint / sw_mean, 0.70, 1.40)
            tint_gain = 0.68
        elif is_core_face and not is_roof:
            tint_ratio = np.clip(obs_tint / sw_mean, 0.76, 1.32)
            tint_gain = 0.52
        else:
            tint_ratio = np.clip(obs_tint / sw_mean, 0.82, 1.20)
            tint_gain = 0.24 if not sw_name.startswith("red_brick") else 0.12
        # Deterministic per-building brightness & warmth variation (+-5% / +-3%)
        b_seed = sum(ord(ch) for ch in str(face["building_id"])) + 13
        b_tone = 1.0 + (((b_seed * 29) % 11) - 5) * 0.010
        b_warm = (((b_seed * 43) % 7) - 3) * 0.010
        b_shift = np.array([b_tone * (1.0 + b_warm), b_tone, b_tone * (1.0 - b_warm)], dtype=np.float32)
        fallback_alb = np.clip(sw_alb * ((1.0 - tint_gain) + tint_gain * tint_ratio) * b_shift, 16.0, 244.0)

        vis_up = cv2.resize(vis_all[fi], (rw, rh), interpolation=cv2.INTER_LINEAR)
        if vis_up.ndim == 2:
            vis_up = vis_up[:, :, None]
        ao_up = cv2.resize(ao_all[fi], (rw, rh), interpolation=cv2.INTER_LINEAR)

        wall_min_cos = float(spec.get("photo_facade_min_cos", 0.66))
        ref_dist_scale = float(spec.get("photo_facade_ref_dist", 58.0))
        max_dist_clip = max(220.0, ref_dist_scale * 4.0)
        should_sample_cams = bake_mode in ("photo_facade", "novel_facade")
        min_cos = 0.34 if is_roof else wall_min_cos
        min_vis = 0.40 if is_roof else 0.75

        pts_flat = world_xyz.reshape(-1, 3)

        # Evaluate CAM_ref for photo_facade or novel synthetic cams for novel_facade
        best_score = np.zeros((rh, rw), dtype=np.float32)
        best_rgb = fallback_alb.copy()
        samples_cache: list[tuple[np.ndarray, np.ndarray]] = []

        if should_sample_cams:
            for ci in range(len(cameras)):
                if bake_mode == "photo_facade" and not cam_is_ref[ci]:
                    continue
                if bake_mode == "novel_facade" and not cam_allow_facade[ci]:
                    continue
                v_map = vis_up[:, :, ci]
                if v_map.max() < min_vis:
                    continue
                fxi, fyi, cxi, cyi, wi, hi = cam_k[ci]
                xci = pts_flat @ cam_r[ci].T + cam_t[ci]
                zi = xci[:, 2].reshape(rh, rw)
                ui = (xci[:, 0] / np.maximum(xci[:, 2], 1e-4) * fxi + cxi).reshape(rh, rw).astype(np.float32)
                vi = (xci[:, 1] / np.maximum(xci[:, 2], 1e-4) * fyi + cyi).reshape(rh, rw).astype(np.float32)
                d_vec = cam_eye[ci] - pts_flat
                dist_i = np.linalg.norm(d_vec, axis=1).reshape(rh, rw)
                cos_i = ((d_vec @ nrm).reshape(rh, rw)) / np.maximum(dist_i, 1e-4)
                wmi = cam_wm[ci]
                bpad = 48.0 if bake_mode == "novel_facade" else 6.0
                in_i = (zi > 2.0) & (ui >= bpad) & (ui < wi - bpad) & (vi >= bpad) & (vi < hi - bpad) & (~((ui > wmi[0] * wi) & (vi > wmi[1] * hi)))
                valid_i = in_i & (cos_i > min_cos) & (v_map > min_vis)
                if not valid_i.any():
                    continue

                sample_i = cv2.remap(cam_images[ci], ui, vi, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                score_i = np.where(
                    valid_i,
                    v_map * (np.clip(cos_i, 0.0, 1.0) ** 2.2) * cam_pri[ci] * (ref_dist_scale / np.clip(dist_i, 10.0, max_dist_clip)) ** 1.18,
                    0.0,
                ).astype(np.float32)
                if ci == ref_cam_idx:
                    score_i *= 2.2 if has_allow_facade else 1.45

                better = score_i > best_score
                best_score[better] = score_i[better]
                best_rgb[better] = sample_i[better]
                samples_cache.append((sample_i, score_i))

        # Consensus-gated refinement over top view (eliminates ghosting while denoising multi-frame variance)
        if samples_cache:
            acc_rgb = best_rgb * (best_score * 1.4)[:, :, None]
            acc_w = best_score * 1.4
            for sample_i, score_i in samples_cache:
                agree = (score_i > 0.05) & (np.abs(sample_i - best_rgb).mean(axis=2) < 26.0)
                w_add = np.where(agree, score_i, 0.0).astype(np.float32)
                acc_rgb += sample_i * w_add[:, :, None]
                acc_w += w_add
            obs_rgb = np.where((acc_w > 1e-4)[:, :, None], acc_rgb / np.maximum(acc_w[:, :, None], 1e-5), fallback_alb)
        else:
            obs_rgb = fallback_alb

        obs_alpha = np.clip(best_score / 0.55, 0.0, 1.0)
        rgh_out = sw_rgh.copy()

        if is_roof:
            if sw_name in ("terracotta_tile", "slate_roof", "corrugated_blue", "corrugated_silver"):
                # Preserve 100% crisp 2K Poly Haven roof scans (zero oblique video cars/roads overwritten!)
                d_eave = np.minimum(v_m, np.maximum(v_len_m - v_m, 0.0))
                eave_dust = 1.0 - 0.08 * np.exp(-np.maximum(d_eave, 0.0) / 0.55)
                sun_bleach = 1.0 + 0.04 * np.clip(1.0 - np.abs(v_m / max(v_len_m, 1e-4) * 2.0 - 1.0), 0.0, 1.0)
                comp = fallback_alb * (eave_dust * sun_bleach)[:, :, None] * (0.64 + 0.36 * ao_up[:, :, None])
            else:
                # Flat concrete roof: frequency-separation fusion + parapet perimeter damp/soot pooling
                fb_low = cv2.GaussianBlur(fallback_alb, (31, 31), 9.0)
                hf_ratio = np.clip(fallback_alb / np.maximum(fb_low, 16.0), 0.72, 1.32)
                if is_core_face and samples_cache:
                    luma_o = obs_rgb.mean(axis=2)
                    green_ex = 2.0 * obs_rgb[:, :, 1] - obs_rgb[:, :, 0] - obs_rgb[:, :, 2]
                    chroma_o = obs_rgb.max(axis=2) - obs_rgb.min(axis=2)
                    sem_ok = (luma_o > 80.0) & (luma_o < 222.0) & (green_ex < 10.0) & (chroma_o < 24.0)
                    if sem_ok.any():
                        med_luma = float(np.median(luma_o[sem_ok]))
                        sem_ok &= np.abs(luma_o - med_luma) < 38.0
                    obs_w = cv2.GaussianBlur((obs_alpha * sem_ok.astype(np.float32)), (19, 19), 5.0) * 0.36
                    obs_safe = np.where(sem_ok[:, :, None], obs_rgb, fb_low)
                    obs_low = cv2.GaussianBlur(obs_safe, (25, 25), 7.0)
                    base_low = obs_w[:, :, None] * obs_low + (1.0 - obs_w[:, :, None]) * fb_low
                    comp = np.clip(base_low * hf_ratio, 18.0, 244.0)
                else:
                    comp = fallback_alb.copy()
                d_edge = np.minimum(
                    np.minimum(u_m, np.maximum(u_len_m - u_m, 0.0)),
                    np.minimum(v_m, np.maximum(v_len_m - v_m, 0.0)),
                )
                streak_n = 0.65 + 0.35 * np.sin(u_m * 2.1 + v_m * 1.9 + fi * 0.7)
                damp_band = np.exp(-np.maximum(d_edge, 0.0) / 0.65) * streak_n
                dry_center = np.clip((d_edge - 0.85) / 1.2, 0.0, 1.0)
                comp = comp * (1.0 - 0.16 * damp_band + 0.04 * dry_center)[:, :, None] * (0.62 + 0.38 * ao_up[:, :, None])
                rgh_out = np.clip(sw_rgh * (1.0 - 0.24 * damp_band) + 164.0 * (0.24 * damp_band) + 8.0 * dry_center, 138.0, 244.0)
            bump_gain = np.zeros_like(obs_alpha)
        elif bake_mode == "photo_facade" and obs_alpha[interior].mean() > 0.30:
            # Feather top/bottom/side borders cleanly into weathered plaster so roof/sky never bleeds
            eroded = cv2.erode(face_mask, np.ones((7, 7), np.uint8))
            edge_soft = cv2.GaussianBlur(eroded.astype(np.float32) / 255.0, (9, 9), 2.5)
            v_trim = np.clip((tg - 0.015) / 0.045, 0.0, 1.0) * np.clip((0.975 - tg) / 0.045, 0.0, 1.0)
            photo_w = (obs_alpha * (0.25 + 0.75 * edge_soft) * v_trim.astype(np.float32))[:, :, None]
            photo_eq = np.clip(obs_rgb * 0.96 + 6.0, 16.0, 246.0)
            comp = photo_w * photo_eq + (1.0 - photo_w) * fallback_alb
            dist_ground = np.maximum(v_len_m - v_m, 0.0)
            dust_w = (0.12 * np.exp(-dist_ground / 0.45))[:, :, None]
            dust_rgb = np.array([178.0, 162.0, 138.0], dtype=np.float32)[None, None, :]
            comp = (comp * (1.0 - 0.06 * np.exp(-dist_ground / 0.45))[:, :, None]) * (1.0 - dust_w) + dust_rgb * dust_w
            comp *= (0.80 + 0.20 * ao_up[:, :, None])
            luma_p = comp.mean(axis=2)
            glass_mask = np.clip((92.0 - luma_p) / 40.0, 0.0, 1.0) * (obs_alpha * edge_soft * v_trim.astype(np.float32))
            rgh_out = np.clip(sw_rgh * (1.0 - glass_mask * 0.72) + 58.0 * glass_mask * 0.72, 48.0, 248.0)
            bump_gain = np.zeros_like(obs_alpha)
            photo_facade_faces += 1
        elif bake_mode == "novel_facade" and obs_alpha[interior].mean() > 0.25:
            comp, novel_ok = composite_novel_facade(
                fallback_alb, obs_rgb, obs_alpha, face_mask, interior, tg, ao_up, u_m, v_m, v_len_m, seed=fi + 7
            )
            bump_gain = np.zeros_like(obs_alpha)
            if novel_ok:
                photo_facade_faces += 1
        else:
            # Unobserved side/rear/infill wall: crisp 2K Poly Haven plaster/brick + subtle weathering + natural crevice AO
            comp = _apply_wall_weathering(fallback_alb, u_m, v_m, v_len_m, seed=fi + 7) * (0.60 + 0.40 * ao_up[:, :, None])
            bump_gain = np.zeros_like(obs_alpha)

        comp_u8 = np.clip(comp, 0.0, 255.0).astype(np.uint8)
        nrm_u8 = np.clip(sw_nrm, 0, 255).astype(np.uint8)
        rgh_u8 = np.clip(rgh_out, 42, 248).astype(np.uint8)

        albedos[aidx][ry0:ry1, rx0:rx1][interior] = comp_u8[interior]
        normals[aidx][ry0:ry1, rx0:rx1][interior] = nrm_u8[interior]
        roughs[aidx][ry0:ry1, rx0:rx1][interior] = rgh_u8[interior]
        masks[aidx][ry0:ry1, rx0:rx1] |= interior

        pad_px = 5 if face.get("is_core", True) else 4
        apply_uv_gutter_padding(albedos[aidx], masks[aidx], face["rect_px"], pad_px=pad_px)
        apply_uv_gutter_padding(normals[aidx], masks[aidx], face["rect_px"], pad_px=pad_px)
        apply_uv_gutter_padding(roughs[aidx], masks[aidx], face["rect_px"], pad_px=pad_px)

        total_obs_weight += float(obs_alpha[interior].mean())
        total_faces += 1

    atlases_dir = out_dir / "atlases"
    atlases_dir.mkdir(parents=True, exist_ok=True)
    # Remove any stale extra atlases from previous runs
    for old in atlases_dir.glob("atlas_*.*"):
        old.unlink(missing_ok=True)

    core_atlas_count = max((int(f["atlas_idx"]) for f in faces if f.get("is_core", True)), default=3) + 1
    atlas_paths = []
    for i in range(num_atlases):
        ap = atlases_dir / f"atlas_{i}_albedo.jpg"
        np_p = atlases_dir / f"atlas_{i}_normal.jpg"
        rp = atlases_dir / f"atlas_{i}_roughness.jpg"
        is_core_atlas = i < core_atlas_count
        alb_q = 88 if is_core_atlas else 85
        nrm_sz, nrm_q = (1536, 85) if is_core_atlas else (1024, 84)
        rgh_sz = 768 if is_core_atlas else 512

        Image.fromarray(albedos[i]).save(ap, format="JPEG", quality=alb_q)
        nrm_resized = cv2.resize(normals[i], (nrm_sz, nrm_sz), interpolation=cv2.INTER_AREA)
        Image.fromarray(nrm_resized).save(np_p, format="JPEG", quality=nrm_q)
        rgh_resized = cv2.resize(roughs[i], (rgh_sz, rgh_sz), interpolation=cv2.INTER_AREA)
        rgh_q = ((cv2.GaussianBlur(rgh_resized, (3, 3), 0.7) // 8) * 8).astype(np.uint8)
        Image.fromarray(rgh_q).save(rp, format="JPEG", quality=82)
        atlas_paths.append({"albedo": str(ap), "normal": str(np_p), "roughness": str(rp)})

    manifest_path = atlases_dir / "atlas_manifest.json"
    manifest_path.write_text(json.dumps(atlas_paths, indent=2), encoding="utf-8")

    run_blender_script(
        Path(__file__).resolve(),
        ["--apply-materials", "--manifest", str(manifest_path), "--blend", str(blend_path)],
        blend_path=blend_path,
    )

    summary = {
        "atlases": atlas_paths,
        "faces_baked": total_faces,
        "photo_facade_faces": photo_facade_faces,
        "mean_observed_coverage": round(total_obs_weight / max(1, total_faces), 4),
        "blend": str(blend_path),
    }
    print("BAKE_MULTIVIEW_OK", json.dumps(summary))
    return summary


def apply_materials_in_blender(manifest_path: Path, blend_path: Path) -> None:
    import bpy  # type: ignore

    atlas_paths = json.loads(manifest_path.read_text(encoding="utf-8"))
    for idx, paths in enumerate(atlas_paths):
        mat = bpy.data.materials.get(f"VillageAtlas_{idx}")
        if mat is None:
            continue
        nt = mat.node_tree
        nt.nodes.clear()
        out_node = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.38
        nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

        tex_a = nt.nodes.new("ShaderNodeTexImage")
        img_a = bpy.data.images.load(paths["albedo"], check_existing=False)
        img_a.colorspace_settings.name = "sRGB"
        tex_a.image = img_a
        nt.links.new(tex_a.outputs["Color"], bsdf.inputs["Base Color"])

        tex_r = nt.nodes.new("ShaderNodeTexImage")
        img_r = bpy.data.images.load(paths["roughness"], check_existing=False)
        img_r.colorspace_settings.name = "Non-Color"
        tex_r.image = img_r
        nt.links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])

        tex_n = nt.nodes.new("ShaderNodeTexImage")
        img_n = bpy.data.images.load(paths["normal"], check_existing=False)
        img_n.colorspace_settings.name = "Non-Color"
        tex_n.image = img_n
        nrm_node = nt.nodes.new("ShaderNodeNormalMap")
        nrm_node.inputs["Strength"].default_value = 0.78
        nt.links.new(tex_n.outputs["Color"], nrm_node.inputs["Color"])
        nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    snap_dir = blend_path.parent / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(snap_dir / "002_multiview_baked.blend"), copy=True)
    print("APPLIED_ATLAS_MATERIALS_OK", len(atlas_paths))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply-materials", action="store_true")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    p.add_argument("--uv-map", type=Path, default=ROOT / "artifacts/blender_village/uv_atlas_map.json")
    p.add_argument("--probes", type=Path, default=ROOT / "artifacts/blender_village/bake_probes.npz")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts/blender_village/village_master.blend")
    return p.parse_args(argv)


if __name__ == "__main__":
    if "--apply-materials" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
        args = parse_args(argv)
        apply_materials_in_blender(args.manifest.resolve(), args.blend.resolve())
    else:
        args = parse_args()
        bake_atlases(
            spec_path=args.spec.resolve(),
            uv_map_path=args.uv_map.resolve(),
            probes_path=args.probes.resolve(),
            out_dir=args.out_dir.resolve(),
            blend_path=args.blend.resolve(),
        )
