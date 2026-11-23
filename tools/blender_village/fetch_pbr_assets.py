"""Download/cache Poly Haven CC0 2K PBR textures, harmonize to village photo, and synthesize foliage cards."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]

POLYHAVEN_ASSETS: dict[str, dict[str, Any]] = {
    "plastered_wall": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Seamless aged off-white rural plaster"},
    "beige_wall_001": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Warm sunlit buff rural stucco"},
    "grey_plaster_02": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Weathered light-gray cement mortar walls"},
    "brick_wall_006": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Northern clay brick compound walls & side facades"},
    "concrete_floor_worn_001": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Pitted stained flat concrete roof slab"},
    "corrugated_iron": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Galvanized & slate-blue corrugated sheds"},
    "clay_roof_tiles_02": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Terracotta clay roof tiles"},
    "asphalt_02": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Clean rural village asphalt aggregate"},
    "concrete_floor_02": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Poured monolithic concrete driveways & yard aprons"},
    "bark_brown_01": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Poplar & broadleaf tree bark"},
    "worn_shutter": {"maps": ("Diffuse", "nor_gl", "Rough"), "note": "Rural rolling shop shutters & doors"},
}


def _fetch_polyhaven_triplet(
    session: requests.Session,
    asset_id: str,
    cache_dir: Path,
    *,
    resolution: str = "2k",
    timeout_sec: float = 25.0,
) -> dict[str, Path] | None:
    """Download and cache Diffuse/nor_gl/Rough jpgs (preferring 2K, falling back to 1K) for one Poly Haven CC0 asset."""
    asset_dir = cache_dir / asset_id
    asset_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "Diffuse": asset_dir / f"{asset_id}_diff_{resolution}.jpg",
        "nor_gl": asset_dir / f"{asset_id}_nor_gl_{resolution}.jpg",
        "Rough": asset_dir / f"{asset_id}_rough_{resolution}.jpg",
    }
    if all(p.is_file() and p.stat().st_size > 2048 for p in expected.values()):
        return expected

    for attempt in range(2):
        try:
            meta_resp = session.get(f"https://api.polyhaven.com/files/{asset_id}", timeout=timeout_sec)
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            for key, dest in expected.items():
                if dest.is_file() and dest.stat().st_size > 2048:
                    continue
                node = meta.get(key) or {}
                res_dict = node.get(resolution) or node.get("1k") or next(iter(node.values()), {})
                file_info = res_dict.get("jpg") or res_dict.get("png")
                if not file_info or "url" not in file_info:
                    raise KeyError(f"{asset_id} missing {resolution}/1k jpg/png for {key}")
                r = session.get(file_info["url"], timeout=timeout_sec)
                r.raise_for_status()
                dest.write_bytes(r.content)
            return expected
        except Exception as exc:
            if attempt == 0:
                time.sleep(0.6)
            else:
                print(f"[WARN] Poly Haven fallback for {asset_id}: {exc}")
                return None
    return None


def _tileable_fbm(size: int, rng: np.random.Generator, octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    acc = np.zeros((size, size), dtype=np.float32)
    wsum = 0.0
    for grid, w in octaves:
        low = rng.normal(0.0, 1.0, (grid, grid)).astype(np.float32)
        tiled = np.tile(low, (3, 3))
        big = cv2.resize(tiled, (size * 3, size * 3), interpolation=cv2.INTER_CUBIC)[size : 2 * size, size : 2 * size]
        acc += big * w
        wsum += w
    acc /= max(wsum, 1e-6)
    return (acc - acc.min()) / max(1e-6, float(acc.max() - acc.min()))


def _height_to_normal_rgb(height_map: np.ndarray, strength: float = 2.2) -> np.ndarray:
    pad = cv2.copyMakeBorder(height_map.astype(np.float32), 1, 1, 1, 1, cv2.BORDER_WRAP)
    gx = cv2.Sobel(pad, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1] * strength
    gy = cv2.Sobel(pad, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1] * strength
    nz = np.ones_like(gx)
    inv_len = 1.0 / np.sqrt(gx * gx + gy * gy + nz * nz)
    nx = -gx * inv_len
    ny = gy * inv_len
    nz_n = nz * inv_len
    return np.clip(
        np.stack([(nx * 0.5 + 0.5) * 255.0, (ny * 0.5 + 0.5) * 255.0, (nz_n * 0.5 + 0.5) * 255.0], -1),
        0,
        255,
    ).astype(np.uint8)


def _fallback_swatch_arrays(asset_id: str, size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic weathered fallback if Poly Haven is unreachable."""
    seed = sum(ord(ch) for ch in asset_id) + 17
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    macro = _tileable_fbm(size, rng, ((6, 0.55), (18, 0.30), (48, 0.15)))
    fine = _tileable_fbm(size, rng, ((32, 0.5), (96, 0.35), (192, 0.15)))

    if "brick" in asset_id:
        bh, bw = 32, 64
        row = (yy // bh).astype(np.int32)
        xs = xx + (row % 2) * (bw * 0.5)
        mortar = (yy % bh < 4) | (xs % bw < 4)
        var = ((row * 37 + (xs // bw).astype(np.int32) * 17) % 11) / 10.0 - 0.5
        rgb = np.stack(
            [138 + var * 26 + (macro - 0.5) * 22, 92 + var * 18 + (macro - 0.5) * 16, 72 + var * 14 + (macro - 0.5) * 14],
            axis=-1,
        )
        rgb[mortar] = np.array([154, 146, 134], dtype=np.float32) + (fine[mortar, None] - 0.5) * 18.0
        ht = np.where(mortar, 0.18 + fine * 0.15, 0.72 + fine * 0.22 + var * 0.08)
        rgh = np.where(mortar, 240.0, 218.0 + (fine - 0.5) * 18.0)
    elif "corrugated" in asset_id or "shutter" in asset_id:
        rib = 0.5 + 0.5 * np.sin(xx * (2.0 * math.pi / 16.0))
        rgb = np.stack(
            [148 + rib * 26 + (macro - 0.5) * 18, 152 + rib * 26 + (macro - 0.5) * 18, 155 + rib * 27 + (macro - 0.5) * 18],
            axis=-1,
        )
        ht = rib * 0.85 + fine * 0.12
        rgh = 128.0 + (1.0 - rib) * 42.0 + macro * 28.0
    elif "clay_roof" in asset_id:
        flute = 0.5 + 0.5 * np.cos(xx * (2.0 * math.pi / 18.0))
        course = 1.0 - np.clip((yy % 28) / 28.0, 0.0, 1.0)
        rgb = np.stack(
            [128 + flute * 16 + (macro - 0.5) * 24, 82 + flute * 12 + (macro - 0.5) * 18, 68 + flute * 10 + (macro - 0.5) * 15],
            axis=-1,
        )
        rgb[(yy % 28) < 3] *= 0.65
        ht = flute * 0.55 + course * 0.45
        rgh = 205.0 + (1.0 - flute) * 24.0
    elif "asphalt" in asset_id:
        base = 88.0 + (macro - 0.5) * 18.0 + (fine - 0.5) * 16.0
        rgb = np.stack([base, base + 2.0, base + 3.0], axis=-1)
        ht = fine * 0.65 + macro * 0.35
        rgh = 200.0 + (macro - 0.5) * 28.0
    elif "bark" in asset_id:
        furrow = _tileable_fbm(size, rng, ((8, 0.4), (36, 0.4), (80, 0.2)))
        bark_streaks = 0.5 + 0.5 * np.sin(xx * 0.22 + furrow * 5.0)
        rgb = np.stack([82 + bark_streaks * 24, 72 + bark_streaks * 20, 56 + bark_streaks * 16], axis=-1)
        ht = bark_streaks * 0.7 + fine * 0.3
        rgh = 232.0 - bark_streaks * 20.0
    elif "concrete" in asset_id:
        damp = np.clip((macro - 0.55) * 2.2, 0.0, 1.0)
        rgb = np.stack(
            [138 - damp * 26 + (fine - 0.5) * 18, 134 - damp * 25 + (fine - 0.5) * 17, 126 - damp * 22 + (fine - 0.5) * 16],
            axis=-1,
        )
        ht = fine * 0.5 + macro * 0.35
        rgh = 222.0 - damp * 22.0 + (fine - 0.5) * 16.0
    else:
        rgb = np.stack(
            [196 + (macro - 0.5) * 26 + (fine - 0.5) * 14, 190 + (macro - 0.5) * 24 + (fine - 0.5) * 13, 180 + (macro - 0.5) * 22 + (fine - 0.5) * 12],
            axis=-1,
        )
        ht = fine * 0.45 + macro * 0.3
        rgh = 218.0 + (fine - 0.5) * 20.0

    alb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    nrm_u8 = _height_to_normal_rgb(ht, strength=2.2)
    rgh_u8 = np.clip(rgh, 35, 250).astype(np.uint8)
    return alb_u8, nrm_u8, rgh_u8


def _load_cached_triplet(asset_id: str, cache_dir: Path, resolution: str = "2k") -> dict[str, Path] | None:
    """Return cached Poly Haven file paths without network access if present on disk."""
    asset_dir = cache_dir / asset_id
    for res in (resolution, "1k"):
        expected = {
            "Diffuse": asset_dir / f"{asset_id}_diff_{res}.jpg",
            "nor_gl": asset_dir / f"{asset_id}_nor_gl_{res}.jpg",
            "Rough": asset_dir / f"{asset_id}_rough_{res}.jpg",
        }
        if all(p.is_file() and p.stat().st_size > 2048 for p in expected.values()):
            return expected
    return None


def _load_or_fallback(
    session: requests.Session | None,
    asset_id: str,
    cache_dir: Path,
    target_size: int = 1024,
    *,
    resolution: str = "2k",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    paths = _load_cached_triplet(asset_id, cache_dir, resolution=resolution)
    if paths is None and session is not None:
        paths = _fetch_polyhaven_triplet(session, asset_id, cache_dir, resolution=resolution)
    if paths is not None:
        try:
            alb = np.asarray(Image.open(paths["Diffuse"]).convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS), dtype=np.uint8)
            nrm = np.asarray(Image.open(paths["nor_gl"]).convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS), dtype=np.uint8)
            rgh = np.asarray(Image.open(paths["Rough"]).convert("L").resize((target_size, target_size), Image.Resampling.LANCZOS), dtype=np.uint8)
            return alb, nrm, rgh, f"polyhaven:{asset_id}"
        except Exception as exc:
            print(f"[WARN] Failed reading cached {asset_id}: {exc}")
    alb, nrm, rgh = _fallback_swatch_arrays(asset_id, size=target_size)
    return alb, nrm, rgh, f"procedural:{asset_id}"


def _regrade_rgb(
    rgb_u8: np.ndarray,
    *,
    target_mean: tuple[float, float, float] | None = None,
    blend_strength: float = 0.55,
    sat_scale: float = 0.85,
    contrast: float = 1.0,
) -> np.ndarray:
    """Shift albedo toward northern rural target palette while preserving all photogrammetry detail."""
    arr = rgb_u8.astype(np.float32)
    if abs(sat_scale - 1.0) > 1e-3:
        gray = arr.mean(axis=2, keepdims=True)
        arr = gray + (arr - gray) * sat_scale
    if abs(contrast - 1.0) > 1e-3:
        m = arr.mean(axis=(0, 1), keepdims=True)
        arr = m + (arr - m) * contrast
    if target_mean is not None:
        cur = np.maximum(arr.mean(axis=(0, 1)), 4.0)
        ratio = np.asarray(target_mean, dtype=np.float32) / cur
        factor = 1.0 + (ratio - 1.0) * blend_strength
        arr *= factor[None, None, :]
    return np.clip(arr, 12.0, 246.0).astype(np.uint8)


def _resize_max_dim(img: np.ndarray, max_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_size:
        return img
    scale = max_size / float(m)
    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _save_jpg_triplet(
    swatches_dir: Path,
    name: str,
    alb: np.ndarray,
    nrm: np.ndarray,
    rgh: np.ndarray,
    *,
    normal_size: int = 1024,
    rough_size: int = 512,
    albedo_quality: int = 90,
    normal_quality: int = 86,
    rough_quality: int = 82,
) -> dict[str, str]:
    swatches_dir.mkdir(parents=True, exist_ok=True)
    ap = swatches_dir / f"{name}_albedo.jpg"
    np_p = swatches_dir / f"{name}_normal.jpg"
    rp = swatches_dir / f"{name}_roughness.jpg"
    Image.fromarray(alb).save(ap, format="JPEG", quality=albedo_quality)
    nrm_resized = _resize_max_dim(nrm, normal_size)
    Image.fromarray(nrm_resized).save(np_p, format="JPEG", quality=normal_quality)
    # Smooth & quantize roughness so Blender's glTF metallicRoughnessTexture PNG stays compact
    rgh_small = _resize_max_dim(rgh, rough_size)
    rgh_blur = cv2.GaussianBlur(rgh_small, (3, 3), 0.7)
    rgh_q = ((rgh_blur // 8) * 8).astype(np.uint8)
    Image.fromarray(rgh_q).save(rp, format="JPEG", quality=rough_quality)
    return {"albedo": str(ap), "normal": str(np_p), "roughness": str(rp)}


def _build_road_asphalt_marked(swatches_dir: Path, road_alb: np.ndarray, road_nrm: np.ndarray, road_rgh: np.ndarray) -> dict[str, str]:
    """Build 2048x2048 2x2-tiled marked rural asphalt with crisp aggregate, worn lane ruts, and dusty shoulders."""
    tile_sz = 1024
    alb_1k = cv2.resize(road_alb, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
    nrm_1k = cv2.resize(road_nrm, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
    rgh_1k = cv2.resize(road_rgh, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)

    arr = np.tile(alb_1k, (2, 2, 1)).astype(np.float32)
    nrm = np.tile(nrm_1k, (2, 2, 1))
    rgh = np.tile(rgh_1k, (2, 2)).astype(np.float32)
    h, w = arr.shape[:2]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    # Subtle longitudinal aggregate/tar variation & micro-speckle without blotchy repeats
    rng = np.random.default_rng(702)
    wear_noise = _tileable_fbm(1024, rng, ((8, 0.50), (28, 0.35), (84, 0.15)))
    wear_2k = np.tile(wear_noise, (2, 2))
    arr *= (0.95 + 0.10 * wear_2k[:, :, None])

    # Polished longitudinal tire-wear tracks in both lanes (lower roughness so grazing sun gleams subtly)
    track = np.exp(-((xs - 0.30) ** 2) / 0.0042) + np.exp(-((xs - 0.70) ** 2) / 0.0042)
    track += 0.75 * (np.exp(-((xs - 0.20) ** 2) / 0.0034) + np.exp(-((xs - 0.80) ** 2) / 0.0034))
    arr *= 1.0 - 0.09 * track[:, :, None]
    rgh -= 34.0 * track

    # Dusty warm sandy rural shoulder transitions encroaching organically along outer edges
    edge_dist = np.minimum(xs, 1.0 - xs) - (wear_2k - 0.5) * 0.028
    shoulder = np.clip((0.105 - edge_dist) / 0.105, 0.0, 1.0)
    dust_rgb = np.array([196.0, 190.0, 178.0], dtype=np.float32)
    arr = arr * (1.0 - 0.76 * shoulder[:, :, None]) + dust_rgb[None, None, :] * (0.76 * shoulder[:, :, None])
    rgh = np.maximum(rgh, 238.0 * shoulder + rgh * (1.0 - shoulder))

    # Sun-faded, chipped white edge lines near u=0.055 and u=0.945
    edge_line = (np.abs(xs - 0.055) < 0.0085) | (np.abs(xs - 0.945) < 0.0085)
    chip = np.clip(0.35 + 0.65 * wear_2k, 0.20, 0.92)
    white_paint = np.array([184.0, 180.0, 170.0], dtype=np.float32)
    edge_w = np.where(edge_line, chip * 0.52, 0.0)
    arr = arr * (1.0 - edge_w[:, :, None]) + white_paint[None, None, :] * edge_w[:, :, None]
    rgh = np.where(edge_line, rgh + 12.0 * edge_w, rgh)

    # Sun-faded dashed yellow centerline (2 dashes along v since tile is 2x repeated)
    center_x = np.abs(xs - 0.50) < 0.0095
    dash_y = ((ys >= 0.08) & (ys <= 0.30)) | ((ys >= 0.58) & (ys <= 0.80))
    center_w = np.where(center_x & dash_y, chip * 0.56, 0.0)
    yellow_paint = np.array([170.0, 146.0, 92.0], dtype=np.float32)
    arr = arr * (1.0 - center_w[:, :, None]) + yellow_paint[None, None, :] * center_w[:, :, None]

    return _save_jpg_triplet(
        swatches_dir,
        "road_asphalt_marked",
        np.clip(arr, 14, 244).astype(np.uint8),
        nrm,
        np.clip(rgh, 152, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
        albedo_quality=88,
    )


def _render_window_variant(cell_size: int, variant_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one 512x512 photorealistic northern rural casement window quadrant."""
    yy, xx = np.mgrid[:cell_size, :cell_size].astype(np.float32)
    u = xx / cell_size
    v = yy / cell_size  # 0 at top of window (+Y), 1 at sill bottom

    outer = (u < 0.042) | (u > 0.958) | (v < 0.040) | (v > 0.960)
    transom = np.abs(v - 0.28) < 0.017
    mull_l = np.abs(u - 0.335) < 0.013
    mull_r = np.abs(u - 0.665) < 0.013
    gasket = (
        ((np.abs(u - 0.045) < 0.006) | (np.abs(u - 0.955) < 0.006) | (np.abs(v - 0.043) < 0.006) | (np.abs(v - 0.957) < 0.006))
        | ((np.abs(v - 0.28) < 0.022) & (~transom))
        | ((np.abs(u - 0.335) < 0.018) & (~mull_l))
        | ((np.abs(u - 0.665) < 0.018) & (~mull_r))
    ) & (~outer) & (~transom) & (~mull_l) & (~mull_r)
    bar = outer | transom | mull_l | mull_r

    # Sky reflection gradient on upper glazing + recessed dark interior depth below
    cloud = 0.5 + 0.5 * np.sin(u * 5.2 + v * 3.8 + variant_idx * 1.4) * np.cos(u * 3.1 - v * 6.0)
    sky_top = np.array([118.0, 140.0, 162.0], dtype=np.float32) if variant_idx != 2 else np.array([104.0, 126.0, 148.0], dtype=np.float32)
    sky_bot = np.array([28.0, 35.0, 42.0], dtype=np.float32)
    refl_w = np.clip((1.0 - v * 0.92) ** 1.35, 0.08, 0.96)
    glass = sky_bot[None, None, :] + (sky_top - sky_bot)[None, None, :] * refl_w[:, :, None]
    glass += cloud[:, :, None] * np.array([16.0, 17.0, 14.0], dtype=np.float32) * (1.0 - v[:, :, None])

    # Crisp diagonal afternoon cloud glint streak on variant 2 and faint sheen on others
    glint_coord = (u * 0.72 + v * 0.68) - (0.58 + 0.06 * variant_idx)
    glint = np.exp(-(glint_coord ** 2) / 0.0045) * (0.28 if variant_idx == 2 else 0.14)
    glass += glint[:, :, None] * np.array([185.0, 182.0, 174.0], dtype=np.float32)

    # Recessed interior room & sheer curtains (behind transom)
    below_transom = np.clip((v - 0.295) / 0.06, 0.0, 1.0)
    fold = 0.5 + 0.5 * np.sin(u * 54.0 + variant_idx * 2.1 + np.sin(v * 7.0) * 1.1)
    if variant_idx == 0:
        curtain_area = np.clip((0.56 - u) / 0.14, 0.0, 1.0) * below_transom
        curtain_rgb = np.array([148.0, 142.0, 126.0], dtype=np.float32)
    elif variant_idx == 1:
        curtain_area = np.clip((u - 0.44) / 0.16, 0.0, 1.0) * below_transom
        curtain_rgb = np.array([128.0, 142.0, 152.0], dtype=np.float32)
    elif variant_idx == 2:
        # Open tilted upper transom shadow slot in center pane
        in_open = (v > 0.05) & (v < 0.265) & (u > 0.345) & (u < 0.655)
        glass[in_open] *= 0.38
        curtain_area = np.clip((0.40 - u) / 0.10, 0.0, 1.0) * below_transom
        curtain_rgb = np.array([134.0, 126.0, 114.0], dtype=np.float32)
    else:
        curtain_area = np.maximum(np.clip((0.36 - u) / 0.08, 0.0, 1.0), np.clip((u - 0.64) / 0.08, 0.0, 1.0)) * below_transom
        curtain_rgb = np.array([162.0, 150.0, 132.0], dtype=np.float32)

    curtain_mix = (curtain_area * (0.34 + 0.22 * fold))[:, :, None]
    glass = glass * (1.0 - curtain_mix) + curtain_rgb[None, None, :] * curtain_mix

    # Dark perimeter ambient shadow inside glass rebate
    edge_rebate = np.clip((0.08 - np.minimum(np.minimum(u, 1.0 - u), np.minimum(v, 1.0 - v))) / 0.04, 0.0, 1.0)
    glass *= 1.0 - 0.28 * edge_rebate[:, :, None]

    # Frame metallic / UPVC colors + bevel shading
    if variant_idx in (0, 1):
        frame_base = np.array([162.0, 165.0, 168.0], dtype=np.float32)
        frame_rgh = 110.0
    elif variant_idx == 2:
        frame_base = np.array([94.0, 90.0, 86.0], dtype=np.float32)
        frame_rgh = 126.0
    else:
        frame_base = np.array([212.0, 208.0, 200.0], dtype=np.float32)
        frame_rgh = 128.0

    # Subtle top-down highlight & bottom dust on aluminum frame
    frame_grad = (1.06 - 0.14 * v)[:, :, None] * (1.0 + 0.04 * np.cos(u * 40.0))[:, :, None]
    glass[gasket] = np.array([34.0, 35.0, 36.0], dtype=np.float32)
    glass[bar] = (frame_base[None, None, :] * frame_grad)[bar]

    # Crisp glass roughness (0.09..0.15) vs frame/gasket/sill dust
    rgh = np.full((cell_size, cell_size), 28.0, dtype=np.float32) + (1.0 - refl_w) * 10.0
    rgh[gasket] = 192.0
    rgh[bar] = frame_rgh + v[bar] * 24.0

    ht = np.where(bar, 0.90, np.where(gasket, 0.42, 0.12)).astype(np.float32)
    return np.clip(glass, 14.0, 242.0).astype(np.uint8), np.clip(rgh, 22.0, 232.0).astype(np.uint8), ht


def _build_window_pane_swatch(swatches_dir: Path, size: int = 1024) -> dict[str, str]:
    """Generate 1024x1024 2x2 4-variant northern rural casement glazing atlas + sharp glass/frame PBR maps."""
    cell = size // 2
    alb = np.zeros((size, size, 3), dtype=np.uint8)
    rgh = np.zeros((size, size), dtype=np.uint8)
    ht = np.zeros((size, size), dtype=np.float32)
    # Image row 0 is UV v=1 (top); image row cell..size is UV v=0..0.5 (bottom)
    row_ranges = [
        (cell, size, 0, cell),
        (cell, size, cell, size),
        (0, cell, 0, cell),
        (0, cell, cell, size),
    ]
    for qi, (r0, r1, c0, c1) in enumerate(row_ranges):
        va, vr, vh = _render_window_variant(cell, qi)
        alb[r0:r1, c0:c1] = va
        rgh[r0:r1, c0:c1] = vr
        ht[r0:r1, c0:c1] = vh

    nrm = _height_to_normal_rgb(ht, strength=2.6)
    return _save_jpg_triplet(
        swatches_dir,
        "window_pane_photo",
        alb,
        nrm,
        rgh,
        normal_size=1024,
        rough_size=512,
        albedo_quality=90,
    )


def _enhance_shop_shutter(
    sh_alb: np.ndarray,
    sh_nrm: np.ndarray,
    sh_rgh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend crisp horizontal interlocking rolling-shutter slats & lower street dust patina."""
    h, w = sh_alb.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    slat_period = h / 46.0
    phase = np.mod(np.arange(h, dtype=np.float32)[:, None], slat_period) / slat_period
    slat_prof = np.sin(phase * math.pi)
    groove = np.exp(-((phase - 0.08) ** 2) / 0.008) + np.exp(-((phase - 0.92) ** 2) / 0.008)

    graded = _regrade_rgb(sh_alb, target_mean=(122.0, 126.0, 130.0), blend_strength=0.78, sat_scale=0.38, contrast=1.08).astype(np.float32)
    graded *= (0.90 + 0.14 * slat_prof[:, :, None]) * (1.0 - 0.24 * groove[:, :, None])
    # Warm lower road dust near ground level
    dust = np.clip((yy - 0.72) / 0.28, 0.0, 1.0) ** 1.4
    dust_rgb = np.array([154.0, 144.0, 128.0], dtype=np.float32)
    graded = graded * (1.0 - 0.38 * dust[:, :, None]) + dust_rgb[None, None, :] * (0.38 * dust[:, :, None])

    slat_nrm = _height_to_normal_rgb(np.broadcast_to(slat_prof * 0.85, (h, w)), strength=2.4)
    nrm_blend = np.clip(sh_nrm.astype(np.float32) * 0.45 + slat_nrm.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    rgh = np.clip(sh_rgh.astype(np.float32) * 0.72 + 48.0 - 22.0 * slat_prof + 55.0 * dust, 72.0, 232.0).astype(np.uint8)
    return np.clip(graded, 14, 244).astype(np.uint8), nrm_blend, rgh


def _build_shop_sign_swatches(swatches_dir: Path) -> dict[str, dict[str, str]]:
    """Synthesize weathered rural enamel storefront signboard textures (crimson & cobalt)."""
    out: dict[str, dict[str, str]] = {}
    configs = [
        ("shop_sign_red", np.array([148.0, 42.0, 36.0], dtype=np.float32), np.array([224.0, 212.0, 182.0], dtype=np.float32), 401),
        ("shop_sign_blue", np.array([38.0, 76.0, 126.0], dtype=np.float32), np.array([226.0, 224.0, 214.0], dtype=np.float32), 402),
    ]
    w, h = 512, 256
    for sname, base_col, accent_col, seed in configs:
        rng = np.random.default_rng(seed)
        noise = cv2.resize(_tileable_fbm(256, rng, ((6, 0.5), (18, 0.35), (48, 0.15))), (w, h), interpolation=cv2.INTER_CUBIC)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        u, v = xx / w, yy / h

        # Subtle painted enamel board with top-down sun fade & weather chalking
        arr = np.broadcast_to(base_col, (h, w, 3)).astype(np.float32).copy()
        arr *= (0.94 + 0.12 * noise[:, :, None]) * (0.92 + 0.14 * (1.0 - v[:, :, None]))

        # Metallic outer bezel + pale double-line painted rural sign border
        bezel = (u < 0.025) | (u > 0.975) | (v < 0.045) | (v > 0.955)
        border_line = (
            ((u > 0.045) & (u < 0.955) & ((np.abs(v - 0.09) < 0.012) | (np.abs(v - 0.91) < 0.012)))
            | ((v > 0.09) & (v < 0.91) & ((np.abs(u - 0.05) < 0.008) | (np.abs(u - 0.95) < 0.008)))
        )
        # Clean weathered rural enamel storefront styling (recessed border + faded title stripe + left circular crest)
        strip = (u > 0.24) & (u < 0.89) & (v > 0.36) & (v < 0.56)
        emblem_d2 = ((u - 0.145) ** 2 / 0.0020 + (v - 0.50) ** 2 / 0.020)
        emblem_ring = (emblem_d2 < 1.0) & (emblem_d2 > 0.56)
        emblem_dot = emblem_d2 < 0.18
        sub_rule = (u > 0.24) & (u < 0.82) & (v > 0.68) & (v < 0.71)

        # Gentle chalky wear mask so enamel signage looks authentically sunlit & weathered
        fade_mask = 0.76 + 0.18 * noise
        for mask, strength in ((border_line, 0.86), (emblem_ring, 0.84), (emblem_dot, 0.72), (sub_rule, 0.64)):
            arr[mask] = arr[mask] * (1.0 - fade_mask[mask, None] * strength) + accent_col[None, :] * (fade_mask[mask, None] * strength)
        arr[strip] = arr[strip] * (1.0 - fade_mask[strip, None] * 0.28) + accent_col[None, :] * (fade_mask[strip, None] * 0.28)

        # Aluminum trim frame + top rain streaking
        arr[bezel] = np.array([162.0, 160.0, 155.0], dtype=np.float32) * (0.90 + 0.14 * noise[bezel, None])
        rain = np.exp(-v / 0.18) * (0.06 + 0.10 * noise)
        arr *= 1.0 - rain[:, :, None]

        ht = np.where(bezel, 0.85, np.where(border_line | emblem_ring | sub_rule, 0.48, 0.35)).astype(np.float32)
        nrm = _height_to_normal_rgb(ht, strength=1.9)
        rgh = np.clip(118.0 + noise * 36.0 + np.where(bezel, 16.0, 0.0) + rain * 90.0, 82.0, 224.0).astype(np.uint8)
        out[sname] = _save_jpg_triplet(
            swatches_dir,
            sname,
            np.clip(arr, 14, 244).astype(np.uint8),
            nrm,
            rgh,
            normal_size=512,
            rough_size=256,
        )
    return out


def _sample_foliage_palette_from_photo(photo_path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample natural vegetation leaf palette directly from the reference drone photo."""
    ref_path = photo_path if photo_path is not None else (ROOT / "assets/scene_sources/village_road_near.png")
    fallback = np.array(
        [
            [38, 48, 24],
            [48, 60, 30],
            [60, 72, 36],
            [32, 42, 20],
            [72, 84, 42],
            [54, 65, 32],
        ],
        dtype=np.float32,
    )
    if not ref_path.is_file():
        return fallback, np.array([62, 54, 44], dtype=np.float32)
    rgb = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32)
    if ref_path.name == "village_road_near.png" and rgb.shape[0] >= 200 and rgb.shape[1] >= 1400:
        # Poplar windbreak sits in upper band (rows 15..165) where green dominates blue & avoids field highlights
        patch = rgb[15:165, 220:1380]
        veg = (patch[:, :, 1] > patch[:, :, 2] + 8.0) & (patch[:, :, 1] >= patch[:, :, 0] - 4.0) & (patch[:, :, 1] < 135.0) & (patch[:, :, 1] > 38.0)
        pts = patch[veg]
    else:
        exg = 2.0 * rgb[:, :, 1] - rgb[:, :, 0] - rgb[:, :, 2]
        veg = (exg > 16.0) & (rgb[:, :, 1] > rgb[:, :, 2] + 8.0) & (rgb[:, :, 1] > 32.0) & (rgb[:, :, 1] < 185.0)
        pts = rgb[veg]
    if len(pts) < 100:
        return fallback, np.array([62, 54, 44], dtype=np.float32)
    qs = np.percentile(pts, [12, 28, 45, 62, 78, 90], axis=0).astype(np.float32)
    qs[:, 0] = np.clip(qs[:, 0] * 0.90, 24.0, 128.0)
    qs[:, 1] = np.clip(qs[:, 1] * 0.96, 36.0, 146.0)
    qs[:, 2] = np.clip(qs[:, 2] * 0.72, 16.0, 90.0)
    return qs.astype(np.float32), np.array([58, 50, 40], dtype=np.float32)


def _bleed_rgb_into_transparent(rgba: np.ndarray, iters: int = 8) -> np.ndarray:
    """Propagate opaque leaf colors into alpha=0 background to prevent mipmap border fringes."""
    out = rgba.copy()
    rgb = out[:, :, :3].copy()
    valid = out[:, :, 3] > 20
    if not valid.any():
        return out
    for _ in range(iters):
        dil_rgb = cv2.dilate(rgb, np.ones((3, 3), np.uint8))
        new_valid = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        fill = new_valid & (~valid)
        rgb[fill] = dil_rgb[fill]
        valid = new_valid
    out[:, :, :3] = rgb
    return out


def _leaf_polygon(cx: float, cy: float, length: float, width: float, angle_rad: float, rng: np.random.Generator) -> np.ndarray:
    """Create a pointed ovate poplar/deciduous leaf polygon in 2D pixel space."""
    pts = []
    # Base -> right waist -> pointed tip -> left waist
    local = [
        (-length * 0.50, 0.0),
        (-length * 0.22, width * 0.48),
        (length * 0.12, width * 0.42),
        (length * 0.54, rng.uniform(-1.5, 1.5)),
        (length * 0.12, -width * 0.42),
        (-length * 0.22, -width * 0.48),
    ]
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    for lx, ly in local:
        jx = lx + rng.uniform(-1.2, 1.2)
        jy = ly + rng.uniform(-1.2, 1.2)
        pts.append([int(round(cx + jx * ca - jy * sa)), int(round(cy + jx * sa + jy * ca))])
    return np.asarray(pts, dtype=np.int32)


def _generate_foliage_card(
    swatches_dir: Path,
    name: str,
    palette: np.ndarray,
    twig_rgb: np.ndarray,
    *,
    kind: str,
    seed: int,
) -> dict[str, str]:
    """Synthesize crisp 512x512 RGBA botanical leaf-branch card + matching normal/roughness maps."""
    size = 512
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    ht = np.zeros((size, size), dtype=np.float32)

    # Woody central branch & forked twigs kept inside leaf cluster silhouette
    num_twigs = 6 if kind == "poplar" else (8 if kind == "broadleaf" else 7)
    cx, cy = size // 2, size - 28
    twig_col = (int(twig_rgb[0]), int(twig_rgb[1]), int(twig_rgb[2]), 255)
    cv2.line(canvas, (cx, cy), (cx + int(rng.integers(-8, 8)), int(size * 0.28)), twig_col, 6, cv2.LINE_AA)
    anchors = [(cx, int(size * 0.46))]
    for ti in range(num_twigs):
        frac = 0.24 + 0.54 * (ti / max(1, num_twigs - 1))
        sy = int(cy - frac * (size * 0.62))
        sx = int(cx + rng.integers(-8, 8))
        side = -1 if ti % 2 == 0 else 1
        ex = int(np.clip(sx + side * rng.integers(58, 138), 72, size - 72))
        ey = int(np.clip(sy - rng.integers(45, 115), 62, size - 78))
        cv2.line(canvas, (sx, sy), (ex, ey), twig_col, 3, cv2.LINE_AA)
        cv2.line(ht, (sx, sy), (ex, ey), 0.75, 3)
        anchors.append((ex, ey))
        anchors.append((int((sx + ex) * 0.5), int((sy + ey) * 0.5)))

    # Crisp pointed overlapping individual leaves clustered around twigs
    n_leaves = 270 if kind == "poplar" else (300 if kind == "broadleaf" else 210)
    for li in range(n_leaves):
        ax, ay = anchors[li % len(anchors)]
        jitter_r = (size * (0.14 if kind == "poplar" else 0.17)) * math.sqrt(rng.uniform(0.02, 1.0))
        r_ang = rng.uniform(0.0, math.tau)
        lx = float(np.clip(ax + math.cos(r_ang) * jitter_r, 42, size - 42))
        ly = float(np.clip(ay + math.sin(r_ang) * jitter_r, 38, size - 36))

        llen = float(rng.uniform(24, 38) if kind == "poplar" else rng.uniform(26, 42))
        lwid = llen * float(rng.uniform(0.56, 0.78))
        rot = float(rng.uniform(0.0, math.tau))
        col = palette[int(rng.integers(0, len(palette)))].copy()
        col *= float(rng.uniform(0.76, 1.18))
        col *= 1.12 - 0.34 * (ly / size)
        c_u8 = (int(np.clip(col[0], 18, 136)), int(np.clip(col[1], 22, 144)), int(np.clip(col[2], 12, 88)), 255)
        poly = _leaf_polygon(lx, ly, llen, lwid, rot, rng)
        cv2.fillPoly(canvas, [poly], c_u8, cv2.LINE_AA)
        cv2.fillPoly(ht, [poly], float(rng.uniform(0.35, 0.95)))
        # Midrib vein
        vx0 = int(lx - math.cos(rot) * (llen * 0.34))
        vy0 = int(ly - math.sin(rot) * (llen * 0.34))
        vx1 = int(lx + math.cos(rot) * (llen * 0.34))
        vy1 = int(ly + math.sin(rot) * (llen * 0.34))
        vein = (max(16, int(c_u8[0] * 0.78)), max(20, int(c_u8[1] * 0.80)), max(12, int(c_u8[2] * 0.76)), 255)
        cv2.line(canvas, (vx0, vy0), (vx1, vy1), vein, 1, cv2.LINE_AA)

    canvas = _bleed_rgb_into_transparent(canvas, iters=10)
    nrm_u8 = _height_to_normal_rgb(cv2.GaussianBlur(ht, (5, 5), 1.0), strength=2.6)
    rgh_u8 = np.clip(192.0 - ht * 35.0 + rng.normal(0.0, 8.0, (size, size)), 120, 236).astype(np.uint8)

    rgba_path = swatches_dir / f"{name}_rgba.png"
    nrm_path = swatches_dir / f"{name}_normal.jpg"
    rgh_path = swatches_dir / f"{name}_roughness.jpg"
    Image.fromarray(canvas, mode="RGBA").save(rgba_path, format="PNG", optimize=True)
    Image.fromarray(nrm_u8).save(nrm_path, format="JPEG", quality=84)
    Image.fromarray(rgh_u8).save(rgh_path, format="JPEG", quality=80)
    return {"rgba": str(rgba_path), "normal": str(nrm_path), "roughness": str(rgh_path)}


def fetch_and_build_pbr_assets(
    out_dir: Path,
    *,
    offline: bool = False,
    cache_dir: Path | None = None,
    photo_path: Path | None = None,
) -> dict[str, Any]:
    """Fetch Poly Haven CC0 textures, harmonize to reference photo, and output pbr_manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        effective_cache = cache_dir
    else:
        local_cache = out_dir / "pbr_cache"
        shared_cache = ROOT / "artifacts/blender_village/pbr_cache"
        effective_cache = local_cache if local_cache.is_dir() and any(local_cache.iterdir()) else (
            shared_cache if shared_cache.is_dir() else local_cache
        )
    swatches_dir = out_dir / "swatches"
    swatches_dir.mkdir(parents=True, exist_ok=True)

    session: requests.Session | None = None
    if not offline:
        session = requests.Session()
        session.headers["User-Agent"] = "scene-data-demo-builder/1.0"

    sources_used: dict[str, str] = {}
    manifest: dict[str, Any] = {"version": 3, "swatches": {}, "foliage": {}, "sources": sources_used}

    def fetch_raw(asset_id: str, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alb, nrm, rgh, src = _load_or_fallback(session, asset_id, effective_cache, target_size=size, resolution="2k")
        sources_used[asset_id] = src
        return alb, nrm, rgh

    # 1. Plasters: seamless whitewashed lime plaster (plastered_wall), buff stucco (beige_wall_001), light grey mortar
    pw_alb, pw_nrm, pw_rgh = fetch_raw("plastered_wall", 1024)
    bw_alb, bw_nrm, bw_rgh = fetch_raw("beige_wall_001", 1024)
    pg_alb, pg_nrm, pg_rgh = fetch_raw("grey_plaster_02", 1024)
    manifest["swatches"]["plaster_white"] = _save_jpg_triplet(
        swatches_dir,
        "plaster_white",
        _regrade_rgb(pw_alb, target_mean=(228.0, 224.0, 214.0), blend_strength=0.88, sat_scale=0.32, contrast=0.52),
        pw_nrm,
        np.clip(pw_rgh.astype(np.float32) * 1.04, 155, 246).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )
    manifest["swatches"]["plaster_warm"] = _save_jpg_triplet(
        swatches_dir,
        "plaster_warm",
        _regrade_rgb(bw_alb, target_mean=(210.0, 200.0, 184.0), blend_strength=0.86, sat_scale=0.45, contrast=0.58),
        bw_nrm,
        np.clip(bw_rgh.astype(np.float32) * 1.06, 160, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )
    manifest["swatches"]["plaster_grey"] = _save_jpg_triplet(
        swatches_dir,
        "plaster_grey",
        _regrade_rgb(pg_alb, target_mean=(184.0, 182.0, 176.0), blend_strength=0.84, sat_scale=0.30, contrast=0.46),
        pg_nrm,
        np.clip(pg_rgh.astype(np.float32) * 1.06, 160, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )

    # 2. Northern weathered clay brick (muted sun-faded brick without dark mortar noise)
    br_alb, br_nrm, br_rgh = fetch_raw("brick_wall_006", 1024)
    manifest["swatches"]["red_brick"] = _save_jpg_triplet(
        swatches_dir,
        "red_brick",
        _regrade_rgb(br_alb, target_mean=(156.0, 118.0, 98.0), blend_strength=0.78, sat_scale=0.62, contrast=0.72),
        br_nrm,
        np.clip(br_rgh.astype(np.float32) * 1.05, 165, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )

    # 3. Aged sunlit monolithic concrete roof slab & bright poured rural concrete aprons
    cr_alb, cr_nrm, cr_rgh = fetch_raw("concrete_floor_worn_001", 1024)
    manifest["swatches"]["concrete_roof"] = _save_jpg_triplet(
        swatches_dir,
        "concrete_roof",
        _regrade_rgb(cr_alb, target_mean=(168.0, 165.0, 158.0), blend_strength=0.82, sat_scale=0.35, contrast=0.82),
        cr_nrm,
        np.clip(cr_rgh.astype(np.float32) * 1.02, 145, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )
    cp_alb, cp_nrm, cp_rgh = fetch_raw("concrete_floor_02", 1024)
    cp_graded = _regrade_rgb(cp_alb, target_mean=(198.0, 193.0, 182.0), blend_strength=0.86, sat_scale=0.35, contrast=0.58).astype(np.float32)
    # Subtle poured-slab control/expansion seams every 512 px
    seam = np.zeros((1024, 1024), dtype=np.float32)
    seam[[0, 1, 511, 512], :] = 1.0
    seam[:, [0, 1, 511, 512]] = 1.0
    cp_graded *= 1.0 - 0.12 * seam[:, :, None]
    seam_nrm = _height_to_normal_rgb(1.0 - seam * 0.35, strength=1.2)
    cp_nrm_mix = np.clip(cp_nrm.astype(np.float32) * 0.85 + seam_nrm.astype(np.float32) * 0.15, 0, 255).astype(np.uint8)
    manifest["swatches"]["concrete_apron"] = _save_jpg_triplet(
        swatches_dir,
        "concrete_apron",
        np.clip(cp_graded, 14, 245).astype(np.uint8),
        cp_nrm_mix,
        np.clip(cp_rgh.astype(np.float32) * 1.08 + 18.0, 190, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )

    # 4. Corrugated galvanized steel & dusty slate-blue steel roof sheets
    ci_alb, ci_nrm, ci_rgh = fetch_raw("corrugated_iron", 1024)
    manifest["swatches"]["corrugated_silver"] = _save_jpg_triplet(
        swatches_dir,
        "corrugated_silver",
        _regrade_rgb(ci_alb, target_mean=(196.0, 196.0, 192.0), blend_strength=0.88, sat_scale=0.25, contrast=0.68),
        ci_nrm,
        np.clip(ci_rgh.astype(np.float32) * 0.85 + 72.0, 132, 224).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )
    manifest["swatches"]["corrugated_blue"] = _save_jpg_triplet(
        swatches_dir,
        "corrugated_blue",
        _regrade_rgb(ci_alb, target_mean=(94.0, 120.0, 144.0), blend_strength=0.92, sat_scale=0.58, contrast=0.68),
        ci_nrm,
        np.clip(ci_rgh.astype(np.float32) * 0.88 + 80.0, 140, 232).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )

    # 5. Terracotta roof tile & dark weathered charcoal slate roof tile
    tt_alb, tt_nrm, tt_rgh = fetch_raw("clay_roof_tiles_02", 1024)
    manifest["swatches"]["terracotta_tile"] = _save_jpg_triplet(
        swatches_dir,
        "terracotta_tile",
        _regrade_rgb(tt_alb, target_mean=(138.0, 94.0, 74.0), blend_strength=0.78, sat_scale=0.62, contrast=0.78),
        tt_nrm,
        np.clip(tt_rgh.astype(np.float32) * 1.04, 150, 245).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )
    manifest["swatches"]["slate_roof"] = _save_jpg_triplet(
        swatches_dir,
        "slate_roof",
        _regrade_rgb(tt_alb, target_mean=(74.0, 76.0, 78.0), blend_strength=0.92, sat_scale=0.14, contrast=0.65),
        tt_nrm,
        np.clip(tt_rgh.astype(np.float32) * 1.06 + 18.0, 178, 248).astype(np.uint8),
        normal_size=1024,
        rough_size=512,
    )

    # 6. Clean leaf-free road asphalt + 2x2 tiled 2K marked road strip
    as_alb, as_nrm, as_rgh = fetch_raw("asphalt_02", 2048)
    as_graded = _regrade_rgb(as_alb, target_mean=(136.0, 134.0, 130.0), blend_strength=0.88, sat_scale=0.20, contrast=0.84)
    as_rgh_graded = np.clip(as_rgh.astype(np.float32) * 1.06 + 22.0, 170, 246).astype(np.uint8)
    manifest["swatches"]["road_asphalt"] = _save_jpg_triplet(
        swatches_dir,
        "road_asphalt",
        cv2.resize(as_graded, (1024, 1024), interpolation=cv2.INTER_AREA),
        as_nrm,
        as_rgh_graded,
        normal_size=1024,
        rough_size=512,
    )
    manifest["swatches"]["road_asphalt_marked"] = _build_road_asphalt_marked(swatches_dir, as_graded, as_nrm, as_rgh_graded)

    # 7. Bark & ribbed rolling shop shutters
    bk_alb, bk_nrm, bk_rgh = fetch_raw("bark_brown_01", 1024)
    manifest["swatches"]["tree_bark"] = _save_jpg_triplet(
        swatches_dir,
        "tree_bark",
        _regrade_rgb(bk_alb, target_mean=(78.0, 68.0, 54.0), blend_strength=0.70, sat_scale=0.75, contrast=1.10),
        bk_nrm,
        np.clip(bk_rgh.astype(np.float32), 185, 250).astype(np.uint8),
        normal_size=512,
        rough_size=256,
    )

    sh_alb, sh_nrm, sh_rgh = fetch_raw("worn_shutter", 1024)
    sh_a, sh_n, sh_r = _enhance_shop_shutter(sh_alb, sh_nrm, sh_rgh)
    manifest["swatches"]["shop_shutter"] = _save_jpg_triplet(
        swatches_dir,
        "shop_shutter",
        sh_a,
        sh_n,
        sh_r,
        normal_size=1024,
        rough_size=512,
    )

    # 8. 4-variant photorealistic casement window glazing, rural storefront signboards & botanical foliage cards
    manifest["swatches"]["window_pane_photo"] = _build_window_pane_swatch(swatches_dir, size=1024)
    manifest["swatches"].update(_build_shop_sign_swatches(swatches_dir))

    leaf_pal, twig_rgb = _sample_foliage_palette_from_photo(photo_path=photo_path)
    manifest["foliage"]["leaf_poplar"] = _generate_foliage_card(swatches_dir, "leaf_poplar", leaf_pal, twig_rgb, kind="poplar", seed=101)
    manifest["foliage"]["leaf_broadleaf"] = _generate_foliage_card(swatches_dir, "leaf_broadleaf", leaf_pal, twig_rgb, kind="broadleaf", seed=202)
    manifest["foliage"]["shrub"] = _generate_foliage_card(swatches_dir, "shrub", leaf_pal, twig_rgb, kind="shrub", seed=303)

    manifest_path = swatches_dir / "pbr_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PBR assets ready -> {manifest_path} ({len(manifest['swatches'])} swatches, {len(manifest['foliage'])} foliage cards)")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village")
    p.add_argument("--offline", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    fetch_and_build_pbr_assets(args.out_dir.resolve(), offline=args.offline)
