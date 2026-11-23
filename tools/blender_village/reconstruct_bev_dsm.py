"""Reconstruct metric BEV orthophoto, nDSM heightmap, and observed building footprints on RTX 5090 via Depth Anything 3."""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/blender_village"),
    str(ROOT / "tools/scene_pipeline"),
]

from blender_bridge_addon import quat_wxyz_to_matrix  # noqa: E402
from layout_solver import (  # noqa: E402
    _extend_centerline,
    _in_allowed_village_zone,
    build_regularized_road,
    ensure_ccw_blender_xz,
    local_road_frame,
    make_oriented_quad_xz,
    regularize_footprint_to_grid,
    solve_building_layout,
)
from prepare_evidence import (  # noqa: E402
    BUILDING_VISUAL_OVERRIDES,
    classify_roof_archetype,
    parse_colmap_text,
    unproject_photo_pixels,
)


def _install_da3_optional_shims() -> None:
    """Register minimal stubs for unused DA3 optional deps (moviepy/evo/open3d) before importing da3.api."""
    for mod_name in ("moviepy", "moviepy.editor", "evo", "evo.core", "evo.core.trajectory", "open3d"):
        if mod_name in sys.modules:
            continue
        mod = types.ModuleType(mod_name)
        if mod_name == "evo.core.trajectory":
            mod.PosePath3D = object
        sys.modules[mod_name] = mod


def load_calibrated_multiview_set(
    *,
    colmap_dir: Path,
    images_dir: Path,
    extra_colmap_dir: Path | None,
    extra_images_dir: Path | None,
    ref_clean_path: Path,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Load calibrated camera frames (grouped by aspect ratio) + filtered COLMAP 3D metric points."""
    cams_m, imgs_m, xyz_m, _ = parse_colmap_text(colmap_dir)
    pts_all = [xyz_m]

    groups: list[dict[str, Any]] = []

    def _pack_item(path: Path, im: dict[str, Any], cmap: dict[int, dict], tag: str, base_weight: float) -> dict[str, Any]:
        c = cmap[im["cam_id"]]
        r_cw = np.asarray(quat_wxyz_to_matrix(im["quat_wxyz"]), dtype=np.float32)
        t_cw = np.asarray(im["t_cw"], dtype=np.float32)
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = r_cw
        ext[:3, 3] = t_cw
        ixt = np.array(
            [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        eye = -r_cw.T @ t_cw
        return {
            "tag": tag,
            "path": str(path),
            "name": im["name"],
            "extrinsic": ext,
            "intrinsic": ixt,
            "eye": eye,
            "is_ref": tag == "ref",
            "base_weight": base_weight,
        }

    # Group 1 (3:2): ref_clean + elevated H3 360 orbit keyframes
    h3_items: list[dict[str, Any]] = []
    ref_item = next((im for im in imgs_m if im["name"].startswith("ref/")), imgs_m[0])
    h3_items.append(_pack_item(ref_clean_path, ref_item, cams_m, "ref", 1.5))

    if extra_colmap_dir is not None and extra_images_dir is not None and (extra_colmap_dir / "images.txt").is_file():
        cams_h, imgs_h, xyz_h, _ = parse_colmap_text(extra_colmap_dir)
        pts_all.append(xyz_h)
        orb = sorted((im for im in imgs_h if not im["name"].startswith("ref/")), key=lambda x: x["name"])
        if orb:
            sel_idx = np.linspace(0, len(orb) - 1, min(12, len(orb)), dtype=int)
            for idx in sel_idx:
                im = orb[int(idx)]
                p = extra_images_dir / im["name"]
                if p.is_file():
                    h3_items.append(_pack_item(p, im, cams_h, "h3_orbit", 0.65))
    groups.append({"name": "aspect_3x2_ref_h3", "frames": h3_items})

    # Group 2 (16:9): clip1 overhead aerial pass (entire village at Y~29..37m)
    clip1 = sorted((im for im in imgs_m if im["name"].startswith("clip1")), key=lambda x: x["name"])
    c1_items: list[dict[str, Any]] = []
    if clip1:
        sel_idx = np.linspace(0, len(clip1) - 1, min(14, len(clip1)), dtype=int)
        for idx in sel_idx:
            im = clip1[int(idx)]
            p = images_dir / im["name"]
            if p.is_file():
                c1_items.append(_pack_item(p, im, cams_m, "clip1_aerial", 1.45))
    if c1_items:
        groups.append({"name": "aspect_16x9_clip1", "frames": c1_items})

    # Group 3 (16:9): elevated clip3 swoop frames (eye_y >= 11m for crisp street/central rooftops)
    clip3_elev = []
    for im in sorted((im for im in imgs_m if im["name"].startswith("clip3")), key=lambda x: x["name"]):
        r_cw = np.asarray(quat_wxyz_to_matrix(im["quat_wxyz"]), dtype=np.float64)
        eye = -r_cw.T @ np.asarray(im["t_cw"], dtype=np.float64)
        if eye[1] >= 11.0:
            clip3_elev.append(im)
    c3_items: list[dict[str, Any]] = []
    if clip3_elev:
        sel_idx = np.linspace(0, len(clip3_elev) - 1, min(10, len(clip3_elev)), dtype=int)
        for idx in sel_idx:
            im = clip3_elev[int(idx)]
            p = images_dir / im["name"]
            if p.is_file():
                c3_items.append(_pack_item(p, im, cams_m, "clip3_elev", 1.25))
    if c3_items:
        groups.append({"name": "aspect_16x9_clip3", "frames": c3_items})

    col_pts = np.vstack(pts_all)
    mask_pts = (
        (np.abs(col_pts[:, 0]) < 165.0)
        & (col_pts[:, 1] > -1.5)
        & (col_pts[:, 1] < 22.0)
        & (col_pts[:, 2] > -220.0)
        & (col_pts[:, 2] < 80.0)
    )
    return groups, col_pts[mask_pts].astype(np.float64)


def _huber_affine_depth_fit(zp: np.ndarray, zc: np.ndarray) -> tuple[float, float, float]:
    """Fit zc ≈ a * zp + b robustly via 6-step Huber IRLS; fallback to median scale if unstable."""
    s_med = float(np.median(zc / np.maximum(zp, 1e-4)))
    if len(zp) < 40 or float(np.std(zp)) < 1e-3:
        err = float(np.median(np.abs(zp * s_med - zc)))
        return s_med, 0.0, err

    A = np.column_stack([zp, np.ones_like(zp)])
    w = np.ones_like(zp)
    ab = np.array([s_med, 0.0], dtype=np.float64)
    for _ in range(6):
        Aw = A * w[:, None]
        ab_cand, *_ = np.linalg.lstsq(Aw, zc * w, rcond=None)
        if ab_cand[0] <= 0.1 * s_med or ab_cand[0] >= 3.0 * s_med:
            break
        ab = ab_cand
        res = np.abs(A @ ab - zc)
        w = 1.0 / np.maximum(res, 1.2)

    err_aff = float(np.median(np.abs(A @ ab - zc)))
    err_scale = float(np.median(np.abs(zp * s_med - zc)))
    if err_aff <= err_scale:
        return float(ab[0]), float(ab[1]), err_aff
    return s_med, 0.0, err_scale


def infer_dense_metric_pointcloud(
    frame_groups: list[dict[str, Any]],
    col_pts: np.ndarray,
    centerline_xz: np.ndarray,
    *,
    model_id: str = "depth-anything/DA3-LARGE-1.1",
    process_res: int = 672,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run pose-conditioned DA3 on RTX 5090, calibrate each view to COLMAP metric points, and unproject XYZ+RGB."""
    import torch

    _install_da3_optional_shims()
    import depth_anything_3.api as da3_api

    da3_api.align_poses_umeyama = lambda pred_ext, in_ext, *a, **kw: (None, None, 1.0, pred_ext)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = da3_api.DepthAnything3.from_pretrained(model_id).to(device)

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_w: list[np.ndarray] = []
    all_cam_id: list[np.ndarray] = []

    frame_stats: list[dict[str, Any]] = []
    global_cam_idx = 0

    for grp in frame_groups:
        frames = grp["frames"]
        if not frames:
            continue
        paths = [f["path"] for f in frames]
        exts = np.stack([f["extrinsic"] for f in frames])
        ixts = np.stack([f["intrinsic"] for f in frames])

        pred = model.inference(
            paths,
            extrinsics=exts,
            intrinsics=ixts,
            align_to_input_ext_scale=True,
            export_dir=None,
            process_res=process_res,
        )

        for fi, fmeta in enumerate(frames):
            dmap = pred.depth[fi].astype(np.float64)
            conf = pred.conf[fi].astype(np.float64)
            rgb_u8 = pred.processed_images[fi]
            H, W = dmap.shape
            K = pred.intrinsics[fi].astype(np.float64)
            ext = exts[fi].astype(np.float64)
            r_cw = ext[:3, :3]
            t_cw = ext[:3, 3]
            c2w = np.linalg.inv(ext)
            k_inv = np.linalg.inv(K)

            # Calibrate predicted depth against COLMAP sparse 3D points
            Xc = col_pts @ r_cw.T + t_cw
            z_col = Xc[:, 2]
            ok = (z_col > 8.0) & (z_col < 220.0)
            u_proj = Xc[:, 0] / np.maximum(z_col, 1e-4) * K[0, 0] + K[0, 2]
            v_proj = Xc[:, 1] / np.maximum(z_col, 1e-4) * K[1, 1] + K[1, 2]
            ui = np.round(u_proj).astype(int)
            vi = np.round(v_proj).astype(int)
            ok &= (ui >= 2) & (ui < W - 2) & (vi >= 2) & (vi < H - 2)
            if not fmeta["is_ref"]:
                ok &= ~((ui > 0.84 * W) & (vi > 0.86 * H))

            zp_raw = dmap[vi[ok], ui[ok]]
            cf_raw = conf[vi[ok], ui[ok]]
            zc_raw = z_col[ok]
            hi = (zp_raw > 1e-3) & (cf_raw >= np.percentile(cf_raw, 45) if len(cf_raw) > 20 else True)
            zp, zc = zp_raw[hi], zc_raw[hi]
            if len(zp) < 25:
                global_cam_idx += 1
                continue

            a_fit, b_fit, med_err = _huber_affine_depth_fit(zp, zc)
            d_metric = np.maximum(a_fit * dmap + b_fit, 0.5)

            # Check median vertical Y error on matched COLMAP anchors
            rays_chk = k_inv @ np.vstack([ui[ok][hi], vi[ok][hi], np.ones(len(zc))])
            xw_pred = (c2w[:3, :3] @ (rays_chk * (a_fit * zp + b_fit)[None, :])).T + c2w[:3, 3]
            xw_col = (c2w[:3, :3] @ (rays_chk * zc[None, :])).T + c2w[:3, 3]
            med_dy = float(np.median(np.abs(xw_pred[:, 1] - xw_col[:, 1])))

            # Fine road-plane vertical offset check (if enough centerline points are visible)
            cl_3d = np.column_stack([centerline_xz[:, 0], np.zeros(len(centerline_xz)), centerline_xz[:, 1]])
            xc_road = cl_3d @ r_cw.T + t_cw
            r_ok = (xc_road[:, 2] > 10.0) & (xc_road[:, 2] < 190.0)
            ru = np.round(xc_road[:, 0] / np.maximum(xc_road[:, 2], 1e-4) * K[0, 0] + K[0, 2]).astype(int)
            rv = np.round(xc_road[:, 1] / np.maximum(xc_road[:, 2], 1e-4) * K[1, 1] + K[1, 2]).astype(int)
            r_ok &= (ru >= 2) & (ru < W - 2) & (rv >= 2) & (rv < H - 2)
            road_y_med = 0.0
            if r_ok.any():
                rz = d_metric[rv[r_ok], ru[r_ok]]
                rrays = k_inv @ np.vstack([ru[r_ok], rv[r_ok], np.ones(int(r_ok.sum()))])
                rxw = (c2w[:3, :3] @ (rrays * rz[None, :])).T + c2w[:3, 3]
                road_y_med = float(np.median(rxw[:, 1]))

            # Skip weak or warped AI views if error is unusually high
            rel_weight = fmeta["base_weight"] / max(0.35, med_dy)
            if med_dy > 1.35:
                rel_weight *= 0.12

            # Dense unprojection of confident surface pixels
            us, vs = np.meshgrid(np.arange(W), np.arange(H))
            valid_px = (
                np.isfinite(d_metric)
                & (d_metric >= 8.0)
                & (d_metric <= 220.0)
                & (conf >= np.percentile(conf, 30))
            )
            if not fmeta["is_ref"]:
                valid_px &= ~((us > 0.84 * W) & (vs > 0.86 * H))

            vidx = np.flatnonzero(valid_px.reshape(-1))
            if len(vidx) > 0:
                pix_v = np.stack([us.reshape(-1)[vidx], vs.reshape(-1)[vidx], np.ones(len(vidx))], axis=-1)
                rays = (k_inv @ pix_v.T)  # (3, M)
                xc_dense = rays * d_metric.reshape(-1)[vidx][None, :]
                xw_dense = ((c2w[:3, :3] @ xc_dense).T + c2w[:3, 3]).astype(np.float32)

                # Steeper downward rays yield much sharper roof elevation measurements
                ray_dirs_w = (c2w[:3, :3] @ rays).T
                ray_norms = np.maximum(np.linalg.norm(ray_dirs_w, axis=1), 1e-6)
                steepness = np.clip(np.abs(ray_dirs_w[:, 1]) / ray_norms * 2.1, 0.20, 1.0).astype(np.float32)
                w_pts = (rel_weight * steepness * np.clip(conf.reshape(-1)[vidx] / 5.0, 0.4, 2.0)).astype(np.float32)

                in_box = (
                    (xw_dense[:, 0] >= -138.0)
                    & (xw_dense[:, 0] <= 132.0)
                    & (xw_dense[:, 1] >= -1.8)
                    & (xw_dense[:, 1] <= 17.5)
                    & (xw_dense[:, 2] >= -178.0)
                    & (xw_dense[:, 2] <= 72.0)
                )
                if in_box.any():
                    all_xyz.append(xw_dense[in_box])
                    all_rgb.append(rgb_u8.reshape(-1, 3)[vidx][in_box])
                    all_w.append(w_pts[in_box])
                    all_cam_id.append(np.full(int(in_box.sum()), global_cam_idx, dtype=np.int16))

            frame_stats.append(
                {
                    "group": grp["name"],
                    "tag": fmeta["tag"],
                    "name": fmeta["name"],
                    "colmap_anchors": len(zc),
                    "med_abs_err_m": round(med_err, 3),
                    "med_dy_m": round(med_dy, 3),
                    "road_y_med_m": round(road_y_med, 3),
                    "weight": round(rel_weight, 3),
                }
            )
            global_cam_idx += 1

    if not all_xyz:
        raise RuntimeError("No valid 3D points reconstructed from DA3 multi-view pass")

    pts_xyz = np.concatenate(all_xyz, axis=0)
    pts_rgb = np.concatenate(all_rgb, axis=0)
    pts_w = np.concatenate(all_w, axis=0)
    pts_cam = np.concatenate(all_cam_id, axis=0)

    report = {
        "model_id": model_id,
        "device": device,
        "process_res": process_res,
        "num_frames_calibrated": len(frame_stats),
        "total_dense_points": int(len(pts_xyz)),
        "median_frame_depth_err_m": round(float(np.median([s["med_abs_err_m"] for s in frame_stats])), 3),
        "median_frame_dy_m": round(float(np.median([s["med_dy_m"] for s in frame_stats])), 3),
        "frame_stats": frame_stats,
    }
    return pts_xyz, pts_rgb, pts_w, pts_cam, report


def rasterize_bev_dsm(
    pts_xyz: np.ndarray,
    pts_rgb: np.ndarray,
    pts_w: np.ndarray,
    pts_cam: np.ndarray,
    centerline_xz: np.ndarray,
    *,
    bounds_xz: tuple[float, float, float, float] = (-135.0, 130.0, -175.0, 70.0),
    res_m: float = 0.25,
) -> dict[str, Any]:
    """Project dense 3D points onto metric BEV grid to compute orthophoto, DTM, nDSM, and vegetation mask."""
    bx0, bx1, bz0, bz1 = bounds_xz
    gw = int(round((bx1 - bx0) / res_m))
    gh = int(round((bz1 - bz0) / res_m))

    ix = np.floor((pts_xyz[:, 0] - bx0) / res_m).astype(np.int32)
    iz = np.floor((pts_xyz[:, 2] - bz0) / res_m).astype(np.int32)
    valid = (ix >= 0) & (ix < gw) & (iz >= 0) & (iz < gh)
    ix = ix[valid]
    iz = iz[valid]
    y_v = pts_xyz[valid, 1].astype(np.float32)
    rgb_v = pts_rgb[valid]
    w_v = pts_w[valid]
    cam_v = pts_cam[valid].astype(np.int32)

    flat = iz.astype(np.int64) * gw + ix.astype(np.int64)
    # Sort primarily by cell index, secondarily by vertical height Y
    order = np.lexsort((y_v, flat))
    flat_s = flat[order]
    y_s = y_v[order]
    rgb_s = rgb_v[order]
    w_s = w_v[order]
    cam_s = cam_v[order]

    uniq_cells, starts, counts = np.unique(flat_s, return_index=True, return_counts=True)

    dsm_flat = np.full(gh * gw, np.nan, dtype=np.float32)
    low_flat = np.full(gh * gw, np.nan, dtype=np.float32)
    ortho_flat = np.zeros((gh * gw, 3), dtype=np.uint8)
    obs_views_flat = np.zeros(gh * gw, dtype=np.uint8)

    # Vectorized percentile slicing for DSM (p78) and ground hint (p20)
    idx_p78 = starts + np.minimum(counts - 1, np.floor(counts * 0.78).astype(np.int64))
    idx_p20 = starts + np.minimum(counts - 1, np.floor(counts * 0.20).astype(np.int64))
    dsm_flat[uniq_cells] = y_s[idx_p78]
    low_flat[uniq_cells] = y_s[idx_p20]

    # Pick weighted-best RGB color near top surface & count distinct cameras per cell
    # Vectorized bitmask across <=64 cameras cleanly computes unique views per cell in O(N)
    cam_bits = (np.uint64(1) << np.clip(cam_s, 0, 62).astype(np.uint64))
    # Segment bitwise OR via Python slice loop only over sparse nonempty cells? Even faster: iterate ufunc reduceat!
    or_mask = np.bitwise_or.reduceat(cam_bits, starts)
    # Count popcount cleanly in NumPy via uint64 byte LUT
    byte_pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    view_counts = byte_pop[or_mask.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.uint8)
    obs_views_flat[uniq_cells] = view_counts

    # Weighted RGB per cell favoring points near the p78 roof elevation & high camera weight
    top_ref = np.repeat(y_s[idx_p78], counts)
    near_top = np.abs(y_s - top_ref) <= 1.25
    pw = np.where(near_top, w_s, w_s * 0.05).astype(np.float32)
    w_sum = np.add.reduceat(pw, starts)
    r_sum = np.add.reduceat(rgb_s[:, 0].astype(np.float32) * pw, starts)
    g_sum = np.add.reduceat(rgb_s[:, 1].astype(np.float32) * pw, starts)
    b_sum = np.add.reduceat(rgb_s[:, 2].astype(np.float32) * pw, starts)
    inv_w = 1.0 / np.maximum(w_sum, 1e-4)
    ortho_flat[uniq_cells, 0] = np.clip(r_sum * inv_w, 0, 255).astype(np.uint8)
    ortho_flat[uniq_cells, 1] = np.clip(g_sum * inv_w, 0, 255).astype(np.uint8)
    ortho_flat[uniq_cells, 2] = np.clip(b_sum * inv_w, 0, 255).astype(np.uint8)

    dsm = dsm_flat.reshape(gh, gw)
    low_y = low_flat.reshape(gh, gw)
    bev_ortho = ortho_flat.reshape(gh, gw, 3)
    obs_views = obs_views_flat.reshape(gh, gw)

    # Interpolate empty holes gently for continuous morphological DTM estimation
    valid_mask = np.isfinite(dsm)
    _, nearest_idx = cv2.distanceTransformWithLabels((~valid_mask).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    # Build seamless filled ground estimate using low_y
    low_valid = np.isfinite(low_y)
    low_filled = np.where(low_valid, low_y, 0.0).astype(np.float32)
    # Simple iterative diffusion on missing low_y cells, anchored by morphological opening (13m kernel erases buildings)
    if not low_valid.all():
        mean_ground = float(np.nanmedian(low_y))
        low_filled = np.where(low_valid, low_y, mean_ground).astype(np.float32)
    k_dtm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (53, 53))
    dtm = cv2.morphologyEx(low_filled, cv2.MORPH_OPEN, k_dtm)
    dtm = cv2.GaussianBlur(dtm, (41, 41), 12.0)

    # Pin DTM toward Y = 0 along calibrated road corridor
    gx = bx0 + (np.arange(gw, dtype=np.float64) + 0.5) * res_m
    gz = bz0 + (np.arange(gh, dtype=np.float64) + 0.5) * res_m
    xx, zz = np.meshgrid(gx, gz)
    # Vectorized extended-road distance mask
    cl, _ = _extend_centerline(centerline_xz, ext_m=180.0)
    min_d2 = np.full((gh, gw), 1e9, dtype=np.float64)
    for a, b in zip(cl, cl[1:]):
        dv = b - a
        l2 = float(np.dot(dv, dv))
        t = np.clip(((xx - a[0]) * dv[0] + (zz - a[1]) * dv[1]) / max(l2, 1e-8), 0.0, 1.0)
        dx = xx - (a[0] + t * dv[0])
        dz = zz - (a[1] + t * dv[1])
        min_d2 = np.minimum(min_d2, dx * dx + dz * dz)
    road_dist_m = np.sqrt(min_d2)
    pin_alpha = np.clip((10.0 - road_dist_m) / 5.0, 0.0, 1.0).astype(np.float32)
    dtm = dtm * (1.0 - pin_alpha * 0.7)

    ndsm = np.where(valid_mask, np.clip(dsm - dtm, 0.0, 16.0), 0.0).astype(np.float32)

    # Detect tree canopy via green excess + HSV green foliage + rough crown variance
    rf = bev_ortho[:, :, 0].astype(np.float32)
    gf = bev_ortho[:, :, 1].astype(np.float32)
    bf = bev_ortho[:, :, 2].astype(np.float32)
    exg = 2.0 * gf - rf - bf
    hsv = cv2.cvtColor(bev_ortho, cv2.COLOR_RGB2HSV)
    # Local height standard deviation highlights fuzzy leaves vs flat planes
    mean_5 = cv2.blur(ndsm, (5, 5))
    sq_5 = cv2.blur(ndsm * ndsm, (5, 5))
    std_5 = np.sqrt(np.maximum(sq_5 - mean_5 * mean_5, 0.0))

    green_px = ((exg > 18.0) & (gf > rf + 3.0) & (gf > bf + 3.0)) | (
        (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 95) & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] <= 170) & (exg > 6.0)
    )
    windbreak_band = zz < (-142.0 - 0.17 * np.maximum(0.0, xx))
    veg_mask = (green_px & (std_5 > 0.35)) | windbreak_band | (green_px & (ndsm > 4.5))
    veg_mask = cv2.dilate(veg_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    return {
        "bounds_xz": bounds_xz,
        "res_m": res_m,
        "dsm": dsm,
        "dtm": dtm,
        "ndsm": ndsm,
        "ortho": bev_ortho,
        "obs_views": obs_views,
        "veg_mask": veg_mask,
        "road_dist_m": road_dist_m,
        "xx": xx,
        "zz": zz,
    }


def _world_poly_to_bev_px(
    poly_xz: np.ndarray,
    bounds_xz: tuple[float, float, float, float],
    gw: int,
    gh: int,
) -> np.ndarray:
    bx0, bx1, bz0, bz1 = bounds_xz
    px = np.clip((poly_xz[:, 0] - bx0) / (bx1 - bx0) * (gw - 1), 0, gw - 1)
    pz = np.clip((poly_xz[:, 1] - bz0) / (bz1 - bz0) * (gh - 1), 0, gh - 1)
    return np.stack([px, pz], axis=-1).astype(np.int32)


def extract_bev_buildings_and_calibrate_core(
    bev: dict[str, Any],
    layout: dict[str, Any],
    road_info: dict[str, Any],
    centerline_xz: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], np.ndarray]:
    """Calibrate core 64 building heights/rays against BEV nDSM and extract additional observed buildings."""
    ndsm = bev["ndsm"]
    ortho = bev["ortho"]
    obs_views = bev["obs_views"]
    veg_mask = bev["veg_mask"]
    bounds_xz = bev["bounds_xz"]
    res_m = float(bev["res_m"])
    gh, gw = ndsm.shape

    apron_mask = (bev["road_dist_m"] <= float(road_info["corridor_half_width"]) + 0.25).astype(np.uint8) * 255

    # Part A: Calibrate Core 64 buildings using multi-view BEV nDSM heightfield
    core_calibrations: dict[str, dict[str, Any]] = {}
    core_records_raw: list[dict[str, Any]] = []
    core_occupied_mask = np.zeros((gh, gw), dtype=np.uint8)
    k_dil_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for b in layout["buildings"]:
        bid = b["id"]
        roof_uv = np.asarray(b["roof"], dtype=np.float64)
        h_prior = float(b["h"])
        override = BUILDING_VISUAL_OVERRIDES.get(bid, {})
        has_fixed_h = "height_m" in override

        best_h = float(override.get("height_m", h_prior))
        best_score = -1e9
        best_pts_xz = ensure_ccw_blender_xz(unproject_photo_pixels(layout, roof_uv, height_y=best_h)[:, [0, 2]])
        prior_score = None

        cands = [best_h] if has_fixed_h else np.arange(max(2.8, h_prior - 1.5), min(8.2, h_prior + 1.55), 0.25).tolist()
        if not has_fixed_h and h_prior not in cands:
            cands.append(h_prior)

        for hc in cands:
            pts3d = unproject_photo_pixels(layout, roof_uv, height_y=float(hc))
            pxz = ensure_ccw_blender_xz(pts3d[:, [0, 2]])
            m_in = np.zeros((gh, gw), dtype=np.uint8)
            cv2.fillPoly(m_in, [_world_poly_to_bev_px(pxz, bounds_xz, gw, gh)], 255)
            inside = m_in > 0
            if int(inside.sum()) < 16:
                continue
            obs_mean = float(obs_views[inside].mean())
            if obs_mean < 1.2:
                continue
            m_ring = (cv2.dilate(m_in, np.ones((9, 9), np.uint8)) > 0) & (~inside)
            in_h = ndsm[inside]
            ring_h = ndsm[m_ring] if m_ring.any() else np.array([0.0], dtype=np.float32)
            p70_in = float(np.percentile(in_h, 70))
            score = float(in_h.mean()) - 0.45 * float(ring_h.mean()) - 0.35 * abs(p70_in - float(hc))
            if abs(float(hc) - h_prior) < 1e-4:
                prior_score = score
            if score > best_score:
                best_score = score
                best_h = float(hc)
                best_pts_xz = pxz

        # Only override h_prior when BEV evidence strongly agrees and shift stays safe (< 4.5m)
        orig_pts = unproject_photo_pixels(layout, roof_uv, height_y=h_prior)[:, [0, 2]]
        shift_m = float(np.linalg.norm(best_pts_xz.mean(0) - orig_pts.mean(0)))
        accept_cal = (not has_fixed_h) and (prior_score is not None) and (best_score > prior_score + 0.30) and (shift_m <= 4.5)
        final_h = best_h if accept_cal else float(override.get("height_m", h_prior))
        final_xz = best_pts_xz if accept_cal else ensure_ccw_blender_xz(orig_pts)

        # Avoid flipping a core polygon across centerline during BEV shift
        if np.dot(local_road_frame(final_xz.mean(0), centerline_xz)[1], local_road_frame(orig_pts.mean(0), centerline_xz)[1]) <= 0:
            final_h = h_prior
            final_xz = ensure_ccw_blender_xz(orig_pts)
            accept_cal = False

        core_calibrations[bid] = {
            "h_prior_m": round(h_prior, 3),
            "h_calibrated_m": round(final_h, 3),
            "accepted": bool(accept_cal),
            "ray_shift_m": round(shift_m, 2),
        }
        reg_xz = regularize_footprint_to_grid(final_xz, centerline_xz)
        core_records_raw.append(
            {
                "id": bid,
                "index": len(core_records_raw),
                "is_core": True,
                "height_m": final_h,
                "footprint_xz": reg_xz.tolist(),
            }
        )

    solved_core = solve_building_layout(
        core_records_raw,
        centerline_xz,
        road_info["road_polygon_xz"],
        road_info["road_apron_polygon_xz"],
        drop_unresolvable_synthetic=False,
    )
    for sc in solved_core:
        cv2.fillPoly(core_occupied_mask, [_world_poly_to_bev_px(np.asarray(sc["footprint_xz"]), bounds_xz, gw, gh)], 255)

    core_occupied_mask = cv2.dilate(core_occupied_mask, k_dil_core)

    # Part B: Segment new observed rooftops from nDSM not covered by core buildings / trees / road apron
    cand = (
        (ndsm >= 1.85)
        & (ndsm <= 11.5)
        & (~veg_mask)
        & (apron_mask == 0)
        & (core_occupied_mask == 0)
        & (obs_views >= 1)
    ).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Watershed splitting: mark background=1, unknown cand pixels=0, peaks=2..K!
    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 5)
    peak_dil = cv2.dilate(dist, np.ones((17, 17), np.uint8))
    peaks = ((dist == peak_dil) & (dist >= 5.0)).astype(np.uint8)
    n_cc, cc_lbl = cv2.connectedComponents(cand)
    for cc_id in range(1, n_cc):
        comp_m = cc_lbl == cc_id
        if int(comp_m.sum()) >= 240 and int(peaks[comp_m].sum()) == 0:
            ys_c, xs_c = np.where(comp_m)
            best_pos = int(np.argmax(dist[ys_c, xs_c]))
            peaks[ys_c[best_pos], xs_c[best_pos]] = 1

    _, peak_lbl = cv2.connectedComponents(peaks)
    markers = np.zeros(cand.shape, dtype=np.int32)
    markers[cand == 0] = 1
    markers[peaks > 0] = peak_lbl[peaks > 0] + 1
    # Use nDSM height gradient magnitude to sharpen watershed boundaries between roofs
    gx = cv2.Sobel(ndsm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ndsm, cv2.CV_32F, 0, 1, ksize=3)
    grad_u8 = np.clip(np.hypot(gx, gy) * 35.0, 0, 255).astype(np.uint8)
    grad_rgb = np.stack([grad_u8, grad_u8, grad_u8], axis=-1)
    ws = cv2.watershed(grad_rgb, markers)

    bx0, bx1, bz0, bz1 = bounds_xz
    extracted: list[dict[str, Any]] = []

    def _emit_region(pts_world_xz: np.ndarray, reg_mask: np.ndarray) -> None:
        c = pts_world_xz.mean(0)
        if not _in_allowed_village_zone(float(c[0]), float(c[1])):
            return
        hs = ndsm[reg_mask]
        if len(hs) < 60:
            return
        area_m2 = len(hs) * (res_m * res_m)
        if area_m2 < 16.0:
            return
        _, _, u_loc, n_loc, _ = local_road_frame(c, centerline_xz)
        pu = (pts_world_xz - c) @ u_loc
        pn = (pts_world_xz - c) @ n_loc
        w_m = float(np.percentile(pu, 96) - np.percentile(pu, 4))
        d_m = float(np.percentile(pn, 96) - np.percentile(pn, 4))
        if min(w_m, d_m) < 3.4:
            return

        h_obs = float(np.clip(np.percentile(hs, 72), 2.9, 7.2))
        rgb_med = np.median(ortho[reg_mask].astype(np.float32), axis=0)
        arch, _ = classify_roof_archetype(None, rgb_med, f"bev_{len(extracted)}")
        views_med = int(round(float(np.median(obs_views[reg_mask]))))

        # Split unusually large blobs along long axis into individual northern terraced row bays
        if w_m > 16.5 or d_m > 12.5:
            n_splits_u = max(1, int(round(w_m / 10.5))) if w_m > 16.5 else 1
            n_splits_n = max(1, int(round(d_m / 9.5))) if d_m > 12.5 else 1
            sub_w = min(13.5, max(5.5, (w_m - (n_splits_u - 1) * 0.7) / n_splits_u))
            sub_d = min(10.5, max(4.8, (d_m - (n_splits_n - 1) * 0.8) / n_splits_n))
            for iu in range(n_splits_u):
                off_u = ((iu + 0.5) / n_splits_u - 0.5) * w_m
                for i_n in range(n_splits_n):
                    off_n = ((i_n + 0.5) / n_splits_n - 0.5) * d_m
                    sc = c + u_loc * off_u + n_loc * off_n
                    if not _in_allowed_village_zone(float(sc[0]), float(sc[1])):
                        continue
                    q = make_oriented_quad_xz(float(sc[0]), float(sc[1]), sub_w, sub_d, u_loc)
                    extracted.append(
                        {
                            "footprint_xz": [[round(float(x), 3), round(float(z), 3)] for x, z in q],
                            "height_m": round(h_obs, 3),
                            "archetype": arch,
                            "observed_roof_rgb": [round(float(x), 1) for x in rgb_med],
                            "obs_views": views_med,
                            "area_m2": round(sub_w * sub_d, 1),
                        }
                    )
        else:
            q = make_oriented_quad_xz(float(c[0]), float(c[1]), min(15.5, max(3.6, w_m)), min(11.5, max(3.6, d_m)), u_loc)
            extracted.append(
                {
                    "footprint_xz": [[round(float(x), 3), round(float(z), 3)] for x, z in q],
                    "height_m": round(h_obs, 3),
                    "archetype": arch,
                    "observed_roof_rgb": [round(float(x), 1) for x in rgb_med],
                    "obs_views": views_med,
                    "area_m2": round(area_m2, 1),
                }
            )

    unique_labels = [int(lbl) for lbl in np.unique(ws) if lbl >= 2]
    for lbl in unique_labels:
        rmask = (ws == lbl) & (cand > 0)
        ys_idx, xs_idx = np.where(rmask)
        if len(ys_idx) < 55:
            continue
        pts_w = np.stack([bx0 + (xs_idx + 0.5) * res_m, bz0 + (ys_idx + 0.5) * res_m], axis=-1)
        _emit_region(pts_w, rmask)

    # Also emit overlay preview for visual QC (red = road+apron contour, green = solved core, yellow = new BEV buildings)
    overlay = ortho.copy()
    apron_cnts, _ = cv2.findContours(apron_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, apron_cnts, -1, (255, 70, 70), 2)
    for sc in solved_core:
        cv2.polylines(overlay, [_world_poly_to_bev_px(np.asarray(sc["footprint_xz"]), bounds_xz, gw, gh)], True, (70, 255, 110), 2)
    for item in extracted:
        cv2.polylines(overlay, [_world_poly_to_bev_px(np.asarray(item["footprint_xz"]), bounds_xz, gw, gh)], True, (255, 220, 40), 2)

    return core_calibrations, extracted, overlay


def run_bev_dsm_pipeline(
    *,
    colmap_dir: Path,
    images_dir: Path,
    extra_colmap_dir: Path | None,
    extra_images_dir: Path | None,
    out_dir: Path,
    model_id: str = "depth-anything/DA3-LARGE-1.1",
) -> dict[str, Any]:
    """Execute end-to-end Route B BEV DSM reconstruction and save outputs to out_dir/bev_dsm."""
    bev_dir = out_dir / "bev_dsm"
    bev_dir.mkdir(parents=True, exist_ok=True)
    layout = json.loads((ROOT / "assets/village_studio/layout.json").read_text(encoding="utf-8"))

    # Reuse watermark-cleaned reference image from prepare_evidence (or create if running standalone)
    ref_clean_path = out_dir / "ref_clean.png"
    if not ref_clean_path.is_file():
        from minimax_h3_orbit import clean_watermark

        with Image.open(ROOT / "assets/scene_sources/village_road_near.png") as im:
            clean_watermark(im.convert("RGB"), box=(0.85, 0.87, 1.0, 1.0)).save(ref_clean_path)

    centerline_uv = [[0, 765], [400, 604], [800, 448], [1200, 307], [1536, 229]]
    centerline_xz = unproject_photo_pixels(layout, centerline_uv, height_y=0.0)[:, [0, 2]]
    road_info = build_regularized_road(centerline_xz, carriageway_half_width=4.4, apron_width=3.2)

    frame_groups, col_pts = load_calibrated_multiview_set(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        extra_colmap_dir=extra_colmap_dir,
        extra_images_dir=extra_images_dir,
        ref_clean_path=ref_clean_path,
    )

    pts_xyz, pts_rgb, pts_w, pts_cam, dsm_report = infer_dense_metric_pointcloud(
        frame_groups, col_pts, centerline_xz, model_id=model_id
    )

    bev = rasterize_bev_dsm(pts_xyz, pts_rgb, pts_w, pts_cam, centerline_xz)
    core_cal, bev_buildings, overlay_rgb = extract_bev_buildings_and_calibrate_core(
        bev, layout, road_info, centerline_xz
    )

    # Save visual & structured artifacts
    ortho_path = bev_dir / "bev_ortho.png"
    ndsm_path = bev_dir / "bev_ndsm.npy"
    ndsm_vis_path = bev_dir / "bev_ndsm_vis.png"
    overlay_path = bev_dir / "bev_candidates_overlay.png"
    buildings_path = bev_dir / "bev_buildings.json"
    report_path = bev_dir / "bev_dsm_report.json"

    Image.fromarray(bev["ortho"]).save(ortho_path)
    np.save(ndsm_path, bev["ndsm"].astype(np.float16))
    np.save(bev_dir / "bev_obs_views.npy", bev["obs_views"])

    ndsm_norm = np.clip(bev["ndsm"] / 9.5 * 255.0, 0, 255).astype(np.uint8)
    ndsm_color_bgr = cv2.applyColorMap(ndsm_norm, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(ndsm_vis_path), ndsm_color_bgr)
    Image.fromarray(overlay_rgb).save(overlay_path)

    payload = {
        "bounds_xz": list(bev["bounds_xz"]),
        "res_m": bev["res_m"],
        "ortho_path": str(ortho_path.resolve()),
        "ndsm_path": str(ndsm_path.resolve()),
        "obs_views_path": str((bev_dir / "bev_obs_views.npy").resolve()),
        "core_calibrations": core_cal,
        "extracted_buildings": bev_buildings,
    }
    buildings_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    accepted_core = sum(1 for v in core_cal.values() if v["accepted"])
    summary = {
        **{k: v for k, v in dsm_report.items() if k != "frame_stats"},
        "core_calibrated_count": accepted_core,
        "extracted_buildings_count": len(bev_buildings),
        "bev_ortho": str(ortho_path),
        "bev_ndsm_vis": str(ndsm_vis_path),
        "bev_candidates_overlay": str(overlay_path),
        "bev_buildings_json": str(buildings_path),
        "frame_stats": dsm_report["frame_stats"],
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"BEV_DSM_OK: {dsm_report['total_dense_points']} pts, med_err={dsm_report['median_frame_depth_err_m']}m, "
        f"med_dY={dsm_report['median_frame_dy_m']}m, core_cal={accepted_core}, extracted={len(bev_buildings)} -> {bev_dir}"
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--colmap-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/sparse/0")
    p.add_argument("--images-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/images")
    p.add_argument("--extra-colmap-dir", type=Path, default=ROOT / "artifacts/blender_village/h3_colmap/colmap/sparse/0")
    p.add_argument("--extra-images-dir", type=Path, default=ROOT / "artifacts/blender_village/h3_colmap/colmap/images")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village")
    p.add_argument("--model-id", default="depth-anything/DA3-LARGE-1.1")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_bev_dsm_pipeline(
        colmap_dir=args.colmap_dir.resolve(),
        images_dir=args.images_dir.resolve(),
        extra_colmap_dir=args.extra_colmap_dir.resolve() if args.extra_colmap_dir else None,
        extra_images_dir=args.extra_images_dir.resolve() if args.extra_images_dir else None,
        out_dir=args.out_dir.resolve(),
        model_id=args.model_id,
    )
