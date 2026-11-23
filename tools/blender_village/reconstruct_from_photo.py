"""General single-drone-photo -> 3D Blender PBR village reconstruction pipeline."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "tools/scene_pipeline"),
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/blender_village"),
]

from bake_multiview import bake_atlases  # noqa: E402
from blender_bridge_addon import (  # noqa: E402
    colmap_intrinsics_to_blender_camera,
    colmap_pose_to_blender_matrix,
    point_in_polygon_2d,
)
from bridge_client import run_blender_script  # noqa: E402
from fetch_pbr_assets import fetch_and_build_pbr_assets  # noqa: E402
from layout_solver import (  # noqa: E402
    audit_layout_geometry,
    build_regularized_road,
    ensure_ccw_blender_xz,
    local_road_frame,
    make_oriented_quad_xz,
    regularize_footprint_to_grid,
    solve_building_layout,
)
from prepare_evidence import _edge_neighbor_gap, build_courtyard_network, laplacian_sharpness  # noqa: E402
from reconstruct_bev_dsm import _huber_affine_depth_fit, _install_da3_optional_shims  # noqa: E402
from synthesize_multiview import synthesize_novel_views  # noqa: E402
from visual_qc import run_visual_qc  # noqa: E402


def run_da3_single_photo(
    photo_path: Path,
    *,
    model_id: str = "depth-anything/DA3-LARGE-1.1",
    process_res: int = 672,
) -> dict[str, Any]:
    """Run locally cached DA3-LARGE-1.1 on a single drone photo and upscale depth/intrinsics to full size."""
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    _install_da3_optional_shims()
    import depth_anything_3.api as da3_api

    with Image.open(photo_path) as im:
        rgb_full = np.asarray(im.convert("RGB"), dtype=np.uint8)
    full_h, full_w = rgb_full.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = da3_api.DepthAnything3.from_pretrained(model_id).to(device)
    pred = model.inference([str(photo_path)], process_res=process_res, export_dir=None)

    d_proc = pred.depth[0].astype(np.float64)
    c_proc = pred.conf[0].astype(np.float64) if pred.conf is not None else np.ones_like(d_proc)
    k_proc = pred.intrinsics[0].astype(np.float64) if pred.intrinsics is not None else np.array(
        [[d_proc.shape[1] * 1.1, 0.0, d_proc.shape[1] * 0.5], [0.0, d_proc.shape[1] * 1.1, d_proc.shape[0] * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ph, pw = d_proc.shape

    depth_full = cv2.resize(d_proc, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    conf_full = cv2.resize(c_proc, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    sx, sy = full_w / float(pw), full_h / float(ph)
    k_full = np.array(
        [
            [k_proc[0, 0] * sx, 0.0, k_proc[0, 2] * sx],
            [0.0, k_proc[1, 1] * sy, k_proc[1, 2] * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "rgb_u8": rgb_full,
        "depth": depth_full,
        "conf": conf_full,
        "intrinsics": k_full,
        "proc_size": [int(pw), int(ph)],
        "image_size": [int(full_w), int(full_h)],
        "device": device,
        "model_id": model_id,
    }


def detect_semantics_gdino(
    rgb_u8: np.ndarray,
    *,
    model_id: str = "IDEA-Research/grounding-dino-tiny",
) -> dict[str, list[dict[str, Any]]]:
    """Run local zero-shot GroundingDINO detection if weights are cached; return empty lists otherwise."""
    empty: dict[str, list[dict[str, Any]]] = {"buildings": [], "roads": [], "cars": [], "trees": []}
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        proc = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    except Exception as exc:
        print(f"[INFO] GroundingDINO not cached locally ({exc.__class__.__name__}); using pure nDSM + photometric extraction.")
        return empty

    h, w = rgb_u8.shape[:2]
    im_area = float(w * h)
    im = Image.fromarray(rgb_u8)
    prompts = [
        ("buildings", "house . building . roof .", 0.18, 0.18, 0.25 * im_area),
        ("roads", "road . street .", 0.18, 0.18, 0.60 * im_area),
        ("cars", "car . van .", 0.22, 0.20, 0.04 * im_area),
        ("trees", "tree .", 0.20, 0.18, 0.18 * im_area),
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for cls_name, text_q, bt, tt, max_area in prompts:
        inputs = proc(images=im, text=text_q, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        res = proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=bt, text_threshold=tt, target_sizes=[(h, w)]
        )[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        kept: list[dict[str, Any]] = []
        if len(boxes):
            xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]).tolist()
            idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), bt, 0.50)
            flat_idx = idxs.flatten().tolist() if len(idxs) > 0 else []
            for ii in flat_idx:
                x0, y0, x1, y1 = [float(v) for v in boxes[ii]]
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                if area <= max_area and area >= 64.0:
                    kept.append({"box": [x0, y0, x1, y1], "score": float(scores[ii])})
        out[cls_name] = kept
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def compute_green_mask(rgb_u8: np.ndarray) -> np.ndarray:
    """Compute strict Excess-Green + HSV foliage/grass mask excluding cool-toned grey slate roofs."""
    rf = rgb_u8[:, :, 0].astype(np.float32)
    gf = rgb_u8[:, :, 1].astype(np.float32)
    bf = rgb_u8[:, :, 2].astype(np.float32)
    exg = 2.0 * gf - rf - bf
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    return (
        ((exg > 16.0) & (gf > rf + 4.0) & (gf > bf + 5.0))
        | ((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 88) & (hsv[:, :, 1] >= 44) & (hsv[:, :, 2] <= 190) & (exg > 9.0))
    )


def fit_ground_plane_and_camera(
    rgb_u8: np.ndarray,
    depth: np.ndarray,
    intrinsics_k: np.ndarray,
    conf: np.ndarray | None = None,
    detections: dict[str, list[dict[str, Any]]] | None = None,
    *,
    target_relief_m: float = 6.8,
    sample_step: int = 1,
) -> dict[str, Any]:
    """Fit dominant ground plane via lower-envelope RANSAC, calibrate metric scale, and solve CAM_ref pose."""
    h, w = depth.shape[:2]
    fx = float(intrinsics_k[0, 0])
    fy = float(intrinsics_k[1, 1])
    cx = float(intrinsics_k[0, 2])
    cy = float(intrinsics_k[1, 2])

    # Keep dense 1px sampling (downsample only for ultra-large >2400px inputs) so BEV has no scanline gaps
    step = max(1, int(sample_step) if max(w, h) <= 2400 else 2)
    ys_s = np.arange(0, h, step, dtype=np.float64)
    xs_s = np.arange(0, w, step, dtype=np.float64)
    us, vs = np.meshgrid(xs_s, ys_s)
    d_sub = depth[::step, ::step].astype(np.float64)
    rgb_sub = rgb_u8[::step, ::step]
    sh, sw = d_sub.shape

    # Conservative sky mask: top band with both low saturation & very low spatial texture + extreme depth tail
    hsv_sub = cv2.cvtColor(rgb_sub, cv2.COLOR_RGB2HSV)
    gray_sub = cv2.cvtColor(rgb_sub, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray_sub, cv2.CV_32F, 1, 0), cv2.Sobel(gray_sub, cv2.CV_32F, 0, 1))
    d_p96 = float(np.percentile(d_sub[np.isfinite(d_sub)], 96))
    sky_mask = (
        (vs < 0.42 * h)
        & (d_sub > d_p96 * 1.15)
        & (hsv_sub[:, :, 1] < 55)
        & (hsv_sub[:, :, 2] > 165)
        & (grad < 14.0)
    )
    valid_px = np.isfinite(d_sub) & (d_sub > 1e-3) & (~sky_mask)
    if int(valid_px.sum()) < 200:
        valid_px = np.isfinite(d_sub) & (d_sub > 1e-3)

    x_cam = (us - cx) / fx * d_sub
    y_cam = (vs - cy) / fy * d_sub
    pts_cam = np.stack([x_cam, y_cam, d_sub], axis=-1).reshape(-1, 3)

    # Candidate ground mask excludes optional building/tree detection boxes when available
    cand_mask = valid_px.copy()
    if detections:
        box_mask = np.zeros((sh, sw), dtype=bool)
        for cls_k in ("buildings", "trees"):
            for det in detections.get(cls_k, []):
                bx0, by0, bx1, by1 = det["box"]
                ix0, iy0 = max(0, int(bx0 // step)), max(0, int(by0 // step))
                ix1, iy1 = min(sw, int(math.ceil(bx1 / step))), min(sh, int(math.ceil(by1 / step)))
                box_mask[iy0:iy1, ix0:ix1] = True
        if int((cand_mask & (~box_mask)).sum()) >= 400:
            cand_mask &= ~box_mask

    cand_pts = pts_cam[cand_mask.reshape(-1)]
    rng = np.random.default_rng(42)
    if len(cand_pts) > 9000:
        cand_pts = cand_pts[rng.choice(len(cand_pts), 9000, replace=False)]

    d_med = float(np.median(d_sub[valid_px]))
    tol = max(1e-4, d_med * 0.015)

    best_score = -1e18
    best_n = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    best_d = 0.0
    n_pts = len(cand_pts)
    for _ in range(500):
        idx3 = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = cand_pts[idx3]
        n = np.cross(p1 - p0, p2 - p0)
        norm_n = float(np.linalg.norm(n))
        if norm_n < 1e-9:
            continue
        n /= norm_n
        if n[1] > 0.0:
            n = -n
        if -n[1] < 0.25:
            continue
        d_plane = -float(np.dot(n, p0))
        signed = cand_pts @ n + d_plane
        inliers = int(((signed >= -tol) & (signed <= tol)).sum())
        below = int((signed < -2.0 * tol).sum())
        score = float(inliers) - 3.0 * float(below)
        if score > best_score:
            best_score = score
            best_n = n
            best_d = d_plane

    # SVD refinement on RANSAC ground inliers
    signed_cand = cand_pts @ best_n + best_d
    inl_pts = cand_pts[(signed_cand >= -tol) & (signed_cand <= tol)]
    if len(inl_pts) >= 25:
        c_mean = inl_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(inl_pts - c_mean, full_matrices=False)
        n_ref = vh[-1]
        if n_ref[1] > 0.0:
            n_ref = -n_ref
        if -n_ref[1] >= 0.25:
            best_n = n_ref / max(float(np.linalg.norm(n_ref)), 1e-9)
            best_d = -float(np.dot(best_n, c_mean))

    # Align zero elevation with the lower peak (RANSAC ground band) so buildings sit at Y >= 0 without burying terrain
    all_signed = pts_cam @ best_n + best_d
    band_signed = all_signed[valid_px.reshape(-1) & (np.abs(all_signed) <= tol * 1.2)]
    if len(band_signed) >= 50:
        ground_off = float(np.percentile(band_signed, 35))
        best_d -= ground_off

    # Intersect optical center ray (or lower-mid ray if nearly horizontal) with ground plane for world origin
    ray0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    denom = float(np.dot(best_n, ray0))
    if abs(denom) < 0.15:
        ray0 = np.array([0.0, (0.62 * h - cy) / fy, 1.0], dtype=np.float64)
        denom = float(np.dot(best_n, ray0))
    t_orig = -best_d / (denom if abs(denom) > 1e-6 else -0.5)
    t_orig = max(0.05 * d_med, min(5.0 * d_med, t_orig))
    p_orig_raw = ray0 * t_orig

    # Build orthonormal right-handed world axes expressed in camera frame: R_cw = [wx | wy | wz]
    wy = best_n / max(float(np.linalg.norm(best_n)), 1e-9)
    fwd = ray0 - float(np.dot(ray0, wy)) * wy
    if float(np.linalg.norm(fwd)) < 1e-6:
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wz = -fwd / float(np.linalg.norm(fwd))
    wx = np.cross(wy, wz)
    wx /= max(float(np.linalg.norm(wx)), 1e-9)
    r_cw = np.column_stack([wx, wy, wz])
    if float(np.linalg.det(r_cw)) < 0.0:
        r_cw[:, 0] = -r_cw[:, 0]

    pts_world_raw = (pts_cam - p_orig_raw) @ r_cw
    eye_w_raw = -r_cw.T @ p_orig_raw

    # Self-calibrating uniform metric scale S (exact projection invariant)
    h_map_raw = pts_world_raw[:, 1].reshape(sh, sw)
    k_sz = max(9, int(round(sw * 0.06)) | 1)
    local_min = cv2.erode(h_map_raw.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sz, k_sz)))
    relief_raw = np.maximum(h_map_raw - local_min, 0.0)
    green_sub = compute_green_mask(rgb_sub)
    struct_mask = valid_px & (~green_sub) & (relief_raw > 1e-5)
    if int(struct_mask.sum()) < 200:
        struct_mask = valid_px & (relief_raw > 1e-5)
    relief_p90 = float(np.percentile(relief_raw[struct_mask], 90)) if int(struct_mask.sum()) >= 50 else 0.04

    if 3.5 <= relief_p90 <= 9.5 and float(eye_w_raw[1]) >= 12.0:
        scale_final = 1.0
        scale_car = None
        scale_relief = 1.0
    else:
        scale_relief = target_relief_m / max(relief_p90, 1e-4)
        scale_car = None
        car_dets = (detections or {}).get("cars", [])
        if len(car_dets) >= 3:
            car_lengths: list[float] = []
            for det in car_dets:
                bx0, by0, bx1, by1 = det["box"]
                c_pts = []
                for ux, vy in ((bx0, 0.5 * (by0 + by1)), (bx1, 0.5 * (by0 + by1)), (0.5 * (bx0 + bx1), by0), (0.5 * (bx0 + bx1), by1)):
                    ray = np.array([(ux - cx) / fx, (vy - cy) / fy, 1.0], dtype=np.float64)
                    dn = float(np.dot(best_n, ray))
                    if abs(dn) > 1e-4:
                        pc = ray * (-best_d / dn)
                        pw_c = (pc - p_orig_raw) @ r_cw
                        c_pts.append(pw_c[[0, 2]])
                if len(c_pts) == 4:
                    car_lengths.append(max(float(np.linalg.norm(c_pts[1] - c_pts[0])), float(np.linalg.norm(c_pts[3] - c_pts[2]))))
            if len(car_lengths) >= 3:
                scale_car = 4.4 / max(float(np.median(car_lengths)), 1e-4)
        if scale_car is not None and 0.5 * scale_relief <= scale_car <= 2.0 * scale_relief:
            scale_final = math.sqrt(scale_relief * scale_car)
        else:
            scale_final = scale_relief
        rel_cal = scale_final * relief_p90
        if rel_cal < 4.8:
            scale_final = 4.8 / max(relief_p90, 1e-4)
        elif rel_cal > 8.0:
            scale_final = 8.0 / max(relief_p90, 1e-4)

    pts_world = (pts_world_raw * scale_final).astype(np.float32)
    t_cw = p_orig_raw * scale_final
    eye_w = -r_cw.T @ t_cw
    pitch_deg = float(np.degrees(math.asin(max(-1.0, min(1.0, float(-np.dot(ray0, wy)))))))

    blender_eye, _, blender_mat4 = colmap_pose_to_blender_matrix(r_cw.tolist(), t_cw.tolist())
    bl_intr = colmap_intrinsics_to_blender_camera(fx, fy, cx, cy, w, h)
    fov_y_deg = math.degrees(2.0 * math.atan(h / (2.0 * fy)))

    conf_sub = conf[::step, ::step].reshape(-1) if conf is not None else np.ones(len(pts_world), dtype=np.float32)
    return {
        "r_cw": r_cw,
        "t_cw": t_cw,
        "eye_w": eye_w,
        "fov_y_deg": float(fov_y_deg),
        "pitch_deg": round(pitch_deg, 2),
        "blender_eye": list(blender_eye),
        "blender_mat4": blender_mat4,
        "blender_intrinsics": bl_intr,
        "scale_factor": float(scale_final),
        "scale_relief": float(scale_relief),
        "scale_car": float(scale_car) if scale_car is not None else None,
        "relief_p90_raw": float(relief_p90),
        "pts_world": pts_world,
        "pts_rgb": rgb_sub.reshape(-1, 3),
        "pts_conf": conf_sub.astype(np.float32),
        "valid_mask_flat": valid_px.reshape(-1),
    }


def project_world_to_photo(
    pts_world: np.ndarray,
    r_cw: np.ndarray,
    t_cw: np.ndarray,
    intrinsics_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) Three.js world points into photo pixel coordinates (uv, z_cam)."""
    xc = np.asarray(pts_world, dtype=np.float64) @ r_cw.T + t_cw
    z = xc[:, 2]
    z_safe = np.maximum(z, 1e-4)
    u = xc[:, 0] / z_safe * float(intrinsics_k[0, 0]) + float(intrinsics_k[0, 2])
    v = xc[:, 1] / z_safe * float(intrinsics_k[1, 1]) + float(intrinsics_k[1, 2])
    return np.stack([u, v], axis=-1), z


def rasterize_single_view_bev(
    pts_world: np.ndarray,
    pts_rgb: np.ndarray,
    pts_conf: np.ndarray,
    valid_flat: np.ndarray,
    *,
    max_span_m: float = 240.0,
    max_grid: int = 680,
) -> dict[str, Any]:
    """Rasterize ground-aligned metric points into BEV orthophoto, hole-filled DTM, nDSM, and road/veg priors."""
    v = valid_flat & np.isfinite(pts_world).all(axis=1) & (pts_world[:, 1] >= -3.0) & (pts_world[:, 1] <= 30.0)
    pw = pts_world[v]
    rgb = pts_rgb[v]
    conf = np.maximum(pts_conf[v], 0.2)

    x0, x1 = float(np.percentile(pw[:, 0], 1)), float(np.percentile(pw[:, 0], 99))
    z0, z1 = float(np.percentile(pw[:, 2], 1)), float(np.percentile(pw[:, 2], 99))
    cx_b, cz_b = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
    hx = min(max_span_m * 0.5, max(24.0, (x1 - x0) * 0.5 + 8.0))
    hz = min(max_span_m * 0.5, max(24.0, (z1 - z0) * 0.5 + 8.0))
    bx0, bx1 = round(cx_b - hx, 1), round(cx_b + hx, 1)
    bz0, bz1 = round(cz_b - hz, 1), round(cz_b + hz, 1)

    res_m = max(0.32, max(bx1 - bx0, bz1 - bz0) / float(max_grid))
    gw = max(64, int(round((bx1 - bx0) / res_m)))
    gh = max(64, int(round((bz1 - bz0) / res_m)))
    bx1 = bx0 + gw * res_m
    bz1 = bz0 + gh * res_m

    ix = np.floor((pw[:, 0] - bx0) / res_m).astype(np.int32)
    iz = np.floor((pw[:, 2] - bz0) / res_m).astype(np.int32)
    in_b = (ix >= 0) & (ix < gw) & (iz >= 0) & (iz < gh)
    ix, iz = ix[in_b], iz[in_b]
    y_v = pw[in_b, 1].astype(np.float32)
    rgb_v = rgb[in_b]
    w_v = conf[in_b]

    flat = iz.astype(np.int64) * gw + ix.astype(np.int64)
    order = np.lexsort((y_v, flat))
    flat_s = flat[order]
    y_s = y_v[order]
    rgb_s = rgb_v[order]
    w_s = w_v[order]

    uniq_cells, starts, counts = np.unique(flat_s, return_index=True, return_counts=True)
    dsm_flat = np.full(gh * gw, np.nan, dtype=np.float32)
    low_flat = np.full(gh * gw, np.nan, dtype=np.float32)
    cnt_flat = np.zeros(gh * gw, dtype=np.int32)
    ortho_flat = np.zeros((gh * gw, 3), dtype=np.uint8)

    idx_p82 = starts + np.minimum(counts - 1, np.floor(counts * 0.82).astype(np.int64))
    idx_p18 = starts + np.minimum(counts - 1, np.floor(counts * 0.18).astype(np.int64))
    dsm_flat[uniq_cells] = y_s[idx_p82]
    low_flat[uniq_cells] = y_s[idx_p18]
    cnt_flat[uniq_cells] = counts.astype(np.int32)

    top_ref = np.repeat(y_s[idx_p82], counts)
    near_top = np.abs(y_s - top_ref) <= 1.25
    pw_col = np.where(near_top, w_s, w_s * 0.08).astype(np.float32)
    w_sum = np.add.reduceat(pw_col, starts)
    inv_w = 1.0 / np.maximum(w_sum, 1e-4)
    for ch in range(3):
        c_sum = np.add.reduceat(rgb_s[:, ch].astype(np.float32) * pw_col, starts)
        ortho_flat[uniq_cells, ch] = np.clip(c_sum * inv_w, 0, 255).astype(np.uint8)

    dsm = dsm_flat.reshape(gh, gw)
    low_y = low_flat.reshape(gh, gw)
    obs_count = cnt_flat.reshape(gh, gw)
    ortho = ortho_flat.reshape(gh, gw, 3)

    # Fill 1-2px oblique perspective scanline gaps so buildings/roads remain solid connected surfaces
    valid_raw = obs_count > 0
    near_valid = cv2.dilate(valid_raw.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for _pass in range(2):
        missing = near_valid & (obs_count == 0)
        if not missing.any():
            break
        w_m = (obs_count > 0).astype(np.float32)
        w_sum = cv2.boxFilter(w_m, -1, (3, 3), normalize=False)
        ok_nb = missing & (w_sum > 0.5)
        if not ok_nb.any():
            break
        d_zero = np.where(obs_count > 0, dsm, 0.0).astype(np.float32)
        l_zero = np.where(np.isfinite(low_y), low_y, 0.0).astype(np.float32)
        d_avg = cv2.boxFilter(d_zero, -1, (3, 3), normalize=False) / np.maximum(w_sum, 1.0)
        l_avg = cv2.boxFilter(l_zero, -1, (3, 3), normalize=False) / np.maximum(w_sum, 1.0)
        dsm[ok_nb] = d_avg[ok_nb]
        low_y[ok_nb] = l_avg[ok_nb]
        for ch in range(3):
            o_ch = np.where(obs_count > 0, ortho[:, :, ch].astype(np.float32), 0.0)
            o_avg = cv2.boxFilter(o_ch, -1, (3, 3), normalize=False) / np.maximum(w_sum, 1.0)
            ortho[ok_nb, ch] = np.clip(o_avg[ok_nb], 0, 255).astype(np.uint8)
        obs_count[ok_nb] = 1

    # Nearest-neighbor hole-fill on low_y handles single-view building shadows cleanly before morphological DTM opening
    low_valid = np.isfinite(low_y)
    if low_valid.all():
        low_filled = low_y.astype(np.float32)
    else:
        dt_zero_y, dt_zero_x = np.where(low_valid)
        if len(dt_zero_y) > 0:
            _, lbl_vor = cv2.distanceTransformWithLabels(
                (~low_valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
            )
            lut = np.zeros(int(lbl_vor.max()) + 1, dtype=np.float32)
            lut_vals = low_y[dt_zero_y, dt_zero_x]
            if len(lut) - 1 == len(lut_vals):
                lut[1:] = lut_vals
                low_filled = lut[lbl_vor]
            else:
                med_g = float(np.nanmedian(low_y))
                low_filled = np.where(low_valid, low_y, med_g).astype(np.float32)
        else:
            low_filled = np.zeros((gh, gw), dtype=np.float32)

    k_dtm = max(21, int(round(18.0 / res_m)) | 1)
    dtm = cv2.morphologyEx(low_filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dtm, k_dtm)))
    k_blur = max(15, (k_dtm // 2) | 1)
    dtm = cv2.GaussianBlur(dtm, (k_blur, k_blur), k_blur * 0.30)

    valid_mask = obs_count > 0
    ndsm = np.where(valid_mask, np.clip(dsm - dtm, 0.0, 22.0), 0.0).astype(np.float32)

    # Photometric vegetation & foliage classification in BEV
    veg_mask_px = compute_green_mask(ortho) & valid_mask
    hsv_bev = cv2.cvtColor(ortho, cv2.COLOR_RGB2HSV)
    rf, gf, bf = ortho[:, :, 0].astype(np.float32), ortho[:, :, 1].astype(np.float32), ortho[:, :, 2].astype(np.float32)
    exg_bev = 2.0 * gf - rf - bf
    chroma = np.maximum(np.maximum(rf, gf), bf) - np.minimum(np.minimum(rf, gf), bf)

    mean_5 = cv2.blur(ndsm, (5, 5))
    sq_5 = cv2.blur(ndsm * ndsm, (5, 5))
    std_5 = np.sqrt(np.maximum(sq_5 - mean_5 * mean_5, 0.0))

    # Separate grass meadows (ndsm < 1.8m) from tall trees/hedgerows WITHOUT eroding nearby slate roofs
    tree_veg_mask = (veg_mask_px & (ndsm >= 2.2)) | (veg_mask_px & (std_5 > 0.50) & (ndsm >= 1.75))

    # Photometric paved road prior: low relief, low chroma, grey asphalt (bridge white lane paint via closing)
    road_prior = (
        valid_mask
        & (ndsm < 1.25)
        & (~veg_mask_px)
        & (exg_bev < 10.0)
        & (chroma < 28.0)
        & (hsv_bev[:, :, 1] < 68)
        & (hsv_bev[:, :, 2] >= 60)
        & (hsv_bev[:, :, 2] <= 205)
    )
    road_prior = cv2.morphologyEx(road_prior.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0

    gx = bx0 + (np.arange(gw, dtype=np.float64) + 0.5) * res_m
    gz = bz0 + (np.arange(gh, dtype=np.float64) + 0.5) * res_m
    xx, zz = np.meshgrid(gx, gz)

    return {
        "bounds_xz": (float(bx0), float(bx1), float(bz0), float(bz1)),
        "res_m": float(res_m),
        "gw": gw,
        "gh": gh,
        "dsm": dsm,
        "dtm": dtm,
        "ndsm": ndsm,
        "ortho": ortho,
        "obs_count": obs_count,
        "veg_mask": tree_veg_mask,
        "grass_mask": veg_mask_px & (~tree_veg_mask),
        "road_prior": road_prior,
        "xx": xx,
        "zz": zz,
    }


def extract_road_centerline(
    bev: dict[str, Any],
    detections: dict[str, list[dict[str, Any]]] | None,
    cam_pose: dict[str, Any],
    intrinsics_k: np.ndarray,
) -> dict[str, Any]:
    """Fit a smooth multi-station carriageway centerline through the dominant paved street corridor."""
    res_m = float(bev["res_m"])
    xx, zz = bev["xx"], bev["zz"]
    road_mask = bev["road_prior"].copy()

    # Reinforce with GDINO road boxes if present
    road_dets = (detections or {}).get("roads", [])
    if road_dets:
        world_grid = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=-1)
        uv, zc = project_world_to_photo(world_grid, cam_pose["r_cw"], cam_pose["t_cw"], intrinsics_k)
        ux = uv[:, 0].reshape(xx.shape)
        vy = uv[:, 1].reshape(xx.shape)
        zc_m = zc.reshape(xx.shape)
        box_hit = np.zeros_like(road_mask)
        for det in road_dets:
            bx0, by0, bx1, by1 = det["box"]
            box_hit |= (zc_m > 2.0) & (ux >= bx0) & (ux <= bx1) & (vy >= by0) & (vy <= by1)
        if int((road_mask & box_hit).sum()) >= int(80.0 / (res_m * res_m)):
            road_mask &= box_hit

    rm_u8 = (road_mask.astype(np.uint8) * 255)
    rm_u8 = cv2.morphologyEx(rm_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rm_u8 = cv2.morphologyEx(rm_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_cc, cc_lbl, stats, _ = cv2.connectedComponentsWithStats(rm_u8)

    bx0, bx1, bz0, bz1 = bev["bounds_xz"]
    best_id = -1
    best_score = 0.0
    for cid in range(1, n_cc):
        area_px = int(stats[cid, cv2.CC_STAT_AREA])
        area_m2 = area_px * (res_m * res_m)
        if area_m2 < 110.0:
            continue
        bw_m = float(stats[cid, cv2.CC_STAT_WIDTH]) * res_m
        bh_m = float(stats[cid, cv2.CC_STAT_HEIGHT]) * res_m
        span_m = math.hypot(bw_m, bh_m)
        if span_m < 24.0:
            continue
        # Favor elongated long-spanning ribbons over blocky car parks
        score = span_m * math.sqrt(area_m2)
        if score > best_score:
            best_score = score
            best_id = cid

    if best_id < 0:
        centerline_xz = np.array([[bx0 + 10.0, bz1 + 9.0], [bx1 - 10.0, bz1 + 9.0]], dtype=np.float64)
        road_info = build_regularized_road(centerline_xz, carriageway_half_width=3.2, apron_width=1.4, ext_m=140.0)
        return {"centerline_xz": centerline_xz.tolist(), "detected": False, **road_info}

    pts_r = np.column_stack([xx[cc_lbl == best_id], zz[cc_lbl == best_id]])
    c_r = pts_r.mean(axis=0)
    _, _, vh = np.linalg.svd(pts_r - c_r, full_matrices=False)
    u_dir = vh[0]
    if u_dir[0] < 0.0 or (abs(u_dir[0]) < 0.1 and u_dir[1] > 0.0):
        u_dir = -u_dir
    n_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)

    s_proj = (pts_r - c_r) @ u_dir
    t_proj = (pts_r - c_r) @ n_dir
    s_lo, s_hi = float(np.percentile(s_proj, 4)), float(np.percentile(s_proj, 96))
    edges = np.linspace(s_lo, s_hi, 8)

    stations: list[np.ndarray] = []
    half_widths: list[float] = []
    prev_t: float | None = None
    for k in range(len(edges) - 1):
        in_bin = (s_proj >= edges[k]) & (s_proj <= edges[k + 1])
        if int(in_bin.sum()) < 15:
            continue
        tb = t_proj[in_bin]
        # Track the continuous primary strand near prev_t (or histogram peak on first station) so junctions don't shortcut across corners
        if prev_t is not None:
            near = tb[np.abs(tb - prev_t) <= 8.5]
            core_t = near if len(near) >= 10 else tb
        else:
            hist, bin_e = np.histogram(tb, bins=max(4, min(20, int(np.ptp(tb) / 2.5) + 1)))
            mode_t = 0.5 * (bin_e[int(np.argmax(hist))] + bin_e[int(np.argmax(hist)) + 1])
            near = tb[np.abs(tb - mode_t) <= 5.5]
            core_t = near if len(near) >= 10 else tb
        med_t = float(np.median(core_t))
        prev_t = med_t
        half_widths.append(float(np.percentile(np.abs(core_t - med_t), 75)))
        s_mid = 0.5 * (edges[k] + edges[k + 1])
        stations.append(c_r + u_dir * s_mid + n_dir * med_t)

    if len(stations) < 2:
        stations = [c_r + u_dir * s_lo, c_r + u_dir * s_hi]

    # Smooth intermediate transverse jitter while preserving true street curvature
    cl = np.asarray(stations, dtype=np.float64)
    if len(cl) >= 4:
        cl_sm = cl.copy()
        cl_sm[1:-1] = 0.25 * cl[:-2] + 0.50 * cl[1:-1] + 0.25 * cl[2:]
        cl = cl_sm

    hw = float(np.clip(np.median(half_widths) if half_widths else 3.4, 2.8, 3.9))
    apron_w = 1.4
    road_info = build_regularized_road(cl, carriageway_half_width=hw, apron_width=apron_w, ext_m=6.0)
    return {"centerline_xz": [[round(float(x), 3), round(float(z), 3)] for x, z in cl], "detected": True, **road_info}


def classify_roof_from_rgb(rgb: np.ndarray, width_m: float, depth_m: float) -> tuple[str, str, float, bool, bool]:
    """Classify roof swatch, archetype, gable ridge rise, and architectural flags from sampled RGB & plan dims."""
    r, g, b = [float(x) for x in rgb]
    u8 = np.uint8([[[np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]]])
    h, s, v = [int(x) for x in cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)[0, 0]]
    short_span = min(width_m, depth_m)

    if (h <= 22 or h >= 165) and s >= 34 and r > b + 12.0:
        rise = float(np.clip(short_span * 0.17, 1.05, 1.95))
        return "terracotta_tile", "gable_red_tile", rise, False, True
    if 95 <= h <= 132 and s >= 52 and b > r + 18.0:
        rise = float(np.clip(short_span * 0.13, 0.85, 1.45))
        return "corrugated_blue", "gable_blue_metal", rise, False, True
    if v < 138 and s < 50:
        rise = float(np.clip(short_span * 0.18, 1.15, 2.20))
        return "slate_roof", "gable_slate_tile", rise, False, True
    if s < 32 and v >= 168:
        return "concrete_roof", "flat_parapet", 0.0, True, False
    rise = float(np.clip(short_span * 0.17, 1.05, 2.00))
    return "slate_roof", "gable_slate_tile", rise, False, True


def classify_wall_from_rgb(rgb: np.ndarray) -> str:
    """Classify exterior wall swatch (whitewashed, warm stone/stucco, grey render, red brick) from sampled RGB."""
    r, g, b = [float(x) for x in rgb]
    u8 = np.uint8([[[np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)]]])
    _, s, v = [int(x) for x in cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)[0, 0]]
    if r > g + 16.0 and r > b + 26.0 and v < 180:
        return "red_brick"
    if v >= 178 and s < 36:
        return "plaster_white"
    if r > b + 12.0 and v >= 112:
        return "plaster_warm"
    return "plaster_grey"


def _sample_quad_median_rgb(
    ref_rgb: np.ndarray,
    pts3d: np.ndarray,
    r_cw: np.ndarray,
    t_cw: np.ndarray,
    intrinsics_k: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Project a 3D quad's interior grid into CAM_ref and return robust median RGB."""
    if len(pts3d) < 3:
        return fallback
    p0, p1, p2, p3 = pts3d[0], pts3d[1], pts3d[2], pts3d[-1]
    grid_samples = []
    for su in (0.25, 0.5, 0.75):
        for sv in (0.25, 0.5, 0.75):
            pt = (1 - su) * (1 - sv) * p0 + su * (1 - sv) * p1 + su * sv * p2 + (1 - su) * sv * p3
            grid_samples.append(pt)
    uv, zc = project_world_to_photo(np.asarray(grid_samples), r_cw, t_cw, intrinsics_k)
    h, w = ref_rgb.shape[:2]
    ok = (zc > 2.0) & (uv[:, 0] >= 4) & (uv[:, 0] < w - 4) & (uv[:, 1] >= 4) & (uv[:, 1] < h - 4)
    if not ok.any():
        return fallback
    ui = np.round(uv[ok, 0]).astype(int)
    vi = np.round(uv[ok, 1]).astype(int)
    return np.median(ref_rgb[vi, ui].astype(np.float32), axis=0)


def extract_buildings_from_bev(
    bev: dict[str, Any],
    road_info: dict[str, Any],
    ref_rgb: np.ndarray,
    cam_pose: dict[str, Any],
    intrinsics_k: np.ndarray,
    *,
    max_buildings: int = 120,
) -> dict[str, Any]:
    """Segment elevated non-vegetation roof regions via BEV nDSM + local PCA ridge tiling, solve layout, and classify materials."""
    ndsm = bev["ndsm"]
    ortho = bev["ortho"]
    veg_mask = bev["veg_mask"]
    obs_count = bev["obs_count"]
    xx, zz = bev["xx"], bev["zz"]
    res_m = float(bev["res_m"])
    gh, gw = ndsm.shape

    cl = np.asarray(road_info["centerline_xz"], dtype=np.float64)
    road_poly_xz = road_info["road_polygon_xz"]
    apron_poly_xz = road_info["road_apron_polygon_xz"]
    corridor_hw = float(road_info["corridor_half_width"])

    apron_mask = np.zeros((gh, gw), dtype=np.uint8)
    bx0, bx1, bz0, bz1 = bev["bounds_xz"]
    ap_px = np.stack(
        [
            np.clip((np.asarray(apron_poly_xz)[:, 0] - bx0) / (bx1 - bx0) * (gw - 1), 0, gw - 1),
            np.clip((np.asarray(apron_poly_xz)[:, 1] - bz0) / (bz1 - bz0) * (gh - 1), 0, gh - 1),
        ],
        axis=-1,
    ).astype(np.int32)
    cv2.fillPoly(apron_mask, [ap_px], 255)
    apron_mask = cv2.dilate(apron_mask, np.ones((3, 3), np.uint8))

    # Avoid dark long-cast shadows on flat lawns/walls pretending to be buildings
    luma_bev = ortho.mean(axis=2)
    cand = (
        (ndsm >= 1.65)
        & (ndsm <= 15.0)
        & (~veg_mask)
        & (obs_count >= 1)
        & (apron_mask == 0)
        & (luma_bev >= 24.0)
    ).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    dist = cv2.distanceTransform(cand, cv2.DIST_L2, 5)
    k_peak = max(7, int(round(4.6 / res_m)) | 1)
    peak_dil = cv2.dilate(dist, np.ones((k_peak, k_peak), np.uint8))
    min_r_px = max(2.2, 1.5 / res_m)
    peaks = ((dist == peak_dil) & (dist >= min_r_px)).astype(np.uint8)
    n_cc, cc_lbl = cv2.connectedComponents(cand)
    min_comp_px = max(30, int(round(14.0 / (res_m * res_m))))
    for cc_id in range(1, n_cc):
        comp_m = cc_lbl == cc_id
        if int(comp_m.sum()) >= min_comp_px and int(peaks[comp_m].sum()) == 0:
            ys_c, xs_c = np.where(comp_m)
            best_i = int(np.argmax(dist[ys_c, xs_c]))
            peaks[ys_c[best_i], xs_c[best_i]] = 1

    _, peak_lbl = cv2.connectedComponents(peaks)
    markers = np.zeros(cand.shape, dtype=np.int32)
    markers[cand == 0] = 1
    markers[peaks > 0] = peak_lbl[peaks > 0] + 1
    gx = cv2.Sobel(ndsm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ndsm, cv2.CV_32F, 0, 1, ksize=3)
    grad_u8 = np.clip(np.hypot(gx, gy) * 35.0, 0, 255).astype(np.uint8)
    ws = cv2.watershed(np.stack([grad_u8, grad_u8, grad_u8], axis=-1), markers)

    def _pca_axis(pts_2d: np.ndarray, fallback_u: np.ndarray) -> np.ndarray:
        if len(pts_2d) < 6:
            return fallback_u
        c_p = pts_2d.mean(axis=0)
        _, _, vh_p = np.linalg.svd(pts_2d - c_p, full_matrices=False)
        u_p = vh_p[0]
        if u_p[0] < 0.0 or (abs(u_p[0]) < 0.15 and u_p[1] > 0.0):
            u_p = -u_p
        return u_p / max(float(np.linalg.norm(u_p)), 1e-8)

    candidates: list[dict[str, Any]] = []
    min_reg_px = max(28, int(round(15.0 / (res_m * res_m))))
    for lbl in np.unique(ws):
        if int(lbl) < 2:
            continue
        rmask = (ws == int(lbl)) & (cand > 0)
        if int(rmask.sum()) < min_reg_px:
            continue
        pts_xz = np.column_stack([xx[rmask], zz[rmask]])
        hs = ndsm[rmask]
        rgb_pts = ortho[rmask].astype(np.float32)
        obs_pts = obs_count[rmask]
        area_m2 = float(len(hs)) * (res_m * res_m)
        if area_m2 < 16.0:
            continue
        c = pts_xz.mean(axis=0)
        _, _, u_road, _, _ = local_road_frame(c, cl)
        u_main = _pca_axis(pts_xz, u_road)
        v_main = np.array([-u_main[1], u_main[0]], dtype=np.float64)
        pu = (pts_xz - c) @ u_main
        pv = (pts_xz - c) @ v_main
        u_lo, u_hi = float(np.percentile(pu, 3)), float(np.percentile(pu, 97))
        v_lo, v_hi = float(np.percentile(pv, 3)), float(np.percentile(pv, 97))
        span_u = u_hi - u_lo
        span_v = v_hi - v_lo
        if min(span_u, span_v) < 3.2:
            continue

        # Tile large or L-shaped terraced blocks into occupied sub-bays using each bay's OWN local orientation
        n_u = max(1, int(round(span_u / 9.8))) if span_u > 14.5 else 1
        n_v = max(1, int(round(span_v / 9.2))) if span_v > 12.5 else 1
        du = span_u / n_u
        dv = span_v / n_v
        cell_area = du * dv
        for iu in range(n_u):
            ua, ub = u_lo + iu * du, u_lo + (iu + 1) * du
            for iv in range(n_v):
                va, vb = v_lo + iv * dv, v_lo + (iv + 1) * dv
                sub_m = (pu >= ua) & (pu <= ub) & (pv >= va) & (pv <= vb)
                sub_area = float(int(sub_m.sum())) * (res_m * res_m)
                if sub_area < max(16.0, 0.34 * cell_area):
                    continue
                s_pts = pts_xz[sub_m]
                sc = s_pts.mean(axis=0)
                su = _pca_axis(s_pts, u_main)
                sv = np.array([-su[1], su[0]], dtype=np.float64)
                spu = (s_pts - sc) @ su
                spv = (s_pts - sc) @ sv
                w_m = float(np.clip(np.percentile(spu, 96) - np.percentile(spu, 4), 4.0, 15.0))
                d_m = float(np.clip(np.percentile(spv, 96) - np.percentile(spv, 4), 3.6, 11.5))
                h_m = float(np.clip(np.percentile(hs[sub_m], 76), 3.6, 9.6))
                rgb_med = np.median(rgb_pts[sub_m], axis=0)
                quad = make_oriented_quad_xz(float(sc[0]), float(sc[1]), w_m, d_m, su)
                candidates.append(
                    {
                        "score": float(obs_pts[sub_m].sum()),
                        "footprint_xz": quad,
                        "height_m": h_m,
                        "w_m": w_m,
                        "d_m": d_m,
                        "bev_roof_rgb": rgb_med,
                    }
                )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    candidates = candidates[:max_buildings]

    r_cw, t_cw = cam_pose["r_cw"], cam_pose["t_cw"]
    raw_buildings: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        quad = ensure_ccw_blender_xz(item["footprint_xz"])
        h_m = float(item["height_m"])
        pts3d = np.column_stack([quad[:, 0], np.full(len(quad), h_m), quad[:, 1]])
        obs_roof_rgb = _sample_quad_median_rgb(ref_rgb, pts3d, r_cw, t_cw, intrinsics_k, item["bev_roof_rgb"])

        # Sample camera-facing visible facade wall quad for wall swatch classification
        bot = pts3d.copy()
        bot[:, 1] = 0.0
        cam_eye = cam_pose["eye_w"]
        best_wall_dot = -1.0
        best_wall_quad = np.array([bot[0], bot[1], pts3d[1], pts3d[0]], dtype=np.float64)
        for ei in range(4):
            ej = (ei + 1) % 4
            edge = quad[ej] - quad[ei]
            elen = float(np.linalg.norm(edge))
            n_out = np.array([-edge[1], edge[0]], dtype=np.float64) / max(elen, 1e-6)
            mid_xz = 0.5 * (quad[ei] + quad[ej])
            if float(np.dot(n_out, mid_xz - quad.mean(0))) < 0.0:
                n_out = -n_out
            to_cam = np.array([cam_eye[0] - mid_xz[0], cam_eye[2] - mid_xz[1]], dtype=np.float64)
            to_cam /= max(float(np.linalg.norm(to_cam)), 1e-6)
            dot_v = float(np.dot(n_out, to_cam))
            if dot_v > best_wall_dot:
                best_wall_dot = dot_v
                best_wall_quad = np.array([bot[ei], bot[ej], pts3d[ej], pts3d[ei]], dtype=np.float64)

        obs_wall_rgb = _sample_quad_median_rgb(
            ref_rgb, best_wall_quad, r_cw, t_cw, intrinsics_k, np.array([210.0, 200.0, 184.0], dtype=np.float32)
        )
        roof_swatch, archetype, ridge_rise, has_parapet, has_eave = classify_roof_from_rgb(
            obs_roof_rgb, item["w_m"], item["d_m"]
        )
        wall_swatch = classify_wall_from_rgb(obs_wall_rgb)
        raw_buildings.append(
            {
                "id": f"house_{idx + 1:02d}",
                "index": idx,
                "is_core": True,
                "source": "single_photo_bev",
                "footprint_xz": [[round(float(x), 4), round(float(z), 4)] for x, z in quad],
                "points_3d": [[round(float(x), 4), round(float(y), 4), round(float(z), 4)] for x, y, z in pts3d],
                "height_m": round(h_m, 3),
                "height_prior_m": round(h_m, 3),
                "stories": 2 if h_m >= 5.1 else 1,
                "archetype": archetype,
                "roof_swatch": roof_swatch,
                "wall_swatch": wall_swatch,
                "side_wall_swatch": wall_swatch,
                "front_cornice_brick": False,
                "ridge_rise": round(ridge_rise, 3),
                "observed_roof_rgb": [round(float(x), 1) for x in obs_roof_rgb],
                "observed_wall_rgb": [round(float(x), 1) for x in obs_wall_rgb],
                "has_parapet": bool(has_parapet),
                "has_eave_overhang": bool(has_eave),
                "solar_heater": False,
            }
        )

    solved = solve_building_layout(
        raw_buildings,
        cl,
        road_poly_xz,
        apron_poly_xz,
        apron_half_width=corridor_hw,
        road_setback=0.35,
        min_building_gap=0.55,
        drop_unresolvable_synthetic=True,
        preserve_orientation=True,
    )
    layout_audit = audit_layout_geometry(solved, cl, road_poly_xz, apron_poly_xz)
    if layout_audit["verdict"] != "pass":
        raise RuntimeError(f"Layout audit gate failed: {layout_audit}")

    # Populate per-wall outward normals, neighbor gaps, and residential facade roles
    all_polys = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in solved]
    for idx, b in enumerate(solved):
        poly = all_polys[idx]
        center_xz = poly.mean(0)
        _, sd_road, _, _, near_road_pt = local_road_frame(center_xz, cl)
        road_d = abs(sd_road)
        to_road = near_road_pt - center_xz
        to_road /= max(float(np.linalg.norm(to_road)), 1e-6)
        b["road_distance_m"] = round(road_d, 2)

        walls = []
        n_e = len(poly)
        for ei in range(n_e):
            p0, p1 = poly[ei], poly[(ei + 1) % n_e]
            edge = p1 - p0
            elen = float(np.linalg.norm(edge))
            cand_n = np.array([-edge[1], edge[0]], dtype=np.float64) / max(elen, 1e-6)
            mid = 0.5 * (p0 + p1)
            if float(np.dot(cand_n, mid - center_xz)) < 0.0:
                cand_n = -cand_n
            gap = _edge_neighbor_gap(p0, p1, idx, all_polys)
            faces_street = float(np.dot(cand_n, to_road)) > 0.42 and road_d < 30.0
            role = "party_wall" if gap < 1.15 else "facade"
            walls.append(
                {
                    "edge_index": ei,
                    "length_m": round(elen, 2),
                    "outward_xz": [round(float(cand_n[0]), 4), round(float(cand_n[1]), 4)],
                    "neighbor_gap_m": round(gap, 2),
                    "role": role,
                    "swatch": b["wall_swatch"] if faces_street else b["side_wall_swatch"],
                }
            )
        b["walls"] = walls
        b["courtyard_segments_xz"] = []

    return {"buildings": solved, "layout_audit": layout_audit, "apron_mask": apron_mask}


def extract_trees_from_bev(
    bev: dict[str, Any],
    buildings: list[dict[str, Any]],
    road_info: dict[str, Any],
    *,
    max_trees: int = 90,
) -> list[dict[str, Any]]:
    """Extract tall vegetation canopy peaks strictly clear of roads, aprons, and buildings."""
    ndsm = bev["ndsm"]
    veg = bev["veg_mask"]
    xx, zz = bev["xx"], bev["zz"]
    res_m = float(bev["res_m"])
    corridor_hw = float(road_info["corridor_half_width"])
    cl = road_info["centerline_xz"]
    road_poly_xz = road_info["road_polygon_xz"]
    apron_poly_xz = road_info["road_apron_polygon_xz"]
    b_shps = [Polygon(b["footprint_xz"]).buffer(1.4) for b in buildings]

    canopy = np.where(veg & (ndsm >= 2.1), ndsm, 0.0).astype(np.float32)
    k_dil = max(7, int(round(4.2 / res_m)) | 1)
    peak_m = (canopy >= 2.1) & (canopy == cv2.dilate(canopy, np.ones((k_dil, k_dil), np.uint8)))
    ys_p, xs_p = np.where(peak_m)
    order = np.argsort(canopy[ys_p, xs_p])[::-1]

    trees: list[dict[str, Any]] = []
    placed_xz: list[np.ndarray] = []
    for k in order:
        if len(trees) >= max_trees:
            break
        py, px = int(ys_p[k]), int(xs_p[k])
        tx, tz = float(xx[py, px]), float(zz[py, px])
        th = float(np.clip(canopy[py, px] * 1.05, 3.6, 13.0))
        trad = float(np.clip(th * 0.34, 1.8, 3.8))
        if point_in_polygon_2d(tx, tz, road_poly_xz) or point_in_polygon_2d(tx, tz, apron_poly_xz):
            continue
        _, sd_r, _, _, _ = local_road_frame((tx, tz), cl)
        # Keep merged leaf-card extents safely outside road aprons (audit checks lowest leaf card vertices!)
        if abs(sd_r) < corridor_hw + trad * 1.85 + 1.2:
            continue
        pt_s = Point(tx, tz)
        if any(bs.contains(pt_s) for bs in b_shps):
            continue
        p_arr = np.array([tx, tz], dtype=np.float64)
        if any(float(np.linalg.norm(p_arr - prev)) < 4.2 for prev in placed_xz):
            continue
        placed_xz.append(p_arr)
        trees.append(
            {
                "id": f"tree_{len(trees):03d}",
                "species": "broadleaf" if len(trees) % 4 != 0 else "poplar",
                "xz": [round(tx, 3), round(tz, 3)],
                "height_m": round(th, 2),
                "radius_m": round(trad, 2),
            }
        )
    return trees


def extract_parked_cars(
    bev: dict[str, Any],
    buildings: list[dict[str, Any]],
    road_info: dict[str, Any],
    detections: dict[str, list[dict[str, Any]]] | None,
    cam_pose: dict[str, Any],
    intrinsics_k: np.ndarray,
    *,
    max_cars: int = 8,
) -> list[dict[str, Any]]:
    """Place parked vehicles from GDINO detections or strict roadside compactness candidates."""
    cl = road_info["centerline_xz"]
    apron_hw = float(road_info["corridor_half_width"])
    b_shps = [Polygon(b["footprint_xz"]).buffer(1.6) for b in buildings]
    cars: list[dict[str, Any]] = []
    placed: list[np.ndarray] = []

    paints = ("white", "silver", "charcoal", "blue")
    car_dets = (detections or {}).get("cars", [])
    if car_dets:
        # Intersect car base centers with ground via CAM_ref ray projection
        r_cw = cam_pose["r_cw"]
        t_cw = cam_pose["t_cw"]
        fx, fy, cx, cy = intrinsics_k[0, 0], intrinsics_k[1, 1], intrinsics_k[0, 2], intrinsics_k[1, 2]
        eye_w = cam_pose["eye_w"]
        for det in car_dets:
            if len(cars) >= max_cars:
                break
            bx0, by0, bx1, by1 = det["box"]
            uc, vc = 0.5 * (bx0 + bx1), 0.78 * by1 + 0.22 * by0
            ray_w = r_cw.T @ np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0], dtype=np.float64)
            if ray_w[1] >= -0.02:
                continue
            pt_w = eye_w + ray_w * (-eye_w[1] / ray_w[1])
            px, pz = float(pt_w[0]), float(pt_w[2])
            _, sd_r, u_dir, _, _ = local_road_frame((px, pz), cl)
            if abs(sd_r) > apron_hw + 8.0:
                continue
            if any(bs.contains(Point(px, pz)) for bs in b_shps):
                continue
            p_arr = np.array([px, pz])
            if any(float(np.linalg.norm(p_arr - prev)) < 5.2 for prev in placed):
                continue
            placed.append(p_arr)
            cars.append(
                {
                    "id": f"car_{len(cars):02d}",
                    "xz": [round(px, 3), round(pz, 3)],
                    "yaw_rad": round(math.atan2(-u_dir[1], u_dir[0]), 4),
                    "paint": paints[len(cars) % len(paints)],
                }
            )
    else:
        # Strict roadside shoulder blob heuristic when GDINO is absent
        ndsm, xx, zz = bev["ndsm"], bev["xx"], bev["zz"]
        res_m = float(bev["res_m"])
        blob = ((ndsm >= 0.45) & (ndsm <= 1.85) & (~bev["veg_mask"])).astype(np.uint8) * 255
        n_cc, _, st, cent = cv2.connectedComponentsWithStats(blob)
        for cid in range(1, n_cc):
            if len(cars) >= max_cars:
                break
            area_m2 = float(st[cid, cv2.CC_STAT_AREA]) * (res_m * res_m)
            if not (4.0 <= area_m2 <= 18.0):
                continue
            gx_i = min(bev["gw"] - 1, max(0, int(round(cent[cid, 0]))))
            gz_i = min(bev["gh"] - 1, max(0, int(round(cent[cid, 1]))))
            px, pz = float(xx[gz_i, gx_i]), float(zz[gz_i, gx_i])
            _, sd_r, u_dir, _, _ = local_road_frame((px, pz), cl)
            if not (2.0 <= abs(sd_r) <= apron_hw + 0.8):
                continue
            if any(bs.contains(Point(px, pz)) for bs in b_shps):
                continue
            p_arr = np.array([px, pz])
            if any(float(np.linalg.norm(p_arr - prev)) < 5.5 for prev in placed):
                continue
            placed.append(p_arr)
            cars.append(
                {
                    "id": f"car_{len(cars):02d}",
                    "xz": [round(px, 3), round(pz, 3)],
                    "yaw_rad": round(math.atan2(-u_dir[1], u_dir[0]), 4),
                    "paint": paints[len(cars) % len(paints)],
                }
            )
    return cars


def build_adaptive_ground_texture(
    out_dir: Path,
    bev: dict[str, Any],
    ground_bounds: list[float],
    buildings: list[dict[str, Any]],
    courtyard_walls: list[dict[str, Any]],
    parked_cars: list[dict[str, Any]],
    road_info: dict[str, Any],
    swatches: dict[str, dict[str, str]],
) -> tuple[dict[str, str], list[float]]:
    """Build 2048x2048 scene-adaptive ground atlas by harmonizing meadow/earth base with BEV orthophoto & contact AO."""
    size = 2048
    bx0, bx1, bz0, bz1 = [float(v) for v in ground_bounds]
    gx = np.linspace(bx0, bx1, size, dtype=np.float64)
    gz = np.linspace(bz0, bz1, size, dtype=np.float64)
    xx, zz = np.meshgrid(gx, gz)

    # Sample scene's dominant open-ground / grass base color from observed sunlit low-nDSM BEV pixels
    low_obs = (bev["obs_count"] >= 1) & (bev["ndsm"] < 0.85) & (~bev["road_prior"])
    grass_obs = low_obs & bev["grass_mask"]
    if int(grass_obs.sum()) >= 50:
        base_rgb = np.percentile(bev["ortho"][grass_obs].astype(np.float32), 62, axis=0)
    elif int(low_obs.sum()) >= 50:
        base_rgb = np.percentile(bev["ortho"][low_obs].astype(np.float32), 58, axis=0)
    else:
        base_rgb = np.array([96.0, 118.0, 68.0], dtype=np.float32)

    rng = np.random.default_rng(109)
    low_n = cv2.resize(rng.normal(0.0, 1.0, (24, 24)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    med_n = cv2.resize(rng.normal(0.0, 1.0, (96, 96)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    fine_n = cv2.resize(rng.normal(0.0, 6.0, (384, 384)).astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    ground = np.clip(
        base_rgb[None, None, :] + low_n[:, :, None] * 10.0 + med_n[:, :, None] * 5.0 + fine_n[:, :, None],
        18.0,
        242.0,
    ).astype(np.float32)

    def world_to_grid(pts_xz: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_xz, dtype=np.float64)
        return np.stack(
            [(p[:, 0] - bx0) / max(bx1 - bx0, 1e-4) * (size - 1), (p[:, 1] - bz0) / max(bz1 - bz0, 1e-4) * (size - 1)],
            axis=-1,
        ).astype(np.int32)

    # Tiled concrete aprons directly around building pads
    apron_src = np.asarray(Image.open(swatches["concrete_apron"]["albedo"]).convert("RGB"), dtype=np.float32)
    ah, aw = apron_src.shape[:2]
    tiled_apron = cv2.remap(
        apron_src,
        np.mod((xx - bx0) / 10.0 * aw, aw).astype(np.float32),
        np.mod((zz - bz0) / 10.0 * ah, ah).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    yard_mask = np.zeros((size, size), dtype=np.uint8)
    contact_occ = np.zeros((size, size), dtype=np.uint8)
    for b in buildings:
        pts_xz = np.asarray(b["footprint_xz"], dtype=np.float64)
        c = pts_xz.mean(0)
        cv2.fillConvexPoly(yard_mask, world_to_grid(c + (pts_xz - c) * 1.18), 255)
        cv2.fillConvexPoly(contact_occ, world_to_grid(pts_xz), 255)
    yard_alpha = cv2.GaussianBlur((yard_mask > 0).astype(np.float32), (11, 11), 3.0) * 0.72
    ground = ground * (1.0 - yard_alpha[:, :, None]) + tiled_apron * yard_alpha[:, :, None]

    # Blend real DA3 BEV orthophoto across ALL observed ground areas (outside buildings & primary road)
    road_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(road_mask, [world_to_grid(np.asarray(road_info["road_polygon_xz"], dtype=np.float64))], 255)

    bx_m0, bx_m1, bz_m0, bz_m1 = bev["bounds_xz"]
    bh, bw = bev["ortho"].shape[:2]
    bu = ((xx - bx_m0) / max(bx_m1 - bx_m0, 1e-4) * bw - 0.5).astype(np.float32)
    bv = ((zz - bz_m0) / max(bz_m1 - bz_m0, 1e-4) * bh - 0.5).astype(np.float32)
    in_bev = (bu >= 1.0) & (bu <= bw - 2.0) & (bv >= 1.0) & (bv <= bh - 2.0)
    ortho_s = cv2.remap(bev["ortho"].astype(np.float32), bu, bv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ndsm_s = cv2.remap(bev["ndsm"].astype(np.float32), bu, bv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=99.0)
    obs_s = cv2.remap(bev["obs_count"].astype(np.float32), bu, bv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    valid_g = in_bev & (obs_s >= 1.0) & (ndsm_s < 1.35) & (contact_occ == 0) & (road_mask == 0)
    if valid_g.any():
        alpha_o = cv2.GaussianBlur(valid_g.astype(np.float32), (17, 17), 4.5) * 0.78
        ground = ground * (1.0 - alpha_o[:, :, None]) + ortho_s * alpha_o[:, :, None]

    for w in courtyard_walls:
        seg = world_to_grid(np.array([w["a_xz"], w["b_xz"]]))
        cv2.line(contact_occ, tuple(seg[0]), tuple(seg[1]), 220, 4)
    for car in parked_cars:
        cg = world_to_grid(np.array([car["xz"]]))[0]
        cv2.circle(contact_occ, (int(cg[0]), int(cg[1])), 9, 200, -1)

    ao_blur = cv2.GaussianBlur(contact_occ.astype(np.float32) / 255.0, (19, 19), 4.5)
    ground *= (1.0 - 0.44 * ao_blur)[:, :, None]

    asphalt_col = np.clip(np.array([116.0, 114.0, 112.0], dtype=np.float32) + med_n[:, :, None] * 4.0, 20, 235)
    ground[road_mask > 0] = asphalt_col[road_mask > 0]

    rough = np.where(road_mask > 0, 0.80 + med_n * 0.04, 0.92 + med_n * 0.04)
    rough_u8 = np.clip(cv2.resize((rough * 255.0).astype(np.uint8), (512, 512), interpolation=cv2.INTER_AREA), 60, 248)
    ground_path = out_dir / "farmland_ground_albedo.jpg"
    rough_path = out_dir / "farmland_ground_roughness.jpg"
    Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8)).save(ground_path, quality=86)
    Image.fromarray(rough_u8).save(rough_path, quality=82)

    # Convert sRGB outer meadow base color to linear RGB for Blender Horizon_Meadow_Skirt
    horizon_lin = [round(float((c / 255.0) ** 2.2), 4) for c in base_rgb]
    return {"albedo": str(ground_path), "roughness": str(rough_path)}, horizon_lin


def render_diagnostic_overlays(
    out_dir: Path,
    ref_rgb: np.ndarray,
    bev: dict[str, Any],
    spec: dict[str, Any],
    cam_pose: dict[str, Any],
    intrinsics_k: np.ndarray,
) -> None:
    """Save visual debugging overlays: photo_detections.png (reprojected 3D roofs+road) and bev_overlay.png."""
    photo_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)
    r_cw, t_cw = cam_pose["r_cw"], cam_pose["t_cw"]

    cl_pts = np.asarray(spec["road_centerline_xz"], dtype=np.float64)
    if len(cl_pts) >= 2:
        cl3d = np.column_stack([cl_pts[:, 0], np.zeros(len(cl_pts)), cl_pts[:, 1]])
        uv_r, zr = project_world_to_photo(cl3d, r_cw, t_cw, intrinsics_k)
        ok_r = zr > 2.0
        if ok_r.sum() >= 2:
            pts_i = np.round(uv_r[ok_r]).astype(np.int32)
            cv2.polylines(photo_bgr, [pts_i], False, (40, 200, 255), 3, cv2.LINE_AA)

    for b in spec["buildings"]:
        top = np.asarray(b["points_3d"], dtype=np.float64)
        uv_t, zt = project_world_to_photo(top, r_cw, t_cw, intrinsics_k)
        if np.all(zt > 2.0):
            cv2.polylines(photo_bgr, [np.round(uv_t).astype(np.int32)], True, (80, 255, 110), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "photo_detections.png"), photo_bgr)

    overlay = cv2.cvtColor(bev["ortho"], cv2.COLOR_RGB2BGR)
    bx0, bx1, bz0, bz1 = bev["bounds_xz"]
    gh, gw = bev["ortho"].shape[:2]

    def _to_px(pts_xz: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.clip((pts_xz[:, 0] - bx0) / (bx1 - bx0) * (gw - 1), 0, gw - 1), np.clip((pts_xz[:, 1] - bz0) / (bz1 - bz0) * (gh - 1), 0, gh - 1)],
            axis=-1,
        ).astype(np.int32)

    cv2.polylines(overlay, [_to_px(np.asarray(spec["road_apron_polygon_xz"], dtype=np.float64))], True, (60, 80, 255), 2)
    for b in spec["buildings"]:
        cv2.polylines(overlay, [_to_px(np.asarray(b["footprint_xz"], dtype=np.float64))], True, (80, 255, 110), 2)
    for tr in spec["trees"]:
        tp = _to_px(np.array([tr["xz"]], dtype=np.float64))[0]
        cv2.circle(overlay, (int(tp[0]), int(tp[1])), max(2, int(round(tr["radius_m"] / bev["res_m"]))), (60, 210, 255), 1)
    ndsm_vis = np.clip(bev["ndsm"] / 10.0 * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "bev_ndsm.png"), cv2.applyColorMap(ndsm_vis, cv2.COLORMAP_TURBO))
    cv2.imwrite(str(out_dir / "bev_overlay.png"), overlay)


def run_da3_multiview_refinement(
    ref_cam: dict[str, Any],
    novel_cams: list[dict[str, Any]],
    stage1_pts_world: np.ndarray,
    stage1_rgb: np.ndarray,
    stage1_conf: np.ndarray,
    stage1_valid: np.ndarray,
    *,
    stage1_bev: dict[str, Any] | None = None,
    model_id: str = "depth-anything/DA3-LARGE-1.1",
    process_res: int = 672,
) -> dict[str, Any]:
    """Run pose-conditioned DA3 across CAM_ref + synthesized views and extract blind-spot 3D points."""
    import torch

    valid_pts = stage1_pts_world[stage1_valid]
    valid_rgb = stage1_rgb[stage1_valid]
    valid_conf = stage1_conf[stage1_valid]
    empty_pts = np.zeros((0, 3), dtype=np.float32)
    empty_rgb = np.zeros((0, 3), dtype=np.uint8)
    empty_conf = np.zeros((0,), dtype=np.float32)
    if not novel_cams:
        return {
            "pts_world": stage1_pts_world,
            "pts_rgb": stage1_rgb,
            "pts_conf": stage1_conf,
            "valid_mask_flat": stage1_valid,
            "blindspot_pts": empty_pts,
            "blindspot_rgb": empty_rgb,
            "blindspot_conf": empty_conf,
            "view_stats": [],
        }

    obs_dilated: np.ndarray | None = None
    bx0 = bz0 = res_m = 0.0
    gw = gh = 0
    if stage1_bev is not None:
        obs_dilated = cv2.dilate((stage1_bev["obs_count"] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        bx0, _, bz0, _ = stage1_bev["bounds_xz"]
        res_m = float(stage1_bev["res_m"])
        gw = int(stage1_bev["gw"])
        gh = int(stage1_bev["gh"])

    # Subsample Stage-1 world anchors inside core elevation band for fast Huber affine depth calibration
    anchor_mask = (valid_pts[:, 1] >= -1.5) & (valid_pts[:, 1] <= 18.0) & np.isfinite(valid_pts).all(axis=1)
    anchors = valid_pts[anchor_mask]
    rng = np.random.default_rng(77)
    if len(anchors) > 18000:
        anchors = anchors[rng.choice(len(anchors), 18000, replace=False)]

    all_cams = [ref_cam] + list(novel_cams)
    paths: list[str] = []
    exts: list[np.ndarray] = []
    ixts: list[np.ndarray] = []
    for c in all_cams:
        ext = np.eye(4, dtype=np.float32)
        ext[:3, :3] = np.asarray(c["r_cw"], dtype=np.float32)
        ext[:3, 3] = np.asarray(c["t_cw"], dtype=np.float32)
        ixt = np.array(
            [[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        paths.append(c["image_path"])
        exts.append(ext)
        ixts.append(ixt)

    os.environ["HF_HUB_OFFLINE"] = "1"
    _install_da3_optional_shims()
    import depth_anything_3.api as da3_api

    da3_api.align_poses_umeyama = lambda pred_ext, in_ext, *a, **kw: (None, None, 1.0, pred_ext)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = da3_api.DepthAnything3.from_pretrained(model_id).to(device)
    pred = model.inference(
        paths,
        extrinsics=np.stack(exts),
        intrinsics=np.stack(ixts),
        align_to_input_ext_scale=True,
        export_dir=None,
        process_res=process_res,
    )

    fused_xyz: list[np.ndarray] = [valid_pts.astype(np.float32)]
    fused_rgb: list[np.ndarray] = [valid_rgb.astype(np.uint8)]
    fused_w: list[np.ndarray] = [(np.maximum(valid_conf, 0.3) * 2.4).astype(np.float32)]
    blind_xyz: list[np.ndarray] = []
    blind_rgb: list[np.ndarray] = []
    blind_w: list[np.ndarray] = []
    view_stats: list[dict[str, Any]] = []

    for fi, cspec in enumerate(all_cams):
        if fi == 0:
            # Stage-1 already contributes full-resolution 1.57M points for CAM_ref
            continue
        dmap = pred.depth[fi].astype(np.float64)
        conf = pred.conf[fi].astype(np.float64)
        rgb_mv = pred.processed_images[fi]
        ph, pw = dmap.shape
        k_proc = pred.intrinsics[fi].astype(np.float64)
        ext = exts[fi].astype(np.float64)
        r_cw, t_cw = ext[:3, :3], ext[:3, 3]
        c2w = np.linalg.inv(ext)
        k_inv = np.linalg.inv(k_proc)

        xc = anchors @ r_cw.T + t_cw
        z_col = xc[:, 2]
        ok = z_col > 4.0
        ui = np.round(xc[:, 0] / np.maximum(z_col, 1e-4) * k_proc[0, 0] + k_proc[0, 2]).astype(int)
        vi = np.round(xc[:, 1] / np.maximum(z_col, 1e-4) * k_proc[1, 1] + k_proc[1, 2]).astype(int)
        ok &= (ui >= 2) & (ui < pw - 2) & (vi >= 2) & (vi < ph - 2)
        if int(ok.sum()) < 40:
            view_stats.append({"cam_key": cspec["cam_key"], "accepted_depth": False, "reason": "too_few_anchors"})
            continue

        zp_raw = dmap[vi[ok], ui[ok]]
        cf_raw = conf[vi[ok], ui[ok]]
        zc_raw = z_col[ok]
        hi = (zp_raw > 1e-3) & (cf_raw >= np.percentile(cf_raw, 40) if len(cf_raw) > 20 else True)
        zp, zc = zp_raw[hi], zc_raw[hi]
        if len(zp) < 30:
            view_stats.append({"cam_key": cspec["cam_key"], "accepted_depth": False, "reason": "low_conf_anchors"})
            continue

        a_fit, b_fit, med_err = _huber_affine_depth_fit(zp, zc)
        d_metric = np.maximum(a_fit * dmap + b_fit, 1.0)
        rays_chk = k_inv @ np.vstack([ui[ok][hi], vi[ok][hi], np.ones(len(zc))])
        xw_pred = (c2w[:3, :3] @ (rays_chk * (a_fit * zp + b_fit)[None, :])).T + c2w[:3, 3]
        xw_col = (c2w[:3, :3] @ (rays_chk * zc[None, :])).T + c2w[:3, 3]
        med_dy = float(np.median(np.abs(xw_pred[:, 1] - xw_col[:, 1])))

        if med_dy > 4.5:
            view_stats.append(
                {"cam_key": cspec["cam_key"], "accepted_depth": False, "med_dy_m": round(med_dy, 3), "med_err_m": round(med_err, 3)}
            )
            continue

        us_g, vs_g = np.meshgrid(np.arange(pw), np.arange(ph))
        valid_px = np.isfinite(d_metric) & (d_metric >= 6.0) & (conf >= np.percentile(conf, 30))
        vidx = np.flatnonzero(valid_px.reshape(-1))
        if len(vidx) > 0:
            pix_v = np.stack([us_g.reshape(-1)[vidx], vs_g.reshape(-1)[vidx], np.ones(len(vidx))], axis=-1)
            rays = k_inv @ pix_v.T
            xw_dense = ((c2w[:3, :3] @ (rays * d_metric.reshape(-1)[vidx][None, :])).T + c2w[:3, 3]).astype(np.float32)
            in_box = (xw_dense[:, 1] >= -2.5) & (xw_dense[:, 1] <= 25.0) & np.isfinite(xw_dense).all(axis=1)
            if in_box.any():
                rel_w = float(1.15 / max(0.55, med_dy))
                w_mv = (rel_w * np.clip(conf.reshape(-1)[vidx] / 2.0, 0.4, 1.8)).astype(np.float32)
                xw_in = xw_dense[in_box]
                rgb_in = rgb_mv.reshape(-1, 3)[vidx][in_box]
                w_in = w_mv[in_box]
                if obs_dilated is not None:
                    ix_g = np.floor((xw_in[:, 0] - bx0) / res_m).astype(np.int32)
                    iz_g = np.floor((xw_in[:, 2] - bz0) / res_m).astype(np.int32)
                    in_grid = (ix_g >= 0) & (ix_g < gw) & (iz_g >= 0) & (iz_g < gh)
                    is_blind = np.zeros(len(xw_in), dtype=bool)
                    is_blind[in_grid] = ~obs_dilated[iz_g[in_grid], ix_g[in_grid]]
                    xw_in = xw_in[is_blind]
                    rgb_in = rgb_in[is_blind]
                    w_in = w_in[is_blind]
                if len(xw_in) > 0:
                    blind_xyz.append(xw_in)
                    blind_rgb.append(rgb_in)
                    blind_w.append(w_in)
                    fused_xyz.append(xw_in)
                    fused_rgb.append(rgb_in)
                    fused_w.append(w_in)
                view_stats.append(
                    {
                        "cam_key": cspec["cam_key"],
                        "accepted_depth": True,
                        "anchors": len(zc),
                        "med_err_m": round(med_err, 3),
                        "med_dy_m": round(med_dy, 3),
                        "points_added": int(len(xw_in)),
                    }
                )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    pts_out = np.concatenate(fused_xyz, axis=0)
    b_pts = np.concatenate(blind_xyz, axis=0) if blind_xyz else empty_pts
    b_rgb = np.concatenate(blind_rgb, axis=0) if blind_rgb else empty_rgb
    b_conf = np.concatenate(blind_w, axis=0) if blind_w else empty_conf
    return {
        "pts_world": pts_out,
        "pts_rgb": np.concatenate(fused_rgb, axis=0),
        "pts_conf": np.concatenate(fused_w, axis=0),
        "valid_mask_flat": np.ones(len(pts_out), dtype=bool),
        "blindspot_pts": b_pts,
        "blindspot_rgb": b_rgb,
        "blindspot_conf": b_conf,
        "view_stats": view_stats,
    }


def _merge_blindspot_bev(
    stage1_bev: dict[str, Any],
    mv_pts: np.ndarray,
    mv_rgb: np.ndarray,
    mv_conf: np.ndarray,
) -> dict[str, Any]:
    """Infill strictly unobserved Stage-1 BEV cells from blind-spot 3D points without altering CAM_ref cells."""
    out = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in stage1_bev.items()
    }
    if len(mv_pts) == 0:
        return out

    bx0, _, bz0, _ = out["bounds_xz"]
    res_m = float(out["res_m"])
    gw, gh = int(out["gw"]), int(out["gh"])
    v = np.isfinite(mv_pts).all(axis=1) & (mv_pts[:, 1] >= -3.0) & (mv_pts[:, 1] <= 30.0)
    if not v.any():
        return out

    pw = mv_pts[v]
    rgb = mv_rgb[v]
    conf = np.maximum(mv_conf[v], 0.2)
    ix = np.floor((pw[:, 0] - bx0) / res_m).astype(np.int32)
    iz = np.floor((pw[:, 2] - bz0) / res_m).astype(np.int32)
    in_b = (ix >= 0) & (ix < gw) & (iz >= 0) & (iz < gh)
    if not in_b.any():
        return out

    ix, iz = ix[in_b], iz[in_b]
    unseen_pt = stage1_bev["obs_count"][iz, ix] == 0
    if not unseen_pt.any():
        return out

    ix, iz = ix[unseen_pt], iz[unseen_pt]
    y_v = pw[in_b, 1][unseen_pt].astype(np.float32)
    rgb_v = rgb[in_b][unseen_pt]
    w_v = conf[in_b][unseen_pt]

    flat = iz.astype(np.int64) * gw + ix.astype(np.int64)
    order = np.lexsort((y_v, flat))
    flat_s, y_s, rgb_s, w_s = flat[order], y_v[order], rgb_v[order], w_v[order]
    uniq, starts, counts = np.unique(flat_s, return_index=True, return_counts=True)

    dsm_flat = np.full(gh * gw, np.nan, dtype=np.float32)
    cnt_flat = np.zeros(gh * gw, dtype=np.int32)
    ortho_flat = np.zeros((gh * gw, 3), dtype=np.uint8)

    idx_p82 = starts + np.minimum(counts - 1, np.floor(counts * 0.82).astype(np.int64))
    dsm_flat[uniq] = y_s[idx_p82]
    cnt_flat[uniq] = counts.astype(np.int32)
    top_ref = np.repeat(y_s[idx_p82], counts)
    pw_col = np.where(np.abs(y_s - top_ref) <= 1.25, w_s, w_s * 0.08).astype(np.float32)
    w_sum = np.add.reduceat(pw_col, starts)
    inv_w = 1.0 / np.maximum(w_sum, 1e-4)
    for ch in range(3):
        c_sum = np.add.reduceat(rgb_s[:, ch].astype(np.float32) * pw_col, starts)
        ortho_flat[uniq, ch] = np.clip(c_sum * inv_w, 0, 255).astype(np.uint8)

    new_mask = (cnt_flat.reshape(gh, gw) > 0) & (stage1_bev["obs_count"] == 0)
    if not new_mask.any():
        return out

    dsm_new = dsm_flat.reshape(gh, gw)
    ortho_new = ortho_flat.reshape(gh, gw, 3)
    out["dsm"][new_mask] = dsm_new[new_mask]
    out["obs_count"][new_mask] = cnt_flat.reshape(gh, gw)[new_mask]
    out["ortho"][new_mask] = ortho_new[new_mask]
    out["ndsm"][new_mask] = np.clip(out["dsm"][new_mask] - out["dtm"][new_mask], 0.0, 22.0)

    veg_new = compute_green_mask(out["ortho"]) & new_mask
    is_tall_tree = veg_new & (out["ndsm"] >= 2.2)
    out["veg_mask"][new_mask] = is_tall_tree[new_mask]
    out["grass_mask"][new_mask] = (veg_new & (~is_tall_tree))[new_mask]
    return out


def _assemble_and_write_spec(
    *,
    photo_path: Path,
    ref_clean_path: Path,
    out_dir: Path,
    scene_code: str,
    da3: dict[str, Any],
    cam_pose: dict[str, Any],
    bev: dict[str, Any],
    road_info: dict[str, Any],
    detections: dict[str, list[dict[str, Any]]],
    cameras: list[dict[str, Any]],
    manifest: dict[str, Any],
    anchor_buildings: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Build buildings/trees/cars/walls/ground atlas from BEV + cameras and write village_spec.json."""
    rgb_u8 = da3["rgb_u8"]
    bldg_out = extract_buildings_from_bev(bev, road_info, rgb_u8, cam_pose, da3["intrinsics"])
    if anchor_buildings:
        buildings = json.loads(json.dumps(anchor_buildings))
        accepted_polys = [Polygon(b["footprint_xz"]).buffer(0.0) for b in buildings]
        apron_shp = Polygon(road_info["road_apron_polygon_xz"]).buffer(0.0)
        road_shp = Polygon(road_info["road_polygon_xz"]).buffer(0.0)
        cl = np.asarray(road_info["centerline_xz"], dtype=np.float64)
        for cand in bldg_out["buildings"]:
            c_poly = Polygon(cand["footprint_xz"]).buffer(0.0)
            if not c_poly.is_valid or c_poly.area < 14.0:
                continue
            if not (2.6 <= float(cand["height_m"]) <= 12.5):
                continue
            if c_poly.intersects(road_shp) or c_poly.distance(apron_shp) < 0.40:
                continue
            if any(c_poly.distance(ap) < 1.2 for ap in accepted_polys):
                continue
            new_b = json.loads(json.dumps(cand))
            new_b["is_core"] = False
            new_b["source"] = "multiview_blindspot_bev"
            new_b["index"] = len(buildings)
            new_b["id"] = f"house_{len(buildings) + 1:02d}"
            poly = np.asarray(new_b["footprint_xz"], dtype=np.float64)
            center_xz = poly.mean(0)
            _, sd_road, _, _, near_road_pt = local_road_frame(center_xz, cl)
            road_d = abs(sd_road)
            to_road = near_road_pt - center_xz
            to_road /= max(float(np.linalg.norm(to_road)), 1e-6)
            all_polys_arr = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in buildings] + [poly]
            new_idx = len(buildings)
            walls = []
            for ei in range(len(poly)):
                p0, p1 = poly[ei], poly[(ei + 1) % len(poly)]
                edge = p1 - p0
                elen = float(np.linalg.norm(edge))
                cand_n = np.array([-edge[1], edge[0]], dtype=np.float64) / max(elen, 1e-6)
                if float(np.dot(cand_n, 0.5 * (p0 + p1) - center_xz)) < 0.0:
                    cand_n = -cand_n
                gap = _edge_neighbor_gap(p0, p1, new_idx, all_polys_arr)
                faces_street = float(np.dot(cand_n, to_road)) > 0.42 and road_d < 30.0
                walls.append(
                    {
                        "edge_index": ei,
                        "length_m": round(elen, 2),
                        "outward_xz": [round(float(cand_n[0]), 4), round(float(cand_n[1]), 4)],
                        "neighbor_gap_m": round(gap, 2),
                        "role": "party_wall" if gap < 1.15 else "facade",
                        "swatch": new_b["wall_swatch"] if faces_street else new_b["side_wall_swatch"],
                    }
                )
            new_b["walls"] = walls
            buildings.append(new_b)
            accepted_polys.append(c_poly)
        layout_audit = audit_layout_geometry(
            buildings, road_info["centerline_xz"], road_info["road_polygon_xz"], road_info["road_apron_polygon_xz"]
        )
        if layout_audit["verdict"] != "pass":
            buildings = json.loads(json.dumps(anchor_buildings))
            layout_audit = audit_layout_geometry(
                buildings, road_info["centerline_xz"], road_info["road_polygon_xz"], road_info["road_apron_polygon_xz"]
            )
    else:
        buildings = bldg_out["buildings"]
        layout_audit = bldg_out["layout_audit"]
    trees = extract_trees_from_bev(bev, buildings, road_info)
    parked_cars = extract_parked_cars(bev, buildings, road_info, detections, cam_pose, da3["intrinsics"])

    all_polys = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in buildings]
    cw_walls, _ = build_courtyard_network(
        buildings, all_polys, road_info["road_polygon_xz"], road_info["centerline_xz"], parked_cars, road_info["road_apron_polygon_xz"]
    )
    cw_walls = [
        {
            **w,
            "height_m": min(float(w.get("height_m", 1.1)), 1.05),
            "swatch": "plaster_warm" if (wi % 2 == 0) else "plaster_grey",
        }
        for wi, w in enumerate(cw_walls[:55])
    ]

    bx0, bx1, bz0, bz1 = bev["bounds_xz"]
    ground_bounds = [round(bx0 - 65.0, 1), round(bx1 + 65.0, 1), round(bz0 - 75.0, 1), round(bz1 + 75.0, 1)]
    swatches = dict(manifest["swatches"])
    foliage_swatches = dict(manifest["foliage"])
    ground_sw, horizon_rgb = build_adaptive_ground_texture(
        out_dir, bev, ground_bounds, buildings, cw_walls, parked_cars, road_info, swatches
    )
    swatches["farmland_ground"] = ground_sw

    full_w, full_h = da3["image_size"]
    eye_dist = float(np.linalg.norm(cam_pose["eye_w"]))
    mv_count = sum(1 for c in cameras if c.get("allow_facade_bake"))
    spec = {
        "version": 3,
        "map_id": scene_code,
        "source_image": str(photo_path),
        "ref_clean_image": str(ref_clean_path.resolve()),
        "adaptive_qc_views": True,
        "horizon_color_rgb": horizon_rgb,
        "road_apron_width_m": float(road_info["apron_width"]),
        "road_extend_m": 6.0,
        "sun_elevation_deg": 40.0,
        "sun_azimuth_deg": 38.0,
        "sun_energy": 4.9,
        "photo_facade_max_dist": round(max(120.0, eye_dist * 1.45), 1),
        "photo_facade_ref_dist": round(max(60.0, eye_dist * 0.85), 1),
        "photo_facade_min_cos": 0.48,
        "camera": {
            "position": [round(float(v), 4) for v in cam_pose["eye_w"]],
            "target": [0.0, 0.0, 0.0],
            "fov": round(float(cam_pose["fov_y_deg"]), 4),
        },
        "imageSize": [int(full_w), int(full_h)],
        "groundBounds": ground_bounds,
        "road_polygon_xz": road_info["road_polygon_xz"],
        "road_apron_polygon_xz": road_info["road_apron_polygon_xz"],
        "road_left_edge_xz": road_info["road_left_edge_xz"],
        "road_right_edge_xz": road_info["road_right_edge_xz"],
        "road_centerline_xz": road_info["centerline_xz"],
        "layout_audit": layout_audit,
        "utility_poles_xz": [],
        "parked_cars": parked_cars,
        "tricycles": [],
        "brick_piles": [],
        "courtyard_walls": cw_walls,
        "courtyard_gates": [],
        "trees": trees,
        "shrubs": [],
        "cameras": cameras,
        "buildings": buildings,
        "swatches": swatches,
        "foliage_swatches": foliage_swatches,
        "calibration": {
            "scale_factor": round(cam_pose["scale_factor"], 3),
            "pitch_deg": cam_pose["pitch_deg"],
            "road_detected": bool(road_info["detected"]),
            "bev_bounds_xz": list(bev["bounds_xz"]),
            "bev_res_m": round(float(bev["res_m"]), 3),
            "multiview_cameras": mv_count,
        },
        "notes": (
            f"{'多视角 AI 融合重建' if mv_count > 0 else '单图通用重建'}：{len(buildings)} 栋闭合建筑 + "
            f"{len(trees)} 棵树木 + {len(parked_cars)} 辆路边车辆 + {1 + mv_count} 视角 DA3 融合与 PBR 摄影烘焙。"
        ),
    }

    spec_path = out_dir / "village_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    render_diagnostic_overlays(out_dir, rgb_u8, bev, spec, cam_pose, da3["intrinsics"])
    obs_cov_pct = round(float((bev["obs_count"] > 0).mean() * 100.0), 2)
    stats = {
        "buildings": len(buildings),
        "trees": len(trees),
        "cars": len(parked_cars),
        "walls": len(cw_walls),
        "cameras": len(cameras),
        "bev_obs_coverage_pct": obs_cov_pct,
    }
    return spec, spec_path, stats


def reconstruct_from_photo(
    *,
    photo_path: Path,
    scene_code: str,
    label: str,
    out_dir: Path,
    output_glb: Path,
    pbr_cache: Path | None = None,
    skip_blender: bool = False,
    multiview_provider: str = "none",
    multiview_angles: tuple[float, ...] | list[float] = (-60.0, 60.0, 180.0),
    comfy_url: str | None = None,
    gateway_upstream: str | None = None,
    gpt_model: str = "gpt-image-2",
    gpt_quality: str = "high",
    force_multiview: bool = False,
) -> dict[str, Any]:
    """Run single-photo or multi-view AI-completed 3D reconstruction -> Blender 4.5 PBR -> audited GLB export."""
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_clean_path = out_dir / "ref_clean.png"
    with Image.open(photo_path) as im:
        im.convert("RGB").save(ref_clean_path)

    print(f"[Stage A] Running DA3-LARGE-1.1 on {photo_path.name}...")
    da3 = run_da3_single_photo(photo_path)
    rgb_u8 = da3["rgb_u8"]
    detections = detect_semantics_gdino(rgb_u8)

    print("[Stage B] Fitting ground plane & calibrating metric camera pose...")
    cam_pose = fit_ground_plane_and_camera(rgb_u8, da3["depth"], da3["intrinsics"], da3["conf"], detections)
    print(
        f"  Calibration: pitch={cam_pose['pitch_deg']}deg, eye_w={np.round(cam_pose['eye_w'], 2).tolist()}, "
        f"scale={cam_pose['scale_factor']:.2f} (relief_raw={cam_pose['relief_p90_raw']:.4f}, fov={cam_pose['fov_y_deg']:.2f}deg)"
    )

    print("[Stage C] Rasterizing BEV nDSM & extracting road/buildings/trees...")
    bev = rasterize_single_view_bev(cam_pose["pts_world"], cam_pose["pts_rgb"], cam_pose["pts_conf"], cam_pose["valid_mask_flat"])
    road_info = extract_road_centerline(bev, detections, cam_pose, da3["intrinsics"])
    manifest = fetch_and_build_pbr_assets(out_dir, offline=True, cache_dir=pbr_cache, photo_path=photo_path)

    full_w, full_h = da3["image_size"]
    k = da3["intrinsics"]
    sharp = laplacian_sharpness(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
    ref_cam = {
        "cam_key": "ref",
        "source_clip": "ref",
        "colmap_name": "ref/ref.png",
        "image_path": str(ref_clean_path.resolve()),
        "width": int(full_w),
        "height": int(full_h),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "r_cw": [[round(float(v), 8) for v in row] for row in cam_pose["r_cw"]],
        "t_cw": [round(float(v), 6) for v in cam_pose["t_cw"]],
        "blender_eye": [round(float(v), 6) for v in cam_pose["blender_eye"]],
        "blender_mat4": cam_pose["blender_mat4"],
        "blender_intrinsics": cam_pose["blender_intrinsics"],
        "sharpness": round(sharp, 2),
        "priority": 4.5,
        "norm_t": 0.0,
        "gain_rgb": [1.0, 1.0, 1.0],
        "watermark_box": [1.0, 1.0, 1.0, 1.0],
    }

    spec, spec_path, pass1_stats = _assemble_and_write_spec(
        photo_path=photo_path,
        ref_clean_path=ref_clean_path,
        out_dir=out_dir,
        scene_code=scene_code,
        da3=da3,
        cam_pose=cam_pose,
        bev=bev,
        road_info=road_info,
        detections=detections,
        cameras=[ref_cam],
        manifest=manifest,
    )
    print(f"  Pass-1 Spec ready -> {spec_path} ({pass1_stats})")

    if skip_blender:
        return {"spec": str(spec_path), **pass1_stats, "elapsed_s": round(time.monotonic() - t0, 1)}

    blend_path = out_dir / "village_master.blend"
    uv_map_path = out_dir / "uv_atlas_map.json"
    probes_path = out_dir / "bake_probes.npz"

    def _run_blender_build_and_bake(stage_prefix: str) -> dict[str, Any]:
        print(f"[{stage_prefix}1] Building watertight 3D PBR scene in Blender 4.5...")
        run_blender_script(ROOT / "tools/blender_village/build_village.py", ["--spec", str(spec_path), "--blend", str(blend_path)])
        print(f"[{stage_prefix}2] Exporting BVH visibility & AO probes...")
        run_blender_script(
            ROOT / "tools/blender_village/export_bake_buffers.py",
            ["--spec", str(spec_path), "--uv-map", str(uv_map_path), "--out", str(probes_path)],
            blend_path=blend_path,
        )
        print(f"[{stage_prefix}3] Baking 2K PBR + multi-view photo-facade atlases...")
        return bake_atlases(spec_path=spec_path, uv_map_path=uv_map_path, probes_path=probes_path, out_dir=out_dir, blend_path=blend_path)

    bake_pass1 = _run_blender_build_and_bake("Pass1-")
    pass1_stats["photo_facade_faces"] = bake_pass1["photo_facade_faces"]
    pass1_stats["mean_observed_coverage"] = bake_pass1["mean_observed_coverage"]

    mv_summary: dict[str, Any] | None = None
    final_stats = dict(pass1_stats)

    if (multiview_provider or "none").lower() != "none":
        print(f"[MultiView] Synthesizing & aligning novel angles {list(multiview_angles)} via '{multiview_provider}'...")
        mv_res = synthesize_novel_views(
            spec_path=spec_path,
            blend_path=blend_path,
            out_dir=out_dir,
            angles_deg=list(multiview_angles),
            provider=multiview_provider,
            comfy_url=comfy_url,
            gateway_upstream=gateway_upstream,
            gpt_model=gpt_model,
            gpt_quality=gpt_quality,
            force=force_multiview,
        )
        novel_cams = mv_res["cameras"]
        if novel_cams:
            print(f"[MultiView] Running pose-conditioned DA3 blind-spot geometry fusion across 1 + {len(novel_cams)} views...")
            mv_cloud = run_da3_multiview_refinement(
                ref_cam,
                novel_cams,
                cam_pose["pts_world"],
                cam_pose["pts_rgb"],
                cam_pose["pts_conf"],
                cam_pose["valid_mask_flat"],
                stage1_bev=bev,
            )
            bev_mv = _merge_blindspot_bev(
                bev,
                mv_cloud["blindspot_pts"],
                mv_cloud["blindspot_rgb"],
                mv_cloud["blindspot_conf"],
            )
            road_mv = road_info if road_info.get("detected") else extract_road_centerline(bev_mv, detections, cam_pose, da3["intrinsics"])
            spec, spec_path, pass2_stats = _assemble_and_write_spec(
                photo_path=photo_path,
                ref_clean_path=ref_clean_path,
                out_dir=out_dir,
                scene_code=scene_code,
                da3=da3,
                cam_pose=cam_pose,
                bev=bev_mv,
                road_info=road_mv,
                detections=detections,
                cameras=[ref_cam] + novel_cams,
                manifest=manifest,
                anchor_buildings=spec["buildings"],
            )
            bake_pass2 = _run_blender_build_and_bake("Pass2-")
            pass2_stats["photo_facade_faces"] = bake_pass2["photo_facade_faces"]
            pass2_stats["mean_observed_coverage"] = bake_pass2["mean_observed_coverage"]
            final_stats = dict(pass2_stats)
            mv_summary = {
                "provider_used": mv_res["provider_used"],
                "pass1_single_view": pass1_stats,
                "pass2_multiview": pass2_stats,
                "synthesis_report": mv_res["report"],
                "da3_fusion_stats": mv_cloud["view_stats"],
            }
            (out_dir / "multiview_summary.json").write_text(json.dumps(mv_summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"  MultiView Gain: facades {pass1_stats['photo_facade_faces']} -> {pass2_stats['photo_facade_faces']}, "
                f"obs_cov {pass1_stats['mean_observed_coverage']:.4f} -> {pass2_stats['mean_observed_coverage']:.4f}, "
                f"bev_cov {pass1_stats['bev_obs_coverage_pct']}% -> {pass2_stats['bev_obs_coverage_pct']}%"
            )
        else:
            print("[WARN] MultiView synthesis produced 0 accepted aligned views; keeping Pass-1 single-view scene.")

    print("[Stage G] Running automated audit gate & multi-angle EEVEE QC...")
    qc_metrics = run_visual_qc(spec_path, blend_path, out_dir / "qc")
    print("[Stage H] Exporting audited GLB + metadata...")
    run_blender_script(
        ROOT / "tools/blender_village/export_village.py",
        ["--spec", str(spec_path), "--blend", str(blend_path), "--output", str(output_glb), "--label", label],
        blend_path=blend_path,
    )

    elapsed = round(time.monotonic() - t0, 1)
    summary = {
        "code": scene_code,
        "glb": str(output_glb),
        "spec": str(spec_path),
        **final_stats,
        "multiview": mv_summary,
        "qc_metrics": qc_metrics,
        "elapsed_s": elapsed,
    }
    print("RECONSTRUCT_FROM_PHOTO_OK", json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--pbr-cache", type=Path, default=ROOT / "artifacts/blender_village/pbr_cache")
    p.add_argument("--skip-blender", action="store_true")
    p.add_argument(
        "--multiview-provider",
        default="none",
        choices=("none", "auto", "gpt_image_2", "minimax_h3_image", "minimax_h3_video", "both"),
    )
    p.add_argument("--multiview-angles", type=float, nargs="+", default=[-60.0, 60.0, 180.0])
    p.add_argument("--comfy-url", default=None)
    p.add_argument("--gateway-upstream", default=None, help="Optional trycloudflare URL to hot-takeover on local gateway")
    p.add_argument("--gpt-model", default="gpt-image-2")
    p.add_argument("--gpt-quality", default="high")
    p.add_argument("--force-multiview", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else (ROOT / "artifacts/generalization_test" / args.code.lower())
    output_glb = args.output if args.output is not None else (ROOT / "assets/scene_meshes" / f"{args.code}.glb")
    label = args.label or f"{args.code} · {'多视角AI融合重建' if args.multiview_provider != 'none' else '单图无人机通用重建'}"
    reconstruct_from_photo(
        photo_path=args.photo.resolve(),
        scene_code=args.code,
        label=label,
        out_dir=out_dir.resolve(),
        output_glb=output_glb.resolve(),
        pbr_cache=args.pbr_cache.resolve() if args.pbr_cache else None,
        skip_blender=args.skip_blender,
        multiview_provider=args.multiview_provider,
        multiview_angles=args.multiview_angles,
        comfy_url=args.comfy_url,
        gateway_upstream=args.gateway_upstream,
        gpt_model=args.gpt_model,
        gpt_quality=args.gpt_quality,
        force_multiview=args.force_multiview,
    )
