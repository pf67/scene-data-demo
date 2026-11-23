"""Ingest photo + calibrated COLMAP video frames/points + CC0 PBR swatches into dense village_spec.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "tools/scene_pipeline"),
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/blender_village"),
    str(ROOT / "builder"),
]

from blender_bridge_addon import (  # noqa: E402
    colmap_intrinsics_to_blender_camera,
    colmap_pose_to_blender_matrix,
    point_in_polygon_2d,
)
from fetch_pbr_assets import fetch_and_build_pbr_assets  # noqa: E402
from layout_solver import (  # noqa: E402
    audit_layout_geometry,
    build_regularized_road,
    ensure_ccw_blender_xz,
    generate_northern_village_fabric,
    local_road_frame,
    regularize_footprint_to_grid,
    solve_building_layout,
)
from minimax_h3_orbit import clean_watermark  # noqa: E402


# Targeted visual overrides verified directly against village_road_near.png & clip1/clip3 overlays
BUILDING_VISUAL_OVERRIDES: dict[str, dict[str, Any]] = {
    "house_01": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_white", "note": "Foreground left silver gable whitewashed walls"},
    "house_02": {"wall_swatch": "plaster_warm", "side_wall_swatch": "plaster_warm", "note": "Foreground left 2-story warm aged stucco"},
    "house_03": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_warm", "note": "Foreground center-left flat whitewashed house"},
    "house_05": {"roof_type": "flat_parapet", "height_m": 5.6, "wall_swatch": "plaster_warm", "side_wall_swatch": "plaster_warm", "note": "Photo shows flat concrete parapet shopfront, not gable"},
    "house_06": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_white", "note": "Foreground center whitewashed 2-story house"},
    "house_08": {"roof_type": "flat_parapet", "height_m": 4.5, "wall_swatch": "plaster_white", "side_wall_swatch": "plaster_warm", "note": "1-story flat roof + rooftop solar heaters confirmed by photo & COLMAP (4.53m)"},
    "house_09": {"roof_type": "gable_red_tile", "height_m": 7.1, "ridge_rise": 1.45, "wall_swatch": "plaster_white", "side_wall_swatch": "plaster_white", "note": "Prominent 2-story terracotta tile gable confirmed by COLMAP"},
    "house_10": {"roof_type": "flat_parapet", "height_m": 4.3, "wall_swatch": "plaster_white", "side_wall_swatch": "plaster_warm", "note": "1-story concrete flat house confirmed by photo & COLMAP"},
    "house_25": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_warm", "note": "Roadside north shopfront"},
    "house_27": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_warm", "note": "Roadside north shopfront"},
    "house_28": {"wall_swatch": "plaster_white", "side_wall_swatch": "plaster_white", "note": "Roadside north shopfront"},
}


def laplacian_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def parse_colmap_text(sparse_dir: Path) -> tuple[dict[int, dict], list[dict], np.ndarray, np.ndarray]:
    """Read cameras.txt, images.txt, points3D.txt into clean Python dicts and NumPy arrays."""
    cams: dict[int, dict] = {}
    for line in (sparse_dir / "cameras.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        cid = int(p[0])
        model = p[1]
        w, h = int(p[2]), int(p[3])
        params = [float(x) for x in p[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            fx, fy, cx, cy = params[:4]
        cams[cid] = {"id": cid, "model": model, "width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    images: list[dict] = []
    img_lines = [
        ln.strip()
        for ln in (sparse_dir / "images.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    for idx in range(0, len(img_lines), 2):
        p = img_lines[idx].split()
        img_id = int(p[0])
        quat_wxyz = [float(x) for x in p[1:5]]
        t_cw = [float(x) for x in p[5:8]]
        cam_id = int(p[8])
        name = " ".join(p[9:])
        images.append(
            {
                "id": img_id,
                "name": name,
                "cam_id": cam_id,
                "quat_wxyz": quat_wxyz,
                "t_cw": t_cw,
            }
        )

    xyz_list: list[list[float]] = []
    rgb_list: list[list[int]] = []
    pts_path = sparse_dir / "points3D.txt"
    if pts_path.is_file():
        for line in pts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            xyz_list.append([float(p[1]), float(p[2]), float(p[3])])
            rgb_list.append([int(p[4]), int(p[5]), int(p[6])])
    pts_xyz = np.asarray(xyz_list, dtype=np.float32) if xyz_list else np.zeros((0, 3), dtype=np.float32)
    pts_rgb = np.asarray(rgb_list, dtype=np.uint8) if rgb_list else np.zeros((0, 3), dtype=np.uint8)
    return cams, images, pts_xyz, pts_rgb


def select_calibrated_frames(
    cams: dict[int, dict],
    images: list[dict],
    images_dir: Path,
    ref_clean_path: Path,
    ref_mean_rgb: np.ndarray,
    *,
    per_clip_count: int = 10,
    tag_prefix: str = "",
) -> list[dict[str, Any]]:
    """Group COLMAP images by clip folder, pick sharpest evenly-spaced keyframes, and compute Blender cameras."""
    groups: dict[str, list[dict]] = {}
    for im in images:
        source = im["name"].split("/")[0] if "/" in im["name"] else "root"
        groups.setdefault(source, []).append(im)

    selected: list[dict[str, Any]] = []
    for source, group in sorted(groups.items()):
        if source == "ref" and tag_prefix:
            continue
        group_sorted = sorted(group, key=lambda item: item["name"])
        if source == "ref":
            chosen = group_sorted[:1]
        elif len(group_sorted) <= per_clip_count:
            chosen = group_sorted
        else:
            indices = np.linspace(0, len(group_sorted) - 1, per_clip_count, dtype=int)
            chosen = [group_sorted[i] for i in indices]

        n_group = max(1, len(group_sorted) - 1)
        rank_map = {item["name"]: idx for idx, item in enumerate(group_sorted)}

        for item in chosen:
            img_path = ref_clean_path if source == "ref" else (images_dir / item["name"])
            if not img_path.is_file():
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            sharpness = laplacian_sharpness(bgr)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            clip_mean = rgb.mean(axis=(0, 1))
            gain = np.clip(ref_mean_rgb / np.maximum(clip_mean, 8.0), 0.86, 1.16)

            cam = cams[item["cam_id"]]
            eye_b, _rot_b2w, mat4 = colmap_pose_to_blender_matrix(item["quat_wxyz"], item["t_cw"])
            bl_intr = colmap_intrinsics_to_blender_camera(
                cam["fx"], cam["fy"], cam["cx"], cam["cy"], cam["width"], cam["height"]
            )
            norm_t = rank_map[item["name"]] / n_group
            if source == "ref":
                priority = 3.8
            elif tag_prefix == "extra_":
                # Extra H3 360 orbit camera frames supply wide wraparound coverage
                priority = 2.4
            elif "clip3" in source:
                priority = float(2.2 * math.exp(-0.72 * norm_t))
            elif "clip1" in source:
                priority = float(1.55 * math.exp(-0.85 * norm_t))
            else:
                priority = 2.0

            from blender_bridge_addon import quat_wxyz_to_matrix

            r_cw = quat_wxyz_to_matrix(item["quat_wxyz"])
            selected.append(
                {
                    "cam_key": f"{tag_prefix}{item['name'].replace('/', '__').rsplit('.', 1)[0]}",
                    "source_clip": f"{tag_prefix}{source}",
                    "colmap_name": item["name"],
                    "image_path": str(img_path.resolve()),
                    "width": int(cam["width"]),
                    "height": int(cam["height"]),
                    "fx": float(cam["fx"]),
                    "fy": float(cam["fy"]),
                    "cx": float(cam["cx"]),
                    "cy": float(cam["cy"]),
                    "r_cw": r_cw,
                    "t_cw": item["t_cw"],
                    "blender_eye": list(eye_b),
                    "blender_mat4": mat4,
                    "blender_intrinsics": bl_intr,
                    "sharpness": round(sharpness, 2),
                    "priority": round(priority, 4),
                    "norm_t": round(norm_t, 4),
                    "gain_rgb": [round(float(x), 4) for x in gain],
                    "watermark_box": [0.85, 0.87, 1.0, 1.0],
                }
            )
    return selected


def unproject_photo_pixels(
    layout: dict[str, Any],
    pixels_uv: np.ndarray | list[list[float]],
    height_y: float = 0.0,
) -> np.ndarray:
    """Intersect reference camera rays at pixel coordinates with horizontal plane Y = height_y."""
    frame = np.asarray(layout["frame"], dtype=np.float64)
    k = np.asarray(layout["intrinsics"], dtype=np.float64)
    eye = np.asarray(layout["camera"]["position"], dtype=np.float64)
    w, h = layout["imageSize"]
    dw, dh = layout["depthSize"]
    pix = np.asarray(pixels_uv, dtype=np.float64)
    uv = pix * np.array([dw / w, dh / h])
    rays = np.stack(
        [(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(uv))],
        axis=-1,
    ) @ frame.T
    t = (float(height_y) - eye[1]) / rays[:, 1]
    return eye + rays * t[:, None]


def sample_polygon_rgb(rgb: np.ndarray, poly_uv: np.ndarray, erode_scale: float = 0.78) -> np.ndarray:
    """Sample median RGB color inside a trimmed image polygon."""
    h, w = rgb.shape[:2]
    center = poly_uv.mean(0)
    trimmed = np.clip(center + (poly_uv - center) * erode_scale, [0, 0], [w - 1, h - 1]).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [trimmed], 255)
    pixels = rgb[mask > 0]
    if len(pixels) == 0:
        return np.array([150.0, 145.0, 136.0], dtype=np.float32)
    return np.median(pixels, axis=0).astype(np.float32)


def sample_wall_rgb(rgb: np.ndarray, roof_uv: np.ndarray) -> np.ndarray:
    """Sample representative wall plaster tone slightly below the frontmost eave edge."""
    h, w = rgb.shape[:2]
    bot_idx = int(np.argmax(roof_uv[:, 1]))
    nxt = (bot_idx + 1) % len(roof_uv)
    prv = (bot_idx - 1) % len(roof_uv)
    other = nxt if roof_uv[nxt, 1] > roof_uv[prv, 1] else prv
    mid = 0.5 * (roof_uv[bot_idx] + roof_uv[other])
    patch_x0 = int(np.clip(mid[0] - 12, 0, w - 1))
    patch_x1 = int(np.clip(mid[0] + 12, patch_x0 + 1, w))
    patch_y0 = int(np.clip(mid[1] + 6, 0, h - 1))
    patch_y1 = int(np.clip(mid[1] + 24, patch_y0 + 1, h))
    patch = rgb[patch_y0:patch_y1, patch_x0:patch_x1]
    if patch.size == 0:
        return np.array([178.0, 173.0, 162.0], dtype=np.float32)
    lum = patch.mean(axis=2)
    bright = patch[lum >= np.percentile(lum, 55)]
    if len(bright) == 0:
        return np.array([178.0, 173.0, 162.0], dtype=np.float32)
    return np.median(bright, axis=0).astype(np.float32)


def classify_roof_archetype(legacy_type: str | None, roof_rgb: np.ndarray, bid: str) -> tuple[str, str]:
    """Classify building architectural archetype and roof material swatch from legacy type + real RGB."""
    if bid in BUILDING_VISUAL_OVERRIDES:
        over_type = BUILDING_VISUAL_OVERRIDES[bid].get("roof_type")
        if over_type == "flat_parapet":
            return "flat_parapet", "concrete_roof"
        if over_type == "gable_red_tile":
            return "gable_red_tile", "terracotta_tile"

    r, g, b = float(roof_rgb[0]), float(roof_rgb[1]), float(roof_rgb[2])
    if legacy_type == "blue" or (b - r > 18.0 and b > g + 6.0):
        return "blue_shed", "corrugated_blue"
    if legacy_type == "metal":
        return "metal_shed", "corrugated_silver"
    if legacy_type == "gable":
        if (r - b > 20.0) and (r > g + 7.0):
            return "gable_red_tile", "terracotta_tile"
        return "gable_silver_metal", "corrugated_silver"
    if (r - b > 28.0) and (r > g + 10.0):
        return "gable_red_tile", "terracotta_tile"
    return "flat_parapet", "concrete_roof"


# ---------------------------------------------------------------------------
# 2D Computational Geometry Helpers for Dense Non-Colliding Village Layout
# ---------------------------------------------------------------------------


def _dist_point_segment(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> tuple[float, float, float]:
    vx, vz = bx - ax, bz - az
    denom = vx * vx + vz * vz
    t = 0.0 if denom < 1e-9 else max(0.0, min(1.0, ((px - ax) * vx + (pz - az) * vz) / denom))
    nx, nz = ax + t * vx, az + t * vz
    return math.hypot(px - nx, pz - nz), nx, nz


def nearest_point_on_polyline(px: float, pz: float, pts: list[list[float]]) -> tuple[float, float, float]:
    best = (float("inf"), pts[0][0], pts[0][1])
    for a, b in zip(pts, pts[1:]):
        cand = _dist_point_segment(px, pz, a[0], a[1], b[0], b[1])
        if cand[0] < best[0]:
            best = cand
    return best


def _segments_intersect_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def ccw(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float((p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

    d1 = ccw(a1, a2, b1) * ccw(a1, a2, b2)
    d2 = ccw(b1, b2, a1) * ccw(b1, b2, a2)
    return d1 <= 1e-7 and d2 <= 1e-7


def _poly_clearance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Minimum distance between two 2D polygons; returns <= 0 if intersecting or overlapping."""
    ca = pts_a.mean(0)
    cb = pts_b.mean(0)
    if np.linalg.norm(ca - cb) > 48.0:
        return 99.0
    poly_a = pts_a.astype(np.float32)
    poly_b = pts_b.astype(np.float32)
    for ia in range(len(pts_a)):
        a1, a2 = pts_a[ia], pts_a[(ia + 1) % len(pts_a)]
        for ib in range(len(pts_b)):
            b1, b2 = pts_b[ib], pts_b[(ib + 1) % len(pts_b)]
            if _segments_intersect_2d(a1, a2, b1, b2):
                return 0.0
    min_d = 999.0
    for p in pts_a:
        d = -cv2.pointPolygonTest(poly_b, (float(p[0]), float(p[1])), True)
        if d < min_d:
            min_d = d
    for p in pts_b:
        d = -cv2.pointPolygonTest(poly_a, (float(p[0]), float(p[1])), True)
        if d < min_d:
            min_d = d
    # Also test edge midpoints so thin crossing edges cannot tunnel between corners
    for i in range(len(pts_a)):
        m = 0.5 * (pts_a[i] + pts_a[(i + 1) % len(pts_a)])
        d = -cv2.pointPolygonTest(poly_b, (float(m[0]), float(m[1])), True)
        if d < min_d:
            min_d = d
    for i in range(len(pts_b)):
        m = 0.5 * (pts_b[i] + pts_b[(i + 1) % len(pts_b)])
        d = -cv2.pointPolygonTest(poly_a, (float(m[0]), float(m[1])), True)
        if d < min_d:
            min_d = d
    return float(min_d)


def _edge_neighbor_gap(p0: np.ndarray, p1: np.ndarray, own_idx: int, all_polys: list[np.ndarray]) -> float:
    mid = 0.5 * (p0 + p1)
    best = 99.0
    for j, poly in enumerate(all_polys):
        if j == own_idx:
            continue
        if np.linalg.norm(poly.mean(0) - mid) > 35.0:
            continue
        d = -cv2.pointPolygonTest(poly.astype(np.float32), (float(mid[0]), float(mid[1])), True)
        if d < best:
            best = d
    return float(best)


def build_courtyard_network(
    buildings: list[dict[str, Any]],
    all_polys: list[np.ndarray],
    road_poly_xz: list[list[float]],
    centerline_xz: list[list[float]],
    parked_cars: list[dict[str, Any]],
    road_apron_poly_xz: list[list[float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Link neighboring compounds with continuous brick courtyard walls and northern rural portal gates."""
    centers = [p.mean(0) for p in all_polys]
    car_pts = [np.asarray(c["xz"], dtype=np.float64) for c in parked_cars]
    walls_out: list[dict[str, Any]] = []
    gates_out: list[dict[str, Any]] = []
    used_pairs: set[tuple[int, int]] = set()

    def road_side_sign(pt_xz: np.ndarray) -> float:
        return local_road_frame(pt_xz, centerline_xz)[1]

    def segment_is_clear(pa: np.ndarray, pb: np.ndarray, skip_i: int, skip_j: int = -1) -> bool:
        for frac in (0.12, 0.25, 0.4, 0.5, 0.6, 0.75, 0.88):
            pt = pa * (1.0 - frac) + pb * frac
            px, pz = float(pt[0]), float(pt[1])
            if point_in_polygon_2d(px, pz, road_poly_xz):
                return False
            if road_apron_poly_xz and point_in_polygon_2d(px, pz, road_apron_poly_xz):
                return False
            _, sd_r, _, _, _ = local_road_frame(pt, centerline_xz)
            if abs(sd_r) < 7.95:
                return False
            for k, poly in enumerate(all_polys):
                if k in (skip_i, skip_j):
                    continue
                if -cv2.pointPolygonTest(poly.astype(np.float32), (px, pz), True) < 0.32:
                    return False
            for cp in car_pts:
                if float(np.linalg.norm(cp - pt)) < 2.2:
                    return False
        for existing in walls_out:
            ea = np.asarray(existing["a_xz"], dtype=np.float64)
            eb = np.asarray(existing["b_xz"], dtype=np.float64)
            if float(np.linalg.norm(0.5 * (pa + pb) - 0.5 * (ea + eb))) < 1.8:
                return False
        return True

    # 1. Link nearest same-side building pairs within 1.8m..11.2m
    degree: dict[int, int] = {}
    for i in range(len(buildings)):
        ci = centers[i]
        si = road_side_sign(ci)
        cands = []
        for j in range(i + 1, len(buildings)):
            cj = centers[j]
            if float(np.linalg.norm(ci - cj)) > 26.0:
                continue
            if si * road_side_sign(cj) <= 0.0:
                continue
            best_pair = None
            best_d = 999.0
            for va in all_polys[i]:
                for vb in all_polys[j]:
                    d = float(np.linalg.norm(va - vb))
                    if 1.8 <= d <= 11.2 and d < best_d:
                        best_d = d
                        best_pair = (va, vb)
            if best_pair is not None:
                cands.append((best_d, j, best_pair[0], best_pair[1]))
        cands.sort(key=lambda x: x[0])
        for dist_w, j, va, vb in cands[:3]:
            if degree.get(i, 0) >= 3 or degree.get(j, 0) >= 3:
                continue
            key = (min(i, j), max(i, j))
            if key in used_pairs:
                continue
            if not segment_is_clear(va, vb, i, j):
                continue
            used_pairs.add(key)
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
            mid = 0.5 * (va + vb)
            d_road_mid = abs(local_road_frame(mid, centerline_xz)[1])
            swatch = "red_brick" if (i + j) % 3 != 1 else "plaster_grey"

            # Street-facing compound spans >= 4.4m get a rural double iron gate portal!
            if dist_w >= 4.4 and d_road_mid <= 28.5 and len(gates_out) < 38:
                u_seg = (vb - va) / max(dist_w, 1e-6)
                gate_w = 2.4
                g0 = mid - u_seg * (gate_w * 0.5)
                g1 = mid + u_seg * (gate_w * 0.5)
                if float(np.linalg.norm(g0 - va)) > 0.65:
                    walls_out.append({"a_xz": va.tolist(), "b_xz": g0.tolist(), "height_m": 2.32, "swatch": swatch})
                if float(np.linalg.norm(vb - g1)) > 0.65:
                    walls_out.append({"a_xz": g1.tolist(), "b_xz": vb.tolist(), "height_m": 2.32, "swatch": swatch})
                gates_out.append(
                    {
                        "center_xz": [round(float(mid[0]), 3), round(float(mid[1]), 3)],
                        "yaw_rad": round(math.atan2(-u_seg[1], u_seg[0]), 4),
                        "width_m": gate_w,
                        "height_m": 2.75,
                        "roof_swatch": "terracotta_tile" if (i + j) % 2 == 0 else "concrete_roof",
                    }
                )
            else:
                walls_out.append({"a_xz": va.tolist(), "b_xz": vb.tolist(), "height_m": 2.30, "swatch": swatch})

    # 2. Enclose open yard flanks on buildings that still have low courtyard degree
    for i, b in enumerate(buildings):
        if len(walls_out) >= 340:
            break
        poly = all_polys[i]
        c_xz = centers[i]
        _, _, _, _, road_near = local_road_frame(c_xz, centerline_xz)
        away = c_xz - road_near
        away /= max(float(np.linalg.norm(away)), 1e-6)
        walls_info = b.get("walls") or []
        if not walls_info:
            continue
        edge_idx = max(range(len(poly)), key=lambda ki: float(np.dot(walls_info[ki]["outward_xz"], away)))
        a = poly[edge_idx]
        d_pt = poly[(edge_idx + 1) % len(poly)]
        out_v = np.asarray(walls_info[edge_idx]["outward_xz"], dtype=np.float64)
        c_pt = a + out_v * 3.4
        e_pt = d_pt + out_v * 3.4
        if segment_is_clear(a, c_pt, i) and segment_is_clear(c_pt, e_pt, i) and segment_is_clear(e_pt, d_pt, i):
            sw = "red_brick" if i % 2 == 0 else "plaster_warm"
            walls_out.append({"a_xz": a.tolist(), "b_xz": c_pt.tolist(), "height_m": 2.20, "swatch": sw})
            walls_out.append({"a_xz": c_pt.tolist(), "b_xz": e_pt.tolist(), "height_m": 2.20, "swatch": sw})
            walls_out.append({"a_xz": e_pt.tolist(), "b_xz": d_pt.tolist(), "height_m": 2.20, "swatch": sw})

    return walls_out, gates_out


def build_photographic_ground(
    out_dir: Path,
    layout: dict[str, Any],
    ref_clean_rgb: np.ndarray,
    buildings: list[dict[str, Any]],
    courtyard_walls: list[dict[str, Any]],
    parked_cars: list[dict[str, Any]],
    road_world: np.ndarray,
    centerline_world: np.ndarray,
    swatches: dict[str, dict[str, str]],
    road_left_edge_xz: np.ndarray | list[list[float]] | None = None,
    road_right_edge_xz: np.ndarray | list[list[float]] | None = None,
    bev_ortho_info: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate 3072x3072 crisp tiled concrete/farmland ground atlas fused with DA3 BEV orthophoto & soft contact AO."""
    size = 3072
    bx0, bx1, bz0, bz1 = layout["groundBounds"]
    gx = np.linspace(bx0, bx1, size, dtype=np.float64)
    gz = np.linspace(bz0, bz1, size, dtype=np.float64)
    xx, zz = np.meshgrid(gx, gz)
    world_pts = np.stack([xx.ravel(), np.zeros(xx.size, dtype=np.float64), zz.ravel()], axis=-1)

    rng = np.random.default_rng(13)
    broad = cv2.resize(rng.normal(0.0, 1.0, (24, 24)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    med = cv2.resize(rng.normal(0.0, 1.0, (64, 64)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    fine = rng.normal(0.0, 2.2, (size, size)).astype(np.float32)

    # Smooth organic crop-field patchwork + directional plow furrows (zero bumpy camouflage blobs)
    rot_ang = 0.36
    rx_base = xx * math.cos(rot_ang) + zz * math.sin(rot_ang)
    rz_base = -xx * math.sin(rot_ang) + zz * math.cos(rot_ang)
    rx = rx_base + np.sin(rz_base * 0.012) * 3.5 + broad * 0.8
    rz = rz_base + np.cos(rx_base * 0.010) * 2.5 + broad * 0.6
    band_z = np.floor(rz / 34.0).astype(np.int32)
    strip_phase = (np.abs(band_z) * 29.0) % 47.0
    strip_w = 52.0 + ((np.abs(band_z) * 19) % 4) * 16.0
    plot_ix = np.floor((rx + strip_phase) / strip_w).astype(np.int32)
    plot_hash = (plot_ix * 73 + band_z * 139) % 7
    plot_pal = np.array(
        [
            [102.0, 112.0, 64.0],
            [88.0, 98.0, 56.0],
            [128.0, 120.0, 78.0],
            [114.0, 118.0, 70.0],
            [138.0, 128.0, 86.0],
            [80.0, 92.0, 52.0],
            [118.0, 114.0, 72.0],
        ],
        dtype=np.float32,
    )
    plot_base = cv2.GaussianBlur(plot_pal[plot_hash], (9, 9), 2.2)
    furrow = np.sin(rz * 2.2 + broad * 0.9) * 3.0 + np.sin(rz * 6.2 + med * 0.4) * 1.2
    ground = np.clip(
        plot_base + broad[:, :, None] * 6.0 + med[:, :, None] * 2.5 + fine[:, :, None] + furrow[:, :, None],
        0,
        255,
    ).astype(np.float32)

    # Crisp field-margin bunds & rural footpaths along crop strips
    bund_z = np.abs(np.mod(rz, 34.0) - 17.0) > 16.3
    bund_x = np.abs(np.mod(rx + strip_phase, strip_w) - strip_w * 0.5) > (strip_w * 0.5 - 0.65)
    bund = bund_z | bund_x
    bund_soft = cv2.GaussianBlur(bund.astype(np.float32), (7, 7), 1.8)
    bund_rgb = np.array([106.0, 98.0, 78.0], dtype=np.float32) + fine[:, :, None] * 0.35
    ground = ground * (1.0 - 0.48 * bund_soft[:, :, None]) + bund_rgb * (0.48 * bund_soft[:, :, None])

    # Project reference open farmland band (v < 165) without building projections
    h_img, w_img = ref_clean_rgb.shape[:2]
    frame = np.asarray(layout["frame"], dtype=np.float64)
    origin = np.asarray(layout["origin"], dtype=np.float64)
    k = np.asarray(layout["intrinsics"], dtype=np.float64)
    scale = float(layout["scale"])
    dw, dh = layout["depthSize"]
    cv_pts = (world_pts / scale) @ frame + origin
    uv = np.stack(
        [
            (cv_pts[:, 0] / np.maximum(cv_pts[:, 2], 1e-4) * k[0, 0] + k[0, 2]) * (w_img / dw),
            (cv_pts[:, 1] / np.maximum(cv_pts[:, 2], 1e-4) * k[1, 1] + k[1, 2]) * (h_img / dh),
        ],
        axis=-1,
    ).reshape(size, size, 2).astype(np.float32)
    observed = cv2.remap(ref_clean_rgb.astype(np.float32), uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    building_uv_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    for b in buildings:
        if not b.get("is_core", True):
            continue
        p3 = np.asarray(b["points_3d"], dtype=np.float64)
        p_base = p3.copy()
        p_base[:, 1] = 0.0
        hull_cv = (np.vstack([p3, p_base]) / scale) @ frame + origin
        hull_uv = np.stack(
            [
                (hull_cv[:, 0] / np.maximum(hull_cv[:, 2], 1e-4) * k[0, 0] + k[0, 2]) * (w_img / dw),
                (hull_cv[:, 1] / np.maximum(hull_cv[:, 2], 1e-4) * k[1, 1] + k[1, 2]) * (h_img / dh),
            ],
            axis=-1,
        )
        hull_i = cv2.convexHull(np.clip(np.round(hull_uv), [0, 0], [w_img - 1, h_img - 1]).astype(np.int32))
        cv2.fillConvexPoly(building_uv_mask, hull_i, 255)
    building_uv_mask = cv2.dilate(building_uv_mask, np.ones((15, 15), np.uint8))

    uv_int_x = np.clip(np.round(uv[:, :, 0]).astype(np.int32), 0, w_img - 1)
    uv_int_y = np.clip(np.round(uv[:, :, 1]).astype(np.int32), 0, h_img - 1)
    hits_building = building_uv_mask[uv_int_y, uv_int_x] > 0
    fields_mask = (cv_pts[:, 2].reshape(size, size) > 8.0) & (uv[:, :, 0] >= 2) & (uv[:, :, 0] < w_img - 2) & (uv[:, :, 1] >= 2) & (uv[:, :, 1] < h_img - 2)
    open_farmland = fields_mask & (~hits_building) & (uv[:, :, 1] < 165.0)
    frustum_margin = np.minimum(np.minimum(uv[:, :, 0], w_img - 1 - uv[:, :, 0]), np.minimum(uv[:, :, 1], 165.0 - uv[:, :, 1]))
    farm_alpha = np.where(open_farmland, np.clip(frustum_margin / 24.0, 0.0, 1.0), 0.0).astype(np.float32)
    ground = ground * (1.0 - farm_alpha[:, :, None]) + observed * farm_alpha[:, :, None]

    def world_to_grid(pts_xz: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_xz, dtype=np.float64)
        return np.stack([(p[:, 0] - bx0) / (bx1 - bx0) * (size - 1), (p[:, 1] - bz0) / (bz1 - bz0) * (size - 1)], axis=-1).astype(np.int32)

    # Dual non-aligned scale sampling (9.0m + rotated 23.0m) removes square grid tiling & ochre noise
    apron_src = np.asarray(Image.open(swatches["concrete_apron"]["albedo"]).convert("RGB"), dtype=np.float32)
    worn_src = np.asarray(Image.open(swatches["concrete_roof"]["albedo"]).convert("RGB"), dtype=np.float32)
    ah, aw = apron_src.shape[:2]
    tiled_apron_1 = cv2.remap(
        apron_src,
        np.mod((xx - bx0) / 9.0 * aw, aw).astype(np.float32),
        np.mod((zz - bz0) / 9.0 * ah, ah).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    ru2 = (xx * 0.80 + zz * 0.60) / 23.0
    rv2 = (-xx * 0.60 + zz * 0.80) / 23.0
    tiled_apron_2 = cv2.remap(
        apron_src,
        np.mod(ru2 * aw, aw).astype(np.float32),
        np.mod(rv2 * ah, ah).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    tiled_apron = tiled_apron_1 * 0.60 + tiled_apron_2 * 0.40
    wh, ww = worn_src.shape[:2]
    tiled_worn = cv2.remap(
        worn_src,
        np.mod(ru2 * ww, ww).astype(np.float32),
        np.mod(rv2 * wh, wh).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    mix_w = np.clip(0.58 + broad * 0.14, 0.30, 0.86)
    tiled_mix = tiled_apron * mix_w[:, :, None] + tiled_worn * (1.0 - mix_w[:, :, None])
    tiled_earth = np.clip(tiled_mix * np.array([0.96, 0.95, 0.90], dtype=np.float32) + fine[:, :, None] * 0.35, 0, 255)

    all_b_pts = world_to_grid(np.vstack([np.asarray(b["footprint_xz"], dtype=np.float64) for b in buildings]))
    core_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(core_mask, cv2.convexHull(all_b_pts), 255)
    core_mask = cv2.dilate(core_mask, np.ones((33, 33), np.uint8))
    core_blur = cv2.GaussianBlur(core_mask.astype(np.float32) / 255.0, (55, 55), 14.0)
    ground = ground * (1.0 - core_blur[:, :, None] * 0.94) + tiled_earth * (core_blur[:, :, None] * 0.94)

    # Feathered concrete yard slabs + contact AO halos around every building & wall
    yard_tint_u8 = np.zeros((size, size), dtype=np.uint8)
    contact_occ = np.zeros((size, size), dtype=np.uint8)
    for idx, b in enumerate(buildings):
        pts_xz = np.asarray(b["footprint_xz"], dtype=np.float64)
        c = pts_xz.mean(0)
        pad_poly = c + (pts_xz - c) * 1.18
        tint_val = int(round(92 + ((idx * 17) % 15) * 1.1))
        cv2.fillConvexPoly(yard_tint_u8, world_to_grid(pad_poly), tint_val)
        cv2.fillConvexPoly(contact_occ, world_to_grid(pts_xz), 255)

    yard_alpha = cv2.GaussianBlur((yard_tint_u8 > 0).astype(np.float32), (13, 13), 3.5)
    yard_tint_dil = cv2.dilate(yard_tint_u8, np.ones((7, 7), np.uint8))
    yard_tint = np.where(yard_tint_dil > 0, yard_tint_dil.astype(np.float32) / 100.0, 1.0)
    yard_rgb = np.clip(tiled_apron * yard_tint[:, :, None] + med[:, :, None] * 2.0, 0, 255)
    ground = ground * (1.0 - yard_alpha[:, :, None]) + yard_rgb * yard_alpha[:, :, None]

    # Build road corridor mask so BEV fusion avoids painting over carriageway strip
    if road_left_edge_xz is not None and road_right_edge_xz is not None:
        left_pts = np.asarray(road_left_edge_xz, dtype=np.float64)
        right_pts = np.asarray(road_right_edge_xz, dtype=np.float64)
    else:
        left_pts = road_world[:5, [0, 2]].copy()
        right_pts = road_world[5:, [0, 2]][::-1].copy()
    d_sw = (left_pts[0] - left_pts[1]) / max(float(np.linalg.norm(left_pts[0] - left_pts[1])), 1e-6)
    d_ne = (left_pts[-1] - left_pts[-2]) / max(float(np.linalg.norm(left_pts[-1] - left_pts[-2])), 1e-6)
    ext_left = np.vstack([left_pts[0] + d_sw * 180.0, left_pts, left_pts[-1] + d_ne * 180.0])
    ext_right = np.vstack([right_pts[0] + d_sw * 180.0, right_pts, right_pts[-1] + d_ne * 180.0])
    road_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(road_mask, [world_to_grid(np.vstack([ext_left, ext_right[::-1]]))], 255)

    # Fuse real DA3 BEV orthophoto on open village alleys (strict shadow & vegetation rejection)
    if bev_ortho_info is not None:
        ortho_img = bev_ortho_info["ortho"].astype(np.float32)
        ndsm_img = bev_ortho_info["ndsm"].astype(np.float32)
        obs_img = bev_ortho_info["obs_views"].astype(np.float32)
        bx_min, bx_max, bz_min, bz_max = bev_ortho_info["bounds_xz"]
        bh, bw = ortho_img.shape[:2]
        bu = ((xx - bx_min) / max(bx_max - bx_min, 1e-4) * bw - 0.5).astype(np.float32)
        bv = ((zz - bz_min) / max(bz_max - bz_min, 1e-4) * bh - 0.5).astype(np.float32)
        in_bev = (bu >= 1.0) & (bu <= bw - 2.0) & (bv >= 1.0) & (bv <= bh - 2.0)
        ortho_s = cv2.remap(ortho_img, bu, bv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ndsm_s = cv2.remap(ndsm_img, bu, bv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=99.0)
        obs_s = cv2.remap(obs_img, bu, bv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        luma_b = ortho_s.mean(axis=2)
        green_ex = 2.0 * ortho_s[:, :, 1] - ortho_s[:, :, 0] - ortho_s[:, :, 2]
        valid_g = (
            in_bev
            & (obs_s >= 3.0)
            & (ndsm_s > -0.30)
            & (ndsm_s < 0.55)
            & (contact_occ == 0)
            & (road_mask == 0)
            & (luma_b > 72.0)
            & (green_ex < 14.0)
        )
        if valid_g.any():
            earth_m = np.maximum(ground[valid_g].mean(axis=0), 8.0)
            ortho_m = np.maximum(ortho_s[valid_g].mean(axis=0), 8.0)
            gain = np.clip(earth_m / ortho_m, 0.80, 1.25)
            ortho_harm = np.clip(ortho_s * gain[None, None, :], 14.0, 245.0)
            alpha_g = cv2.GaussianBlur(valid_g.astype(np.float32), (19, 19), 5.0) * 0.36
            ground = ground * (1.0 - alpha_g[:, :, None]) + ortho_harm * alpha_g[:, :, None]

    for w in courtyard_walls:
        seg = world_to_grid(np.array([w["a_xz"], w["b_xz"]]))
        cv2.line(contact_occ, tuple(seg[0]), tuple(seg[1]), 220, 5)

    for car in parked_cars:
        cg = world_to_grid(np.array([car["xz"]]))[0]
        cv2.circle(contact_occ, (int(cg[0]), int(cg[1])), 11, 200, -1)

    ao_blur = cv2.GaussianBlur(contact_occ.astype(np.float32) / 255.0, (23, 23), 5.5)
    ground_ao = 1.0 - 0.46 * ao_blur
    ground *= ground_ao[:, :, None]

    # Extend asphalt under-layer beneath elevated carriageway mesh so far LOD matches
    asphalt = np.clip(np.array([124.0, 122.0, 118.0], dtype=np.float32) + med[:, :, None] * 3.5 + fine[:, :, None] * 0.5, 0, 255)
    ground[road_mask > 0] = asphalt[road_mask > 0]

    rough = np.where(road_mask > 0, 0.80 + med * 0.04, 0.92 + med * 0.04)
    rough_small = cv2.resize((rough * 255.0).astype(np.uint8), (768, 768), interpolation=cv2.INTER_AREA)
    rough_q = ((cv2.GaussianBlur(rough_small, (3, 3), 0.7) // 8) * 8).astype(np.uint8)
    ground_path = out_dir / "farmland_ground_albedo.jpg"
    rough_path = out_dir / "farmland_ground_roughness.jpg"
    Image.fromarray(np.clip(ground, 0, 255).astype(np.uint8)).save(ground_path, quality=86)
    Image.fromarray(np.clip(rough_q, 60, 248).astype(np.uint8)).save(rough_path, quality=82)
    return {"albedo": str(ground_path), "roughness": str(rough_path)}


def prepare_evidence(
    *,
    colmap_dir: Path,
    images_dir: Path,
    out_dir: Path,
    extra_colmap_dir: Path | None = None,
    extra_images_dir: Path | None = None,
    bev_dsm_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate clean reference image, regularized dense village_spec (180+ buildings), and CC0 PBR swatches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    swatches_dir = out_dir / "swatches"
    manifest_path = swatches_dir / "pbr_manifest.json"
    pbr_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        pbr_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest_path.is_file() or int(pbr_manifest.get("version", 1)) < 3:
        pbr_manifest = fetch_and_build_pbr_assets(out_dir)
    swatches: dict[str, Any] = dict(pbr_manifest["swatches"])
    foliage_swatches: dict[str, Any] = dict(pbr_manifest["foliage"])

    layout = json.loads((ROOT / "assets/village_studio/layout.json").read_text(encoding="utf-8"))
    ref_src = ROOT / "assets/scene_sources/village_road_near.png"
    with Image.open(ref_src) as im:
        ref_clean_img = clean_watermark(im.convert("RGB"), box=(0.85, 0.87, 1.0, 1.0))
    ref_clean_path = out_dir / "ref_clean.png"
    ref_clean_img.save(ref_clean_path)
    ref_rgb = np.asarray(ref_clean_img, dtype=np.float32)
    ref_mean_rgb = ref_rgb.mean(axis=(0, 1))

    cams, images, pts_xyz, pts_rgb = parse_colmap_text(colmap_dir)
    if len(pts_xyz) > 15000:
        idx = np.linspace(0, len(pts_xyz) - 1, 15000, dtype=int)
        guide_xyz, guide_rgb = pts_xyz[idx], pts_rgb[idx]
    else:
        guide_xyz, guide_rgb = pts_xyz, pts_rgb
    np.savez_compressed(out_dir / "colmap_guide.npz", xyz=guide_xyz, rgb=guide_rgb)

    selected_cameras = select_calibrated_frames(
        cams, images, images_dir, ref_clean_path, ref_mean_rgb, per_clip_count=12
    )
    if extra_colmap_dir is not None and extra_images_dir is not None and (extra_colmap_dir / "images.txt").is_file():
        ecams, eimgs, _, _ = parse_colmap_text(extra_colmap_dir)
        selected_cameras.extend(
            select_calibrated_frames(
                ecams, eimgs, extra_images_dir, ref_clean_path, ref_mean_rgb, per_clip_count=16, tag_prefix="extra_"
            )
        )

    centerline_uv = [[0, 765], [400, 604], [800, 448], [1200, 307], [1536, 229]]
    centerline_world = unproject_photo_pixels(layout, centerline_uv, height_y=0.0)
    centerline_xz = [[round(float(p[0]), 3), round(float(p[2]), 3)] for p in centerline_world]

    # Constant-width carriageway (8.8m) & concrete aprons (3.2m each side => 7.6m half-corridor)
    road_info = build_regularized_road(centerline_xz, carriageway_half_width=4.4, apron_width=3.2)
    road_poly_xz = road_info["road_polygon_xz"]
    road_apron_poly_xz = road_info["road_apron_polygon_xz"]
    road_left_edge = road_info["road_left_edge_xz"]
    road_right_edge = road_info["road_right_edge_xz"]
    road_world = np.column_stack([
        np.asarray(road_left_edge + road_right_edge[::-1], dtype=np.float64)[:, 0],
        np.zeros(len(road_left_edge) + len(road_right_edge)),
        np.asarray(road_left_edge + road_right_edge[::-1], dtype=np.float64)[:, 1],
    ])

    # Load Route B BEV DSM data if present
    resolved_bev_dir = bev_dsm_dir if bev_dsm_dir is not None else (out_dir / "bev_dsm")
    bev_core_cal: dict[str, Any] = {}
    bev_extracted: list[dict[str, Any]] = []
    bev_ortho_info: dict[str, Any] | None = None
    bev_json_path = resolved_bev_dir / "bev_buildings.json"
    if bev_json_path.is_file():
        bev_payload = json.loads(bev_json_path.read_text(encoding="utf-8"))
        bev_core_cal = bev_payload.get("core_calibrations", {})
        bev_extracted = bev_payload.get("extracted_buildings", [])
        ortho_p = Path(bev_payload.get("ortho_path", ""))
        ndsm_p = Path(bev_payload.get("ndsm_path", ""))
        obs_p = Path(bev_payload.get("obs_views_path", ""))
        if ortho_p.is_file() and ndsm_p.is_file() and obs_p.is_file():
            bev_ortho_info = {
                "ortho": np.asarray(Image.open(ortho_p).convert("RGB"), dtype=np.uint8),
                "ndsm": np.load(ndsm_p).astype(np.float32),
                "obs_views": np.load(obs_p),
                "bounds_xz": tuple(bev_payload["bounds_xz"]),
            }

    # Core 64 silhouette-locked buildings with orthogonal street-grid regularization + BEV calibration
    core_buildings: list[dict[str, Any]] = []
    for idx, b in enumerate(layout["buildings"]):
        bid = b["id"]
        roof_uv = np.asarray(b["roof"], dtype=np.float64)
        h_prior = float(b["h"])
        override = BUILDING_VISUAL_OVERRIDES.get(bid, {})
        cal = bev_core_cal.get(bid, {})
        if "height_m" in override:
            h_final = float(override["height_m"])
        elif cal.get("accepted"):
            h_final = float(cal["h_calibrated_m"])
        else:
            h_final = h_prior
        pts3d = unproject_photo_pixels(layout, roof_uv, height_y=h_final) if abs(h_final - h_prior) > 1e-4 else np.asarray(b["points"], dtype=np.float64)

        raw_xz = ensure_ccw_blender_xz(np.stack([pts3d[:, 0], pts3d[:, 2]], axis=-1))
        poly_xz = regularize_footprint_to_grid(raw_xz, centerline_xz)
        pts3d = np.column_stack([poly_xz[:, 0], np.full(len(poly_xz), h_final), poly_xz[:, 1]])

        roof_rgb = sample_polygon_rgb(ref_rgb, roof_uv)
        wall_rgb = sample_wall_rgb(ref_rgb, roof_uv)
        archetype, roof_swatch = classify_roof_archetype(b.get("type"), roof_rgb, bid)
        ridge_rise = float(override.get("ridge_rise", 1.25 if archetype.startswith("gable") else 0.0))

        wall_warmth = float(wall_rgb[0] - wall_rgb[2])
        default_wall = "plaster_white" if (wall_rgb.mean() > 135 and wall_warmth < 24) else "plaster_warm"
        wall_swatch = str(override.get("wall_swatch", default_wall))
        side_choices = (
            ("plaster_white", "plaster_grey", "plaster_warm")
            if (wall_rgb.mean() > 125.0 and wall_warmth < 28.0)
            else ("plaster_grey", "red_brick", "plaster_warm")
        )
        side_wall_swatch = str(override.get("side_wall_swatch", side_choices[idx % 3]))

        core_buildings.append(
            {
                "id": bid,
                "index": idx,
                "is_core": True,
                "source": "core_layout",
                "roof_uv": roof_uv.tolist(),
                "points_3d": pts3d.tolist(),
                "footprint_xz": poly_xz.tolist(),
                "height_m": round(h_final, 3),
                "height_prior_m": round(h_prior, 3),
                "stories": 2 if h_final >= 5.1 else 1,
                "archetype": archetype,
                "roof_swatch": roof_swatch,
                "wall_swatch": wall_swatch,
                "side_wall_swatch": side_wall_swatch,
                "front_cornice_brick": bool(archetype == "flat_parapet" and idx % 4 != 2),
                "ridge_rise": round(ridge_rise, 3),
                "observed_roof_rgb": [round(float(x), 1) for x in roof_rgb],
                "observed_wall_rgb": [round(float(x), 1) for x in wall_rgb],
                "has_parapet": archetype == "flat_parapet",
                "has_eave_overhang": archetype in ("gable_red_tile", "gable_silver_metal"),
                "solar_heater": bool(archetype == "flat_parapet" and (idx in (3, 5, 7, 12, 23, 25, 29) or idx % 9 == 4)),
            }
        )

    # Solve core footprints first (clearing house_33/37 off road+apron and separating core overlaps)
    core_buildings = solve_building_layout(
        core_buildings,
        centerline_xz,
        road_poly_xz,
        road_apron_poly_xz,
        apron_half_width=road_info["corridor_half_width"],
        min_building_gap=0.45,
        drop_unresolvable_synthetic=False,
    )

    # Expand with BEV-DSM extracted rooftops + continuous roadside shopfronts + dense northern courtyard fabric
    cam_eye_xz = np.array([layout["camera"]["position"][0], layout["camera"]["position"][2]], dtype=np.float64)
    infill_buildings = generate_northern_village_fabric(
        core_buildings,
        bev_extracted,
        centerline_xz,
        road_poly_xz,
        road_apron_poly_xz,
        cam_eye_xz,
        bev_ortho_info=bev_ortho_info,
        corridor_half_width=road_info["corridor_half_width"],
    )
    raw_buildings = solve_building_layout(
        core_buildings + infill_buildings,
        centerline_xz,
        road_poly_xz,
        road_apron_poly_xz,
        apron_half_width=road_info["corridor_half_width"],
        min_building_gap=0.45,
        drop_unresolvable_synthetic=True,
    )
    layout_audit = audit_layout_geometry(raw_buildings, centerline_xz, road_poly_xz, road_apron_poly_xz)
    if layout_audit["verdict"] != "pass":
        raise RuntimeError(f"Layout audit gate failed: {layout_audit}")

    # Compute wall roles across ALL buildings (core + BEV + courtyard fabric)
    all_polys = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in raw_buildings]
    for idx, b in enumerate(raw_buildings):
        poly = all_polys[idx]
        center_xz = poly.mean(0)
        _, sd_road, _, _, near_road_pt = local_road_frame(center_xz, centerline_xz)
        road_d = abs(sd_road)
        to_road = near_road_pt - center_xz
        to_road /= max(float(np.linalg.norm(to_road)), 1e-6)
        b["road_distance_m"] = round(road_d, 2)

        walls = []
        n_edges = len(poly)
        for ei in range(n_edges):
            p0 = poly[ei]
            p1 = poly[(ei + 1) % n_edges]
            edge = p1 - p0
            elen = float(np.linalg.norm(edge))
            cand_n = np.array([-edge[1], edge[0]], dtype=np.float64) / max(elen, 1e-6)
            mid = 0.5 * (p0 + p1)
            if float(np.dot(cand_n, mid - center_xz)) < 0:
                cand_n = -cand_n
            gap = _edge_neighbor_gap(p0, p1, idx, all_polys)
            faces_street = float(np.dot(cand_n, to_road)) > 0.42 and road_d < 30.0
            role = "party_wall" if gap < 1.15 else ("shopfront" if faces_street and b["archetype"] in ("flat_parapet", "gable_red_tile", "gable_silver_metal") else "facade")
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

    # Street utility poles
    pole_uvs = [[392, 595], [666, 455], [867, 370], [1091, 304], [1300, 246]]
    poles_world = unproject_photo_pixels(layout, pole_uvs, height_y=0.0)
    poles_xz = [[round(float(p[0]), 3), round(float(p[2]), 3)] for p in poles_world]

    # Observed parked roadside vehicles on concrete aprons (4.8m .. 7.1m from centerline, clear of walls)
    car_uvs = [
        (195, 608, "white", 0),
        (292, 566, "silver", 1),
        (518, 478, "silver", 2),
        (752, 408, "charcoal", 3),
        (802, 391, "white", 4),
        (1208, 272, "white", 5),
        (1284, 249, "silver", 6),
    ]
    cars = []
    for u, v, color_tag, cid in car_uvs:
        p3 = unproject_photo_pixels(layout, [[u, v]], height_y=0.0)[0]
        _, sd_c, u_loc, n_loc, near_c = local_road_frame((float(p3[0]), float(p3[2])), centerline_xz)
        side_c = -1.0 if sd_c <= 0.0 else 1.0
        target_d = min(6.6, max(5.35, abs(sd_c)))
        pt_c = near_c + n_loc * (side_c * target_d)
        cx, cz = float(pt_c[0]), float(pt_c[1])
        yaw_three = math.atan2(-u_loc[1], u_loc[0])
        if any(-cv2.pointPolygonTest(poly.astype(np.float32), (cx, cz), True) < 0.85 for poly in all_polys):
            pt_c = near_c + n_loc * (side_c * 5.5)
            cx, cz = float(pt_c[0]), float(pt_c[1])
        if any(-cv2.pointPolygonTest(poly.astype(np.float32), (cx, cz), True) < 0.80 for poly in all_polys):
            continue
        cars.append({"id": f"parked_car_{cid:02d}", "xz": [round(cx, 3), round(cz, 3)], "yaw_rad": round(yaw_three, 4), "paint": color_tag})

    # Courtyard compound network & rural portal gates
    courtyard_walls, courtyard_gates = build_courtyard_network(
        raw_buildings, all_polys, road_poly_xz, centerline_xz, cars, road_apron_poly_xz=road_apron_poly_xz
    )

    # Rural flatbed electric tricycles on concrete aprons & stacked courtyard brick piles
    cl = np.asarray(centerline_xz, dtype=np.float64)
    r_dir = (cl[-1] - cl[0]) / max(float(np.linalg.norm(cl[-1] - cl[0])), 1e-6)
    r_perp = np.array([-r_dir[1], r_dir[0]], dtype=np.float64)
    tri_cands = [
        cl[0] + r_dir * 22.0 + r_perp * 6.2,
        cl[1] + r_dir * 26.0 - r_perp * 6.2,
        cl[2] + r_dir * 19.0 + r_perp * 6.3,
    ]
    tricycles = []
    for ti, tp in enumerate(tri_cands):
        tx, tz = float(tp[0]), float(tp[1])
        d_tri = abs(local_road_frame((tx, tz), centerline_xz)[1])
        if not point_in_polygon_2d(tx, tz, road_poly_xz) and d_tri >= 5.2:
            if all(-cv2.pointPolygonTest(poly.astype(np.float32), (tx, tz), True) >= 0.95 for poly in all_polys):
                if all(math.hypot(tx - c["xz"][0], tz - c["xz"][1]) > 3.2 for c in cars):
                    tricycles.append({"id": f"tricycle_{ti:02d}", "xz": [round(tx, 3), round(tz, 3)], "yaw_rad": round(math.atan2(-r_dir[1], r_dir[0]), 4)})

    brick_piles = []
    for wi, w in enumerate(courtyard_walls[::12]):
        if len(brick_piles) >= 12:
            break
        mx = 0.5 * (w["a_xz"][0] + w["b_xz"][0])
        mz = 0.5 * (w["a_xz"][1] + w["b_xz"][1])
        dx, dz = w["b_xz"][0] - w["a_xz"][0], w["b_xz"][1] - w["a_xz"][1]
        olen = max(math.hypot(dx, dz), 1e-6)
        px = mx - dz / olen * 0.9
        pz = mz + dx / olen * 0.9
        d_r = abs(local_road_frame((px, pz), centerline_xz)[1])
        if d_r > 8.1 and all(-cv2.pointPolygonTest(poly.astype(np.float32), (px, pz), True) >= 0.9 for poly in all_polys):
            brick_piles.append({"xz": [round(px, 3), round(pz, 3)], "yaw_rad": round(math.atan2(-dz, dx), 4)})

    # Dense 220+ categorized trees (poplar windbreak + broadleaf courtyard/alley, clear of road & aprons) + shrubs
    rng_tree = np.random.default_rng(2026)
    tree_specs: list[dict[str, Any]] = []
    shrub_specs: list[dict[str, Any]] = []
    raw_trees: list[tuple[float, float, float, str]] = []

    observed_tree_uv = [
        (310, 131, "poplar"), (320, 124, "poplar"), (370, 116, "poplar"),
        (983, 156, "poplar"), (999, 144, "poplar"), (1152, 182, "poplar"),
        (949, 157, "poplar"), (923, 157, "poplar"), (970, 186, "poplar"),
        (1108, 207, "broadleaf"), (1454, 287, "broadleaf"), (1470, 272, "broadleaf"),
        (711, 219, "broadleaf"),
    ]
    obs_pts = unproject_photo_pixels(layout, [[u, v] for u, v, _ in observed_tree_uv], 0.0)
    for p, (_, _, species) in zip(obs_pts, observed_tree_uv):
        px, pz = float(p[0]), float(p[2])
        d_r = abs(local_road_frame((px, pz), centerline_xz)[1])
        h_tr = float(rng_tree.uniform(4.2, 5.4)) if d_r < 9.5 else (float(rng_tree.uniform(9.5, 13.5)) if species == "poplar" else float(rng_tree.uniform(5.8, 8.6)))
        raw_trees.append((px, pz, h_tr, species))

    # Multi-row northern poplar shelterbelt matching upper edge of village_road_near.png
    for row_idx, row_z_off in enumerate((0.0, -8.0, -17.0, -27.0)):
        count_x = 64 if row_idx < 2 else 46
        for wx in np.linspace(-138.0, 168.0, count_x):
            tx = float(wx + rng_tree.uniform(-2.4, 2.4))
            tz = float(rng_tree.uniform(-164.0, -142.0) - 0.17 * max(0.0, tx) + row_z_off)
            raw_trees.append((tx, tz, float(rng_tree.uniform(10.5, 15.2)), "poplar"))

    # Courtyard, alley, and eastern/western perimeter shade broadleaf trees
    for _ in range(110):
        tx = float(rng_tree.uniform(-118.0, 118.0))
        tz = float(rng_tree.uniform(-142.0, 52.0))
        raw_trees.append((tx, tz, float(rng_tree.uniform(4.6, 8.2)), "broadleaf"))

    def is_clear_tree_spot(tx: float, tz: float) -> bool:
        if point_in_polygon_2d(tx, tz, road_poly_xz) or point_in_polygon_2d(tx, tz, road_apron_poly_xz):
            return False
        d_road = abs(local_road_frame((tx, tz), centerline_xz)[1])
        if d_road < 8.8:
            return False
        for poly in all_polys:
            if -cv2.pointPolygonTest(poly.astype(np.float32), (tx, tz), True) < 1.85:
                return False
        return True

    for tx, tz, th, species in raw_trees:
        if len(tree_specs) >= 240:
            break
        if not is_clear_tree_spot(tx, tz):
            continue
        min_sp = 2.1 if species == "poplar" else 4.2
        if any(math.hypot(tx - e["xz"][0], tz - e["xz"][1]) < min_sp for e in tree_specs):
            continue
        rad_mul = 0.27 if species == "poplar" else 0.36
        tree_specs.append(
            {
                "id": f"tree_{len(tree_specs):03d}",
                "species": species,
                "xz": [round(tx, 3), round(tz, 3)],
                "height_m": round(th, 2),
                "radius_m": round(th * rad_mul, 2),
            }
        )

    # Roadside & courtyard shrubs
    for _ in range(130):
        if len(shrub_specs) >= 36:
            break
        along = float(rng_tree.uniform(-55.0, 75.0))
        side_s = -1.0 if rng_tree.random() < 0.5 else 1.0
        pt = cl[2] + r_dir * along + r_perp * (side_s * float(rng_tree.uniform(7.9, 18.0)))
        sx, sz = float(pt[0]), float(pt[1])
        if point_in_polygon_2d(sx, sz, road_poly_xz) or point_in_polygon_2d(sx, sz, road_apron_poly_xz):
            continue
        d_r = abs(local_road_frame((sx, sz), centerline_xz)[1])
        if d_r < 7.8:
            continue
        if all(-cv2.pointPolygonTest(poly.astype(np.float32), (sx, sz), True) >= 1.1 for poly in all_polys):
            shrub_specs.append({"xz": [round(sx, 3), round(sz, 3)], "height_m": round(float(rng_tree.uniform(0.95, 1.55)), 2)})

    ground_tex = build_photographic_ground(
        swatches_dir,
        layout,
        ref_rgb,
        raw_buildings,
        courtyard_walls,
        cars,
        road_world,
        centerline_world,
        swatches,
        road_left_edge_xz=road_left_edge,
        road_right_edge_xz=road_right_edge,
        bev_ortho_info=bev_ortho_info,
    )
    swatches["farmland_ground"] = ground_tex

    spec = {
        "version": 3,
        "source_image": str(ref_src.relative_to(ROOT)),
        "ref_clean_image": str(ref_clean_path.resolve()),
        "colmap_sparse_dir": str(colmap_dir.resolve()),
        "colmap_guide_npz": str((out_dir / "colmap_guide.npz").resolve()),
        "camera": layout["camera"],
        "imageSize": layout["imageSize"],
        "depthSize": layout["depthSize"],
        "groundBounds": layout["groundBounds"],
        "road_polygon_xz": road_poly_xz,
        "road_apron_polygon_xz": road_apron_poly_xz,
        "road_left_edge_xz": road_left_edge,
        "road_right_edge_xz": road_right_edge,
        "road_centerline_xz": centerline_xz,
        "layout_audit": layout_audit,
        "utility_poles_xz": poles_xz,
        "parked_cars": cars,
        "tricycles": tricycles,
        "brick_piles": brick_piles,
        "courtyard_walls": courtyard_walls,
        "courtyard_gates": courtyard_gates,
        "trees": tree_specs,
        "shrubs": shrub_specs,
        "cameras": selected_cameras,
        "buildings": raw_buildings,
        "swatches": swatches,
        "foliage_swatches": foliage_swatches,
    }
    spec_path = out_dir / "village_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    digest_hex = hashlib.sha256(spec_path.read_bytes()).hexdigest()[:12]
    print(
        f"Prepared dense village_spec v3 ({digest_hex}): {len(raw_buildings)} buildings "
        f"({layout_audit['core_count']} core + {layout_audit['bev_count']} BEV + {layout_audit['infill_count']} fabric, "
        f"density={layout_audit['footprint_density_pct']}%, skew<={layout_audit['max_corner_skew_deg']}deg), "
        f"{len(courtyard_walls)} compound walls, {len(courtyard_gates)} gates, {len(tree_specs)} trees, {len(cars)} cars -> {spec_path}"
    )
    return spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/sparse/0")
    parser.add_argument("--images-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/images")
    parser.add_argument("--extra-colmap-dir", type=Path, default=None)
    parser.add_argument("--extra-images-dir", type=Path, default=None)
    parser.add_argument("--bev-dsm-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    prepare_evidence(
        colmap_dir=args.colmap_dir,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        extra_colmap_dir=args.extra_colmap_dir,
        extra_images_dir=args.extra_images_dir,
        bev_dsm_dir=args.bev_dsm_dir,
    )
