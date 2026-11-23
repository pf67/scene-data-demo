"""Street-grid road regularization, 2D hard collision/setback solver, and northern courtyard village generator."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union


def ensure_ccw_blender_xz(pts_xz: np.ndarray) -> np.ndarray:
    """Order 2D XZ vertices CCW in Blender XY (x, -z) so solid meshes produce outward normals."""
    pts = np.asarray(pts_xz, dtype=np.float64)
    xy_bl = np.stack([pts[:, 0], -pts[:, 1]], axis=-1)
    area = float(np.sum(xy_bl[:, 0] * np.roll(xy_bl[:, 1], -1) - xy_bl[:, 1] * np.roll(xy_bl[:, 0], -1)))
    return pts[::-1].copy() if area < 0.0 else pts.copy()


def make_oriented_quad_xz(cx: float, cz: float, width: float, depth: float, dir_xz: np.ndarray) -> np.ndarray:
    """Construct a strict 90-degree CCW rectangular XZ footprint aligned to unit dir_xz."""
    u = np.asarray(dir_xz, dtype=np.float64).copy()
    u /= max(float(np.linalg.norm(u)), 1e-8)
    v = np.array([-u[1], u[0]], dtype=np.float64)
    hw, hd = float(width) * 0.5, float(depth) * 0.5
    c = np.array([cx, cz], dtype=np.float64)
    quad = np.array(
        [
            c - u * hw - v * hd,
            c + u * hw - v * hd,
            c + u * hw + v * hd,
            c - u * hw + v * hd,
        ],
        dtype=np.float64,
    )
    return ensure_ccw_blender_xz(quad)


def _extend_centerline(centerline_xz: np.ndarray | list[list[float]], ext_m: float = 180.0) -> tuple[np.ndarray, float]:
    cl = np.asarray(centerline_xz, dtype=np.float64)
    d0 = cl[1] - cl[0]
    d0 /= max(float(np.linalg.norm(d0)), 1e-8)
    d1 = cl[-1] - cl[-2]
    d1 /= max(float(np.linalg.norm(d1)), 1e-8)
    ext = np.vstack([cl[0] - d0 * ext_m, cl, cl[-1] + d1 * ext_m])
    return ext, ext_m


def local_road_frame(
    pt_xz: np.ndarray | list[float] | tuple[float, float],
    centerline_xz: np.ndarray | list[list[float]],
    *,
    ext_m: float = 180.0,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (along_s, signed_dist, u_tangent, n_south, nearest_pt) on the extended road centerline.

    u_tangent points roughly ESE along road from cl[0] to cl[-1].
    n_south = [-u_tangent[1], u_tangent[0]] points roughly +Z/+X (South side of street in VILLAGE_CALIBRATED).
    signed_dist < 0 denotes North side (-Z), signed_dist > 0 denotes South side (+Z).
    """
    ext, base_off = _extend_centerline(centerline_xz, ext_m=ext_m)
    px, pz = float(pt_xz[0]), float(pt_xz[1])
    p = np.array([px, pz], dtype=np.float64)

    best_d2 = float("inf")
    best_along = 0.0
    best_u = np.array([1.0, 0.0], dtype=np.float64)
    best_near = ext[0].copy()

    cum_s = -base_off
    for i in range(len(ext) - 1):
        a, b = ext[i], ext[i + 1]
        dv = b - a
        seg_len2 = float(np.dot(dv, dv))
        seg_len = math.sqrt(max(seg_len2, 1e-12))
        t = 0.0 if seg_len2 < 1e-12 else max(0.0, min(1.0, float(np.dot(p - a, dv) / seg_len2)))
        near = a + t * dv
        diff = p - near
        d2 = float(np.dot(diff, diff))
        if d2 < best_d2:
            best_d2 = d2
            best_near = near
            best_u = dv / max(seg_len, 1e-8)
            best_along = cum_s + t * seg_len
        cum_s += seg_len

    best_n = np.array([-best_u[1], best_u[0]], dtype=np.float64)
    signed_dist = float(np.dot(p - best_near, best_n))
    return best_along, signed_dist, best_u, best_n, best_near


def build_regularized_road(
    centerline_xz: np.ndarray | list[list[float]],
    *,
    carriageway_half_width: float = 4.4,
    apron_width: float = 3.2,
    ext_m: float = 160.0,
) -> dict[str, Any]:
    """Construct constant-width asphalt carriageway & concrete apron polygons from centerline_xz."""
    cl = np.asarray(centerline_xz, dtype=np.float64)
    tangents: list[np.ndarray] = []
    for i in range(len(cl)):
        if i == 0:
            t = cl[1] - cl[0]
        elif i == len(cl) - 1:
            t = cl[-1] - cl[-2]
        else:
            t1 = cl[i] - cl[i - 1]
            t1 /= max(float(np.linalg.norm(t1)), 1e-8)
            t2 = cl[i + 1] - cl[i]
            t2 /= max(float(np.linalg.norm(t2)), 1e-8)
            t = t1 + t2
        t /= max(float(np.linalg.norm(t)), 1e-8)
        tangents.append(t)

    left_pts = []
    right_pts = []
    for p, u in zip(cl, tangents):
        n_south = np.array([-u[1], u[0]], dtype=np.float64)
        # Keep edge offset perpendicular without sharp miter overshoot
        left_pts.append(p - n_south * carriageway_half_width)
        right_pts.append(p + n_south * carriageway_half_width)

    left_arr = np.asarray(left_pts, dtype=np.float64)
    right_arr = np.asarray(right_pts, dtype=np.float64)
    u0, u1 = tangents[0], tangents[-1]
    n0 = np.array([-u0[1], u0[0]], dtype=np.float64)
    n1 = np.array([-u1[1], u1[0]], dtype=np.float64)

    def _corridor_polygon(half_w: float) -> list[list[float]]:
        l_ext = np.vstack(
            [
                cl[0] - u0 * ext_m - n0 * half_w,
                cl - np.array([np.array([-u[1], u[0]]) * half_w for u in tangents]),
                cl[-1] + u1 * ext_m - n1 * half_w,
            ]
        )
        r_ext = np.vstack(
            [
                cl[0] - u0 * ext_m + n0 * half_w,
                cl + np.array([np.array([-u[1], u[0]]) * half_w for u in tangents]),
                cl[-1] + u1 * ext_m + n1 * half_w,
            ]
        )
        poly = np.vstack([l_ext, r_ext[::-1]])
        return [[round(float(x), 3), round(float(z), 3)] for x, z in poly]

    corridor_hw = carriageway_half_width + apron_width
    return {
        "road_left_edge_xz": [[round(float(x), 3), round(float(z), 3)] for x, z in left_arr],
        "road_right_edge_xz": [[round(float(x), 3), round(float(z), 3)] for x, z in right_arr],
        "road_polygon_xz": _corridor_polygon(carriageway_half_width),
        "road_apron_polygon_xz": _corridor_polygon(corridor_hw),
        "carriageway_half_width": float(carriageway_half_width),
        "apron_width": float(apron_width),
        "corridor_half_width": float(corridor_hw),
    }


def polygon_corner_skew_deg(poly_xz: np.ndarray | list[list[float]]) -> float:
    """Return max deviation (degrees) of interior corners from 90 degrees."""
    pts = np.asarray(poly_xz, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return 0.0
    max_dev = 0.0
    for i in range(n):
        v1 = pts[(i - 1) % n] - pts[i]
        v2 = pts[(i + 1) % n] - pts[i]
        l1 = float(np.linalg.norm(v1))
        l2 = float(np.linalg.norm(v2))
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        cos_a = max(-1.0, min(1.0, float(np.dot(v1, v2) / (l1 * l2))))
        ang = math.degrees(math.acos(cos_a))
        max_dev = max(max_dev, abs(ang - 90.0))
    return float(max_dev)


def regularize_footprint_to_grid(
    poly_xz: np.ndarray | list[list[float]],
    centerline_xz: np.ndarray | list[list[float]],
    *,
    min_side: float = 3.4,
    max_w: float = 22.0,
    max_d: float = 14.0,
) -> np.ndarray:
    """Snap any skewed 2D XZ polygon to an exact 90-degree CCW quad aligned to the local road frame."""
    pts = np.asarray(poly_xz, dtype=np.float64)
    shp = Polygon(pts)
    if not shp.is_valid:
        shp = shp.buffer(0.0)
    area_orig = max(float(shp.area), min_side * min_side)
    c_pt = shp.centroid
    cx, cz = float(c_pt.x), float(c_pt.y)
    _, _, u, n_s, _ = local_road_frame((cx, cz), centerline_xz)

    # Estimate along-road width (u) vs cross-road depth (n) from edge orientations without shear inflation
    u_edges: list[float] = []
    n_edges: list[float] = []
    m = len(pts)
    for i in range(m):
        e = pts[(i + 1) % m] - pts[i]
        elen = float(np.linalg.norm(e))
        if elen < 0.4:
            continue
        eu = abs(float(np.dot(e, u))) / elen
        en = abs(float(np.dot(e, n_s))) / elen
        if eu >= en:
            u_edges.append(abs(float(np.dot(e, u))) * 0.55 + elen * 0.45)
        else:
            n_edges.append(abs(float(np.dot(e, n_s))) * 0.55 + elen * 0.45)

    if u_edges and n_edges:
        w_raw = float(np.mean(u_edges))
        d_raw = float(np.mean(n_edges))
    else:
        pu = pts @ u
        pn = pts @ n_s
        w_raw = max(float(pu.max() - pu.min()), min_side)
        d_raw = max(float(pn.max() - pn.min()), min_side)

    aspect = max(0.38, min(2.8, w_raw / max(d_raw, 1.0)))
    w = math.sqrt(area_orig * aspect)
    d = math.sqrt(area_orig / aspect)
    w = max(min_side, min(max_w, w))
    d = max(min_side, min(max_d, d))
    return make_oriented_quad_xz(cx, cz, w, d, u)


def _rect_params(poly_xz: np.ndarray, centerline_xz: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, float]:
    pts = np.asarray(poly_xz, dtype=np.float64)
    c = pts.mean(0)
    _, sd, u, n_s, _ = local_road_frame(c, centerline_xz)
    pu = (pts - c) @ u
    pn = (pts - c) @ n_s
    w = max(2.5, float(pu.max() - pu.min()))
    d = max(2.5, float(pn.max() - pn.min()))
    side_sign = -1.0 if sd <= 0.0 else 1.0
    return c, w, d, u, n_s, side_sign


def _update_building_record_poly(b: dict[str, Any], poly_xz: np.ndarray) -> None:
    poly_ccw = ensure_ccw_blender_xz(poly_xz)
    h = float(b["height_m"])
    pts3d = np.column_stack([poly_ccw[:, 0], np.full(len(poly_ccw), h), poly_ccw[:, 1]])
    b["footprint_xz"] = [[round(float(x), 4), round(float(z), 4)] for x, z in poly_ccw]
    b["points_3d"] = [[round(float(x), 4), round(float(y), 4), round(float(z), 4)] for x, y, z in pts3d]


def solve_building_layout(
    buildings: list[dict[str, Any]],
    centerline_xz: np.ndarray | list[list[float]],
    road_poly_xz: list[list[float]],
    road_apron_poly_xz: list[list[float]],
    *,
    apron_half_width: float = 7.6,
    road_setback: float = 0.35,
    min_building_gap: float = 0.45,
    min_side: float = 3.2,
    max_iters: int = 60,
    drop_unresolvable_synthetic: bool = True,
    preserve_orientation: bool = False,
) -> list[dict[str, Any]]:
    """Iteratively enforce hard zero road/apron intrusion and zero building overlap.

    Preserves each building's original side of the road centerline and either local street-grid
    orthogonality (default) or each building's own detected roof orientation when preserve_orientation=True.
    """
    if not buildings:
        return []

    cl = np.asarray(centerline_xz, dtype=np.float64)
    apron_shp = Polygon(road_apron_poly_xz).buffer(0.0)
    road_shp = Polygon(road_poly_xz).buffer(0.0)
    min_clear_cl = apron_half_width + road_setback

    states: list[dict[str, Any]] = []
    for b in buildings:
        raw_fp = np.asarray(b["footprint_xz"], dtype=np.float64)
        if preserve_orientation and len(raw_fp) == 4:
            e0 = raw_fp[1] - raw_fp[0]
            e1 = raw_fp[2] - raw_fp[1]
            u_own = e0 if float(np.linalg.norm(e0)) >= float(np.linalg.norm(e1)) else e1
            u_own = u_own / max(float(np.linalg.norm(u_own)), 1e-8)
            c_own = raw_fp.mean(0)
            v_own = np.array([-u_own[1], u_own[0]], dtype=np.float64)
            w_own = max(min_side, float(np.ptp((raw_fp - c_own) @ u_own)))
            d_own = max(min_side, float(np.ptp((raw_fp - c_own) @ v_own)))
            poly = make_oriented_quad_xz(float(c_own[0]), float(c_own[1]), w_own, d_own, u_own)
            _, sd_own, _, _, _ = local_road_frame(c_own, cl)
            c, w, d, u, n_s, side_sign = c_own, w_own, d_own, u_own, v_own, (-1.0 if sd_own <= 0.0 else 1.0)
        else:
            poly = regularize_footprint_to_grid(raw_fp, cl)
            c, w, d, u, n_s, side_sign = _rect_params(poly, cl)
        states.append(
            {
                "rec": b,
                "orig_c": c.copy(),
                "c": c.copy(),
                "w": w,
                "d": d,
                "u": u,
                "n": n_s,
                "side": side_sign,
                "is_core": bool(b.get("is_core", True)),
                "quad": poly,
                "shp": Polygon(poly),
                "road_pushed": False,
            }
        )

    def rebuild_state(st: dict[str, Any]) -> None:
        if not preserve_orientation:
            _, _, u, n_s, _ = local_road_frame(st["c"], cl)
            st["u"] = u
            st["n"] = n_s
        q = make_oriented_quad_xz(float(st["c"][0]), float(st["c"][1]), st["w"], st["d"], st["u"])
        st["quad"] = q
        st["shp"] = Polygon(q)

    def push_off_road(st: dict[str, Any]) -> None:
        side = st["side"]
        for _ in range(4):
            corner_sd = [local_road_frame(p, cl)[1] for p in st["quad"]]
            # Distance of the closest corner along building's designated road side (+side * sd)
            min_side_dist = min(side * sd for sd in corner_sd)
            deficit = min_clear_cl - min_side_dist
            if deficit > 1e-3:
                _, _, _, n_loc, _ = local_road_frame(st["c"], cl)
                st["c"] = st["c"] + n_loc * (side * (deficit + 0.08))
                st["road_pushed"] = True
                rebuild_state(st)
            elif st["shp"].intersects(apron_shp) or st["shp"].intersects(road_shp):
                _, _, _, n_loc, _ = local_road_frame(st["c"], cl)
                st["c"] = st["c"] + n_loc * (side * 0.45)
                st["road_pushed"] = True
                rebuild_state(st)
            else:
                break

    # Initial pass: immediately clear road & aprons
    for st in states:
        push_off_road(st)

    # Iterative collision separation
    n_b = len(states)
    for it in range(max_iters):
        moved = False
        for i in range(n_b):
            si = states[i]
            bmin_i = si["shp"].bounds
            for j in range(i + 1, n_b):
                sj = states[j]
                bmin_j = sj["shp"].bounds
                # Fast AABB pre-filter
                if (
                    bmin_i[2] + min_building_gap < bmin_j[0]
                    or bmin_j[2] + min_building_gap < bmin_i[0]
                    or bmin_i[3] + min_building_gap < bmin_j[1]
                    or bmin_j[3] + min_building_gap < bmin_i[1]
                ):
                    continue
                dist = float(si["shp"].distance(sj["shp"]))
                if dist >= min_building_gap - 1e-3:
                    continue

                moved = True
                mid_c = 0.5 * (si["c"] + sj["c"])
                _, _, u_m, n_m, _ = local_road_frame(mid_c, cl)
                pi_u = (si["quad"] - mid_c) @ u_m
                pj_u = (sj["quad"] - mid_c) @ u_m
                pi_n = (si["quad"] - mid_c) @ n_m
                pj_n = (sj["quad"] - mid_c) @ n_m

                # Positive value means separated; negative means overlapping along that axis
                sep_u = max(float(pj_u.min() - pi_u.max()), float(pi_u.min() - pj_u.max()))
                sep_n = max(float(pj_n.min() - pi_n.max()), float(pi_n.min() - pj_n.max()))
                req_u = min_building_gap + 0.04 - sep_u
                req_n = min_building_gap + 0.04 - sep_n

                # Prefer separating along the axis with smaller overlap, or cross-road outward when both small
                if req_u <= req_n:
                    axis = u_m
                    req = max(0.08, req_u)
                    sign_ij = 1.0 if float(np.dot(sj["c"] - si["c"], u_m)) >= 0.0 else -1.0
                else:
                    axis = n_m
                    req = max(0.08, req_n)
                    sign_ij = 1.0 if float(np.dot(sj["c"] - si["c"], n_m)) >= 0.0 else -1.0

                # Relative mobility weights (synthetic infill moves more than observed core)
                wi = 0.28 if si["is_core"] else 0.78
                wj = 0.28 if sj["is_core"] else 0.78
                wsum = wi + wj
                fi, fj = wi / wsum, wj / wsum

                # Late-iteration core/crowded conflict resolution via gentle dimension trim
                if it >= 18 and req > 0.2:
                    trim_total = min(req * 0.6, 1.2)
                    if not si["is_core"] or (si["w"] >= sj["w"] and req_u <= req_n) or (si["d"] >= sj["d"] and req_n < req_u):
                        if req_u <= req_n and si["w"] > min_side:
                            dt = min(si["w"] - min_side, trim_total * 0.5)
                            si["w"] -= dt
                            si["c"] -= axis * (sign_ij * dt * 0.5)
                        elif req_n < req_u and si["d"] > min_side:
                            dt = min(si["d"] - min_side, trim_total * 0.5)
                            si["d"] -= dt
                            si["c"] -= axis * (sign_ij * dt * 0.5)
                    if req_u <= req_n and sj["w"] > min_side:
                        dt = min(sj["w"] - min_side, trim_total * 0.5)
                        sj["w"] -= dt
                        sj["c"] += axis * (sign_ij * dt * 0.5)
                    elif req_n < req_u and sj["d"] > min_side:
                        dt = min(sj["d"] - min_side, trim_total * 0.5)
                        sj["d"] -= dt
                        sj["c"] += axis * (sign_ij * dt * 0.5)

                # Apply translation (capped for non-road-pushed core buildings to preserve CAM_ref photo alignment)
                shift_i = -axis * (sign_ij * req * fi)
                shift_j = axis * (sign_ij * req * fj)
                for st, sh_v in ((si, shift_i), (sj, shift_j)):
                    cand_c = st["c"] + sh_v
                    if st["is_core"] and not st["road_pushed"]:
                        drift = cand_c - st["orig_c"]
                        d_norm = float(np.linalg.norm(drift))
                        if d_norm > 4.2:
                            cand_c = st["orig_c"] + drift * (4.2 / d_norm)
                    st["c"] = cand_c
                    rebuild_state(st)
                    push_off_road(st)

        if not moved:
            break

    # Final safety pass: drop any unresolvable non-core synthetic infill if STILL < min_building_gap * 0.8
    keep_mask = [True] * n_b
    if drop_unresolvable_synthetic:
        for i in range(n_b):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n_b):
                if not keep_mask[j]:
                    continue
                if float(states[i]["shp"].distance(states[j]["shp"])) < min_building_gap * 0.75:
                    if not states[j]["is_core"]:
                        keep_mask[j] = False
                    elif not states[i]["is_core"]:
                        keep_mask[i] = False
                        break

    # Final pairwise trim if two CORE buildings still touch (e.g., locked tight without dropping)
    kept_indices = [idx for idx, flag in enumerate(keep_mask) if flag]
    for _ in range(12):
        any_bad = False
        for pos_a in range(len(kept_indices)):
            ia = kept_indices[pos_a]
            sa = states[ia]
            for pos_b in range(pos_a + 1, len(kept_indices)):
                ib = kept_indices[pos_b]
                sb = states[ib]
                d_ab = float(sa["shp"].distance(sb["shp"]))
                if d_ab >= min_building_gap - 1e-3:
                    continue
                any_bad = True
                mid_c = 0.5 * (sa["c"] + sb["c"])
                _, _, u_m, n_m, _ = local_road_frame(mid_c, cl)
                pi_u = (sa["quad"] - mid_c) @ u_m
                pj_u = (sb["quad"] - mid_c) @ u_m
                pi_n = (sa["quad"] - mid_c) @ n_m
                pj_n = (sb["quad"] - mid_c) @ n_m
                sep_u = max(float(pj_u.min() - pi_u.max()), float(pi_u.min() - pj_u.max()))
                sep_n = max(float(pj_n.min() - pi_n.max()), float(pi_n.min() - pj_n.max()))
                if sep_u >= sep_n:
                    sign_ab = 1.0 if float(np.dot(sb["c"] - sa["c"], u_m)) >= 0.0 else -1.0
                    dt = max(0.25, (min_building_gap - sep_u) * 0.6)
                    for st, sgn in ((sa, -sign_ab), (sb, sign_ab)):
                        if st["w"] > 2.8:
                            cut = min(st["w"] - 2.8, dt)
                            st["w"] -= cut
                            st["c"] -= u_m * (sgn * cut * 0.5)
                            rebuild_state(st)
                            push_off_road(st)
                else:
                    sign_ab = 1.0 if float(np.dot(sb["c"] - sa["c"], n_m)) >= 0.0 else -1.0
                    dt = max(0.25, (min_building_gap - sep_n) * 0.6)
                    for st, sgn in ((sa, -sign_ab), (sb, sign_ab)):
                        if st["d"] > 2.8:
                            cut = min(st["d"] - 2.8, dt)
                            st["d"] -= cut
                            st["c"] -= n_m * (sgn * cut * 0.5)
                            rebuild_state(st)
                            push_off_road(st)
        if not any_bad:
            break

    solved: list[dict[str, Any]] = []
    for idx_new, st_idx in enumerate(kept_indices):
        st = states[st_idx]
        rec = st["rec"]
        rec["index"] = idx_new
        _update_building_record_poly(rec, st["quad"])
        solved.append(rec)
    return solved


def audit_layout_geometry(
    buildings: list[dict[str, Any]],
    centerline_xz: np.ndarray | list[list[float]],
    road_poly_xz: list[list[float]],
    road_apron_poly_xz: list[list[float]],
    *,
    skew_tol_deg: float = 1.5,
    overlap_tol_m2: float = 0.05,
) -> dict[str, Any]:
    """Compute exact Shapely topology, road/apron clearance, skew, and density metrics."""
    road_shp = Polygon(road_poly_xz).buffer(0.0)
    apron_shp = Polygon(road_apron_poly_xz).buffer(0.0)

    shps = [Polygon(b["footprint_xz"]).buffer(0.0) for b in buildings]
    road_intr_ids: list[str] = []
    road_intr_area = 0.0
    apron_intr_ids: list[str] = []
    apron_intr_area = 0.0
    max_skew = 0.0
    total_fp_area = 0.0

    for b, shp in zip(buildings, shps):
        total_fp_area += float(shp.area)
        max_skew = max(max_skew, polygon_corner_skew_deg(b["footprint_xz"]))
        ra = float(shp.intersection(road_shp).area)
        if ra > overlap_tol_m2:
            road_intr_ids.append(b["id"])
            road_intr_area += ra
        aa = float(shp.intersection(apron_shp).area)
        if aa > overlap_tol_m2:
            apron_intr_ids.append(b["id"])
            apron_intr_area += aa

    overlap_pairs: list[tuple[str, str, float]] = []
    max_ov = 0.0
    for i in range(len(buildings)):
        bi = shps[i].bounds
        for j in range(i + 1, len(buildings)):
            bj = shps[j].bounds
            if bi[2] < bj[0] or bj[2] < bi[0] or bi[3] < bj[1] or bj[3] < bi[1]:
                continue
            ia = float(shps[i].intersection(shps[j]).area)
            if ia > overlap_tol_m2:
                max_ov = max(max_ov, ia)
                overlap_pairs.append((buildings[i]["id"], buildings[j]["id"], round(ia, 3)))

    village_hull = unary_union(shps).convex_hull if shps else Polygon()
    non_road_hull = float(village_hull.difference(road_shp).area) if not village_hull.is_empty else 1.0
    density_pct = 100.0 * total_fp_area / max(non_road_hull, 1.0)

    core_n = sum(1 for b in buildings if b.get("is_core", True))
    bev_n = sum(1 for b in buildings if b.get("source") == "bev_dsm")
    infill_n = len(buildings) - core_n - bev_n

    passed = (
        len(road_intr_ids) == 0
        and len(apron_intr_ids) == 0
        and len(overlap_pairs) == 0
        and max_skew <= skew_tol_deg + 1e-3
    )
    return {
        "verdict": "pass" if passed else "fail",
        "building_count": len(buildings),
        "core_count": core_n,
        "bev_count": bev_n,
        "infill_count": infill_n,
        "road_intrusion_count": len(road_intr_ids),
        "road_intrusion_ids": road_intr_ids,
        "road_intrusion_area_m2": round(road_intr_area, 3),
        "apron_intrusion_count": len(apron_intr_ids),
        "apron_intrusion_ids": apron_intr_ids,
        "apron_intrusion_area_m2": round(apron_intr_area, 3),
        "building_overlap_count": len(overlap_pairs),
        "building_overlap_pairs": overlap_pairs[:20],
        "max_building_overlap_m2": round(max_ov, 3),
        "max_corner_skew_deg": round(max_skew, 3),
        "village_hull_non_road_m2": round(non_road_hull, 1),
        "total_footprint_m2": round(total_fp_area, 1),
        "footprint_density_pct": round(density_pct, 2),
    }


def build_synthetic_building_record(
    bid: str,
    idx: int,
    poly_xz: np.ndarray,
    height_m: float,
    archetype: str,
    wall_swatch: str,
    side_wall_swatch: str,
    *,
    solar_heater: bool = False,
    source: str = "procedural_fabric",
    obs_roof_rgb: list[float] | None = None,
) -> dict[str, Any]:
    """Create a complete spec entry matching core buildings for BEV/courtyard/infill structures."""
    poly_ccw = ensure_ccw_blender_xz(poly_xz)
    pts3d = np.column_stack([poly_ccw[:, 0], np.full(len(poly_ccw), float(height_m)), poly_ccw[:, 1]])
    roof_swatch_map = {
        "flat_parapet": "concrete_roof",
        "gable_red_tile": "terracotta_tile",
        "gable_silver_metal": "corrugated_silver",
        "blue_shed": "corrugated_blue",
        "metal_shed": "corrugated_silver",
    }
    roof_swatch = roof_swatch_map[archetype]
    ridge_rise = 1.15 if archetype.startswith("gable") else 0.0
    default_obs_roof = {
        "concrete_roof": [132.0, 128.0, 120.0],
        "terracotta_tile": [132.0, 88.0, 70.0],
        "corrugated_silver": [158.0, 162.0, 166.0],
        "corrugated_blue": [76.0, 102.0, 132.0],
    }[roof_swatch]
    obs_wall = {
        "plaster_white": [204.0, 198.0, 188.0],
        "plaster_warm": [186.0, 177.0, 164.0],
        "plaster_grey": [152.0, 148.0, 142.0],
        "red_brick": [136.0, 96.0, 78.0],
    }[wall_swatch]
    return {
        "id": bid,
        "index": idx,
        "is_core": False,
        "source": source,
        "roof_uv": [],
        "points_3d": [[round(float(x), 4), round(float(y), 4), round(float(z), 4)] for x, y, z in pts3d],
        "footprint_xz": [[round(float(x), 4), round(float(z), 4)] for x, z in poly_ccw],
        "height_m": round(float(height_m), 3),
        "height_prior_m": round(float(height_m), 3),
        "stories": 2 if height_m >= 5.0 else 1,
        "archetype": archetype,
        "roof_swatch": roof_swatch,
        "wall_swatch": wall_swatch,
        "side_wall_swatch": side_wall_swatch,
        "front_cornice_brick": bool(archetype == "flat_parapet" and idx % 3 != 1),
        "ridge_rise": round(ridge_rise, 3),
        "observed_roof_rgb": obs_roof_rgb if obs_roof_rgb is not None else default_obs_roof,
        "observed_wall_rgb": obs_wall,
        "has_parapet": archetype == "flat_parapet",
        "has_eave_overhang": archetype in ("gable_red_tile", "gable_silver_metal"),
        "solar_heater": solar_heater,
    }


def _in_allowed_village_zone(cx: float, cz: float) -> bool:
    """Respect northern poplar windbreak, northwest open crop fields, and foreground camera zone."""
    if not (-120.0 <= cx <= 116.0 and -156.0 <= cz <= 55.0):
        return False
    # North poplar windbreak boundary
    if cz < -142.0 - 0.17 * max(0.0, cx):
        return False
    # Northwest open farmland visible in upper-left of village_road_near.png
    if cx < -46.0 and cz < -80.0:
        return False
    if cx < -74.0 and cz < -48.0:
        return False
    if cx < -95.0 and cz < -20.0:
        return False
    # Foreground bottom-right near camera margin
    if cz > 49.0 and cx > -8.0:
        return False
    return True


def _sample_bev_roof_clue(
    poly_xz: np.ndarray,
    bev_ortho_info: dict[str, Any] | None,
) -> tuple[str | None, list[float] | None, float | None]:
    """Sample observed BEV orthophoto RGB & nDSM height inside candidate parcel when multi-view coverage exists."""
    if not bev_ortho_info:
        return None, None, None
    ortho = bev_ortho_info.get("ortho")
    ndsm = bev_ortho_info.get("ndsm")
    obs = bev_ortho_info.get("obs_views")
    bx0, bx1, bz0, bz1 = bev_ortho_info["bounds_xz"]
    if ortho is None or ndsm is None or obs is None:
        return None, None, None
    gh, gw = ndsm.shape
    mask = np.zeros((gh, gw), dtype=np.uint8)
    px_poly = np.stack(
        [
            np.clip((poly_xz[:, 0] - bx0) / (bx1 - bx0) * (gw - 1), 0, gw - 1),
            np.clip((poly_xz[:, 1] - bz0) / (bz1 - bz0) * (gh - 1), 0, gh - 1),
        ],
        axis=-1,
    ).astype(np.int32)
    cv2.fillPoly(mask, [px_poly], 255)
    sel = (mask > 0) & (obs >= 1)
    if int(sel.sum()) < 24:
        return None, None, None
    hs = ndsm[sel]
    rgbs = ortho[sel].astype(np.float32)
    h_p70 = float(np.percentile(hs, 70)) if len(hs) > 0 else 0.0
    rgb_med = np.median(rgbs, axis=0)
    r, g, b = float(rgb_med[0]), float(rgb_med[1]), float(rgb_med[2])
    if b - r > 14.0 and b > g + 4.0:
        arch = "blue_shed"
    elif r - b > 20.0 and r > g + 7.0:
        arch = "gable_red_tile"
    elif rgb_med.mean() > 148.0 and abs(r - b) < 14.0:
        arch = "gable_silver_metal"
    else:
        arch = "flat_parapet"
    obs_h = float(np.clip(h_p70, 3.0, 7.2)) if h_p70 >= 2.6 else None
    return arch, [round(r, 1), round(g, 1), round(b, 1)], obs_h


def generate_northern_village_fabric(
    core_buildings: list[dict[str, Any]],
    bev_buildings: list[dict[str, Any]],
    centerline_xz: np.ndarray | list[list[float]],
    road_poly_xz: list[list[float]],
    road_apron_poly_xz: list[list[float]],
    cam_eye_xz: np.ndarray,
    *,
    bev_ortho_info: dict[str, Any] | None = None,
    corridor_half_width: float = 7.6,
    target_density_pct: float = 47.0,
) -> list[dict[str, Any]]:
    """Build dense 180-230 northern rural buildings from BEV observations + streetfront/courtyard packing."""
    rng = np.random.default_rng(20261121)
    cl = np.asarray(centerline_xz, dtype=np.float64)
    apron_shp = Polygon(road_apron_poly_xz).buffer(0.0)

    placed_shps: list[Polygon] = [Polygon(b["footprint_xz"]).buffer(0.0) for b in core_buildings]
    placed_polys: list[np.ndarray] = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in core_buildings]
    infill: list[dict[str, Any]] = []
    next_idx = len(core_buildings)

    def can_place(quad: np.ndarray, min_gap: float = 0.45, min_road_dist: float = 8.05) -> Polygon | None:
        c = quad.mean(0)
        if not _in_allowed_village_zone(float(c[0]), float(c[1])):
            return None
        for pt in list(quad) + [c]:
            _, sd, _, _, _ = local_road_frame(pt, cl)
            if abs(sd) < min_road_dist:
                return None
        cand_shp = Polygon(quad)
        if not cand_shp.is_valid:
            cand_shp = cand_shp.buffer(0.0)
        if cand_shp.intersects(apron_shp):
            return None
        bmin = cand_shp.bounds
        for ex_shp in placed_shps:
            eb = ex_shp.bounds
            if bmin[2] + min_gap < eb[0] or eb[2] + min_gap < bmin[0] or bmin[3] + min_gap < eb[1] or eb[3] + min_gap < bmin[1]:
                continue
            if float(cand_shp.distance(ex_shp)) < min_gap:
                return None
        return cand_shp

    def accept_building(
        bid: str,
        quad: np.ndarray,
        cand_shp: Polygon,
        height_m: float,
        archetype: str,
        wall_sw: str,
        side_sw: str,
        *,
        solar: bool = False,
        source: str = "procedural_fabric",
        obs_rgb: list[float] | None = None,
    ) -> None:
        nonlocal next_idx
        rec = build_synthetic_building_record(
            bid,
            next_idx,
            quad,
            height_m,
            archetype,
            wall_sw,
            side_sw,
            solar_heater=solar,
            source=source,
            obs_roof_rgb=obs_rgb,
        )
        infill.append(rec)
        placed_shps.append(cand_shp)
        placed_polys.append(quad)
        next_idx += 1

    # Pass 0: Accept non-overlapping multi-view BEV-DSM observed buildings first
    bev_sorted = sorted(
        bev_buildings,
        key=lambda item: (-int(item.get("obs_views", 1)), -float(item.get("score", item.get("area_m2", 50.0)))),
    )
    bev_n = 0
    for item in bev_sorted:
        raw_poly = np.asarray(item["footprint_xz"], dtype=np.float64)
        reg = regularize_footprint_to_grid(raw_poly, cl, min_side=3.5, max_w=16.5, max_d=12.0)
        c = reg.mean(0)
        _, sd, _, n_loc, _ = local_road_frame(c, cl)
        shp = can_place(reg, min_gap=0.45, min_road_dist=corridor_half_width + 0.42)
        if shp is None:
            # Gently nudge outward along road normal once if merely grazing the apron
            side = -1.0 if sd <= 0 else 1.0
            corner_dists = [side * local_road_frame(p, cl)[1] for p in reg]
            deficit = (corridor_half_width + 0.45) - min(corner_dists)
            if 0.0 < deficit <= 3.8:
                reg = reg + n_loc * (side * (deficit + 0.1))
                shp = can_place(reg, min_gap=0.45, min_road_dist=corridor_half_width + 0.40)
        if shp is None:
            continue
        bev_n += 1
        arch = str(item.get("archetype", "flat_parapet"))
        h_val = float(np.clip(item.get("height_m", 4.4), 2.9, 7.2))
        w_sw = ["plaster_white", "plaster_warm", "plaster_grey", "red_brick"][bev_n % 4]
        s_sw = ["red_brick", "plaster_grey", "plaster_warm"][bev_n % 3]
        accept_building(
            f"house_bev_{bev_n:03d}",
            reg,
            shp,
            h_val,
            arch,
            w_sw,
            s_sw,
            solar=(arch == "flat_parapet" and h_val >= 4.2 and bev_n % 3 == 0),
            source="bev_dsm",
            obs_rgb=item.get("observed_roof_rgb"),
        )

    # Pass 1: Continuous Roadside Shopfront Row Completion (North & South Frontages flush behind concrete apron)
    shop_n = 0
    for side_sign in (-1.0, 1.0):
        along = -8.0
        while along <= 165.0:
            w_shop = float(rng.uniform(8.4, 11.2))
            d_shop = float(rng.uniform(6.8, 8.2))
            c_along = along + w_shop * 0.5
            ref_guess = cl[0] + (cl[-1] - cl[0]) / max(float(np.linalg.norm(cl[-1] - cl[0])), 1e-6) * c_along
            _, _, u_loc, n_loc, near_pt = local_road_frame(ref_guess, cl)
            center = near_pt + n_loc * (side_sign * (corridor_half_width + 0.48 + d_shop * 0.5))
            quad = make_oriented_quad_xz(float(center[0]), float(center[1]), w_shop, d_shop, u_loc)
            used_w = w_shop
            shp = can_place(quad, min_gap=0.45, min_road_dist=corridor_half_width + 0.40)
            if shp is None:
                used_w = max(6.4, w_shop * 0.72)
                d_alt = min(d_shop, 6.4)
                c_alt = along + used_w * 0.5
                ref_alt = cl[0] + (cl[-1] - cl[0]) / max(float(np.linalg.norm(cl[-1] - cl[0])), 1e-6) * c_alt
                _, _, u_loc, n_loc, near_alt = local_road_frame(ref_alt, cl)
                center2 = near_alt + n_loc * (side_sign * (corridor_half_width + 0.48 + d_alt * 0.5))
                quad2 = make_oriented_quad_xz(float(center2[0]), float(center2[1]), used_w, d_alt, u_loc)
                shp = can_place(quad2, min_gap=0.45, min_road_dist=corridor_half_width + 0.40)
                if shp is not None:
                    quad = quad2
                    c_along = c_alt
            if shp is not None:
                shop_n += 1
                clue_arch, clue_rgb, clue_h = _sample_bev_roof_clue(quad, bev_ortho_info)
                arch = clue_arch or ["flat_parapet", "flat_parapet", "gable_red_tile", "gable_silver_metal", "flat_parapet"][shop_n % 5]
                h_val = clue_h if clue_h is not None else float(rng.uniform(4.2, 6.3) if arch != "blue_shed" else rng.uniform(3.1, 3.7))
                w_sw = ["plaster_white", "plaster_warm", "plaster_grey"][shop_n % 3]
                s_sw = "red_brick" if shop_n % 2 == 0 else "plaster_grey"
                accept_building(
                    f"house_shop_{shop_n:03d}",
                    quad,
                    shp,
                    h_val,
                    arch,
                    w_sw,
                    s_sw,
                    solar=(arch == "flat_parapet" and shop_n % 3 == 0),
                    obs_rgb=clue_rgb,
                )
                along = c_along + used_w * 0.5 + 0.55
            else:
                along += 2.2

    # Pass 2: Structured Northern Courtyard Parcel Rows inside authentic village boundary
    row_n = 0
    cl_dir = (cl[-1] - cl[0]) / max(float(np.linalg.norm(cl[-1] - cl[0])), 1e-6)
    row_plan = [
        (-1.0, [20.2, 32.0, 44.2, 56.8, 70.0, 84.0, 98.5], -10.0, 170.0),
        (1.0, [20.2, 32.0, 44.2, 56.5], -10.0, 166.0),
    ]
    for side_sign, offsets_list, along_start, along_end in row_plan:
        for r_idx, dist_perp in enumerate(offsets_list):
            along = along_start + (r_idx % 2) * 2.5
            while along <= along_end:
                if len(infill) + len(core_buildings) >= 215:
                    break
                w_m = float(rng.uniform(9.4, 12.2))
                d_m = float(rng.uniform(6.8, 8.6))
                c_along = along + w_m * 0.5
                guess = cl[0] + cl_dir * c_along
                _, _, u_loc, n_loc, near_pt = local_road_frame(guess, cl)
                jit_n = float(rng.uniform(-0.6, 0.6))
                center = near_pt + n_loc * (side_sign * (dist_perp + jit_n))
                quad = make_oriented_quad_xz(float(center[0]), float(center[1]), w_m, d_m, u_loc)
                used_w = w_m
                shp = can_place(quad, min_gap=0.48, min_road_dist=corridor_half_width + 0.50)
                if shp is None:
                    used_w = 7.8
                    d_alt = 5.8
                    c_alt = along + used_w * 0.5
                    guess2 = cl[0] + cl_dir * c_alt
                    _, _, u_loc2, n_loc2, near_pt2 = local_road_frame(guess2, cl)
                    center2 = near_pt2 + n_loc2 * (side_sign * (dist_perp + jit_n))
                    quad2 = make_oriented_quad_xz(float(center2[0]), float(center2[1]), used_w, d_alt, u_loc2)
                    shp = can_place(quad2, min_gap=0.45, min_road_dist=corridor_half_width + 0.50)
                    if shp is not None:
                        quad = quad2
                        c_along = c_alt
                if shp is not None:
                    row_n += 1
                    clue_arch, clue_rgb, clue_h = _sample_bev_roof_clue(quad, bev_ortho_info)
                    arch_pool = (
                        "flat_parapet", "flat_parapet", "flat_parapet",
                        "gable_red_tile", "gable_red_tile", "gable_silver_metal", "blue_shed",
                    )
                    arch = clue_arch or arch_pool[row_n % len(arch_pool)]
                    if clue_h is not None:
                        h_val = clue_h
                    elif arch == "blue_shed":
                        h_val = float(rng.uniform(3.0, 3.6))
                    elif arch == "flat_parapet":
                        h_val = float(rng.uniform(3.8, 6.2))
                    else:
                        h_val = float(rng.uniform(4.2, 6.4))
                    w_sw = ["plaster_white", "plaster_warm", "plaster_grey", "red_brick"][row_n % 4]
                    s_sw = ["red_brick", "plaster_grey", "plaster_warm"][row_n % 3]
                    accept_building(
                        f"house_row_{row_n:03d}",
                        quad,
                        shp,
                        h_val,
                        arch,
                        w_sw,
                        s_sw,
                        solar=(arch == "flat_parapet" and row_n % 3 == 1),
                        obs_rgb=clue_rgb,
                    )
                    along = c_along + used_w * 0.5 + float(rng.uniform(0.55, 1.05))
                else:
                    along += 2.4

    # Pass 3: Tight Courtyard Wings & Lean-To Sheds (`厢房 / 彩钢棚`) beside core + main row houses
    annex_n = 0
    parents = list(core_buildings) + [b for b in infill if not b["id"].startswith("house_annex_")]
    for b in parents:
        if annex_n >= 75 or len(infill) + len(core_buildings) >= 265:
            break
        # Check live density ceiling so village remains realistic rather than solid concrete slab
        if len(infill) + len(core_buildings) >= 200 and annex_n >= 40:
            road_shp_tmp = Polygon(road_poly_xz).buffer(0.0)
            hull_area = max(float(unary_union(placed_shps).convex_hull.difference(road_shp_tmp).area), 1.0)
            fp_area = sum(float(s.area) for s in placed_shps)
            if (100.0 * fp_area / hull_area) >= target_density_pct + 2.5:
                break

        poly = np.asarray(b["footprint_xz"], dtype=np.float64)
        c_b = poly.mean(0)
        away_cam = c_b - cam_eye_xz
        away_cam /= max(float(np.linalg.norm(away_cam)), 1e-6)
        is_fg_core = bool(b.get("is_core", False)) and float(np.linalg.norm(c_b - cam_eye_xz)) < 95.0

        best_cand = None
        best_score = -9.0
        n_e = len(poly)
        for ei in range(n_e):
            p0, p1 = poly[ei], poly[(ei + 1) % n_e]
            evec = p1 - p0
            elen = float(np.linalg.norm(evec))
            if elen < 4.5:
                continue
            u_e = evec / elen
            n_out = np.array([-u_e[1], u_e[0]], dtype=np.float64)
            if float(np.dot(n_out, 0.5 * (p0 + p1) - c_b)) < 0.0:
                n_out = -n_out
            # Protect foreground CAM_ref visible facades
            cam_align = float(np.dot(n_out, away_cam))
            if is_fg_core and cam_align < -0.20:
                continue
            mid_e = 0.5 * (p0 + p1)
            for w_len, dep in ((min(elen * 0.85, 7.6), 4.0), (5.4, 3.4)):
                cand_c = mid_e + n_out * (dep * 0.5 + 0.55)
                al_c, sd_c, _, _, _ = local_road_frame(cand_c, cl)
                if sd_c > 60.5 or al_c < -11.0:
                    continue
                cand = make_oriented_quad_xz(float(cand_c[0]), float(cand_c[1]), w_len, dep, u_e)
                shp = can_place(cand, min_gap=0.45, min_road_dist=corridor_half_width + 0.70)
                if shp is not None:
                    score = cam_align + (0.2 if dep > 3.5 else 0.0)
                    if score > best_score:
                        best_score = score
                        best_cand = cand, shp
                    break
        if best_cand is not None:
            cand, shp = best_cand
            annex_n += 1
            h_core = float(b["height_m"])
            is_shed = (annex_n % 3 != 1)
            arch = ("blue_shed" if annex_n % 5 == 0 else "metal_shed") if is_shed else "flat_parapet"
            h_annex = float(rng.uniform(2.85, 3.35)) if is_shed else max(2.9, min(h_core - 0.7, float(rng.uniform(3.2, 3.9))))
            w_sw = str(rng.choice(["red_brick", "plaster_grey", "plaster_warm"], p=[0.46, 0.34, 0.20]))
            s_sw = str(rng.choice(["red_brick", "plaster_grey"], p=[0.6, 0.4]))
            accept_building(f"house_annex_{annex_n:03d}", cand, shp, h_annex, arch, w_sw, s_sw, source="courtyard_wing")

    return infill
