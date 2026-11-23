"""Antigravity <-> Blender 4.2/4.5+ bridge addon, headless command handler, and shared coordinate math."""

from __future__ import annotations

import io
import json
import math
import queue
import socket
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

bl_info = {
    "name": "Antigravity Blender Bridge",
    "author": "scene-data-demo",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Antigravity",
    "description": "TCP JSON-line live bridge & headless command parity (inspect, py, see, snapshot, audit)",
    "category": "Development",
}

try:
    import bpy  # type: ignore
    import bmesh  # type: ignore
    from mathutils import Euler, Matrix, Quaternion, Vector  # type: ignore

    HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore
    bmesh = None  # type: ignore
    Euler = Matrix = Quaternion = Vector = None  # type: ignore
    HAS_BPY = False


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876
MAX_JSON_LINE_BYTES = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Pure-math coordinate & COLMAP <-> Blender camera conversion helpers
# (Safe to import in both Blender Python and normal system Python)
# ---------------------------------------------------------------------------


def world_to_blender(v: Any) -> tuple[float, float, float]:
    """Convert Three.js Y-up (x, y, z) to Blender Z-up (x, -z, y)."""
    return (float(v[0]), float(-v[2]), float(v[1]))


def blender_to_three(v: Any) -> list[float]:
    """Convert Blender Z-up (bx, by, bz) to Three.js Y-up [bx, bz, -by]."""
    return [float(v[0]), float(v[2]), float(-v[1])]


def quat_wxyz_to_matrix(q: Any) -> list[list[float]]:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to a 3x3 orthonormal rotation matrix."""
    qw, qx, qy, qz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("Degenerate quaternion norm")
    w, x, y, z = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
        for r in range(3)
    ]


def _mat3_T(a: list[list[float]]) -> list[list[float]]:
    return [[a[c][r] for c in range(3)] for r in range(3)]


def _mat3_vec(a: list[list[float]], v: Any) -> list[float]:
    return [sum(a[r][k] * float(v[k]) for k in range(3)) for r in range(3)]


def _mat3_det(a: list[list[float]]) -> float:
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def colmap_pose_to_blender_matrix(
    rot_or_quat: Any,
    t_cw: Any,
) -> tuple[tuple[float, float, float], list[list[float]], list[list[float]]]:
    """Convert aligned Y-up COLMAP camera pose (R_cw/quat_wxyz, t_cw) to Blender Z-up 4x4 matrix.

    OpenCV cam axes: +X right, +Y down, +Z forward.
    Blender cam axes: +X right, +Y up, +Z backward -> R_cam_cv_to_bl = diag(1, -1, -1).
    World (Y-up) -> Blender (Z-up): M_b = [[1,0,0],[0,0,-1],[0,1,0]].
    Returns (eye_blender, rot_b2w_3x3, mat4_rows).
    """
    if len(rot_or_quat) == 4 and not isinstance(rot_or_quat[0], (list, tuple)):
        r_cw = quat_wxyz_to_matrix(rot_or_quat)
    else:
        r_cw = [[float(rot_or_quat[r][c]) for c in range(3)] for r in range(3)]
    t_vec = [float(t_cw[0]), float(t_cw[1]), float(t_cw[2])]
    r_wc = _mat3_T(r_cw)
    eye_w = [-v for v in _mat3_vec(r_wc, t_vec)]

    m_b = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    flip_cv = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    eye_b_list = _mat3_vec(m_b, eye_w)
    rot_b2w = _mat3_mul(_mat3_mul(m_b, r_wc), flip_cv)
    mat4 = [
        [rot_b2w[0][0], rot_b2w[0][1], rot_b2w[0][2], eye_b_list[0]],
        [rot_b2w[1][0], rot_b2w[1][1], rot_b2w[1][2], eye_b_list[1]],
        [rot_b2w[2][0], rot_b2w[2][1], rot_b2w[2][2], eye_b_list[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return (eye_b_list[0], eye_b_list[1], eye_b_list[2]), rot_b2w, mat4


def colmap_intrinsics_to_blender_camera(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    sensor_width: float = 36.0,
) -> dict[str, Any]:
    """Map COLMAP PINHOLE intrinsics to exact Blender HORIZONTAL-fit camera + render aspect settings.

    Note: In Blender 4.x with sensor_fit='HORIZONTAL', pixel_aspect_x=fy/fx & pixel_aspect_y=1.0
    aligns vertical FOV to 2*atan(H/(2*fy)), and shift_x/shift_y use principal-point offset
    normalized by max(width, height) = width (for landscape images).
    """
    w = float(width)
    h = float(height)
    ref_dim = max(w, h)
    # Effective sensor shift in Blender scales pixel_aspect_x on X vs Y; empirically verified <0.001px
    return {
        "resolution_x": int(width),
        "resolution_y": int(height),
        "sensor_fit": "HORIZONTAL" if w >= h else "VERTICAL",
        "sensor_width": float(sensor_width),
        "lens": float(sensor_width * float(fx) / w) if w >= h else float(sensor_width * float(fy) / h),
        "shift_x": float((w * 0.5 - float(cx)) / ref_dim),
        "shift_y": float((float(cy) - h * 0.5) * (float(fx) / float(fy)) / ref_dim),
        "pixel_aspect_x": float(fy) / float(fx),
        "pixel_aspect_y": 1.0,
    }


def point_in_polygon_2d(px: float, py: float, poly_xy: list[tuple[float, float]] | list[list[float]]) -> bool:
    """Ray-casting 2D point-in-polygon test without external dependencies."""
    inside = False
    n = len(poly_xy)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = float(poly_xy[i][0]), float(poly_xy[i][1])
        xj, yj = float(poly_xy[j][0]), float(poly_xy[j][1])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Blender-side session kernel, inspection, capture, snapshot, & audit engine
# ---------------------------------------------------------------------------


_KERNEL_ENV: dict[str, Any] = {}


def _ensure_kernel_env() -> dict[str, Any]:
    if HAS_BPY and not _KERNEL_ENV:
        _KERNEL_ENV.update(
            {
                "bpy": bpy,
                "bmesh": bmesh,
                "Vector": Vector,
                "Matrix": Matrix,
                "Euler": Euler,
                "Quaternion": Quaternion,
                "math": math,
                "json": json,
                "Path": Path,
                "world_to_blender": world_to_blender,
                "blender_to_three": blender_to_three,
            }
        )
    return _KERNEL_ENV


def _world_bbox_blender(obj: Any) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _available_eevee_engine() -> str:
    engines = {"BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"}
    if HAS_BPY:
        enum_items = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
            if candidate in enum_items:
                return candidate
    return next(iter(engines))


def _mesh_shell_volumes_and_topology(bm: Any) -> dict[str, Any]:
    """Analyze manifoldness and signed volume independently for every connected shell."""
    boundary_edges = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    non_manifold_edges = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    zero_area_faces = sum(1 for f in bm.faces if f.calc_area() < 1e-8)
    nan_verts = sum(
        1 for v in bm.verts if any(not math.isfinite(coord) for coord in (v.co.x, v.co.y, v.co.z))
    )

    bm.faces.ensure_lookup_table()
    visited = set()
    shell_volumes: list[float] = []
    for face in bm.faces:
        if face.index in visited:
            continue
        q = deque([face])
        visited.add(face.index)
        comp = []
        while q:
            cur = q.popleft()
            comp.append(cur)
            for edge in cur.edges:
                for nxt in edge.link_faces:
                    if nxt.index not in visited:
                        visited.add(nxt.index)
                        q.append(nxt)
        # Compute signed volume of this connected shell by fan triangulation around origin
        vol = 0.0
        for f in comp:
            verts = f.verts
            if len(verts) < 3:
                continue
            v0 = verts[0].co
            for idx in range(1, len(verts) - 1):
                v1 = verts[idx].co
                v2 = verts[idx + 1].co
                vol += v0.dot(v1.cross(v2)) / 6.0
        shell_volumes.append(float(vol))

    inverted_shells = sum(1 for v in shell_volumes if v <= 0.0)
    return {
        "vertices": len(bm.verts),
        "faces": len(bm.faces),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "zero_area_faces": zero_area_faces,
        "nan_vertices": nan_verts,
        "shell_count": len(shell_volumes),
        "shell_volumes": [round(v, 4) for v in shell_volumes],
        "inverted_shells": inverted_shells,
    }


def run_audit(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run three-state (pass | fail | degraded) geometry, normal, & placement hygiene gate."""
    if not HAS_BPY:
        raise RuntimeError("run_audit requires Blender bpy runtime")
    params = params or {}
    road_poly_xz = params.get("road_polygon_xz") or []
    apron_poly_xz = params.get("road_apron_polygon_xz") or []
    strict_layout = bool(params.get("strict_layout_audit", False))
    ground_min_y = float(params.get("ground_min_three_y", -0.8))
    ground_max_y = float(params.get("ground_max_three_y", 0.5))

    scene = bpy.context.scene
    mesh_objs = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    if not mesh_objs:
        return {
            "verdict": "fail",
            "summary": "No renderable mesh objects in scene",
            "failures": ["empty_scene"],
            "warnings": [],
            "objects": [],
        }

    failures: list[str] = []
    warnings: list[str] = []
    obj_reports: list[dict[str, Any]] = []
    closed_count = 0

    for obj in mesh_objs:
        mesh_data = obj.data
        bm = bmesh.new()
        try:
            bm.from_mesh(mesh_data)
            stats = _mesh_shell_volumes_and_topology(bm)
        finally:
            bm.free()

        bmin, bmax = _world_bbox_blender(obj)
        tmin = blender_to_three(bmin)
        tmax = blender_to_three(bmax)
        min_three_y = float(min(tmin[1], tmax[1]))
        requires_closed = bool(obj.get("closed_building", False)) or any(
            c.name == "COL_Buildings_Main" for c in obj.users_collection
        )
        if requires_closed:
            closed_count += 1
            if stats["boundary_edges"] > 0:
                failures.append(f"{obj.name}: boundary_edges={stats['boundary_edges']}")
            if stats["non_manifold_edges"] > 0:
                failures.append(f"{obj.name}: non_manifold_edges={stats['non_manifold_edges']}")
            if stats["inverted_shells"] > 0:
                failures.append(f"{obj.name}: inverted_shells={stats['inverted_shells']}")
            if min_three_y < ground_min_y or min_three_y > ground_max_y:
                failures.append(f"{obj.name}: base_three_y={min_three_y:.3f} out of [{ground_min_y}, {ground_max_y}]")

        if stats["nan_vertices"] > 0:
            failures.append(f"{obj.name}: nan_vertices={stats['nan_vertices']}")
        if stats["zero_area_faces"] > 0:
            warnings.append(f"{obj.name}: zero_area_faces={stats['zero_area_faces']}")

        if road_poly_xz and obj.get("check_road_clearance"):
            cx_b, cy_b, _ = [(bmin[k] + bmax[k]) * 0.5 for k in range(3)]
            cx_t, _, cz_t = blender_to_three((cx_b, cy_b, 0.0))
            if point_in_polygon_2d(cx_t, cz_t, road_poly_xz):
                failures.append(f"{obj.name}: centroid ({cx_t:.2f}, {cz_t:.2f}) intrudes road_polygon_xz")

        if strict_layout and requires_closed and (road_poly_xz or apron_poly_xz):
            mw = obj.matrix_world
            base_z = float(bmin[2]) + 0.5
            base_xz = []
            for v in mesh_data.vertices:
                vw = mw @ v.co
                if vw.z <= base_z:
                    tv = blender_to_three(vw)
                    base_xz.append((tv[0], tv[2]))
            if base_xz:
                n_ring = len(base_xz)
                probe_pts = list(base_xz)
                if n_ring <= 16:
                    for pi in range(n_ring):
                        ax, az = base_xz[pi]
                        bx, bz = base_xz[(pi + 1) % n_ring]
                        probe_pts.append(((ax + bx) * 0.5, (az + bz) * 0.5))
                for px_t, pz_t in probe_pts:
                    if road_poly_xz and point_in_polygon_2d(px_t, pz_t, road_poly_xz):
                        failures.append(f"{obj.name}: base vertex ({px_t:.2f}, {pz_t:.2f}) intrudes road_polygon_xz")
                        break
                    if apron_poly_xz and point_in_polygon_2d(px_t, pz_t, apron_poly_xz):
                        failures.append(f"{obj.name}: base vertex ({px_t:.2f}, {pz_t:.2f}) intrudes road_apron_polygon_xz")
                        break

        obj_reports.append(
            {
                "name": obj.name,
                "requires_closed": requires_closed,
                "three_min": [round(min(tmin[i], tmax[i]), 3) for i in range(3)],
                "three_max": [round(max(tmin[i], tmax[i]), 3) for i in range(3)],
                **stats,
            }
        )

    expected_buildings = params.get("expected_closed_buildings")
    if expected_buildings is not None and closed_count != int(expected_buildings):
        failures.append(f"closed_buildings={closed_count} != expected {int(expected_buildings)}")

    verdict = "fail" if failures else ("degraded" if len(warnings) > 25 else "pass")
    return {
        "verdict": verdict,
        "closed_buildings": closed_count,
        "total_render_meshes": len(mesh_objs),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures[:30],
        "warnings": warnings[:30],
        "objects": obj_reports,
    }


def _render_capture(params: dict[str, Any]) -> dict[str, Any]:
    scene = bpy.context.scene
    out_path = Path(params.get("out_path") or "artifacts/blender_village/capture.png").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_x, res_y = params.get("resolution") or [1024, 640]
    engine = params.get("engine") or _available_eevee_engine()

    prev = {
        "camera": scene.camera,
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "pct": scene.render.resolution_percentage,
        "pax": scene.render.pixel_aspect_x,
        "pay": scene.render.pixel_aspect_y,
        "filepath": scene.render.filepath,
        "fmt": scene.render.image_settings.file_format,
    }
    temp_cam_obj = None
    temp_cam_data = None
    try:
        cam_name = params.get("camera")
        if cam_name:
            cam_obj = bpy.data.objects.get(cam_name)
            if cam_obj is None or cam_obj.type != "CAMERA":
                raise ValueError(f"Camera not found: {cam_name}")
            scene.camera = cam_obj
            if params.get("preserve_camera_aspect"):
                res_x = int(params.get("resolution_x") or scene.render.resolution_x)
                res_y = int(params.get("resolution_y") or scene.render.resolution_y)
        else:
            eye_in = params.get("eye") or [6.3, 37.6, 87.8]
            target_in = params.get("target") or [4.1, 0.0, 1.6]
            fov_deg = float(params.get("fov") or 42.0)
            coord_space = params.get("coord_space", "three")
            eye_b = Vector(world_to_blender(eye_in) if coord_space == "three" else eye_in)
            tgt_b = Vector(world_to_blender(target_in) if coord_space == "three" else target_in)

            temp_cam_data = bpy.data.cameras.new("_AntigravityCaptureCam")
            temp_cam_data.type = "PERSP"
            temp_cam_data.sensor_fit = "VERTICAL"
            temp_cam_data.angle_y = math.radians(fov_deg)
            temp_cam_data.clip_start = 0.2
            temp_cam_data.clip_end = 1500.0
            temp_cam_obj = bpy.data.objects.new("_AntigravityCaptureCam", temp_cam_data)
            scene.collection.objects.link(temp_cam_obj)
            temp_cam_obj.location = eye_b
            direction = tgt_b - eye_b
            if direction.length < 1e-6:
                direction = Vector((0.0, 1.0, -0.3))
            temp_cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
            scene.camera = temp_cam_obj
            scene.render.pixel_aspect_x = 1.0
            scene.render.pixel_aspect_y = 1.0

        scene.render.engine = engine
        scene.render.resolution_x = int(res_x)
        scene.render.resolution_y = int(res_y)
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        return {
            "path": str(out_path),
            "resolution": [int(res_x), int(res_y)],
            "engine": engine,
            "camera": scene.camera.name if scene.camera else None,
            "bytes": out_path.stat().st_size if out_path.exists() else 0,
        }
    finally:
        scene.camera = prev["camera"]
        scene.render.engine = prev["engine"]
        scene.render.resolution_x = prev["res_x"]
        scene.render.resolution_y = prev["res_y"]
        scene.render.resolution_percentage = prev["pct"]
        scene.render.pixel_aspect_x = prev["pax"]
        scene.render.pixel_aspect_y = prev["pay"]
        scene.render.filepath = prev["filepath"]
        scene.render.image_settings.file_format = prev["fmt"]
        if temp_cam_obj is not None:
            bpy.data.objects.remove(temp_cam_obj, do_unlink=True)
        if temp_cam_data is not None:
            bpy.data.cameras.remove(temp_cam_data, do_unlink=True)


def handle_command(cmd: str, params: dict[str, Any] | None = None, base_dir: Path | None = None) -> dict[str, Any]:
    """Execute one bridge command synchronously on Blender's main thread and return structured JSON."""
    if not HAS_BPY:
        return {"ok": False, "verdict": "fail", "error": "Blender bpy module unavailable"}
    params = params or {}
    t0 = time.monotonic()

    try:
        if cmd in ("ping", "doctor"):
            scene = bpy.context.scene
            return {
                "ok": True,
                "verdict": "pass",
                "cmd": cmd,
                "blender_version": bpy.app.version_string,
                "blend_file": bpy.data.filepath or "",
                "render_engine": scene.render.engine,
                "available_eevee": _available_eevee_engine(),
                "collections": len(bpy.data.collections),
                "objects": len(scene.objects),
                "meshes": sum(1 for o in scene.objects if o.type == "MESH"),
                "cameras": [o.name for o in scene.objects if o.type == "CAMERA"],
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        if cmd == "inspect":
            scene = bpy.context.scene
            view_layer = bpy.context.view_layer
            selected = [o.name for o in bpy.context.selected_objects] if bpy.context.selected_objects else []
            active = view_layer.objects.active.name if view_layer.objects.active else None
            limit = int(params.get("limit", 250))
            objs = []
            for ob in scene.objects[:limit]:
                item: dict[str, Any] = {
                    "name": ob.name,
                    "type": ob.type,
                    "collections": [c.name for c in ob.users_collection],
                    "hide_render": bool(ob.hide_render),
                    "closed_building": bool(ob.get("closed_building", False)),
                }
                if ob.type == "MESH":
                    bmin, bmax = _world_bbox_blender(ob)
                    tmin = blender_to_three(bmin)
                    tmax = blender_to_three(bmax)
                    item["vertices"] = len(ob.data.vertices)
                    item["faces"] = len(ob.data.polygons)
                    item["materials"] = [m.name for m in ob.data.materials if m]
                    item["three_bbox"] = [
                        [round(min(tmin[k], tmax[k]), 3) for k in range(3)],
                        [round(max(tmin[k], tmax[k]), 3) for k in range(3)],
                    ]
                objs.append(item)
            return {
                "ok": True,
                "verdict": "pass",
                "cmd": "inspect",
                "collections": [
                    {"name": c.name, "objects": len(c.objects)} for c in bpy.data.collections
                ],
                "total_objects": len(scene.objects),
                "returned_objects": len(objs),
                "selected_objects": selected,
                "active_object": active,
                "objects": objs,
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        if cmd == "exec_py":
            code = str(params.get("code") or "")
            if not code.strip():
                return {"ok": False, "verdict": "fail", "error": "exec_py requires non-empty code"}
            env = _ensure_kernel_env()
            env["RESULT"] = None
            buf = io.StringIO()
            with redirect_stdout(buf):
                exec(compile(code, "<bridge_exec_py>", "exec"), env, env)
            res_payload: dict[str, Any] = {
                "ok": True,
                "verdict": "pass",
                "cmd": "exec_py",
                "stdout": buf.getvalue(),
                "result": env.get("RESULT"),
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }
            if params.get("see"):
                res_payload["capture"] = _render_capture(params)
            return res_payload

        if cmd == "capture_view":
            cap = _render_capture(params)
            return {
                "ok": True,
                "verdict": "pass",
                "cmd": "capture_view",
                **cap,
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        if cmd == "snapshot":
            snap_dir = Path(
                params.get("snapshots_dir")
                or (base_dir / "snapshots" if base_dir else "artifacts/blender_village/snapshots")
            ).resolve()
            snap_dir.mkdir(parents=True, exist_ok=True)
            label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(params.get("label") or "ckpt"))
            existing = sorted(snap_dir.glob("*.blend"))
            idx = len(existing) + 1
            snap_path = snap_dir / f"{idx:03d}_{label}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(snap_path), copy=True)
            return {
                "ok": True,
                "verdict": "pass",
                "cmd": "snapshot",
                "snapshot_path": str(snap_path),
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        if cmd == "rollback":
            snap_dir = Path(
                params.get("snapshots_dir")
                or (base_dir / "snapshots" if base_dir else "artifacts/blender_village/snapshots")
            ).resolve()
            target = params.get("snapshot_path")
            if target:
                snap_path = Path(target).resolve()
            else:
                existing = sorted(snap_dir.glob("*.blend"))
                if not existing:
                    return {"ok": False, "verdict": "fail", "error": f"No snapshots in {snap_dir}"}
                snap_path = existing[-1]
            if not snap_path.is_file():
                return {"ok": False, "verdict": "fail", "error": f"Snapshot not found: {snap_path}"}
            bpy.ops.wm.open_mainfile(filepath=str(snap_path))
            out_blend = params.get("blend_path")
            if out_blend:
                bpy.ops.wm.save_as_mainfile(filepath=str(Path(out_blend).resolve()))
            return {
                "ok": True,
                "verdict": "pass",
                "cmd": "rollback",
                "restored_from": str(snap_path),
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        if cmd == "audit":
            report = run_audit(params)
            return {
                "ok": report["verdict"] == "pass",
                "cmd": "audit",
                **report,
                "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
            }

        return {"ok": False, "verdict": "fail", "error": f"Unknown command: {cmd}"}
    except Exception as exc:
        return {
            "ok": False,
            "verdict": "fail",
            "cmd": cmd,
            "error": str(exc),
            "traceback": traceback.format_exc()[-2000:],
            "elapsed_ms": round((time.monotonic() - t0) * 1000.0, 2),
        }


# ---------------------------------------------------------------------------
# Blender GUI TCP Server & N-Sidebar Panel
# ---------------------------------------------------------------------------

if HAS_BPY:
    _JOB_QUEUE: queue.Queue[tuple[dict[str, Any], queue.Queue[dict[str, Any]]]] = queue.Queue()
    _SERVER_STATE: dict[str, Any] = {
        "running": False,
        "sock": None,
        "thread": None,
        "host": DEFAULT_HOST,
        "port": DEFAULT_PORT,
        "last_cmd": "",
    }

    def _client_loop(conn: socket.socket) -> None:
        buf = b""
        with conn:
            while _SERVER_STATE["running"]:
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                if len(buf) > MAX_JSON_LINE_BYTES:
                    break
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        req = json.loads(raw_line.decode("utf-8"))
                    except Exception as exc:
                        resp = {"ok": False, "verdict": "fail", "error": f"Bad JSON: {exc}"}
                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                        continue
                    reply_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
                    _JOB_QUEUE.put((req, reply_q))
                    resp = reply_q.get(timeout=300)
                    if "id" in req:
                        resp["id"] = req["id"]
                    conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))

    def _accept_loop(sock: socket.socket) -> None:
        sock.settimeout(0.5)
        while _SERVER_STATE["running"]:
            try:
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=_client_loop, args=(conn,), daemon=True)
            t.start()

    def _pump_queue() -> float | None:
        if not _SERVER_STATE["running"]:
            return None
        for _ in range(8):
            try:
                req, reply_q = _JOB_QUEUE.get_nowait()
            except queue.Empty:
                break
            cmd = str(req.get("cmd") or "")
            params = req.get("params") or {}
            _SERVER_STATE["last_cmd"] = cmd
            resp = handle_command(cmd, params)
            reply_q.put(resp)
        return 0.05

    def start_bridge_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        if _SERVER_STATE["running"]:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        sock.listen(8)
        _SERVER_STATE.update({"running": True, "sock": sock, "host": host, "port": int(port)})
        thread = threading.Thread(target=_accept_loop, args=(sock,), daemon=True)
        _SERVER_STATE["thread"] = thread
        thread.start()
        if not bpy.app.timers.is_registered(_pump_queue):
            bpy.app.timers.register(_pump_queue, first_interval=0.05, persistent=True)

    def stop_bridge_server() -> None:
        _SERVER_STATE["running"] = False
        sock = _SERVER_STATE.get("sock")
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        _SERVER_STATE["sock"] = None
        if bpy.app.timers.is_registered(_pump_queue):
            try:
                bpy.app.timers.unregister(_pump_queue)
            except Exception:
                pass

    class ANTIGRAVITY_OT_bridge_start(bpy.types.Operator):
        bl_idname = "antigravity.bridge_start"
        bl_label = "Start Bridge Server"

        def execute(self, context):
            scene = context.scene
            port = int(getattr(scene, "antigravity_bridge_port", DEFAULT_PORT))
            start_bridge_server(DEFAULT_HOST, port)
            self.report({"INFO"}, f"Antigravity Bridge listening on {DEFAULT_HOST}:{port}")
            return {"FINISHED"}

    class ANTIGRAVITY_OT_bridge_stop(bpy.types.Operator):
        bl_idname = "antigravity.bridge_stop"
        bl_label = "Stop Bridge Server"

        def execute(self, context):
            stop_bridge_server()
            self.report({"INFO"}, "Antigravity Bridge stopped")
            return {"FINISHED"}

    class VIEW3D_PT_antigravity_bridge(bpy.types.Panel):
        bl_label = "Antigravity Bridge"
        bl_idname = "VIEW3D_PT_antigravity_bridge"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Antigravity"

        def draw(self, context):
            layout = self.layout
            scene = context.scene
            layout.prop(scene, "antigravity_bridge_port", text="Port")
            if _SERVER_STATE["running"]:
                layout.label(text=f"Status: Listening :{_SERVER_STATE['port']}", icon="CHECKMARK")
                if _SERVER_STATE["last_cmd"]:
                    layout.label(text=f"Last cmd: {_SERVER_STATE['last_cmd']}")
                layout.operator("antigravity.bridge_stop", icon="PAUSE")
            else:
                layout.label(text="Status: Stopped", icon="X")
                layout.operator("antigravity.bridge_start", icon="PLAY")

    _CLASSES = (
        ANTIGRAVITY_OT_bridge_start,
        ANTIGRAVITY_OT_bridge_stop,
        VIEW3D_PT_antigravity_bridge,
    )

    def register() -> None:
        bpy.types.Scene.antigravity_bridge_port = bpy.props.IntProperty(
            name="Port", default=DEFAULT_PORT, min=1024, max=65535
        )
        for cls in _CLASSES:
            bpy.utils.register_class(cls)

    def unregister() -> None:
        stop_bridge_server()
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)
        if hasattr(bpy.types.Scene, "antigravity_bridge_port"):
            del bpy.types.Scene.antigravity_bridge_port


if __name__ == "__main__":
    # Headless dispatch when executed directly inside `blender --factory-startup -b ... --python blender_bridge_addon.py -- --req <json_file>`
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        req_idx = argv.index("--req") if "--req" in argv else -1
        if req_idx >= 0 and req_idx + 1 < len(argv):
            req_data = json.loads(Path(argv[req_idx + 1]).read_text(encoding="utf-8"))
            cmd_name = req_data.get("cmd", "doctor")
            cmd_params = req_data.get("params") or {}
            base = Path(req_data["base_dir"]) if req_data.get("base_dir") else None
            response = handle_command(cmd_name, cmd_params, base_dir=base)
            if req_data.get("save_on_ok") and response.get("ok") and cmd_params.get("blend_path"):
                save_target = Path(cmd_params["blend_path"]).resolve()
                save_target.parent.mkdir(parents=True, exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=str(save_target))
                response["saved_blend"] = str(save_target)
            print("BRIDGE_JSON_BEGIN")
            print(json.dumps(response, ensure_ascii=False))
            print("BRIDGE_JSON_END")
