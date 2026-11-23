"""Export BVH camera visibility and 16-ray hemisphere ambient occlusion probes for all building atlas faces."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

from blender_bridge_addon import world_to_blender  # noqa: E402


def _fibonacci_hemisphere(n_dirs: int = 16) -> np.ndarray:
    dirs = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_dirs):
        z = (i + 0.5) / n_dirs
        r = math.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * i
        dirs.append([math.cos(phi) * r, math.sin(phi) * r, z])
    return np.asarray(dirs, dtype=np.float64)


def _build_collection_bvh(collection_names: tuple[str, ...]) -> BVHTree:
    bm_all = bmesh.new()
    for col_name in collection_names:
        col = bpy.data.collections.get(col_name)
        if col is None:
            continue
        for ob in col.objects:
            if ob.type != "MESH" or ob.hide_render:
                continue
            bm_sub = bmesh.new()
            bm_sub.from_mesh(ob.data)
            bmesh.ops.transform(bm_sub, matrix=ob.matrix_world, verts=bm_sub.verts)
            mesh_tmp = bpy.data.meshes.new("_bvh_tmp")
            bm_sub.to_mesh(mesh_tmp)
            bm_sub.free()
            bm_all.from_mesh(mesh_tmp)
            bpy.data.meshes.remove(mesh_tmp)
    bvh = BVHTree.FromBMesh(bm_all)
    bm_all.free()
    return bvh


def export_bake_buffers(spec_path: Path, uv_map_path: Path, out_npz: Path, probe_grid: int = 8) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    uv_map = json.loads(uv_map_path.read_text(encoding="utf-8"))
    faces = uv_map["faces"]
    cameras = spec["cameras"]

    bvh_cam = _build_collection_bvh(("COL_Buildings_Main", "COL_Courtyards_Walls"))
    bvh_ao = _build_collection_bvh(("COL_Buildings_Main", "COL_Building_Details", "COL_Courtyards_Walls", "COL_Terrain_Road"))

    cam_eyes_b = [Vector(c["blender_eye"]) for c in cameras]
    hemi_local = _fibonacci_hemisphere(16)

    n_faces = len(faces)
    n_cams = len(cameras)
    vis = np.zeros((n_faces, probe_grid, probe_grid, n_cams), dtype=np.float32)
    ao = np.ones((n_faces, probe_grid, probe_grid), dtype=np.float32)

    grid_coords_full = np.linspace(0.05, 0.95, probe_grid)
    grid_coords_coarse = np.linspace(0.08, 0.92, probe_grid // 2)

    for fi, f in enumerate(faces):
        origin = np.asarray(f["origin_3d"], dtype=np.float64)
        u_span = np.asarray(f["u_span_3d"], dtype=np.float64)
        v_span = np.asarray(f["v_span_3d"], dtype=np.float64)
        nrm_w = np.asarray(f["normal_world"], dtype=np.float64)
        nrm_b = Vector(world_to_blender(nrm_w)).normalized()
        if f["role"] == "bottom":
            ao[fi].fill(0.68)
            continue

        is_core = bool(f.get("is_core", True))
        ref_v = Vector((0.0, 0.0, 1.0)) if abs(nrm_b.z) < 0.9 else Vector((1.0, 0.0, 0.0))
        t1 = nrm_b.cross(ref_v).normalized()
        t2 = nrm_b.cross(t1).normalized()

        if is_core:
            for row_i, sv in enumerate(grid_coords_full):
                for col_i, su in enumerate(grid_coords_full):
                    pw = origin + su * u_span + sv * v_span
                    pb = Vector(world_to_blender(pw))
                    start_vis = pb + nrm_b * 0.22
                    for ci, eye_b in enumerate(cam_eyes_b):
                        delta = eye_b - pb
                        dist = delta.length
                        if dist < 0.4:
                            vis[fi, row_i, col_i, ci] = 1.0
                            continue
                        direction = delta / dist
                        if direction.dot(nrm_b) <= 0.16:
                            continue
                        hit_loc, _, _, _ = bvh_cam.ray_cast(start_vis, direction, dist - 0.35)
                        if hit_loc is None:
                            vis[fi, row_i, col_i, ci] = 1.0

                    start_ao = pb + nrm_b * 0.12
                    hits = 0
                    for hx, hy, hz in hemi_local:
                        ao_dir = t1 * hx + t2 * hy + nrm_b * hz
                        hit_loc, _, _, _ = bvh_ao.ray_cast(start_ao, ao_dir, 4.8)
                        if hit_loc is not None:
                            hits += 1
                    ao[fi, row_i, col_i] = 1.0 - 0.36 * (hits / len(hemi_local))
        else:
            half = probe_grid // 2
            vis_sub = np.zeros((half, half, n_cams), dtype=np.float32)
            ao_sub = np.ones((half, half), dtype=np.float32)
            for row_i, sv in enumerate(grid_coords_coarse):
                for col_i, su in enumerate(grid_coords_coarse):
                    pw = origin + su * u_span + sv * v_span
                    pb = Vector(world_to_blender(pw))
                    start_vis = pb + nrm_b * 0.22
                    for ci, eye_b in enumerate(cam_eyes_b):
                        delta = eye_b - pb
                        dist = delta.length
                        if dist < 0.4:
                            vis_sub[row_i, col_i, ci] = 1.0
                            continue
                        direction = delta / dist
                        if direction.dot(nrm_b) <= 0.16:
                            continue
                        hit_loc, _, _, _ = bvh_cam.ray_cast(start_vis, direction, dist - 0.35)
                        if hit_loc is None:
                            vis_sub[row_i, col_i, ci] = 1.0

                    start_ao = pb + nrm_b * 0.12
                    hits = 0
                    for hx, hy, hz in hemi_local:
                        ao_dir = t1 * hx + t2 * hy + nrm_b * hz
                        hit_loc, _, _, _ = bvh_ao.ray_cast(start_ao, ao_dir, 4.8)
                        if hit_loc is not None:
                            hits += 1
                    ao_sub[row_i, col_i] = 1.0 - 0.36 * (hits / len(hemi_local))
            vis[fi] = np.repeat(np.repeat(vis_sub, 2, axis=0), 2, axis=1)
            ao[fi] = np.repeat(np.repeat(ao_sub, 2, axis=0), 2, axis=1)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, vis=vis, ao=ao)
    res = {"out_npz": str(out_npz), "faces": n_faces, "cameras": n_cams, "mean_ao": round(float(ao.mean()), 4)}
    print("BAKE_PROBES_OK", json.dumps(res))
    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    p.add_argument("--uv-map", type=Path, default=ROOT / "artifacts/blender_village/uv_atlas_map.json")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/blender_village/bake_probes.npz")
    return p.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)
    export_bake_buffers(args.spec.resolve(), args.uv_map.resolve(), args.out.resolve())
