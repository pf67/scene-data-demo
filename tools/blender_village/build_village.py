"""Blender 4.5 village builder: 110+ dense buildings, dedicated road, alpha leaf-card trees, sculpted cars & BVH facade gating."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore
import bmesh  # type: ignore
from mathutils import Euler, Matrix, Vector  # type: ignore
from mathutils.bvhtree import BVHTree  # type: ignore
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

from blender_bridge_addon import world_to_blender  # noqa: E402

ATLAS_SIZE = 2048
CORE_PER_ATLAS = 16
INFILL_PER_ATLAS = 36


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablock in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights, bpy.data.collections):
        for item in list(datablock):
            if item.users == 0:
                datablock.remove(item)


def get_or_create_collection(name: str, *, hide_render: bool = False) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    col.hide_render = hide_render
    return col


def link_only(ob: bpy.types.Object, col: bpy.types.Collection) -> None:
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    col.objects.link(ob)


def make_pbr_material(
    name: str,
    *,
    base_color: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0),
    roughness: float = 0.82,
    metallic: float = 0.0,
    specular: float = 0.35,
    albedo_path: str | None = None,
    normal_path: str | None = None,
    rough_path: str | None = None,
    uv_scale: float | tuple[float, float] = 1.0,
    normal_strength: float = 0.72,
    double_sided: bool = False,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = not double_sided
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular

    sx, sy = (uv_scale, uv_scale) if isinstance(uv_scale, (int, float)) else uv_scale
    mapping = None
    if (abs(sx - 1.0) > 1e-4 or abs(sy - 1.0) > 1e-4) and (albedo_path or normal_path or rough_path):
        tex_coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (sx, sy, 1.0)
        nt.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

    def add_img(path_str: str, colorspace: str) -> bpy.types.ShaderNodeTexImage:
        node = nt.nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(path_str, check_existing=True)
        img.colorspace_settings.name = colorspace
        node.image = img
        if mapping is not None:
            nt.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
        return node

    if albedo_path:
        tex_a = add_img(albedo_path, "sRGB")
        nt.links.new(tex_a.outputs["Color"], bsdf.inputs["Base Color"])
    if rough_path:
        tex_r = add_img(rough_path, "Non-Color")
        nt.links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])
    if normal_path:
        tex_n = add_img(normal_path, "Non-Color")
        nrm_node = nt.nodes.new("ShaderNodeNormalMap")
        nrm_node.inputs["Strength"].default_value = normal_strength
        nt.links.new(tex_n.outputs["Color"], nrm_node.inputs["Color"])
        nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_alpha_mask_material(
    name: str,
    *,
    rgba_path: str,
    normal_path: str | None = None,
    rough_path: str | None = None,
    cutoff: float = 0.45,
) -> bpy.types.Material:
    """Create double-sided glTF-compatible alpha-MASK botanical leaf material."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = False
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.78
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.14

    tex_rgba = nt.nodes.new("ShaderNodeTexImage")
    img_rgba = bpy.data.images.load(rgba_path, check_existing=True)
    img_rgba.colorspace_settings.name = "sRGB"
    tex_rgba.image = img_rgba
    nt.links.new(tex_rgba.outputs["Color"], bsdf.inputs["Base Color"])

    math_cut = nt.nodes.new("ShaderNodeMath")
    math_cut.operation = "GREATER_THAN"
    math_cut.inputs[1].default_value = float(cutoff)
    nt.links.new(tex_rgba.outputs["Alpha"], math_cut.inputs[0])
    nt.links.new(math_cut.outputs["Value"], bsdf.inputs["Alpha"])

    if rough_path:
        tex_r = nt.nodes.new("ShaderNodeTexImage")
        img_r = bpy.data.images.load(rough_path, check_existing=True)
        img_r.colorspace_settings.name = "Non-Color"
        tex_r.image = img_r
        nt.links.new(tex_r.outputs["Color"], bsdf.inputs["Roughness"])

    if normal_path:
        tex_n = nt.nodes.new("ShaderNodeTexImage")
        img_n = bpy.data.images.load(normal_path, check_existing=True)
        img_n.colorspace_settings.name = "Non-Color"
        tex_n.image = img_n
        nrm_node = nt.nodes.new("ShaderNodeNormalMap")
        nrm_node.inputs["Strength"].default_value = 0.5
        nt.links.new(tex_n.outputs["Color"], nrm_node.inputs["Color"])
        nt.links.new(nrm_node.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


def compute_face_basis_st(face_pts: np.ndarray, *, prefer_up_v: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[float]]]:
    pts = np.asarray(face_pts, dtype=np.float64)
    a, b, c = pts[0], pts[1], pts[2]
    normal = np.cross(b - a, c - a)
    n_norm = float(np.linalg.norm(normal))
    normal = np.array([0.0, 1.0, 0.0]) if n_norm < 1e-9 else (normal / n_norm)

    if prefer_up_v and abs(normal[1]) < 0.45:
        u_dir = b - a
        u_dir[1] = 0.0
        ul = float(np.linalg.norm(u_dir))
        if ul < 1e-6:
            u_dir = np.cross(np.array([0.0, 1.0, 0.0]), normal)
            ul = float(np.linalg.norm(u_dir))
        u_dir /= max(ul, 1e-6)
        v_dir = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        v_ortho = v_dir - np.dot(v_dir, u_dir) * u_dir - np.dot(v_dir, normal) * normal
        if np.linalg.norm(v_ortho) > 0.2:
            v_dir = v_ortho / np.linalg.norm(v_ortho)
    else:
        u_dir = b - a
        u_dir /= max(float(np.linalg.norm(u_dir)), 1e-6)
        v_dir = np.cross(normal, u_dir)
        v_dir /= max(float(np.linalg.norm(v_dir)), 1e-6)

    rel = pts - a
    su = rel @ u_dir
    sv = rel @ v_dir
    u_min, u_max = float(su.min()), float(su.max())
    v_min, v_max = float(sv.min()), float(sv.max())
    du, dv = max(u_max - u_min, 1e-3), max(v_max - v_min, 1e-3)
    origin = a + u_dir * u_min + v_dir * v_min
    st_poly = [[float((u - u_min) / du), float((v - v_min) / dv)] for u, v in zip(su, sv)]
    return origin, u_dir * du, v_dir * dv, st_poly


def pack_building_rects(is_gable: bool, cell_x0: int, cell_y0: int, cell_size: int = 512) -> dict[str, list[int]]:
    """Scale-invariant UV rects inside a cell_size x cell_size building tile with >= 6px gutters."""
    if not is_gable:
        base = {
            "bottom": [480, 8, 504, 32],
            "roof_0": [8, 8, 472, 236],
            "wall_0": [8, 246, 250, 370],
            "wall_1": [260, 246, 504, 370],
            "wall_2": [8, 380, 250, 504],
            "wall_3": [260, 380, 504, 504],
        }
    else:
        base = {
            "bottom": [480, 8, 504, 32],
            "roof_0": [8, 8, 472, 118],
            "roof_1": [8, 126, 472, 236],
            "gable_0": [8, 244, 250, 296],
            "gable_1": [260, 244, 504, 296],
            "wall_0": [8, 304, 250, 400],
            "wall_1": [260, 304, 504, 400],
            "wall_2": [8, 408, 250, 504],
            "wall_3": [260, 408, 504, 504],
        }
    s = cell_size / 512.0
    return {
        k: [
            cell_x0 + int(round(x0 * s)),
            cell_y0 + int(round(y0 * s)),
            cell_x0 + int(round(x1 * s)),
            cell_y0 + int(round(y1 * s)),
        ]
        for k, (x0, y0, x1, y1) in base.items()
    }


def add_bm_box(
    bm: bmesh.types.BMesh,
    center_w: np.ndarray,
    size_w: np.ndarray,
    rot_yaw_w: float = 0.0,
    pitch_w: float = 0.0,
    uv_layer: Any = None,
    uv_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    flip_u: bool = False,
) -> None:
    hx, hy, hz = float(size_w[0]) * 0.5, float(size_w[1]) * 0.5, float(size_w[2]) * 0.5
    local = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz],
            [-hx, hy, -hz], [hx, hy, -hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    if abs(pitch_w) > 1e-6:
        cp, sp = math.cos(pitch_w), math.sin(pitch_w)
        local = local @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64).T
    if abs(rot_yaw_w) > 1e-6:
        cy, sy = math.cos(rot_yaw_w), math.sin(rot_yaw_w)
        local = local @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64).T
    world = local + np.asarray(center_w, dtype=np.float64)
    verts = [bm.verts.new(world_to_blender(p)) for p in world]
    u0, v0, u1, v1 = uv_rect
    if flip_u:
        u0, u1 = u1, u0
    quad_uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        face = bm.faces.new([verts[i] for i in f])
        if uv_layer is not None:
            for loop, uv in zip(face.loops, quad_uvs):
                loop[uv_layer].uv = uv


def add_bm_beam(bm: bmesh.types.BMesh, a_w: np.ndarray, b_w: np.ndarray, thickness: float, height: float, *, y_base: float | None = None) -> None:
    delta = np.asarray(b_w, dtype=np.float64) - np.asarray(a_w, dtype=np.float64)
    horiz_len = math.hypot(delta[0], delta[2])
    if horiz_len < 1e-4:
        return
    mid = 0.5 * (np.asarray(a_w, dtype=np.float64) + np.asarray(b_w, dtype=np.float64))
    if y_base is not None:
        mid[1] = y_base + height * 0.5
    yaw = math.atan2(-delta[2], delta[0])
    add_bm_box(bm, mid, np.array([horiz_len + thickness * 0.6, height, thickness]), rot_yaw_w=yaw)


def add_bm_prism_beam(bm: bmesh.types.BMesh, a_w: np.ndarray, b_w: np.ndarray, thickness: float, height: float) -> None:
    va, vb = np.asarray(a_w, dtype=np.float64), np.asarray(b_w, dtype=np.float64)
    d = vb - va
    length = float(np.linalg.norm(d))
    if length < 1e-4:
        return
    u = d / length
    up_ref = np.array([0.0, 1.0, 0.0]) if abs(float(np.dot(u, np.array([0.0, 1.0, 0.0])))) <= 0.95 else np.array([1.0, 0.0, 0.0])
    side = np.cross(u, up_ref)
    side /= max(float(np.linalg.norm(side)), 1e-6)
    up = np.cross(side, u)
    up /= max(float(np.linalg.norm(up)), 1e-6)
    hs, hh = thickness * 0.5, height * 0.5
    pts = [
        va - side * hs - up * hh, va + side * hs - up * hh, va + side * hs + up * hh, va - side * hs + up * hh,
        vb - side * hs - up * hh, vb + side * hs - up * hh, vb + side * hs + up * hh, vb - side * hs + up * hh,
    ]
    verts = [bm.verts.new(world_to_blender(p)) for p in pts]
    for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
        bm.faces.new([verts[i] for i in f])


def add_bm_tube(bm: bmesh.types.BMesh, pts_w: list[np.ndarray], radii: list[float], sides: int = 7) -> None:
    rings = []
    n = len(pts_w)
    for i, (pt, rad) in enumerate(zip(pts_w, radii)):
        t_vec = pts_w[min(i + 1, n - 1)] - pts_w[max(i - 1, 0)]
        t_hat = t_vec / max(float(np.linalg.norm(t_vec)), 1e-6)
        ref = np.array([0.0, 0.0, 1.0]) if abs(t_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        b1 = np.cross(t_hat, ref)
        b1 /= max(float(np.linalg.norm(b1)), 1e-6)
        b2 = np.cross(t_hat, b1)
        ring = []
        for s in range(sides):
            ang = s * math.tau / sides
            p = pt + (b1 * math.cos(ang) + b2 * math.sin(ang)) * rad
            ring.append(bm.verts.new(world_to_blender(p)))
        rings.append(ring)
    bm.faces.new(rings[0][::-1])
    for r0, r1 in zip(rings, rings[1:]):
        for s in range(sides):
            sn = (s + 1) % sides
            bm.faces.new([r0[s], r0[sn], r1[sn], r1[s]])
    bm.faces.new(rings[-1])


def add_bm_leaf_card(
    bm: bmesh.types.BMesh,
    uv_layer: Any,
    center_w: np.ndarray,
    width: float,
    height: float,
    yaw: float,
    pitch: float,
    roll: float = 0.0,
) -> None:
    """Append a single crossed/tilted botanical leaf card quad with clean [0..1] UVs."""
    hw, hh = width * 0.5, height * 0.5
    # Slight fold across vertical midline (+0.12*width Z curve) so cards never look paper-flat edge-on
    local = np.array([[-hw, -hh, 0.08 * hw], [0.0, -hh, -0.08 * hw], [hw, -hh, 0.08 * hw], [-hw, hh, 0.08 * hw], [0.0, hh, -0.08 * hw], [hw, hh, 0.08 * hw]], dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    r_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    world = (local @ r_z.T @ r_x.T @ r_y.T) + np.asarray(center_w, dtype=np.float64)
    v = [bm.verts.new(world_to_blender(p)) for p in world]
    f1 = bm.faces.new([v[0], v[1], v[4], v[3]])
    for loop, uv in zip(f1.loops, [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]):
        loop[uv_layer].uv = uv
    f2 = bm.faces.new([v[1], v[2], v[5], v[4]])
    for loop, uv in zip(f2.loops, [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]):
        loop[uv_layer].uv = uv


def add_bm_sedan(
    bm_paint: bmesh.types.BMesh,
    bm_glass: bmesh.types.BMesh,
    bm_wheels: bmesh.types.BMesh,
    bm_trim: bmesh.types.BMesh,
    bm_lights: bmesh.types.BMesh,
    bm_taillights: bmesh.types.BMesh,
    base_c: np.ndarray,
    yaw: float,
) -> None:
    """Build a sculpted 3-station hatchback/sedan silhouette instead of raw stacked boxes."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    f_dir = np.array([cy, 0.0, -sy], dtype=np.float64)
    r_dir = np.array([sy, 0.0, cy], dtype=np.float64)

    def to_w(lx: float, ly: float, lz: float) -> np.ndarray:
        return base_c + f_dir * lx + np.array([0.0, ly, 0.0]) + r_dir * lz

    # 6 longitudinal loft stations: x in [-2.22 .. +2.22], tapered front bumper -> hood -> cowl -> roof -> trunk -> tail
    stations = [
        (-2.22, 0.24, 0.62, 0.78),
        (-2.10, 0.22, 0.76, 0.86),
        (-0.88, 0.22, 0.88, 0.88),
        (1.05, 0.22, 0.90, 0.88),
        (2.08, 0.23, 0.80, 0.85),
        (2.22, 0.26, 0.66, 0.76),
    ]
    rings = []
    for lx, y_lo, y_hi, hw in stations:
        rings.append([
            bm_paint.verts.new(world_to_blender(to_w(lx, y_lo, -hw))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_lo, hw))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_hi, hw * 0.94))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_hi, -hw * 0.94))),
        ])
    bm_paint.faces.new(rings[0][::-1])
    for r0, r1 in zip(rings, rings[1:]):
        for k in range(4):
            kn = (k + 1) % 4
            bm_paint.faces.new([r0[k], r0[kn], r1[kn], r1[k]])
    bm_paint.faces.new(rings[-1])

    # Sculpted greenhouse cabin (sloped windshield + arched roof + sloped rear glass)
    cab_stations = [
        (-0.92, 0.84, 0.86, 0.78),
        (-0.32, 0.84, 1.38, 0.68),
        (0.88, 0.84, 1.36, 0.68),
        (1.52, 0.84, 0.88, 0.76),
    ]
    c_rings = []
    for lx, y_lo, y_hi, hw in cab_stations:
        c_rings.append([
            bm_paint.verts.new(world_to_blender(to_w(lx, y_lo, -hw))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_lo, hw))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_hi, hw * 0.92))),
            bm_paint.verts.new(world_to_blender(to_w(lx, y_hi, -hw * 0.92))),
        ])
    bm_paint.faces.new(c_rings[0][::-1])
    for r0, r1 in zip(c_rings, c_rings[1:]):
        for k in range(4):
            kn = (k + 1) % 4
            bm_paint.faces.new([r0[k], r0[kn], r1[kn], r1[k]])
    bm_paint.faces.new(c_rings[-1])

    # Slightly inset dark window band overlapping cabin sides, windshield, and rear glass
    w_rings = []
    for lx, y_lo, y_hi, hw in [(-0.85, 0.88, 0.91, 0.77), (-0.30, 0.88, 1.32, 0.695), (0.85, 0.88, 1.30, 0.695), (1.44, 0.88, 0.91, 0.75)]:
        w_rings.append([
            bm_glass.verts.new(world_to_blender(to_w(lx, y_lo, -hw))),
            bm_glass.verts.new(world_to_blender(to_w(lx, y_lo, hw))),
            bm_glass.verts.new(world_to_blender(to_w(lx, y_hi, hw * 0.93))),
            bm_glass.verts.new(world_to_blender(to_w(lx, y_hi, -hw * 0.93))),
        ])
    bm_glass.faces.new(w_rings[0][::-1])
    for r0, r1 in zip(w_rings, w_rings[1:]):
        for k in range(4):
            kn = (k + 1) % 4
            bm_glass.faces.new([r0[k], r0[kn], r1[kn], r1[k]])
    bm_glass.faces.new(w_rings[-1])

    # Headlights, grille, taillights & 4 wheels with alloy rims
    for side in (-1.0, 1.0):
        add_bm_box(bm_lights, to_w(2.14, 0.56, side * 0.58), np.array([0.14, 0.16, 0.28]), rot_yaw_w=yaw)
        add_bm_box(bm_taillights, to_w(-2.14, 0.56, side * 0.56), np.array([0.12, 0.14, 0.30]), rot_yaw_w=yaw)
    add_bm_box(bm_trim, to_w(2.16, 0.44, 0.0), np.array([0.10, 0.20, 0.72]), rot_yaw_w=yaw)

    for wx in (-1.34, 1.36):
        for wz in (-0.82, 0.82):
            w_pos = base_c + f_dir * wx + r_dir * wz + np.array([0.0, 0.31, 0.0])
            add_bm_tube(bm_wheels, [w_pos - r_dir * 0.11, w_pos + r_dir * 0.11], [0.31, 0.31], sides=12)
            out_dir = r_dir * (1.0 if wz > 0 else -1.0)
            add_bm_tube(bm_trim, [w_pos + out_dir * 0.08, w_pos + out_dir * 0.12], [0.18, 0.18], sides=10)


def finalize_bmesh_object(
    name: str,
    bm: bmesh.types.BMesh,
    mat: bpy.types.Material,
    col: bpy.types.Collection,
    *,
    smooth: bool = False,
    skip_auto_uv: bool = False,
    force_up_normals: bool = False,
) -> bpy.types.Object:
    if force_up_normals:
        bm.normal_update()
        down_faces = [f for f in bm.faces if f.normal.z < 0.0]
        if down_faces:
            bmesh.ops.reverse_faces(bm, faces=down_faces)
        bm.normal_update()
    else:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    mesh.materials.append(mat)
    if not skip_auto_uv and not mesh.uv_layers:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        verts = mesh.vertices
        for poly in mesh.polygons:
            n = poly.normal
            ax, ay, az = abs(n.x), abs(n.y), abs(n.z)
            for loop_idx in poly.loop_indices:
                vco = verts[mesh.loops[loop_idx].vertex_index].co
                if az >= ax and az >= ay:
                    uv_layer.data[loop_idx].uv = (vco.x * 0.25, vco.y * 0.25)
                elif ax >= ay:
                    uv_layer.data[loop_idx].uv = (vco.y * 0.25, vco.z * 0.25)
                else:
                    uv_layer.data[loop_idx].uv = (vco.x * 0.25, vco.z * 0.25)
    ob = bpy.data.objects.new(name, mesh)
    link_only(ob, col)
    return ob


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
            mesh_tmp = bpy.data.meshes.new("_bvh_vis_tmp")
            bm_sub.to_mesh(mesh_tmp)
            bm_sub.free()
            bm_all.from_mesh(mesh_tmp)
            bpy.data.meshes.remove(mesh_tmp)
    bvh = BVHTree.FromBMesh(bm_all)
    bm_all.free()
    return bvh


def _compute_photo_wall_score(
    bvh: BVHTree,
    origin: np.ndarray,
    u_span: np.ndarray,
    v_span: np.ndarray,
    nrm_w: np.ndarray,
    photo_cams: list[dict[str, Any]],
    *,
    max_dist: float = 75.0,
    min_cos: float = 0.70,
) -> float:
    """Fraction of 3x3 wall probes crisply observed unoccluded by CAM_ref."""
    if not photo_cams:
        return 0.0
    nrm_b = Vector(world_to_blender(nrm_w)).normalized()
    visible_probes = 0
    total_probes = 9
    for su in (0.2, 0.5, 0.8):
        for sv in (0.2, 0.5, 0.8):
            pw = origin + su * u_span + sv * v_span
            pb = Vector(world_to_blender(pw))
            start_b = pb + nrm_b * 0.24
            ok_cam = False
            for cam in photo_cams:
                eye_b = Vector(cam["blender_eye"])
                delta = eye_b - pb
                dist = delta.length
                if dist > max_dist or dist < 1.5:
                    continue
                d_b = delta / dist
                if d_b.dot(nrm_b) < min_cos:
                    continue
                r_cw = np.asarray(cam["r_cw"], dtype=np.float64)
                t_cw = np.asarray(cam["t_cw"], dtype=np.float64)
                xc = r_cw @ pw + t_cw
                if xc[2] <= 2.0:
                    continue
                u_px = xc[0] / xc[2] * cam["fx"] + cam["cx"]
                v_px = xc[1] / xc[2] * cam["fy"] + cam["cy"]
                w_c, h_c = cam["width"], cam["height"]
                if not (10.0 <= u_px <= w_c - 10.0 and 10.0 <= v_px <= h_c - 10.0):
                    continue
                wm = cam["watermark_box"]
                if u_px > wm[0] * w_c and v_px > wm[1] * h_c:
                    continue
                hit_loc, _, _, _ = bvh.ray_cast(start_b, d_b, dist - 0.35)
                if hit_loc is None:
                    ok_cam = True
                    break
            if ok_cam:
                visible_probes += 1
    return visible_probes / total_probes


def build_village(spec_path: Path, blend_out: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    clear_scene()
    scene = bpy.context.scene

    col_guide = get_or_create_collection("COL_Ref_Guide", hide_render=True)
    col_cams = get_or_create_collection("COL_Cameras")
    col_terrain = get_or_create_collection("COL_Terrain_Road")
    col_buildings = get_or_create_collection("COL_Buildings_Main")
    col_details = get_or_create_collection("COL_Building_Details")
    col_yards = get_or_create_collection("COL_Courtyards_Walls")
    col_props = get_or_create_collection("COL_Street_Props")
    col_foliage = get_or_create_collection("COL_Foliage")

    # 1. COL_Ref_Guide (COLMAP aligned sparse point cloud for GUI reference only)
    guide_raw = str(spec.get("colmap_guide_npz") or "").strip()
    guide_npz = Path(guide_raw) if guide_raw else None
    if guide_npz is not None and guide_npz.is_file():
        npz = np.load(guide_npz)
        pts_mesh = bpy.data.meshes.new("COLMAP_Sparse_Points")
        pts_mesh.from_pydata([world_to_blender(v) for v in npz["xyz"]], [], [])
        pts_obj = bpy.data.objects.new("COLMAP_Sparse_Points", pts_mesh)
        pts_obj.hide_render = True
        pts_obj["guide_only"] = True
        link_only(pts_obj, col_guide)

    # 2. COL_Cameras
    cams_spec = spec["cameras"]
    ref_cam_spec = next((c for c in cams_spec if c["source_clip"] == "ref"), cams_spec[0])
    chosen_cams = [ref_cam_spec]
    for cspec in cams_spec:
        if cspec is not ref_cam_spec and cspec.get("allow_facade_bake", False):
            chosen_cams.append(cspec)
    for clip_prefix in ("clip1", "clip3"):
        cands = [c for c in cams_spec if clip_prefix in c["source_clip"]]
        if cands:
            chosen_cams.append(cands[len(cands) // 2])
            if len(cands) >= 3:
                chosen_cams.append(cands[-1])

    for cspec in chosen_cams:
        cname = "CAM_ref" if cspec["source_clip"] == "ref" else f"CAM_{cspec['cam_key']}"
        intr = cspec["blender_intrinsics"]
        cd = bpy.data.cameras.new(cname)
        cd.type = "PERSP"
        cd.sensor_fit = intr["sensor_fit"]
        cd.sensor_width = intr["sensor_width"]
        cd.lens = intr["lens"]
        cd.shift_x = intr["shift_x"]
        cd.shift_y = intr["shift_y"]
        cd.clip_start = 0.2
        cd.clip_end = 1500.0
        img_p = Path(cspec["image_path"])
        if img_p.is_file():
            bg = cd.background_images.new()
            bg.image = bpy.data.images.load(str(img_p), check_existing=True)
            bg.alpha = 0.35
            bg.display_depth = "BACK"
            cd.show_background_images = True
        co = bpy.data.objects.new(cname, cd)
        link_only(co, col_cams)
        co.matrix_world = Matrix(cspec["blender_mat4"])
        if cspec["source_clip"] == "ref":
            scene.camera = co
            scene.render.resolution_x = intr["resolution_x"]
            scene.render.resolution_y = intr["resolution_y"]
            scene.render.pixel_aspect_x = intr["pixel_aspect_x"]
            scene.render.pixel_aspect_y = intr["pixel_aspect_y"]

    # 3. Materials from real CC0 Poly Haven manifest
    sw = spec["swatches"]
    fol = spec["foliage_swatches"]
    mat_ground = make_pbr_material(
        "Ground_Farmland_Road", roughness=0.92, specular=0.18,
        albedo_path=sw["farmland_ground"]["albedo"], rough_path=sw["farmland_ground"]["roughness"],
    )
    # Add fine tiled ground micro-normal relief without scaling farmland albedo UVs
    g_norm_src = sw.get("concrete_apron", {}).get("normal") or sw.get("concrete_roof", {}).get("normal")
    if g_norm_src:
        g_nt = mat_ground.node_tree
        g_bsdf = next(n for n in g_nt.nodes if n.type == "BSDF_PRINCIPLED")
        g_tc = g_nt.nodes.new("ShaderNodeTexCoord")
        g_mp = g_nt.nodes.new("ShaderNodeMapping")
        g_mp.inputs["Scale"].default_value = (42.0, 42.0, 1.0)
        g_nt.links.new(g_tc.outputs["UV"], g_mp.inputs["Vector"])
        g_tn = g_nt.nodes.new("ShaderNodeTexImage")
        g_im = bpy.data.images.load(g_norm_src, check_existing=True)
        g_im.colorspace_settings.name = "Non-Color"
        g_tn.image = g_im
        g_nt.links.new(g_mp.outputs["Vector"], g_tn.inputs["Vector"])
        g_nm = g_nt.nodes.new("ShaderNodeNormalMap")
        g_nm.inputs["Strength"].default_value = 0.44
        g_nt.links.new(g_tn.outputs["Color"], g_nm.inputs["Color"])
        g_nt.links.new(g_nm.outputs["Normal"], g_bsdf.inputs["Normal"])
    mat_road = make_pbr_material(
        "Road_Asphalt_PBR", roughness=0.80, specular=0.28,
        albedo_path=sw["road_asphalt_marked"]["albedo"],
        normal_path=sw["road_asphalt_marked"]["normal"],
        rough_path=sw["road_asphalt_marked"]["roughness"],
        normal_strength=0.95,
    )
    mat_apron = make_pbr_material(
        "Road_Concrete_Apron_PBR", roughness=0.86, specular=0.22,
        albedo_path=sw["concrete_apron"]["albedo"],
        normal_path=sw["concrete_apron"]["normal"],
        rough_path=sw["concrete_apron"]["roughness"],
        normal_strength=0.82,
    )
    mat_earth = make_pbr_material("Earth_Foundation", base_color=(0.18, 0.16, 0.12, 1.0), roughness=0.95)
    mat_concrete = make_pbr_material(
        "Detail_ConcreteTrim", roughness=0.86, specular=0.25,
        albedo_path=sw["plaster_warm"]["albedo"], normal_path=sw["plaster_warm"]["normal"], rough_path=sw["plaster_warm"]["roughness"], uv_scale=0.6,
    )
    mat_plaster_wall = make_pbr_material(
        "Detail_PlasterGrey", roughness=0.88, specular=0.22,
        albedo_path=sw["plaster_grey"]["albedo"], normal_path=sw["plaster_grey"]["normal"], rough_path=sw["plaster_grey"]["roughness"], uv_scale=0.4,
    )
    mat_brick = make_pbr_material(
        "Detail_Brick", roughness=0.90, specular=0.18,
        albedo_path=sw["red_brick"]["albedo"], normal_path=sw["red_brick"]["normal"], rough_path=sw["red_brick"]["roughness"], uv_scale=0.45, normal_strength=0.9,
    )
    mat_roof_terracotta = make_pbr_material(
        "Detail_RoofTerracotta", roughness=0.84, specular=0.22,
        albedo_path=sw["terracotta_tile"]["albedo"], normal_path=sw["terracotta_tile"]["normal"], rough_path=sw["terracotta_tile"]["roughness"], uv_scale=0.78, normal_strength=0.92,
    )
    slate_sw = sw.get("slate_roof") or sw["concrete_roof"]
    mat_roof_slate = make_pbr_material(
        "Detail_RoofSlate", roughness=0.82, specular=0.24,
        albedo_path=slate_sw["albedo"], normal_path=slate_sw["normal"], rough_path=slate_sw["roughness"], uv_scale=0.85, normal_strength=0.88,
    )
    mat_roof_silver = make_pbr_material(
        "Detail_RoofSilver", roughness=0.48, metallic=0.36, specular=0.42,
        albedo_path=sw["corrugated_silver"]["albedo"], normal_path=sw["corrugated_silver"]["normal"], rough_path=sw["corrugated_silver"]["roughness"], uv_scale=1.15, normal_strength=0.95,
    )
    mat_roof_blue = make_pbr_material(
        "Detail_RoofBlue", roughness=0.54, metallic=0.22, specular=0.36,
        albedo_path=sw["corrugated_blue"]["albedo"], normal_path=sw["corrugated_blue"]["normal"], rough_path=sw["corrugated_blue"]["roughness"], uv_scale=1.15, normal_strength=0.95,
    )
    mat_window_glass = make_pbr_material(
        "Detail_WindowGlass", roughness=0.14, specular=0.72,
        albedo_path=sw["window_pane_photo"]["albedo"], normal_path=sw["window_pane_photo"]["normal"], rough_path=sw["window_pane_photo"]["roughness"], normal_strength=0.85,
    )
    mat_window_frame = make_pbr_material("Detail_WindowFrame", base_color=(0.56, 0.57, 0.58, 1.0), roughness=0.38, metallic=0.60, specular=0.52)
    mat_shutter = make_pbr_material(
        "Detail_ShopShutter", roughness=0.46, metallic=0.52, specular=0.50,
        albedo_path=sw["shop_shutter"]["albedo"], normal_path=sw["shop_shutter"]["normal"], rough_path=sw["shop_shutter"]["roughness"], uv_scale=0.5, normal_strength=0.95,
    )
    horizon_rgb = spec.get("horizon_color_rgb") or [0.165, 0.172, 0.095]
    mat_horizon = make_pbr_material("Horizon_Meadow_Skirt", base_color=(float(horizon_rgb[0]), float(horizon_rgb[1]), float(horizon_rgb[2]), 1.0), roughness=0.96, specular=0.05)
    mat_solar_tube = make_pbr_material("Detail_SolarCollector", base_color=(0.018, 0.024, 0.036, 1.0), roughness=0.26, specular=0.42)
    mat_solar_tank = make_pbr_material("Detail_SolarTank", base_color=(0.68, 0.67, 0.64, 1.0), roughness=0.38, metallic=0.55, specular=0.5)
    mat_steel = make_pbr_material("Detail_Steel", base_color=(0.24, 0.25, 0.25, 1.0), roughness=0.55, metallic=0.58)
    mat_iron_gate = make_pbr_material("Detail_IronGate", base_color=(0.30, 0.11, 0.09, 1.0), roughness=0.58, metallic=0.38)
    sign_red_sw = sw.get("shop_sign_red", {})
    sign_blue_sw = sw.get("shop_sign_blue", {})
    mat_sign_red = make_pbr_material(
        "Detail_ShopSign_Red", base_color=(0.46, 0.13, 0.11, 1.0), roughness=0.58, specular=0.32,
        albedo_path=sign_red_sw.get("albedo"), normal_path=sign_red_sw.get("normal"), rough_path=sign_red_sw.get("roughness"), normal_strength=0.65,
    )
    mat_sign_blue = make_pbr_material(
        "Detail_ShopSign_Blue", base_color=(0.14, 0.27, 0.42, 1.0), roughness=0.58, specular=0.32,
        albedo_path=sign_blue_sw.get("albedo"), normal_path=sign_blue_sw.get("normal"), rough_path=sign_blue_sw.get("roughness"), normal_strength=0.65,
    )
    mat_pole = make_pbr_material("Detail_PoleConcrete", base_color=(0.64, 0.62, 0.58, 1.0), roughness=0.85)
    mat_bark = make_pbr_material(
        "Tree_Bark", roughness=0.92,
        albedo_path=sw["tree_bark"]["albedo"], normal_path=sw["tree_bark"]["normal"], rough_path=sw["tree_bark"]["roughness"], uv_scale=0.8,
    )
    mat_leaf_poplar = make_alpha_mask_material(
        "Tree_Leaf_Poplar", rgba_path=fol["leaf_poplar"]["rgba"], normal_path=fol["leaf_poplar"]["normal"], rough_path=fol["leaf_poplar"]["roughness"],
    )
    mat_leaf_broadleaf = make_alpha_mask_material(
        "Tree_Leaf_Broadleaf", rgba_path=fol["leaf_broadleaf"]["rgba"], normal_path=fol["leaf_broadleaf"]["normal"], rough_path=fol["leaf_broadleaf"]["roughness"],
    )
    mat_shrub = make_alpha_mask_material(
        "Foliage_Shrub", rgba_path=fol["shrub"]["rgba"], normal_path=fol["shrub"]["normal"], rough_path=fol["shrub"]["roughness"],
    )
    car_mats = {
        "white": make_pbr_material("Car_Paint_White", base_color=(0.68, 0.67, 0.64, 1.0), roughness=0.28, specular=0.68),
        "silver": make_pbr_material("Car_Paint_Silver", base_color=(0.56, 0.58, 0.62, 1.0), roughness=0.24, metallic=0.55, specular=0.78),
        "charcoal": make_pbr_material("Car_Paint_Charcoal", base_color=(0.11, 0.12, 0.14, 1.0), roughness=0.24, metallic=0.35, specular=0.72),
        "blue": make_pbr_material("Car_Paint_Blue", base_color=(0.12, 0.25, 0.42, 1.0), roughness=0.38, metallic=0.2, specular=0.55),
    }
    mat_car_glass = make_pbr_material("Car_Glass", base_color=(0.03, 0.04, 0.055, 1.0), roughness=0.10, specular=0.85)
    mat_car_headlight = make_pbr_material("Car_Headlight", base_color=(0.92, 0.90, 0.82, 1.0), roughness=0.14, specular=0.8)
    mat_car_taillight = make_pbr_material("Car_Taillight", base_color=(0.68, 0.05, 0.05, 1.0), roughness=0.20, specular=0.7)
    mat_tire = make_pbr_material("Car_Tire", base_color=(0.055, 0.055, 0.058, 1.0), roughness=0.86)

    # Dynamic multi-tier atlas count: Atlases 0..3 for 64 core buildings, Atlases 4+ for dense infill (25 per atlas)
    core_list = [b for b in spec["buildings"] if b.get("is_core", True)]
    infill_list = [b for b in spec["buildings"] if not b.get("is_core", True)]
    core_atlases = max(1, math.ceil(len(core_list) / CORE_PER_ATLAS))
    infill_atlases = math.ceil(len(infill_list) / INFILL_PER_ATLAS)
    num_atlases = core_atlases + infill_atlases

    atlas_mats = [
        make_pbr_material(f"VillageAtlas_{i}", roughness=0.84, specular=0.28, albedo_path=sw["plaster_white"]["albedo"])
        for i in range(num_atlases)
    ]

    # 4. COL_Terrain_Road: global farmland ground + dedicated high-res tiled asphalt road + concrete aprons + curbs
    bx0, bx1, bz0, bz1 = spec["groundBounds"]
    bm_g = bmesh.new()
    nx_g, nz_g = 24, 24
    grid_v = []
    for j in range(nz_g + 1):
        z = bz0 + (bz1 - bz0) * (j / nz_g)
        grid_v.append([bm_g.verts.new(world_to_blender((bx0 + (bx1 - bx0) * (i / nx_g), -0.02, z))) for i in range(nx_g + 1)])
    uv_g = bm_g.loops.layers.uv.new("UVMap")
    for j in range(nz_g):
        for i in range(nx_g):
            f = bm_g.faces.new([grid_v[j][i], grid_v[j + 1][i], grid_v[j + 1][i + 1], grid_v[j][i + 1]])
            coords = [
                (i / nx_g, 1.0 - j / nz_g),
                (i / nx_g, 1.0 - (j + 1) / nz_g),
                ((i + 1) / nx_g, 1.0 - (j + 1) / nz_g),
                ((i + 1) / nx_g, 1.0 - j / nz_g),
            ]
            for loop, uv_pt in zip(f.loops, coords):
                loop[uv_g].uv = uv_pt
    finalize_bmesh_object("village_ground", bm_g, mat_ground, col_terrain, skip_auto_uv=True, force_up_normals=True)

    bm_earth = bmesh.new()
    add_bm_box(bm_earth, np.array([(bx0 + bx1) * 0.5, -1.05, (bz0 + bz1) * 0.5]), np.array([bx1 - bx0, 2.0, bz1 - bz0]))
    finalize_bmesh_object("earth_foundation", bm_earth, mat_earth, col_terrain)

    bm_horizon = bmesh.new()
    add_bm_box(bm_horizon, np.array([(bx0 + bx1) * 0.5, -0.55, (bz0 + bz1) * 0.5]), np.array([1500.0, 0.96, 1500.0]))
    finalize_bmesh_object("horizon_meadow_skirt", bm_horizon, mat_horizon, col_terrain)

    # Dedicated high-res tiled Road_Asphalt_Carriageway & Road_Concrete_Aprons!
    left_arr = np.asarray(spec.get("road_left_edge_xz") or [], dtype=np.float64)
    right_arr = np.asarray(spec.get("road_right_edge_xz") or [], dtype=np.float64)
    if len(left_arr) >= 2 and len(right_arr) >= 2:
        dl0 = (left_arr[0] - left_arr[1]) / max(float(np.linalg.norm(left_arr[0] - left_arr[1])), 1e-6)
        dl1 = (left_arr[-1] - left_arr[-2]) / max(float(np.linalg.norm(left_arr[-1] - left_arr[-2])), 1e-6)
        dr0 = (right_arr[0] - right_arr[1]) / max(float(np.linalg.norm(right_arr[0] - right_arr[1])), 1e-6)
        dr1 = (right_arr[-1] - right_arr[-2]) / max(float(np.linalg.norm(right_arr[-1] - right_arr[-2])), 1e-6)
        road_ext_m = float(spec.get("road_extend_m", 145.0))
        ext_l = np.vstack([left_arr[0] + dl0 * road_ext_m, left_arr, left_arr[-1] + dl1 * road_ext_m])
        ext_r = np.vstack([right_arr[0] + dr0 * road_ext_m, right_arr, right_arr[-1] + dr1 * road_ext_m])

        # Ensure matching station count (resample uniformly into 28 clean longitudinal strips)
        n_road_seg = 28
        def _resample_polyline(pts: np.ndarray, n_out: int) -> np.ndarray:
            seg_l = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg_l)])
            ts = np.linspace(0.0, cum[-1], n_out)
            return np.stack([np.interp(ts, cum, pts[:, 0]), np.interp(ts, cum, pts[:, 1])], axis=-1)

        rs_l = _resample_polyline(ext_l, n_road_seg + 1)
        rs_r = _resample_polyline(ext_r, n_road_seg + 1)
        cum_dist = 0.0
        bm_road = bmesh.new()
        uv_r = bm_road.loops.layers.uv.new("UVMap")
        road_verts_l = [bm_road.verts.new(world_to_blender((p[0], 0.015, p[1]))) for p in rs_l]
        road_verts_r = [bm_road.verts.new(world_to_blender((p[0], 0.015, p[1]))) for p in rs_r]
        v_accum = [0.0]
        for idx_s in range(n_road_seg):
            step_len = float(np.linalg.norm(0.5 * (rs_l[idx_s + 1] + rs_r[idx_s + 1]) - 0.5 * (rs_l[idx_s] + rs_r[idx_s])))
            cum_dist += step_len
            v_accum.append(cum_dist / 6.5)
        for idx_s in range(n_road_seg):
            f = bm_road.faces.new([road_verts_l[idx_s], road_verts_r[idx_s], road_verts_r[idx_s + 1], road_verts_l[idx_s + 1]])
            v0, v1 = v_accum[idx_s], v_accum[idx_s + 1]
            for loop, uv_pt in zip(f.loops, [(0.0, v0), (1.0, v0), (1.0, v1), (0.0, v1)]):
                loop[uv_r].uv = uv_pt
        finalize_bmesh_object("Road_Asphalt_Carriageway", bm_road, mat_road, col_terrain, skip_auto_uv=True, force_up_normals=True)

        # Poured concrete shoulder aprons flanking both road sides (4.6m wide outside curbstones)
        apron_w_m = float(spec.get("road_apron_width_m", 3.2))
        bm_apron = bmesh.new()
        uv_ap = bm_apron.loops.layers.uv.new("UVMap")
        for side_edge, opp_edge in ((rs_l, rs_r), (rs_r, rs_l)):
            out_pts = []
            for pa, pb in zip(side_edge, opp_edge):
                out_d = pa - pb
                out_d /= max(float(np.linalg.norm(out_d)), 1e-6)
                out_pts.append(pa + out_d * apron_w_m)
            v_in = [bm_apron.verts.new(world_to_blender((p[0], 0.002, p[1]))) for p in side_edge]
            v_out = [bm_apron.verts.new(world_to_blender((p[0], 0.002, p[1]))) for p in out_pts]
            for idx_s in range(n_road_seg):
                f = bm_apron.faces.new([v_in[idx_s], v_out[idx_s], v_out[idx_s + 1], v_in[idx_s + 1]])
                v0, v1 = v_accum[idx_s] * 1.3, v_accum[idx_s + 1] * 1.3
                for loop, uv_pt in zip(f.loops, [(0.0, v0), (0.9, v0), (0.9, v1), (0.0, v1)]):
                    loop[uv_ap].uv = uv_pt
        finalize_bmesh_object("Road_Concrete_Aprons", bm_apron, mat_apron, col_terrain, skip_auto_uv=True, force_up_normals=True)

        bm_curb = bmesh.new()
        for pa, pb in zip(ext_l, ext_l[1:]):
            add_bm_beam(bm_curb, np.array([pa[0], 0.0, pa[1]]), np.array([pb[0], 0.0, pb[1]]), thickness=0.18, height=0.05, y_base=-0.015)
        for pa, pb in zip(ext_r, ext_r[1:]):
            add_bm_beam(bm_curb, np.array([pa[0], 0.0, pa[1]]), np.array([pb[0], 0.0, pb[1]]), thickness=0.18, height=0.05, y_base=-0.015)
        finalize_bmesh_object("Road_Curbs", bm_curb, mat_concrete, col_terrain)

    # 5. COL_Courtyards_Walls (built BEFORE BVH photo_facade gating so brick courtyard walls properly occlude rear houses)
    bm_yard_brick = bmesh.new()
    bm_yard_plaster = bmesh.new()
    bm_gate_iron = bmesh.new()
    bm_gate_roof = bmesh.new()
    bm_trim_conc = bmesh.new()
    bm_trim_brick = bmesh.new()

    for cw in spec.get("courtyard_walls", []):
        pa = np.array([cw["a_xz"][0], 0.0, cw["a_xz"][1]], dtype=np.float64)
        pb = np.array([cw["b_xz"][0], 0.0, cw["b_xz"][1]], dtype=np.float64)
        wh = float(cw.get("height_m", 2.3))
        tgt_bm = bm_yard_brick if cw.get("swatch", "red_brick") == "red_brick" else bm_yard_plaster
        add_bm_beam(tgt_bm, pa, pb, thickness=0.25, height=wh, y_base=0.0)
        add_bm_beam(bm_trim_conc, pa, pb, thickness=0.33, height=0.08, y_base=wh)
        add_bm_box(tgt_bm, pa + np.array([0.0, wh * 0.52, 0.0]), np.array([0.38, wh * 1.04, 0.38]))
        add_bm_box(tgt_bm, pb + np.array([0.0, wh * 0.52, 0.0]), np.array([0.38, wh * 1.04, 0.38]))

    for cg in spec.get("courtyard_gates", []):
        cx, cz = cg["center_xz"]
        yaw = float(cg["yaw_rad"])
        gw, gh = float(cg["width_m"]), float(cg["height_m"])
        u_g = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
        ctr = np.array([cx, 0.0, cz], dtype=np.float64)
        # Gate brick pillars
        for side in (-0.5, 0.5):
            add_bm_box(bm_yard_brick, ctr + u_g * (side * gw) + np.array([0.0, gh * 0.5, 0.0]), np.array([0.46, gh, 0.46]), rot_yaw_w=yaw)
            add_bm_box(bm_trim_conc, ctr + u_g * (side * gw) + np.array([0.0, gh + 0.05, 0.0]), np.array([0.56, 0.10, 0.56]), rot_yaw_w=yaw)
        # Over-gate lintel canopy (Menlou portal header)
        add_bm_box(bm_yard_brick, ctr + np.array([0.0, gh - 0.16, 0.0]), np.array([gw + 0.55, 0.32, 0.50]), rot_yaw_w=yaw)
        add_bm_box(bm_gate_roof, ctr + np.array([0.0, gh + 0.09, 0.0]), np.array([gw + 0.85, 0.16, 0.92]), rot_yaw_w=yaw)
        # Dark iron double gate leaves
        add_bm_box(bm_gate_iron, ctr + u_g * (-gw * 0.24) + np.array([0.0, 1.05, 0.0]), np.array([gw * 0.46, 2.02, 0.06]), rot_yaw_w=yaw)
        add_bm_box(bm_gate_iron, ctr + u_g * (gw * 0.24) + np.array([0.0, 1.05, 0.0]), np.array([gw * 0.46, 2.02, 0.06]), rot_yaw_w=yaw)

    if len(bm_yard_brick.verts) > 0:
        finalize_bmesh_object("Courtyard_Brick_Walls", bm_yard_brick, mat_brick, col_yards)
    else:
        bm_yard_brick.free()
    if len(bm_yard_plaster.verts) > 0:
        finalize_bmesh_object("Courtyard_Plaster_Walls", bm_yard_plaster, mat_plaster_wall, col_yards)
    else:
        bm_yard_plaster.free()
    if len(bm_gate_iron.verts) > 0:
        finalize_bmesh_object("Courtyard_Iron_Gates", bm_gate_iron, mat_iron_gate, col_yards)
    else:
        bm_gate_iron.free()
    if len(bm_gate_roof.verts) > 0:
        finalize_bmesh_object("Courtyard_Gate_Canopies", bm_gate_roof, mat_roof_terracotta, col_yards)
    else:
        bm_gate_roof.free()

    # 6. COL_Buildings_Main (build ALL 110+ closed watertight buildings FIRST so BVH is complete)
    bm_roof_tile = bmesh.new()
    bm_roof_silv = bmesh.new()
    bm_roof_blue = bmesh.new()
    bm_roof_slate = bmesh.new()

    uv_atlas_map: dict[str, Any] = {"atlas_size": ATLAS_SIZE, "atlases": num_atlases, "faces": []}
    building_wall_meta: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    infill_counter = 0

    for b in spec["buildings"]:
        idx = int(b["index"])
        bid = b["id"]
        is_core = bool(b.get("is_core", True))
        if is_core:
            atlas_idx = idx // CORE_PER_ATLAS
            slot = idx % CORE_PER_ATLAS
            cols_n, cell_sz = 4, 512
        else:
            atlas_idx = core_atlases + (infill_counter // INFILL_PER_ATLAS)
            slot = infill_counter % INFILL_PER_ATLAS
            cols_n, cell_sz = 6, 340
            infill_counter += 1

        col_cell = slot % cols_n
        row_cell = slot // cols_n
        cell_x0, cell_y0 = col_cell * cell_sz, row_cell * cell_sz

        top = np.asarray(b["points_3d"], dtype=np.float64)
        n = len(top)
        bot = top.copy()
        bot[:, 1] = 0.0
        rise = float(b["ridge_rise"])
        is_gable = rise > 0.05 and n == 4
        rects = pack_building_rects(is_gable, cell_x0, cell_y0, cell_size=cell_sz)

        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new("UVMap")
        bot_v = [bm.verts.new(world_to_blender(p)) for p in bot]
        top_v = [bm.verts.new(world_to_blender(p)) for p in top]

        face_records: list[tuple[bmesh.types.BMFace, np.ndarray, str, list[int], str, str, int]] = []
        f_bot = bm.faces.new(bot_v[::-1])
        face_records.append((f_bot, bot[::-1], "bottom", rects["bottom"], "bottom", "concrete_roof", -1))

        for ei in range(n):
            ej = (ei + 1) % n
            f_w = bm.faces.new([bot_v[ei], bot_v[ej], top_v[ej], top_v[ei]])
            w_pts = np.array([bot[ei], bot[ej], top[ej], top[ei]], dtype=np.float64)
            w_meta = b["walls"][ei]
            face_records.append((f_w, w_pts, "wall", rects[f"wall_{ei % 4}"], w_meta["role"], w_meta.get("swatch", b["wall_swatch"]), ei))

        if is_gable:
            edge_lens = np.linalg.norm(top - np.roll(top, -1, axis=0), axis=1)
            if edge_lens[0] + edge_lens[2] >= edge_lens[1] + edge_lens[3]:
                rot_top, rot_top_v = top, top_v
            else:
                rot_top = np.roll(top, -1, axis=0)
                rot_top_v = [top_v[(k + 1) % 4] for k in range(4)]
            a3, b3, c3, d3 = rot_top
            va, vb, vc, vd = rot_top_v
            r1 = 0.5 * (a3 + d3) + np.array([0.0, rise, 0.0])
            r2 = 0.5 * (b3 + c3) + np.array([0.0, rise, 0.0])
            vr1 = bm.verts.new(world_to_blender(r1))
            vr2 = bm.verts.new(world_to_blender(r2))
            f_r0 = bm.faces.new([va, vb, vr2, vr1])
            f_r1 = bm.faces.new([vr1, vr2, vc, vd])
            f_g0 = bm.faces.new([va, vr1, vd])
            f_g1 = bm.faces.new([vb, vc, vr2])
            face_records.append((f_r0, np.array([a3, b3, r2, r1]), "roof", rects["roof_0"], "roof", b["roof_swatch"], -1))
            face_records.append((f_r1, np.array([r1, r2, c3, d3]), "roof", rects["roof_1"], "roof", b["roof_swatch"], -1))
            face_records.append((f_g0, np.array([a3, r1, d3]), "wall", rects["gable_0"], "gable_end", b.get("side_wall_swatch", b["wall_swatch"]), -1))
            face_records.append((f_g1, np.array([b3, c3, r2]), "wall", rects["gable_1"], "gable_end", b.get("side_wall_swatch", b["wall_swatch"]), -1))

            bm_eave = (
                bm_roof_tile
                if b["roof_swatch"] == "terracotta_tile"
                else (
                    bm_roof_blue
                    if b["roof_swatch"] == "corrugated_blue"
                    else (bm_roof_slate if b["roof_swatch"] == "slate_roof" else bm_roof_silv)
                )
            )
            for slope_quad in ([a3, b3, r2, r1], [d3, c3, r2, r1]):
                sq = np.array(slope_quad, dtype=np.float64)
                e0, e1, p_top0, p_top1 = sq[0], sq[1], sq[3], sq[2]
                down_slope = 0.5 * (e0 + e1) - 0.5 * (p_top0 + p_top1)
                out_slope = down_slope / max(float(np.linalg.norm(down_slope)), 1e-6)
                ridge_dir = e1 - e0
                out_ridge = ridge_dir / max(float(np.linalg.norm(ridge_dir)), 1e-6)
                q_e0 = e0 + out_slope * 0.42 - out_ridge * 0.34
                q_e1 = e1 + out_slope * 0.42 + out_ridge * 0.34
                q_r0 = p_top0 - out_ridge * 0.34
                q_r1 = p_top1 + out_ridge * 0.34
                s_n = np.cross(q_e1 - q_e0, q_r0 - q_e0)
                if s_n[1] < 0:
                    s_n = -s_n
                s_n /= max(float(np.linalg.norm(s_n)), 1e-6)
                sv = [bm_eave.verts.new(world_to_blender(p)) for p in (q_e0 + s_n * 0.015, q_e1 + s_n * 0.015, q_r1 + s_n * 0.015, q_r0 + s_n * 0.015, q_e0 + s_n * 0.13, q_e1 + s_n * 0.13, q_r1 + s_n * 0.13, q_r0 + s_n * 0.13)]
                for ff in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
                    bm_eave.faces.new([sv[k] for k in ff])
            add_bm_prism_beam(bm_trim_conc, r1 - out_ridge * 0.36 + [0, 0.13, 0], r2 + out_ridge * 0.36 + [0, 0.13, 0], 0.24, 0.16)
        else:
            f_top = bm.faces.new(top_v)
            face_records.append((f_top, top, "roof", rects["roof_0"], "roof", b["roof_swatch"], -1))
            if b["archetype"] in ("blue_shed", "metal_shed") and n == 4:
                bm_shed = bm_roof_blue if b["roof_swatch"] == "corrugated_blue" else bm_roof_silv
                c_shed = top.mean(0)
                sh_b = [c_shed + (p - c_shed) * 1.09 + np.array([0.0, 0.015, 0.0]) for p in top]
                sh_t = [c_shed + (p - c_shed) * 1.09 + np.array([0.0, 0.11, 0.0]) for p in top]
                sh_t[2][1] += 0.18
                sh_t[3][1] += 0.18
                sv = [bm_shed.verts.new(world_to_blender(p)) for p in (sh_b + sh_t)]
                for ff in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
                    bm_shed.faces.new([sv[k] for k in ff])

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.faces.ensure_lookup_table()

        wall_recs_for_b = []
        for f_obj, f_pts, role, rect_px, wall_role, sw_name, edge_ei in face_records:
            rx0, ry0, rx1, ry1 = rect_px
            origin_3d, u_span_3d, v_span_3d, st_poly = compute_face_basis_st(f_pts, prefer_up_v=(role == "wall"))
            for loop, (su, sv) in zip(f_obj.loops, st_poly):
                loop[uv_layer].uv = ((rx0 + su * (rx1 - rx0)) / ATLAS_SIZE, 1.0 - (ry0 + sv * (ry1 - ry0)) / ATLAS_SIZE)
            n_bl = f_obj.normal
            nrm_w_out = np.array([float(n_bl.x), float(n_bl.z), float(-n_bl.y)], dtype=np.float64)
            nrm_w_out /= max(float(np.linalg.norm(nrm_w_out)), 1e-6)

            frec = {
                "building_id": bid,
                "is_core": is_core,
                "atlas_idx": atlas_idx,
                "role": role,
                "wall_role": wall_role,
                "edge_index": edge_ei,
                "swatch": sw_name,
                "observed_rgb": b["observed_roof_rgb"] if role == "roof" else b["observed_wall_rgb"],
                "rect_px": rect_px,
                "origin_3d": [round(float(x), 5) for x in origin_3d],
                "u_span_3d": [round(float(x), 5) for x in u_span_3d],
                "v_span_3d": [round(float(x), 5) for x in v_span_3d],
                "normal_world": [round(float(x), 5) for x in nrm_w_out],
                "st_poly": [[round(c[0], 5), round(c[1], 5)] for c in st_poly],
                "bake_mode": "pbr_procedural",
                "photo_wall_score": 0.0,
            }
            uv_atlas_map["faces"].append(frec)
            if edge_ei >= 0:
                wall_recs_for_b.append(frec)

        mesh = bpy.data.meshes.new(bid)
        bm.to_mesh(mesh)
        bm.free()
        mesh.materials.append(atlas_mats[atlas_idx])
        ob = bpy.data.objects.new(bid, mesh)
        ob["closed_building"] = True
        ob["check_road_clearance"] = True
        ob["is_core"] = is_core
        ob["archetype"] = b["archetype"]
        ob["stories"] = b["stories"]
        link_only(ob, col_buildings)
        building_wall_meta.append((b, wall_recs_for_b))

    # 7. BVH Visibility Gating: CAM_ref walls -> photo_facade (no 3D windows); synthetic novel walls -> novel_facade (keeps 3D windows + soft PBR transfer)
    bvh_vis = _build_collection_bvh(("COL_Buildings_Main", "COL_Courtyards_Walls"))
    ref_photo_cams = [c for c in cams_spec if c["source_clip"] == "ref"]
    novel_photo_cams = [c for c in cams_spec if c.get("allow_facade_bake", False)]
    facade_max_dist = float(spec.get("photo_facade_max_dist", 75.0))
    facade_min_cos = float(spec.get("photo_facade_min_cos", 0.70))
    n_photo_facades = 0
    for b, wall_recs in building_wall_meta:
        if not b.get("is_core", True):
            continue
        for frec in wall_recs:
            if frec["wall_role"] == "party_wall":
                continue
            ref_score = _compute_photo_wall_score(
                bvh_vis,
                np.asarray(frec["origin_3d"], dtype=np.float64),
                np.asarray(frec["u_span_3d"], dtype=np.float64),
                np.asarray(frec["v_span_3d"], dtype=np.float64),
                np.asarray(frec["normal_world"], dtype=np.float64),
                ref_photo_cams,
                max_dist=facade_max_dist,
                min_cos=facade_min_cos,
            )
            if ref_score >= 0.85:
                frec["photo_wall_score"] = round(ref_score, 3)
                frec["bake_mode"] = "photo_facade"
                n_photo_facades += 1
            elif novel_photo_cams:
                mv_score = _compute_photo_wall_score(
                    bvh_vis,
                    np.asarray(frec["origin_3d"], dtype=np.float64),
                    np.asarray(frec["u_span_3d"], dtype=np.float64),
                    np.asarray(frec["v_span_3d"], dtype=np.float64),
                    np.asarray(frec["normal_world"], dtype=np.float64),
                    novel_photo_cams,
                    max_dist=facade_max_dist,
                    min_cos=facade_min_cos,
                )
                frec["photo_wall_score"] = round(mv_score, 3)
                if mv_score >= 0.85:
                    frec["bake_mode"] = "novel_facade"
                    n_photo_facades += 1
            else:
                frec["photo_wall_score"] = round(ref_score, 3)

    # 8. COL_Building_Details: parapets, brick cornices, solar heaters, & NON-photo_facade realistic windows
    bm_win_glass = bmesh.new()
    uv_win_glass = bm_win_glass.loops.layers.uv.new("UVMap")
    bm_win_frame = bmesh.new()
    bm_shutter = bmesh.new()
    bm_solar_col = bmesh.new()
    bm_solar_tank = bmesh.new()
    bm_steel = bmesh.new()
    bm_sign_red = bmesh.new()
    uv_sign_red = bm_sign_red.loops.layers.uv.new("UVMap")
    bm_sign_blue = bmesh.new()
    uv_sign_blue = bm_sign_blue.loops.layers.uv.new("UVMap")

    for b, wall_recs in building_wall_meta:
        idx = int(b["index"])
        top = np.asarray(b["points_3d"], dtype=np.float64)
        n = len(top)
        bot = top.copy()
        bot[:, 1] = 0.0
        center_top = top.mean(0)
        mode_by_ei = {r["edge_index"]: r["bake_mode"] for r in wall_recs}

        if b["has_parapet"]:
            for p_a, p_b in zip(top, np.roll(top, -1, axis=0)):
                out_vec = 0.5 * (p_a + p_b) - center_top
                out_vec[1] = 0.0
                out_hat = out_vec / max(float(np.linalg.norm(out_vec)), 1e-6)
                in_a = p_a - out_hat * 0.08
                in_b = p_b - out_hat * 0.08
                add_bm_beam(bm_trim_conc, in_a, in_b, 0.22, 0.40, y_base=float(p_a[1]) - 0.02)
                add_bm_beam(bm_trim_brick, in_a, in_b, 0.28, 0.09, y_base=float(p_a[1]) + 0.36)

        if b["has_parapet"] and b["stories"] == 2 and idx % 6 == 0:
            edge_dir = top[1] - top[0]
            add_bm_box(bm_trim_conc, center_top + np.array([1.1, 0.22, 0.6]), np.array([1.4, 0.44, 1.4]), rot_yaw_w=math.atan2(-edge_dir[2], edge_dir[0]))

        if b["solar_heater"]:
            sh_base = center_top + np.array([1.15, 0.0, -0.75])
            sh_yaw = math.radians(-28.0 + (idx % 5) * 8.0)
            add_bm_box(bm_steel, sh_base + np.array([0.0, 0.42, 0.0]), np.array([1.10, 0.72, 0.06]), rot_yaw_w=sh_yaw)
            add_bm_box(bm_solar_col, sh_base + np.array([0.0, 0.48, 0.30]), np.array([1.15, 0.05, 0.95]), rot_yaw_w=sh_yaw, pitch_w=math.radians(38.0))
            add_bm_box(bm_solar_tank, sh_base + np.array([0.0, 0.88, -0.14]), np.array([1.25, 0.28, 0.28]), rot_yaw_w=sh_yaw)

        for ei in range(n):
            wall = b["walls"][ei]
            if wall["role"] == "party_wall":
                continue
            p0, p1 = bot[ei], bot[(ei + 1) % n]
            w_dir = p1 - p0
            w_len = float(np.linalg.norm(w_dir))
            if w_len < 3.0:
                continue
            u_hat = w_dir / w_len
            n_out = np.array([wall["outward_xz"][0], 0.0, wall["outward_xz"][1]], dtype=np.float64)
            w_yaw = math.atan2(-u_hat[2], u_hat[0])
            h_wall = float(b["height_m"])

            # Northern brick plinth + upper eave tile/brick cornice strip + 2-story belt course
            add_bm_beam(bm_trim_brick, p0 + n_out * 0.03, p1 + n_out * 0.03, 0.07, 0.48, y_base=0.0)
            if b.get("front_cornice_brick") and wall["role"] == "shopfront":
                add_bm_beam(bm_trim_brick, p0 + n_out * 0.06, p1 + n_out * 0.06, 0.14, 0.22, y_base=h_wall - 0.14)
            if b["stories"] == 2 and h_wall >= 5.2:
                add_bm_beam(bm_trim_conc, p0 + n_out * 0.04, p1 + n_out * 0.04, 0.08, 0.14, y_base=2.88)

            mid_w = 0.5 * (p0 + p1)
            if (
                wall["role"] == "shopfront"
                and b.get("is_core")
                and w_len >= 5.2
                and b["id"] in ("house_21", "house_24", "house_25", "house_27", "house_28", "house_05", "house_08", "house_26", "house_31")
            ):
                sign_w = min(w_len * 0.52, 4.4)
                use_red = idx % 2 == 0
                sign_bm = bm_sign_red if use_red else bm_sign_blue
                sign_uv = uv_sign_red if use_red else uv_sign_blue
                add_bm_box(sign_bm, mid_w + n_out * 0.10 + np.array([0.0, 2.72, 0.0]), np.array([sign_w, 0.62, 0.08]), rot_yaw_w=w_yaw, uv_layer=sign_uv)
                add_bm_box(bm_trim_conc, mid_w + n_out * 0.06 + np.array([0.0, 2.72, 0.0]), np.array([sign_w + 0.16, 0.72, 0.05]), rot_yaw_w=w_yaw)
                canopy_bm = bm_roof_blue if idx % 2 == 1 else bm_roof_silv
                add_bm_box(canopy_bm, mid_w + n_out * 0.38 + np.array([0.0, 2.42, 0.0]), np.array([min(w_len * 0.76, 6.2), 0.06, 0.75]), rot_yaw_w=w_yaw)

            if b["archetype"] in ("blue_shed", "metal_shed") and h_wall < 3.6:
                continue

            bake_mode = mode_by_ei.get(ei, "pbr_procedural")
            if bake_mode == "photo_facade":
                # Photo-baked facade: NO overlapping procedural black windows! Keep only sparse AC condenser
                if b["stories"] == 2 and idx % 3 == 1:
                    ac_c = p0 + u_hat * (w_len * 0.18) + n_out * 0.18 + np.array([0.0, 3.15, 0.0])
                    add_bm_box(bm_solar_tank, ac_c, np.array([0.74, 0.52, 0.30]), rot_yaw_w=w_yaw)
                continue

            usable = w_len - 2.0
            n_bays = max(1, min(5, int(usable // 2.4)))
            floor_heights = [1.45] if (b["stories"] == 1 or h_wall < 5.0) else [1.45, 3.85]
            for fi, fy in enumerate(floor_heights):
                if fy + 0.9 > h_wall:
                    continue
                if fi == 0 and wall["role"] == "shopfront":
                    n_shutters = max(1, min(3, int(usable // 3.2)))
                    for si in range(n_shutters):
                        dist_along = 1.0 + ((si + 0.5) / n_shutters) * usable
                        sc = p0 + u_hat * dist_along + np.array([0.0, 1.22, 0.0])
                        add_bm_box(bm_shutter, sc + n_out * 0.035, np.array([1.95, 2.38, 0.07]), rot_yaw_w=w_yaw)
                        add_bm_box(bm_win_frame, sc + n_out * 0.055 + np.array([0.0, 1.22, 0.0]), np.array([2.1, 0.10, 0.11]), rot_yaw_w=w_yaw)
                else:
                    for bi in range(n_bays):
                        frac = (bi + 0.5) / n_bays if n_bays > 1 else 0.5
                        dist_along = 1.0 + frac * usable
                        wc = p0 + u_hat * dist_along + np.array([0.0, fy, 0.0])
                        win_w = 1.38 if w_len > 7.0 else 1.08
                        win_h = 1.34
                        q = (idx * 7 + ei * 3 + fi * 5 + bi) % 4
                        qx, qy = (q % 2) * 0.5, (q // 2) * 0.5
                        add_bm_box(
                            bm_win_glass,
                            wc + n_out * 0.025,
                            np.array([win_w, win_h, 0.04]),
                            rot_yaw_w=w_yaw,
                            uv_layer=uv_win_glass,
                            uv_rect=(qx + 0.008, qy + 0.008, qx + 0.492, qy + 0.492),
                            flip_u=((idx + bi) % 2 == 1),
                        )
                        add_bm_box(bm_win_frame, wc + n_out * 0.045 + u_hat * (-win_w * 0.5), np.array([0.06, win_h + 0.06, 0.055]), rot_yaw_w=w_yaw)
                        add_bm_box(bm_win_frame, wc + n_out * 0.045 + u_hat * (win_w * 0.5), np.array([0.06, win_h + 0.06, 0.055]), rot_yaw_w=w_yaw)
                        add_bm_box(bm_trim_conc, wc + n_out * 0.065 + np.array([0.0, -win_h * 0.5 - 0.05, 0.0]), np.array([win_w + 0.22, 0.08, 0.13]), rot_yaw_w=w_yaw)
                        add_bm_box(bm_trim_conc, wc + n_out * 0.055 + np.array([0.0, win_h * 0.5 + 0.05, 0.0]), np.array([win_w + 0.22, 0.08, 0.10]), rot_yaw_w=w_yaw)
                        if fi == 1 and bi == 0 and idx % 3 == 1:
                            ac_c = wc + u_hat * (win_w * 0.5 + 0.55) + n_out * 0.18 + np.array([0.0, -0.35, 0.0])
                            add_bm_box(bm_solar_tank, ac_c, np.array([0.74, 0.52, 0.30]), rot_yaw_w=w_yaw)

    finalize_bmesh_object("Detail_Window_Glass", bm_win_glass, mat_window_glass, col_details, skip_auto_uv=True) if len(bm_win_glass.verts) else bm_win_glass.free()
    finalize_bmesh_object("Detail_ShopSign_Red", bm_sign_red, mat_sign_red, col_details, skip_auto_uv=True) if len(bm_sign_red.verts) else bm_sign_red.free()
    finalize_bmesh_object("Detail_ShopSign_Blue", bm_sign_blue, mat_sign_blue, col_details, skip_auto_uv=True) if len(bm_sign_blue.verts) else bm_sign_blue.free()
    detail_specs = [
        ("Detail_Concrete_Trim", bm_trim_conc, mat_concrete),
        ("Detail_Brick_Plinth", bm_trim_brick, mat_brick),
        ("Detail_Window_Frame", bm_win_frame, mat_window_frame),
        ("Detail_Shop_Shutters", bm_shutter, mat_shutter),
        ("Detail_Solar_Collectors", bm_solar_col, mat_solar_tube),
        ("Detail_Solar_Tanks", bm_solar_tank, mat_solar_tank),
        ("Detail_Steel_Mounts", bm_steel, mat_steel),
        ("Detail_Roof_Terracotta", bm_roof_tile, mat_roof_terracotta),
        ("Detail_Roof_Silver", bm_roof_silv, mat_roof_silver),
        ("Detail_Roof_Blue", bm_roof_blue, mat_roof_blue),
        ("Detail_Roof_Slate", bm_roof_slate, mat_roof_slate),
    ]
    for dname, dbm, dmat in detail_specs:
        if len(dbm.verts) > 0:
            finalize_bmesh_object(dname, dbm, dmat, col_details)
        else:
            dbm.free()

    # 9. COL_Street_Props: utility poles/lamps/wires, sculpted sedans, tricycles, courtyard brick piles
    bm_poles = bmesh.new()
    bm_wires = bmesh.new()
    pole_pts = [np.array([p[0], 0.0, p[1]], dtype=np.float64) for p in spec["utility_poles_xz"]]
    for pp in pole_pts:
        add_bm_tube(bm_poles, [pp, pp + np.array([0.0, 9.5, 0.0])], [0.15, 0.085], sides=8)
        add_bm_box(bm_wires, pp + np.array([0.0, 8.6, 0.0]), np.array([1.65, 0.10, 0.10]), rot_yaw_w=0.45)
        add_bm_box(bm_wires, pp + np.array([-0.55, 6.2, -0.45]), np.array([1.25, 0.08, 0.08]), rot_yaw_w=-0.65)
        for off in (-0.55, 0.0, 0.55):
            add_bm_box(bm_poles, pp + np.array([off * 0.9, 8.72, off * 0.4]), np.array([0.08, 0.12, 0.08]))
    for pa, pb in zip(pole_pts, pole_pts[1:]):
        for off in (-0.55, 0.0, 0.55):
            cable_pts = [pa * (1.0 - k / 8.0) + pb * (k / 8.0) + np.array([off * 0.9, 8.65 - math.sin(k / 8.0 * math.pi) * 0.65, off * 0.4]) for k in range(9)]
            add_bm_tube(bm_wires, cable_pts, [0.022] * len(cable_pts), sides=5)
    if len(bm_poles.verts) > 0:
        finalize_bmesh_object("Utility_Poles", bm_poles, mat_pole, col_props)
    else:
        bm_poles.free()
    if len(bm_wires.verts) > 0:
        finalize_bmesh_object("Utility_Wires_And_Arms", bm_wires, mat_steel, col_props)
    else:
        bm_wires.free()

    bm_car_glass = bmesh.new()
    bm_car_wheels = bmesh.new()
    bm_car_trim = bmesh.new()
    bm_car_hl = bmesh.new()
    bm_car_tl = bmesh.new()
    cars_by_paint: dict[str, bmesh.types.BMesh] = {k: bmesh.new() for k in car_mats}
    for car in spec["parked_cars"]:
        cx, cz = car["xz"]
        add_bm_sedan(cars_by_paint[car["paint"]], bm_car_glass, bm_car_wheels, bm_car_trim, bm_car_hl, bm_car_tl, np.array([cx, 0.0, cz], dtype=np.float64), float(car["yaw_rad"]))
    for tri in spec.get("tricycles", []):
        cx, cz = tri["xz"]
        yaw = float(tri["yaw_rad"])
        f_d = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
        base_t = np.array([cx, 0.0, cz], dtype=np.float64)
        add_bm_box(cars_by_paint["blue"], base_t - f_d * 0.45 + np.array([0.0, 0.55, 0.0]), np.array([1.75, 0.55, 1.15]), rot_yaw_w=yaw)
        add_bm_box(bm_car_trim, base_t + f_d * 0.68 + np.array([0.0, 0.85, 0.0]), np.array([0.85, 1.05, 0.85]), rot_yaw_w=yaw)
    bm_brick_piles = bmesh.new()
    for bp in spec.get("brick_piles", []):
        add_bm_box(bm_brick_piles, np.array([bp["xz"][0], 0.42, bp["xz"][1]]), np.array([1.45, 0.84, 0.95]), rot_yaw_w=float(bp["yaw_rad"]))

    for paint, c_bm in cars_by_paint.items():
        finalize_bmesh_object(f"Parked_Cars_{paint}", c_bm, car_mats[paint], col_props, smooth=True) if len(c_bm.verts) else c_bm.free()
    finalize_bmesh_object("Parked_Cars_Glass", bm_car_glass, mat_car_glass, col_props, smooth=True) if len(bm_car_glass.verts) else bm_car_glass.free()
    finalize_bmesh_object("Parked_Cars_Wheels", bm_car_wheels, mat_tire, col_props, smooth=True) if len(bm_car_wheels.verts) else bm_car_wheels.free()
    finalize_bmesh_object("Parked_Cars_Trim", bm_car_trim, mat_window_frame, col_props) if len(bm_car_trim.verts) else bm_car_trim.free()
    finalize_bmesh_object("Parked_Cars_Headlights", bm_car_hl, mat_car_headlight, col_props) if len(bm_car_hl.verts) else bm_car_hl.free()
    finalize_bmesh_object("Parked_Cars_Taillights", bm_car_tl, mat_car_taillight, col_props) if len(bm_car_tl.verts) else bm_car_tl.free()
    finalize_bmesh_object("Courtyard_Brick_Piles", bm_brick_piles, mat_brick, col_props) if len(bm_brick_piles.verts) else bm_brick_piles.free()

    # 10. COL_Foliage: split North vs South of road centerline so no merged bbox straddles the road
    cl_pts = np.asarray(spec.get("road_centerline_xz") or [], dtype=np.float64)
    if len(cl_pts) >= 2:
        r_dir = (cl_pts[-1] - cl_pts[0]) / max(float(np.linalg.norm(cl_pts[-1] - cl_pts[0])), 1e-6)
        r_perp = np.array([-r_dir[1], r_dir[0]], dtype=np.float64)
        cl_mid = cl_pts[len(cl_pts) // 2]

        def _side_key(px: float, pz: float) -> str:
            return "North" if float(np.dot(np.array([px, pz]) - cl_mid, r_perp)) >= 0.0 else "South"
    else:
        def _side_key(px: float, pz: float) -> str:
            return "North"

    bm_trunks = {"North": bmesh.new(), "South": bmesh.new()}
    bm_poplar = {"North": bmesh.new(), "South": bmesh.new()}
    uv_poplar = {k: bm.loops.layers.uv.new("UVMap") for k, bm in bm_poplar.items()}
    bm_broad = {"North": bmesh.new(), "South": bmesh.new()}
    uv_broad = {k: bm.loops.layers.uv.new("UVMap") for k, bm in bm_broad.items()}
    bm_shrub = {"North": bmesh.new(), "South": bmesh.new()}
    uv_shrub = {k: bm.loops.layers.uv.new("UVMap") for k, bm in bm_shrub.items()}
    rng_tree = random.Random(202607)

    for tr in spec["trees"]:
        tx, tz = tr["xz"]
        th = float(tr["height_m"])
        trad = float(tr["radius_m"])
        species = tr.get("species", "broadleaf")
        side_k = _side_key(tx, tz)
        base_p = np.array([tx, 0.0, tz], dtype=np.float64)

        if species == "poplar":
            add_bm_tube(bm_trunks[side_k], [base_p, base_p + np.array([0.0, th * 0.45, 0.0]), base_p + np.array([0.0, th * 0.85, 0.0])], [0.22, 0.14, 0.06], sides=6)
            n_cards = 16
            for ci in range(n_cards):
                y_frac = 0.28 + 0.64 * (ci / (n_cards - 1))
                taper = math.sin(math.pi * ((y_frac - 0.22) / 0.78)) ** 0.55
                r_off = rng_tree.uniform(0.0, trad * 0.55 * taper)
                ang = rng_tree.uniform(0.0, math.tau)
                c_pos = np.array([tx + math.cos(ang) * r_off, th * y_frac, tz + math.sin(ang) * r_off], dtype=np.float64)
                cw = trad * rng_tree.uniform(1.35, 1.85) * max(0.55, taper)
                ch = cw * rng_tree.uniform(1.1, 1.35)
                add_bm_leaf_card(
                    bm_poplar[side_k], uv_poplar[side_k], c_pos, cw, ch,
                    yaw=rng_tree.uniform(0.0, math.tau), pitch=rng_tree.uniform(-0.45, 0.45), roll=rng_tree.uniform(-0.35, 0.35),
                )
        else:
            fork_h = th * 0.38
            add_bm_tube(bm_trunks[side_k], [base_p, base_p + np.array([rng_tree.uniform(-0.08, 0.08), fork_h * 0.5, 0.0]), base_p + np.array([0.0, fork_h, 0.0])], [0.19, 0.13, 0.09], sides=6)
            for bi in range(3):
                ang = bi * math.tau / 3.0 + rng_tree.uniform(-0.35, 0.35)
                limb_tip = base_p + np.array([math.cos(ang) * trad * 0.6, th * rng_tree.uniform(0.68, 0.84), math.sin(ang) * trad * 0.6])
                add_bm_tube(bm_trunks[side_k], [base_p + np.array([0.0, fork_h * 0.9, 0.0]), limb_tip], [0.075, 0.035], sides=5)
            n_cards = 22
            for ci in range(n_cards):
                ang = rng_tree.uniform(0.0, math.tau)
                r_off = math.sqrt(rng_tree.random()) * trad * 0.72
                cy_h = th * rng_tree.uniform(0.46, 0.90)
                c_pos = np.array([tx + math.cos(ang) * r_off, cy_h, tz + math.sin(ang) * r_off], dtype=np.float64)
                cw = trad * rng_tree.uniform(0.95, 1.45)
                add_bm_leaf_card(
                    bm_broad[side_k], uv_broad[side_k], c_pos, cw, cw * rng_tree.uniform(0.92, 1.15),
                    yaw=rng_tree.uniform(0.0, math.tau), pitch=rng_tree.uniform(-0.65, 0.65), roll=rng_tree.uniform(-0.5, 0.5),
                )

    for sh in spec.get("shrubs", []):
        sx, sz = sh["xz"]
        sh_h = float(sh["height_m"])
        side_k = _side_key(sx, sz)
        for si in range(5):
            add_bm_leaf_card(
                bm_shrub[side_k], uv_shrub[side_k],
                np.array([sx + rng_tree.uniform(-0.3, 0.3), sh_h * 0.5, sz + rng_tree.uniform(-0.3, 0.3)]),
                sh_h * 1.45, sh_h * 1.15,
                yaw=(si * math.pi / 5.0) + rng_tree.uniform(-0.2, 0.2), pitch=rng_tree.uniform(-0.4, 0.4), roll=0.0,
            )

    for side_k in ("North", "South"):
        if len(bm_trunks[side_k].verts) > 0:
            t_ob = finalize_bmesh_object(f"Village_Trees_Trunks_{side_k}", bm_trunks[side_k], mat_bark, col_foliage, smooth=True)
            t_ob["check_road_clearance"] = True
        else:
            bm_trunks[side_k].free()
        for label_f, fbm_dict, fmat in (
            ("Poplar_Leaves", bm_poplar, mat_leaf_poplar),
            ("Broadleaf_Leaves", bm_broad, mat_leaf_broadleaf),
            ("Shrubs", bm_shrub, mat_shrub),
        ):
            fbm = fbm_dict[side_k]
            if len(fbm.verts) > 0:
                lob = finalize_bmesh_object(f"Village_{label_f}_{side_k}", fbm, fmat, col_foliage, smooth=True, skip_auto_uv=True)
                lob["check_road_clearance"] = True
            else:
                fbm.free()

    # 11. Scene Sun & Cool Sky Bounce matched to afternoon village_road_near.png balance (or spec overrides)
    bpy.ops.object.light_add(type="SUN", location=(35.0, -35.0, 65.0))
    sun = bpy.context.object
    sun.name = "Village_Sun"
    sun.data.energy = float(spec.get("sun_energy", 4.3))
    sun.data.color = (1.0, 0.96, 0.90)
    sun.data.angle = math.radians(2.2)
    sun_az = math.radians(float(spec.get("sun_azimuth_deg", 32.0)))
    sun_el = math.radians(float(spec.get("sun_elevation_deg", 24.0)))
    sun_vec_w = (math.cos(sun_el) * math.cos(sun_az), math.sin(sun_el), math.cos(sun_el) * math.sin(sun_az))
    sun_vec_b = Vector(world_to_blender(sun_vec_w)).normalized()
    sun.rotation_euler = (-sun_vec_b).to_track_quat("-Z", "Y").to_euler()
    link_only(sun, col_cams)

    world = scene.world or bpy.data.worlds.new("VillageWorld")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.64, 0.74, 0.88, 1.0)
        bg_node.inputs["Strength"].default_value = 0.30

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 24
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    scene["preserve_scale"] = True
    scene["scene_kind"] = "photo_completed_village"
    scene["camera_position"] = spec["camera"]["position"]
    scene["camera_target"] = spec["camera"]["target"]
    scene["camera_fov"] = spec["camera"]["fov"]
    scene["building_count"] = len(spec["buildings"])
    scene["source_image"] = Path(spec.get("source_image", "village_road_near.png")).name

    blend_out.parent.mkdir(parents=True, exist_ok=True)
    uv_map_path = blend_out.parent / "uv_atlas_map.json"
    uv_map_path.write_text(json.dumps(uv_atlas_map, ensure_ascii=False), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_out))
    snap_dir = blend_out.parent / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(snap_dir / "001_geometry_blockout.blend"), copy=True)

    summary = {
        "blend": str(blend_out),
        "uv_atlas_map": str(uv_map_path),
        "buildings": len(spec["buildings"]),
        "atlases": num_atlases,
        "atlas_faces": len(uv_atlas_map["faces"]),
        "photo_facade_walls": n_photo_facades,
        "trees": len(spec["trees"]),
        "cars": len(spec["parked_cars"]),
    }
    print("VILLAGE_BLOCKOUT_OK", json.dumps(summary))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    parser.add_argument("--blend", type=Path, default=ROOT / "artifacts/blender_village/village_master.blend")
    return parser.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)
    build_village(args.spec.resolve(), args.blend.resolve())
