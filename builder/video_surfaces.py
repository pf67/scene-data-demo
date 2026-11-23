"""Transfer measured roof heights and rectified video textures onto existing scene faces."""

from collections import defaultdict
import hashlib
import json
import re
from os import environ
from pathlib import Path

import numpy as np
from PIL import Image


def array(glb, index):
    a = glb.doc["accessors"][index]
    v = glb.doc["bufferViews"][a["bufferView"]]
    width = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[a["type"]]
    dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
    return (
        np.frombuffer(
            glb.blob,
            dtype,
            a["count"] * width,
            v.get("byteOffset", 0) + a.get("byteOffset", 0),
        )
        .reshape(-1, width)
        .copy()
    )


def primitives(glb, node):
    mesh = glb.doc["meshes"][node["mesh"]]
    out = []
    for p in mesh["primitives"]:
        attrs = p["attributes"]
        if not all(k in attrs for k in ("POSITION", "NORMAL", "TEXCOORD_0")):
            return []
        arrays = [array(glb, attrs[k]) for k in ("POSITION", "NORMAL", "TEXCOORD_0")]
        if "indices" in p:
            indices = array(glb, p["indices"]).ravel()
            arrays = [a[indices] for a in arrays]
        out.append((p["material"], *(a.reshape(-1, 3, a.shape[1]) for a in arrays)))
    return out


def signature(parts):
    h = hashlib.sha256()
    for _, v, n, uv in parts:
        for a in (v, n, uv):
            h.update(np.ascontiguousarray(a, dtype="<f4"))
    return h.hexdigest()


def planar_faces(parts):
    for primitive, (_, vertices, normals, _) in enumerate(parts):
        n = normals.mean(1).astype(float)
        offset = (n * vertices[:, 0]).sum(1)
        plane = np.rint(np.c_[n * 500, offset * 50]).astype(int)
        for key in np.unique(plane, axis=0):
            idx = np.flatnonzero((plane == key).all(1))
            v = vertices[idx]
            if len(idx) < 2 or (abs(n[idx, 1].mean()) > 0.45 and v[:, :, 1].max() < 1):
                continue
            area = (
                0.5
                * np.linalg.norm(
                    np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
                ).sum()
            )
            if area < 5:
                continue
            normal = n[idx].mean(0)
            normal /= np.linalg.norm(normal)
            axis = np.cross([0.0, 1.0, 0.0], normal)
            if np.linalg.norm(axis) < 0.05:
                axis = v[0, 1] - v[0, 0]
            axis /= np.linalg.norm(axis)
            vertical = np.cross(normal, axis)
            basis = np.stack([axis, vertical])
            local = v.reshape(-1, 3) @ basis.T
            lo, hi = local.min(0), local.max(0)
            if min(hi - lo) < 0.55 or np.prod(hi - lo) > area * 2.3:
                continue
            yield dict(
                primitive=primitive,
                indices=idx,
                vertices=v,
                normal=normal,
                basis=basis,
                lo=lo,
                hi=hi,
                area=float(area),
            )


def image_key(image):
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def inspection_camera(glb, names):
    positions = []
    for i in glb.doc["scenes"][0]["nodes"]:
        node = glb.doc["nodes"][i]
        if node["name"] in names:
            positions.extend(v.reshape(-1, 3) for _, v, _, _ in primitives(glb, node))
    if not positions:
        return None
    points = np.concatenate(positions)
    lo, hi = points.min(0), points.max(0)
    center = (lo + hi) * 0.5
    center[1] = 0.72 * hi[1]
    origin = np.array(glb.doc["scenes"][0]["extras"]["camera_position"])
    radial = (origin - center)[[0, 2]]
    radial /= max(np.linalg.norm(radial), 0.01)
    angle = np.radians(-60.0)
    cs, sn = np.cos(angle), np.sin(angle)
    toward = np.array([[cs, sn], [-sn, cs]]) @ radial
    span = max(8.0, np.linalg.norm((hi - lo)[[0, 2]]) * 0.5)
    eye = center + np.array(
        [2.85 * span * toward[0], 2.15 * span, 2.85 * span * toward[1]]
    )
    return dict(
        position=eye.tolist(),
        target=center.tolist(),
        up=[0, 1, 0],
        size=[1200, 800],
        principal=[0.5, 0.5],
        focalRatio=1.0,
        fov=38.0,
    )


def add_evidence(glb, plan, photo):
    key = plan.get("videoEvidence")
    if not key:
        return None
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("无效的视频证据编号")
    root = Path(__file__).resolve().parent / "video-surfaces"

    roots = (
        [Path(environ["SCENE_VIDEO_EVIDENCE_DIR"]), root]
        if environ.get("SCENE_VIDEO_EVIDENCE_DIR")
        else [root]
    )
    folder = next(
        (r / key for r in roots if (r / key / "manifest.json").is_file()), None
    )
    if not folder:
        raise ValueError("场景所需的视频表面证据未安装")
    raw = (folder / "manifest.json").read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != key:
        raise ValueError("视频表面清单校验失败")
    spec = json.loads(raw)
    if spec["imageKey"] != image_key(photo):
        return dict(
            buildings=0,
            adjustedRoofHeights=0,
            correctedFootprints=0,
            joinedEaves=0,
            roofTextures=0,
            wallTextures=0,
            skippedBuildings=len(spec["nodes"]),
            evidence=key,
            reason="当前参考图与视频标定时的图片不同，已跳过视频修正。",
        )
    atlas_file = folder / "atlas.jpg"
    if hashlib.sha256(atlas_file.read_bytes()).hexdigest() != spec["atlasSha256"]:
        raise ValueError("视频纹理校验失败")
    atlas_mat = None
    with Image.open(atlas_file) as im:
        atlas = im.convert("RGB")
    geometry = walls = roofs = footprints = 0
    applied = []
    nodes = glb.doc["nodes"]
    matched = {}
    for node_id in glb.doc["scenes"][0]["nodes"]:
        node = nodes[node_id]
        name = node.get("name", "")
        row = spec["nodes"].get(name)
        if not row:
            continue
        parts = primitives(glb, node)
        if signature(parts) != row["signature"]:
            continue
        matched[name] = (node, parts)
    # A change to either side of a shared roof invalidates that joint.
    pending = set(spec["nodes"]) - matched.keys()
    neighbours = defaultdict(set)
    for joint in spec.get("sharedEaves", []):
        a, b = joint["buildings"]
        neighbours[a].add(b)
        neighbours[b].add(a)
    while pending:
        for other in neighbours[pending.pop()]:
            if matched.pop(other, None) is not None:
                pending.add(other)
    for name, (node, parts) in matched.items():
        row = spec["nodes"][name]
        groups = defaultdict(list)
        for i, (mat, v, n, uv) in enumerate(parts):
            modified = np.zeros(len(v), bool)
            v1 = v
            n1 = n
            if "scale" in row:
                vec = np.array(row["scale"])
                shift = np.array(row["translation"])
                v1 = v * vec + shift
                n1 = n / vec
                n1 /= np.maximum(np.linalg.norm(n1, axis=2, keepdims=True), 1.0e-8)
            if "footprintHomography" in row:
                h = np.asarray(row["footprintHomography"])
                raw = v[:, :, [0, 2]].astype(np.float64)
                denom = raw @ h[2, :2] + h[2, 2]
                mapped = (raw @ h[:2, :2].T + h[:2, 2]) / denom[:, :, None]
                jac = (h[:2, :2] - mapped[:, :, :, None] * h[2, :2]) / denom[
                    :, :, None, None
                ]
                v1 = v1.copy()
                v1[:, :, [0, 2]] = mapped
                n1 = n.astype(float, copy=True)
                n1[:, :, 1] /= row.get("scale", [1.0, 1.0, 1.0])[1]
                n1[:, :, [0, 2]] = (
                    n[:, :, [0, 2]][:, :, None, :] @ np.linalg.inv(jac)
                ).squeeze(-2)
                n1 /= np.maximum(np.linalg.norm(n1, axis=2, keepdims=True), 1.0e-8)
            for face in row["faces"]:
                if face["primitive"] != i:
                    continue
                idx = np.asarray(face["triangles"], int)
                if modified[idx].any():
                    raise ValueError("重复的视频表面")
                if atlas_mat is None:
                    atlas_mat = glb.material(
                        "Measured video surfaces", [1, 1, 1], atlas
                    )
                basis = np.asarray(face["basis"])
                uv1 = v[idx] @ basis[:, :3].T + basis[:, 3]
                groups[atlas_mat].append(
                    (v1[idx].reshape(-1, 3), n1[idx].reshape(-1, 3), uv1.reshape(-1, 2))
                )
                modified[idx] = True
                if face["kind"] == "roof":
                    roofs += 1
                else:
                    walls += 1
            keep = ~modified
            if keep.any():
                groups[mat].append(
                    (
                        v1[keep].reshape(-1, 3),
                        n1[keep].reshape(-1, 3),
                        uv[keep].reshape(-1, 2),
                    )
                )
        # The construction and default scenes may share a mesh. Only replace
        # this instance's mesh, preserving its node, name and scene membership.
        glb.mesh(name, groups)
        new_node = glb.doc["scenes"][0]["nodes"].pop()
        node["mesh"] = nodes[new_node]["mesh"]
        nodes.pop()
        geometry += int("scale" in row)
        footprints += int("footprintHomography" in row)
        applied.append(name)
    summary = dict(
        buildings=len(applied),
        adjustedRoofHeights=geometry,
        correctedFootprints=footprints,
        joinedEaves=sum(
            all(name in applied for name in g["buildings"])
            for g in spec.get("sharedEaves", [])
        ),
        roofTextures=roofs,
        wallTextures=walls,
        skippedBuildings=len(spec["nodes"]) - len(applied),
        evidence=key,
    )
    if summary["joinedEaves"]:
        group = next(
            g for g in spec["sharedEaves"] if all(n in applied for n in g["buildings"])
        )
        summary["inspectionCamera"] = inspection_camera(glb, group["buildings"])
    if not applied:
        summary["reason"] = "当前建筑轮廓与视频标定时的网格不同，已跳过视频修正。"
    glb.doc["scenes"][0]["extras"]["video_surfaces"] = summary.copy()
    return summary
