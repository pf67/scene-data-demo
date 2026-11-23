"""Embed camera-local, textured depth surfaces as optional glTF scenes beside the closed model."""

import numpy as np
from PIL import Image


def triangulate(field):
    cam = field.camera
    height, width = field.depth.shape
    scale = min(1.0, 288 / max(width, height), np.sqrt(60000 / (width * height)))
    xs = np.unique(np.rint(np.linspace(0, width - 1, max(2, round(width * scale)))).astype(int))
    ys = np.unique(np.rint(np.linspace(0, height - 1, max(2, round(height * scale)))).astype(int))
    x, y = np.meshgrid(xs, ys)
    z = field.depth[y, x]
    rays = (
        np.stack([(x - cam.k[0, 2]) / cam.k[0, 0], (y - cam.k[1, 2]) / cam.k[1, 1], np.ones_like(x)], -1)
        @ cam.r
    )
    far_limit = max(30, 16 * cam.eye[1])
    # Where the calibrated inverse depth tends to infinity, a downward ray can
    # still locate distant ground. Upward rays remain outside the mesh.
    distant = np.isfinite(z) & (z >= far_limit) & (rays[:, :, 1] < -0.045)
    ground_z = -cam.eye[1] / np.minimum(rays[:, :, 1], -0.001)
    z = np.where(distant, np.minimum(ground_z, 2.5 * far_limit), z)
    vertices = (cam.eye + rays * z[:, :, None]).reshape(-1, 3)
    normals = field.normals[y, x].reshape(-1, 3).copy()
    normals[np.sum(normals * (cam.eye - vertices), axis=1) < 0] *= -1
    weak = np.linalg.norm(normals, axis=1) < 0.1
    normals[weak] = (cam.eye - vertices[weak]) / np.maximum(
        np.linalg.norm(cam.eye - vertices[weak], axis=1, keepdims=True), 1e-7
    )
    normals[distant.ravel()] = [0, 1, 0]
    uv = np.stack([(x + 0.5) / width, (y + 0.5) / height], -1).reshape(-1, 2)
    ids = np.arange(z.size).reshape(z.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    depth = z.ravel()
    # Use the diagonal of lesser depth change. Reject triangles spanning an
    # occlusion contour so roofs cannot turn into vertical curtains when moving.
    ad = np.abs(depth[a] - depth[d]) < np.abs(depth[b] - depth[c])
    first = np.stack([a, c, np.where(ad, d, b)], -1)
    second = np.where(ad[:, :, None], np.stack([a, d, b], -1), np.stack([b, c, d], -1))
    triangles = np.stack([first, second], axis=-2).reshape(-1, 3)
    depths = depth[triangles]
    near, far = depths.min(1), depths.max(1)
    valid = (
        np.isfinite(depths).all(1)
        & (near > 0.15)
        & ((depth < far_limit) | distant.ravel())[triangles].all(1)
    )
    keep = valid & ((far - near) < np.maximum(0.95, 0.115 * near))
    coverage = float(keep.mean())
    if coverage < 0.30:
        return None
    used, indices = np.unique(triangles[keep], return_inverse=True)
    radius = float(np.clip(0.11 * np.median(depth[used]), 1.5, 14.0))
    seam = triangles[valid & ~keep]
    margin = None
    if len(seam):
        # A separate surface continues the background under silhouettes. Its vertices
        # share the farther depth; no wall joins a near roof to a distant field.
        outside = np.argmax(depth[seam], axis=1)
        rear = seam[np.arange(len(seam)), outside]
        direction = rays.reshape(-1, 3)[seam]
        near_side = (depth[seam] < depth[rear, None] * 0.95)[:, :, None]
        ray_center = direction.mean(1, keepdims=True)
        direction = direction + near_side * (direction - ray_center) * 1.5
        pos = cam.eye + direction * (depth[rear, None, None] * 1.005)
        tex = uv[seam] + near_side * (uv[seam] - uv[seam].mean(1, keepdims=True)) * 1.5
        margin = (
            pos.reshape(-1, 3),
            np.repeat(normals[rear], 3, axis=0),
            np.clip(tex.reshape(-1, 2), 0, 1),
        )
    return (
        (vertices[used], normals[used], uv[used]),
        indices.ravel().astype("<u2"),
        coverage,
        radius,
        margin,
    )


def add_view_surfaces(glb, priors, images, materials):
    result = {}
    for source, field in priors.items():
        if field.anchor_error > 6.5:
            continue
        mesh = triangulate(field)
        if mesh is None:
            continue
        arrays, indices, coverage, radius, margin = mesh
        if source not in materials:
            materials[source] = glb.material("View " + source, [1, 1, 1], Image.fromarray(images[source]))
        extras = {
            "surface_source": source,
            "camera_position": field.camera.eye.tolist(),
            "camera_forward": field.camera.r[2].tolist(),
            "working_radius": round(radius, 2),
        }
        index = glb.add("scenes", {"name": "Depth surface " + source, "extras": extras, "nodes": []})
        glb.mesh("Depth surface " + source, {materials[source]: [arrays]}, indices, scene=index)
        if margin is not None:
            glb.mesh("Occlusion margins " + source, {materials[source]: [margin]}, scene=index)
        result[source] = {
            "sceneIndex": index,
            "triangles": len(indices) // 3,
            "marginTriangles": 0 if margin is None else len(margin[0]) // 3,
            "coveragePercent": round(coverage * 100, 2),
            "workingRadiusM": round(radius, 2),
        }
    return result
