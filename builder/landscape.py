"""Continue a reviewed rural site into a fixed, textured terrain around all four sides."""

import math

import cv2
import numpy as np
from PIL import Image

from ground import field_pattern, reviewed_streets, road_distance
from ground_material import GroundMaterials


def add_landscape(glb, plan, marks, ground, texture, ground_vertices):
    sources = [m for m in marks.values() if m.data.get("surroundingFields")]
    if not sources or ground["surface"] != "fields":
        return {}
    lo, hi = np.array(ground["boundsXZ"])
    span = max(hi - lo)
    padding = 1.25 * span
    outer_lo, outer_hi = lo - padding, hi + padding
    samples = 1536
    x, z = np.meshgrid(*[np.linspace(a, b, samples, dtype=np.float32) for a, b in zip(outer_lo, outer_hi)])
    outside = np.maximum.reduce([lo[0] - x, x - hi[0], lo[1] - z, z - hi[1]])
    settings = reviewed_streets(plan["ground"], marks, {v["source"] for v in plan["views"]})
    roads = settings["roads"]
    direction = np.subtract(roads[0]["points"][-1], roads[0]["points"][0]) if roads else [1, 0.28]
    atlas = field_pattern(
        x,
        z,
        np.random.default_rng(612),
        x.shape,
        max(0.3, min(hi - lo) / 260),
        math.atan2(direction[1], direction[0]),
        GroundMaterials(marks),
    ).astype(np.float32)
    best = np.zeros(x.shape, np.float32)
    # Bake reviewed fields wherever a real ground ray reaches the photograph.
    # Erode the mask softly to keep rooflines and tree silhouettes off the terrain.
    for mark in sources:
        support = np.zeros(mark.image.shape[:2], np.uint8)
        for poly in mark.data["surroundingFields"]:
            cv2.fillPoly(support, [np.int32(np.rint(mark.pixels(poly)))], 1)
        weight = np.clip(cv2.distanceTransform(support, cv2.DIST_L2, 3) / 7, 0, 1)
        for row in range(0, samples, 64):
            sl = slice(row, min(samples, row + 64))
            xx, zz = x[sl], z[sl]
            world = np.stack([xx, np.zeros_like(xx), zz], -1)
            uv, depth = mark.cam.pixels(world.reshape(-1, 3))
            inside = (
                (depth > 1) & np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < mark.cam.size - 1).all(1)
            )
            uv[~inside] = 0
            uv = uv.astype(np.float32).reshape(*xx.shape, 2)
            alpha = (
                cv2.remap(weight, uv, None, cv2.INTER_LINEAR) * inside.reshape(xx.shape) * (outside[sl] > 0)
            )
            quality = alpha / np.maximum(1.0, depth.reshape(xx.shape))
            use = (quality > best[sl]) & (alpha > 0.02)
            rgb = cv2.remap(mark.image, uv, None, cv2.INTER_LINEAR)
            atlas[sl][use] += alpha[use, None] * (rgb[use] - atlas[sl][use])
            best[sl][use] = quality[use]
    for road in roads:
        if not road.get("extendEnds", True):
            continue
        distance, _ = road_distance(x, z, road)
        width = road["widthM"]
        for radius, tone in (
            (width * 0.68, road.get("shoulder", [143, 132, 110])),
            (width * 0.50, road.get("asphalt", [82, 85, 83])),
        ):
            alpha = np.clip((radius - distance) * 2, 0, 1)[:, :, None]
            atlas = atlas * (1 - alpha) + np.asarray(tone) * alpha
    inner = np.asarray(texture)
    uv = np.float32(
        np.stack(
            [
                (x - lo[0]) / (hi[0] - lo[0]) * (inner.shape[1] - 1),
                (z - lo[1]) / (hi[1] - lo[1]) * (inner.shape[0] - 1),
            ],
            -1,
        )
    )
    rim = cv2.remap(inner, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    blend = np.clip(1 - outside / 6, 0, 1)[:, :, None]
    atlas = np.uint8(np.clip(atlas * (1 - blend) + rim * blend, 0, 255))

    # Include every boundary vertex of the inner terrain to meet the surveyed
    # site without a height step or an overlapping coplanar sheet.
    source_x, source_z = (np.unique(ground_vertices[:, i]) for i in (0, 2))
    heights = ground_vertices[:, 1].reshape(len(source_z), len(source_x))

    def axis(inner_axis):
        return np.r_[
            np.linspace(inner_axis[0] - padding, inner_axis[0], 33)[:-1],
            inner_axis,
            np.linspace(inner_axis[-1], inner_axis[-1] + padding, 33)[1:],
        ]

    gx, gz = np.meshgrid(axis(source_x), axis(source_z))
    dx = np.clip((gx - lo[0]) / (hi[0] - lo[0]), 0, 1) * (len(source_x) - 1)
    dz = np.clip((gz - lo[1]) / (hi[1] - lo[1]), 0, 1) * (len(source_z) - 1)
    edge_h = heights[np.rint(dz).astype(int), np.rint(dx).astype(int)]
    dist = np.maximum.reduce([lo[0] - gx, gx - hi[0], lo[1] - gz, gz - hi[1]])
    fade = np.clip(dist / (padding * 0.25), 0, 1)
    h = edge_h * (1 - fade) + fade * (
        1.7 * np.cos(gx / 58) * np.sin(gz / 38) + 1.2 * np.sin(gx / 89 + gz / 66)
    )
    gx_axis, gz_axis = gx[0], gz[:, 0]
    dhz, dhx = np.gradient(h, gz_axis, gx_axis)
    normals = np.stack([-dhx, np.ones_like(h), -dhz], -1).reshape(-1, 3)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    ids = np.arange(gx.size).reshape(gx.shape)
    a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]
    cx, cz = (gx[:-1, :-1] + gx[1:, 1:]) * 0.5, (gz[:-1, :-1] + gz[1:, 1:]) * 0.5
    annulus = (cx < lo[0]) | (cx > hi[0]) | (cz < lo[1]) | (cz > hi[1])
    indices = np.stack([a, b, c, a, c, d], -1)[annulus].ravel()
    used, indices = np.unique(indices, return_inverse=True)
    vertices = np.stack([gx, h, gz], -1).reshape(-1, 3)[used]
    coords = (vertices[:, [0, 2]] - outer_lo) / (outer_hi - outer_lo)
    material = glb.material("Fields beyond the village", [1, 1, 1], Image.fromarray(atlas))
    glb.mesh("Surrounding fields", {material: [(vertices, normals[used], coords)]}, indices)
    return {
        "triangles": len(indices) // 3,
        "boundsXZ": [outer_lo.tolist(), outer_hi.tolist()],
        "reviewedPhotoFraction": round(float((best[outside > 0] > 0).mean()), 4),
    }
