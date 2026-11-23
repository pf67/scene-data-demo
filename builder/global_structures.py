"""Fit solid buildings to the reviewed roof and facade contours of the anchor photograph."""

from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from mesh import SurfaceAtlas, building_faces, subdivide
from surface_context import inferred_roof, overlaps, slab_edge
from surface_marks import FaceMapping


def secondary_face(face, normal, vertices, anchor, fields, images):
    """Assign one photograph to a complete hidden elevation only when depth agrees across it."""
    if abs(normal[1]) > 0.15:
        return None
    samples = subdivide(face, 2).reshape(-1, 3, 3).mean(axis=1)
    best = None
    for key, other in fields.items():
        if key == anchor:
            continue
        cam = other.camera
        ray = cam.eye - face.mean(0)
        cosine = normal @ ray / np.linalg.norm(ray)
        if cosine < 0.42:
            continue
        coords, z = cam.pixels(samples)
        if z.min() <= 0.1 or (coords < 1).any() or (coords >= cam.size - 1).any():
            continue
        valid = other.check(coords, z, normal, cam.size)
        if valid.mean() < 0.75:
            continue
        uv, depth = cam.pixels(vertices)
        if depth.min() <= 0.1 or (uv < 0).any() or (uv >= cam.size - 1).any():
            continue
        if other.accepted_region is not None:
            # This branch assigns one photo to the whole elevation. A regional
            # registration cannot supply an unverified corner or window bay.
            region_xy = np.rint(np.vstack([uv, coords])).astype(int)
            if not other.accepted_region[region_xy[:, 1], region_xy[:, 0]].all():
                continue
        xy = np.rint(coords / cam.size * np.array(images[key].shape[1::-1])).astype(int)
        rgb = images[key][xy[:, 1], xy[:, 0]].astype(float)
        foliage = (rgb[:, 1] > rgb[:, 2] + 10) & (rgb[:, 0] < rgb[:, 1] * 1.07) & (rgb[:, 1] < 130)
        if foliage.mean() > 0.35:
            continue
        quality = valid.mean() * cosine * cam.k[0, 0] / np.median(z)
        if best is None or quality > best[0]:
            best = (quality, key, uv / cam.size)
    return None if best is None else best[1:]


def add_buildings(glb, mark, roofs, plan, materials, bank, field, fields=None, images=None, existing=()):
    if mark is None:
        return []
    indexed = {b["id"]: (i, r) for i, (b, r) in enumerate(zip(plan["buildings"], roofs))}
    rows = list(mark.buildings.values()) + mark.data.get("infill", []) + mark.data.get("globalRoofs", [])
    if not rows:
        return []
    if mark.source not in materials:
        materials[mark.source] = glb.material(
            "Reviewed " + mark.source, [1, 1, 1], Image.fromarray(mark.image)
        )
    neutral = glb.material("Eaves and foundations", [0.37, 0.33, 0.28])
    atlas = SurfaceAtlas(glb)
    ticks = np.linspace(0, 1, atlas.tile)
    u, v = np.meshgrid(ticks, ticks)
    built = []
    previous_names = {glb.doc["nodes"][i]["name"] for i in glb.doc["scenes"][0]["nodes"]}
    for index, row in enumerate(rows):
        name = row.get("id") or row["name"]
        if "Photo building " + name in previous_names:
            continue
        original_index, surveyed = indexed.get(name, (-index - 1, None))
        nominal = row.get("height", float(surveyed[:, 1].mean()) if surveyed is not None else 3.0)
        corners = np.asarray(row["roof"], dtype=float)
        feet = row.get("feet", [None] * 4)
        height = inferred_roof(mark, {**row, "height": nominal, "feet": feet})[:, 1].mean()
        # Bind the eaves to the same local coordinate frame as the fused terrain.
        # Sample an inset of the roof so adjacent walls and streets stay outside.
        inset = corners * 0.72 + corners.mean(0) * 0.28
        poly = np.int32(np.rint(inset * (field.camera.size / mark.size)))
        x, y, w, h = cv2.boundingRect(poly)
        x0, y0 = max(0, x), max(0, y)
        patch = field.height[y0 : min(y + h, field.height.shape[0]), x0 : min(x + w, field.height.shape[1])]
        if patch.size and w > 2 and h > 2:
            region = np.zeros(patch.shape, np.uint8)
            cv2.fillConvexPoly(region, poly - [x0, y0], 1)
            sampled = patch[(region != 0) & np.isfinite(patch)]
            if sampled.size > 5:
                depth_height = np.clip(np.median(sampled), 0.55 * height, 1.55 * height)
                height = 0.65 * depth_height + 0.35 * height
        roof = mark.cam.plane(corners / mark.size, height)
        if overlaps(roof, [*existing, *built]):
            continue
        feet_px = [mark.pixels(pt) if pt is not None else None for pt in feet]
        mapping = FaceMapping(mark, roof, mark.pixels(corners), feet_px)
        alternate_roof = None
        if surveyed is not None and original_index in bank.reviewed_roofs:
            size_px = abs(cv2.contourArea(np.float32(mark.pixels(corners))))
            if bank.reviewed_roofs[original_index][0] > 1.20 * size_px:
                projected = mark.cam.pixels(surveyed)[0]
                orders = [np.roll(np.arange(4)[::direction], j) for direction in (1, -1) for j in range(4)]
                order = min(orders, key=lambda ids: np.linalg.norm(projected - mark.pixels(corners[ids])))
                alternate_roof = roof[order]
        hip = row.get("hipRiseM", 0.0)
        groups = defaultdict(list)
        donors = []
        faces = building_faces(roof, hip, hip=bool(hip))
        for face, n in faces:
            if abs(n[1]) > 0.05 or len(face) != 4:
                continue
            edge = np.cross([0.0, 1.0, 0.0], n)
            along = face @ edge
            # Match image-space corners to the same building's visible wall and rectify it once.
            extent = np.ptp(along)
            order = sorted(range(4), key=lambda i: (-face[i, 1], along[i] if face[i, 1] > 0 else -along[i]))
            uv = mapping.texture(face, n, face[order])
            if uv is None or np.any(uv <= 0.001) or np.any(uv >= 0.999):
                continue
            px = np.float32(uv * mark.cam.size)
            if abs(cv2.contourArea(px)) < 110:
                continue
            dst = np.float32(
                [[0, 0], [atlas.tile - 1, 0], [atlas.tile - 1, atlas.tile - 1], [0, atlas.tile - 1]]
            )
            image = cv2.warpPerspective(
                mark.image, cv2.getPerspectiveTransform(px, dst), (atlas.tile, atlas.tile)
            )
            donors.append((extent, image))
        for face, n in faces:
            verts = subdivide(face, 2)
            tex = mapping.texture(face, n, verts)
            mat = materials[mark.source]
            if alternate_roof is not None and n[1] > .35:
                patch = bank.reviewed_roof_uv(original_index, alternate_roof, verts)
                if patch is not None:
                    source, image, tex = patch
                    if source not in materials:
                        materials[source] = glb.material("Reviewed roof " + source, [1, 1, 1], Image.fromarray(image))
                    mat = materials[source]
                    bank.roof_sources[original_index] = source
            # When the anchor cannot see this wall, prefer the rectified photograph
            # of its named facade to an uncorrected projection from another camera.
            reviewed = bank.reviewed_wall(original_index, n, u, v) if tex is None else None
            if tex is None and reviewed is None and fields:
                observed = secondary_face(face, n, verts, mark.source, fields, images)
                if observed is not None:
                    key, tex = observed
                    if key not in materials:
                        materials[key] = glb.material(
                            "Fusion view " + key, [1, 1, 1], Image.fromarray(images[key])
                        )
                    mat = materials[key]
            if tex is None:
                if abs(n[1]) < 0.15:
                    along = np.cross([0.0, 1.0, 0.0], n)
                    coords = np.column_stack((verts @ along, -verts[:, 1]))
                    size = np.maximum(np.ptp(coords, axis=0), 0.001)
                    local = (coords - coords.min(0)) / size
                    tile = reviewed
                    if tile is None and donors:
                        span, tile = min(donors, key=lambda t: abs(np.log(t[0] / size[0])))
                        repeats = max(1, round(size[0] / span))
                        tile = cv2.remap(
                            tile,
                            np.float32((1 - np.abs(1 - u * repeats % 2)) * (atlas.tile - 1)),
                            np.float32(v * (atlas.tile - 1)),
                            cv2.INTER_LINEAR,
                        )
                    if tile is None:
                        tile = bank.fill(
                            face, n, original_index, u, v, np.full((*u.shape, 3), [161, 149, 127], np.uint8)
                        )
                    mat, tex = atlas.put(tile, local)
                else:
                    mat, tex = neutral, np.zeros((len(verts), 2))
            groups[mat].append((verts, np.tile(n, (len(verts), 1)), tex))
        groups[neutral].extend(slab_edge(roof))
        glb.mesh("Photo building " + name, groups)
        built.append(roof)
    atlas.finish()
    return built
