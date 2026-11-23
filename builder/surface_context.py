"""Construct closed courtyard buildings and volumetric vegetation from reviewed image observations."""

from collections import defaultdict
import math

import cv2
import numpy as np
from PIL import Image

from ground import ground_anchors, ground_bounds, reviewed_streets, road_distance, variation
from surface_marks import FaceMapping


def slab_edge(roof):
    """A closed narrow coping around the eave; leaves the photograph on the roof interior."""
    points = roof[:, [0, 2]]
    center = points.mean(0)
    towards = points - center
    width = np.linalg.norm(towards, axis=1)[:, None]
    offset = towards / np.maximum(width, 0.1) * min(0.22, 0.035 * np.min(np.ptp(points, axis=0)))
    outer = roof.copy()
    outer[:, [0, 2]] += offset
    inner = roof.copy()
    inner[:, [0, 2]] -= offset * 0.28
    up = np.array([0.0, 0.07, 0.0])
    down = np.array([0.0, -0.095, 0.0])
    for i in range(len(roof)):
        j = (i + 1) % len(roof)
        for quad, hint in (
            ([inner[i] + up, inner[j] + up, outer[j] + up, outer[i] + up], [0, 1, 0]),
            ([outer[i] + down, outer[j] + down, inner[j] + down, inner[i] + down], [0, -1, 0]),
            (
                [outer[i] + up, outer[j] + up, outer[j] + down, outer[i] + down],
                outer[i] + outer[j] - 2 * roof.mean(0),
            ),
            (
                [inner[i] + down, inner[j] + down, inner[j] + up, inner[i] + up],
                2 * roof.mean(0) - inner[i] - inner[j],
            ),
        ):
            quad = np.array(quad)
            n = np.cross(quad[1] - quad[0], quad[2] - quad[0])
            if n @ hint < 0:
                quad, n = quad[::-1], -n
            verts = quad[[0, 1, 2, 0, 2, 3]]
            yield verts, np.tile(n / max(np.linalg.norm(n), 1e-8), (6, 1)), np.zeros((6, 2))


def inferred_roof(mark, item):
    corners = np.array(item["roof"]) / mark.size
    feet = item["feet"]
    h = item["height"]
    ground = mark.cam.plane(corners, 0)
    ray = mark.cam.rays(corners)
    delta = ray[:, [0, 2]] / ray[:, 1:2]
    estimates = []
    for i, foot in enumerate(feet):
        if foot is not None:
            known = mark.cam.plane([np.array(foot) / mark.size], 0)[0]
            dy = (known[[0, 2]] - ground[i, [0, 2]]) @ delta[i] / max(delta[i] @ delta[i], 1e-8)
            if 0.50 * h < dy < 1.85 * h:
                estimates.append(dy)
    h = float(np.median(estimates)) if estimates else h
    return mark.cam.plane(corners, h)


def overlaps(roof, others):
    poly = roof[:, [0, 2]].astype(np.float32)
    area = abs(cv2.contourArea(poly))
    if area < 2 or not cv2.isContourConvex(poly):
        return True
    for other in others:
        ground = other[:, [0, 2]].astype(np.float32)
        if not ((ground.max(0) >= poly.min(0)) & (poly.max(0) >= ground.min(0))).all():
            continue
        intersection, _ = cv2.intersectConvexConvex(poly, ground)
        if intersection > 0.17 * min(area, abs(cv2.contourArea(ground))):
            return True
    return False


def leaf_texture(mark):
    """Make a repeating leaf texture from foliage pixels only, excluding sky, roofs and masonry."""
    samples = []
    for tree in mark.data.get("trees", []):
        x0, y0, x1, y1 = np.rint(mark.pixels(np.asarray(tree["box"]).reshape(2, 2))).astype(int).ravel()
        patch = mark.image[max(0, y0) : y1, max(0, x0) : x1].reshape(-1, 3).astype(float)
        red, green, blue = patch.T
        foliage = (green > blue + 8) & (green > red * 0.88) & (green < 150) & (green > 28)
        samples.extend(patch[foliage][::5])
    if len(samples) < 20:
        samples = [[44, 51, 32], [72, 78, 45], [100, 103, 59]]
    samples = np.asarray(samples)
    palette = np.percentile(samples, [12, 55, 90], axis=0)
    rng = np.random.default_rng(947)
    shape = (128, 128)
    value = 0.35 * variation(rng, shape, 13) + 0.48 * variation(rng, shape, 3) + rng.normal(0, 0.39, shape)
    level = np.clip(value + 0.94, 0, 1.999)
    band = level.astype(int)
    weight = (level - band)[:, :, None]
    color = palette[band] * (1 - weight) + palette[band + 1] * weight
    return Image.fromarray(np.uint8(np.clip(color, 0, 255)))


def tree_geometry(mark, item, seed):
    x0, y0, x1, y1 = item["box"]
    foot = np.asarray(item["foot"]) / mark.size
    base = mark.cam.plane([foot], 0)[0]
    _, z = mark.cam.pixels([base])
    pixel_scale = z[0] / mark.cam.k[1, 1] * mark.cam.size[1] / mark.size[1]
    radius = np.clip((x1 - x0) * pixel_scale * 0.46, 0.65, 6.5)
    top = np.clip((item["foot"][1] - y0) * pixel_scale * 1.07, 2.1, 13)
    crown = np.clip((y1 - y0) * pixel_scale * 0.52, 0.6, top * 0.44)
    trunk_top = max(0.8, top - 1.6 * crown)
    rng = np.random.default_rng(seed)
    parts = []
    lon, lat = np.meshgrid(np.linspace(0, 2 * np.pi, 11), np.linspace(-np.pi / 2, np.pi / 2, 7))
    unit = np.stack([np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], -1)
    ids = np.arange(lon.size).reshape(lon.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    tri = np.stack([a, c, b, b, c, d], -1).ravel()
    for i in range(21):
        angle = i * math.pi * (3 - math.sqrt(5)) + rng.uniform(-0.12, 0.12)
        layer = 1.7 * (i + 0.5) / 21 - 0.85
        fraction = math.sqrt(1 - layer**2) * rng.uniform(0.48, 0.71)
        middle = base + [
            math.cos(angle) * radius * fraction,
            top - crown + layer * crown * 0.69,
            math.sin(angle) * radius * fraction,
        ]
        axes = np.array(
            [
                radius * rng.uniform(0.30, 0.47),
                crown * rng.uniform(0.34, 0.57),
                radius * rng.uniform(0.28, 0.47),
            ]
        )
        noise = (
            1
            + 0.13 * np.sin(lon * 4 + i) * np.cos(lat * 2) * np.cos(lat)
            + 0.10 * np.cos(lon * 3 - i) * np.cos(lat * 4) * np.cos(lat)
        )
        verts = (middle + unit * axes * noise[:, :, None]).reshape(-1, 3)
        normals = (unit / axes).reshape(-1, 3)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-7)
        uv = np.stack([lon / (2 * np.pi), 0.5 - lat / np.pi], -1).reshape(-1, 2)
        # The reconstruction has finite volume and can be seen from the opposite side.
        parts.append((verts[tri], normals[tri], uv[tri]))
    trunk = []
    ring = base + np.array(
        [
            [math.cos(a) * radius * 0.075, 0, math.sin(a) * radius * 0.075]
            for a in np.arange(7) * 2 * np.pi / 7
        ]
    )
    for i, p in enumerate(ring):
        q = ring[(i + 1) % len(ring)]
        lift = np.array([0, trunk_top, 0])
        verts = np.array([p, p + lift, q, q, p + lift, q + lift])
        n = (p + q) / 2 - base
        n /= max(np.linalg.norm(n), 1e-7)
        trunk.append((verts, np.tile(n, (6, 1)), np.zeros((6, 2))))
    return base, radius, parts, trunk


def boundary_segments(mark, roofs, roads, lo, hi):
    """Clip ground-traced courtyard runs to free ground around the reconstructed buildings."""
    from mesh import building_faces, subdivide

    polygons = [np.float32(roof[:, [0, 2]]) for roof in roofs]
    for line in mark.data.get("boundaries", []):
        points = mark.cam.plane(np.asarray(line) / mark.size, 0)
        for a, b in zip(points, points[1:]):
            delta = b - a
            length = np.linalg.norm(delta)
            if length < 0.55 or length > 35:
                continue
            intervals = max(1, math.ceil(length / 0.65))
            midpoints = a + (np.arange(intervals)[:, None] + 0.5) * delta / intervals
            x, z = midpoints[:, 0], midpoints[:, 2]
            clear = (x > lo[0]) & (x < hi[0]) & (z > lo[1]) & (z < hi[1])
            for road in roads:
                clear &= road_distance(x, z, road)[0] > road["widthM"] / 2 + 0.22
            for i in np.flatnonzero(clear):
                if any(cv2.pointPolygonTest(poly, (x[i], z[i]), False) >= 0 for poly in polygons):
                    clear[i] = False
            changes = np.diff(np.r_[0, clear, 0].astype(int))
            across = np.array([-delta[2], 0, delta[0]]) * (0.14 / length)
            for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                if (last - first) * length / intervals < 0.55:
                    continue
                s, t = a + delta * first / intervals, a + delta * last / intervals
                roof = np.array([s + across, t + across, t - across, s - across]) + [0, 1.35, 0]
                group = []
                for face, normal in building_faces(roof, 0):
                    v = subdivide(face, 0)
                    group.append((v, np.tile(normal, (len(v), 1)), np.zeros((len(v), 2))))
                yield group


def add_context(glb, plan, cameras, roofs, marks, materials, bank):
    from mesh import building_faces, subdivide, SurfaceAtlas

    if not marks or not roofs:
        return {}
    settings = reviewed_streets(plan.get("ground", {}), marks, cameras)
    lo, hi = ground_bounds(ground_anchors(plan, cameras, roofs), settings.get("marginPercent", 25))
    existing = list(roofs)
    canopies = []
    result = {}
    coping = glb.material("Courtyard eaves", [0.40, 0.33, 0.26])
    bark = glb.material("Tree bark", [0.17, 0.115, 0.071])
    masonry = glb.material("Courtyard masonry", [0.46, 0.37, 0.29])
    atlas = SurfaceAtlas(glb)
    for key, mark in marks.items():
        if key not in materials:
            materials[key] = glb.material("Reviewed view " + key, [1, 1, 1], Image.fromarray(mark.image))
        counts = {"infillBuildings": 0, "trees": 0, "boundaryRuns": 0}
        for row, item in enumerate(mark.data.get("infill", [])):
            roof = inferred_roof(mark, item)
            center = roof.mean(0)[[0, 2]]
            if (center < lo).any() or (center > hi).any() or overlaps(roof, existing):
                continue
            if any(
                road_distance(center[0], center[1], road)[0] < road["widthM"] * 0.52
                for road in settings.get("roads", [])
            ):
                continue
            mapping = FaceMapping(
                mark,
                roof,
                mark.pixels(item["roof"]),
                [mark.pixels(uv) if uv else None for uv in item["feet"]],
            )
            groups = defaultdict(list)
            for face, n in building_faces(roof, 0):
                vertices = subdivide(face, 2)
                tex = mapping.texture(face, n, vertices)
                mat = materials[key]
                if tex is None:
                    if abs(n[1]) < 0.1:
                        axis = np.cross([0.0, 1.0, 0.0], n)
                        basis = np.stack((axis, np.cross(axis, n)))
                        rect = (vertices - face[0]) @ basis.T
                        uv = (rect - rect.min(0)) / np.maximum(np.ptp(rect, axis=0), 1e-7)
                        ticks = np.linspace(0, 1, SurfaceAtlas.tile)
                        u, v = np.meshgrid(ticks, ticks)
                        tile = bank.fill(
                            face, n, -row-1, u, v, np.full((*u.shape, 3), [159, 148, 131], np.uint8)
                        )
                        mat, tex = atlas.put(tile, uv)
                    else:
                        mat, tex = coping, np.zeros((len(vertices), 2))
                groups[mat].append((vertices, np.tile(n, (len(vertices), 1)), tex))
            groups[coping].extend(slab_edge(roof))
            glb.mesh("Inferred " + item["name"], groups)
            existing.append(roof)
            counts["infillBuildings"] += 1
        tree_groups = defaultdict(list)
        foliage = None
        for i, item in enumerate(mark.data.get("trees", [])):
            base, radius, leaves, trunk = tree_geometry(mark, item, i + 41)
            xz = base[[0, 2]]
            if (
                (xz < lo).any()
                or (xz > hi).any()
                or any(np.linalg.norm(xz - pos) < 0.65 * (radius + r) for pos, r in canopies)
            ):
                continue
            if any(
                cv2.pointPolygonTest(np.float32(roof[:, [0, 2]]), tuple(xz), False) > 0 for roof in existing
            ):
                continue
            if foliage is None:
                foliage = glb.material("Foliage " + key, [1, 1, 1], leaf_texture(mark))
            tree_groups[foliage].extend(leaves)
            tree_groups[bark].extend(trunk)
            canopies.append((xz, radius))
            counts["trees"] += 1
        if tree_groups:
            glb.mesh("Tree canopies " + key, tree_groups)
        boundaries = []
        for run in boundary_segments(mark, existing, settings["roads"], lo, hi):
            boundaries.extend(run)
            counts["boundaryRuns"] += 1
        if boundaries:
            glb.mesh("Courtyard walls " + key, {masonry: boundaries})
        result[key] = counts
    atlas.finish()
    return result
