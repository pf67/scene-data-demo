"""Fuse registered depth into one world-space surface for unrestricted aerial orbit."""

import cv2
import numpy as np
from global_material import unseen_texture
from PIL import Image
from surface_depth import rays_for


def grid_depth(field, lo, step, shape, max_height):
    """Project actual depth samples onto the common XZ grid. Occluded columns stay empty."""
    points = rays_for(field.camera) * field.depth[:, :, None] + field.camera.eye
    y = points[:, :, 1]
    ix = np.rint((points[:, :, 0] - lo[0]) / step[0]).astype(int)
    iz = np.rint((points[:, :, 2] - lo[1]) / step[1]).astype(int)
    h, w = shape
    good = (
        (ix >= 0)
        & (ix < w)
        & (iz >= 0)
        & (iz < h)
        & np.isfinite(y)
        & (y > -max_height * 0.45)
        & (y < max_height)
    )
    top = np.full(shape, -np.inf, np.float32)
    if field.accepted_region is not None:
        good &= field.accepted_region
    np.maximum.at(top.ravel(), (iz[good] * w + ix[good]), y[good])
    known = np.isfinite(top)
    if not known.any():
        return np.zeros(shape, np.float32), np.full(shape, np.inf, np.float32)
    # Preserve the upper surface at silhouettes. Only bridge holes within the
    # world-space sample radius, never a roof-sized optical shadow.
    distance, labels = cv2.distanceTransformWithLabels(
        (~known).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros(int(labels.max()) + 1, np.float32)
    lut[labels[known]] = top[known]
    return np.maximum(lut[labels], 0), distance * max(step)


def camera_samples(field, points):
    cam = field.camera
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels, z = cam.pixels(points)
    inside = np.isfinite(pixels).all(1) & (pixels >= 0).all(1) & (pixels < cam.size - 1).all(1) & (z > 0.1)
    xy = np.rint(np.where(inside[:, None], pixels, 0)).astype(int)
    if field.accepted_region is not None:
        inside &= field.accepted_region[xy[:, 1], xy[:, 0]]
    return pixels, z, inside, field.depth[xy[:, 1], xy[:, 0]]


def shell_prior(roofs, lo, step, shape):
    height = np.zeros(shape, np.float32)
    for roof in sorted(roofs, key=lambda r: r[:, 1].mean()):
        poly = np.rint((roof[:, [0, 2]] - lo) / step).astype(np.int32)
        cv2.fillConvexPoly(height, poly, float(roof[:, 1].mean()))
    return height


def height_fusion(fields, lo, step, x, z, shell):
    hmax = max(12.0, float(shell.max()) + 12.0, 0.45 * max(f.camera.eye[1] for f in fields))
    shape = x.shape
    samples = [grid_depth(field, lo, step, shape, hmax) for field in fields]
    height = np.array(shell, dtype=np.float32, copy=True)
    owner = np.full(shape, -1, np.int8)
    best = np.full(shape, -np.inf)
    for index, (field, (y, distance)) in enumerate(zip(fields, samples)):
        close = distance < 1.7 * max(step)
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        score = np.maximum(0, 2.0 - distance / max(step))
        for other, independent in enumerate(fields):
            if other == index:
                continue
            _, depth, inside, observed = camera_samples(independent, points)
            tolerance = np.clip(0.018 * depth, 0.65, 3.0)
            agreement = inside & (np.abs(depth - observed) < tolerance)
            score += agreement.reshape(shape) * 1.8
            if other == 0:
                # A rear observation can fill an optical shadow, but a new
                # house standing in front of observed open space is a conflict.
                close &= ~(inside & (depth < observed - 2.0 * tolerance)).reshape(shape)
        if index == 0:
            score += 12.0
        score += np.clip(1 - np.abs(y - shell) / 4, 0, 1) * 0.4
        use = close & (score > best)
        owner[use], height[use], best[use] = index, y[use], score[use]
    known = owner >= 0
    if not known.any():
        return height, owner, known, known.copy()
    distance, labels = cv2.distanceTransformWithLabels(
        (~known).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lookup = np.zeros(labels.max() + 1, np.float32)
    lookup[labels[known]] = height[known]
    fill = (~known) & (distance * max(step) < 2.4)
    height[fill] = lookup[labels[fill]]
    # Remove isolated samples (e.g. an aerial cable) which otherwise grow
    # into tall spikes. Keep real steps in height at continuous roof edges.
    height = cv2.medianBlur(height, 3)
    active = distance * max(step) < 5.5
    # A tall return needs independent support at the same physical location.
    # A reprojected silhouette or two nearly coincident cameras cannot establish
    # the back edge of a roof or canopy.
    confirmed = np.zeros(shape, bool)
    for i, (field, (sample, distance)) in enumerate(zip(fields, samples)):
        rayx, rayy, rayz = field.camera.eye[0] - x, field.camera.eye[1] - height, field.camera.eye[2] - z
        magnitude = np.sqrt(rayx * rayx + rayy * rayy + rayz * rayz)
        for j in range(i):
            other, other_dist = samples[j]
            dx, dy, dz = (
                fields[j].camera.eye[0] - x,
                fields[j].camera.eye[1] - height,
                fields[j].camera.eye[2] - z,
            )
            cosine = (rayx * dx + rayy * dy + rayz * dz) / np.maximum(
                magnitude * np.sqrt(dx * dx + dy * dy + dz * dz), 1e-6
            )
            tolerance = np.clip(0.16 * height, 0.60, 1.35)
            confirmed |= (
                (cosine < 0.965)
                & (distance < 1.65 * max(step))
                & (other_dist < 1.65 * max(step))
                & (np.abs(sample - height) < tolerance)
                & (np.abs(other - height) < tolerance)
            )
    return height, owner, active, confirmed


def low_relief(fields, images, keys, x, z, height, owner, confirmed, step):
    """Keep grounded garden and street detail; tall architecture is represented by solids."""
    world = np.column_stack([x.ravel(), height.ravel(), z.ravel()])
    vegetation = np.zeros(x.shape, bool)
    for i, field in enumerate(fields):
        rgb = images[keys[i]]
        px, _, inside, _ = camera_samples(field, world)
        uv = px / field.camera.size * np.array(rgb.shape[1::-1])
        observed = cv2.remap(
            rgb, uv.reshape(*x.shape, 2).astype(np.float32), None, cv2.INTER_LINEAR
        ).astype(np.float32)
        red, green, blue = np.moveaxis(observed, 2, 0)
        vegetation |= (
            (owner == i)
            & inside.reshape(x.shape)
            & (green > 0.92 * red)
            & (green > blue + 9)
            & (green < 150)
        )
    local = cv2.boxFilter(confirmed.astype(np.float32), -1, (7, 7))
    canopy = cv2.boxFilter(vegetation.astype(np.float32), -1, (7, 7)) > 0.50
    canopy &= local > 0.38
    n, labels, stats, _ = cv2.connectedComponentsWithStats(canopy.astype(np.uint8))
    compact = np.zeros(n, bool)
    if n > 1:
        area = stats[1:, cv2.CC_STAT_AREA] * np.prod(step)
        compact[1:] = (area >= 2.5) & (area <= 125)
    crowns = []
    for label in np.flatnonzero(compact):
        elevated = (labels == label) & (height > 2.0)
        if np.count_nonzero(elevated) * np.prod(step) < 2.1:
            continue
        center = np.array([np.median(x[elevated]), np.median(z[elevated])])
        radius = np.sqrt(np.count_nonzero(elevated) * np.prod(step) / np.pi)
        top = np.percentile(height[elevated], 78)
        crowns.append((center, float(np.clip(radius, 1.0, 4.8)), float(np.clip(top, 2.2, 10.0))))
    # Unverified elevated pixels are often projecting a neighbour's roof. A
    # flat, editable yard is preferable to a wall stretched down to the ground.
    detail = (height < 1.4) & (local > 0.30)
    discard = (~detail) & (height > 0.48)
    owner[discard] = -1
    height[discard] = 0
    height[:] = cv2.medianBlur(height, 3)
    return crowns


def add_confirmed_trees(glb, marks, crowns, roofs):
    from collections import defaultdict

    from surface_context import leaf_texture, tree_geometry

    references = [m for m in marks.values() if m.data.get("trees")]
    if not references or not crowns:
        return 0
    occupied = []
    for mark in references:
        for tree in mark.data["trees"]:
            point = mark.cam.plane([np.asarray(tree["foot"]) / mark.size], 0)[0]
            _, depth = mark.cam.pixels([point])
            scale = depth[0] / mark.cam.k[1, 1] * mark.cam.size[1] / mark.size[1]
            radius = np.clip((tree["box"][2] - tree["box"][0]) * scale * 0.46, 0.65, 6.5)
            occupied.append((point[[0, 2]], radius))
    polygons = [np.float32(roof[:, [0, 2]]) for roof in roofs]
    mats = {m["name"]: i for i, m in enumerate(glb.doc["materials"])}
    group = defaultdict(list)
    count = 0
    for center, radius, top in sorted(crowns, key=lambda c: -c[1]):
        if any(np.linalg.norm(center - pos) < 0.9 * (r + radius) for pos, r in occupied):
            continue
        if any(cv2.pointPolygonTest(poly, tuple(center), True) > -0.55 * radius for poly in polygons):
            continue
        base = np.array([center[0], 0.0, center[1]])
        for mark in references:
            xy, depth = mark.cam.pixels([base])
            if depth[0] < 1 or (xy[0] < 0).any() or (xy[0] > mark.cam.size).any():
                continue
            scale = depth[0] / mark.cam.k[1, 1] * mark.cam.size[1] / mark.size[1]
            px, py = xy[0] / mark.cam.size * mark.size
            half = radius / (0.92 * scale)
            y0 = py - top / (1.07 * scale)
            box = [px - half, y0, px + half, min(py - 0.5 / scale, y0 + 0.72 * top / scale)]
            _, _, leaves, trunk = tree_geometry(mark, {"box": box, "foot": [px, py]}, count + 93)
            key = "Foliage " + mark.source
            if key not in mats:
                mats[key] = glb.material(key, [1, 1, 1], leaf_texture(mark))
            if "Tree bark" not in mats:
                mats["Tree bark"] = glb.material("Tree bark", [0.17, 0.115, 0.071])
            group[mats[key]].extend(leaves)
            group[mats["Tree bark"]].extend(trunk)
            occupied.append((center, radius))
            count += 1
            break
    if group:
        glb.mesh("Multi-view tree crowns", group)
    return count


def write_indexed(glb, name, mat, vertices, normals, uv, faces):
    """Keep each indexed glTF primitive within its 16-bit vertex range."""
    for offset in range(0, len(faces), 20000):
        chunk = faces[offset : offset + 20000]
        used, indices = np.unique(chunk, return_inverse=True)
        glb.mesh(
            name + " " + str(offset // 20000 + 1),
            {mat: [(vertices[used], normals[used], uv[used])]},
            indices.ravel().astype("<u2"),
        )


def graph_triangles(height, active):
    ids = np.arange(height.size, dtype=np.int32).reshape(height.shape)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, :-1], ids[1:, 1:]
    h = height.ravel()
    ad = np.abs(h[a] - h[d]) < np.abs(h[b] - h[c])
    f1 = np.stack([a, c, np.where(ad, d, b)], -1)
    f2 = np.where(ad[:, :, None], np.stack([a, d, b], -1), np.stack([b, c, d], -1))
    tri = np.stack([f1, f2], -2).reshape(-1, 3)
    return tri[np.all(active.ravel()[tri], axis=1)]


def texture_ownership(fields, points, normal):
    source = np.full(len(points), -1, np.int8)
    best = np.full(len(points), -np.inf)
    for i, field in enumerate(fields):
        _, depth, inside, observed = camera_samples(field, points)
        delta = field.camera.eye - points
        cosine = np.sum(delta * normal, axis=1) / np.maximum(np.linalg.norm(delta, axis=1), 0.1)
        tolerance = np.clip(0.030 * depth, 0.8, 4.6)
        difference = np.abs(depth - observed)
        use = inside & (cosine > 0.07) & (difference < tolerance)
        score = cosine + 0.55 * (1.0 - difference / tolerance) + (0.78 if i == 0 else 0.0)
        use &= score > best
        source[use], best[use] = i, score[use]
    return source


def blended_atlas(fields, images, keys, x, z, height, owner, active, texture, locked):
    """Use one continuous material where individual cameras disagree about a surface."""
    step = (x[0, 1] - x[0, 0], z[1, 0] - z[0, 0])
    dz, dx = np.gradient(height, step[1], step[0])
    normals = np.stack([-dx, np.ones_like(dx), -dz], -1)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-5)
    shape = x.shape
    world = np.column_stack([x.ravel(), height.ravel(), z.ravel()])
    total = np.zeros(shape, np.float32)
    colors = np.zeros((*shape, 3), np.float32)
    weights = []
    for i, field in enumerate(fields):
        pixels, depth, inside, observed = camera_samples(field, world)
        rgb = images[keys[i]]
        texcoords = pixels / field.camera.size * np.array(rgb.shape[1::-1])
        mapped = cv2.remap(rgb, texcoords.reshape(*shape, 2).astype(np.float32), None, cv2.INTER_LINEAR)
        delta = field.camera.eye - world
        cosine = (normals.reshape(-1, 3) * delta).sum(1) / np.maximum(np.linalg.norm(delta, axis=1), 0.1)
        scale = np.clip(0.025 * depth, 1.2, 4.0)
        agreement = np.exp(-0.5 * (np.abs(depth - observed) / scale) ** 2)
        strength = (inside * agreement * np.clip((cosine - 0.02) / 0.50, 0, 1)).reshape(shape)
        weight = np.float32(strength * (4.0 if i == 0 else 1.0))
        # A colour observation is useful as broad albedo even on the side
        # where no individual camera can supply a plausible window/roof match.
        albedo = np.maximum(weight, np.float32(owner == i) * 0.15)
        colors += mapped * albedo[:, :, None]
        total += albedo
        weights.append(weight)
    sampled = total > 0.12
    base = cv2.resize(np.asarray(texture), shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    if sampled.any():
        dist, labels = cv2.distanceTransformWithLabels(
            (~sampled).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        lut = np.zeros((int(labels.max()) + 1, 3), np.float32)
        lut[labels[sampled]] = colors[sampled] / total[sampled, None]
        extrapolated = lut[labels]
        # Remove directional windows and scattered occluding foreground from
        # estimates of unseen albedo. The observed high frequencies are restored below.
        extrapolated = cv2.bilateralFilter(extrapolated, 9, 42.0, 4)
        local = np.clip(1.0 - dist * max(step) / 7.0, 0, 1)[:, :, None]
        base = base * (1 - local) + extrapolated * local
    width, size_z = texture.size
    fallback = cv2.resize(
        unseen_texture(base, height, step), (width, size_z), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    level = cv2.resize(height, (width, size_z), interpolation=cv2.INTER_LINEAR)
    sampled_owner = cv2.resize(
        np.where(active, owner, -1).astype(np.float32), (width, size_z), interpolation=cv2.INTER_NEAREST
    )
    source_weights = [cv2.GaussianBlur(w, (5, 5), 0.85) for w in weights]
    atlas = np.empty_like(np.asarray(texture))
    ground = np.array(texture)
    count = 0
    xx_axis = np.linspace(x[0, 0], x[0, -1], width)
    zz_axis = np.linspace(z[0, 0], z[-1, 0], size_z)
    for start in range(0, size_z, 64):
        sl = slice(start, min(start + 64, size_z))
        xx, zz = np.meshgrid(xx_axis, zz_axis[sl])
        yy = level[sl]
        world = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        sum_weight = np.full(xx.shape, 0.28, np.float32)
        sum_rgb = fallback[sl] * 0.28
        ground_weight = np.zeros(xx.shape, np.float32)
        ground_rgb = np.zeros((*xx.shape, 3), np.float32)
        u = ((xx - x[0, 0]) / step[0]).astype(np.float32)
        v = ((zz - z[0, 0]) / step[1]).astype(np.float32)
        for i, field in enumerate(fields):
            px, depth, inside, observed = camera_samples(field, world)
            agreement = np.clip(1.0 - np.abs(depth - observed) / np.clip(0.054 * depth, 2.0, 7.0), 0, 1)
            weight = cv2.remap(source_weights[i], u, v, cv2.INTER_LINEAR)
            weight *= (inside * agreement).reshape(xx.shape)
            rgb = images[keys[i]]
            uv = px / field.camera.size * np.array(rgb.shape[1::-1])
            mapped = cv2.remap(rgb, uv.reshape(*xx.shape, 2).astype(np.float32), None, cv2.INTER_LINEAR)
            sum_rgb += mapped * weight[:, :, None]
            sum_weight += weight
            # The level floor is shared by every orbit. It may only borrow
            # actual near-horizontal, near-ground rays from each donor image.
            floor = field.check(np.where(inside[:, None], px, 0), depth, None, field.camera.size)
            floor &= inside & (np.abs(depth - observed) < np.clip(0.025 * depth, 1.1, 3.0))
            weight_ground = weight * floor.reshape(xx.shape)
            ground_weight += weight_ground
            ground_rgb += mapped * weight_ground[:, :, None]
        color = np.uint8(np.clip(sum_rgb / sum_weight[:, :, None], 0, 255))
        atlas[sl] = color
        change = (sampled_owner[sl] >= 0) & (yy < 0.48) & ~locked[sl] & (ground_weight > 0.32)
        alpha = np.minimum(ground_weight[change], 1.0)[:, None]
        observed_color = ground_rgb[change] / ground_weight[change, None]
        ground[sl][change] = np.uint8(
            np.clip(ground[sl][change] * (1 - alpha) + observed_color * alpha, 0, 255)
        )
        count += int(change.sum())
    return Image.fromarray(atlas), Image.fromarray(ground), count, weights


def add_global_surface(glb, priors, marks, images, materials, roofs, bank, ground, texture, locked, plan):
    fields_by_key = {k: f for k, f in priors.items() if f.anchor_error <= 6.5}
    if not fields_by_key or not roofs:
        return None, texture
    # Prefer the image with most reviewed building correspondences. With
    # equally reviewed observations, keep the original registration order.
    keys = sorted(fields_by_key, key=lambda k: (
        fields_by_key[k].accepted_region is not None, -len(marks[k].buildings) if k in marks else 0
    ))
    fields = [fields_by_key[k] for k in keys]
    lo, hi = np.array(ground["boundsXZ"])
    dims = (
        np.maximum(12, np.minimum(520, np.rint((hi - lo) / max(0.28, max(hi - lo) / 520)).astype(int))) + 1
    )
    step = (hi - lo) / (dims - 1)
    x, z = np.meshgrid(np.linspace(lo[0], hi[0], dims[0]), np.linspace(lo[1], hi[1], dims[1]))
    shell = shell_prior(roofs, lo, step, x.shape)
    height, owner, active, confirmed = height_fusion(fields, lo, step, x, z, shell)
    if active.mean() < 0.05 or np.count_nonzero((owner >= 0) & (height > 0.8)) < 12:
        return None, texture
    # Keep editable input shells available to external GLB editors. The
    # default scene contains each completed physical building only once.
    base = glb.doc["scenes"][0]
    original = base["nodes"].copy()
    construction = glb.add("scenes", {"name": "Construction shells", "nodes": original})
    base["extras"]["construction_scene"] = construction
    base["nodes"] = []

    shell_wall_hits = bank.reviewed_wall_hits.copy()
    shell_sources = bank.wall_sources.copy()
    shell_roofs = bank.roof_sources.copy()
    bank.wall_sources.clear()
    bank.roof_sources.clear()
    bank.reviewed_wall_hits.clear()
    fixed = []
    seen_triangles = {key: 0 for key in keys}
    building_faces_inferred = 0
    for key in keys:
        if key not in marks:
            continue
        from global_structures import add_buildings

        first_building_node = len(base["nodes"])
        new = add_buildings(glb, marks[key], roofs, plan, materials, bank,
                            fields_by_key[key], fields_by_key, images, fixed)
        fixed.extend(new)
        by_mat = {materials[k]: k for k in keys if k in materials}
        for node_id in base["nodes"][first_building_node:]:
            mesh = glb.doc["meshes"][glb.doc["nodes"][node_id]["mesh"]]
            for primitive in mesh["primitives"]:
                accessor = primitive.get("indices", primitive["attributes"]["POSITION"])
                count = glb.doc["accessors"][accessor]["count"] // 3
                if primitive["material"] in by_mat:
                    seen_triangles[by_mat[primitive["material"]]] += count
                else:
                    building_faces_inferred += count
    reviewed_count = len(fixed)
    from surface_context import overlaps

    names = {glb.doc["nodes"][nid]["name"].removeprefix("Photo building ") for nid in base["nodes"]}
    candidates = {b["id"]: r for b, r in zip(plan["buildings"], roofs)}
    restored = 0
    restored_names = set()
    for node_id in original:
        name = glb.doc["nodes"][node_id]["name"]
        if name.startswith(("Tree canopies ", "Courtyard walls ")):
            base["nodes"].append(glb.add("nodes", dict(glb.doc["nodes"][node_id])))
        elif name in candidates and name not in names and not overlaps(candidates[name], fixed):
            base["nodes"].append(glb.add("nodes", dict(glb.doc["nodes"][node_id])))
            fixed.append(candidates[name])
            restored += 1
            restored_names.add(name)
    visible_wall_hits = bank.reviewed_wall_hits | {
        (b, edge) for b, edge in shell_wall_hits if plan["buildings"][b]["id"] in restored_names
    }
    active_sources = {**shell_sources, **bank.wall_sources}
    local_wall_counts = {}
    for face in visible_wall_hits:
        name = active_sources[face]
        local_wall_counts[name] = local_wall_counts.get(name, 0) + 1
    bank.reviewed_wall_hits.update(shell_wall_hits)
    roof_sources = {i: source for i, source in shell_roofs.items() if plan["buildings"][i]["id"] in restored_names}
    roof_sources.update(bank.roof_sources)
    structures = shell.copy()
    cut = np.zeros(height.shape, np.uint8)
    for polygon in fixed:
        points = np.rint((np.asarray(polygon)[:, [0, 2]] - lo) / step).astype(np.int32)
        cv2.fillConvexPoly(cut, points, 1)
        cv2.fillConvexPoly(structures, points, float(np.mean(np.asarray(polygon)[:, 1])))
    dist = cv2.distanceTransform(np.uint8(structures < 0.3), cv2.DIST_L2, 5) * max(step)
    # Monocular axial distance loses metric scale in the distant fields.
    # Limit high relief to the observed settlement and its immediate gardens.
    height *= np.clip(1.0 - (dist - 8.0) / 15.0, 0, 1)
    height = cv2.bilateralFilter(cv2.medianBlur(height, 5), 5, 1.35, 2.0)
    crowns = low_relief(fields, images, keys, x, z, height, owner, confirmed, step)
    confirmed_trees = add_confirmed_trees(glb, marks, crowns, fixed)
    if cut.any():
        cut = cv2.dilate(cut, np.ones((5, 5), np.uint8)).astype(bool)
        height[cut] = 0
        owner[cut] = -1
    height[height < 0.48] = 0.0
    vertices = np.column_stack([x.ravel(), height.ravel(), z.ravel()])
    triangles = graph_triangles(height, active)
    # Level patches live on the editable ground mesh and share its texture.
    triangles = triangles[np.max(height.ravel()[triangles], axis=1) >= 0.48]
    v = vertices[triangles]
    plane = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    normal = plane / np.maximum(np.linalg.norm(plane, axis=1, keepdims=True), 1e-8)
    centers = v.mean(1)
    atlas, texture, count, weights = blended_atlas(
        fields, images, keys, x, z, height, owner, active, texture, locked
    )
    # Strong, contiguous support gets the original photograph at full
    # resolution. Around visibility boundaries both sides meet the same
    # continuously baked multi-view texture rather than alternating photos.
    votes = np.stack([cv2.GaussianBlur(w, (5, 5), 1.2).ravel()[triangles].mean(1) for w in weights])
    source = np.argmax(votes, axis=0)
    confidence = np.max(votes, axis=0)
    chosen = texture_ownership(fields, centers, normal)
    direct = (chosen == source) & (confidence > np.where(source == 0, 1.0, 0.45))
    source[confidence < 0.30] = -1
    normals = np.zeros(vertices.shape)
    for k in range(3):
        np.add.at(normals, triangles[:, k], plane)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    stats = {
        "method": "world-height-fusion",
        "primarySource": keys[0],
        "gridSpacingM": np.round(step, 3).tolist(),
        "coveragePercent": round(100 * float(active.mean()), 2),
        "byView": {},
    }
    for i, (key, field) in enumerate(zip(keys, fields)):
        selected = triangles[(source == i) & direct]
        stats["byView"][key] = int(np.count_nonzero(source == i))
        if not len(selected):
            continue
        if key not in materials:
            materials[key] = glb.material("Fusion view " + key, [1, 1, 1], Image.fromarray(images[key]))
        coords, _, _, _ = camera_samples(field, vertices)
        uv = np.clip(np.nan_to_num(coords / field.camera.size), 0, 1)
        write_indexed(glb, "Photographed surface " + key, materials[key], vertices, normals, uv, selected)
    if np.any(~direct):
        material = glb.material("Continuous surface atlas", [1, 1, 1], atlas)
        uv = (vertices[:, [0, 2]] - lo) / (hi - lo)
        write_indexed(glb, "Fused context", material, vertices, normals, uv, triangles[~direct])
    for key, value in seen_triangles.items():
        stats["byView"][key] += value
    building_faces_seen = sum(seen_triangles.values())
    stats.update(
        triangles=len(triangles) + building_faces_seen + building_faces_inferred,
        contextTriangles=len(triangles),
        closedBuildingTriangles=building_faces_seen + building_faces_inferred,
        inferredTriangles=int((source < 0).sum()) + building_faces_inferred,
        photoGroundTexels=count,
        blendedTriangles=int((~direct).sum()),
        closedAnchorBuildings=reviewed_count,
        retainedBuildings=restored,
        reviewedRearWalls=len(visible_wall_hits),
        localWallsBySource=local_wall_counts,
        localRoofsBySource={s: list(roof_sources.values()).count(s) for s in sorted(set(roof_sources.values()))},
        additionalTrees=confirmed_trees,
    )
    base["extras"]["global_surface"] = True
    return stats, texture
