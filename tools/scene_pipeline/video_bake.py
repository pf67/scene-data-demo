"""Bake video detail and triangulated eaves that improve measurements on unseen frames."""

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from video_materials import fit_patch, load_mesh, scene_faces
from video_tracks import digest
from geometry import Camera
from register_step import mesh_depth
from video_surfaces import image_key, primitives
from worker import prepare
from orbit import surface_map
from mesh import building_faces
from video_scale import ground_scale


def project(cam, world):
    return (
        cam.pixels(world.reshape(-1, 3))[0]
        .reshape(*world.shape[:-1], 2)
        .astype(np.float32)
    )


def transform(h, xy):
    return cv2.perspectiveTransform(xy.reshape(1, -1, 2).astype(np.float32), h).reshape(
        xy.shape
    )


def remap(image, xy):
    return cv2.remap(
        image,
        xy.astype(np.float32),
        None,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def plane_world(cam, xy, face):
    rays = (
        np.column_stack(((xy - cam.k[:2, 2]) / np.diag(cam.k)[:2], np.ones(len(xy))))
        @ cam.r
    )
    normal = face["normal"]
    depth = ((face["corners"][0] - cam.eye) @ normal) / (rays @ normal)
    return cam.eye + rays * depth[:, None]


class Shot:
    def __init__(self, folder, video, base):
        self.folder = folder
        self.video = video
        self.id = folder.name
        report = json.loads((folder / "reconstruction.json").read_text("utf8"))
        self.size = np.array(report["size"])
        k = np.array(report["k"])
        self.start = Camera("start", self.size, k, base.r, base.t)
        self.cams = {
            f["frame"]: Camera(
                str(f["frame"]), self.size, k, np.array(f["r"]), np.array(f["t"])
            )
            for f in report["frames"]
        }
        self.fps = report["fps"]
        self.images = {}
        self.observations = defaultdict(dict)
        self.boundaries = defaultdict(dict)
        tracks = np.load(folder / "face-tracks.npz", allow_pickle=False)
        self.held = set(tracks["heldout"].tolist())
        initial, owners, boundary = (
            tracks["start"],
            tracks["owners"],
            tracks["boundary"],
        )
        self.frames = sorted(set(self.cams) | self.held)
        for i in self.frames:
            ix, pts = tracks[f"i{i}"], tracks[f"p{i}"]
            for f in np.unique(owners[ix]):
                group = owners[ix] == f
                use = group & ~boundary[ix]
                x = initial[ix[use]]
                y = pts[use]
                fit = fit_patch(x, y)
                if fit is not None:
                    h, hull, error, check, valid = fit
                    self.observations[int(f)][i] = (
                        h,
                        hull,
                        error,
                        x[valid],
                        y[valid],
                        check[valid],
                    )
                    e = group & boundary[ix]
                    src = initial[ix[e]]
                    if len(src) >= 3:
                        dist = np.linalg.norm(transform(h, src) - pts[e], axis=1)
                        corners = src[dist < 2.8]
                        if len(corners) >= 3:
                            self.boundaries[int(f)][i] = corners
        tracks.close()
        cap = cv2.VideoCapture(str(video))
        try:
            for i in range(max(self.frames) + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise ValueError("录像中途解码失败")
                if i in self.frames:
                    self.images[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

    def survey_ground(self, plan):
        polygons = []
        for road in plan.get("ground", {}).get("roads", []):
            center = np.asarray(road["points"], float)
            for a, b in zip(center, center[1:]):
                v = b - a
                side = (
                    v[[1, 0]]
                    * [1, -1]
                    / max(1e-6, np.linalg.norm(v))
                    * road["widthM"]
                    * 0.37
                )
                xz = np.array([a - side, b - side, b + side, a + side])
                world = np.column_stack((xz[:, 0], np.zeros(4), xz[:, 1]))
                poly, z = self.start.pixels(world)
                if z.min() > 1:
                    polygons.append(poly)
        self.scale_quality = (
            ground_scale(self.video, self.cams, self.start, polygons)
            if polygons
            else None
        )
        self.ground_cams = {}
        if not self.scale_quality:
            return
        audit = self.scale_quality
        for i, pose in audit.pop("local", {}).items():
            self.ground_cams[i] = Camera(
                str(i),
                self.size,
                self.start.k,
                np.array(pose["r"]),
                np.array(pose["t"]),
            )
        audit["locallyCalibratedFrames"] = len(self.ground_cams)
        audit["applied"] = bool(
            audit["afterPx"] < 5
            and audit["afterPx"] < audit["beforePx"]
            and audit["spread"] < 0.075
        )
        if audit["applied"]:
            for c in self.cams.values():
                eye = self.start.eye + audit["scale"] * (c.eye - self.start.eye)
                c.t = -c.r @ eye


def semantic_reference(plan, roofs, cam):
    shells = []
    names = []
    normals = []
    for b, roof in zip(plan["buildings"], roofs):
        for face, n in building_faces(roof, b.get("ridgeRiseM", 0)):
            shells.append((face, n))
            names.append(b["id"])
            normals.append(n)
    _, _, ids = surface_map(cam, shells)
    return ids, np.array(names), np.array(normals)


def base_pixels(glb, material, cache):
    if material in cache:
        return cache[material]
    pbr = glb.doc["materials"][material].get("pbrMetallicRoughness", {})
    base = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3])
    texture = pbr.get("baseColorTexture")
    if texture:
        image = glb.doc["images"][glb.doc["textures"][texture["index"]]["source"]]
        view = glb.doc["bufferViews"][image["bufferView"]]
        i = view.get("byteOffset", 0)
        source = np.asarray(
            Image.open(io.BytesIO(glb.blob[i : i + view["byteLength"]])).convert("RGB")
        )
        if np.any(base != 1):
            source = np.uint8(np.clip(source * base ** (1 / 2.2), 0, 255))
    else:
        source = np.uint8(np.clip(255 * base ** (1 / 2.2), 0, 255))[None, None, :]
    cache[material] = source
    return source


def tile_grid(face, part, image):
    basis, lo, hi = face["basis"], face["lo"], face["hi"]
    extent = hi - lo
    width, height = np.maximum(
        24, np.rint(extent * min(20.0, 420 / max(extent))).astype(int)
    )
    wh = np.array([width, height])
    xs, ys = np.meshgrid(
        np.linspace(lo[0], hi[0], width), np.linspace(lo[1], hi[1], height)
    )
    world = np.stack([xs, ys], -1) @ basis + face["normal"] * (
        face["corners"][0] @ face["normal"]
    )
    uv = np.zeros((height, width, 2), np.float32)
    inside = np.zeros((height, width), bool)
    for tri, old in zip(face["vertices"], part[3][face["indices"]]):
        a, b, c = (tri @ basis.T - lo) / extent * (wh - 1)
        start = np.maximum(0, np.floor(np.minimum(np.minimum(a, b), c)).astype(int))
        end = np.minimum(wh, np.ceil(np.maximum(np.maximum(a, b), c)).astype(int) + 1)
        x0, y0 = start
        x1, y1 = end
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = np.array([b - a, c - a]).T
        if abs(np.linalg.det(m)) < 1.0e-6:
            continue
        weights = (np.stack([xx, yy], -1) - a) @ np.linalg.inv(m).T
        used = (weights.min(-1) >= -0.001) & (weights.sum(-1) <= 1.001)
        uv[y0:y1, x0:x1][used] = old[0] + weights[used] @ (old[1:] - old[0])
        inside[y0:y1, x0:x1] |= used
    # Extend valid boundary texels for bilinear filtering.
    if not inside.any():
        return None
    if not inside.all():
        _, labels = cv2.distanceTransformWithLabels(
            np.uint8(~inside), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        table = np.zeros((labels.max() + 1, 2), np.float32)
        table[labels[inside]] = uv[inside]
        uv = table[labels]
    base = remap(image, np.clip(uv, 0, 1) * image.shape[1::-1] - 0.5)
    return world, inside, base


def hull_mask(hull, xy):
    # Test each convex supporting polygon in the coordinates of the first frame.
    poly = cv2.convexHull(np.float32(hull), clockwise=False).reshape(-1, 2)
    mask = np.ones(xy.shape[:-1], bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        v = b - a
        mask &= v[0] * (xy[..., 1] - a[1]) - v[1] * (xy[..., 0] - a[0]) >= 0
    return mask


def observations(shot, face, stage, source_xy, full=False):
    h, hull, *_ = shot.observations[face["number"]][stage]
    xy = transform(h, source_xy)
    supported = (
        np.vstack((hull, shot.boundaries[face["number"]].get(stage, [])))
        if full and stage in shot.boundaries[face["number"]]
        else hull
    )
    good = (
        hull_mask(supported, source_xy)
        & (xy > 2).all(-1)
        & (xy < shot.size - 3).all(-1)
    )
    return remap(shot.images[stage], xy), good


def evidence_for_face(shot, face):
    obs = shot.observations[face["number"]]
    coords = project(shot.start, face["corners"])
    original = max(1, cv2.contourArea(coords))
    centroid = coords.mean(0)
    choices = []
    for i in sorted(set(obs) & set(shot.cams)):
        if i / shot.fps < 0.50:
            continue
        h, hull, err, *_ = obs[i]
        current = transform(h, coords)
        if not np.isfinite(current).all() or not cv2.isContourConvex(current):
            continue
        coverage = cv2.contourArea(hull) / original
        gain = cv2.contourArea(current) / original
        if (
            not 0.65 < gain < 14
            or coverage < 0.24
            or np.linalg.norm(current.mean(0) - centroid) < 5
        ):
            continue
        # The camera can drift across a generated shot. A measured local
        # homography must still describe the same size and orientation as the physical face.
        model = project(shot.cams[i], face["corners"])
        local = current - current.mean(0)
        metric = model - model.mean(0)
        span = max(16, np.linalg.norm(np.ptp(current, axis=0)))
        if np.median(np.linalg.norm(local - metric, axis=1)) > max(4, 0.13 * span):
            continue
        if np.linalg.norm(current.mean(0) - model.mean(0)) > max(10, 0.36 * span):
            continue
        others = sorted(
            j
            for j in obs
            if j in shot.cams and j != i and abs(j - i) / shot.fps >= 0.16
        )
        if len(others) < 2:
            continue
        near = sorted(others, key=lambda j: abs(j - i))
        supports = [near[0]]
        alternate = next((j for j in near[1:] if (j - i) * (near[0] - i) < 0), near[1])
        supports.append(alternate)
        withheld = sorted(shot.held & set(obs), key=lambda j: abs(j - i))
        if not withheld or abs(i - withheld[0]) / shot.fps > 1:
            continue
        choices.append(
            (
                min(3.0, gain) ** 0.6 * min(0.95, coverage) / (1 + 0.2 * err),
                i,
                supports,
                withheld[:2],
            )
        )
    return sorted(choices, reverse=True)[:3]


def color_error(a, b):
    return np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=-1)


def find_texture(face, grid, shots, base_cam, mesh_z, semantics):
    world, inside, old = grid
    refpix = project(base_cam, world)
    _, z = base_cam.pixels(world.reshape(-1, 3))
    z = z.reshape(inside.shape)
    visible = inside & (refpix > 2).all(-1) & (refpix < base_cam.size - 3).all(-1)
    visible &= np.abs(remap(mesh_z, refpix) - z) < np.maximum(0.32, 0.008 * z)
    ids, names, normals = semantics
    ix, iy = np.clip(
        np.rint(refpix).astype(int), 0, base_cam.size.astype(int) - 1
    ).transpose(2, 0, 1)
    index = ids[iy, ix]
    name = face["node"].removeprefix("Photo building ")
    expected = (
        (index >= 0)
        & (names[np.maximum(index, 0)] == name)
        & ((normals[np.maximum(index, 0)] @ face["normal"]) > 0.80)
    )
    if np.count_nonzero(visible & expected) / max(1, visible.sum()) < 0.70:
        return None
    visible &= expected
    if visible.mean() < 0.22:
        return None
    best = None
    for shot in shots:
        opts = evidence_for_face(shot, face)
        if not opts:
            continue
        xy = project(shot.start, world)
        for _, i, neighbours, withheld in opts:
            boundary = shot.boundaries[face["number"]]
            boundary_ok = all(j in boundary for j in [i, *neighbours, *withheld])
            proposed, valid = observations(shot, face, i, xy, boundary_ok)
            valid &= visible
            costs = []
            old_costs = []
            for j in neighbours:
                check, ok = observations(shot, face, j, xy, boundary_ok)
                valid &= ok
                costs.append(cv2.blur(color_error(check, proposed), (5, 5)))
                old_costs.append(cv2.blur(color_error(check, old), (5, 5)))
            residual = np.maximum(*costs)
            valid &= residual < 20
            valid = cv2.morphologyEx(
                valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            distance = cv2.distanceTransform(valid, cv2.DIST_L2, 5)
            alpha = np.minimum(1, distance / 6.0)
            covered = (alpha > 0.16).sum() / inside.sum()
            if (
                covered < (0.48 if face["normal"][1] > 0.4 else 0.30)
                or (alpha > 0.65).sum() < 90
            ):
                continue
            strong = alpha > 0.65
            diffuse_old = cv2.GaussianBlur(old.astype(np.float32), (0, 0), 3.0)
            diffuse_new = cv2.GaussianBlur(proposed.astype(np.float32), (0, 0), 3.0)
            shift = color_error(diffuse_old, diffuse_new)
            if boundary_ok and covered > 0.85 and np.mean(visible[inside]) > 0.91:
                correction = proposed.astype(float) - old
                mode = "rectified"
            else:
                # A fragment cannot introduce a new window or a different
                # roof colour. Retain the underlying material and transfer fine
                # detail only where the coarse appearance also agrees.
                if np.median(shift[strong]) > 18:
                    continue
                alpha *= np.clip((25 - shift) / 12, 0, 1)
                correction = np.clip(
                    (proposed - diffuse_new) - (old - diffuse_old), -24, 24
                )
                mode = "detail"
            new = np.uint8(
                np.rint(np.clip(old + correction * alpha[:, :, None], 0, 255))
            )
            before = after = 0.0
            n = 0
            validation = []
            for j in withheld:
                check, ok = observations(shot, face, j, xy, boundary_ok)
                ok &= visible & (alpha > 0.16)
                if ok.sum() < 60:
                    continue
                before += float(color_error(old, check)[ok].sum())
                after += float(color_error(new, check)[ok].sum())
                n += int(ok.sum())
                validation.append(j)
            if not n or before - after < max(0.65 * n, 0.075 * before):
                continue
            quality = covered * (before - after) / n
            if best is None or quality > best[0]:
                row = dict(
                    source=shot.id,
                    frame=i,
                    support=neighbours,
                    heldOutFrames=validation,
                    mode=mode,
                    coverage=round(float(covered), 3),
                    heldOutTexels=n,
                    beforeMAE=round(before / n, 2),
                    afterMAE=round(after / n, 2),
                )
                best = (
                    quality,
                    row,
                    Image.fromarray(new),
                    Image.fromarray(old),
                    Image.fromarray(np.uint8(alpha * 255)),
                )
    return best


def geometry_measure(face, shot):
    result = []
    obs = shot.observations[face["number"]]
    cameras = getattr(shot, "ground_cams", shot.cams)
    for i in sorted(set(obs) & set(cameras)):
        cam = cameras[i]
        h, hull, err, x, y, reserved = obs[i]
        if i == 0 or len(x) < 14:
            continue
        world = plane_world(shot.start, x, face)
        direction = (world - shot.start.eye) @ cam.r.T
        center = (shot.start.eye - cam.eye) @ cam.r.T
        observed = (y - cam.k[:2, 2]) / np.diag(cam.k)[:2]
        a = direction[:, :2] - observed * direction[:, 2:]
        b = center[:2] - observed * center[2]
        scales = -(a * b).sum(1) / np.maximum((a * a).sum(1), 1e-9)
        parallax = np.linalg.norm(b, axis=1) / np.linalg.norm(world - cam.eye, axis=1)
        train = (
            (~reserved)
            & (parallax > 0.045)
            & np.isfinite(scales)
            & (scales > 0.65)
            & (scales < 1.35)
        )
        if train.sum() < 7 or reserved.sum() < 3:
            continue
        s = float(np.median(scales[train]))
        if (
            np.median(np.abs(scales[train] - s))
            * abs(face["corners"][0, 1] - shot.start.eye[1])
            > 1.6
        ):
            continue
        check = (world[reserved], y[reserved], cam)
        result.append((s, check, i))
    return result


def measure_height(face, shots, eye):
    if face["normal"][1] < 0.997 or face["area"] < 12:
        return None
    prior = float(face["corners"][:, 1].mean())
    samples = [geometry_measure(face, s) for s in shots]
    per_clip = [
        (k, float(np.median([x[0] for x in rows])))
        for k, rows in enumerate(samples)
        if len(rows) >= 2
    ]
    fits = []
    for k, s in per_clip:
        corroborated = [
            (j, v) for j, v in per_clip if abs((s - v) * (prior - eye[1])) < 0.65
        ]
        if len(corroborated) < 2:
            continue
        fitted = float(np.median([v for _, v in corroborated]))
        new = eye[1] + fitted * (prior - eye[1])
        if not 0.14 < abs(new - prior) < 1.8 or not 0.8 < new < 25:
            continue
        if (
            np.linalg.norm(
                (face["corners"][:, [0, 2]] - eye[[0, 2]]) * (fitted - 1), axis=1
            ).max()
            > 2.25
        ):
            continue
        prior_err = []
        now_err = []
        clips = []
        for j, _ in corroborated:
            a = []
            b = []
            for _, (w, p, cam), _ in samples[j]:
                a.extend(np.linalg.norm(project(cam, w) - p, axis=1))
                b.extend(
                    np.linalg.norm(project(cam, eye + fitted * (w - eye)) - p, axis=1)
                )
            if np.median(b) + 0.4 >= np.median(a) or np.median(b) > 5.5:
                continue
            prior_err.extend(a)
            now_err.extend(b)
            clips.append(shots[j].id)
        if len(clips) < 2:
            continue
        before, after = map(float, (np.median(prior_err), np.median(now_err)))
        if after > 0.80 * before:
            continue
        fits.append(
            (
                len(clips),
                before - after,
                dict(
                    scale=[fitted, new / prior, fitted],
                    translation=[eye[0] * (1 - fitted), 0, eye[2] * (1 - fitted)],
                    oldHeight=round(prior, 3),
                    newHeight=round(float(new), 3),
                    clips=clips,
                    checks=len(prior_err),
                    beforePx=round(before, 3),
                    afterPx=round(after, 3),
                ),
            )
        )
    return max(fits, key=lambda v: v[:2])[-1] if fits else None


def clash(old, new, others):
    def area(a, b):
        return max(0, cv2.intersectConvexConvex(a, b)[0])

    return any(area(new, b) - area(old, b) > 0.40 for b in others)


def bake(args):
    plan = json.loads(args.plan.read_text("utf-8-sig"))
    key = plan["views"][0]["source"]
    _, cams, photos, roofs, _ = prepare(plan, {key: args.reference})
    base = cams[key]
    glb = load_mesh(args.geometry)
    faces = scene_faces(glb)
    parts = {}
    for i in glb.doc["scenes"][0]["nodes"]:
        node = glb.doc["nodes"][i]
        if any(f["node"] == node["name"] for f in faces):
            parts[node["name"]] = primitives(glb, node)
    by_sha = {digest(p)[:12]: p for p in args.videos}
    shots = [
        Shot(args.sfm / h, p, base)
        for h, p in sorted(by_sha.items())
        if (args.sfm / h / "face-tracks.npz").is_file()
    ]
    if not shots:
        raise ValueError("没有已跟踪并校准的视频")
    for s in shots:
        s.survey_ground(plan)
    args.out.mkdir(parents=True, exist_ok=True)
    qa = args.out / "tiles"
    qa.mkdir(exist_ok=True)
    mesh_z = mesh_depth(args.geometry, base)
    semantics = semantic_reference(plan, roofs, base)
    textures = {}
    entries = []
    nodes = {}
    footprints = {}
    (args.out / "ground-scale.json").write_text(
        json.dumps({s.id: s.scale_quality for s in shots}, indent=2)
    )
    by_name = {
        b["id"]: np.float32(
            next(o for o in b["observations"] if o["source"] == key)["roof"]
        )
        * base.size.astype(np.float32)
        for b in plan["buildings"]
        if any(o["source"] == key for o in b["observations"])
    }
    identifiable = set()
    for name, nodeparts in parts.items():
        if name.startswith("Courtyard walls "):
            continue
        xyz = np.vstack([v.reshape(-1, 3) for _, v, _, _ in nodeparts])
        footprints[name] = cv2.convexHull(np.float32(xyz[:, [0, 2]]))
        roof = [
            f["corners"]
            for f in faces
            if f["node"] == name and f["normal"][1] > 0.5 and f["area"] > 10
        ]
        drawn = by_name.get(name.removeprefix("Photo building "))
        if not roof or drawn is None:
            continue
        poly = cv2.convexHull(project(base, np.concatenate(roof)))
        intersection = max(0, cv2.intersectConvexConvex(drawn, poly)[0])
        if (
            intersection
            / max(1, cv2.contourArea(drawn) + cv2.contourArea(poly) - intersection)
            > 0.56
        ):
            identifiable.add(name)
    for f in faces:
        name = f["node"]
        if name not in identifiable:
            continue
        if not any(f["number"] in shot.observations for shot in shots):
            continue
        row = nodes.setdefault(name, dict(signature=f["signature"], faces=[]))
        if "scale" not in row and name in footprints:
            change = measure_height(f, shots, base.eye)
            if change:
                shape = footprints[name]
                s = change["scale"]
                t = change["translation"]
                new = np.float32(shape * s[0] + [t[0], t[2]])
                if not clash(
                    shape, new, [q for n, q in footprints.items() if n != name]
                ):
                    row.update(change)
                    footprints[name] = new
        part = parts[name][f["primitive"]]
        grid = tile_grid(f, part, base_pixels(glb, part[0], textures))
        if grid is None:
            continue
        accepted = find_texture(f, grid, shots, base, mesh_z, semantics)
        if accepted:
            _, stats, tile, old, alpha = accepted
            code = f"{f['number']:04d}"
            for suffix, im in (("before", old), ("after", tile), ("mask", alpha)):
                im.save(qa / (code + "-" + suffix + ".png"))
            face = dict(
                kind="roof" if f["normal"][1] > 0.4 else "wall",
                primitive=f["primitive"],
                triangles=f["indices"].tolist(),
                **stats,
            )
            row["faces"].append(face)
            entries.append((tile, f, face))
    nodes = {name: row for name, row in nodes.items() if row["faces"] or "scale" in row}
    if not nodes:
        raise ValueError("没有通过复核的表面")
    # Use one padded atlas with one continuous material per verified face.
    width = 2048
    pad = 3
    x = y = line = 0
    places = []
    for tile, _, _ in entries:
        w, h = tile.size
        if x + w + 2 * pad > width:
            x = 0
            y += line
            line = 0
        places.append((x + pad, y + pad))
        x += w + 2 * pad
        line = max(line, h + 2 * pad)
    height = max(y + line, 8)
    atlas = Image.new("RGB", (width, height))
    for (tile, f, face), (left, top) in zip(entries, places):
        atlas.paste(
            Image.fromarray(
                np.pad(np.asarray(tile), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            ),
            (left - pad, top - pad),
        )
        wh = np.array(tile.size)
        size = np.array(atlas.size)
        coef = (wh - 1) / (f["hi"] - f["lo"]) / size
        shift = (np.array([left, top]) + 0.5) / size - f["lo"] * coef
        face["basis"] = np.column_stack((f["basis"] * coef[:, None], shift)).tolist()
    atlas.save(args.out / "atlas.jpg", quality=96, subsampling=0)
    manifest = dict(
        version=1,
        imageKey=image_key(photos[key]),
        atlasSha256=digest(args.out / "atlas.jpg"),
        nodes=nodes,
    )
    (args.out / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf8")
    )
    print(
        "accepted",
        len(nodes),
        "buildings",
        sum("scale" in row for row in nodes.values()),
        "heights",
        len(entries),
        "faces",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference", "geometry", "sfm", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--videos", nargs="+", type=Path, required=True)
    cv2.setNumThreads(4)
    bake(p.parse_args())


if __name__ == "__main__":
    main()
