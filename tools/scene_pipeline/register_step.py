"""Register a neighbouring photograph using repeatable details tied to a supported parent surface."""

import argparse
import json
import struct
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from step_view import report_camera, warp_parent
from orbit import make_layout, register, root_sift, scaled, surface_map
from surface_depth import cached_depth, fit_surface_depth, image_digest
from worker import prepare, read_photo, scene_faces
from mesh import building_faces
from photo_projectors import render_triangles, visibility_image


def read(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def save(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )


def mesh_depth(path, src):
    blob = path.read_bytes()
    size = struct.unpack_from("<I", blob, 12)[0]
    glb = SimpleNamespace(
        doc=json.loads(blob[20 : 20 + size]), blob=memoryview(blob)[28 + size :]
    )
    tris = render_triangles(glb, np.empty((0, 3)), np.empty((0, 3), int))
    depth, units = visibility_image(src, tris, longest=int(max(src.size)))
    code = np.array(depth, dtype=np.float32)
    return (code[:, :, 0] * 256 + code[:, :, 1]) * units


def mesh_landmarks(src, target, zbuffer, uv):
    ix, iy = np.clip(np.rint(uv).astype(int), 0, src.size.astype(int) - 1).T
    z = zbuffer[iy, ix]
    ray = np.c_[(uv - src.k[:2, 2]) / np.diag(src.k)[:2], np.ones(len(uv))] @ src.r
    world = src.eye + ray * z[:, None]
    pix, front = target.camera.pixels(world)
    valid = (
        (uv >= 1).all(1)
        & (uv < src.size - 2).all(1)
        & (z > 0.2)
        & (front > 0.2)
        & (pix >= 1).all(1)
        & (pix < target.camera.size - 2).all(1)
    )
    return world, uv, z, pix, valid


def mutual_features(a, b):
    sift = cv2.SIFT_create(nfeatures=18000, contrastThreshold=0.012, edgeThreshold=12)

    def features(image):
        key, desc = sift.detectAndCompute(
            cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY), None
        )
        if desc is None:
            raise ValueError("图像中缺少可复核的纹理特征")
        return np.float32([k.pt for k in key]), root_sift(desc)

    ak, ad = features(a)
    bk, bd = features(b)
    matcher = cv2.BFMatcher()

    def pairs(x, y):
        return {
            m.queryIdx: (m.trainIdx, m.distance / max(n.distance, 1.0e-9))
            for m, n in matcher.knnMatch(x, y, k=2)
            if m.distance < 0.84 * n.distance
        }

    forward, backward = pairs(ad, bd), pairs(bd, ad)
    rows = [
        (i, j, max(r, backward[j][1]))
        for i, (j, r) in forward.items()
        if j in backward and backward[j][0] == i
    ]
    if not rows:
        raise ValueError("相邻照片没有互相匹配的特征")
    i, j, r = np.array(rows).T
    return ak[i.astype(int)], bk[j.astype(int)], r


def spread(indices, a, b, ratios, limit, separation=17):
    selected = []
    for i in indices[np.argsort(ratios[indices], kind="stable")]:
        if all(
            np.linalg.norm(a[i] - a[j]) >= separation
            and np.linalg.norm(b[i] - b[j]) >= separation
            for j in selected
        ):
            selected.append(i)
            if len(selected) == limit:
                break
    return selected


def track_corners(guide, image, a, b):
    """Forward/backward patch tracking with a two-image displacement seed and a contrast check."""
    g = cv2.cvtColor(np.asarray(guide), cv2.COLOR_RGB2GRAY)
    c = cv2.cvtColor(
        np.asarray(image.resize(guide.size, Image.Resampling.LANCZOS)),
        cv2.COLOR_RGB2GRAY,
    )
    ratio = np.array(image.size) / guide.size
    flow = b / ratio - a
    consistent = []
    for i in range(len(a)):
        near = (np.linalg.norm(a - a[i], axis=1) < 75) & (
            np.linalg.norm(a - a[i], axis=1) > 5
        )
        if np.sum(near & (np.linalg.norm(flow - flow[i], axis=1) < 9)) >= 2:
            consistent.append(i)
    if len(consistent) < 4:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    anchors = a[consistent]
    flows = flow[consistent]
    pts = cv2.goodFeaturesToTrack(g, 5500, 0.003, 5, blockSize=5).reshape(-1, 2)
    seeds = []
    start = []
    for pt in pts:
        distances = np.linalg.norm(anchors - pt, axis=1)
        near = np.argsort(distances)[:7]
        if distances[near[0]] > 110:
            continue
        steps = flows[near]
        nominal = np.median(steps, axis=0)
        if np.median(np.linalg.norm(steps - nominal, axis=1)) > 8:
            continue
        start.append(pt)
        seeds.append(pt + nominal)
    if not start:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    start = np.float32(start).reshape(-1, 1, 2)
    kw = dict(
        winSize=(19, 19),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.015),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
    )
    end, ok, _ = cv2.calcOpticalFlowPyrLK(
        g, c, start, np.float32(seeds).reshape(-1, 1, 2), **kw
    )
    rev, back, _ = cv2.calcOpticalFlowPyrLK(c, g, end, start.copy(), **kw)
    keep = (
        (ok.ravel() != 0)
        & (back.ravel() != 0)
        & (np.linalg.norm((rev - start)[:, 0], axis=1) < 0.8)
    )
    pairs = []
    for i in np.flatnonzero(keep):
        p, q = start[i, 0], end[i, 0]
        if (np.minimum(p, q) < 8).any() or (
            np.maximum(p, q) > np.array(guide.size) - 9
        ).any():
            continue
        patch = cv2.getRectSubPix(g, (15, 15), tuple(p))
        other = cv2.getRectSubPix(c, (15, 15), tuple(q))
        if min(patch.std(), other.std()) < 8:
            continue
        corr = cv2.matchTemplate(patch, other, cv2.TM_CCOEFF_NORMED)[0, 0]
        if corr >= 0.84:
            pairs.append((i, corr))
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    ids = np.array([i for i, _ in pairs])
    return start[ids, 0], end[ids, 0] * ratio, np.array([1 - c for _, c in pairs])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--source", nargs=2, action="append", required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--parent-image", type=Path, required=True)
    p.add_argument("--parent-region", type=Path)
    p.add_argument(
        "--geometry",
        type=Path,
        help="Use the previously accepted scene GLB for physical landmark depth",
    )
    args = p.parse_args()
    directory = args.directory
    plan, report, meta = (
        read(args.plan),
        read(args.report),
        read(directory / "generation.json"),
    )
    if (
        image_digest(args.parent_image) != meta["parentSha256"]
        or image_digest(directory / "view.jpg") != meta["sha256"]
    ):
        raise ValueError("输入图像与生成记录不一致")
    first = plan["views"][0]["source"]
    parent = next(
        s
        for s in report["supplements"]
        if s["source"] == meta["parent"] and s["status"] == "aligned"
    )
    _, cameras, images, roofs, _ = prepare(plan, dict(args.source))
    faces = scene_faces(plan, roofs)

    def layout(yaw):
        return make_layout(plan, cameras[first], images[first], roofs, yaw)

    target = layout(meta["targetYaw"])
    source_img = read_photo(args.parent_image)
    src = scaled(report_camera(parent["source"], parent["camera"]), source_img.size)
    if args.parent_region:
        support = (
            np.array(
                Image.open(args.parent_region).resize(
                    source_img.size, Image.Resampling.NEAREST
                )
            )
            > 0
        )
    else:
        spec = next(s for s in plan["supplements"] if s["source"] == parent["source"])
        old, _, _ = register(
            layout(spec["angle"]),
            images[first],
            source_img,
            spec["angle"],
            faces,
            spec["points"],
            relative_view=True,
            image_api=True,
        )
        if not old:
            raise ValueError("父机位复标失败")
        support = old.registration_region.copy()
        mark = (
            Path(__file__).resolve().parents[2]
            / "builder"
            / "surface-marks"
            / (meta["parentSha256"] + ".json")
        )
        if mark.exists():
            data = read(mark)
            mask = np.uint8(support)
            for poly in data.get("projectionExclusions", []):
                cv2.fillPoly(
                    mask,
                    [np.int32(np.rint(np.array(poly) * src.size / data["size"]))],
                    0,
                )
            support = mask.astype(bool)
    candidate = read_photo(directory / "view.jpg")
    guide = read_photo(directory / "guide.jpg")
    a, b, ratios = mutual_features(guide, candidate)
    ca, cb, cr = track_corners(guide, candidate, a, b)
    a = np.vstack([a, ca])
    b = np.vstack([b, cb])
    ratios = np.r_[ratios, cr]
    x, y = np.rint(a).astype(int).T
    world = target.world[y, x]
    ids = target.ids[y, x]
    uv, z = src.pixels(world)
    px = np.clip(np.rint(uv).astype(int), 0, src.size.astype(int) - 1)
    sx, sy = px.T
    if args.geometry:
        da, db, dr = mutual_features(source_img, candidate)
        ca, cb, cr = track_corners(source_img, candidate, da, db)
        direct_uv = np.vstack([da, ca])
        direct_b = np.vstack([db, cb])
        zbuffer = mesh_depth(args.geometry, src)
        mapping = {}
        warp_parent(
            target,
            src,
            np.asarray(source_img),
            faces,
            cached_depth(args.parent_image),
            plan.get("ground", {}).get("roads", []),
            mapping,
        )
        world, uv, z, gp, good = mesh_landmarks(
            src, target, zbuffer, mapping["parent"][y, x]
        )
        good &= np.linalg.norm(gp - a, axis=1) < 4.5
        dw, du, dz, dg, dok = mesh_landmarks(src, target, zbuffer, direct_uv)
        inherited = len(a)
        world = np.vstack([world, dw])
        uv = np.vstack([uv, du])
        z = np.r_[z, dz]
        gp = np.vstack([gp, dg])
        a = np.vstack([a, dg])
        b = np.vstack([b, direct_b])
        ratios = np.r_[ratios, dr, cr]
        gx, gy = np.clip(
            np.rint(dg).astype(int), 0, target.camera.size.astype(int) - 1
        ).T
        ids = np.r_[ids, target.ids[gy, gx]]
        good = np.r_[good, dok]
        sx, sy = np.clip(np.rint(uv).astype(int), 0, src.size.astype(int) - 1).T
        # Outside the previous projection patch, accept only direct original
        # photograph-to-photograph tracks confirmed by both physical mesh depth
        # and the independent monocular estimate. The warped guide alone cannot extend it.
        good[:inherited] &= support[sy[:inherited], sx[:inherited]]
    else:
        gp = np.column_stack([x, y])
        _, parent_z, parent_ids = surface_map(src, faces)
        good = (ids != -2) & (z > 0.2) & (uv >= 2).all(1) & (uv < src.size - 2).all(1)
        good &= (ids == parent_ids[sy, sx]) & (
            np.abs(z - parent_z[sy, sx]) < np.maximum(0.45, 0.009 * z)
        )
        good &= support[sy, sx]
    depth = fit_surface_depth(
        src,
        cached_depth(args.parent_image),
        faces,
        plan.get("ground", {}).get("roads", []),
    )
    if depth is not None:
        dx, dy = np.clip(
            np.rint(uv * (depth.camera.size / src.size)).astype(int),
            0,
            depth.camera.size.astype(int) - 1,
        ).T
        source_z = depth.depth[dy, dx]
        good &= np.abs(source_z - z) < np.maximum(0.8, 0.027 * z)
    elif args.geometry:
        good &= support[sy, sx]
    # Independently rotating/repeated roof textures can match a distant house.
    # Require a consistent measured displacement in neighbouring photo details.
    delta = b - a * (np.array(candidate.size) / guide.size)
    for i in np.flatnonzero(good):
        neighbours = (np.linalg.norm(a - a[i], axis=1) < 72) & (
            np.linalg.norm(delta - delta[i], axis=1) < 10
        )
        neighbours &= np.linalg.norm(a - a[i], axis=1) > 5
        if np.count_nonzero(neighbours) < 2:
            good[i] = False
    # Split by physical house or 8-m ground cell, before pose fitting. Every
    # point on a held-out building is excluded from training across all faces.
    house_for_face = np.repeat(
        np.arange(len(roofs)),
        [
            len(list(building_faces(roof, b.get("ridgeRiseM", 0))))
            for roof, b in zip(roofs, plan["buildings"])
        ],
    )
    ground = np.floor(world[:, [0, 2]] / 8).astype(int)
    blocks = np.where(
        ids >= 0,
        house_for_face[np.maximum(ids, 0)],
        10000 + (ground[:, 0] + 1000) * 2048 + ground[:, 1],
    )
    # Balance the split by number of *spatially distinct* features, without
    # consulting fitted errors. A mod hash alone can reserve only empty plots.
    groups = sorted(
        np.unique(blocks[good]),
        key=lambda k: (
            -len(spread(np.flatnonzero(good & (blocks == k)), a, b, ratios, 64)),
            k,
        ),
    )
    counts = [0, 0, 0]
    bins = [[], [], []]
    for k in groups:
        n = len(spread(np.flatnonzero(good & (blocks == k)), a, b, ratios, 64))
        j = int(np.argmin(counts))
        bins[j].append(k)
        counts[j] += n
    check = np.isin(blocks, bins[2])
    train = spread(np.flatnonzero(good & ~check), a, b, ratios, 47)
    test = spread(np.flatnonzero(good & check), a, b, ratios, 17)
    print(
        "matched features",
        len(a),
        "on parent's supported surfaces",
        int(good.sum()),
        "train",
        len(train),
        "check",
        len(test),
        flush=True,
    )
    if len(train) < 6 or len(test) < 3:
        raise ValueError("当前重叠区的独立地物不足")
    indices = train + test
    pts = [
        dict(
            guide=(gp[i] / (target.camera.size - 1)).tolist(),
            image=(b[i] / (np.array(candidate.size) - 1)).tolist(),
            **({"world": world[i].tolist()} if args.geometry else {}),
            **({"check": True} if i in test else {}),
        )
        for i in indices
    ]
    save(directory / "track-points.json", pts)
    view, q, overlay = register(
        target,
        images[first],
        candidate,
        meta["targetYaw"],
        faces,
        pts,
        relative_view=True,
        image_api=True,
    )
    if view:
        step = (q["actualAngle"] - parent["actualAngle"] + 180) % 360 - 180
        q["parentStepDeg"] = round(step, 1)
        expected_step = meta["requestedStep"]
        minimum = min(abs(expected_step) * 0.35, 4.0)
        if np.sign(step) * np.sign(expected_step) <= 0 or abs(step) < minimum:
            view = None
            q.update(
                status="needs-angle",
                message="相对父图的实测机位没有按请求方向前进，不能作为下一步的父图",
            )
            (directory / "track-region.png").unlink(missing_ok=True)
    save(directory / "track-quality.json", q)
    overlay.save(directory / "track-overlay.jpg", quality=94)
    print({k: v for k, v in q.items() if k not in ("camera", "message")}, flush=True)
    if view:
        region = np.uint8(view.registration_region) * 255
        Image.fromarray(region).save(directory / "track-region.png")
    np.savez_compressed(
        directory / "tracks.npz",
        guide=a,
        image=b,
        parent=uv,
        world=world,
        ids=ids,
        good=good,
        check=check,
        ratios=ratios,
        chosen=indices,
    )


if __name__ == "__main__":
    main()
