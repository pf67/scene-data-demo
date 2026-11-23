"""Find locally repeatable photo detail on visible surfaces of an existing world-space model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from register_step import mesh_depth, read, save
from step_view import report_camera
from orbit import scaled
from surface_depth import cached_depth, fit_surface_depth, image_digest, rays_for
from worker import prepare, read_photo, scene_faces


def correlation(x, y):
    blur = lambda t: cv2.blur(t, (11, 11))
    x, y = x.astype(np.float32), y.astype(np.float32)
    mx, my = blur(x), blur(y)
    vx = np.maximum(blur(x * x) - mx * mx, 0.0)
    vy = np.maximum(blur(y * y) - my * my, 0.0)
    return (blur(x * y) - mx * my) / np.sqrt(np.maximum(vx * vy, 1)), np.minimum(vx, vy)


def audit(args):
    directory = args.directory
    meta = read(directory / "generation.json")
    plan, report = read(args.plan), read(args.report)
    old = next(
        s
        for s in report["supplements"]
        if s["source"] == meta["parent"] and s["status"] == "aligned"
    )
    new = read(directory / "track-quality.json")
    if (
        new["status"] != "aligned"
        or image_digest(args.parent_image) != meta["parentSha256"]
    ):
        raise ValueError("相邻机位需要已校准的原图")
    pair = [args.parent_image, directory / "view.jpg"]
    photos = [read_photo(f) for f in pair]
    cams = [
        scaled(report_camera("", q["camera"]), photo.size)
        for q, photo in zip([old, new], photos)
    ]
    _, _, _, roofs, _ = prepare(plan, dict(args.source))
    faces = scene_faces(plan, roofs)
    fields = [
        fit_surface_depth(
            c, cached_depth(f), faces, plan.get("ground", {}).get("roads", [])
        )
        for c, f in zip(cams, pair)
    ]
    if any(f is None for f in fields):
        raise ValueError("缺少独立深度图")
    # A common resolution and physical metre scale keep the optical-flow and
    # photometric thresholds independent of the original photograph's size.
    cams = [scaled(c, np.rint(c.size * (768 / max(c.size))).astype(int)) for c in cams]
    p, c = cams
    colors = [
        np.array(im.resize(tuple(cam.size.astype(int)), Image.Resampling.LANCZOS))
        for im, cam in zip(photos, cams)
    ]
    zp, zc = [mesh_depth(args.geometry, cam) for cam in cams]
    h, w = zc.shape
    xyz = c.eye + rays_for(c) * zc[:, :, None]
    uv, z = p.pixels(xyz.reshape(-1, 3))
    uv = np.nan_to_num(uv, nan=-1).reshape(h, w, 2).astype(np.float32)
    z = z.reshape(h, w)
    inside = (zc > 0.1) & (z > 0.1) & (uv > 2).all(2) & (uv < p.size - 3).all(2)
    ref = lambda img: cv2.remap(
        img, uv, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    inside &= np.abs(ref(zp) - z) < np.maximum(0.40, 0.010 * z)
    depth = [
        cv2.resize(
            f.depth, tuple(cam.size.astype(int)), interpolation=cv2.INTER_NEAREST_EXACT
        )
        for f, cam in zip(fields, cams)
    ]
    inside &= np.abs(depth[1] - zc) < np.maximum(1.05, 0.035 * zc)
    inside &= np.abs(ref(depth[0]) - z) < np.maximum(1.05, 0.035 * z)
    base = ref(colors[0])
    gray = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    pred, actual = gray(base), gray(colors[1])
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(actual, pred, None)
    ry, rx = np.mgrid[:h, :w].astype(np.float32)
    grid = np.stack([rx, ry], axis=-1)
    sample = grid + flow
    back = dis.calc(pred, actual, None)
    cycle = np.linalg.norm(
        flow + cv2.remap(back, sample, None, cv2.INTER_LINEAR), axis=-1
    )
    warped = cv2.remap(pred, sample, None, cv2.INTER_LINEAR)
    ncc, var = correlation(warped, actual)
    close = (np.linalg.norm(flow, axis=-1) < 6.4) & (cycle < 1.4)
    parent_valid = cv2.remap(np.uint8(inside), sample, None, cv2.INTER_NEAREST) != 0
    eligible = inside & parent_valid & close
    matched = np.uint8(eligible & (ncc > 0.66) & (var > 18))
    # Allow small textureless gaps inside one corroborated patch. Every added
    # pixel must still pass reference visibility, depth and bidirectional motion.
    coarse = cv2.morphologyEx(
        matched, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    matched = eligible & (coarse != 0)
    independent = (
        np.asarray(
            Image.open(directory / "track-region.png").resize(
                (w, h), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    # The old model is an anchor, not a requirement that the two new depth
    # observations reproduce its mistakes. Reproject the independently fitted
    # candidate depth, checking colour, visibility and normal direction in the parent.
    optical = depth[1]
    observed_xyz = c.eye + rays_for(c) * optical[:, :, None]
    puv, oz = p.pixels(observed_xyz.reshape(-1, 3))
    puv = np.nan_to_num(puv, nan=-1).reshape(h, w, 2).astype(np.float32)
    oz = oz.reshape(h, w)
    remap = lambda arr: cv2.remap(arr, puv, None, cv2.INTER_LINEAR)
    depth_pair = (oz > 0.2) & (puv >= 3).all(2) & (puv < p.size - 4).all(2)
    depth_pair &= np.abs(remap(depth[0]) - oz) < np.maximum(0.70, 0.024 * oz)
    norm = [
        cv2.resize(
            f.normals, tuple(cam.size.astype(int)), interpolation=cv2.INTER_LINEAR
        )
        for f, cam in zip(fields, cams)
    ]
    depth_pair &= np.abs((remap(norm[0]) * norm[1]).sum(2)) > 0.68
    pred_depth = remap(gray(colors[0]))
    df = dis.calc(actual, pred_depth, None)
    db = dis.calc(pred_depth, actual, None)
    ds = grid + df
    delta = np.linalg.norm(df, axis=2)
    cycle3 = np.linalg.norm(df + cv2.remap(db, ds, None, cv2.INTER_LINEAR), axis=2)
    n3, v3 = correlation(actual, cv2.remap(pred_depth, ds, None, cv2.INTER_LINEAR))
    local = depth_pair & (delta < 6.4) & (cycle3 < 1.4)
    photo3 = local & (n3 > 0.66) & (v3 > 18)
    photo3 = local & (
        cv2.morphologyEx(
            np.uint8(photo3),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        != 0
    )
    matched = (matched | photo3) & independent
    n, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(matched))
    use = stats[:, cv2.CC_STAT_AREA] >= 30
    use[0] = False
    matched = use[labels]
    mask = directory / "paired-mask.png"
    Image.fromarray(np.uint8(matched) * 255).save(mask)
    ground = read(args.ground_polygons) if args.ground_polygons else []
    floor = np.zeros((h, w), np.uint8)
    for poly in ground:
        cv2.fillPoly(
            floor, [np.int32(np.rint(np.asarray(poly) * c.size / photos[1].size))], 1
        )
    normals = [
        cv2.resize(
            np.abs(f.normals[:, :, 1]),
            tuple(cam.size.astype(int)),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        for f, cam in zip(fields, cams)
    ]
    # An old shell may incorrectly occupy a garden. Independently register the
    # surveyed ground plane in the two photographs before adding a local material.
    rays = rays_for(c)
    floor_z = -c.eye[1] / np.minimum(-1.0e-6, rays[:, :, 1])
    ground_xyz = c.eye + rays * floor_z[:, :, None]
    floor_uv, floor_other_z = p.pixels(ground_xyz.reshape(-1, 3))
    floor_uv = floor_uv.reshape(h, w, 2).astype(np.float32)
    on_ground = (
        (floor != 0)
        & (rays[:, :, 1] < -0.02)
        & (floor_uv >= 2).all(2)
        & (floor_uv < p.size - 3).all(2)
    )
    ground_view = cv2.remap(gray(colors[0]), floor_uv, None, cv2.INTER_LINEAR)
    gf = dis.calc(actual, ground_view, None)
    gb = dis.calc(ground_view, actual, None)
    ground_sample = grid + gf
    gn, gv = correlation(
        actual, cv2.remap(ground_view, ground_sample, None, cv2.INTER_LINEAR)
    )
    gcycle = np.linalg.norm(
        gf + cv2.remap(gb, ground_sample, None, cv2.INTER_LINEAR), axis=2
    )
    prev_depth = cv2.remap(depth[0], floor_uv, None, cv2.INTER_LINEAR)
    floor_other_z = floor_other_z.reshape(h, w)
    on_ground &= (np.abs(depth[1] - floor_z) < np.maximum(1.25, 0.036 * floor_z)) & (
        np.abs(prev_depth - floor_other_z) < np.maximum(1.25, 0.036 * floor_other_z)
    )
    on_ground &= (normals[1] > 0.70) & (
        cv2.remap(normals[0], floor_uv, None, cv2.INTER_LINEAR) > 0.67
    )
    ground_ok = (
        on_ground
        & (gn > 0.70)
        & (gv > 16)
        & (np.linalg.norm(gf, axis=2) < 6.5)
        & (gcycle < 1.2)
    )
    # Independently triangulate the photos' corresponding soil pixels. Keep a
    # sloping yard only when collapsing its height to the established floor
    # shifts neither observation by more than six native pixels.
    look = (
        (floor != 0)
        & photo3
        & (normals[1] > 0.70)
        & (cv2.remap(normals[0], puv, None, cv2.INTER_LINEAR) > 0.67)
    )
    iy, ix = np.nonzero(look)
    if len(ix):
        observed_parent = cv2.remap(puv, ds, None, cv2.INTER_LINEAR)[iy, ix].astype(
            float
        )
        observed_new = np.column_stack([ix, iy]).astype(float)
        proj = lambda v: v.k @ np.column_stack([v.r, v.t])
        homogeneous = cv2.triangulatePoints(
            proj(p), proj(c), observed_parent.T, observed_new.T
        ).T
        terrain = homogeneous[:, :3] / homogeneous[:, 3:]
        flat = terrain.copy()
        flat[:, 1] = 0
        good = np.isfinite(terrain).all(1) & (np.abs(terrain[:, 1]) < 1.1)
        for cam, pts in ((p, observed_parent), (c, observed_new)):
            uv0, z0 = cam.pixels(terrain)
            uv1, _ = cam.pixels(flat)
            scale = float(photos[1].width / c.size[0])
            good &= (
                (z0 > 0.2)
                & (np.linalg.norm(uv0 - pts, axis=1) * scale < 2.5)
                & (np.linalg.norm(uv1 - pts, axis=1) * scale < 6.0)
            )
        ground_ok[iy[good], ix[good]] = True
    yards = []
    step = 0.85
    grid_cells = np.unique(
        np.floor(ground_xyz[ground_ok][:, [0, 2]] / step).astype(int), axis=0
    )
    # Every texel footprint of a proposed ground quad must fit inside a
    # hand-traced yard and its two-photo verified, unobstructed ground.
    for cell in grid_cells:
        xz = (cell + np.array([[0, 0], [1, 0], [1, 1], [0, 1]])) * step
        base_world = np.column_stack([xz[:, 0], np.zeros(4), xz[:, 1]])
        pix, z = c.pixels(base_world)
        if (pix < 1).any() or (pix >= c.size - 1).any() or min(z) < 0.1:
            continue
        x, y, ww, hh = cv2.boundingRect(np.float32(pix))
        roi = np.zeros((hh, ww), np.uint8)
        cv2.fillConvexPoly(roi, np.int32(np.rint(pix - [x, y])), 1)
        if (
            np.count_nonzero(roi) < 5
            or not ground_ok[y : y + hh, x : x + ww][roi != 0].all()
        ):
            continue
        yards.append(
            dict(
                verified=True,
                groundQuad=np.round(xz, 4).tolist(),
                quad=np.round(pix * (np.array(photos[1].size) / c.size), 3).tolist(),
            )
        )
    # The continuous area includes small verified shapes that cannot contain
    # a full rectangular ground cell. The builder clips this same bitmap per texel.
    footprint = np.abs(
        np.gradient(ground_xyz[:, :, 0], axis=1)
        * np.gradient(ground_xyz[:, :, 2], axis=0)
        - np.gradient(ground_xyz[:, :, 0], axis=0)
        * np.gradient(ground_xyz[:, :, 2], axis=1)
    )
    info = dict(
        parent=meta["parent"],
        parentSha256=meta["parentSha256"],
        targetSha256=meta["sha256"],
        size=[w, h],
        pairMatchedPercent=round(100 * matched.mean(), 3),
        closeGroundMeters2=round(len(yards) * step**2, 2),
        matchedYardPixels=int(ground_ok.sum()),
        surveyedGroundM2=round(float(footprint[ground_ok].sum()), 2),
        yardQuads=len(yards),
    )
    save(directory / "surface-audit.json", info)
    save(directory / "yard-quads.json", yards)
    Image.fromarray(np.uint8(ground_ok) * 255).save(directory / "yard-mask.png")
    preview = colors[1].copy()
    preview[matched] = np.uint8(preview[matched] * 0.64 + [16, 76, 44])
    preview[ground_ok] = np.uint8(colors[1][ground_ok] * 0.45 + [113, 73, 3])
    Image.fromarray(preview).save(directory / "verified-areas.jpg", quality=92)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "report", "geometry", "directory", "parent-image"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--source", nargs=2, action="append", required=True)
    p.add_argument("--ground-polygons", type=Path)
    audit(p.parse_args())
