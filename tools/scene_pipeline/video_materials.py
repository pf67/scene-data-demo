"""Measure the motion of individual roofs and elevations instead of forcing the whole shot onto the mesh."""

import argparse
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from register_step import mesh_depth
from video_tracks import digest, follow
from geometry import Camera
from video_surfaces import planar_faces, primitives, signature
from worker import prepare


def load_mesh(path):
    blob = path.read_bytes()
    size = struct.unpack_from("<I", blob, 12)[0]
    return SimpleNamespace(doc=json.loads(blob[20 : 20 + size]), blob=blob[28 + size :])


def scene_faces(glb):
    faces = []
    for i in glb.doc["scenes"][0]["nodes"]:
        node = glb.doc["nodes"][i]
        name = node.get("name", "")
        if not name.startswith(("house_", "Photo building ", "Courtyard walls ")):
            continue
        parts = primitives(glb, node)
        key = signature(parts)
        for f in planar_faces(parts):
            if f["normal"][1] < -0.2:
                continue
            if name.startswith("Courtyard walls ") and (
                f["normal"][1] > 0.3 or f["area"] > 260
            ):
                continue
            hull = cv2.convexHull(
                np.float32(f["vertices"].reshape(-1, 3) @ f["basis"].T),
                returnPoints=False,
            ).ravel()
            faces.append(
                dict(
                    **f,
                    number=len(faces),
                    node=name,
                    signature=key,
                    corners=f["vertices"].reshape(-1, 3)[hull],
                )
            )
    return faces


def tracking_seeds(gray, camera, depth, faces):
    points = []
    groups = []
    boundary = []
    corners_map = cv2.cornerMinEigenVal(gray, 5)
    for f in faces:
        corners = f["corners"]
        normal = f["normal"]
        direction = camera.eye - corners.mean(0)
        if normal @ direction < 0.13 * np.linalg.norm(direction):
            continue
        uv, z = camera.pixels(corners)
        if z.min() < 1.0 or not np.isfinite(uv).all():
            continue
        lo = np.maximum(0, np.floor(uv.min(0)).astype(int))
        hi = np.minimum(camera.size.astype(int), np.ceil(uv.max(0)).astype(int) + 1)
        if min(hi - lo) < 9:
            continue
        x, y = lo
        xx, yy = hi
        mask = np.zeros((yy - y, xx - x), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(np.rint(uv - lo)), 255)
        py, px = np.mgrid[y:yy, x:xx]
        rays = (
            np.stack(
                [
                    (px - camera.k[0, 2]) / camera.k[0, 0],
                    (py - camera.k[1, 2]) / camera.k[1, 1],
                    np.ones_like(px),
                ],
                -1,
            )
            @ camera.r
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            target = (normal @ (corners[0] - camera.eye)) / (rays @ normal)
        mask[np.abs(target - depth[y:yy, x:xx]) > np.maximum(0.27, 0.007 * target)] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        if np.count_nonzero(mask) < 84:
            continue
        pts = cv2.goodFeaturesToTrack(
            gray[y:yy, x:xx], 190, 0.003, 3.3, mask=mask, blockSize=3
        )
        if pts is None or len(pts) < 10:
            continue
        pts = pts.reshape(-1, 2) + lo
        outer = []
        for corner in cv2.approxPolyDP(np.float32(uv), 1.0, True).reshape(-1, 2):
            cx, cy = np.int32(np.rint(corner))
            if cx < 6 or cy < 6 or cx > gray.shape[1] - 7 or cy > gray.shape[0] - 7:
                continue
            response = corners_map[cy - 4 : cy + 5, cx - 4 : cx + 5]
            dy, dx = np.unravel_index(np.argmax(response), response.shape)
            if response[dy, dx] < 1.0e-5:
                continue
            outer.append([cx + dx - 4, cy + dy - 4])
        # Eave and wall-boundary checks must not participate in fitting the homography.
        if outer:
            pts = pts[
                np.linalg.norm(pts[:, None] - np.asarray(outer)[None], axis=2).min(1)
                > 4.5
            ]
        points.extend(pts)
        groups.extend([f["number"]] * len(pts))
        boundary.extend([False] * len(pts))
        points.extend(outer)
        groups.extend([f["number"]] * len(outer))
        boundary.extend([True] * len(outer))
    return np.float32(points).reshape(-1, 2), np.int32(groups), np.bool_(boundary)


def collect(path, out, base, faces, mesh):
    solution = json.loads((out / "reconstruction.json").read_text())
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("视频无法打开")
    ok, bgr = cap.read()
    if not ok:
        raise ValueError("视频首帧无法解码")
    prev = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    first = Camera(
        "video", np.array(solution["size"]), np.array(solution["k"]), base.r, base.t
    )
    depth = mesh_depth(mesh, first)
    seed, owners, boundary = tracking_seeds(prev, first, depth, faces)
    if len(seed) < 12:
        raise ValueError("可见立面和屋面的起始观测不足")
    pts = seed.copy()
    ids = np.arange(len(seed))
    frames = sorted(f["frame"] for f in solution["frames"])
    # Interleaved frames were excluded from the global reconstruction and from texture selection.
    midpoints = [(a + b) // 2 for a, b in zip(frames, frames[1:]) if b - a >= 3]
    samples = sorted(set(frames + midpoints))
    rows = {}
    sample_set = set(samples)
    try:
        for frame in range(max(samples) + 1):
            if frame:
                ok, bgr = cap.read()
                if not ok:
                    break
                if frame % 2 and frame not in sample_set:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                pts, use = follow(prev, gray, pts)
                pts, ids = pts[use], ids[use]
                prev = gray
            if frame in sample_set:
                rows[f"i{frame}"] = ids.astype(np.int32)
                rows[f"p{frame}"] = pts
    finally:
        cap.release()
    np.savez_compressed(
        out / "face-tracks.npz",
        start=seed,
        owners=owners,
        boundary=boundary,
        heldout=midpoints,
        **rows,
    )
    print(
        "tracked",
        out.name,
        len(seed),
        "corners",
        len(np.unique(owners)),
        "surfaces",
        len(samples),
        "observations",
        flush=True,
    )


def fit_patch(x, y):
    if len(x) < 13:
        return None
    size = np.maximum(np.ptp(x, axis=0), 1.0)
    units = (x - x.min(0)) / size
    cell = np.clip(np.int32(units * [5, 4]), 0, [4, 3])
    check = (cell[:, 0] + 2 * cell[:, 1]) % 4 == 0
    if check.sum() < 3 or (~check).sum() < 8:
        return None
    h, inl = cv2.findHomography(x[~check], y[~check], cv2.RANSAC, 1.9, maxIters=500)
    if h is None or inl.sum() < 7:
        return None
    pred = cv2.perspectiveTransform(x[None].astype(np.float32), h)[0]
    errors = np.linalg.norm(pred - y, axis=1)
    keep = errors < 2.8
    if (
        np.median(errors[check]) > 2.0
        or np.count_nonzero(keep & check) < 3
        or keep.mean() < 0.72
    ):
        return None
    hull = cv2.convexHull(x[keep].astype(np.float32)).reshape(-1, 2)
    if cv2.contourArea(hull) < 48:
        return None
    return h, hull, round(float(np.median(errors[check])), 3), check, keep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference", "geometry", "video", "sfm"):
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    cv2.setNumThreads(4)
    plan = json.loads(a.plan.read_text("utf-8-sig"))
    key = plan["views"][0]["source"]
    _, cameras, _, _, _ = prepare(plan, {key: a.reference})
    collect(
        a.video,
        a.sfm / digest(a.video)[:12],
        cameras[key],
        scene_faces(load_mesh(a.geometry)),
        a.geometry,
    )


if __name__ == "__main__":
    main()
