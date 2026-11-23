"""Anchor the scale of each moving camera to annotated ground, with held-out road features."""

import cv2
import numpy as np

from video_tracks import follow


def ground_scale(path, frames, start, polygon):
    if len(frames) < 2:
        return None
    cap = cv2.VideoCapture(str(path))
    saved = {}
    last = max(frames)
    first = previous = None
    try:
        for i in range(last + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            if i % 2 and i not in frames:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if i == 0:
                mask = np.zeros(gray.shape, np.uint8)
                for q in polygon:
                    cv2.fillPoly(mask, [np.int32(np.rint(q))], 255)
                mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
                points = cv2.goodFeaturesToTrack(
                    gray, 1700, 0.001, 3, mask=mask, blockSize=3
                )
                if points is None:
                    return None
                first = points.reshape(-1, 2)
                points = first.copy()
                ids = np.arange(len(first))
            else:
                points, keep = follow(previous, gray, points)
                points, ids = points[keep], ids[keep]
            previous = gray
            if i in frames and len(points) > 10:
                saved[i] = (ids.copy(), points.copy())
    finally:
        cap.release()
    if first is None or len(saved) < 2:
        return None
    ray = (
        np.column_stack(
            ((first - start.k[:2, 2]) / np.diag(start.k)[:2], np.ones(len(first)))
        )
        @ start.r
    )
    world = ray * (-start.eye[1] / np.minimum(ray[:, 1:2], -0.001))
    check = ((first[:, 0] // 64 + 3 * (first[:, 1] // 48)) % 5) == 0
    observations = []
    scales = []
    local = {}
    for i, (ids, pts) in saved.items():
        camera = frames[i]
        if i == 0:
            continue
        down = ray[ids, 1] < -0.001
        ids, pts = ids[down], pts[down]
        if len(ids) < 10:
            continue
        fit = ~check[ids]
        test = check[ids]
        metric = world[ids] + start.eye
        if fit.sum() >= 30 and test.sum() >= 9:
            ok, rv, tv, ins = cv2.solvePnPRansac(
                metric[fit],
                pts[fit],
                camera.k,
                None,
                iterationsCount=450,
                reprojectionError=2.7,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok and ins is not None and len(ins) >= 25:
                rv, tv = cv2.solvePnPRefineLM(
                    metric[fit][ins.ravel()],
                    pts[fit][ins.ravel()],
                    camera.k,
                    None,
                    rv,
                    tv,
                )
                r = cv2.Rodrigues(rv)[0]
                eye = -r.T @ tv.ravel()
                image = cv2.projectPoints(metric[test], rv, tv, camera.k, None)[
                    0
                ].reshape(-1, 2)
                error = np.linalg.norm(image - pts[test], axis=1)
                angle = np.degrees(
                    np.arccos(np.clip((np.trace(r @ camera.r.T) - 1) / 2, -1, 1))
                )
                span = cv2.contourArea(
                    cv2.convexHull(pts[fit][ins.ravel()].astype(np.float32))
                ) / np.prod(start.size)
                if (
                    np.median(error) < 2.5
                    and (error < 4).mean() > 0.70
                    and span > 0.012
                    and eye[1] > 2
                    and angle < 5.5
                ):
                    local[i] = dict(
                        r=r.tolist(),
                        t=tv.ravel().tolist(),
                        checks=int(test.sum()),
                        medianPx=round(float(np.median(error)), 3),
                    )
        delta = (start.eye - camera.eye) @ camera.r.T
        direction = world[ids] @ camera.r.T
        norm = (pts - camera.k[:2, 2]) / np.diag(camera.k)[:2]
        a = direction[:, :2] - norm * direction[:, 2:]
        b = delta[:2] - norm * delta[2]
        scale = -(a * b).sum(1) / np.maximum((b * b).sum(1), 1.0e-8)
        use = (
            (np.linalg.norm(b, axis=1) / np.linalg.norm(world[ids], axis=1) > 0.025)
            & (scale > 0.55)
            & (scale < 1.7)
        )
        train = use & ~check[ids]
        if train.sum() < 18 or (use & check[ids]).sum() < 5:
            continue
        median = np.median(scale[train])
        if np.median(np.abs(scale[train] - median)) > 0.11:
            continue
        scales.append(median)
        m = use & check[ids]
        observations.append((delta, direction[m], pts[m], camera.k))
    if len(scales) < 3:
        return dict(
            scale=1.0,
            frames=0,
            checks=0,
            beforePx=0.0,
            afterPx=0.0,
            spread=0.0,
            local=local,
        )
    factor = float(np.median(scales))

    def error(s):
        residual = []
        for delta, dir, uv, k in observations:
            p = dir + s * delta
            projected = p[:, :2] / p[:, 2:] * np.diag(k)[:2] + k[:2, 2]
            residual.extend(np.linalg.norm(projected - uv, axis=1))
        return np.asarray(residual)

    before, after = error(1), error(factor)
    return dict(
        scale=factor,
        frames=len(scales),
        checks=len(after),
        beforePx=round(float(np.median(before)), 3),
        afterPx=round(float(np.median(after)), 3),
        spread=round(float(np.median(np.abs(np.asarray(scales) - factor))), 4),
        local=local,
    )
