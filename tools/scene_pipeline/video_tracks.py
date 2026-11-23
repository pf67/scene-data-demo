"""Match the video crop to the reference and track image features in both temporal directions."""

import hashlib
from pathlib import Path

import cv2
import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(stream.read()).hexdigest()
        )


def photo_features(gray):
    detector = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.017)
    keys, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        raise ValueError("照片中缺少稳定特征")
    descriptors /= np.maximum(descriptors.sum(1, keepdims=True), 1e-9)
    return np.float32([key.pt for key in keys]), np.sqrt(descriptors)


def camera_match(first, second):
    a, da = first
    b, db = second
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))

    def near(x, y):
        return {
            m.queryIdx: m.trainIdx
            for m, n in matcher.knnMatch(x, y, k=2)
            if m.distance < 0.75 * n.distance
        }

    fwd, rev = near(da, db), near(db, da)
    pairs = [(i, j) for i, j in fwd.items() if rev.get(j) == i]
    if not pairs:
        return np.empty((0, 2)), np.empty((0, 2))
    i, j = np.array(pairs).T
    return a[i], b[j]


def initial_intrinsics(reference, first, cam):
    a, b = camera_match(reference, photo_features(first))
    if len(a) < 30:
        raise ValueError("视频首帧与参考照片的对应不足")
    _, mask = cv2.estimateAffine2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if mask is None or mask.sum() < 30:
        raise ValueError("首帧未能配准到参考照片")
    a, b = a[mask.ravel() != 0], b[mask.ravel() != 0]
    # Video encoders crop or stretch the source photo. Measure both axes.
    sx, tx = np.polyfit(a[:, 0], b[:, 0], 1)
    sy, ty = np.polyfit(a[:, 1], b[:, 1], 1)
    scale, shift = np.array([sx, sy]), np.array([tx, ty])
    err = np.linalg.norm(a * scale + shift - b, axis=1)
    if (
        min(scale) <= 0
        or np.median(err) > 2
        or np.prod(np.ptp(b, axis=0)) / first.size < 0.18
    ):
        raise ValueError("该视频首帧需要单独相机标定")
    k = cam.k.copy()
    k[:2] *= scale[:, None]
    k[:2, 2] += shift
    return k, scale, shift, round(float(np.median(err)), 3)


def world_anchors(camera, depth, uv):
    xy = np.clip(np.rint(uv).astype(int), 0, camera.size.astype(int) - 1)
    x, y = xy.T
    z = depth[y, x]
    ray = (
        np.column_stack(
            ((uv - camera.k[:2, 2]) / np.diag(camera.k)[:2], np.ones(len(uv)))
        )
        @ camera.r
    )
    good = (uv >= 3).all(1) & (uv < camera.size - 3).all(1) & (z > 1) & (z < 350)
    for dx, dy in ((-2, 0), (2, 0), (0, 2), (0, -2)):
        nbr = depth[
            np.clip(y + dy, 0, depth.shape[0] - 1),
            np.clip(x + dx, 0, depth.shape[1] - 1),
        ]
        good &= np.abs(nbr - z) < np.maximum(0.45, 0.012 * z)
    return camera.eye + ray * z[:, None], good


def follow(before, after, xy):
    if not len(xy):
        return np.empty((0, 2), np.float32), np.zeros(0, bool)
    args = dict(
        winSize=(27, 27),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 25, 0.02),
    )
    q, keep1, err = cv2.calcOpticalFlowPyrLK(
        before, after, xy.astype(np.float32), None, **args
    )
    p, keep2, _ = cv2.calcOpticalFlowPyrLK(after, before, q, None, **args)
    good = (
        (keep1.ravel() != 0)
        & (keep2.ravel() != 0)
        & (err.ravel() < 31)
        & (np.linalg.norm(p - xy, axis=1) < 1.25)
    )
    good &= (q >= 5).all(axis=1) & (q < np.array(after.shape[::-1]) - 5).all(axis=1)
    return q, good
