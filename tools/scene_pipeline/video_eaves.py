"""Track surveyed eaves through moving shots and verify them against independently marked frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from video_bake import transform
from video_materials import fit_patch
from video_tracks import digest, follow
from worker import prepare


def tracks(video, data, base, survey):
    k = np.array(data["k"])
    size = np.array(data["size"])
    cap = cv2.VideoCapture(str(video))
    ok, bgr = cap.read()
    if not ok:
        raise ValueError("初始帧读取失败")
    first = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    scale = np.diag(k)[:2] / np.diag(base.k)[:2]
    shift = k[:2, 2] - scale * base.k[:2, 2]
    pts = []
    groups = []
    polys = {}
    for i, (name, shape) in enumerate(survey.items()):
        q = np.float32(shape) * scale + shift
        polys[name] = q
        mask = np.zeros(first.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.int32(np.rint(0.91 * q + 0.09 * q.mean(0))), 255)
        corners = cv2.goodFeaturesToTrack(
            first, 350, 0.0015, 2.8, mask=mask, blockSize=3
        )
        if corners is not None:
            pts.extend(corners.reshape(-1, 2))
            groups.extend([i] * len(corners))
    start = np.float32(pts)
    ids = np.arange(len(pts))
    pts = start.copy()
    groups = np.array(groups)
    frames = {f["frame"] for f in data["frames"]}
    frames.update((a + b) // 2 for a, b in zip(sorted(frames), sorted(frames)[1:]))
    prev = first
    result = {name: {} for name in survey}
    try:
        for i in range(max(frames) + 1):
            if i:
                ok, img = cap.read()
                if not ok:
                    break
                if i % 2 and i not in frames:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                pts, keep = follow(prev, gray, pts)
                pts, ids = pts[keep], ids[keep]
                prev = gray
            if i not in frames:
                continue
            for j, name in enumerate(survey):
                use = groups[ids] == j
                x = start[ids[use]]
                y = pts[use]
                match = fit_patch(x, y)
                if match is None:
                    continue
                h, hull, med, check, valid = match
                q = transform(h, polys[name])
                ratio = cv2.contourArea(hull) / max(
                    1.0, cv2.contourArea(np.float32(polys[name]))
                )
                if (
                    ratio < 0.30
                    or not cv2.isContourConvex(q)
                    or (q < 2).any()
                    or (q > size - 3).any()
                ):
                    continue
                result[name][i] = dict(
                    h=h.tolist(),
                    quad=q.tolist(),
                    heldMedianPx=med,
                    heldPoints=int(check.sum()),
                    spread=round(ratio, 3),
                )
    finally:
        cap.release()
    return dict(firstScale=scale.tolist(), firstShift=shift.tolist(), surfaces=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference", "sfm", "survey", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--videos", required=True, nargs="+", type=Path)
    a = p.parse_args()
    cv2.setNumThreads(4)
    plan = json.loads(a.plan.read_text("utf-8-sig"))
    key = plan["views"][0]["source"]
    _, cams, _, _, _ = prepare(plan, {key: a.reference})
    survey = json.loads(a.survey.read_text("utf8"))
    a.out.mkdir(parents=True, exist_ok=True)
    for sha, path in {digest(v)[:12]: v for v in a.videos}.items():
        rec = json.loads((a.sfm / sha / "reconstruction.json").read_text())
        surfaces = dict(survey["roofs"])
        surfaces.update(
            {n + "-wall": w["track"] for n, w in survey.get("walls", {}).items()}
        )
        out = tracks(path, rec, cams[key], surfaces)
        (a.out / (sha + ".json")).write_text(json.dumps(out, indent=2), encoding="utf8")
        print(sha, [(k, len(v)) for k, v in out["surfaces"].items()], flush=True)


if __name__ == "__main__":
    main()
