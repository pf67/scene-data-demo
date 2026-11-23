"""Exclude contradictory pixels where the two ends of a calibrated view chain meet."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from register_step import mesh_depth, read, save
from step_view import report_camera
from orbit import scaled
from surface_depth import rays_for
from verify_surface_pair import correlation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directories", type=Path, nargs=2, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = args.directories
    masks = [np.asarray(Image.open(d / "paired-mask.png")) > 0 for d in paths]
    sizes = [m.shape[::-1] for m in masks]
    quality = [read(d / "track-quality.json") for d in paths]
    if not all(q["status"] == "aligned" for q in quality):
        raise ValueError("两端图像都需要完成校准")
    cams = [
        scaled(report_camera("", q["camera"]), size) for q, size in zip(quality, sizes)
    ]
    photos = [
        cv2.cvtColor(
            np.asarray(
                Image.open(d / "view.jpg")
                .convert("RGB")
                .resize(size, Image.Resampling.LANCZOS)
            ),
            cv2.COLOR_RGB2GRAY,
        )
        for d, size in zip(paths, sizes)
    ]
    depth = [mesh_depth(args.geometry, c) for c in cams]
    result = {}
    for i, j in ((0, 1), (1, 0)):
        xyz = cams[i].eye + rays_for(cams[i]) * depth[i][:, :, None]
        coords, z = cams[j].pixels(xyz.reshape(-1, 3))
        uv = coords.reshape(*masks[i].shape, 2).astype(np.float32)
        z = z.reshape(masks[i].shape)
        warp = lambda a, method: cv2.remap(a, uv, None, method)
        both = masks[i] & (warp(np.uint8(masks[j]), cv2.INTER_NEAREST) != 0) & (z > 0.2)
        both &= np.abs(warp(depth[j], cv2.INTER_LINEAR) - z) < np.maximum(
            0.5, 0.012 * z
        )
        ncc, var = correlation(photos[i], warp(photos[j], cv2.INTER_LINEAR))
        conflict = both & ((ncc < 0.55) | (var < 15))
        keep = masks[i] & ~conflict
        Image.fromarray(np.uint8(keep) * 255).save(paths[i] / "joined-mask.png")
        result[paths[i].name] = dict(
            overlap=int(both.sum()),
            removed=int(conflict.sum()),
            retained=int((both & ~conflict).sum()),
            finalPercent=round(float(100 * keep.mean()), 3),
        )
    save(args.output, result)


if __name__ == "__main__":
    main()
