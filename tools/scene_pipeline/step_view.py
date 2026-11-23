"""Generate one gpt-image step from an accepted neighbour in the original scene coordinates."""

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "builder"))
from geometry import Camera
from orbit import make_layout, scaled, surface_map
from surface_depth import cached_depth, fit_surface_depth, rays_for
from worker import prepare, read_photo, scene_faces


def report_camera(source, pose):
    eye = np.array(pose["position"])
    forward = np.array(pose["target"]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, pose["up"])
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    size = np.array(pose["size"], float)
    cyx = size * np.array(pose["principal"])
    fy = size[1] / (2 * math.tan(math.radians(pose["fov"]) / 2))
    k = np.array([[fy * pose["focalRatio"], 0, cyx[0]], [0, fy, cyx[1]], [0, 0, 1]])
    return Camera(source, size, k, rotation, -rotation @ eye)


def warp_parent(layout, camera, image, faces, raw_depth, roads, correspondence=None):
    """Fill a target view from the neighbour; carry the nearest surface through disocclusions."""
    w, h = layout.image.size
    base = np.array(layout.image)
    _, front, ids = surface_map(camera, faces)
    xy, z = camera.pixels(layout.world.reshape(-1, 3))
    xy = np.nan_to_num(xy, nan=-1).reshape(h, w, 2).astype(np.float32)
    ij = np.clip(np.rint(xy), 0, camera.size - 1).astype(int)
    inside = (xy >= 0).all(2) & (xy < camera.size - 1).all(2) & (z.reshape(h, w) > 0.2)
    supported = (
        inside & (layout.ids == ids[ij[:, :, 1], ij[:, :, 0]]) & (layout.ids != -2)
    )
    supported &= np.abs(z.reshape(h, w) - front[ij[:, :, 1], ij[:, :, 0]]) < np.maximum(
        0.4, 0.009 * z.reshape(h, w)
    )
    base[supported] = cv2.remap(image, xy, None, cv2.INTER_LINEAR)[supported]
    if correspondence is not None:
        correspondence["parent"] = np.where(supported[:, :, None], xy, -1.0).astype(
            np.float32
        )

    field = fit_surface_depth(camera, raw_depth, faces, roads)
    if field is None:
        return Image.fromarray(base)
    xyz = (rays_for(field.camera) * field.depth[:, :, None] + field.camera.eye).reshape(
        -1, 3
    )
    dst, tz = layout.camera.pixels(xyz)
    keep = (
        np.isfinite(dst).all(1)
        & (tz > 0.2)
        & (dst >= 0).all(1)
        & (dst < [w - 1, h - 1]).all(1)
    )
    indices = np.flatnonzero(keep)
    order = indices[np.argsort(tz[indices])[::-1]]
    ix, iy = np.rint(dst[order]).astype(int).T
    rgb = cv2.resize(
        image, field.depth.shape[::-1], interpolation=cv2.INTER_AREA
    ).reshape(-1, 3)
    # Sparse depth can expose holes at a roof silhouette. Interpolate within
    # a few pixels only; the structural guide remains behind larger gaps.
    samples = np.zeros((h, w, 3), np.uint8)
    samples[iy, ix] = rgb[order]
    known = np.zeros((h, w), np.uint8)
    known[iy, ix] = 1
    distance, labels = cv2.distanceTransformWithLabels(
        1 - known, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
    lut[labels[known != 0]] = samples[known != 0]
    base[distance <= 4] = lut[labels[distance <= 4]]
    if correspondence is not None:
        original_uv = np.empty((h, w, 2), np.float32)
        original_uv[iy, ix] = np.column_stack(
            np.unravel_index(order, field.depth.shape)
        )[:, ::-1] * (camera.size / field.camera.size)
        table = np.zeros((int(labels.max()) + 1, 2), np.float32)
        table[labels[known != 0]] = original_uv[known != 0]
        use = distance <= 4
        correspondence["parent"][use] = table[labels[use]]
    return Image.fromarray(base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--source", action="append", nargs=2, required=True, metavar=("ID", "PHOTO")
    )
    p.add_argument("--parent", required=True)
    p.add_argument("--parent-image", type=Path, required=True)
    p.add_argument(
        "--target",
        type=float,
        required=True,
        help="signed azimuth relative to the first real photo",
    )
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--model", default="gpt-image-2")
    p.add_argument(
        "--detail-swatches",
        action="store_true",
        help="Use the parent's detail crops as the second reference when a candidate copied its camera pose",
    )
    p.add_argument("--guide-only", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text("utf-8-sig"))
    report = json.loads(args.report.read_text("utf-8-sig"))
    parent = next(
        (
            v
            for v in report["supplements"]
            if v["source"] == args.parent and v["status"] == "aligned"
        ),
        None,
    )
    if parent is None:
        raise ValueError("邻接图需要有已通过的配准报告")
    first = plan["views"][0]["source"]
    if parent["reference"] != first:
        raise ValueError("目标方位和父机位必须使用同一原始参考坐标")
    step = (args.target - parent["actualAngle"] + 180) % 360 - 180
    if abs(step) < 0.1:
        raise ValueError("目标机位与父机位重合")
    target = args.directory
    target.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.parent_image.read_bytes()).hexdigest()
    meta = dict(
        parent=args.parent,
        parentSha256=digest,
        parentYaw=parent["actualAngle"],
        targetYaw=args.target,
        requestedStep=round(step, 3),
        model=args.model,
        quality="high",
    )
    if args.detail_swatches:
        meta["referenceMode"] = "detail-swatches"
    if (target / "view.jpg").is_file():
        previous = json.loads((target / "generation.json").read_text("utf8"))
        if (
            all(previous.get(k) == v for k, v in meta.items())
            and previous.get("referenceMode") == meta.get("referenceMode")
            and previous.get("sha256")
            == hashlib.sha256((target / "view.jpg").read_bytes()).hexdigest()
        ):
            print("Using saved generated image:", target / "view.jpg")
            return
        raise ValueError("输出目录已有另一组机位的候选图片")
    _, cameras, images, roofs, _ = prepare(plan, dict(args.source))
    layout = make_layout(plan, cameras[first], images[first], roofs, args.target)
    cam = report_camera(args.parent, parent["camera"])
    photo = read_photo(args.parent_image)
    cam = scaled(cam, photo.size)
    guide = warp_parent(
        layout,
        cam,
        np.array(photo),
        scene_faces(plan, roofs),
        cached_depth(args.parent_image),
        plan.get("ground", {}).get("roads", []),
    )
    guide.save(target / "guide.jpg", quality=96, subsampling=0)
    layout.image.save(target / "registration-guide.jpg", quality=95)
    if args.guide_only:
        return
    instruction = (
        "The target image already incorporates the requested small camera step. Match its composition "
        "exactly. The other input is the immediately preceding photograph. Carry the same roof ridges, "
        "cornerstones, roof colour, boundary walls, gate openings and individual planted or paved "
        "courtyards forwards. Maintain the positions of the foreground walls and ground-level corners. "
        "Restore photographic surfaces and only the thin newly exposed strips. No shifted buildings, "
        "additional houses, altered plots, zoom, annotations or second camera move."
    )
    reference = photo
    if args.detail_swatches:
        # The first image still derives from the entire parent photograph with
        # its calibrated depth. Local swatches cannot reintroduce the old composition.
        cols, rows = 3, 2
        w, h = photo.size
        reference = Image.new("RGB", photo.size)
        for i, (x, y) in enumerate(
            (
                (0.48, 0.39),
                (0.32, 0.64),
                (0.80, 0.44),
                (0.63, 0.68),
                (0.17, 0.32),
                (0.44, 0.84),
            )
        ):
            cx, cy = round(x * w), round(y * h)
            crop = photo.crop((cx - w // 9, cy - h // 9, cx + w // 9, cy + h // 9))
            box = (
                i % cols * w // cols,
                i // cols * h // rows,
                (i % cols + 1) * w // cols,
                (i // cols + 1) * h // rows,
            )
            reference.paste(
                crop.resize(
                    (box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS
                ),
                box,
            )
        instruction = (
            "Restore the first image at precisely its existing viewpoint, including its shifted foreground. "
            "The second image is a disconnected contact sheet of material samples from the previous camera. "
            "Use it only for fine detail. Every complete roof, courtyard and boundary of the output must "
            "remain at exactly its position, orientation and scale in the first image. Deliver one aerial photograph."
        )
    from refinement.image_api import generate_view
    blob, info = generate_view(
        reference, guide, step, args.model, "layout", "high", instruction
    )
    out = read_photo(io.BytesIO(blob))
    out.save(target / "view.tmp", format="JPEG", quality=96, subsampling=0)
    meta.update(
        size=list(out.size),
        guidance=info["guidance"],
        inputRoles=info["inputRoles"],
        sha256=hashlib.sha256((target / "view.tmp").read_bytes()).hexdigest(),
    )
    (target / "generation.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf8"
    )
    (target / "view.tmp").replace(target / "view.jpg")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
