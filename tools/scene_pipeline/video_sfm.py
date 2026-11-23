"""Reconstruct one moving shot and anchor its measured cameras and points to the reference photo."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pycolmap

from video_tracks import digest, initial_intrinsics, photo_features, world_anchors
from register_step import mesh_depth
from geometry import Camera
from orbit import signed_yaw
from worker import prepare


def extract(path, folder, stride):
    folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    names = []
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for i in range(count):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"视频第 {i} 帧解码失败")
            if i % stride == 0 or i == count - 1:
                name = f"{i:05d}.jpg"
                if not (folder / name).exists():
                    cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 97])[
                        1
                    ].tofile(folder / name)
                names.append(name)
    finally:
        cap.release()
    if len(names) < 3:
        raise ValueError("相邻机位至少需要三帧")
    return names, fps


def populate(folder, names, dbpath, pairsfile, k):
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "PINHOLE"
    opts.camera_params = ",".join(
        map(str, [k[0, 0], k[1, 1], k[0, 2] + 0.5, k[1, 2] + 0.5])
    )
    with pycolmap.Database.open(dbpath):
        pass
    pycolmap.import_images(dbpath, folder, pycolmap.CameraMode.SINGLE, names, opts)
    with pycolmap.Database.open(dbpath) as db:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        features, indexes = [], []
        for name in names:
            im = cv2.imdecode(
                np.fromfile(folder / name, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            xy, desc = photo_features(im)
            if len(desc) > 5200:
                xy, desc = xy[:5200], desc[:5200]
            db.write_keypoints(ids[name], xy + 0.5)
            features.append(desc)
            tree = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
            tree.add([desc])
            tree.train()
            indexes.append(tree)
        pairs = []
        for i in range(len(names)):
            candidates = set(range(i + 1, min(i + 6, len(names))))
            candidates.update(
                j for j in (i + 7, i + 10, i + 15, len(names) - 1) if i < j < len(names)
            )
            for j in sorted(candidates):

                def matches(a, b):
                    return {
                        m.queryIdx: m.trainIdx
                        for m, n in indexes[b].knnMatch(features[a], k=2)
                        if m.distance < 0.79 * n.distance
                    }

                forward = matches(i, j)
                if len(forward) < 24:
                    continue
                reverse = matches(j, i)
                matched = np.uint32(
                    [(a, b) for a, b in forward.items() if reverse.get(b) == a]
                )
                if len(matched) < 24:
                    continue
                db.write_matches(ids[names[i]], ids[names[j]], matched)
                pairs.append(f"{names[i]} {names[j]}")
    pairsfile.write_text("\n".join(pairs) + "\n", encoding="ascii")
    opts = pycolmap.TwoViewGeometryOptions()
    opts.ransac.max_error = 2.5
    pycolmap.verify_matches(dbpath, pairsfile, opts)


def reconstruction(out, names):
    folder, db, sparse = out / "frames", out / "features.db", out / "sparse"
    sparse.mkdir(exist_ok=True)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = 4
    opts.mapper.num_threads = 4
    opts.random_seed = 19
    opts.ba_use_gpu = False
    opts.ba_refine_focal_length = False
    opts.ba_refine_extra_params = False
    opts.mapper.abs_pose_refine_focal_length = False
    opts.mapper.abs_pose_refine_extra_params = False
    opts.mapper.init_min_tri_angle = 4.0
    opts.mapper.init_max_forward_motion = 0.995
    opts.mapper.filter_max_reproj_error = 2.5
    opts.min_model_size = 4
    opts.ba_global_max_num_iterations = 36
    reconstructions = [
        pycolmap.Reconstruction(s.parent) for s in sparse.glob("*/images.bin")
    ]
    if not reconstructions:
        reconstructions = list(
            pycolmap.incremental_mapping(db, folder, sparse, opts).values()
        )
    for model in sorted(reconstructions, key=lambda m: -m.num_reg_images()):
        if any(model.images[i].name == names[0] for i in model.reg_image_ids()):
            return model
    raise ValueError("重建分量没有连接到源照片")


def anchor(
    model, out, base, pivot, depth, k, scale, shift, video, names, fps, first_error
):
    images = [model.images[i] for i in model.reg_image_ids()]
    start = next(im for im in images if im.name == names[0])
    pose = start.cam_from_world()
    rotation = base.r.T @ pose.rotation.matrix()
    origin = start.projection_center()
    pts = [
        (m.xy - 0.5, m.point3D_id)
        for m in start.points2D
        if m.has_point3D() and model.points3D[m.point3D_id].track.length() >= 4
    ]
    if len(pts) < 30:
        raise ValueError("首帧上稳定的三维锚点不足")
    uv = np.array([p for p, _ in pts])
    ids = [i for _, i in pts]
    xyz = np.array([model.points3D[i].xyz for i in ids])
    original = (uv - shift) / scale
    prior, valid = world_anchors(base, depth, original)
    prior_z = (prior - base.eye) @ base.r[2]
    solved_z = (xyz - origin) @ pose.rotation.matrix()[2]
    valid &= solved_z > 0
    check = ((original[:, 0] // 70 + 3 * (original[:, 1] // 70)) % 5) == 0
    ratio = prior_z[valid & ~check] / solved_z[valid & ~check]
    if len(ratio) < 24 or (valid & check).sum() < 8:
        raise ValueError("重建的尺度缺少分布在不同区域的参考点")
    factor = np.median(ratio)
    errors = np.abs(solved_z[valid & check] * factor / prior_z[valid & check] - 1)
    size = np.array(
        [model.cameras[start.camera_id].width, model.cameras[start.camera_id].height]
    )
    frames = []
    for image in sorted(images, key=lambda im: im.name):
        pose = image.cam_from_world()
        r = pose.rotation.matrix() @ rotation.T
        eye = base.eye + (image.projection_center() - origin) @ rotation.T * factor
        cam = Camera(image.name, size, k, r, -r @ eye)
        visible = [p.point3D_id for p in image.points2D if p.has_point3D()]
        err = np.array([model.points3D[i].error for i in visible])
        frames.append(
            dict(
                frame=int(Path(image.name).stem),
                eye=cam.eye.tolist(),
                r=cam.r.tolist(),
                t=cam.t.tolist(),
                yaw=round(signed_yaw(base.eye - pivot, cam.eye - pivot), 2),
                points=len(visible),
                medianTrackPx=round(float(np.median(err)), 3),
            )
        )
    data = dict(
        videoSha256=digest(video),
        fps=fps,
        framesSampled=len(names),
        size=size.tolist(),
        k=k.tolist(),
        scale=float(factor),
        rotation=rotation.tolist(),
        origin=origin.tolist(),
        referenceEye=base.eye.tolist(),
        firstMedianPx=first_error,
        scaleChecks=int(len(errors)),
        scaleMedianDepthErrorPercent=round(100 * float(np.median(errors)), 2),
        points3D=model.num_points3D(),
        frames=frames,
    )
    (out / "reconstruction.json").write_text(
        json.dumps(data, indent=2), encoding="utf8"
    )
    print(
        "reconstructed",
        out.name,
        len(frames),
        "/",
        len(names),
        "views",
        data["points3D"],
        "points",
        data["scaleMedianDepthErrorPercent"],
        "percent scale check",
        [(f["frame"], f["yaw"]) for f in frames[::4]],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "reference", "geometry", "video", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stride", type=int, default=9)
    a = p.parse_args()
    if a.stride < 1:
        p.error("stride must be positive")
    cv2.setNumThreads(4)
    cv2.setRNGSeed(19)
    out = a.out / digest(a.video)[:12]
    out.mkdir(exist_ok=True, parents=True)
    tag = out / "inputs.json"
    source_inputs = dict(
        video=digest(a.video), stride=a.stride, reference=digest(a.reference)
    )
    if tag.exists() and any(
        json.loads(tag.read_text())[k] != v for k, v in source_inputs.items()
    ):
        raise ValueError("输入或抽帧间隔已变，请使用新的输出目录")
    plan = json.loads(a.plan.read_text("utf-8-sig"))
    key = plan["views"][0]["source"]
    _, cams, photos, roofs, _ = prepare(plan, {key: a.reference})
    base = cams[key]
    names, fps = extract(a.video, out / "frames", a.stride)
    first = cv2.imdecode(
        np.fromfile(out / "frames" / names[0], np.uint8), cv2.IMREAD_GRAYSCALE
    )
    ref = photo_features(cv2.cvtColor(np.asarray(photos[key]), cv2.COLOR_RGB2GRAY))
    k, scale, shift, error = initial_intrinsics(ref, first, base)
    inputs = dict(**source_inputs, k=k.tolist())
    if tag.exists() and json.loads(tag.read_text()) != inputs:
        raise ValueError("输入或抽帧间隔已变，请使用新的输出目录")
    tag.write_text(json.dumps(inputs), encoding="utf8")
    if not (out / "pairs.txt").is_file() or not (out / "features.db").is_file():
        with TemporaryDirectory(dir=out) as tmp:
            db, pairs = Path(tmp) / "features.db", Path(tmp) / "pairs.txt"
            populate(out / "frames", names, db, pairs, k)
            db.replace(out / "features.db")
            pairs.replace(out / "pairs.txt")
    model = reconstruction(out, names)
    anchor(
        model,
        out,
        base,
        np.median(np.concatenate(roofs), axis=0),
        mesh_depth(a.geometry, base),
        k,
        scale,
        shift,
        a.video,
        names,
        fps,
        error,
    )


if __name__ == "__main__":
    main()
