"""Build a unified world-aligned COLMAP workspace from reference photo + aerial videos and train 3DGS via Brush."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/scene_pipeline"), str(ROOT / "builder")]

from register_step import mesh_depth  # noqa: E402
from video_tracks import digest, initial_intrinsics, photo_features, world_anchors  # noqa: E402
from worker import prepare, read_photo  # noqa: E402


DEFAULT_BRUSH_BIN = ROOT / "tools/bin/brush/brush_app.exe"


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def write_watermark_mask(
    width: int,
    height: int,
    dest: Path,
    box: tuple[float, float, float, float] = (0.86, 0.88, 1.0, 1.0),
) -> Path:
    """Create a Brush/COLMAP mask image (255=keep, 0=ignore) masking out the corner watermark."""
    mask = np.full((height, width), 255, dtype=np.uint8)
    x0 = int(np.clip(round(box[0] * width), 0, width - 1))
    y0 = int(np.clip(round(box[1] * height), 0, height - 1))
    x1 = int(np.clip(round(box[2] * width), x0 + 1, width))
    y1 = int(np.clip(round(box[3] * height), y0 + 1, height))
    mask[y0:y1, x0:x1] = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(dest)
    return dest


def extract_sharp_frames(
    video_path: Path,
    dest_dir: Path,
    prefix: str,
    stride: int,
    max_frames: int | None = None,
    min_sharpness_ratio: float = 0.28,
) -> list[str]:
    """Extract frames at stride, keeping frame 0 and filtering out severe motion-blur frames."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates: list[tuple[int, np.ndarray, float]] = []
    try:
        for i in range(total):
            ok, bgr = cap.read()
            if not ok:
                break
            if i == 0 or i % stride == 0 or i == total - 1:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                candidates.append((i, bgr, laplacian_sharpness(gray)))
    finally:
        cap.release()
    if len(candidates) < 3:
        raise ValueError(f"视频 {video_path.name} 可用抽样帧不足 3 帧")
    median_sharp = float(np.median([s for _, _, s in candidates]))
    threshold = max(12.0, min_sharpness_ratio * median_sharp)
    kept = [
        item for item in candidates if item[0] == 0 or item[2] >= threshold
    ]
    if max_frames and len(kept) > max_frames:
        indices = np.linspace(0, len(kept) - 1, max_frames, dtype=int)
        kept = [kept[idx] for idx in indices]
    rel_names: list[str] = []
    for frame_idx, bgr, _ in kept:
        rel = f"{prefix}/{frame_idx:05d}.jpg"
        out_file = dest_dir.parent / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])[1].tofile(out_file)
        rel_names.append(rel)
    return rel_names


def deduplicate_videos(video_paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for vp in video_paths:
        h = digest(vp)
        if h not in seen:
            seen.add(h)
            unique.append(vp)
    return unique


def rotation_matrix_to_colmap_quat(r_cw: np.ndarray) -> np.ndarray:
    """Convert 3x3 world-to-camera rotation matrix to COLMAP quaternion (qw, qx, qy, qz)."""
    q_xyzw = Rotation.from_matrix(r_cw).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)


def compute_sim3_alignment(
    model,
    ref_image_name: str,
    base_cam,
    ref_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve Sim(3) from COLMAP coordinates into base_cam's Y-up world coordinates."""
    images = [model.images[i] for i in model.reg_image_ids()]
    start = next((im for im in images if im.name == ref_image_name), None)
    if start is None:
        raise ValueError(f"参考图 {ref_image_name} 未进入重建分量")
    pose = start.cam_from_world()
    rotation = base_cam.r.T @ pose.rotation.matrix()
    origin = start.projection_center()

    for min_track in (4, 3, 2):
        pts = [
            (m.xy - 0.5, m.point3D_id)
            for m in start.points2D
            if m.has_point3D() and model.points3D[m.point3D_id].track.length() >= min_track
        ]
        if len(pts) >= 24:
            break
    if len(pts) < 16:
        raise ValueError(f"参考图上稳定的三维锚点不足: {len(pts)}")
    uv = np.array([p for p, _ in pts], dtype=np.float64)
    ids = [i for _, i in pts]
    xyz = np.array([model.points3D[i].xyz for i in ids], dtype=np.float64)
    prior, valid = world_anchors(base_cam, ref_depth, uv)
    prior_z = (prior - base_cam.eye) @ base_cam.r[2]
    solved_z = (xyz - origin) @ pose.rotation.matrix()[2]
    valid &= (solved_z > 0.05) & (prior_z > 0.05)
    if valid.sum() < 12:
        raise ValueError("对齐尺度有效参考点不足")
    ratios = prior_z[valid] / solved_z[valid]
    factor = float(np.median(ratios))
    errors = np.abs(solved_z[valid] * factor / prior_z[valid] - 1.0)
    scale_err_pct = round(100.0 * float(np.median(errors)), 2)
    return rotation, origin, factor, scale_err_pct


def write_aligned_colmap_sparse(
    model,
    sparse_dir: Path,
    ref_image_name: str,
    base_cam,
    rotation: np.ndarray,
    origin: np.ndarray,
    factor: float,
    scene_center: np.ndarray,
    max_reproj_px: float = 2.0,
    min_image_points: int = 50,
    max_radius_m: float = 650.0,
) -> dict:
    """Transform COLMAP reconstruction into Y-up world coords and export text sparse model."""
    sparse_dir.mkdir(parents=True, exist_ok=True)
    used_cameras = set()
    kept_images = []
    dropped_images = []

    for img_id in sorted(model.reg_image_ids()):
        im = model.images[img_id]
        visible_pts = [p.point3D_id for p in im.points2D if p.has_point3D()]
        errs = [model.points3D[pid].error for pid in visible_pts]
        median_err = float(np.median(errs)) if errs else 999.0
        pose = im.cam_from_world()
        r_cw = pose.rotation.matrix() @ rotation.T
        eye = base_cam.eye + (im.projection_center() - origin) @ rotation.T * factor
        t_cw = -r_cw @ eye
        horiz_dist = float(np.linalg.norm(eye[[0, 2]] - scene_center[[0, 2]]))
        if im.name != ref_image_name and (
            len(visible_pts) < min_image_points
            or median_err > max_reproj_px
            or eye[1] < 0.25
            or eye[1] > 350.0
            or horiz_dist > max_radius_m
        ):
            dropped_images.append((im.name, len(visible_pts), round(median_err, 3)))
            continue
        quat = rotation_matrix_to_colmap_quat(r_cw)
        kept_images.append((img_id, im, quat, t_cw, eye, median_err, len(visible_pts)))
        used_cameras.add(im.camera_id)

    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for cid in sorted(used_cameras):
        cam = model.cameras[cid]
        params = " ".join(f"{v:.8f}" for v in cam.params)
        cameras_lines.append(f"{cid} {cam.model.name} {cam.width} {cam.height} {params}")
    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf8")

    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for img_id, im, quat, t_cw, _, _, _ in kept_images:
        q_str = " ".join(f"{v:.10f}" for v in quat)
        t_str = " ".join(f"{v:.8f}" for v in t_cw)
        images_lines.append(f"{img_id} {q_str} {t_str} {im.camera_id} {im.name}")
        images_lines.append("")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf8")

    points_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    kept_points = 0
    xs, ys, zs = [], [], []
    for pid, pt in model.points3D.items():
        if pt.error > 2.5 or pt.track.length() < 3:
            continue
        xyz = base_cam.eye + (pt.xyz - origin) @ rotation.T * factor
        horiz_dist = float(np.linalg.norm(xyz[[0, 2]] - scene_center[[0, 2]]))
        if horiz_dist > max_radius_m or xyz[1] < -15.0 or xyz[1] > 140.0:
            continue
        r, g, b = (int(np.clip(c, 0, 255)) for c in pt.color)
        points_lines.append(
            f"{pid} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} {r} {g} {b} {pt.error:.4f}"
        )
        kept_points += 1
        xs.append(xyz[0])
        ys.append(xyz[1])
        zs.append(xyz[2])
    (sparse_dir / "points3D.txt").write_text("\n".join(points_lines) + "\n", encoding="utf8")

    per_source: dict[str, int] = {}
    for _, im, *_ in kept_images:
        folder = im.name.split("/")[0] if "/" in im.name else "root"
        per_source[folder] = per_source.get(folder, 0) + 1

    return {
        "registeredImages": len(kept_images),
        "droppedImages": len(dropped_images),
        "droppedDetails": dropped_images[:20],
        "perSource": per_source,
        "medianReprojPx": round(float(np.median([x[5] for x in kept_images])), 3)
        if kept_images
        else None,
        "points3D": kept_points,
        "worldBounds": {
            "min": [round(min(xs), 2), round(min(ys), 2), round(min(zs), 2)],
            "max": [round(max(xs), 2), round(max(ys), 2), round(max(zs), 2)],
        }
        if kept_points
        else None,
    }


def build_match_pairs(
    ref_rel: str,
    clip_groups: list[tuple[str, np.ndarray, list[str]]],
) -> list[tuple[str, str]]:
    """Build targeted intra-clip, loop-closure, reference-anchor, and cross-clip image pairs."""
    pairs: set[tuple[str, str]] = set()

    def add_pair(a: str, b: str) -> None:
        if a != b:
            pairs.add((a, b) if a < b else (b, a))

    clip_lists = [names for tag, _, names in clip_groups if tag != "ref" and names]
    offsets = (1, 2, 3, 4, 6, 9, 14)

    for names in clip_lists:
        n = len(names)
        for i, name in enumerate(names):
            for d in offsets:
                if i + d < n:
                    add_pair(name, names[i + d])
        # Loop closure between start and tail frames
        for a in names[:6]:
            for b in names[-6:]:
                add_pair(a, b)
        # Anchor reference image strongly to start frames, sampled frames, and tail frame
        for a in names[:6]:
            add_pair(ref_rel, a)
        for a in names[::3]:
            add_pair(ref_rel, a)
        add_pair(ref_rel, names[-1])

    for i in range(len(clip_lists)):
        for j in range(i + 1, len(clip_lists)):
            a_names = clip_lists[i]
            b_names = clip_lists[j]
            for a in a_names[:4]:
                for b in b_names[:4]:
                    add_pair(a, b)
            for a in a_names[::5]:
                for b in b_names[::5]:
                    add_pair(a, b)

    return sorted(pairs)


def run_joint_sfm(
    plan_path: Path,
    reference_path: Path,
    geometry_path: Path,
    video_paths: list[Path],
    colmap_dir: Path,
    stride_short: int = 2,
    stride_long: int = 5,
    max_frames_per_video: int = 75,
    max_num_features: int = 4500,
    num_threads: int = min(16, os.cpu_count() or 8),
) -> dict:
    """Extract frames, run unified pycolmap SfM anchored to reference, and align to world coords."""
    import pycolmap

    plan = json.loads(plan_path.read_text("utf-8-sig"))
    key = plan["views"][0]["source"]
    _, cams, photos, roofs, _ = prepare(plan, {key: reference_path})
    base_cam = cams[key]
    scene_center = np.median(np.concatenate(roofs), axis=0) if roofs else np.zeros(3)

    sfm_work = colmap_dir.parent / "sfm_work"
    images_dir = colmap_dir / "images"
    masks_dir = colmap_dir / "masks"
    sparse_dir = colmap_dir / "sparse" / "0"
    db_path = sfm_work / "database.db"

    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    if sfm_work.exists():
        shutil.rmtree(sfm_work)
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    sfm_work.mkdir(parents=True, exist_ok=True)

    ref_rel = "ref/0000_reference.jpg"
    ref_out = images_dir / ref_rel
    ref_out.parent.mkdir(parents=True, exist_ok=True)
    ref_img = photos[key]
    ref_img.save(ref_out, format="JPEG", quality=96, subsampling=0)
    write_watermark_mask(ref_img.width, ref_img.height, masks_dir / f"{ref_rel}.png")
    # Brush resolves image masks at <images_dir>/masks/<file_stem>.png
    write_watermark_mask(ref_img.width, ref_img.height, images_dir / "masks" / "0000_reference.png")
    ref_features = photo_features(cv2.cvtColor(np.asarray(ref_img), cv2.COLOR_RGB2GRAY))

    clip_groups: list[tuple[str, np.ndarray, list[str]]] = [
        ("ref", base_cam.k, [ref_rel])
    ]
    unique_videos = deduplicate_videos(video_paths)
    for idx, vp in enumerate(unique_videos):
        clip_tag = f"clip{idx}_{digest(vp)[:8]}"
        cap = cv2.VideoCapture(str(vp))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok0, bgr0 = cap.read()
        cap.release()
        if not ok0:
            continue
        stride = stride_short if frame_count <= 120 else stride_long
        names = extract_sharp_frames(
            vp,
            images_dir / clip_tag,
            clip_tag,
            stride=stride,
            max_frames=max_frames_per_video,
        )
        first_gray = cv2.cvtColor(bgr0, cv2.COLOR_BGR2GRAY)
        k_clip, _, _, _ = initial_intrinsics(ref_features, first_gray, base_cam)
        clip_groups.append((clip_tag, k_clip, names))

    ext_opts = pycolmap.FeatureExtractionOptions()
    ext_opts.num_threads = num_threads
    ext_opts.sift.max_num_features = max_num_features

    for clip_tag, k_mat, names in clip_groups:
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = ",".join(
            f"{v:.8f}"
            for v in (k_mat[0, 0], k_mat[1, 1], k_mat[0, 2] + 0.5, k_mat[1, 2] + 0.5)
        )
        if clip_tag == "ref":
            ro.mask_path = str(masks_dir)
        pycolmap.extract_features(
            db_path,
            images_dir,
            image_names=names,
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=ro,
            extraction_options=ext_opts,
        )

    pairs = build_match_pairs(ref_rel, clip_groups)
    pairs_path = sfm_work / "pairs.txt"
    pairs_path.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n", encoding="utf8")
    match_opts = pycolmap.FeatureMatchingOptions()
    match_opts.num_threads = num_threads
    pair_opts = pycolmap.ImportedPairingOptions()
    pair_opts.match_list_path = str(pairs_path)
    pair_opts.block_size = 50
    pycolmap.match_image_pairs(db_path, matching_options=match_opts, pairing_options=pair_opts)

    raw_sparse = sfm_work / "raw_sparse"
    raw_sparse.mkdir(parents=True, exist_ok=True)
    map_opts = pycolmap.IncrementalPipelineOptions()
    map_opts.num_threads = num_threads
    map_opts.mapper.num_threads = num_threads
    map_opts.random_seed = 19
    map_opts.ba_use_gpu = False
    map_opts.ba_refine_focal_length = False
    map_opts.ba_refine_extra_params = False
    map_opts.mapper.abs_pose_refine_focal_length = False
    map_opts.mapper.abs_pose_refine_extra_params = False
    map_opts.mapper.init_min_tri_angle = 4.0
    map_opts.mapper.init_max_forward_motion = 0.995
    map_opts.mapper.filter_max_reproj_error = 2.5
    map_opts.ba_global_max_num_iterations = 30

    reconstructions = list(
        pycolmap.incremental_mapping(db_path, images_dir, raw_sparse, map_opts).values()
    )
    chosen = None
    for model in sorted(reconstructions, key=lambda m: -m.num_reg_images()):
        if any(model.images[i].name == ref_rel for i in model.reg_image_ids()):
            chosen = model
            break
    if chosen is None:
        raise ValueError("联合 SfM 未生成包含参考图的重建分量")

    ref_depth = mesh_depth(geometry_path, base_cam)
    rotation, origin, factor, scale_err_pct = compute_sim3_alignment(
        chosen, ref_rel, base_cam, ref_depth
    )
    summary = write_aligned_colmap_sparse(
        chosen,
        sparse_dir,
        ref_rel,
        base_cam,
        rotation,
        origin,
        factor,
        scene_center,
    )
    summary["scaleMedianDepthErrorPercent"] = scale_err_pct
    summary["inputImages"] = sum(len(names) for _, _, names in clip_groups)
    summary["videosUsed"] = [str(vp.name) for vp in unique_videos]
    return summary


_PLY_TYPE_MAP = {
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
    "uchar": "<u1",
    "uint8": "<u1",
    "char": "<i1",
    "int8": "<i1",
    "ushort": "<u2",
    "uint16": "<u2",
    "short": "<i2",
    "int16": "<i2",
    "uint": "<u4",
    "uint32": "<u4",
    "int": "<i4",
    "int32": "<i4",
}


def prune_splat_ply(
    src_ply: Path,
    dst_ply: Path,
    center_xz: tuple[float, float] = (0.0, 0.0),
    max_radius_m: float = 1200.0,
    min_y: float = -8.0,
    max_y: float = 20.0,
    min_raw_opacity: float = -3.6,
    max_log_scale: float = 1.9,
    max_needle_ratio: float = 14.0,
    needle_min_scale: float = 1.2,
    bounds_xz: tuple[float, float, float, float] | None = None,
) -> dict:
    """Filter NaN, ultra-transparent, oversized floater, and out-of-bounds splats from a binary PLY."""
    raw = src_ply.read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        marker = b"end_header\r\n"
        idx = raw.find(marker)
    if idx < 0:
        raise ValueError(f"无法识别 PLY header: {src_ply}")

    header_bytes = raw[: idx + len(marker)]
    body = raw[idx + len(marker) :]
    header_lines = header_bytes.decode("ascii").splitlines()

    vertex_count = None
    in_vertex = False
    fields: list[tuple[str, str]] = []
    new_header_lines: list[str] = []

    for line in header_lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            ptype, pname = parts[1], parts[2]
            if ptype not in _PLY_TYPE_MAP:
                raise ValueError(f"不支持的 PLY 属性类型: {ptype}")
            fields.append((pname, _PLY_TYPE_MAP[ptype]))
        elif parts == ["format", "ascii", "1.0"]:
            raise ValueError("仅支持 binary_little_endian PLY")
        new_header_lines.append(line)

    if vertex_count is None or not fields:
        raise ValueError(f"PLY 缺少 vertex 定义: {src_ply}")

    dtype = np.dtype(fields)
    expected_bytes = vertex_count * dtype.itemsize
    if len(body) < expected_bytes:
        raise ValueError(f"PLY 数据截断: expected {expected_bytes}, got {len(body)}")

    arr = np.frombuffer(body[:expected_bytes], dtype=dtype)
    keep = np.ones(vertex_count, dtype=bool)
    for name, _ in fields:
        if np.issubdtype(arr[name].dtype, np.floating):
            keep &= np.isfinite(arr[name])

    if all(k in arr.dtype.names for k in ("x", "y", "z")):
        dx = arr["x"] - center_xz[0]
        dz = arr["z"] - center_xz[1]
        keep &= (dx * dx + dz * dz) <= (max_radius_m * max_radius_m)
        keep &= (arr["y"] >= min_y) & (arr["y"] <= max_y)
        if bounds_xz is not None:
            min_x, max_x, min_z, max_z = bounds_xz
            keep &= (arr["x"] >= min_x) & (arr["x"] <= max_x)
            keep &= (arr["z"] >= min_z) & (arr["z"] <= max_z)

    if "opacity" in arr.dtype.names:
        keep &= arr["opacity"] >= min_raw_opacity

    scale_names = [k for k in ("scale_0", "scale_1", "scale_2") if k in arr.dtype.names]
    if scale_names:
        max_scale = np.maximum.reduce([arr[k] for k in scale_names])
        keep &= max_scale <= max_log_scale
        if len(scale_names) == 3:
            # Prune only 1D needle spikes (s_max >> s_mid), preserving flat 2D roof/road disks (s_max ~= s_mid >> s_min).
            s_sorted = np.sort(np.column_stack([arr[k] for k in scale_names]), axis=1)
            s_mid = s_sorted[:, 1]
            s_max_log = s_sorted[:, 2]
            s_exp = np.exp(np.clip(s_max_log, -20.0, 20.0))
            needle_ratio = np.exp(np.clip(s_max_log - s_mid, 0.0, 30.0))
            keep &= ~((s_exp > needle_min_scale) & (needle_ratio > max_needle_ratio))

    pruned = np.ascontiguousarray(arr[keep])
    kept_count = int(len(pruned))
    out_header = "\n".join(
        f"element vertex {kept_count}" if l.strip().startswith("element vertex ") else l
        for l in new_header_lines
    )
    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    with dst_ply.open("wb") as f:
        f.write((out_header + "\n").encode("ascii"))
        f.write(pruned.tobytes())

    radius_p99 = None
    if kept_count > 0 and all(k in pruned.dtype.names for k in ("x", "z")):
        pruned_r = np.hypot(pruned["x"] - center_xz[0], pruned["z"] - center_xz[1])
        radius_p99 = round(float(np.percentile(pruned_r, 99)), 2)

    scale_p99 = None
    if kept_count > 0 and scale_names:
        pruned_max_scale = np.maximum.reduce([pruned[k] for k in scale_names])
        scale_p99 = round(float(np.percentile(pruned_max_scale, 99)), 3)

    return {
        "inputSplats": int(vertex_count),
        "keptSplats": kept_count,
        "removedSplats": int(vertex_count - kept_count),
        "radiusP99": radius_p99,
        "maxLogScaleP99": scale_p99,
        "bytes": dst_ply.stat().st_size,
    }


def train_with_brush(
    brush_bin: Path,
    colmap_dir: Path,
    export_dir: Path,
    total_steps: int = 24000,
    max_splats: int = 450000,
    sh_degree: int = 1,
    refine_every: int = 200,
    growth_stop_iter: int = 12000,
    eval_every: int = 4000,
    eval_split_every: int | None = 12,
    max_resolution: int = 1536,
    scale_loss_weight: float = 1e-8,
    opac_loss_weight: float = 1e-9,
) -> Path:
    """Invoke Brush CLI headless to train 3DGS on a COLMAP dataset."""
    if not brush_bin.is_file():
        raise FileNotFoundError(f"找不到 Brush 可执行文件: {brush_bin}")
    has_txt_sparse = (colmap_dir / "sparse" / "0" / "cameras.txt").is_file()
    for p in colmap_dir.rglob("*"):
        if not p.is_file():
            continue
        if has_txt_sparse and p.name in ("cameras.bin", "images.bin"):
            raise ValueError(
                f"检测到冲突的二进制 COLMAP 文件 {p}：Brush 会优先读取 .bin 而忽略对齐后的 sparse/0/*.txt"
            )
        if p.suffix.lower() == ".ply":
            raise ValueError(
                f"检测到冲突的 PLY 文件 {p}：Brush 会误将其当作 init.ply 覆盖 points3D.txt"
            )
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(brush_bin),
        str(colmap_dir),
        "--total-steps",
        str(total_steps),
        "--max-splats",
        str(max_splats),
        "--sh-degree",
        str(sh_degree),
        "--refine-every",
        str(refine_every),
        "--growth-stop-iter",
        str(growth_stop_iter),
        "--eval-every",
        str(eval_every),
        "--eval-save-to-disk",
        "--scale-loss-weight",
        str(scale_loss_weight),
        "--opac-loss-weight",
        str(opac_loss_weight),
        "--max-resolution",
        str(max_resolution),
        "--export-every",
        str(total_steps),
        "--export-path",
        str(export_dir),
        "--export-name",
        "export_{iter}.ply",
    ]
    if eval_split_every and eval_split_every > 0:
        cmd.extend(["--eval-split-every", str(eval_split_every)])

    print("Running Brush:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Brush 训练失败 (exit={proc.returncode})")

    plys = sorted(export_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not plys:
        raise FileNotFoundError(f"Brush 未导出任何 PLY 文件到 {export_dir}")
    return plys[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, default=ROOT / "builder/example.json")
    p.add_argument("--reference", type=Path, default=ROOT / "assets/scene_sources/village_road_near.png")
    p.add_argument("--geometry", type=Path, default=ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.glb")
    p.add_argument("--videos", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/3dgs_build")
    p.add_argument("--export-ply", type=Path, default=ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.ply")
    p.add_argument("--brush-bin", type=Path, default=DEFAULT_BRUSH_BIN)
    p.add_argument("--stride-short", type=int, default=2)
    p.add_argument("--stride-long", type=int, default=5)
    p.add_argument("--max-frames-per-video", type=int, default=75)
    p.add_argument("--steps", type=int, default=24000)
    p.add_argument("--max-splats", type=int, default=450000)
    p.add_argument("--sh-degree", type=int, default=1)
    p.add_argument("--scale-loss-weight", type=float, default=1e-8)
    p.add_argument("--opac-loss-weight", type=float, default=1e-9)
    p.add_argument("--refine-every", type=int, default=200)
    p.add_argument("--growth-stop-iter", type=int, default=12000)
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--eval-split-every", type=int, default=12)
    p.add_argument("--sfm-only", action="store_true")
    p.add_argument("--skip-sfm", action="store_true")
    p.add_argument("--prune-only", action="store_true")
    return p.parse_args(argv)


def compute_sparse_bounds_xz(
    colmap_dir: Path,
    ref_image_name: str = "ref/0000_reference.jpg",
) -> tuple[float, float, float, float] | None:
    """Compute padded XZ bounds from sparse/0/points3D.txt, clamped behind reference camera corridor."""
    sparse_dir = colmap_dir / "sparse" / "0"
    pts_file = sparse_dir / "points3D.txt"
    if not pts_file.is_file():
        return None
    xs: list[float] = []
    zs: list[float] = []
    for line in pts_file.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            xs.append(float(parts[1]))
            zs.append(float(parts[3]))
    if len(xs) < 10:
        return None
    min_x = float(np.percentile(xs, 1)) - 65.0
    max_x = float(np.percentile(xs, 99)) + 65.0
    min_z = float(np.percentile(zs, 1)) - 65.0
    max_z = float(np.percentile(zs, 99)) + 65.0

    imgs_file = sparse_dir / "images.txt"
    if imgs_file.is_file():
        for line in imgs_file.read_text(encoding="utf8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[9] == ref_image_name:
                qw, qx, qy, qz = (float(v) for v in parts[1:5])
                t = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
                r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                eye = -r.T @ t
                max_z = min(max_z, float(eye[2]) - 16.0)
                break
    return (min_x, max_x, min_z, max_z)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    work = args.out
    work.mkdir(parents=True, exist_ok=True)
    colmap_dir = work / "colmap"
    export_dir = work / "export"
    report_path = work / "alignment_report.json"

    if not args.skip_sfm and not args.prune_only:
        sfm_report = run_joint_sfm(
            args.plan,
            args.reference,
            args.geometry,
            args.videos,
            colmap_dir,
            stride_short=args.stride_short,
            stride_long=args.stride_long,
            max_frames_per_video=args.max_frames_per_video,
        )
        report_path.write_text(json.dumps(sfm_report, indent=2, ensure_ascii=False), encoding="utf8")
        print("SfM report:", json.dumps(sfm_report, ensure_ascii=False), flush=True)
    else:
        sfm_report = json.loads(report_path.read_text(encoding="utf8"))

    if args.sfm_only:
        return

    if args.prune_only:
        plys = sorted(export_dir.glob("*.ply"), key=lambda p: p.stat().st_mtime)
        if not plys:
            raise FileNotFoundError(f"未在 {export_dir} 找到可裁剪的 PLY 文件")
        raw_ply = plys[-1]
    else:
        raw_ply = train_with_brush(
            args.brush_bin,
            colmap_dir,
            export_dir,
            total_steps=args.steps,
            max_splats=args.max_splats,
            sh_degree=args.sh_degree,
            refine_every=args.refine_every,
            growth_stop_iter=args.growth_stop_iter,
            eval_every=args.eval_every,
            eval_split_every=args.eval_split_every,
            scale_loss_weight=args.scale_loss_weight,
            opac_loss_weight=args.opac_loss_weight,
        )
    center = (0.0, 0.0)
    if sfm_report.get("worldBounds"):
        bmin, bmax = sfm_report["worldBounds"]["min"], sfm_report["worldBounds"]["max"]
        center = (0.5 * (bmin[0] + bmax[0]), 0.5 * (bmin[2] + bmax[2]))
    bounds_xz = compute_sparse_bounds_xz(colmap_dir)
    prune_stats = prune_splat_ply(raw_ply, args.export_ply, center_xz=center, bounds_xz=bounds_xz)
    sfm_report["plyExport"] = {**prune_stats, "boundsXZ": list(bounds_xz) if bounds_xz else None, "path": str(args.export_ply)}
    report_path.write_text(json.dumps(sfm_report, indent=2, ensure_ascii=False), encoding="utf8")
    print("3DGS export:", json.dumps(sfm_report["plyExport"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
