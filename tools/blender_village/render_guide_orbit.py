"""Render a geometry-locked 360-degree Blender orbital turntable MP4 + 4 quadrant guides for MiniMax-H3."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

from blender_bridge_addon import world_to_blender  # noqa: E402
from bridge_client import run_blender_script  # noqa: E402


def compute_orbit_poses(
    spec: dict[str, Any],
    *,
    num_frames: int = 36,
    target_y: float = 4.0,
) -> dict[str, Any]:
    """Compute a smooth clockwise 360-degree orbit centered on core village anchored to CAM_ref azimuth."""
    core_pts = np.vstack([np.asarray(b["footprint_xz"], dtype=np.float64) for b in spec["buildings"] if b.get("is_core", True)])
    cx, cz = float(core_pts[:, 0].mean()), float(core_pts[:, 1].mean())
    eye0 = spec["camera"]["position"]
    dx0, dz0 = float(eye0[0]) - cx, float(eye0[2]) - cz
    radius = math.hypot(dx0, dz0)
    theta0 = math.atan2(dz0, dx0)
    eye_y = float(eye0[1])
    fov = float(spec["camera"].get("fov", 40.0))

    poses = []
    for k in range(num_frames):
        ang_offset = (k / num_frames) * math.tau
        theta = theta0 - ang_offset  # clockwise aerial orbit from CAM_ref
        ex = cx + radius * math.cos(theta)
        ez = cz + radius * math.sin(theta)
        poses.append(
            {
                "frame_index": k,
                "angle_deg": round(k * 360.0 / num_frames, 2),
                "eye_three": [round(ex, 4), round(eye_y, 4), round(ez, 4)],
                "target_three": [round(cx, 4), round(target_y, 4), round(cz, 4)],
                "fov": fov,
            }
        )
    return {"center_three": [round(cx, 3), target_y, round(cz, 3)], "radius_m": round(radius, 3), "poses": poses}


def render_orbit_frames_in_blender(
    spec_path: Path,
    out_dir: Path,
    *,
    num_frames: int = 36,
    width: int = 1280,
    height: int = 720,
    samples: int = 12,
) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    traj = compute_orbit_poses(spec, num_frames=num_frames)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = int(samples)
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"

    cam_data = bpy.data.cameras.new("_GuideOrbitCam")
    cam_data.type = "PERSP"
    cam_data.sensor_fit = "VERTICAL"
    cam_data.clip_start = 0.2
    cam_data.clip_end = 1500.0
    cam_obj = bpy.data.objects.new("_GuideOrbitCam", cam_data)
    scene.collection.objects.link(cam_obj)
    prev_cam = scene.camera
    scene.camera = cam_obj

    try:
        for p in traj["poses"]:
            k = p["frame_index"]
            eye_b = Vector(world_to_blender(p["eye_three"]))
            tgt_b = Vector(world_to_blender(p["target_three"]))
            cam_data.angle_y = math.radians(float(p["fov"]))
            cam_obj.location = eye_b
            cam_obj.rotation_euler = (tgt_b - eye_b).to_track_quat("-Z", "Y").to_euler()
            fpath = frames_dir / f"frame_{k:04d}.png"
            scene.render.filepath = str(fpath)
            bpy.ops.render.render(write_still=True)
            p["image_path"] = str(fpath)
    finally:
        scene.camera = prev_cam
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)

    (out_dir / "orbit_cameras.json").write_text(json.dumps(traj, indent=2), encoding="utf-8")
    print("ORBIT_GUIDE_BLENDER_OK", len(traj["poses"]))


def render_guide_orbit(
    *,
    spec_path: Path,
    blend_path: Path,
    out_dir: Path,
    num_frames: int = 36,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> dict[str, Any]:
    """Render 360 turntable keyframes via Blender and encode smooth orbit_guide.mp4 + 4 quadrant guides."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_blender_script(
        Path(__file__).resolve(),
        [
            "--blender-render",
            "--spec",
            str(spec_path),
            "--out-dir",
            str(out_dir),
            "--num-frames",
            str(num_frames),
            "--width",
            str(width),
            "--height",
            str(height),
        ],
        blend_path=blend_path,
    )

    traj = json.loads((out_dir / "orbit_cameras.json").read_text(encoding="utf-8"))
    frame_paths = [Path(p["image_path"]) for p in traj["poses"]]

    quadrants = {
        "quadrant_000.png": frame_paths[0],
        "quadrant_090.png": frame_paths[num_frames // 4],
        "quadrant_180.png": frame_paths[num_frames // 2],
        "quadrant_270.png": frame_paths[(3 * num_frames) // 4],
    }
    for q_name, src in quadrants.items():
        shutil.copyfile(src, out_dir / q_name)

    shutil.copyfile(frame_paths[0], out_dir / "first_frame.png")
    shutil.copyfile(frame_paths[0], out_dir / "last_frame.png")

    mp4_path = out_dir / "orbit_guide.mp4"
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    try:
        # Smooth ~6s 144-frame turntable repeating each rendered keyframe 4x + final loop frame
        repeat_n = max(1, round(144 / max(1, len(frame_paths))))
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            for _ in range(repeat_n):
                writer.write(img)
        img0 = cv2.imread(str(frame_paths[0]))
        if img0 is not None:
            writer.write(img0)
    finally:
        writer.release()

    res = {
        "mp4": str(mp4_path),
        "cameras_json": str(out_dir / "orbit_cameras.json"),
        "keyframes": len(frame_paths),
        "quadrants": [str(out_dir / q) for q in quadrants],
    }
    print("ORBIT_GUIDE_OK", json.dumps(res))
    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender-render", action="store_true")
    p.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts/blender_village/village_master.blend")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village/orbit_guide")
    p.add_argument("--num-frames", type=int, default=36)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=24)
    return p.parse_args(argv)


if __name__ == "__main__":
    if "--blender-render" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
        args = parse_args(argv)
        render_orbit_frames_in_blender(
            args.spec.resolve(), args.out_dir.resolve(),
            num_frames=args.num_frames, width=args.width, height=args.height,
        )
    else:
        args = parse_args()
        render_guide_orbit(
            spec_path=args.spec.resolve(),
            blend_path=args.blend.resolve(),
            out_dir=args.out_dir.resolve(),
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
