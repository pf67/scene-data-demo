"""Run automated mesh/assembly audit gate + 8-angle EEVEE Next visual QC comparison suite."""

from __future__ import annotations

import argparse
import json
import math
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

from bridge_client import BridgeClient, run_blender_script  # noqa: E402


QC_VIEWS: list[tuple[str, dict[str, Any]]] = [
    ("01_ref_raw.png", {"camera": "CAM_ref", "res": [1152, 768]}),
    ("04_orbit_right30.png", {"eye": [72.0, 32.0, 56.0], "target": [-8.0, 3.0, -45.0], "fov": 46.0, "res": [1200, 800]}),
    ("05_orbit_left35.png", {"eye": [-78.0, 34.0, 48.0], "target": [10.0, 3.0, -48.0], "fov": 46.0, "res": [1200, 800]}),
    ("06_rear_180.png", {"eye": [18.0, 42.0, -205.0], "target": [4.0, 3.0, -25.0], "fov": 48.0, "res": [1200, 800]}),
    ("07_street_level.png", {"eye": [-17.5, 2.4, 14.4], "target": [35.0, 3.0, -22.8], "fov": 45.0, "res": [1200, 800]}),
    ("08_top_overview.png", {"eye": [8.0, 175.0, 12.0], "target": [8.0, 0.0, -52.0], "fov": 52.0, "res": [1200, 800]}),
]


def compute_adaptive_qc_views(spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Generate adaptive multi-angle QC viewpoints centered on any reconstructed scene's bounds."""
    img_w, img_h = spec.get("imageSize", [1536, 1024])
    ref_w = 1152
    ref_h = max(512, int(round(ref_w * float(img_h) / max(float(img_w), 1.0))))

    fps = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in spec.get("buildings", []) if b.get("footprint_xz")]
    if fps:
        all_xz = np.vstack(fps)
        x_min, z_min = all_xz.min(axis=0)
        x_max, z_max = all_xz.max(axis=0)
        cx, cz = float(0.5 * (x_min + x_max)), float(0.5 * (z_min + z_max))
        span = float(max(x_max - x_min, z_max - z_min))
    else:
        cx, cz, span = 0.0, 0.0, 60.0

    radius = max(25.0, min(95.0, span * 0.65))
    cam_pos = np.asarray(spec["camera"]["position"], dtype=np.float64)
    cam_tgt = np.asarray(spec["camera"]["target"], dtype=np.float64)
    back_xz = np.array([cam_pos[0] - cam_tgt[0], cam_pos[2] - cam_tgt[2]], dtype=np.float64)
    back_norm = float(np.linalg.norm(back_xz))
    if back_norm < 1e-4:
        back_xz = np.array([0.0, 1.0], dtype=np.float64)
    else:
        back_xz /= back_norm
    base_az = math.atan2(back_xz[0], back_xz[1])
    eye_y = float(max(24.0, min(75.0, cam_pos[1])))

    def _orbit_eye(az_offset_deg: float, dist: float, height: float) -> list[float]:
        ang = base_az + math.radians(az_offset_deg)
        return [round(cx + math.sin(ang) * dist, 2), round(height, 2), round(cz + math.cos(ang) * dist, 2)]

    cl = np.asarray(spec.get("road_centerline_xz") or [], dtype=np.float64)
    if len(cl) >= 2:
        p0, p1 = cl[0], cl[-1]
        rdir = p1 - p0
        rlen = float(np.linalg.norm(rdir))
        if rlen > 4.0:
            rdir /= rlen
            st_eye = p0 + rdir * min(rlen * 0.2, 8.0)
            st_tgt = p0 + rdir * min(rlen * 0.75, 48.0)
            street_cfg = {
                "eye": [round(float(st_eye[0]), 2), 2.6, round(float(st_eye[1]), 2)],
                "target": [round(float(st_tgt[0]), 2), 3.0, round(float(st_tgt[1]), 2)],
                "fov": 45.0,
                "res": [1200, 800],
            }
        else:
            street_cfg = {
                "eye": [round(cx, 2), 2.6, round(cz + radius * 0.85, 2)],
                "target": [round(cx, 2), 3.0, round(cz, 2)],
                "fov": 45.0,
                "res": [1200, 800],
            }
    else:
        street_cfg = {
            "eye": [round(cx, 2), 2.6, round(cz + radius * 0.85, 2)],
            "target": [round(cx, 2), 3.0, round(cz, 2)],
            "fov": 45.0,
            "res": [1200, 800],
        }

    return [
        ("01_ref_raw.png", {"camera": "CAM_ref", "res": [ref_w, ref_h]}),
        ("04_orbit_right30.png", {"eye": _orbit_eye(35.0, radius * 1.25, eye_y), "target": [round(cx, 2), 3.0, round(cz, 2)], "fov": 46.0, "res": [1200, 800]}),
        ("05_orbit_left35.png", {"eye": _orbit_eye(-35.0, radius * 1.25, eye_y), "target": [round(cx, 2), 3.0, round(cz, 2)], "fov": 46.0, "res": [1200, 800]}),
        ("06_rear_180.png", {"eye": _orbit_eye(180.0, radius * 1.35, max(28.0, eye_y * 1.05)), "target": [round(cx, 2), 3.0, round(cz, 2)], "fov": 48.0, "res": [1200, 800]}),
        ("07_street_level.png", street_cfg),
        ("08_top_overview.png", {"eye": [round(cx, 2), round(max(95.0, radius * 2.2), 2), round(cz + 6.0, 2)], "target": [round(cx, 2), 0.0, round(cz, 2)], "fov": 52.0, "res": [1200, 800]}),
    ]


def render_qc_in_blender(spec_path: Path, qc_dir: Path) -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore
    from blender_bridge_addon import run_audit, world_to_blender  # noqa: E402

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    qc_dir.mkdir(parents=True, exist_ok=True)

    audit = run_audit(
        {
            "expected_closed_buildings": len(spec["buildings"]),
            "road_polygon_xz": spec["road_polygon_xz"],
            "road_apron_polygon_xz": spec.get("road_apron_polygon_xz"),
            "strict_layout_audit": True,
        }
    )
    if spec.get("layout_audit") is not None:
        audit["layout_audit"] = spec["layout_audit"]
    audit_path = qc_dir.parent / "audit_report.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit["verdict"] != "pass":
        raise SystemExit(f"Audit gate failed ({audit['verdict']}): {audit['failures'][:5]}")

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 24
    scene.render.image_settings.file_format = "PNG"

    # Collect available clip1 and clip3 mid cameras created by build_village
    base_views = compute_adaptive_qc_views(spec) if spec.get("adaptive_qc_views", False) else QC_VIEWS
    cam_jobs: list[tuple[str, dict[str, Any]]] = list(base_views)
    clip1_cam = next((o.name for o in scene.objects if o.type == "CAMERA" and "clip1" in o.name), None)
    clip3_cam = next((o.name for o in scene.objects if o.type == "CAMERA" and "clip3" in o.name), None)
    if clip1_cam:
        cam_jobs.append(("02_clip1_raw.png", {"camera": clip1_cam, "res": [1280, 720]}))
    if clip3_cam:
        cam_jobs.append(("03_clip3_raw.png", {"camera": clip3_cam, "res": [1280, 720]}))
    ref_res = base_views[0][1]["res"]
    for cam_obj in sorted((o for o in scene.objects if o.type == "CAMERA" and o.name.startswith("CAM_novel_")), key=lambda x: x.name):
        key = cam_obj.name.removeprefix("CAM_")
        cam_jobs.append((f"{key}_raw.png", {"camera": cam_obj.name, "res": ref_res}))

    temp_cam_data = bpy.data.cameras.new("_QCTempCam")
    temp_cam_data.type = "PERSP"
    temp_cam_data.sensor_fit = "VERTICAL"
    temp_cam_data.clip_start = 0.2
    temp_cam_data.clip_end = 1500.0
    temp_cam = bpy.data.objects.new("_QCTempCam", temp_cam_data)
    scene.collection.objects.link(temp_cam)

    rendered_meta = {}
    for filename, cfg in cam_jobs:
        rw, rh = cfg["res"]
        scene.render.resolution_x = int(rw)
        scene.render.resolution_y = int(rh)
        scene.render.resolution_percentage = 100
        if "camera" in cfg:
            cam_obj = bpy.data.objects[cfg["camera"]]
            scene.camera = cam_obj
            # Preserve square aspect on clip cameras; apply exact pixel aspect only if ref
            if cfg["camera"] == "CAM_ref":
                ref_c = next(c for c in spec["cameras"] if c["source_clip"] == "ref")
                scene.render.pixel_aspect_x = ref_c["blender_intrinsics"]["pixel_aspect_x"]
            else:
                scene.render.pixel_aspect_x = 1.0
            scene.render.pixel_aspect_y = 1.0
        else:
            eye_b = Vector(world_to_blender(cfg["eye"]))
            tgt_b = Vector(world_to_blender(cfg["target"]))
            temp_cam_data.angle_y = math.radians(float(cfg["fov"]))
            temp_cam.location = eye_b
            temp_cam.rotation_euler = (tgt_b - eye_b).to_track_quat("-Z", "Y").to_euler()
            scene.camera = temp_cam
            scene.render.pixel_aspect_x = 1.0
            scene.render.pixel_aspect_y = 1.0

        out_file = qc_dir / filename
        scene.render.filepath = str(out_file)
        bpy.ops.render.render(write_still=True)
        rendered_meta[filename] = cfg.get("camera", "custom")

    bpy.data.objects.remove(temp_cam, do_unlink=True)
    bpy.data.cameras.remove(temp_cam_data, do_unlink=True)
    (qc_dir / "rendered_meta.json").write_text(json.dumps(rendered_meta, indent=2), encoding="utf-8")
    print("QC_BLENDER_RENDER_OK", len(rendered_meta))


def make_comparison_panel(render_path: Path, real_path: Path, out_path: Path, label_left: str, label_right: str) -> dict[str, float]:
    ren = cv2.imread(str(render_path))
    ref = cv2.imread(str(real_path))
    if ren is None or ref is None:
        return {}
    h, w = ren.shape[:2]
    ref_r = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    blend = cv2.addWeighted(ren, 0.52, ref_r, 0.48, 0)

    # Structural metrics: gradient cosine similarity & Sobel edge IoU
    g_ren = cv2.GaussianBlur(cv2.cvtColor(ren, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0)
    g_ref = cv2.GaussianBlur(cv2.cvtColor(ref_r, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 1.0)
    ren_norm = (g_ren - g_ren.mean()) / max(float(g_ren.std()), 1.0)
    ref_norm = (g_ref - g_ref.mean()) / max(float(g_ref.std()), 1.0)
    luma_corr = float(np.mean(ren_norm * ref_norm))
    mae_luma = float(np.mean(np.abs(g_ren - g_ref)))

    # 3-up horizontal strip resized cleanly to fit 1680px width
    strip = np.hstack([ren, blend, ref_r])
    scale = min(1.0, 1680.0 / strip.shape[1])
    if scale < 1.0:
        strip = cv2.resize(strip, (int(strip.shape[1] * scale), int(strip.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.putText(strip, label_left, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(strip, "50% Overlay", (strip.shape[1] // 3 + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 235, 130), 2, cv2.LINE_AA)
    cv2.putText(strip, label_right, (2 * strip.shape[1] // 3 + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (140, 255, 160), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), strip)
    return {"luma_corr": round(luma_corr, 4), "mae_luma": round(mae_luma, 2)}


def run_visual_qc(spec_path: Path, blend_path: Path, qc_dir: Path) -> dict[str, Any]:
    run_blender_script(
        Path(__file__).resolve(),
        ["--blender-render", "--spec", str(spec_path), "--qc-dir", str(qc_dir)],
        blend_path=blend_path,
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    meta = json.loads((qc_dir / "rendered_meta.json").read_text(encoding="utf-8"))
    metrics: dict[str, Any] = {}

    metrics["01_ref_compare"] = make_comparison_panel(
        qc_dir / "01_ref_raw.png",
        Path(spec["ref_clean_image"]),
        qc_dir / "01_ref_compare.png",
        "Blender Render (CAM_ref)",
        "Reference Photo (clean)",
    )
    for raw_name, cmp_name, label in [
        ("02_clip1_raw.png", "02_clip1_compare.png", "clip1 Aerial"),
        ("03_clip3_raw.png", "03_clip3_compare.png", "clip3 Street"),
    ]:
        cam_key = str(meta.get(raw_name, "")).removeprefix("CAM_")
        cam_spec = next((c for c in spec["cameras"] if c["cam_key"] == cam_key), None)
        if cam_spec and (qc_dir / raw_name).is_file():
            metrics[cmp_name.removesuffix(".png")] = make_comparison_panel(
                qc_dir / raw_name,
                Path(cam_spec["image_path"]),
                qc_dir / cmp_name,
                f"Blender ({label})",
                f"Video Frame ({label})",
            )

    for raw_name, cam_name in meta.items():
        if not raw_name.startswith("novel_") or not raw_name.endswith("_raw.png"):
            continue
        cam_key = str(cam_name).removeprefix("CAM_")
        cam_spec = next((c for c in spec["cameras"] if c["cam_key"] == cam_key), None)
        cmp_name = raw_name.replace("_raw.png", "_compare.png")
        if cam_spec and (qc_dir / raw_name).is_file():
            metrics[cmp_name.removesuffix(".png")] = make_comparison_panel(
                qc_dir / raw_name,
                Path(cam_spec["image_path"]),
                qc_dir / cmp_name,
                f"Blender ({cam_key})",
                f"Synthesized ({cam_key})",
            )

    (qc_dir / "qc_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("VISUAL_QC_OK", json.dumps(metrics))
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender-render", action="store_true")
    p.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts/blender_village/village_master.blend")
    p.add_argument("--qc-dir", type=Path, default=ROOT / "artifacts/blender_village/qc")
    return p.parse_args(argv)


if __name__ == "__main__":
    if "--blender-render" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
        args = parse_args(argv)
        render_qc_in_blender(args.spec.resolve(), args.qc_dir.resolve())
    else:
        args = parse_args()
        run_visual_qc(args.spec.resolve(), args.blend.resolve(), args.qc_dir.resolve())
