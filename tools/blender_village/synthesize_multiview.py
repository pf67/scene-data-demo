"""Multi-angle novel-view synthesis (GPT Image 2 + Colab MiniMax-H3), geometric ECC verification, and camera export."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:  # Blender's bundled Python only needs numpy + bpy for guide rendering
    cv2 = None  # type: ignore
    Image = None  # type: ignore

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/blender_village"),
    str(ROOT / "tools/scene_pipeline"),
]

from blender_bridge_addon import (  # noqa: E402
    colmap_intrinsics_to_blender_camera,
    colmap_pose_to_blender_matrix,
)
from bridge_client import run_blender_script  # noqa: E402

if cv2 is not None and Image is not None:
    from prepare_evidence import laplacian_sharpness  # noqa: E402
    from refinement import image_api  # noqa: E402
    from refinement import minimax_h3_image as mm_img  # noqa: E402
    import minimax_h3_orbit as mm_orb  # noqa: E402
else:
    laplacian_sharpness = None  # type: ignore
    image_api = None  # type: ignore
    mm_img = None  # type: ignore
    mm_orb = None  # type: ignore


def load_workspace_env(env_path: Path | None = None) -> None:
    """Load non-secret and API keys from .env into os.environ without overwriting caller env or printing values."""
    target = env_path if env_path is not None else (ROOT / ".env")
    if not target.is_file():
        return
    for raw in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        k = key.strip()
        v = val.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)


def _angle_tag(angle_deg: float) -> str:
    a = int(round(angle_deg))
    if abs(abs(a) - 180) <= 5:
        return "rear180"
    if a > 0:
        return f"right{a}"
    if a < 0:
        return f"left{abs(a)}"
    return "front0"


def _opencv_extrinsics_from_lookat(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build right-handed OpenCV world->cam (R_cw, t_cw) from Three.js Y-up look-at vectors."""
    fwd = target - eye
    fwd /= max(float(np.linalg.norm(fwd)), 1e-8)
    up_w = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up_w))) > 0.99:
        up_w = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up_w)
    right /= max(float(np.linalg.norm(right)), 1e-8)
    down = np.cross(fwd, right)
    down /= max(float(np.linalg.norm(down)), 1e-8)
    r_cw = np.stack([right, down, fwd], axis=0)
    t_cw = -r_cw @ eye
    return r_cw, t_cw


def compute_multiview_orbit_cameras(
    spec: dict[str, Any],
    angles_deg: list[float] | tuple[float, ...],
    *,
    width: int = 1248,
    height: int = 832,
) -> list[dict[str, Any]]:
    """Compute calibrated orbit camera poses around building cluster centroid matching CAM_ref altitude & FOV."""
    b_pts = [np.asarray(b["footprint_xz"], dtype=np.float64) for b in spec.get("buildings", []) if b.get("footprint_xz")]
    if b_pts:
        all_xz = np.vstack(b_pts)
        cx_w, cz_w = float(all_xz[:, 0].mean()), float(all_xz[:, 1].mean())
    else:
        tgt0 = spec["camera"]["target"]
        cx_w, cz_w = float(tgt0[0]), float(tgt0[2])
    target = np.array([cx_w, 3.0, cz_w], dtype=np.float64)

    eye0 = np.asarray(spec["camera"]["position"], dtype=np.float64)
    rel0 = eye0 - target
    fov_y = float(spec["camera"].get("fov", 40.0))
    fy = height / (2.0 * math.tan(math.radians(fov_y) * 0.5))
    fx = fy
    cx = width * 0.5
    cy = height * 0.5
    bl_intr = colmap_intrinsics_to_blender_camera(fx, fy, cx, cy, width, height)

    out: list[dict[str, Any]] = []
    for angle in angles_deg:
        rad = math.radians(float(angle))
        # Orbit +angle clockwise seen from bottom-up -> moves camera to photographer's RIGHT (+X when facing -Z)
        ca, sa = math.cos(rad), math.sin(rad)
        rot_y = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=np.float64)
        eye_a = target + rot_y @ rel0
        eye_a[1] = float(eye0[1])
        r_cw, t_cw = _opencv_extrinsics_from_lookat(eye_a, target)
        blender_eye, _, blender_mat4 = colmap_pose_to_blender_matrix(r_cw.tolist(), t_cw.tolist())
        tag = _angle_tag(float(angle))
        cam_key = f"novel_{tag}"
        out.append(
            {
                "angle_deg": float(angle),
                "cam_key": cam_key,
                "source_clip": cam_key,
                "colmap_name": f"novel/{cam_key}.png",
                "width": int(width),
                "height": int(height),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "eye_three": [round(float(v), 4) for v in eye_a],
                "target_three": [round(float(v), 4) for v in target],
                "r_cw": [[round(float(v), 8) for v in row] for row in r_cw],
                "t_cw": [round(float(v), 6) for v in t_cw],
                "blender_eye": [round(float(v), 6) for v in blender_eye],
                "blender_mat4": blender_mat4,
                "blender_intrinsics": bl_intr,
                "priority": 2.4,
                "norm_t": 0.0,
                "gain_rgb": [1.0, 1.0, 1.0],
                "watermark_box": [1.0, 1.0, 1.0, 1.0],
                "allow_facade_bake": True,
            }
        )
    return out


def _render_guides_in_blender(cameras_json: Path, guides_dir: Path) -> None:
    import bpy  # type: ignore
    from mathutils import Matrix  # type: ignore

    cams = json.loads(cameras_json.read_text(encoding="utf-8"))
    guides_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 16
    scene.render.image_settings.file_format = "PNG"

    temp_data = bpy.data.cameras.new("_MultiViewGuideCam")
    temp_data.type = "PERSP"
    temp_data.clip_start = 0.2
    temp_data.clip_end = 1500.0
    temp_obj = bpy.data.objects.new("_MultiViewGuideCam", temp_data)
    scene.collection.objects.link(temp_obj)
    prev_cam = scene.camera
    scene.camera = temp_obj
    try:
        for cam in cams:
            intr = cam["blender_intrinsics"]
            temp_data.sensor_fit = intr["sensor_fit"]
            temp_data.sensor_width = intr["sensor_width"]
            temp_data.lens = intr["lens"]
            temp_data.shift_x = intr["shift_x"]
            temp_data.shift_y = intr["shift_y"]
            temp_obj.matrix_world = Matrix(cam["blender_mat4"])
            scene.render.resolution_x = int(intr["resolution_x"])
            scene.render.resolution_y = int(intr["resolution_y"])
            scene.render.resolution_percentage = 100
            scene.render.pixel_aspect_x = float(intr["pixel_aspect_x"])
            scene.render.pixel_aspect_y = float(intr["pixel_aspect_y"])
            out_path = guides_dir / f"guide_{cam['cam_key']}.png"
            scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
    finally:
        scene.camera = prev_cam
        bpy.data.objects.remove(temp_obj, do_unlink=True)
        bpy.data.cameras.remove(temp_data, do_unlink=True)
    print("MULTIVIEW_GUIDES_OK", len(cams))


def render_multiview_guides(
    spec_path: Path,
    blend_path: Path,
    out_dir: Path,
    angles_deg: list[float] | tuple[float, ...],
    *,
    width: int = 1248,
    height: int = 832,
) -> list[dict[str, Any]]:
    """Render target-viewpoint EEVEE layout guides from initial Blender blockout."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cams = compute_multiview_orbit_cameras(spec, list(angles_deg), width=width, height=height)
    mv_dir = out_dir / "multiview"
    guides_dir = mv_dir / "guides"
    guides_dir.mkdir(parents=True, exist_ok=True)
    cams_json = guides_dir / "orbit_cams.json"
    cams_json.write_text(json.dumps(cams, indent=2), encoding="utf-8")

    run_blender_script(
        Path(__file__).resolve(),
        ["--blender-guides", "--cameras-json", str(cams_json), "--guides-dir", str(guides_dir)],
        blend_path=blend_path,
    )
    for c in cams:
        c["guide_path"] = str((guides_dir / f"guide_{c['cam_key']}.png").resolve())
    return cams


def _edge_structure_map(rgb: np.ndarray, ksize: int = 11) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), (3, 3), 0.8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (ksize, ksize), ksize * 0.28)
    m = float(mag.mean())
    s = max(float(mag.std()), 1e-4)
    return ((mag - m) / s).astype(np.float32)


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - float(a.mean())
    b0 = b - float(b.mean())
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    return float(np.sum(a0 * b0) / max(denom, 1e-8))


def align_and_verify_novel_view(
    guide_rgb: np.ndarray,
    synth_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    *,
    min_corr: float = 0.35,
) -> dict[str, Any]:
    """Two-stage coarse translation + Sobel ECC affine alignment against Blender guide."""
    gh, gw = guide_rgb.shape[:2]
    synth_r = cv2.resize(synth_rgb, (gw, gh), interpolation=cv2.INTER_AREA) if synth_rgb.shape[:2] != (gh, gw) else synth_rgb.copy()

    g_edge = _edge_structure_map(guide_rgb, ksize=11)
    s_edge = _edge_structure_map(synth_r, ksize=11)
    g_broad = _edge_structure_map(guide_rgb, ksize=21)
    s_broad = _edge_structure_map(synth_r, ksize=21)
    before_edge = _norm_corr(g_edge, s_edge)
    before_broad = _norm_corr(g_broad, s_broad)
    base_score = max(before_edge, before_broad)

    aligned = synth_r
    warp_applied = False
    max_shift_px = 0.0
    after_edge = before_edge
    after_broad = before_broad
    max_allowed_shift = 0.115 * max(gw, gh)

    # Stage 1: Coarse translation via windowed FFT phase correlation + 0.5x translation ECC
    best_warp = np.eye(2, 3, dtype=np.float32)
    best_coarse_score = base_score

    def _eval_warp(w_mat: np.ndarray) -> tuple[float, float, float, np.ndarray | None]:
        det_w = float(w_mat[0, 0] * w_mat[1, 1] - w_mat[0, 1] * w_mat[1, 0])
        corners_w = np.array([[0.0, 0.0, 1.0], [gw, 0.0, 1.0], [gw, gh, 1.0], [0.0, gh, 1.0]], dtype=np.float32)
        shift_w = float(np.linalg.norm(corners_w @ w_mat.T - corners_w[:, :2], axis=1).max())
        if not (0.88 <= det_w <= 1.12 and shift_w <= max_allowed_shift):
            return -1.0, -1.0, shift_w, None
        cand = cv2.warpAffine(
            synth_r,
            w_mat,
            (gw, gh),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        ce = _norm_corr(g_edge, _edge_structure_map(cand, ksize=11))
        cb = _norm_corr(g_broad, _edge_structure_map(cand, ksize=21))
        return ce, cb, shift_w, cand

    try:
        hann = cv2.createHanningWindow((gw, gh), cv2.CV_32F)
        (dx_p, dy_p), _ = cv2.phaseCorrelate(g_broad * hann, s_broad * hann)
        for sx, sy in ((dx_p, dy_p), (-dx_p, -dy_p)):
            if float(np.hypot(sx, sy)) <= max_allowed_shift:
                w_pc = np.array([[1.0, 0.0, float(sx)], [0.0, 1.0, float(sy)]], dtype=np.float32)
                ce, cb, _, _ = _eval_warp(w_pc)
                if max(ce, cb) > best_coarse_score:
                    best_coarse_score = max(ce, cb)
                    best_warp = w_pc
    except cv2.error:
        pass

    g_norm = np.clip((g_broad + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)
    s_norm = np.clip((s_broad + 3.0) / 6.0, 0.0, 1.0).astype(np.float32)
    try:
        hw, hh = max(64, gw // 2), max(64, gh // 2)
        g_half = cv2.resize(g_norm, (hw, hh), interpolation=cv2.INTER_AREA)
        s_half = cv2.resize(s_norm, (hw, hh), interpolation=cv2.INTER_AREA)
        for init_w in (best_warp, np.eye(2, 3, dtype=np.float32)):
            w_t = init_w.copy()
            w_t[0, 2] *= 0.5
            w_t[1, 2] *= 0.5
            try:
                _, w_t = cv2.findTransformECC(
                    g_half,
                    s_half,
                    w_t,
                    cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-4),
                )
                w_full = np.array([[1.0, 0.0, float(w_t[0, 2] * 2.0)], [0.0, 1.0, float(w_t[1, 2] * 2.0)]], dtype=np.float32)
                ce, cb, _, _ = _eval_warp(w_full)
                if max(ce, cb) > best_coarse_score:
                    best_coarse_score = max(ce, cb)
                    best_warp = w_full
            except cv2.error:
                continue
    except cv2.error:
        pass

    # Stage 2: Full-res affine ECC refinement starting from best coarse translation (plus raw identity fallback)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-4)
    best_final = (before_edge, before_broad, 0.0, None)
    for start_w in (best_warp, np.eye(2, 3, dtype=np.float32)):
        for w_seed in (start_w,):
            ce0, cb0, sh0, cand0 = _eval_warp(w_seed)
            if cand0 is not None and max(ce0, cb0) > max(best_final[0], best_final[1]):
                best_final = (ce0, cb0, sh0, cand0)
        try:
            warp = start_w.copy()
            _, warp = cv2.findTransformECC(g_norm, s_norm, warp, cv2.MOTION_AFFINE, criteria)
            ce1, cb1, sh1, cand1 = _eval_warp(warp)
            if cand1 is not None and max(ce1, cb1) > max(best_final[0], best_final[1]):
                best_final = (ce1, cb1, sh1, cand1)
        except cv2.error:
            pass

    if best_final[3] is not None and max(best_final[0], best_final[1]) > base_score + 0.006:
        after_edge, after_broad, max_shift_px, aligned = best_final
        warp_applied = True

    final_score = max(after_edge, after_broad)
    accepted = bool(final_score >= min_corr)
    ref_mean = ref_rgb.astype(np.float32).mean(axis=(0, 1))
    al_mean = aligned.astype(np.float32).mean(axis=(0, 1))
    gain = np.clip(ref_mean / np.maximum(al_mean, 8.0), 0.86, 1.16)
    sharp = laplacian_sharpness(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))

    return {
        "aligned_rgb": aligned,
        "accepted": accepted,
        "warp_applied": warp_applied,
        "max_shift_px": round(max_shift_px, 2),
        "before_corr": round(max(before_edge, before_broad), 4),
        "after_corr": round(final_score, 4),
        "sharpness": round(sharp, 2),
        "gain_rgb": [round(float(x), 4) for x in gain],
    }


def _save_alignment_panel(guide_rgb: np.ndarray, aligned_rgb: np.ndarray, out_path: Path, header: str) -> None:
    gh, gw = guide_rgb.shape[:2]
    blend = cv2.addWeighted(guide_rgb, 0.45, aligned_rgb, 0.55, 0)
    strip = cv2.cvtColor(np.hstack([guide_rgb, blend, aligned_rgb]), cv2.COLOR_RGB2BGR)
    scale = min(1.0, 1680.0 / strip.shape[1])
    if scale < 1.0:
        strip = cv2.resize(strip, (int(strip.shape[1] * scale), int(strip.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.putText(strip, f"Guide ({header})", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(strip, "55% Align Overlay", (strip.shape[1] // 3 + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(strip, "Synthesized Aligned", (2 * strip.shape[1] // 3 + 14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 255, 160), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), strip)


def _minimax_orbit_prompt(angle_deg: float) -> str:
    side = "right" if angle_deg > 0 else "left"
    turn_desc = (
        "180 degrees around to the exact opposite rear view"
        if abs(abs(angle_deg) - 180.0) <= 5.0
        else f"{abs(angle_deg):g} degrees to the {side}"
    )
    return (
        f"Image 1 (<Picture 1>) is a 3D target-camera architectural blockout orbited {turn_desc}. "
        "Render a realistic full-frame drone photograph strictly locked to Image 1's exact camera viewpoint, "
        "perspective, rooflines, street curves, house spacing, and green meadow boundaries, while faithfully "
        "reproducing the authentic photographic stone/plaster walls, slate roof tiles, windows, garden fences, "
        "vehicles, daylight lighting, and crisp optical realism of the reference photo in Image 2 (<Picture 2>). "
        "Keep every newly visible side and rear facade architecturally coherent with crisp residential windows. "
        "Zero text, zero borders, zero watermarks."
    )


def _synth_one_gpt_image_2(
    ref_img: Image.Image,
    guide_img: Image.Image,
    angle_deg: float,
    *,
    model: str,
    quality: str,
) -> tuple[bytes, dict[str, Any]]:
    extra_instr = (
        "Lock strictly onto every building footprint, gable slope, road edge, and tree location in Image 1; "
        "paint realistic side and rear residential facades, doors, windows, roof tiles, and garden surfaces "
        "matching the architectural identity and sunny daylight optics of Image 2."
    )
    prev_size = os.environ.get("ORBIT_IMAGE_SIZE")
    size_candidates = (
        ["1536x1024", "auto"] if ref_img.width >= ref_img.height else ["1024x1536", "auto"]
    )
    last_err: Exception | None = None
    try:
        for sz in size_candidates:
            os.environ["ORBIT_IMAGE_SIZE"] = sz
            for attempt in range(2):
                try:
                    return image_api.generate_view(
                        ref_img,
                        guide_img,
                        angle_deg,
                        model=model,
                        guidance="layout",
                        quality=quality,
                        instruction=extra_instr,
                    )
                except Exception as exc:
                    last_err = exc
                    if attempt == 0:
                        time.sleep(1.0)
        assert last_err is not None
        raise last_err
    finally:
        if prev_size is None:
            os.environ.pop("ORBIT_IMAGE_SIZE", None)
        else:
            os.environ["ORBIT_IMAGE_SIZE"] = prev_size


def _synth_one_minimax_h3_image(
    ref_img: Image.Image,
    guide_img: Image.Image,
    angle_deg: float,
    *,
    comfy_url: str | None,
    cam_key: str,
) -> tuple[bytes, dict[str, Any]]:
    prompt = _minimax_orbit_prompt(angle_deg)
    url = _resolve_comfy_url(comfy_url)
    if url:
        return mm_img.call_comfyui_minimax_h3(
            guide_img,
            ref_img,
            prompt,
            comfy_url=url,
            steps=6,
            megapixels=1.5,
            task_id=f"mv_{cam_key}",
        )
    api_key = (os.environ.get("MINIMAX_API_KEY") or "").strip()
    if api_key:
        return mm_img.call_minimax_api(
            guide_img,
            ref_img,
            prompt,
            api_key=api_key,
            base_url=os.environ.get("MINIMAX_API_BASE_URL") or mm_img.DEFAULT_MINIMAX_BASE_URL,
            size="2k" if ref_img.width >= 1500 else "1k",
        )
    raise RuntimeError("未配置可用的 MiniMax-H3-Image（需要 Colab ComfyUI URL 或 MINIMAX_API_KEY）")


def _extract_calibrated_video_views(
    video_path: Path,
    ref_path: Path,
    ref_cam: dict[str, Any],
    target_cams: list[dict[str, Any]],
    out_dir: Path,
) -> dict[str, tuple[Image.Image, dict[str, Any]]]:
    """Sample sharp frames from a generated MiniMax-H3 orbit MP4 and match them against target orbit guides."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法读取 MiniMax-H3 视频: {video_path}")
    frames: list[tuple[int, float, np.ndarray]] = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % 8 == 0:
            sharp = laplacian_sharpness(bgr)
            frames.append((idx, sharp, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"视频未解码出任何帧: {video_path}")

    with Image.open(ref_path) as r_im:
        ref_rgb = np.asarray(r_im.convert("RGB"))

    matched: dict[str, tuple[Image.Image, dict[str, Any]]] = {}
    for tc in target_cams:
        with Image.open(tc["guide_path"]) as g_im:
            guide_rgb = np.asarray(g_im.convert("RGB"))
        g_edge = _edge_structure_map(guide_rgb, ksize=15)
        gh, gw = guide_rgb.shape[:2]
        best_frame = None
        best_score = -1.0
        for f_idx, sharp, f_rgb in frames:
            r_rgb = cv2.resize(f_rgb, (gw, gh), interpolation=cv2.INTER_AREA)
            sc = _norm_corr(g_edge, _edge_structure_map(r_rgb, ksize=15))
            if sc > best_score:
                best_score = sc
                best_frame = (f_idx, r_rgb)
        if best_frame is not None:
            f_idx, rgb_best = best_frame
            al = align_and_verify_novel_view(guide_rgb, rgb_best, ref_rgb)
            matched[tc["cam_key"]] = (
                Image.fromarray(al["aligned_rgb"]),
                {"provider": "minimax_h3_video", "frame_index": f_idx, "guide_edge_corr": round(best_score, 4)},
            )
    return matched


def _resolve_comfy_url(comfy_url: str | None = None) -> str:
    """Resolve explicit/env ComfyUI URL or auto-discover fixed local Colab gateway http://127.0.0.1:8188."""
    url = (comfy_url or os.environ.get("MINIMAX_H3_COMFY_URL") or os.environ.get("COMFYUI_URL") or "").strip()
    if url:
        return url
    if mm_img is not None and mm_img.probe_comfy_gateway(mm_img.DEFAULT_LOCAL_COMFY_GATEWAY) is not None:
        return mm_img.DEFAULT_LOCAL_COMFY_GATEWAY
    return ""


def resolve_multiview_provider(requested: str, *, comfy_url: str | None = None) -> str:
    """Resolve effective synthesis provider ('auto' prioritizes live Colab MiniMax-H3, then GPT Image 2)."""
    req = (requested or "auto").strip().lower()
    if req in ("gpt_image_2", "minimax_h3_image", "minimax_h3_video", "both"):
        return req
    if req != "auto":
        raise ValueError(f"Unsupported multiview provider: {requested}")

    url = _resolve_comfy_url(comfy_url)
    if url:
        ping = mm_img.ping_comfyui(url)
        if ping.get("ok") and ping.get("has_minimax_h3_image"):
            return "minimax_h3_image"
        gw_state = ping.get("gateway_state")
        if gw_state in (None, "ready"):
            vid = mm_orb.check_comfy_health(url)
            if vid.get("ok"):
                return "minimax_h3_video"
        else:
            print(f"[MultiView] Colab 网关暂未就绪 ({ping.get('message')}); auto 模式回退至下一可用引擎")
    if (os.environ.get("ORBIT_IMAGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("key") or "").strip():
        return "gpt_image_2"
    if (os.environ.get("MINIMAX_API_KEY") or "").strip():
        return "minimax_h3_image"
    raise RuntimeError("未检测到可用的多视角生成通道（既无连通的 Colab ComfyUI，也未配置 OpenAI/MiniMax API Key）")


def synthesize_novel_views(
    *,
    spec_path: Path,
    blend_path: Path,
    out_dir: Path,
    angles_deg: list[float] | tuple[float, ...] = (-60.0, 60.0, 180.0),
    provider: str = "auto",
    comfy_url: str | None = None,
    gateway_upstream: str | None = None,
    gpt_model: str = "gpt-image-2",
    gpt_quality: str = "high",
    force: bool = False,
) -> dict[str, Any]:
    """Render target orbit guides, synthesize novel aerial photographs, verify ECC alignment, and return cameras."""
    load_workspace_env()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    ref_path = Path(spec["ref_clean_image"])
    with Image.open(ref_path) as im_r:
        ref_img = im_r.convert("RGB")
        ref_rgb = np.asarray(ref_img)

    if gateway_upstream and str(gateway_upstream).strip():
        gw_target = _resolve_comfy_url(comfy_url) or mm_img.DEFAULT_LOCAL_COMFY_GATEWAY
        mm_img.set_comfy_gateway_upstream(gw_target, str(gateway_upstream).strip())
        comfy_url = gw_target

    effective_provider = resolve_multiview_provider(provider, comfy_url=comfy_url)
    guide_cams = render_multiview_guides(
        spec_path, blend_path, out_dir, list(angles_deg), width=1248, height=832
    )

    mv_root = out_dir / "multiview" / effective_provider
    mv_root.mkdir(parents=True, exist_ok=True)
    ref_cam = next((c for c in spec["cameras"] if c["source_clip"] == "ref"), spec["cameras"][0])

    # Pre-run video generator once if minimax_h3_video
    video_matches: dict[str, tuple[Image.Image, dict[str, Any]]] = {}
    video_err: str | None = None
    if effective_provider in ("minimax_h3_video", "both"):
        url_v = _resolve_comfy_url(comfy_url)
        if effective_provider == "minimax_h3_video" or (url_v and mm_orb.check_comfy_health(url_v).get("ok")):
            try:
                vid_out = out_dir / "multiview" / "h3_video"
                vid_out.mkdir(parents=True, exist_ok=True)
                orb_args = mm_orb.parse_args(
                    [
                        "--comfy-url", url_v,
                        "--first-frame", str(ref_path),
                        "--last-frame", str(ref_path),
                        "--mode", "loop360",
                        "--width", "1152",
                        "--height", "768",
                        "--length", "175",
                        "--no-clean-watermark",
                        "--out-dir", str(vid_out),
                    ]
                )
                man = mm_orb.run_orbit_generation(orb_args)
                video_matches = _extract_calibrated_video_views(
                    Path(man["videoPath"]), ref_path, ref_cam, guide_cams, out_dir
                )
            except Exception as exc:
                video_err = str(exc)
                if effective_provider == "minimax_h3_video":
                    raise

    accepted_cams: list[dict[str, Any]] = []
    view_reports: list[dict[str, Any]] = []

    for cam in guide_cams:
        key = cam["cam_key"]
        angle = float(cam["angle_deg"])
        guide_path = Path(cam["guide_path"])
        with Image.open(guide_path) as im_g:
            guide_img = im_g.convert("RGB")
            guide_rgb = np.asarray(guide_img)

        synth_raw_path = mv_root / f"{key}_synth.png"
        synth_aligned_path = mv_root / f"{key}_aligned.png"
        meta_gen: dict[str, Any] = {}

        if synth_raw_path.is_file() and not force:
            with Image.open(synth_raw_path) as cached_im:
                synth_rgb = np.asarray(cached_im.convert("RGB"))
            meta_gen = {"cached": True, "provider": effective_provider}
        else:
            candidates_gen: list[tuple[np.ndarray, dict[str, Any]]] = []
            errors_gen: list[str] = []

            if effective_provider in ("gpt_image_2", "both"):
                try:
                    raw_bytes, m_gpt = _synth_one_gpt_image_2(
                        ref_img, guide_img, angle, model=gpt_model, quality=gpt_quality
                    )
                    with Image.open(io.BytesIO(raw_bytes)) as g_out:
                        candidates_gen.append((np.asarray(g_out.convert("RGB")), m_gpt))
                except Exception as exc:
                    errors_gen.append(f"gpt_image_2: {exc}")

            if effective_provider in ("minimax_h3_image", "both"):
                try:
                    raw_bytes, m_mm = _synth_one_minimax_h3_image(
                        ref_img, guide_img, angle, comfy_url=comfy_url, cam_key=key
                    )
                    with Image.open(io.BytesIO(raw_bytes)) as m_out:
                        candidates_gen.append((np.asarray(m_out.convert("RGB")), m_mm))
                except Exception as exc:
                    errors_gen.append(f"minimax_h3_image: {exc}")

            if effective_provider in ("minimax_h3_video", "both") and key in video_matches:
                v_img, m_vid = video_matches[key]
                candidates_gen.append((np.asarray(v_img.convert("RGB")), m_vid))

            if not candidates_gen:
                raise RuntimeError(f"多视角图像生成失败 ({key}): {'; '.join(errors_gen or [video_err or 'no output'])}")

            # Pick the synthesized candidate with highest geometric edge alignment against Blender guide
            best_pick = None
            best_al = None
            for cand_rgb, cand_meta in candidates_gen:
                al_cand = align_and_verify_novel_view(guide_rgb, cand_rgb, ref_rgb)
                if best_al is None or al_cand["after_corr"] > best_al["after_corr"]:
                    best_al = al_cand
                    best_pick = (cand_rgb, cand_meta)
            assert best_pick is not None
            synth_rgb, meta_gen = best_pick
            Image.fromarray(synth_rgb).save(synth_raw_path, format="PNG")

        al = align_and_verify_novel_view(guide_rgb, synth_rgb, ref_rgb)
        Image.fromarray(al["aligned_rgb"]).save(synth_aligned_path, format="PNG")
        _save_alignment_panel(guide_rgb, al["aligned_rgb"], mv_root / f"{key}_align_compare.png", f"{key} ({angle:+.0f}°)")

        vrep = {
            "cam_key": key,
            "angle_deg": angle,
            "accepted": al["accepted"],
            "before_corr": al["before_corr"],
            "after_corr": al["after_corr"],
            "warp_applied": al["warp_applied"],
            "max_shift_px": al["max_shift_px"],
            "sharpness": al["sharpness"],
            "gain_rgb": al["gain_rgb"],
            "synth_path": str(synth_aligned_path.resolve()),
            "meta": meta_gen,
        }
        view_reports.append(vrep)
        print(
            f"  [MultiView {key}] corr={al['before_corr']:.3f}->{al['after_corr']:.3f} "
            f"shift={al['max_shift_px']}px accepted={al['accepted']} ({meta_gen.get('provider', effective_provider)})"
        )

        if al["accepted"]:
            cam_entry = dict(cam)
            cam_entry["image_path"] = str(synth_aligned_path.resolve())
            cam_entry["sharpness"] = al["sharpness"]
            cam_entry["gain_rgb"] = al["gain_rgb"]
            accepted_cams.append(cam_entry)

    report = {
        "requested_provider": provider,
        "effective_provider": effective_provider,
        "angles_deg": [float(a) for a in angles_deg],
        "accepted_count": len(accepted_cams),
        "views": view_reports,
    }
    report_path = mv_root / "multiview_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"provider_used": effective_provider, "cameras": accepted_cams, "report": report, "report_path": str(report_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--blender-guides", action="store_true")
    p.add_argument("--cameras-json", type=Path, default=None)
    p.add_argument("--guides-dir", type=Path, default=None)
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--blend", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--provider", default="auto", choices=("auto", "gpt_image_2", "minimax_h3_image", "minimax_h3_video", "both"))
    p.add_argument("--angles", type=float, nargs="+", default=[-60.0, 60.0, 180.0])
    p.add_argument("--comfy-url", default=None)
    p.add_argument("--gateway-upstream", default=None, help="Optional trycloudflare URL to hot-takeover on local gateway")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    if "--blender-guides" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
        args = parse_args(argv)
        _render_guides_in_blender(args.cameras_json.resolve(), args.guides_dir.resolve())
    else:
        args = parse_args()
        if not args.spec or not args.blend or not args.out_dir:
            raise SystemExit("--spec, --blend, and --out-dir are required")
        res = synthesize_novel_views(
            spec_path=args.spec.resolve(),
            blend_path=args.blend.resolve(),
            out_dir=args.out_dir.resolve(),
            angles_deg=args.angles,
            provider=args.provider,
            comfy_url=args.comfy_url,
            gateway_upstream=args.gateway_upstream,
            force=args.force,
        )
        print("SYNTHESIZE_MULTIVIEW_OK", json.dumps(res["report"], ensure_ascii=False))
