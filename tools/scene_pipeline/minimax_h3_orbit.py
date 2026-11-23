"""Generate smooth 360-degree or wide-arc aerial orbit videos from a reference photo via MiniMax-H3 ComfyUI."""

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from refinement import minimax_h3_image as mm_img  # noqa: E402


DEFAULT_MODELS = {
    "unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "turbo_lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
}


def snap_length(frames: int) -> int:
    """Snap frame count up to MiniMax-H3's 17*k + 5 grid (minimum 5, trained range 124..362)."""
    if frames <= 5:
        return 5
    k = (frames - 5 + 16) // 17
    return 17 * k + 5


def snap_dimension(value: int) -> int:
    """Round width/height to nearest multiple of 32 >= 32."""
    return max(32, int(round(value / 32) * 32))


def clean_watermark(image: Image.Image, box: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Inpaint a corner watermark (default: bottom-right 14% x 12%) without touching the rest of the frame."""
    rgb = np.asarray(image.convert("RGB")).copy()
    h, w = rgb.shape[:2]
    x0_f, y0_f, x1_f, y1_f = box if box is not None else (0.86, 0.88, 1.0, 1.0)
    x0 = int(np.clip(round(x0_f * w), 0, w - 1))
    y0 = int(np.clip(round(y0_f * h), 0, h - 1))
    x1 = int(np.clip(round(x1_f * w), x0 + 1, w))
    y1 = int(np.clip(round(y1_f * h), y0 + 1, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    repaired = cv2.inpaint(bgr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    out = rgb
    out[y0:y1, x0:x1] = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
    return Image.fromarray(out)


def default_orbit_prompt(mode: str, direction: str) -> str:
    """Build a photogrammetry-friendly static-scene drone orbit prompt."""
    turn = "clockwise" if direction == "cw" else "counter-clockwise"
    static_tail = (
        "Completely frozen static architectural scene: zero moving vehicles, zero pedestrians, "
        "zero swaying trees, zero smoke, constant daylight sun angle and exposure, fixed focal length "
        "wide drone lens, crisp sharp details across every roof and courtyard wall, no cuts, no dissolves, "
        "no text, no logos, no watermarks. Audio: quiet distant rural breeze, no music."
    )
    if mode == "loop360":
        return (
            f"Continuous smooth aerial drone {turn} 360-degree orbital shot circling steadily around the "
            "center of the rural village at constant altitude and constant radius, smoothly revealing all sides, "
            "back facades, courtyards, and surrounding fields before returning seamlessly to the exact starting camera. "
            + static_tail
        )
    if mode == "elevated":
        return (
            f"Smooth aerial drone {turn} elevated orbit rising gently while circling around the village center, "
            "keeping the street grid and roof geometry crisp and rigid with clear view into rear alleys and yards. "
            + static_tail
        )
    return (
        f"Smooth constant-altitude aerial drone {turn} sweeping arc around the village center, "
        "keeping building proportions, roof ridges, and road topology rigid and consistent. "
        + static_tail
    )


def build_workflow(
    *,
    first_image_name: str,
    last_image_name: str | None = None,
    mode: str = "loop360",
    direction: str = "cw",
    width: int = 1344,
    height: int = 768,
    length: int = 243,
    steps: int = 8,
    turbo: bool = True,
    seed: int = 42,
    prompt: str | None = None,
    filename_prefix: str = "video/village_h3_orbit",
    models: dict[str, str] | None = None,
) -> dict:
    """Construct a ComfyUI API prompt graph for MiniMax-H3 video generation."""
    cfg = {**DEFAULT_MODELS, **(models or {})}
    snapped_w = snap_dimension(width)
    snapped_h = snap_dimension(height)
    snapped_len = snap_length(length)
    text_prompt = prompt.strip() if prompt and prompt.strip() else default_orbit_prompt(mode, direction)

    graph: dict[str, dict] = {
        "101": {"class_type": "LoadImage", "inputs": {"image": first_image_name}},
        "119": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["video_vae"]}},
        "120": {"class_type": "VAELoader", "inputs": {"vae_name": cfg["audio_vae"]}},
        "127": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg["unet"], "weight_dtype": "default"},
        },
        "128": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": cfg["clip"], "type": "minimax", "device": "default"},
        },
    }

    model_ref: list = ["127", 0]
    if turbo:
        graph["134"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["127", 0],
                "lora_name": cfg["turbo_lora"],
                "strength_model": 1.0,
            },
        }
        model_ref = ["134", 0]

    h3_inputs: dict = {
        "clip": ["128", 0],
        "vae": ["119", 0],
        "prompt": text_prompt,
        "width": snapped_w,
        "height": snapped_h,
        "length": snapped_len,
        "first_frame": ["101", 0],
    }

    if last_image_name:
        graph["102"] = {"class_type": "LoadImage", "inputs": {"image": last_image_name}}
        h3_inputs["last_frame"] = ["102", 0]
    elif mode == "loop360":
        h3_inputs["last_frame"] = ["101", 0]

    graph["131"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": h3_inputs}
    graph["126"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model_ref, "conditioning": ["131", 0]},
    }
    graph["123"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}}
    graph["124"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": model_ref, "scheduler": "simple", "steps": int(steps), "denoise": 1.0},
    }
    graph["129"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}}
    graph["125"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["129", 0],
            "guider": ["126", 0],
            "sampler": ["123", 0],
            "sigmas": ["124", 0],
            "latent_image": ["131", 1],
        },
    }
    graph["122"] = {"class_type": "VAEDecode", "inputs": {"samples": ["125", 0], "vae": ["119", 0]}}
    graph["121"] = {
        "class_type": "VAEDecodeAudio",
        "inputs": {"samples": ["125", 0], "vae": ["120", 0]},
    }
    graph["130"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["122", 0], "audio": ["121", 0], "fps": 24, "bit_depth": 8},
    }
    graph["92"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["130", 0],
            "filename_prefix": filename_prefix,
            "format": "auto",
            "codec": "auto",
        },
    }
    return graph


def upload_image(
    session: requests.Session,
    base_url: str,
    image: Image.Image,
    filename: str,
) -> str:
    """Upload an in-memory PNG image to ComfyUI /upload/image and return its server filename."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    resp = session.post(
        f"{base_url.rstrip('/')}/upload/image",
        files={"image": (filename, buf.getvalue(), "image/png")},
        data={"overwrite": "true"},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    name = payload.get("name") or filename
    subfolder = payload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def submit_and_wait(
    session: requests.Session,
    base_url: str,
    graph: dict,
    timeout_sec: int = 900,
    poll_interval_sec: float = 5.0,
) -> tuple[str, dict]:
    """Submit a ComfyUI prompt graph and poll /history until completion."""
    base = base_url.rstrip("/")
    resp = session.post(f"{base}/prompt", json={"prompt": graph}, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"ComfyUI /prompt rejected HTTP {resp.status_code}: {resp.text[:500]}")
    prompt_id = resp.json()["prompt_id"]
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        time.sleep(poll_interval_sec)
        hist_resp = session.get(f"{base}/history/{prompt_id}", timeout=20)
        if not hist_resp.ok:
            continue
        hist = hist_resp.json()
        entry = hist.get(prompt_id)
        if not entry:
            continue
        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI generation failed: {json.dumps(status, ensure_ascii=False)}")
        if status.get("completed") is True or entry.get("outputs"):
            return prompt_id, entry
    raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout_sec}s")


def download_video_output(
    session: requests.Session,
    base_url: str,
    entry: dict,
    dest_path: Path,
) -> Path:
    """Download the generated video from ComfyUI SaveVideo output metadata."""
    outputs = entry.get("outputs") or {}
    node_out = outputs.get("92") or next(iter(outputs.values()), {})
    files = node_out.get("images") or node_out.get("videos") or node_out.get("gifs") or []
    if not files:
        raise RuntimeError(f"No video output found in ComfyUI history: {json.dumps(outputs)}")
    meta = files[0]
    query = urllib.parse.urlencode(
        {
            "filename": meta["filename"],
            "subfolder": meta.get("subfolder", ""),
            "type": meta.get("type", "output"),
        }
    )
    url = f"{base_url.rstrip('/')}/view?{query}"
    resp = session.get(url, timeout=180, stream=True)
    resp.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as handle:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return dest_path


def run_orbit_generation(args: argparse.Namespace) -> dict:
    """Prepare frame(s), build workflow, optionally execute remotely, and write manifest."""
    if not args.first_frame.is_file():
        raise FileNotFoundError(f"First frame not found: {args.first_frame}")
    if args.last_frame is not None and not args.last_frame.is_file():
        raise FileNotFoundError(f"Last frame not found: {args.last_frame}")

    with Image.open(args.first_frame) as im:
        first_img = im.convert("RGB")
    if args.clean_watermark_flag:
        first_img = clean_watermark(first_img, args.watermark_box)

    last_img = None
    if args.last_frame is not None:
        with Image.open(args.last_frame) as im:
            last_img = im.convert("RGB")
        if args.clean_watermark_flag:
            last_img = clean_watermark(last_img, args.watermark_box)

    steps = args.steps if args.steps is not None else (8 if not args.no_turbo else 20)
    stem = f"h3_{args.mode}_{args.direction}_seed{args.seed}"

    if args.dry_run:
        graph = build_workflow(
            first_image_name=args.first_frame.name,
            last_image_name=args.last_frame.name if args.last_frame else None,
            mode=args.mode,
            direction=args.direction,
            width=args.width,
            height=args.height,
            length=args.length,
            steps=steps,
            turbo=not args.no_turbo,
            seed=args.seed,
            prompt=args.prompt,
            filename_prefix=f"video/{stem}",
        )
        return {"dryRun": True, "workflow": graph}

    base_url = (args.comfy_url or os.environ.get("MINIMAX_H3_COMFY_URL") or os.environ.get("COMFYUI_URL") or "").strip()
    if not base_url and mm_img.probe_comfy_gateway(mm_img.DEFAULT_LOCAL_COMFY_GATEWAY) is not None:
        base_url = mm_img.DEFAULT_LOCAL_COMFY_GATEWAY
    if not base_url:
        raise ValueError("Missing ComfyUI URL: pass --comfy-url or set MINIMAX_H3_COMFY_URL")

    with requests.Session() as session:
        mm_img.wait_for_comfy_gateway_ready(base_url, wait_timeout_sec=420.0, session=session)
        first_remote = upload_image(session, base_url, first_img, f"{stem}_first.png")
        last_remote = (
            upload_image(session, base_url, last_img, f"{stem}_last.png")
            if last_img is not None
            else None
        )
        graph = build_workflow(
            first_image_name=first_remote,
            last_image_name=last_remote,
            mode=args.mode,
            direction=args.direction,
            width=args.width,
            height=args.height,
            length=args.length,
            steps=steps,
            turbo=not args.no_turbo,
            seed=args.seed,
            prompt=args.prompt,
            filename_prefix=f"video/{stem}",
        )
        prompt_id, entry = submit_and_wait(
            session,
            base_url,
            graph,
            timeout_sec=args.timeout,
            poll_interval_sec=args.poll_interval,
        )
        out_video = args.out_dir / f"{stem}.mp4"
        download_video_output(session, base_url, entry, out_video)

    manifest = {
        "promptId": prompt_id,
        "comfyUrl": base_url,
        "mode": args.mode,
        "direction": args.direction,
        "width": snap_dimension(args.width),
        "height": snap_dimension(args.height),
        "length": snap_length(args.length),
        "steps": steps,
        "turbo": not args.no_turbo,
        "seed": args.seed,
        "firstFrame": str(args.first_frame),
        "lastFrame": str(args.last_frame) if args.last_frame else None,
        "cleanWatermark": args.clean_watermark_flag,
        "videoPath": str(out_video),
    }
    manifest_path = args.out_dir / f"{stem}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    return manifest


def check_comfy_health(base_url: str, timeout_sec: float = 10.0) -> dict:
    """Verify live ComfyUI /system_stats (and /_gateway/status if present) and MiniMaxH3ImageToVideo node readiness."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return {"ok": False, "error": "Empty ComfyUI URL"}
    try:
        with requests.Session() as session:
            gw = mm_img.probe_comfy_gateway(base, session=session, timeout=2.5)
            if gw is not None and gw.get("state") != "ready":
                return {
                    "ok": False,
                    "comfyUrl": base,
                    "gatewayState": gw.get("state"),
                    "recoveryStage": gw.get("recovery_stage"),
                    "error": gw.get("last_error") or f"Gateway not ready (state={gw.get('state')})",
                }
            stats_r = session.get(f"{base}/system_stats", timeout=timeout_sec)
            stats_r.raise_for_status()
            stats = stats_r.json()
            info_r = session.get(f"{base}/object_info/MiniMaxH3ImageToVideo", timeout=timeout_sec)
            info_r.raise_for_status()
            info = info_r.json()
            has_h3_node = "MiniMaxH3ImageToVideo" in info
            devices = (stats.get("devices") or [])
            return {
                "ok": bool(has_h3_node),
                "comfyUrl": base,
                "comfyuiVersion": (stats.get("system") or {}).get("comfyui_version"),
                "hasMiniMaxH3Node": has_h3_node,
                "deviceCount": len(devices),
            }
    except Exception as exc:
        return {"ok": False, "comfyUrl": base, "error": str(exc)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", default=None, help="ComfyUI base URL (or MINIMAX_H3_COMFY_URL)")
    parser.add_argument("--first-frame", type=Path, default=None)
    parser.add_argument("--last-frame", type=Path, default=None)
    parser.add_argument("--mode", choices=("loop360", "arc", "elevated"), default="loop360")
    parser.add_argument("--direction", choices=("cw", "ccw"), default="cw")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--length", type=int, default=243)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-turbo", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("video/h3_orbits"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument(
        "--watermark-box",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Normalized [0..1] box to inpaint before upload (default bottom-right corner)",
    )
    wm = parser.add_mutually_exclusive_group()
    wm.add_argument("--clean-watermark", dest="clean_watermark_flag", action="store_true")
    wm.add_argument("--no-clean-watermark", dest="clean_watermark_flag", action="store_false")
    parser.set_defaults(clean_watermark_flag=True)
    args = parser.parse_args(argv)
    if not args.health_check and args.first_frame is None:
        parser.error("--first-frame is required unless --health-check is passed")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.health_check:
        url = (args.comfy_url or os.environ.get("MINIMAX_H3_COMFY_URL") or os.environ.get("COMFYUI_URL") or "").strip()
        if not url and mm_img.probe_comfy_gateway(mm_img.DEFAULT_LOCAL_COMFY_GATEWAY) is not None:
            url = mm_img.DEFAULT_LOCAL_COMFY_GATEWAY
        res = check_comfy_health(url)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if not res.get("ok"):
            raise SystemExit(1)
        return
    result = run_orbit_generation(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
