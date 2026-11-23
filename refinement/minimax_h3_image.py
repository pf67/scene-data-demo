"""Dual-engine MiniMax-H3-Image style redraw worker (MiniMax Cloud API + ComfyUI)."""

from __future__ import annotations

import base64
import binascii
import io
import json
import math
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from PIL import Image

DEFAULT_MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_LOCAL_COMFY_GATEWAY = "http://127.0.0.1:8188"
DEFAULT_MINIMAX_MODELS = ("MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed")
DEFAULT_COMFY_MODELS = {
    "unet": os.environ.get("MINIMAX_H3_IMAGE_UNET", "minimax_h3_image_pruned_int8_convrot.safetensors"),
    "clip": os.environ.get("MINIMAX_H3_IMAGE_CLIP", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "vae": os.environ.get("MINIMAX_H3_IMAGE_VAE", "minimax_h3_image_vae_fp16.safetensors"),
}

MAX_INPUT_PIXELS = 20_000_000
MAX_JSON_PAYLOAD_BYTES = 48 * 1024 * 1024

STYLE_PRESETS: dict[str, dict[str, str]] = {
    "faithful": {
        "label": "高保真还原原图",
        "description": "严格锁定 3D 截图几何与机位，忠实复现原图航拍色彩、建筑材质肌理与自然光照氛围。",
        "prompt": (
            "Generate a crisp photorealistic aerial photograph matching the exact camera angle, focal length, "
            "framing, building footprints, rooflines, street layout, and horizon of <Picture 1>, while strictly "
            "reproducing the authentic photographic style, tonal palette, natural sunlight, atmospheric haze, "
            "weathered masonry, clay roof tiles, asphalt patina, and camera optics of <Picture 2>."
        ),
    },
    "detailed": {
        "label": "强化写实细节",
        "description": "保留几何布局的同时显著增强红砖、灰泥风化、屋瓦层次、路面颗粒与植被枝叶真实感。",
        "prompt": (
            "Transform <Picture 1> into a richly detailed real-world drone photograph that preserves every building, "
            "roof ridge, courtyard wall, road, and camera perspective from <Picture 1>, rendered in the exact visual "
            "style and color grading of <Picture 2> with fine brick courses, weathered concrete plaster, individual "
            "terracotta roof tiles, crisp eaves shadows, textured asphalt roads, and natural multi-toned foliage."
        ),
    },
    "match_3d_light": {
        "label": "匹配当前3D光照",
        "description": "以 3D 截图的太阳方位与明暗情绪为准，迁移原图的真实材质质感与相机光学表现。",
        "prompt": (
            "Relight and render <Picture 1> into a genuine real-world aerial photograph preserving both the exact "
            "geometry, camera framing, and directional sun shadow mood of <Picture 1>, while applying the real-world "
            "building textures, color harmony, surface weathering, and photographic sensor look of <Picture 2>."
        ),
    },
}


def decode_image_input(value: str | None = None, file_path: str | Path | None = None) -> Image.Image:
    """Load an RGB PIL image from a base64/data-URI string or a local file path."""
    if value:
        raw = value.strip()
        encoded = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("输入图像 Base64 编码无效") from exc
        buf = io.BytesIO(data)
        with Image.open(buf) as img:
            if img.width * img.height > MAX_INPUT_PIXELS:
                raise ValueError("输入图像像素数超出 20MP 上限")
            img.load()
            return img.convert("RGB")
    if file_path is not None:
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"找不到参考图像文件: {path.name}")
        with Image.open(path) as img:
            if img.width * img.height > MAX_INPUT_PIXELS:
                raise ValueError("参考图像像素数超出 20MP 上限")
            img.load()
            return img.convert("RGB")
    raise ValueError("缺少图像数据")


def validate_http_url(raw: str, label: str = "服务地址") -> str:
    """Validate an http(s) URL without embedded credentials, query, or fragment."""
    cleaned = (raw or "").strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"请检查{label}格式（需以 http:// 或 https:// 开头，且不含账号密码或查询参数）")
    return cleaned


def normalize_minimax_base_url(raw: str | None) -> str:
    """Normalize a MiniMax API base URL and append /v1 when only an origin is supplied."""
    cleaned = validate_http_url(raw or DEFAULT_MINIMAX_BASE_URL, "MiniMax API 地址")
    parsed = urlsplit(cleaned)
    if not parsed.path or parsed.path == "/":
        return f"{cleaned}/v1"
    return cleaned


def snap_multiple_of_32(value: float, lo: int = 128, hi: int = 8192) -> int:
    """Round a pixel dimension to the nearest multiple of 32 within [lo, hi]."""
    snapped = int(round(value / 32.0) * 32)
    return max(lo, min(hi, snapped))


def compute_target_size(width: int, height: int, size_mode: str = "auto") -> str:
    """Compute the MiniMax API `size` parameter preserving input aspect ratio."""
    mode = (size_mode or "auto").strip()
    if mode.lower() == "auto":
        return "auto"
    ratio = width / max(1, height)
    if not (0.25 <= ratio <= 4.0):
        raise ValueError("输入画面宽高比超出 1:4 – 4:1 支持范围")

    tier_long_edge = {"1k": 1024, "2k": 2048, "4k": 4096}.get(mode.lower())
    if tier_long_edge is not None:
        if width >= height:
            out_w = snap_multiple_of_32(tier_long_edge)
            out_h = snap_multiple_of_32(out_w / ratio)
        else:
            out_h = snap_multiple_of_32(tier_long_edge)
            out_w = snap_multiple_of_32(out_h * ratio)
        return f"{out_w}x{out_h}"

    if mode.lower() == "match":
        out_w = snap_multiple_of_32(width)
        out_h = snap_multiple_of_32(height)
        return f"{out_w}x{out_h}"

    if "x" in mode.lower():
        parts = mode.lower().split("x", 1)
        try:
            w_val = snap_multiple_of_32(int(parts[0]))
            h_val = snap_multiple_of_32(int(parts[1]))
            return f"{w_val}x{h_val}"
        except ValueError as exc:
            raise ValueError(f"无效的输出尺寸参数: {size_mode}") from exc

    raise ValueError(f"不支持的输出尺寸模式: {size_mode}")


def compute_comfy_dimensions(width: int, height: int, megapixels: float = 4.0) -> tuple[int, int]:
    """Scale (width, height) proportionally to ~megapixels (1..16 MP), snapped to multiples of 32."""
    mp = max(1.0, min(16.0, float(megapixels)))
    target_pixels = mp * 1_000_000.0
    ratio = width / max(1, height)
    if not (0.25 <= ratio <= 4.0):
        raise ValueError("输入画面宽高比超出 1:4 – 4:1 支持范围")
    out_w = snap_multiple_of_32(math.sqrt(target_pixels * ratio), lo=256, hi=4096)
    out_h = snap_multiple_of_32(out_w / ratio, lo=256, hi=4096)
    return out_w, out_h


def _prepare_image_bounds(image: Image.Image, max_edge: int) -> Image.Image:
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w < 128 or h < 128:
        scale = max(128 / max(1, w), 128 / max(1, h))
        rgb = rgb.resize((max(128, round(w * scale)), max(128, round(h * scale))), Image.Resampling.BICUBIC)
        w, h = rgb.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        rgb = rgb.resize((max(128, round(w * scale)), max(128, round(h * scale))), Image.Resampling.LANCZOS)
    return rgb


def encode_data_uri(image: Image.Image, max_edge: int = 2048, prefer_jpeg: bool = False) -> str:
    """Encode a PIL RGB image as a data URI suitable for MiniMax-H3-Image JSON payloads."""
    prepared = _prepare_image_bounds(image, max_edge=max_edge)
    if not prefer_jpeg:
        buf = io.BytesIO()
        prepared.save(buf, format="PNG")
        raw = buf.getvalue()
        if len(raw) <= 10 * 1024 * 1024:
            return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    buf = io.BytesIO()
    prepared.save(buf, format="JPEG", quality=94)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_style_redraw_prompt(
    style_preset: str = "faithful",
    custom_prompt: str = "",
    protect_equipment: bool = False,
) -> str:
    """Construct the dual-reference prompt anchoring <Picture 1> layout and <Picture 2> style."""
    preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["faithful"])
    parts = [preset["prompt"]]
    if protect_equipment:
        parts.append(
            "Keep the cleared road segments and courtyard spaces in <Picture 1> completely unobstructed without "
            "inventing extra parked vehicles or obstacles, while preserving consistent ground contact shadows "
            "and realistic rural village textures."
        )
    parts.append("Zero text overlays, zero watermarks, zero borders. Output one coherent full-frame photograph.")
    extra = (custom_prompt or "").strip()
    if extra:
        parts.append(f"Additional user instruction: {extra}")
    return " ".join(parts)


def _extract_minimax_error(resp: requests.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"
    if isinstance(payload, dict):
        base_resp = payload.get("base_resp")
        if isinstance(base_resp, dict) and base_resp.get("status_msg"):
            code = base_resp.get("status_code")
            return f"MiniMax API 错误 ({code}): {base_resp['status_msg']}"
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return f"MiniMax API 错误: {err['message']}"
        if payload.get("message"):
            return f"MiniMax API 错误 ({resp.status_code}): {payload['message']}"
    return f"MiniMax API 返回 HTTP {resp.status_code}"


def call_minimax_api(
    screenshot: Image.Image,
    reference: Image.Image,
    prompt: str,
    *,
    api_key: str,
    base_url: str = DEFAULT_MINIMAX_BASE_URL,
    model: str = "MiniMax-H3-Image",
    size: str = "auto",
    session: requests.Session | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Call MiniMax /v1/images/edits with Picture 1 (3D render) and Picture 2 (reference photo)."""
    token = (api_key or "").strip()
    if not token:
        raise ValueError("请先配置 MINIMAX_API_KEY")
    normalized_base = normalize_minimax_base_url(base_url)
    selected_model = (model or DEFAULT_MINIMAX_MODELS[0]).strip()
    resolved_size = compute_target_size(screenshot.width, screenshot.height, size)

    def make_body(prefer_jpeg: bool, max_edge: int) -> dict[str, Any]:
        return {
            "model": selected_model,
            "prompt": prompt,
            "images": [
                {"image_url": encode_data_uri(screenshot, max_edge=max_edge, prefer_jpeg=prefer_jpeg)},
                {"image_url": encode_data_uri(reference, max_edge=max_edge, prefer_jpeg=prefer_jpeg)},
            ],
            "size": resolved_size,
            "response_format": "b64_json",
        }

    payload = make_body(prefer_jpeg=False, max_edge=2048)
    raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw_bytes) > MAX_JSON_PAYLOAD_BYTES:
        payload = make_body(prefer_jpeg=True, max_edge=2048)
        raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if len(raw_bytes) > MAX_JSON_PAYLOAD_BYTES:
        payload = make_body(prefer_jpeg=True, max_edge=1536)

    client = session or requests.Session()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        resp = client.post(
            f"{normalized_base}/images/edits",
            json=payload,
            headers=headers,
            timeout=(20, 600),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"连接 MiniMax API 失败（{type(exc).__name__}）: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(_extract_minimax_error(resp))

    try:
        body = resp.json()
    except ValueError as exc:
        raise RuntimeError("MiniMax API 返回的不是有效 JSON") from exc

    base_resp = body.get("base_resp") if isinstance(body, dict) else None
    if isinstance(base_resp, dict) and base_resp.get("status_code") not in (None, 0):
        raise RuntimeError(_extract_minimax_error(resp))

    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list) or not items:
        raise RuntimeError("MiniMax API 未返回生成图像数据")

    first = items[0]
    if not isinstance(first, dict):
        raise RuntimeError("MiniMax API 返回数据格式异常")
    if first.get("error"):
        raise RuntimeError(f"MiniMax 图像生成失败: {first['error']}")

    image_bytes: bytes | None = None
    b64_value = first.get("b64_json")
    if isinstance(b64_value, str) and b64_value.strip():
        cleaned = b64_value.split(",", 1)[1] if b64_value.startswith("data:") and "," in b64_value else b64_value
        image_bytes = base64.b64decode(cleaned.strip(), validate=True)
    elif isinstance(first.get("url"), str) and first["url"].strip():
        image_url = validate_http_url_download(first["url"].strip())
        dl = client.get(image_url, timeout=(15, 120), allow_redirects=True)
        dl.raise_for_status()
        image_bytes = dl.content

    if not image_bytes:
        raise RuntimeError("MiniMax API 未返回 b64_json 或可下载的图像 URL")

    return image_bytes, {
        "provider": "minimax_api",
        "model": selected_model,
        "requested_size": resolved_size,
        "output_size_reported": first.get("size") or resolved_size,
        "revised_prompt": first.get("revised_prompt") or "",
        "request_id": body.get("request_id") or "",
    }


def validate_http_url_download(raw: str) -> str:
    """Allow http/https URLs with query tokens when downloading signed CDN outputs."""
    parsed = urlsplit(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise RuntimeError("MiniMax API 返回了无效的图像下载地址")
    return raw


def build_comfyui_minimax_h3_graph(
    image1_name: str,
    image2_name: str,
    prompt: str,
    *,
    width: int,
    height: int,
    ref_width: int,
    ref_height: int,
    steps: int = 6,
    cfg: float = 1.0,
    seed: int | None = None,
    filename_prefix: str = "minimax_h3_redraw",
    models: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the ComfyUI API prompt graph matching Comfy-Org image_minimax_h3_image_edit.json."""
    cfg_models = {**DEFAULT_COMFY_MODELS, **(models or {})}
    chosen_seed = secrets.randbelow(2**48) if seed is None else int(seed)
    return {
        "29": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": cfg_models["unet"], "weight_dtype": "default"},
        },
        "30": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": cfg_models["clip"], "type": "minimax", "device": "default"},
        },
        "31": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": cfg_models["vae"]},
        },
        "45": {
            "class_type": "LoadImage",
            "inputs": {"image": image1_name},
        },
        "46": {
            "class_type": "LoadImage",
            "inputs": {"image": image2_name},
        },
        "48": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["45", 0],
                "upscale_method": "lanczos",
                "width": int(width),
                "height": int(height),
                "crop": "disabled",
            },
        },
        "49": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["46", 0],
                "upscale_method": "lanczos",
                "width": int(ref_width),
                "height": int(ref_height),
                "crop": "disabled",
            },
        },
        "32": {
            "class_type": "MiniMaxH3ImageConditioning",
            "inputs": {
                "clip": ["30", 0],
                "vae": ["31", 0],
                "images.image_1": ["48", 0],
                "images.image_2": ["49", 0],
                "prompt": prompt,
                "width": int(width),
                "height": int(height),
            },
        },
        "33": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["32", 0]},
        },
        "38": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["29", 0],
                "positive": ["32", 0],
                "negative": ["33", 0],
                "latent_image": ["32", 1],
                "seed": chosen_seed,
                "steps": max(4, min(30, int(steps))),
                "cfg": float(cfg),
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "42": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["38", 0], "vae": ["31", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["42", 0], "filename_prefix": filename_prefix},
        },
    }


def _get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 15,
    attempts: int = 3,
) -> requests.Response:
    """Retry idempotent GET calls against transient TLS/tunnel glitches."""
    last_exc: Exception | None = None
    for idx in range(max(1, attempts)):
        try:
            return session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            if idx + 1 < attempts:
                time.sleep(0.5 * (idx + 1))
    assert last_exc is not None
    raise last_exc


def _format_gateway_status_msg(gw: dict[str, Any]) -> str:
    gw_state = str(gw.get("state") or "unknown")
    stage = gw.get("recovery_stage") or ""
    last_err = str(gw.get("last_error") or "").strip()
    if gw_state == "reconnecting":
        return f"Colab 网关冷启动/重连中（state={gw_state}, stage={stage or 'boot'}），生图将自动等待就绪，也可热接管已有隧道"
    if gw_state == "error":
        first_line = last_err.splitlines()[0].strip() if last_err else ""
        if len(first_line) > 84:
            first_line = first_line[:84].rstrip() + "..."
        reason = first_line or "tier 启动失败"
        return f"Colab 网关恢复异常（{reason}）；可在下方热接管已有隧道"
    if gw_state in ("stopped", "idle", "disconnected"):
        return f"Colab 网关待机中（state={gw_state}），首次生图将自动按需唤醒 A100 实例（不用时执行 .\\stop_colab.ps1 停止计费）"
    return f"Colab 网关状态：{gw_state}" + (f" ({stage})" if stage else "")


def probe_comfy_gateway(
    comfy_url: str,
    *,
    session: requests.Session | None = None,
    timeout: float = 2.5,
) -> dict[str, Any] | None:
    """Return parsed /_gateway/status payload when talking to the local Colab ComfyGateway, else None."""
    base = (comfy_url or "").strip().rstrip("/")
    if not base:
        return None
    client = session or requests.Session()
    try:
        resp = client.get(f"{base}/_gateway/status", timeout=timeout)
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) and "state" in data else None


def set_comfy_gateway_upstream(
    gateway_url: str,
    upstream_url: str,
    *,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Hot-takeover a running Colab ComfyUI tunnel on POST /_gateway/set-upstream."""
    gw = validate_http_url(gateway_url, "ComfyUI 网关地址")
    upstream = validate_http_url(upstream_url, "上游隧道地址")
    client = session or requests.Session()
    endpoint = f"{gw}/_gateway/set-upstream"
    last_resp: requests.Response | None = None
    for payload in ({"url": upstream}, {"upstream_url": upstream}):
        try:
            resp = client.post(endpoint, json=payload, timeout=15)
        except requests.RequestException as exc:
            raise RuntimeError(f"连接网关热接管接口失败: {exc}") from exc
        last_resp = resp
        if resp.ok:
            try:
                body = resp.json()
                return body if isinstance(body, dict) else {"ok": True, "upstream_url": upstream}
            except ValueError:
                return {"ok": True, "upstream_url": upstream, "raw": resp.text[:200]}
        if resp.status_code not in (400, 422):
            break
    status_code = last_resp.status_code if last_resp is not None else "unknown"
    detail = last_resp.text[:300] if last_resp is not None else ""
    raise RuntimeError(f"网关热接管失败 HTTP {status_code}: {detail}")


def wait_for_comfy_gateway_ready(
    comfy_url: str,
    *,
    wait_timeout_sec: float = 420.0,
    poll_interval_sec: float = 4.0,
    trigger_wake: bool = True,
    session: requests.Session | None = None,
) -> dict[str, Any] | None:
    """If comfy_url is the local Colab gateway, wake/wait until state=='ready' or raise cleanly on terminal error."""
    base = validate_http_url(comfy_url, "ComfyUI 服务地址")
    client = session or requests.Session()
    gw = probe_comfy_gateway(base, session=client)
    if gw is None:
        return None
    if gw.get("state") == "ready":
        return gw

    if trigger_wake and gw.get("state") in ("stopped", "idle", "disconnected"):
        def _wake_ping() -> None:
            try:
                requests.get(f"{base}/system_stats", timeout=15)
            except requests.RequestException:
                pass
        threading.Thread(target=_wake_ping, daemon=True).start()

    deadline = time.monotonic() + max(5.0, float(wait_timeout_sec))
    last_log = ""
    consec_err = 0
    while time.monotonic() < deadline:
        gw = probe_comfy_gateway(base, session=client) or gw
        state = str(gw.get("state") or "")
        stage = str(gw.get("recovery_stage") or "")
        if state == "ready":
            return gw
        msg = f"{state}:{stage}"
        if msg != last_log:
            print(f"[ComfyGateway] {base} -> state={state} stage={stage}", flush=True)
            last_log = msg
        cooldown = float(gw.get("cooldown_remaining_sec") or 0.0)
        active = int(gw.get("active_requests") or 0)
        if state == "error" and cooldown <= 0.0 and active == 0:
            consec_err += 1
            if consec_err >= 2:
                err_txt = str(gw.get("last_error") or "gateway entered error state")
                raise RuntimeError(f"Colab 网关就绪失败（{err_txt}）；可通过 POST {base}/_gateway/set-upstream 热接管已有隧道")
        else:
            consec_err = 0
        time.sleep(poll_interval_sec)

    raise TimeoutError(f"等待 Colab 网关 {base} 就绪超时（>{wait_timeout_sec:.0f}s，当前状态={gw.get('state')}/{gw.get('recovery_stage')}）")


def ping_comfyui(comfy_url: str, *, session: requests.Session | None = None) -> dict[str, Any]:
    """Probe a ComfyUI server (and /_gateway/status if present) without blocking on cold boot."""
    url = validate_http_url(comfy_url, "ComfyUI 服务地址")
    client = session or requests.Session()
    gw = probe_comfy_gateway(url, session=client)
    if gw is not None and gw.get("state") != "ready":
        return {
            "ok": False,
            "url": url,
            "comfyui_version": None,
            "has_minimax_h3_image": False,
            "has_minimax_h3_video": False,
            "gateway": gw,
            "gateway_state": gw.get("state"),
            "recovery_stage": gw.get("recovery_stage"),
            "upstream_url": gw.get("upstream_url"),
            "message": _format_gateway_status_msg(gw),
        }

    try:
        stats_resp = _get_with_retry(client, f"{url}/system_stats", timeout=10, attempts=2)
    except requests.RequestException as exc:
        return {
            "ok": False,
            "url": url,
            "comfyui_version": None,
            "has_minimax_h3_image": False,
            "has_minimax_h3_video": False,
            "gateway": gw,
            "gateway_state": gw.get("state") if gw else None,
            "recovery_stage": gw.get("recovery_stage") if gw else None,
            "upstream_url": gw.get("upstream_url") if gw else None,
            "message": f"无法连接 ComfyUI：{exc}",
        }
    if not stats_resp.ok:
        return {
            "ok": False,
            "url": url,
            "comfyui_version": None,
            "has_minimax_h3_image": False,
            "has_minimax_h3_video": False,
            "gateway": gw,
            "gateway_state": gw.get("state") if gw else None,
            "recovery_stage": gw.get("recovery_stage") if gw else None,
            "upstream_url": gw.get("upstream_url") if gw else None,
            "message": f"ComfyUI 返回 HTTP {stats_resp.status_code}",
        }
    stats = stats_resp.json() if stats_resp.text else {}
    version = (stats.get("system") or {}).get("comfyui_version")
    try:
        node_resp = _get_with_retry(client, f"{url}/object_info/MiniMaxH3ImageConditioning", timeout=10, attempts=2)
        has_node = node_resp.ok and "MiniMaxH3ImageConditioning" in node_resp.text
    except requests.RequestException:
        has_node = False
    try:
        vid_resp = _get_with_retry(client, f"{url}/object_info/MiniMaxH3ImageToVideo", timeout=10, attempts=1)
        has_video = vid_resp.ok and "MiniMaxH3ImageToVideo" in vid_resp.text
    except requests.RequestException:
        has_video = False
    gw_tag = f" [网关 ready -> {gw.get('upstream_url')}]" if gw and gw.get("upstream_url") else (" [网关 ready]" if gw else "")
    return {
        "ok": True,
        "url": url,
        "comfyui_version": version,
        "has_minimax_h3_image": has_node,
        "has_minimax_h3_video": has_video,
        "gateway": gw,
        "gateway_state": gw.get("state") if gw else None,
        "recovery_stage": gw.get("recovery_stage") if gw else None,
        "upstream_url": gw.get("upstream_url") if gw else None,
        "message": (
            f"ComfyUI 已连通（{version or 'ready'}{gw_tag}），已检测到 MiniMaxH3ImageConditioning 节点"
            if has_node
            else f"ComfyUI 已连通（{version or 'ready'}{gw_tag}），但未检测到 MiniMaxH3ImageConditioning 节点"
        ),
    }


def _upload_comfy_image(session: requests.Session, base_url: str, image: Image.Image, filename: str) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    resp = session.post(
        f"{base_url}/upload/image",
        files={"image": (filename, buf.getvalue(), "image/png")},
        data={"overwrite": "true", "type": "input"},
        timeout=120,
    )
    resp.raise_for_status()
    payload = resp.json()
    name = payload.get("name") or filename
    subfolder = payload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def call_comfyui_minimax_h3(
    screenshot: Image.Image,
    reference: Image.Image,
    prompt: str,
    *,
    comfy_url: str,
    steps: int = 6,
    megapixels: float = 4.0,
    seed: int | None = None,
    task_id: str = "redraw",
    directory: Path | None = None,
    session: requests.Session | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Execute MiniMax-H3-Image dual-image edit on a ComfyUI server (with gateway wait support)."""
    base = validate_http_url(comfy_url, "ComfyUI 服务地址")
    client = session or requests.Session()
    wait_for_comfy_gateway_ready(base, wait_timeout_sec=420.0, session=client)
    try:
        info_resp = _get_with_retry(client, f"{base}/object_info/MiniMaxH3ImageConditioning", timeout=60, attempts=3)
    except requests.RequestException as exc:
        raise RuntimeError(f"无法连接 ComfyUI 服务 {base}: {exc}") from exc

    if not info_resp.ok or "MiniMaxH3ImageConditioning" not in info_resp.text:
        raise RuntimeError(
            "目标 ComfyUI 实例未检测到 MiniMaxH3ImageConditioning 节点，"
            "请升级 ComfyUI 至最新版并放置 MiniMax-H3-Image 模型权重"
        )

    out_w, out_h = compute_comfy_dimensions(screenshot.width, screenshot.height, megapixels)
    ref_w, ref_h = compute_comfy_dimensions(reference.width, reference.height, megapixels)
    img1_name = _upload_comfy_image(client, base, screenshot, f"{task_id}_picture1.png")
    img2_name = _upload_comfy_image(client, base, reference, f"{task_id}_picture2.png")

    graph = build_comfyui_minimax_h3_graph(
        img1_name,
        img2_name,
        prompt,
        width=out_w,
        height=out_h,
        ref_width=ref_w,
        ref_height=ref_h,
        steps=steps,
        seed=seed,
        filename_prefix=task_id,
    )
    submit_resp = client.post(f"{base}/prompt", json={"prompt": graph}, timeout=120)
    if not submit_resp.ok:
        raise RuntimeError(f"ComfyUI 拒绝提交工作流 HTTP {submit_resp.status_code}: {submit_resp.text[:300]}")
    prompt_id = submit_resp.json()["prompt_id"]
    if directory is not None:
        (directory / "comfy-task.json").write_text(json.dumps({"prompt_id": prompt_id}), encoding="utf-8")

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        try:
            hist_resp = _get_with_retry(client, f"{base}/history/{prompt_id}", timeout=30, attempts=2)
        except requests.RequestException:
            time.sleep(2.0)
            continue
        if hist_resp.ok:
            entry = hist_resp.json().get(prompt_id) or {}
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI 工作流执行失败: {json.dumps(status, ensure_ascii=False)}")
            images = (entry.get("outputs") or {}).get("9", {}).get("images") or []
            if images:
                view_resp = _get_with_retry(client, f"{base}/view", params=images[0], timeout=120, attempts=3)
                view_resp.raise_for_status()
                return view_resp.content, {
                    "provider": "comfyui",
                    "model": DEFAULT_COMFY_MODELS["unet"],
                    "requested_size": f"{out_w}x{out_h}",
                    "output_size_reported": f"{out_w}x{out_h}",
                    "prompt_id": prompt_id,
                    "steps": int(steps),
                }
        time.sleep(2.0)

    raise TimeoutError(f"ComfyUI 任务 {prompt_id} 等待超时（>30 分钟）")


def composite_equipment_protection(
    source: Image.Image,
    mask_img: Image.Image,
    repainted_bytes: bytes,
    directory: Path,
) -> tuple[Image.Image, float]:
    """Keep exact 3D equipment pixels intact over the repainted background."""
    with Image.open(io.BytesIO(repainted_bytes)) as raw:
        edited = raw.convert("RGB")
    (directory / "result-model.png").write_bytes(repainted_bytes)

    src_ratio = source.width / max(1, source.height)
    edt_ratio = edited.width / max(1, edited.height)
    if abs(edt_ratio / src_ratio - 1.0) > 0.08:
        raise ValueError(
            f"模型输出宽高比偏移过大（原图 {source.width}x{source.height}，重绘 {edited.width}x{edited.height}），已拒绝直接合成"
        )

    if edited.size != source.size:
        edited = edited.resize(source.size, Image.Resampling.LANCZOS)

    mask_l = mask_img.convert("L")
    if mask_l.size != source.size:
        mask_l = mask_l.resize(source.size, Image.Resampling.BILINEAR)

    alpha = mask_l.point(lambda v: 255 if v >= 200 else (0 if v < 12 else min(255, int(v * 255 / 200))))
    alpha.save(directory / "protection.png", format="PNG")

    total_px = max(1, source.width * source.height)
    mask_data = mask_l.get_flattened_data() if hasattr(mask_l, "get_flattened_data") else mask_l.getdata()
    covered_px = sum(1 for v in mask_data if v >= 128)
    coverage_pct = round(100.0 * covered_px / total_px, 3)

    final_img = Image.composite(source.convert("RGB"), edited, alpha)
    final_img.save(directory / "result.png", format="PNG")
    return final_img, coverage_pct


def run_task(directory: Path) -> dict[str, Any]:
    """Execute one redraw task directory (reads input.json, writes PNGs + result.meta.json)."""
    params = json.loads((directory / "input.json").read_text("utf-8"))
    source_img = decode_image_input(value=params.get("source_image"))
    reference_img = decode_image_input(
        value=params.get("reference_image"),
        file_path=params.get("reference_image_path"),
    )
    env_img = decode_image_input(value=params["environment_image"]) if params.get("environment_image") else None
    mask_img = decode_image_input(value=params["protection_mask"]) if params.get("protection_mask") else None

    source_img.save(directory / "source.png", format="PNG")
    reference_img.save(directory / "reference.png", format="PNG")

    want_protection = bool(params.get("protect_equipment", True))
    has_visible_equipment = False
    if want_protection and mask_img is not None:
        mask_binary = mask_img.convert("L").point(lambda v: 255 if v >= 32 else 0)
        has_visible_equipment = mask_binary.getbbox() is not None

    model_input = env_img if (has_visible_equipment and env_img is not None) else source_img
    if has_visible_equipment and env_img is not None:
        env_img.save(directory / "environment.png", format="PNG")

    style_preset = str(params.get("style_preset") or "faithful")
    custom_prompt = str(params.get("prompt") or "")
    effective_prompt = build_style_redraw_prompt(
        style_preset=style_preset,
        custom_prompt=custom_prompt,
        protect_equipment=bool(has_visible_equipment and env_img is not None),
    )

    provider = str(params.get("provider") or "minimax_api").strip()
    t0 = time.monotonic()
    if provider == "minimax_api":
        api_key = os.environ.get("MINIMAX_RUNTIME_API_KEY") or os.environ.get("MINIMAX_API_KEY") or ""
        base_url = params.get("api_base_url") or os.environ.get("MINIMAX_API_BASE_URL") or DEFAULT_MINIMAX_BASE_URL
        model = params.get("model") or os.environ.get("MINIMAX_IMAGE_MODEL") or DEFAULT_MINIMAX_MODELS[0]
        size = str(params.get("size") or "auto")
        repainted_bytes, engine_meta = call_minimax_api(
            model_input,
            reference_img,
            effective_prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            size=size,
        )
    elif provider == "comfyui":
        comfy_url = (
            params.get("comfyui_url")
            or os.environ.get("MINIMAX_H3_COMFY_URL")
            or os.environ.get("COMFYUI_URL")
            or ""
        ).strip()
        if not comfy_url and probe_comfy_gateway(DEFAULT_LOCAL_COMFY_GATEWAY) is not None:
            comfy_url = DEFAULT_LOCAL_COMFY_GATEWAY
        steps = int(params.get("steps") or 6)
        megapixels = float(params.get("megapixels") or 4.0)
        seed = int(params["seed"]) if params.get("seed") not in (None, "") else None
        repainted_bytes, engine_meta = call_comfyui_minimax_h3(
            model_input,
            reference_img,
            effective_prompt,
            comfy_url=comfy_url,
            steps=steps,
            megapixels=megapixels,
            seed=seed,
            task_id=directory.name,
            directory=directory,
        )
    else:
        raise ValueError(f"不支持的渲染引擎: {provider}")

    elapsed_sec = round(time.monotonic() - t0, 2)

    if has_visible_equipment and mask_img is not None:
        final_img, coverage_pct = composite_equipment_protection(
            source_img, mask_img, repainted_bytes, directory
        )
        out_width, out_height = final_img.size
    else:
        with Image.open(io.BytesIO(repainted_bytes)) as raw:
            out_rgb = raw.convert("RGB")
            out_width, out_height = out_rgb.size
            out_rgb.save(directory / "result-model.png", format="PNG")
            out_rgb.save(directory / "result.png", format="PNG")
        coverage_pct = 0.0

    meta = {
        **engine_meta,
        "task_id": directory.name,
        "style_preset": style_preset,
        "effective_prompt": effective_prompt,
        "equipment_protected": bool(has_visible_equipment),
        "protection_coverage_pct": coverage_pct,
        "source_size": [source_img.width, source_img.height],
        "reference_size": [reference_img.width, reference_img.height],
        "output_size": [out_width, out_height],
        "elapsed_sec": elapsed_sec,
        "scene_code": params.get("scene_code") or "",
        "scene_label": params.get("scene_label") or "",
    }
    (directory / "result.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) >= 2 and args[0] == "run":
        target = Path(args[1]).resolve()
        run_task(target)
        return 0
    if len(args) >= 2 and args[0] == "ping":
        print(json.dumps(ping_comfyui(args[1]), ensure_ascii=False))
        return 0
    if len(args) >= 3 and args[0] == "set-upstream":
        print(json.dumps(set_comfy_gateway_upstream(args[1], args[2]), ensure_ascii=False))
        return 0
    print(
        "Usage: python minimax_h3_image.py (run <task_dir> | ping <comfy_url> | set-upstream <gateway_url> <upstream_url>)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
