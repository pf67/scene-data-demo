"""Generate an architectural novel view using an OpenAI-compatible Images Edits API."""
import base64
import binascii
import io
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests

DEFAULTS = json.loads((Path(__file__).resolve().parents[1] / "builder/orbit-defaults.json").read_text("utf-8"))["imageApi"]


def orbit_prompt(angle, guidance, instruction=""):
    side = "right" if angle > 0 else "left"
    move = (f"Orbit the camera {abs(angle):g} degrees to the original photographer's {side} "
            "around the centre of the same group of buildings. Keep altitude, distance, lens and lighting consistent.")
    if guidance == "layout":
        inputs = ("Image 1 is the target-camera rendering. Make a photorealistic photograph in EXACTLY its camera, "
                  "framing and perspective. Keep all roof corners, road edges, gaps and occlusion boundaries at "
                  "their Image 1 positions. Image 2 is the original photograph of those same buildings from "
                  "a different camera. Use it for the identity and materials of the scene. Fill newly visible "
                  "surfaces with plausible architectural detail, replacing patchy projection textures. ")
    else:
        inputs = "The input is the original photograph. Show the same group of buildings from the new camera. "
    prompt = (inputs + move + " Preserve building density, roof shapes, street topology and courtyard boundaries. "
              "Return one full-frame photograph.")
    return prompt + (" Additional visual instruction: " + instruction.strip() if instruction.strip() else "")


def output_size(photo):
    configured = os.environ.get("ORBIT_IMAGE_SIZE")
    if configured:
        return configured
    ratio = photo.width / photo.height
    if not 1/3 <= ratio <= 3:
        raise ValueError("参考图过宽或过高，请调整裁切或配置图像服务的输出尺寸")
    width = max(16, round(math.sqrt(1024**2 * ratio) / 16) * 16)
    height = max(16, round(width / ratio / 16) * 16)
    width = min(width, 3 * height)
    height = min(height, 3 * width)
    return f"{width}x{height}"


def final_image(response):
    if "text/event-stream" not in response.headers.get("Content-Type", ""):
        return response.json()["data"][0]["b64_json"]
    data = []
    def event_image():
        if not data or data == [b"[DONE]"]:
            return None
        item = json.loads(b"\n".join(data))
        if item.get("type") == "error" or item.get("error"):
            raise ValueError("上游图像生成失败")
        if item.get("type") == "image_edit.completed":
            return item["b64_json"]
        return None
    for line in response.iter_lines(chunk_size=65536):
        if line.startswith(b"data:"):
            data.append(line[5:].lstrip())
        elif not line:
            completed = event_image()
            if completed:
                return completed
            data.clear()
    return event_image()


def generate_view(photo, guide, angle, model=None, guidance=None, quality=None, instruction=""):
    base = os.environ.get("ORBIT_IMAGE_BASE_URL", "").strip().rstrip("/") or "https://api.openai.com/v1"
    parsed = urlsplit(base)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("请检查场景服务配置的图像 API 基址")
    key = os.environ.get("ORBIT_IMAGE_API_KEY")
    if not key and parsed.hostname == "api.openai.com" and parsed.scheme == "https" and parsed.port in (None, 443):
        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("key")
        if not key:
            raise ValueError("请为场景服务配置图像 API 密钥")
    models = [m.strip() for m in (os.environ.get("ORBIT_IMAGE_MODELS") or DEFAULTS["model"]).split(",") if m.strip()]
    if not models or (model is not None and model not in models):
        raise ValueError("所选图像模型未在场景服务中启用")
    model = model or models[0]
    guidance = guidance or DEFAULTS["guidance"]
    quality = quality or DEFAULTS["quality"]
    if guidance not in [m["value"] for m in DEFAULTS["guidanceModes"]] or quality not in DEFAULTS["qualities"]:
        raise ValueError("无效的机位引导方式或出图质量")
    prompt = orbit_prompt(angle, guidance, instruction)
    inputs = [("target-layout", guide), ("reference", photo)] if guidance == "layout" else [("reference", photo)]
    files = []
    for role, image in inputs:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        files.append(("image[]" if len(inputs) > 1 else "image", (role + ".png", buffer.getvalue(), "image/png")))
    size = output_size(photo)
    fields = {"model": model, "prompt": prompt, "n": "1", "size": size, "quality": quality}
    streaming = os.environ.get("ORBIT_IMAGE_STREAM", "").lower() or ("true" if parsed.hostname == "api.openai.com" else "false")
    if streaming in ("true", "1"):
        fields.update(stream="true", partial_images="1")
    try:
        with requests.post(base + "/images/edits", files=files, headers={"Authorization": "Bearer " + key} if key else {},
                           data=fields, timeout=(15, 600), allow_redirects=False, stream=True) as response:
            if response.status_code != 200:
                raise ValueError(f"图像编辑 API 返回 HTTP {response.status_code}，请检查模型、访问权限和额度")
            try:
                image_bytes = base64.b64decode(final_image(response), validate=True)
                if not image_bytes:
                    raise ValueError("empty image")
            except (ValueError, TypeError, KeyError, IndexError, binascii.Error):
                raise ValueError("图像编辑 API 未返回有效的最终图像，请检查 b64_json 或流结束事件") from None
    except requests.RequestException as error:
        raise ValueError(f"图像 API 连接中断（{type(error).__name__}），请检查场景服务到模型接口的连接") from None
    return image_bytes, {"provider": "openai", "model": model, "guidance": guidance, "quality": quality,
                         "size": size, "inputRoles": [role for role, _ in inputs], "prompt": prompt}
