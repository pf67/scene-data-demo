"""Generate a camera view from the original photograph using the QMA HTTP API."""
import base64
import io
import os
import time
from urllib.parse import quote

import requests


# Suggestions are offsets from the connected QMA service's advertised default.
# An explicitly selected strength always takes precedence, including zero.
DIRECTIONS = (
    (-135, "左后方", "back-left quarter view", .40),
    (-90, "左侧", "left side view", .25),
    (-45, "左前方", "front-left quarter view", 0),
    (45, "右前方", "front-right quarter view", 0),
    (90, "右侧", "right side view", .25),
    (135, "右后方", "back-right quarter view", .40),
    (180, "正后方", "back view", .50),
)


def camera_direction(angle):
    magnitude = abs(angle)
    available = (d for d in DIRECTIONS if d[0] == 180 or (d[0] > 0) == (angle > 0))
    return min(available, key=lambda d: (abs(abs(d[0]) - magnitude), -abs(d[0])))


def camera_options(session):
    api = os.environ.get("QMA_API_URL", "").rstrip("/")
    if not api:
        raise ValueError("请先为场景服务配置 QMA_API_URL")
    response = session.get(api + "/v1/options", timeout=(2, 4))
    response.raise_for_status()
    options = response.json()
    if not isinstance(options, dict) or not isinstance(options.get("azimuths"), list):
        raise ValueError("QMA 未返回有效的拍摄方向列表")
    default = options.get("default_strength")
    if type(default) not in (int, float) or not 0 <= default <= 1:
        raise ValueError("QMA 的视角选项缺少有效的默认 LoRA 强度")
    supported = options["azimuths"]
    directions = [{"angle": angle, "label": label, "azimuth": azimuth,
                   "suggestedStrength": min(1, round(default + increase, 6))}
                  for angle, label, azimuth, increase in DIRECTIONS if azimuth in supported]
    if not directions:
        raise ValueError("QMA 没有提供可用的旋转方向")
    return {"defaultStrength": default, "directions": directions}


def generate_view(photo, angle, seed, strength=None):
    _, _, azimuth, _ = camera_direction(angle)
    encoded = io.BytesIO()
    photo.save(encoded, format="PNG")
    api = os.environ["QMA_API_URL"].rstrip("/")
    with requests.Session() as session:
        if strength is None:
            choices = camera_options(session)["directions"]
            strength = next((d["suggestedStrength"] for d in choices if d["azimuth"] == azimuth), None)
            if strength is None:
                raise ValueError("所选旋转方向未在 QMA 服务上启用")
        if type(strength) not in (int, float) or not 0 <= strength <= 1:
            raise ValueError("LoRA 强度须在 0 到 1 之间")
        settings = {"azimuth": azimuth, "elevation": "high-angle shot", "distance": "wide shot",
                    "strength": strength, "seed": seed, "steps": 30, "cfg": 3, "resolution": 1024}
        payload = {**settings, "image_base64": base64.b64encode(encoded.getvalue()).decode("ascii")}
        response = session.post(api + "/v1/jobs", json=payload, timeout=90)
        response.raise_for_status()
        job_id = str(response.json()["job_id"])
        route = api + "/v1/jobs/" + quote(job_id, safe="")
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            response = session.get(route, timeout=30)
            response.raise_for_status()
            state = response.json()
            if state["status"] == "succeeded":
                if not state.get("images"):
                    raise ValueError("QMA 任务完成但没有返回图像")
                image = session.get(route + "/images/0", timeout=120)
                image.raise_for_status()
                return image.content, {"provider": "qma", "model": "Qwen-Image-2.1", **settings}
            if state["status"] in ("unknown", "failed"):
                raise ValueError("QMA 生成任务失败或已失效，请在模型服务检查任务 " + job_id[:80])
            time.sleep(2)
    raise TimeoutError("QMA 生成超时")
