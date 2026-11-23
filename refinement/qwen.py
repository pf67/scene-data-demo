"""Run the bundled Qwen-Image-2.1 graph with its spatial and photographic references."""
import io
import json
import math
import os
import secrets
import time
from pathlib import Path

import requests


def generate(image, prompt, directory, steps=40, cfg=1, *, reference=None, resolution=1024, seed=None, novel_mask=None):
    """Submit both spatial and appearance references when the caller supplies two images."""
    api = os.environ["COMFYUI_URL"].rstrip("/")
    workflow = json.loads(Path(__file__).with_name("workflow.json").read_text("utf-8"))
    with requests.Session() as session:
        workflow["1"]["inputs"]["image"] = upload_image(session, api, image, directory.name + ".png")
        if reference is not None:
            reference_name = upload_image(session, api, reference, directory.name + "-reference.png")
            workflow["10"] = {"class_type": "LoadImage", "inputs": {"image": reference_name}}
            workflow["6"]["inputs"]["images.image_2"] = ["10", 0]
        if novel_mask is not None:
            # Anchor the new viewpoint at observed roof corners and let Qwen generate
            # the surfaces revealed by the camera move at full sampling strength.
            mask_name = upload_image(session, api, novel_mask, directory.name + "-disocclusions.png")
            ratio = image.width / image.height
            width = max(32, round(resolution * math.sqrt(ratio) / 32) * 32)
            height = max(32, round(resolution / math.sqrt(ratio) / 32) * 32)
            workflow["11"] = {"class_type": "ImageScale", "inputs": {
                "image": ["1", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled"}}
            workflow["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["4", 0]}}
            workflow["13"] = {"class_type": "LoadImageMask", "inputs": {"image": mask_name, "channel": "red"}}
            workflow["14"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["13", 0]}}
            workflow["7"]["inputs"]["latent_image"] = ["14", 0]
        workflow["6"]["inputs"].update(prompt=prompt, resolution=resolution)
        workflow["7"]["inputs"].update(seed=secrets.randbelow(2**53) if seed is None else seed,
                                       steps=max(20, min(100, int(steps))), cfg=max(0, min(30, float(cfg))))
        workflow["9"]["inputs"]["filename_prefix"] = directory.name
        return execute(session, api, workflow, directory)


def upload_image(session, api, image, name):
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    upload = session.post(api + "/upload/image", files={"image": (name, encoded.getvalue(), "image/png")},
                          data={"type": "input"}, timeout=90)
    upload.raise_for_status()
    uploaded = upload.json()
    return "/".join(p for p in [uploaded.get("subfolder"), uploaded["name"]] if p)


def execute(session, api, workflow, directory):
    submitted = session.post(api + "/prompt", json={"prompt": workflow}, timeout=60)
    submitted.raise_for_status()
    prompt_id = submitted.json()["prompt_id"]
    (directory / "comfy-task.json").write_text(json.dumps({"prompt_id": prompt_id}), "utf-8")
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        history = session.get(api + "/history/" + prompt_id, timeout=60)
        history.raise_for_status()
        entry = history.json().get(prompt_id, {})
        if entry.get("status", {}).get("status_str") == "error":
            raise RuntimeError("ComfyUI execution failed. Inspect its task history.")
        outputs = entry.get("outputs", {}).get("9", {}).get("images", [])
        if outputs:
            output = session.get(api + "/view", params=outputs[0], timeout=120)
            output.raise_for_status()
            return output.content
        time.sleep(2)
    raise TimeoutError("ComfyUI task did not finish within 30 minutes")
