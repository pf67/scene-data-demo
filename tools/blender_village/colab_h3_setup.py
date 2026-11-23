"""Generate turnkey Colab A100 ComfyUI + MiniMax-H3 setup cells and portable .ipynb for Layer 4 360 orbits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

INTRO_MARKDOWN = """## MiniMax-H3 多视角补全服务（Google Colab A100-40GB）

用于在云端启动 **ComfyUI + MiniMax-H3（同时支持 Image 双图视角生成与 FL2VA 360° 视频生成）** 并暴露临时公网 API URL，供本地 Blender 重建管线调用：

1. 在 Colab 菜单 **代码执行程序 → 更改运行时类型** 选择 **A100 GPU**。
2. 依次运行下方 3 个单元格（安装 ComfyUI → 下载共享编码器与 MiniMax-H3 Image/Video 权重 → 启动服务并打印 `MINIMAX_H3_COMFY_URL`）。
3. 接入方式（二选一）：
   - **固定本地网关**：直接访问固定地址 `http://127.0.0.1:8188`（或使用热接管 `POST http://127.0.0.1:8188/_gateway/set-upstream` 绑定下方打印的 `https://*.trycloudflare.com`）；不用时执行 `.\\stop_colab.ps1` 停止计费。
   - **直连隧道**：`python tools/blender_village/reconstruct_from_photo.py --photo <photo> --code <CODE> --multiview-provider minimax_h3_image --comfy-url <URL>`"""

CELL_1_INSTALL = r'''import importlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request

def run(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)

print("=== 1. Check GPU & Disk ===")
gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if gpu_check.returncode != 0:
    raise RuntimeError("No NVIDIA GPU detected! Please switch Colab Runtime -> Change runtime type -> A100 GPU first.")
print(gpu_check.stdout.splitlines()[0:12])
free_gib = shutil.disk_usage("/content").free / (1024 ** 3)
print(f"Free space on /content: {free_gib:.2f} GiB")
assert free_gib >= 65.0, f"Need >=65 GiB free on /content, found {free_gib:.1f} GiB"

run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "-qq", "aria2"])

COMFY = "/content/ComfyUI"
if not os.path.isdir(COMFY):
    run(["git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", COMFY])
else:
    run(["git", "-C", COMFY, "pull", "--ff-only"])

req_src = open(os.path.join(COMFY, "requirements.txt"), encoding="utf-8").read().splitlines()
filtered = [line for line in req_src if not re.match(r"^\s*(torch|torchvision|torchaudio)\b", line, re.I)]
req_filtered = "/tmp/comfy_req_no_torch.txt"
open(req_filtered, "w", encoding="utf-8").write("\n".join(filtered) + "\n")
run([sys.executable, "-m", "pip", "install", "-q", "-r", req_filtered])
importlib.invalidate_caches()

cf_bin = "/usr/local/bin/cloudflared"
if not os.path.isfile(cf_bin):
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        cf_bin,
    )
    os.chmod(cf_bin, 0o755)
print("CELL_1_SETUP_OK")'''

CELL_2_MODELS = r'''import json
import os
import subprocess
import time
import urllib.request

MINIMAX_H3_MODE = os.environ.get("MINIMAX_H3_MODE", "both")  # "image" | "video" | "both"

BASE = "/content/ComfyUI"
DIRS = {
    "text_encoders": os.path.join(BASE, "models", "text_encoders"),
    "diffusion_models": os.path.join(BASE, "models", "diffusion_models"),
    "loras": os.path.join(BASE, "models", "loras"),
    "vae": os.path.join(BASE, "models", "vae"),
    "workflows": os.path.join(BASE, "user", "default", "workflows"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

MODELS = [
    (
        DIRS["text_encoders"],
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    ),
]

if MINIMAX_H3_MODE in ("image", "both"):
    MODELS.extend([
        (
            DIRS["diffusion_models"],
            "minimax_h3_image_pruned_int8_convrot.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
        ),
        (
            DIRS["vae"],
            "minimax_h3_image_vae_fp16.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/vae/minimax_h3_image_vae_fp16.safetensors",
        ),
    ])

if MINIMAX_H3_MODE in ("video", "both"):
    MODELS.extend([
        (
            DIRS["diffusion_models"],
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        ),
        (
            DIRS["vae"],
            "minimax_h3_video_vae_int8_convrot.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        ),
        (
            DIRS["vae"],
            "minimax_h3_audio_vae_fp32.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        ),
        (
            DIRS["loras"],
            "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        ),
    ])

for target_dir, filename, url in MODELS:
    full_path = os.path.join(target_dir, filename)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 100_000_000:
        print(f"[SKIP] {filename} ({os.path.getsize(full_path)/1024**3:.2f} GiB)")
        continue
    t0 = time.time()
    print(f"[DOWNLOAD] {filename}...")
    subprocess.run(
        [
            "aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M",
            "--file-allocation=none", "--console-log-level=warn", "--summary-interval=30",
            "-d", target_dir, "-o", filename, url,
        ],
        check=True,
    )
    print(f"[OK] {filename} in {time.time() - t0:.1f}s")

if MINIMAX_H3_MODE in ("video", "both"):
    wf_url = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_i2v.json"
    raw = urllib.request.urlopen(wf_url, timeout=30).read().decode("utf-8")
    patched = raw.replace(
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    )
    json.loads(patched)
    open(os.path.join(DIRS["workflows"], "video_minimax_h3_i2v.json"), "w", encoding="utf-8").write(patched)
print("CELL_2_MODELS_OK", MINIMAX_H3_MODE)'''

CELL_3_LAUNCH = r'''import json
import re
import subprocess
import time
import urllib.request

for pat in ["/content/ComfyUI/main.py", "cloudflared"]:
    subprocess.run(["pkill", "-f", pat], check=False)
time.sleep(1)

comfy_log = open("/tmp/comfyui.log", "w", encoding="utf-8")
subprocess.Popen(
    ["python3", "/content/ComfyUI/main.py", "--listen", "127.0.0.1", "--port", "8188"],
    stdout=comfy_log,
    stderr=subprocess.STDOUT,
)

for _ in range(90):
    try:
        stats = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3).read())
        info_img = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/object_info/MiniMaxH3ImageConditioning", timeout=3).read())
        info_vid = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/object_info/MiniMaxH3ImageToVideo", timeout=3).read())
        if "MiniMaxH3ImageConditioning" in info_img or "MiniMaxH3ImageToVideo" in info_vid:
            print("ComfyUI Ready:", stats["system"]["comfyui_version"],
                  "image_node=", "MiniMaxH3ImageConditioning" in info_img,
                  "video_node=", "MiniMaxH3ImageToVideo" in info_vid)
            break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("ComfyUI failed to become healthy within 180s; check /tmp/comfyui.log")

cf_log_path = "/tmp/cloudflared.log"
cf_log = open(cf_log_path, "w", encoding="utf-8")
subprocess.Popen(
    ["/usr/local/bin/cloudflared", "tunnel", "--url", "http://127.0.0.1:8188"],
    stdout=cf_log,
    stderr=subprocess.STDOUT,
)

tunnel_url = None
for _ in range(30):
    txt = open(cf_log_path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
    if m:
        tunnel_url = m.group(0)
        break
    time.sleep(1)

assert tunnel_url, "Failed to obtain trycloudflare.com URL"
print("======================================================")
print("MINIMAX_H3_COMFY_URL =", tunnel_url)
print("Fixed Local Gateway:  http://127.0.0.1:8188")
print(f"Gateway Hot-Takeover: curl -X POST http://127.0.0.1:8188/_gateway/set-upstream -H \"Content-Type: application/json\" -d '{{\"url\":\"{tunnel_url}\"}}'")
print(f"Image Multiview:      python tools/blender_village/reconstruct_from_photo.py --photo assets/scene_sources/uk_houses_drone.png --code UK_VILLAGE_BLENDER --multiview-provider minimax_h3_image --comfy-url http://127.0.0.1:8188 --gateway-upstream {tunnel_url}")
print(f"Video Orbit:          python tools/blender_village/run_full_pipeline.py --comfy-url http://127.0.0.1:8188")
print("Stop Billing:         .\\stop_colab.ps1")
print("======================================================")'''


def get_colab_cells() -> list[dict[str, str]]:
    """Return ordered list of Markdown/code cell definitions for Colab A100 setup."""
    return [
        {"type": "markdown", "source": INTRO_MARKDOWN},
        {"type": "code", "source": CELL_1_INSTALL},
        {"type": "code", "source": CELL_2_MODELS},
        {"type": "code", "source": CELL_3_LAUNCH},
    ]


def write_colab_notebook(out_ipynb: Path) -> Path:
    """Emit a portable Jupyter .ipynb file ready to open/run on Google Colab A100."""
    nb_cells = []
    for item in get_colab_cells():
        if item["type"] == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": item["source"].splitlines(keepends=True)})
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": item["source"].splitlines(keepends=True),
                }
            )
    doc: dict[str, Any] = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"name": "MiniMax_H3_Village_360_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": nb_cells,
    }
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    out_ipynb.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote Colab A100 notebook -> {out_ipynb}")
    return out_ipynb


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/blender_village/MiniMax_H3_Village_360_Colab.ipynb")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    write_colab_notebook(args.out.resolve())
