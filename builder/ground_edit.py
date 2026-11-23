"""Apply an orthographic ground edit to a versioned GLB while preserving all geometry."""
import io
import json
import shutil
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from mesh import write_glb


def read_edit(source):
    try:
        with Image.open(source) as image:
            if image.format not in ("PNG", "JPEG") or image.width * image.height > 32_000_000 or min(image.size) < 32:
                raise ValueError("补图须为 PNG/JPEG，短边至少 32 px，总像素最多 3200 万")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError("无法读取补图，请检查图像文件") from error


def generated_ortho(base, prompt, directory):
    # A square reference gives the model a fixed coordinate frame even for narrow sites.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from refinement.qwen import generate
    frame_size = 1024
    w, h = base.size
    scale = frame_size / max(w, h)
    width, height = max(1, round(w*scale)), max(1, round(h*scale))
    x, y = (frame_size-width)//2, (frame_size-height)//2
    small = np.asarray(base.resize((width, height), Image.Resampling.LANCZOS))
    frame = np.pad(small, ((y, frame_size-height-y), (x, frame_size-width-x), (0, 0)), mode="edge")
    instruction = ("Generate a photorealistic orthographic terrain texture from this aligned ground atlas. "
                   "View vertically downward. Keep road axes, parcel boundaries, orientation and scale fixed. "
                   "Rebuild natural soil, gravel, vegetation and pavement detail. No buildings, vehicles, "
                   "trees with height, horizon, labels, perspective or cast shadows. " + prompt)
    data = generate(Image.fromarray(frame), instruction, directory)
    output = read_edit(io.BytesIO(data))
    if abs(output.width/output.height-1) > .02:
        raise ValueError("模型输出改变了俯视构图比例，请重新生成")
    return output.resize((frame_size, frame_size), Image.Resampling.LANCZOS).crop((x, y, x+width, y+height))


def replace_texture(model, target, png):
    raw = model.read_bytes()
    meta_size = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20:20 + meta_size])
    binary = memoryview(raw)[28 + meta_size:]
    # The exporter writes the terrain as the last mesh, including for building IDs named Ground.
    ground = doc["meshes"][-1]
    if ground.get("name") != "Ground":
        raise ValueError("此场景缺少可编辑的地面网格")
    if any(scene.get("extras", {}).get("photo_projectors", {}).get("cameras") for scene in doc["scenes"]):
        ground_mesh = len(doc["meshes"]) - 1
        for node in doc["nodes"]:
            if node.get("mesh") == ground_mesh:
                node.setdefault("extras", {})["photo_projectors"] = False
    pbr = doc["materials"][ground["primitives"][0]["material"]]["pbrMetallicRoughness"]
    image = doc["images"][doc["textures"][pbr["baseColorTexture"]["index"]]["source"]]
    image["mimeType"] = "image/png"
    blob = bytearray()
    for index, view in enumerate(doc["bufferViews"]):
        offset = view.get("byteOffset", 0)
        chunk = png if index == image["bufferView"] else binary[offset:offset+view["byteLength"]]
        blob.extend(bytes(-len(blob) % 4))
        view.update(byteOffset=len(blob), byteLength=len(chunk))
        blob.extend(chunk)
    write_glb(target, doc, blob)


def edit_ground(model, ground, target, output, image=None, prompt=None):
    if model.resolve() == target.resolve() or ground.resolve() == output.resolve():
        raise ValueError("请为新版本指定独立的 GLB 路径和地面图层目录")
    with Image.open(ground / "texture.png") as source:
        base = source.convert("RGB")
    with Image.open(ground / "mask.png") as mask:
        editable = np.asarray(mask.convert("L")) >= 128
    if not editable.any():
        raise ValueError("地面已全部由照片覆盖")
    if image:
        edited = read_edit(image)
        if abs((edited.width/edited.height)/(base.width/base.height)-1) > .01:
            raise ValueError("补图须与下载的地面纹理方向和宽高比一致")
    else:
        edited = generated_ortho(base, prompt or "", target.parent)
    candidate = np.asarray(edited.resize(base.size, Image.Resampling.LANCZOS))
    # Feather inward only; observed texels remain bit-for-bit identical.
    alpha = np.minimum(cv2.distanceTransform(editable.astype(np.uint8), cv2.DIST_L2, 5)/4, 1)[:, :, None]
    merged = np.rint(np.asarray(base)*(1-alpha) + candidate*alpha).astype(np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    texture = output / "texture.png"
    Image.fromarray(merged).save(texture, compress_level=3)
    for name in ("mask.png", "coverage.png"):
        shutil.copyfile(ground / name, output / name)
    replace_texture(model, target, texture.read_bytes())
    return {"method": "imported" if image else "qwen-image-2.1"}
