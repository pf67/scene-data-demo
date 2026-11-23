"""Prepare offline, image-bound depth evidence for the GLB builder."""

import argparse
import hashlib
import io
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(str(args.model_dir.resolve())).to(args.device).eval()
    args.output.mkdir(exist_ok=True, parents=True)
    for path in args.images:
        encoded = path.read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        with Image.open(io.BytesIO(encoded)) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        # Supply an exact patch grid so DA3's internal crop cannot shift pixels.
        shape = tuple(max(14, round(n / max(photo.size) * args.resolution / 14) * 14) for n in photo.size)
        image = photo.resize(shape, Image.Resampling.LANCZOS)
        with torch.inference_mode():
            prediction = model.inference([image], process_res=max(shape))
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if depth.shape != shape[::-1] or not (np.isfinite(depth) & (depth > 0)).all():
            raise ValueError("Depth inference returned an invalid grid")
        # A scale-free optical-axis distance, calibrated to the scene at build time.
        relative = (depth / np.median(depth)).astype(np.float16)
        with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as temporary:
            stage = Path(temporary.name)
        try:
            with stage.open("wb") as stream:
                np.savez_compressed(stream, relative=relative)
            stage.replace(args.output / (digest + ".npz"))
        finally:
            stage.unlink(missing_ok=True)
        print(path.name, shape, digest, flush=True)


if __name__ == "__main__":
    main()
