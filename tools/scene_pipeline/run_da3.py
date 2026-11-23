"""Offline single/multi-image DA3 inference. Produces evidence, not a finished scene."""
import argparse
import json
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=504)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        parser.error('An existing local model snapshot is required; this script never downloads weights.')
    images = []
    for source in args.images:
        candidates = sorted(source.iterdir()) if source.is_dir() else [source]
        images.extend(p.resolve() for p in candidates if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    if not images or any(not p.is_file() for p in images):
        parser.error('No valid input images')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use an empty output folder to preserve earlier evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; install a matching PyTorch/CUDA environment or use --device cpu')
    model = DepthAnything3.from_pretrained(str(args.model_dir.resolve())).to(args.device)
    with torch.inference_mode():
        prediction = model.inference([str(p) for p in images], process_res=args.resolution,
                                     export_dir=str(args.output), export_format='mini_npz-glb')
    for name in ['depth', 'conf', 'intrinsics', 'extrinsics', 'processed_images']:
        value = getattr(prediction, name, None)
        if value is not None:
            np.save(args.output / f'{name}.npy', value)
    evidence = {'images': [str(p) for p in images], 'model_directory': str(args.model_dir.resolve()),
                'resolution': args.resolution, 'device': args.device, 'torch': torch.__version__,
                'note': 'Raw DA3 output may be points or an open surface. Complete/texture/validate geometry before deploying.'}
    (args.output / 'input-manifest.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
