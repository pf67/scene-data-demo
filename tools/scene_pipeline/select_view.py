"""Convert DA3 per-view evidence into the village preparation cache convention."""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--index', type=int, default=0)
args = p.parse_args()
depth = np.load(args.input / 'depth.npy')
intrinsics = np.load(args.input / 'intrinsics.npy')
if depth.ndim == 3:
    depth = depth[args.index]
elif args.index != 0:
    p.error('Single-view depth accepts index 0 only')
if intrinsics.ndim == 3:
    intrinsics = intrinsics[args.index]
if depth.ndim != 2 or intrinsics.shape != (3, 3):
    p.error('Expected per-view HxW depth and 3x3 intrinsics')
if args.output.exists() and any(args.output.iterdir()):
    p.error('Output directory must be empty')
args.output.mkdir(parents=True, exist_ok=True)
np.save(args.output / 'depth.npy', depth)
np.save(args.output / 'K.npy', intrinsics)
print(args.output)
