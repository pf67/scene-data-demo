"""Bake locally matched courtyard photographs into their surveyed ground polygons."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image


def apply_local_yards(texture, ground, locked, bank, roofs):
    """Return the amended RGB texture and per-source counts of changed, unlocked texels."""
    paired_yards = getattr(bank,"paired_yards",[])
    if not bank.reviewed_yards and not paired_yards:
        return texture, {}
    rgb = np.array(texture.convert("RGB"))
    h, w = rgb.shape[:2]
    lo, hi = np.asarray(ground["boundsXZ"])
    scale = np.array([w - 1, h - 1]) / (hi - lo)
    protected = np.uint8(locked).copy()
    if protected.shape != (h, w):
        raise ValueError("Ground observation mask and texture must have the same dimensions")
    for roof in roofs:
        poly = (np.asarray(roof)[:, [0, 2]] - lo) * scale
        cv2.fillConvexPoly(protected, np.int32(np.rint(poly)), 1)
    # Prefer the observation with the highest source sampling density if
    # adjacent, independently checked courtyard patches happen to overlap.
    yards = sorted(bank.reviewed_yards, key=lambda row:
                   abs(cv2.contourArea(row[2]) / cv2.contourArea(row[3])), reverse=True)
    counts = Counter()
    for source, photo, image_quad, world_quad in yards:
        target = np.float32((world_quad - lo) * scale)
        x0, y0 = np.maximum(0, np.floor(target.min(0)).astype(int))
        x1, y1 = np.minimum([w, h], np.ceil(target.max(0)).astype(int) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        size = (int(x1 - x0), int(y1 - y0))
        local = np.float32(target - [x0, y0])
        mask = np.zeros(size[::-1], np.uint8)
        cv2.fillConvexPoly(mask, np.int32(np.rint(local)), 1)
        window = np.s_[y0:y1, x0:x1]
        mask[protected[window] != 0] = 0
        # One-pixel guard removes the source-side rounding uncertainty. A
        # three-pixel ramp entirely inside the polygon blends its inner rim.
        dist = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        alpha = np.clip((dist - 1) / 3, 0, 1)
        used = alpha > 0
        if not used.any():
            continue
        warp = cv2.getPerspectiveTransform(image_quad, local)
        color = cv2.warpPerspective(photo, warp, size, flags=cv2.INTER_LINEAR)
        old = rgb[window]
        values = np.rint(old + (color.astype(np.float32) - old) * alpha[:, :, None]).astype(np.uint8)
        counts[source] += int(np.count_nonzero(used & np.any(old != values, axis=2)))
        old[used] = values[used]
        protected[window][used] = 1
    for mark in paired_yards:
        mask = mark.paired_ground.astype(np.uint8)
        # Project only the small verified yard extent. Nearest-neighbour
        # sampling cannot extend the support across a fence or roof edge.
        yy,xx=np.where(mask)
        if not len(xx):continue
        norm=np.column_stack([xx,yy])/np.array(mask.shape[::-1])
        world=mark.cam.plane(norm,0)[:,[0,2]]
        atlas=(world-lo)*scale
        x0,y0=np.maximum(0,np.floor(atlas.min(0)-2).astype(int))
        x1,y1=np.minimum([w,h],np.ceil(atlas.max(0)+3).astype(int))
        if x1<=x0 or y1<=y0:continue
        row,col=np.mgrid[y0:y1,x0:x1]
        xyz=np.stack([lo[0]+col/scale[0],np.zeros_like(col),lo[1]+row/scale[1]],axis=-1)
        pixels,z=mark.cam.pixels(xyz.reshape(-1,3))
        uv=pixels.reshape(*row.shape,2).astype(np.float32)
        support=cv2.remap(mask, np.float32(uv*np.array(mask.shape[::-1])/mark.cam.size),None,cv2.INTER_NEAREST)
        area=np.s_[y0:y1,x0:x1]
        use=(support!=0)&(protected[area]==0)&(z.reshape(row.shape)>.1)
        if not use.any():continue
        src=cv2.remap(mark.image,uv,None,cv2.INTER_LINEAR)
        inside=cv2.distanceTransform(np.pad(np.uint8(use),1),cv2.DIST_L2,5)[1:-1,1:-1]
        alpha=np.minimum(1,inside/2.5)[:,:,None]
        window=rgb[area]
        mixed=np.uint8(np.rint(window*(1-alpha)+src*alpha))
        counts[mark.source]+=int(np.count_nonzero(use&np.any(window!=mixed,axis=2)))
        window[use]=mixed[use];protected[area][use]=1
    return Image.fromarray(rgb), dict(counts)
