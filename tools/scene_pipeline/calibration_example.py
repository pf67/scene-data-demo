"""Create a portable two-view calibration exercise with known dimensions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = (960, 640)
FOV = 48
HEIGHT = 5.4
FOOTPRINT = np.array([[-4, 3], [4, 3], [4, -3], [-4, -3]], float)
CONTROLS = [(-10, -5), (10, -5), (7, 5), (-7, 5), (0, 7), (-9, 0), (9, 0), (3, 7), (-3, 7), (6, 8)]


def create(directory):
    directory.mkdir(exist_ok=True, parents=True)
    views, observations, files = [], [], {}
    for index, position in enumerate(([10, 13, 23], [-14, 12, 19])):
        eye = np.array(position)
        forward = np.array([0, 2, 0]) - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        focal = SIZE[1] / (2 * math.tan(math.radians(FOV) / 2))

        def project(vertices):
            delta = vertices - eye
            uv = np.stack([delta @ right, delta @ down], axis=1)
            return uv * focal / (delta @ forward)[:, None] + np.array(SIZE) / 2

        def at_height(xz, h):
            xz = np.array(xz)
            return np.column_stack([xz[:, 0], np.full(len(xz), h), xz[:, 1]])

        image = Image.new("RGB", SIZE, "#bfc6b0")
        draw = ImageDraw.Draw(image)
        for a in range(-14, 16, 2):
            for line in [[(a, -12), (a, 12)], [(-14, a), (14, a)]]:
                draw.line([tuple(p) for p in project(at_height(line, 0))], fill="#a5b09b", width=1)
        base, roof = at_height(FOOTPRINT, 0), at_height(FOOTPRINT, HEIGHT)
        walls = []
        for j in range(4):
            k = (j + 1) % 4
            face = np.array([base[j], base[k], roof[k], roof[j]])
            walls.append((float(np.linalg.norm(face.mean(0) - eye)), face, ["#c6b38c", "#d9c8a8", "#eee3c9", "#b4a88d"][j]))
        silhouette = Image.new("L", SIZE)
        mask = ImageDraw.Draw(silhouette)
        for _, face, color in sorted(walls, key=lambda x: -x[0]):
            polygon = [tuple(p) for p in project(face)]
            draw.polygon(polygon, fill=color, outline="#645d4e", width=2)
            mask.polygon(polygon, fill=255)
        roof_px = project(roof)
        draw.polygon([tuple(p) for p in roof_px], fill="#718891", outline="#394e56", width=2)
        mask.polygon([tuple(p) for p in roof_px], fill=255)
        silhouette = silhouette.filter(ImageFilter.MaxFilter(9))
        for j, (u, v) in enumerate(roof_px):
            draw.text((u + 3, v - 15), str(j + 1), fill="white")
        visible = []
        for xz, pt in zip(CONTROLS, project(at_height(CONTROLS, 0))):
            if (pt > 12).all() and (pt < np.array(SIZE) - 12).all() and not silhouette.getpixel(tuple(pt.astype(int))):
                visible.append(xz)
        visible = visible[:6]
        ground_px = project(at_height(visible, 0))
        for j, (u, v) in enumerate(ground_px):
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill="#0b5760")
            draw.text((u + 4, v), chr(65 + j), fill="#163a3b")
        filename = directory / f"view-{index + 1}.png"
        image.save(filename)
        source = "src_" + hashlib.sha256(filename.read_bytes()).hexdigest()
        files[source] = filename
        controls = [{"uv": (p / SIZE).tolist(), "xz": list(xz), "check": j >= 5} for j, (p, xz) in enumerate(zip(ground_px, visible))]
        views.append({"source": source, "fovY": FOV, "controls": controls})
        observations.append({"source": source, "roof": (roof_px / SIZE).tolist(), "base": (project(base)[0] / SIZE).tolist()})
    plan = {"version": 1, "name": "双视图标定样房", "scaleBasis": "measured", "views": views,
            "buildings": [{"id": "studio", "observations": observations}]}
    (directory / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), "utf-8")
    return plan, files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    create(parser.parse_args().directory)
