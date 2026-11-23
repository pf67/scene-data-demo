"""Convert the curated annotations and cached depth into a reusable builder plan."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def make_example():
    layout = json.loads((ROOT / "assets/village_studio/layout.json").read_text("utf-8"))
    depth = np.load(ROOT / "assets/da3_out/village_road_near/depth.npy")
    frame, origin, k = [np.array(layout[key]) for key in ("frame", "origin", "intrinsics")]
    scale = layout["scale"]
    eye = -frame @ origin * scale
    h, w = depth.shape
    y, x = np.mgrid[:h, :w]
    rays = np.stack([(x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)], -1)
    height = eye[1] + (rays @ frame[1]) * depth * scale
    size = np.array(layout["imageSize"])
    def ground_point(u, v):
        ray = frame @ np.array([(u * w - k[0, 2]) / k[0, 0], (v * h - k[1, 2]) / k[1, 1], 1])
        ground = eye - eye[1] / ray[1] * ray
        return np.round(ground[[0, 2]], 6).tolist()

    controls = [{"uv": [u, v], "xz": ground_point(u, v)}
                for u, v in ((.08, .744), (.24, .610), (.54, .467), (.85, .304), (.7, .345), (.38, .558))]
    # All example controls come from the same prior, so none are independent checks.
    # Keep terrain evidence near the fitted plane. Distant tree crowns and
    # untraced roofs would otherwise stretch into streaks in the orthophoto.
    support = np.isfinite(height) & (np.abs(height) < 1.6)
    Image.fromarray(np.uint8(support)*255).save(ROOT / "builder/village-ground.png")
    view = {"source": "village", "groundMask": "village-depth-ground", "fovY": math.degrees(2 * math.atan(h / (2 * k[1, 1]))),
            "focalRatio": (k[0, 0] * size[0] / w) / (k[1, 1] * size[1] / h), "controls": controls}
    buildings, before, after = [], [], []
    for old in layout["buildings"]:
        uv = np.array(old["roof"]) / size
        mask = np.zeros_like(depth, dtype=np.uint8)
        cv2.fillPoly(mask, [np.int32(np.round(uv * [w, h]))], 1)
        core = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool) & np.isfinite(height)
        holdout = (x + 3 * y) % 5 == 0
        samples = height[core & ~holdout]
        estimate = old["h"]
        gable = old.get("type") == "gable"
        if len(samples) >= 15 and not gable:
            median = np.median(samples)
            if 1 < median < 25 and np.median(np.abs(samples - median)) <= 1.7 and (core & holdout).any():
                # In this village the legacy estimates reflect visible storeys.
                # A monocular depth prior can drift with distance and should not
                # turn a one-storey traced house into a tall block.
                estimate = round(float(np.clip(median, .65 * old["h"], 1.35 * old["h"])), 3)
                check = height[core & holdout]
                before.append(float(np.median(abs(check - old["h"]))))
                after.append(float(np.median(abs(check - estimate))))
        b = {"id": old["id"], "heightM": estimate, "heightSource": "estimated",
             "observations": [{"source": "village", "roof": uv.round(7).tolist()}]}
        if gable:
            b["ridgeRiseM"] = 1.25
        buildings.append(b)
    plan = {"version": 1, "name": "村庄沿街建筑", "scaleBasis": "estimated", "views": [view], "buildings": buildings}
    centerline = [ground_point(px / size[0], py / size[1])
                  for px, py in ((0,765), (400,604), (800,448), (1200,307), (1536,229))]
    plan["ground"] = {"marginPercent": 10, "surface": "fields", "reliefM": 1.1,
                      "roads": [{"points": centerline, "widthM": 6.5, "extendEnds": True}]}
    samples = ROOT / "builder/sample-views"
    sample_views = json.loads((samples / "manifest.json").read_text("utf-8"))
    plan["supplements"] = []
    for row in sample_views["views"]:
        spec = {"source": row["source"], "reference": sample_views["reference"], "angle": row["angle"],
                "provider": row["provider"], "provenance": row.get("provenance", "qwen-image-2.1")}
        if row.get("provenance") == "image-api":
            spec["model"] = row["model"]
        if row.get("pointsFile"):
            spec["points"] = json.loads((samples / row["pointsFile"]).read_text("utf-8"))
        plan["supplements"].append(spec)
    path = ROOT / "builder/example.json"
    path.write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(buildings)} buildings; {len(after)} depth-adjusted flat roofs; median per-roof held-out depth-height error "
          f"{np.median(before):.3f} -> {np.median(after):.3f} scene units (relative depth proxy)")


if __name__ == "__main__":
    make_example()
