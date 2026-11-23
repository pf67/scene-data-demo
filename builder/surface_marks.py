"""Image-space annotations for reviewed surfaces, material swatches and inferred objects."""

import json
import os
from pathlib import Path

import cv2
import numpy as np


class SurfaceMarks:
    def __init__(self, source, camera, image, data, paired_mask=None, paired_ground=None):
        self.source, self.cam, self.image = source, camera, image
        self.data = data
        self.paired_mask = paired_mask
        self.paired_ground = paired_ground
        self.size = np.asarray(data["size"], float)
        self.buildings = {b["id"]: b for b in data.get("buildings", [])}
        self.ground = None
        if data.get("ground"):
            self.ground = np.zeros(image.shape[:2], np.uint8)
            scale = self.cam.size / self.size
            for polygon in data["ground"]:
                cv2.fillPoly(self.ground, [np.rint(np.array(polygon) * scale).astype("int32")], 1)

    def pixels(self, points):
        return np.asarray(points, np.float32) * (self.cam.size / self.size)

    def match(self, roof, label):
        item = self.buildings.get(label)
        if not item or len(roof) != 4:
            return None
        target = self.pixels(item["roof"])
        projected, z = self.cam.pixels(roof)
        if min(z) < 0.2:
            return None
        candidates = [np.roll(np.arange(4)[::direction], n) for n in range(4) for direction in (1, -1)]
        # Each corner keeps its identity around the eave, even when the image model
        # translated or resized the house within its globally calibrated camera.
        span = max(np.linalg.norm(np.ptp(projected, axis=0)), 1)
        order = min(candidates, key=lambda order: np.linalg.norm(projected - target[order]))
        if np.linalg.norm(projected - target[order], axis=1).mean() > 0.68 * span:
            return None
        feet = [self.pixels(item["feet"][i]) if item["feet"][i] is not None else None for i in order]
        return FaceMapping(self, roof, target[order], feet)

    @classmethod
    def load(cls, digest, source, camera, image):
        folders = [Path(__file__).parent / "surface-marks"]
        if os.environ.get("SCENE_SURFACE_MARKS_DIR"):
            folders.insert(0, Path(os.environ["SCENE_SURFACE_MARKS_DIR"]))
        for folder in folders:
            path = folder / (digest + ".json")
            if path.is_file():
                data = json.loads(path.read_text("utf8"))
                def mask_for(key, suffix):
                    if not data.get(key):
                        return None
                    from PIL import Image
                    import hashlib
                    mask_path = folder / (digest + suffix)
                    if hashlib.sha256(mask_path.read_bytes()).hexdigest() != data[key]:
                        raise ValueError("双视图对齐区域与已验收的记录不一致")
                    with Image.open(mask_path) as bitmap:
                        return np.asarray(bitmap.convert("L")) != 0
                return cls(source, camera, image, data, mask_for("pairedMask",".png"),mask_for("pairedGround","-ground.png"))
        return None


class FaceMapping:
    def __init__(self, marks, roof, corners, feet):
        self.marks, self.roof, self.corners, self.feet = marks, roof, corners, feet
        source, _ = marks.cam.pixels(roof)
        self.roof_transform = cv2.getPerspectiveTransform(np.float32(source), np.float32(corners))

    def texture(self, face, normal, vertices):
        cam = self.marks.cam
        if normal @ (cam.eye - face.mean(0)) <= 0:
            return None
        if normal[1] > 0.3:
            transform = self.roof_transform
        elif abs(normal[1]) < 0.1 and len(face) == 4:
            order = np.argmin(
                np.linalg.norm(face[:, [0, 2]][:, None] - self.roof[:, [0, 2]][None, :], axis=2), axis=1
            )
            mapped = [self.corners[i] if p[1] > 0.1 else self.feet[i] for i, p in zip(order, face)]
            if any(v is None for v in mapped):
                return None
            projected, _ = cam.pixels(face)
            dst = np.float32(mapped)
            if not cv2.isContourConvex(dst) or abs(cv2.contourArea(dst)) < 18:
                return None
            transform = cv2.getPerspectiveTransform(np.float32(projected), dst)
        else:
            return None
        uv, _ = cam.pixels(vertices)
        pixels = cv2.perspectiveTransform(np.float32(uv)[None], transform)[0]
        return np.clip(pixels / cam.size, [0.0001, 0.0001], [0.9999, 0.9999])
