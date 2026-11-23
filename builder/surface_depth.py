"""Calibrate a cached monocular depth grid to a scene and sample it on physical surfaces."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from ground import road_distance


def image_digest(path):
    with Path(path).open("rb") as source:
        return (
            hashlib.file_digest(source, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else hashlib.sha256(source.read()).hexdigest()
        )


def cached_depth(path, digest=None):
    folder = Path(__file__).resolve().parent / "surface-depth"
    roots = [folder]
    if os.environ.get("SCENE_SURFACE_DEPTH_DIR"):
        roots.insert(0, Path(os.environ["SCENE_SURFACE_DEPTH_DIR"]))
    if not any(root.is_dir() for root in roots):
        return None
    name = digest or image_digest(path)
    for root in roots:
        cache = root / (name + ".npz")
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as data:
                return data["relative"].astype(np.float32)
    return None


def rays_for(cam):
    width, height = cam.size.astype(int)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return (
        np.stack([(x - cam.k[0, 2]) / cam.k[0, 0], (y - cam.k[1, 2]) / cam.k[1, 1], np.ones_like(x)], -1)
        @ cam.r
    ).astype(np.float32)


def normals_for(points):
    dx = cv2.Sobel(points, cv2.CV_32F, 1, 0, ksize=5)
    dy = cv2.Sobel(points, cv2.CV_32F, 0, 1, ksize=5)
    normal = np.cross(dx, dy)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-8)
    return normal


@dataclass
class SurfaceDepth:
    camera: object
    depth: np.ndarray
    normals: np.ndarray
    height: np.ndarray
    anchor_error: float
    accepted_region: object = None

    def check(self, uv, z, normal, image_size):
        px = np.clip(
            np.rint(uv * (self.camera.size / image_size)).astype(int), 0, self.camera.size.astype(int) - 1
        )
        x, y = px[:, 0], px[:, 1]
        local = True if self.accepted_region is None else self.accepted_region[y, x]
        n = self.normals[y, x]
        if normal is None:
            # Ground cannot borrow roofs, tree crowns or the vertically projected face of a house.
            return local & (np.abs(self.height[y, x]) < 1.35) & (np.abs(n[:, 1]) > 0.65)
        tolerance = np.clip(0.025 * z, 0.85, 3.5)
        return local & (np.abs(z - self.depth[y, x]) < tolerance) & (np.abs(n @ normal) > 0.57)


def fit_surface_depth(camera, raw, faces, roads, ground_mask=None, registration_region=None):
    from orbit import scaled, surface_map

    if raw is None or not faces:
        return None
    h, w = raw.shape
    cam = scaled(camera, (w, h))
    rays = rays_for(cam)
    shell, shell_z, ids = surface_map(cam, faces)
    raw = cv2.bilateralFilter(raw, 5, 0.012, 2)
    normal = normals_for(rays * raw[:, :, None])
    upward = np.abs(normal[:, :, 1]) > 0.60
    roof = np.array([n[1] > 0.55 for _, n in faces])[np.maximum(ids, 0)] & (ids >= 0)
    roof = cv2.erode(roof.astype(np.uint8), np.ones((7, 7), np.uint8)) != 0
    ground = np.zeros((h, w), bool)
    if ground_mask is not None:
        ground |= cv2.resize(ground_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) != 0
    for road in roads:
        dist, _ = road_distance(shell[:, :, 0], shell[:, :, 2], road)
        ground |= dist < road["widthM"] * 0.38
    anchors = upward & (roof | (ground & (ids == -1))) & np.isfinite(shell_z) & (shell_z > 0.3) & (raw > 0)
    accepted_region = None
    if registration_region is not None:
        accepted_region = cv2.resize(np.uint8(registration_region), (w,h), interpolation=cv2.INTER_NEAREST_EXACT).astype(bool)
        anchors &= accepted_region
    if anchors.sum() < 80:
        return None
    # Fit inverse distance to structural anchors over the depth range. Subsample to
    # keep the robust solve inexpensive on high resolution maps.
    observed, target = raw[anchors][::5].astype(float), shell_z[anchors][::5].astype(float)
    design = np.column_stack([1 / observed, np.ones_like(observed)]) * target[:, None]
    weights = np.ones_like(target)
    for _ in range(6):
        coef = np.linalg.lstsq(design * weights[:, None], weights, rcond=None)[0]
        residual = np.abs(design @ coef - 1)
        weights = np.sqrt(np.minimum(1, max(0.008, 1.5 * np.median(residual)) / np.maximum(residual, 1e-7)))
    if coef[0] <= 0 or np.median(residual) > 0.10:
        return None
    depth = np.float32(1 / np.maximum(coef[0] / raw + coef[1], 1e-5))
    # Correct smooth local scale drift without flattening unmodelled objects.
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.log(shell_z / depth)
    weight = (anchors & (np.abs(correction) < 0.065)).astype(np.float32)
    sigma = max(8, w * 0.032)
    smooth = cv2.GaussianBlur(np.where(weight > 0, correction, 0).astype(np.float32), (0, 0), sigma)
    mass = cv2.GaussianBlur(weight, (0, 0), sigma)
    depth *= np.exp(smooth / np.maximum(mass, 0.05))
    world = rays * depth[:, :, None]
    height = world[:, :, 1] + cam.eye[1]
    return SurfaceDepth(cam, depth, normals_for(world), height, round(float(np.median(residual) * 100), 2), accepted_region)
