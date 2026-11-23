"""Fit calibrated cameras and recover horizontal eaves in a shared Y-up world."""
import math
from dataclasses import dataclass

import cv2
import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def rounded(value):
    return round(float(value), 4)


@dataclass
class Camera:
    source: str
    size: np.ndarray
    k: np.ndarray
    r: np.ndarray
    t: np.ndarray

    @property
    def eye(self):
        return -self.r.T @ self.t

    def pixels(self, points):
        cv = np.atleast_2d(points) @ self.r.T + self.t
        screen = cv @ self.k.T
        return screen[:, :2] / screen[:, 2:], cv[:, 2]

    def rays(self, uv):
        pixels = np.atleast_2d(uv) * self.size
        hom = np.column_stack((pixels, np.ones(len(pixels))))
        directions = hom @ np.linalg.inv(self.k).T @ self.r
        return directions / np.linalg.norm(directions, axis=1)[:, None]

    def plane(self, uv, height):
        rays = self.rays(uv)
        if np.any(np.abs(rays[:, 1]) < 1e-5):
            raise ValueError(f"{self.source} 有点太接近地平线")
        distance = (height - self.eye[1]) / rays[:, 1]
        if np.any(distance <= 0) or np.any(distance > 30000):
            raise ValueError(f"{self.source} 射线与目标平面没有可用交点，请核对相机和标点")
        return self.eye + distance[:, None] * rays


def fit_camera(view, size):
    size = np.array(size, dtype=float)
    fy = size[1] / (2 * math.tan(math.radians(view["fovY"]) / 2))
    cx, cy = np.array(view.get("principal", [.5, .5])) * size
    k = np.array([[fy * view.get("focalRatio", 1), 0, cx], [0, fy, cy], [0, 0, 1]])
    controls = [p for p in view["controls"] if not p.get("check")]
    checks = [p for p in view["controls"] if p.get("check")]
    source = view["source"]
    if "approxPose" in view:
        pose = view["approxPose"]
        pitch = math.radians(pose["pitchDeg"])
        roll = math.radians(pose.get("rollDeg", 0))
        forward = np.array([0, -math.sin(pitch), -math.cos(pitch)])
        right = np.array([1., 0, 0])
        down = np.cross(forward, right)
        rotation = np.stack([math.cos(roll)*right - math.sin(roll)*down, math.sin(roll)*right + math.cos(roll)*down, forward])
        eye = np.array([0, pose["heightM"], 0])
        return Camera(source, size, k, rotation, -rotation @ eye), {
            "source": source, "cameraMethod": "estimated", "fitPoints": 0, "fitRmsPx": None,
            "checkPoints": 0, "checkRmsPx": None, "checkRmsM": None}
    if len(controls) < 4:
        raise ValueError(f"{source} 至少需要 4 个参与拟合的地面控制点")

    def coordinates(items):
        ground = np.array([p["xz"] for p in items], float)
        return np.column_stack((ground[:, 0], np.zeros(len(items)), ground[:, 1])), np.array([p["uv"] for p in items]) * size

    world, pixels = coordinates(controls)
    for coords in (world[:, [0, 2]], pixels):
        spread = np.linalg.svd(coords - coords.mean(axis=0), compute_uv=False)
        if len(np.unique(coords, axis=0)) != len(coords) or spread[-1] < max(1e-6, spread[0] * .003):
            raise ValueError(f"{source} 控制点重复或接近共线，请把控制点分布到地面不同区域")
    try:
        solved = cv2.solvePnPGeneric(world, pixels, k, None, flags=cv2.SOLVEPNP_IPPE)
        candidates = list(zip(solved[1], solved[2]))
        ok, rv, tv = cv2.solvePnP(world, pixels, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            candidates.append((rv, tv))
    except cv2.error as error:
        raise ValueError(f"{source} 相机标定失败，请核对地面控制点") from error
    fits = []
    for rv, tv in candidates:
        if not np.isfinite(rv).all() or not np.isfinite(tv).all():
            continue
        rv, tv = cv2.solvePnPRefineLM(world, pixels, k, None, rv.copy(), tv.copy())
        cam = Camera(source, size, k, cv2.Rodrigues(rv)[0], tv.ravel())
        image, depth = cam.pixels(world)
        if cam.eye[1] > .1 and (depth > .01).all():
            fits.append((rms(np.linalg.norm(image - pixels, axis=1)), cam))
    if not fits:
        raise ValueError(f"{source} 无法在地面上方解出相机，请核对 X/Z 方向和控制点顺序")
    error, cam = min(fits, key=lambda fit: fit[0])
    if error > max(5, np.linalg.norm(size) * .009):
        raise ValueError(f"{source} 控制点误差 {error:.1f} px 过大，请核对视场角和坐标")
    report = {"source": source, "cameraMethod": "ground_controls", "fitPoints": len(controls), "fitRmsPx": rounded(error), "checkPoints": len(checks),
              "checkRmsPx": None, "checkRmsM": None}
    if checks:
        truth, marked = coordinates(checks)
        image, depth = cam.pixels(truth)
        if (depth <= 0).any():
            raise ValueError(f"{source} 检查点位于相机后方，请核对坐标")
        report.update(checkRmsPx=rounded(rms(np.linalg.norm(image - marked, axis=1))),
                      checkRmsM=rounded(rms(np.linalg.norm(cam.plane(marked / size, 0) - truth, axis=1))))
    return cam, report


def triangulate(eyes, rays):
    angles = [math.degrees(math.acos(np.clip(abs(a @ b), 0, 1))) for i, a in enumerate(rays) for b in rays[i + 1:]]
    best = max(angles)
    if best < 2:
        raise ValueError(f"视线交角仅 {best:.2f}°，请增加有侧向位移的参考视图")
    perpendicular = np.eye(3) - rays[:, :, None] * rays[:, None, :]
    point = np.linalg.solve(perpendicular.sum(0), np.einsum("nij,nj->i", perpendicular, eyes))
    if np.any(np.sum((point - eyes) * rays, axis=1) <= 0):
        raise ValueError("多视图对应点在相机后方，请核对屋角顺序")
    return point, best


def reconstruct_building(building, cameras):
    observations = building["observations"]
    sources = [o["source"] for o in observations]
    if len(set(sources)) != len(sources) or any(s not in cameras for s in sources):
        raise ValueError("观测的图片编号重复或未标定")
    cams = [cameras[s] for s in sources]
    polys = [np.array(o["roof"], float) for o in observations]
    n = len(polys[0])
    if any(len(p) != n or not cv2.isContourConvex(p.astype(np.float32)) for p in polys):
        raise ValueError("屋檐需要同序、等点数的凸多边形，复杂建筑可分成多个体块")
    if building.get("ridgeRiseM", 0) and n != 4:
        raise ValueError("双坡屋脊需要四个檐角")
    angle = None
    if len(cams) > 1:
        eyes = np.array([cam.eye for cam in cams])
        directions = np.array([cam.rays(uv) for cam, uv in zip(cams, polys)])
        fits = [triangulate(eyes, directions[:, j]) for j in range(n)]
        height = np.median([p[1] for p, _ in fits])
        angle = min(a for _, a in fits)
        method = "multiview"
    elif "base" in observations[0]:
        cam = cams[0]
        base = cam.plane(observations[0]["base"], 0)[0]
        ray = cam.rays(polys[0][0])[0]
        horizontal = ray[[0, 2]]
        if horizontal @ horizontal < 1e-5:
            raise ValueError("视图过于接近正俯视，请输入檐高或增加侧视图")
        t = (base - cam.eye)[[0, 2]] @ horizontal / (horizontal @ horizontal)
        height = (cam.eye + t * ray)[1]
        base[1] = height
        if t <= 0 or np.linalg.norm(cam.pixels(base)[0][0] - polys[0][0] * cam.size) > max(4, np.linalg.norm(cam.size) * .006):
            raise ValueError("墙脚和第一个檐角不在同一竖线上")
        method = "vertical"
    else:
        if "heightM" not in building:
            raise ValueError("单图需要檐高或第一个檐角正下方的墙脚")
        height = building["heightM"]
        method = building.get("heightSource", "estimated")
    if not .1 < height <= 300:
        raise ValueError(f"求得的檐高 {height:.2f} m 超出范围，请核对观测和尺度")
    # Refit each corner on the common eave plane against all camera rays.
    seeds = np.array([cam.plane(uv, height) for cam, uv in zip(cams, polys)])
    roof = seeds[0].copy()
    if len(cams) > 1:
        for j in range(n):
            mats = np.eye(3) - directions[:, j, :, None] * directions[:, j, None, :]
            lhs = mats.sum(0)
            rhs = np.einsum("nij,nj->i", mats, eyes) - lhs[:, 1] * height
            roof[j, [0, 2]] = np.linalg.solve(lhs[np.ix_([0, 2], [0, 2])], rhs[[0, 2]])
    footprint = np.ascontiguousarray(roof[:, [0, 2]], dtype=np.float32)
    if not cv2.isContourConvex(footprint) or cv2.contourArea(footprint) < .04:
        raise ValueError("重建轮廓退化或自交，请核对屋角")
    residuals = []
    for cam, uv in zip(cams, polys):
        projected, depths = cam.pixels(roof)
        distances = np.linalg.norm(projected - uv * cam.size, axis=1)
        if (depths <= 0).any() or distances.max() > max(5, np.linalg.norm(cam.size) * .009):
            raise ValueError(f"{cam.source} 屋角重投影偏差过大，请核对同名角点和屋檐平面")
        residuals.extend(distances)
    return roof, {"id": building["id"], "heightM": rounded(height), "heightMethod": method,
                  "views": len(cams), "minRayAngleDeg": rounded(angle) if angle is not None else None,
                  "roofRmsPx": rounded(rms(residuals)), "areaM2": rounded(cv2.contourArea(footprint))}
