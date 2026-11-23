"""Bake local photographic ground materials in ground coordinates, including unobserved courtyards."""

import cv2
import numpy as np


class GroundMaterials:
    def __init__(self, marks):
        self.tiles = {}
        for mark in marks.values():
            for item in mark.data.get("terrainMaterials", []):
                quad = np.float32(mark.pixels(item["quad"]))
                dimensions = np.asarray(item["sizeM"])
                width, height = np.clip(np.rint(dimensions * 18), 32, 384).astype(int)
                target = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
                warp = cv2.getPerspectiveTransform(quad, target)
                patch = cv2.warpPerspective(
                    mark.image, warp, (width, height), borderMode=cv2.BORDER_REPLICATE
                )
                self.tiles.setdefault(item["kind"], []).append((patch, dimensions))

    def sample(self, kind, u, v):
        choices = self.tiles.get(kind)
        if not choices:
            return None
        result = np.empty((*u.shape, 3), np.float32)
        for start in range(0, len(u), 96):
            rows = slice(start, start + 96)
            result[rows] = self._sample_rows(choices, u[rows], v[rows])
        return result

    @staticmethod
    def _sample_rows(choices, u, v):
        # Randomly translate the source swatch at each vertex of a triangular
        # lattice. Barycentric blending between three translates keeps boundaries
        # continuous while preserving crop rows and pavement orientation.
        result = None
        for index, (patch, (width, height)) in enumerate(choices):
            h, w = patch.shape[:2]
            x = (u + index * 3.7 + 0.11 * np.sin(0.13 * v)) / width
            y = (v - index * 2.3 + 0.09 * np.cos(0.23 * u)) / height
            gx, gy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
            fx, fy = np.float32(x - gx), np.float32(y - gy)
            upper = (fx + fy > 1).astype(np.int32)
            color = np.zeros((*u.shape, 3), np.float32)
            total = np.zeros(u.shape, np.float32)
            for dx, dy, weight in (
                (upper, upper, np.abs(1 - fx - fy)),
                (1, 0, np.minimum(fx, 1 - fy)),
                (0, 1, np.minimum(fy, 1 - fx)),
            ):
                seed = (gx + dx).astype(np.uint32) * 0x61C88647 ^ (gy + dy).astype(np.uint32) * 0x5BD1E995
                seed = (seed ^ (seed >> 15)) * 0x45D9F3B
                offset_x = (seed & 65535).astype(np.float32) / 32768
                offset_y = (seed >> 16).astype(np.float32) / 32768
                px = (1 - np.abs(1 - np.mod(x + offset_x, 2))) * (w - 1)
                py = (1 - np.abs(1 - np.mod(y + offset_y, 2))) * (h - 1)
                local = cv2.remap(patch, np.float32(px), np.float32(py), cv2.INTER_LINEAR)
                weight = weight**2
                color += local * weight[:, :, None]
                total += weight
            color /= total[:, :, None]
            if result is None:
                result = color
            else:
                blend = (
                    0.52 + 0.27 * np.sin(0.32 * u + 0.14 * v + index) + 0.18 * np.cos(0.18 * u - 0.29 * v)
                )[:, :, None]
                result = result * (1 - blend) + color * blend
        return result

    def settlement(self, u, v, clearance, roads, guide):
        yard = self.sample("yard", u, v)
        soil = self.sample("soil", u + 0.27 * v, v)
        if yard is None or soil is None:
            return guide
        # Paved forecourts near buildings and along shopfronts, softer ground
        # behind the street block, and narrow worn paths connecting the two.
        verge_distance = np.full(u.shape, np.inf, np.float32)
        for road, distance, _ in roads:
            verge_distance = np.minimum(verge_distance, distance - road["widthM"] / 2)
        near_street = np.clip(1 - (verge_distance - 1.3) / 4.5, 0, 1)
        near_wall = np.clip(1 - (clearance - 0.6) / 2.8, 0, 1)
        path = np.abs(np.mod(u + 1.2 * np.sin(v * 0.08), 13.7) - 6.85)
        paving = np.maximum(
            near_street, np.maximum(near_wall, 0.68 * np.clip(1 - (path - 0.55) / 0.65, 0, 1))
        )
        dirt = soil
        grass = self.sample("weeds", u, v)
        if grass is not None:
            growth = np.clip(
                0.35 + 0.65 * np.sin(u * 0.48 + np.cos(v * 0.38)) * np.cos(v * 0.29 - u * 0.16), 0, 1
            )
            growth *= np.clip((clearance - 1.2) / 3, 0, 1) * np.clip((verge_distance - 1.5) / 3, 0, 1)
            dirt = dirt * (1 - growth[:, :, None]) + grass * growth[:, :, None]
        return dirt * (1 - paving[:, :, None]) + yard * paving[:, :, None]

    def farmland(self, u, v, family, original):
        result = original
        for kind, types in (("crops", (0, 2, 4, 6)), ("stubble", (1, 3, 5))):
            tile = self.sample(kind, u, v)
            if tile is None:
                continue
            # Retain broad variations between distinct parcels.
            shade = (0.80 + 0.065 * family)[:, :, None]
            result = np.where(np.isin(family, types)[:, :, None], tile * shade, result)
        return result
