"""Rectify photographed wall materials for the unobserved sides of the same settlement."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class Patch:
    building: int
    center: np.ndarray
    width: float
    height: float
    quality: float
    image: np.ndarray

    def __post_init__(self):
        self.tone = material_tone(self.image[::3, ::3].reshape(-1, 3))


def material_tone(pixels):
    """Separate brightness from chroma so blue sheet, clay tile and cement stay distinct."""
    red, green, blue = np.median(pixels, axis=0)
    return np.array([0.55 * (0.27 * red + 0.57 * green + 0.16 * blue), red - green, blue - green])


def observed_material(surfaces, cameras, images, buffers, priors):
    tones = {}
    for index, shell in enumerate(surfaces):
        samples = {}
        for face, normal in shell:
            if normal[1] < -0.01:
                continue
            kind = "roof" if normal[1] > 0.35 else "wall"
            center = face.mean(0)
            # Stay inside the face. Traced parapets and shadows at its perimeter
            # should not control the inferred roof covering or facade finish.
            probes = np.vstack([center + (face - center) * r for r in (0.2, 0.36, 0.55, 0.70)])
            for key, cam in cameras.items():
                towards = cam.eye - center
                if normal @ towards < 0.28 * np.linalg.norm(towards):
                    continue
                uv, z = cam.pixels(probes)
                valid = (z > 0.1) & (uv > 2).all(1) & (uv < cam.size - 3).all(1)
                buf, scale = buffers[key]
                xy = np.clip(np.rint(uv * scale).astype(int), 0, np.array(buf.shape[::-1]) - 1)
                valid &= z < buf[xy[:, 1], xy[:, 0]] + np.maximum(0.18, 0.008 * z)
                if key in priors:
                    valid &= priors[key].check(uv, z, normal, cam.size)
                ij = np.rint(uv[valid]).astype(int)
                if len(ij) >= 4:
                    samples.setdefault(kind, []).append(images[key][ij[:, 1], ij[:, 0]])
        for kind, colors in samples.items():
            tones[(index, kind)] = material_tone(np.concatenate(colors))
    return tones


class FacadeMaterials:
    def __init__(self, surfaces, cameras, images, buffers, priors, supplements, marks, building_ids=None):
        self.patches = []
        self.walls = []
        self.roofs = []
        self.used = 0
        self.house_tones = {}
        self.reviewed_materials = {}
        self.reviewed_wall_hits = set()
        self.wall_sources = {}
        self.reviewed_roofs = {}
        self.roof_sources = {}
        self.reference_specs = {}
        self.reviewed_yards = []
        self.paired_yards = [mark for mark in marks.values() if mark.paired_ground is not None]
        if building_ids is not None:
            identifiers = {name: index for index, name in enumerate(building_ids)}
            references = {name for review in marks.values() for name in review.data.get("materialReferences", [])}
            for name in sorted(references):
                self.load_reviewed_walls(name, identifiers, surfaces)
        for review in marks.values():
            for name, library in (("walls", self.walls), ("roofs", self.roofs)):
                for item in review.data.get(name, []):
                    pw, ph = item["width"], item["height"]
                    w, h = np.clip(np.rint([pw * 24, ph * 24]), 32, 512).astype(int)
                    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                    transform = cv2.getPerspectiveTransform(np.float32(review.pixels(item["quad"])), dst)
                    image = cv2.warpPerspective(review.image, transform, (w, h))
                    library.append(Patch(-1, None, pw, ph, 1, image))
        if self.walls or self.roofs:
            self.house_tones = observed_material(surfaces, cameras, images, buffers, priors)
        gx, gy = np.meshgrid(np.linspace(0.025, 0.975, 36), np.linspace(0.045, 0.955, 24))
        views = [(c, images[k], None) for k, c in cameras.items()]
        views += [(v.camera, v.image, v) for v in supplements if v.surface_depth is not None]
        for b, shell in enumerate(surfaces):
            for face, normal in shell:
                if abs(normal[1]) > 0.1:
                    continue
                axis = np.cross([0.0, 1.0, 0.0], normal)
                axis /= np.linalg.norm(axis)
                width, height = np.ptp(face @ axis), np.ptp(face[:, 1])
                if min(width, height) < 2:
                    continue
                origin = face[np.argmin(face @ axis)].copy()
                origin[1] = face[:, 1].max()

                def locate(u, v):
                    return (
                        origin
                        + (u * width)[..., None] * axis
                        - (v * height)[..., None] * np.array([0, 1, 0])
                    )

                world = locate(gx, gy)
                best = None
                for cam, rgb, novel in views:
                    delta = cam.eye - face.mean(0)
                    cosine = normal @ delta / np.linalg.norm(delta)
                    if cosine < 0.40:
                        continue
                    uv, z = cam.pixels(world.reshape(-1, 3))
                    valid = (uv >= 2).all(1) & (uv < cam.size - 2).all(1) & (z > 0.2)
                    if novel is None:
                        buf, scale = buffers[cam.source]
                        xy = np.clip((uv * scale).astype(int), 0, np.array(buf.shape[::-1]) - 1)
                        valid &= z <= buf[xy[:, 1], xy[:, 0]] + np.maximum(0.16, 0.006 * z)
                    else:
                        xy = np.clip(np.rint(uv).astype(int), 0, cam.size.astype(int) - 1)
                        valid &= np.abs(z - novel.depth[xy[:, 1], xy[:, 0]]) < np.maximum(0.18, 0.004 * z)
                    if cam.source in priors:
                        valid &= priors[cam.source].check(uv, z, normal, cam.size)
                    valid = valid.reshape(gx.shape)
                    for y0, y1 in ((0, 24), (1, 20), (5, 23)):
                        for x0, x1 in ((0, 36), (0, 25), (11, 36), (6, 30)):
                            if valid[y0:y1, x0:x1].mean() < 0.88:
                                continue
                            u0, u1 = gx[0, [x0, x1 - 1]]
                            v0, v1 = gy[[y0, y1 - 1], 0]
                            pw, ph = (u1 - u0) * width, (v1 - v0) * height
                            corners = locate(np.array([u0, u1, u1, u0]), np.array([v0, v0, v1, v1]))
                            src = cam.pixels(corners)[0].astype(np.float32)
                            area = abs(cv2.contourArea(src))
                            if area < 180 or min(pw, ph) < 1.7:
                                continue
                            quality = cosine * math.sqrt(area) / (1 + 0.22 * (novel is not None))
                            if best is None or quality > best[0]:
                                best = (quality, src, rgb, pw, ph)
                if best is not None:
                    quality, src, rgb, pw, ph = best
                    w, h = np.clip(np.rint([pw * 17, ph * 17]), 32, 384).astype(int)
                    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
                    tile = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h))
                    self.patches.append(Patch(b, face.mean(0), pw, ph, quality, tile))

    def load_reviewed_walls(self, name, identifiers, surfaces):
        # Material observations have local surface correspondences. Importing
        # them does not register the entire photo as a scene camera.
        if not isinstance(name, str) or not name.replace("-", "").isalnum():
            raise ValueError("Invalid wall material reference")
        folder = Path(__file__).resolve().parent / "sample-views"
        spec = json.loads((folder / (name + "-materials.json")).read_text(encoding="utf8"))
        self.reference_specs[name] = spec
        photo = folder / (name + ".jpg")
        if hashlib.sha256(photo.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError("Wall material reference image does not match its annotations")
        with Image.open(photo) as im:
            rgb = np.array(im.convert("RGB"))
        scale = np.array(rgb.shape[1::-1]) / spec["size"]
        def image_quad(item):
            q = np.asarray(item["quad"], dtype=np.float32)
            if q.shape != (4, 2):
                return None
            q = np.float32(q * scale)
            if (not np.isfinite(q).all() or np.any(q < 0) or np.any(q > np.array(rgb.shape[1::-1]) - 1)
                    or not cv2.isContourConvex(q) or abs(cv2.contourArea(q)) < 8):
                return None
            return q

        for item in spec.get("walls", []):
            building = identifiers.get(item["building"])
            edge = item["edge"]
            if building is None or not 0 <= edge < len(surfaces[building][0][0]):
                continue
            _, normal = surfaces[building][edge + 1]
            if abs(normal[1]) >= 0.1:
                continue
            src = image_quad(item)
            if src is None:
                continue
            sizes = np.linalg.norm(src - np.roll(src, -1, axis=0), axis=1)
            w, h = np.clip(np.rint([(sizes[0]+sizes[2])/2, (sizes[1]+sizes[3])/2]), 8, 512).astype(int)
            dst = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            tile = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h))
            area = abs(cv2.contourArea(src))
            self.reviewed_materials.setdefault(building, []).append((normal, tile, edge, name, area))
        for item in spec.get("roofs", []):
            building = identifiers.get(item["building"])
            quad = image_quad(item)
            if building is None or quad is None:
                continue
            area = abs(cv2.contourArea(quad))
            previous = self.reviewed_roofs.get(building)
            if previous is None or previous[0] < area:
                self.reviewed_roofs[building] = (area, name, rgb, quad)
        for item in spec.get("yards", []):
            if item.get("verified") is not True:
                continue
            world = np.float32(item["groundQuad"])
            quad = image_quad(item)
            if (quad is not None and world.shape == (4, 2) and np.isfinite(world).all()
                    and cv2.isContourConvex(world) and abs(cv2.contourArea(world)) > .05):
                self.reviewed_yards.append((name, rgb, quad, world))

    def reviewed_roof_uv(self, building_index, canonical_roof, vertices):
        observation = self.reviewed_roofs.get(building_index)
        if observation is None or np.shape(canonical_roof) != (4, 3):
            return None
        _, name, rgb, quad = observation
        roof = np.ascontiguousarray(np.asarray(canonical_roof)[:, [0, 2]], dtype=np.float32)
        if not cv2.isContourConvex(roof) or abs(cv2.contourArea(roof)) < .05:
            return None
        transform = cv2.getPerspectiveTransform(roof, quad)
        xz = np.asarray(vertices, np.float32)[:, [0, 2]].reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(xz, transform).reshape(-1, 2)
        # Reject a mismatched footprint rather than sampling the adjoining yard.
        if (not np.isfinite(projected).all()
                or any(cv2.pointPolygonTest(quad, (float(x), float(y)), True) < -1.5
                       for x, y in projected)):
            return None
        return name, rgb, (projected + .5) / np.array(rgb.shape[1::-1])

    def reviewed_wall(self, building_index, normal, u, v):
        if abs(normal[1]) >= 0.1:
            return None
        matches = [entry for entry in self.reviewed_materials.get(building_index, ())
                   if np.dot(entry[0], normal) >= .90]
        if not matches:
            return None
        _, tile, edge, source, _ = max(matches, key=lambda entry: entry[4])
        self.reviewed_wall_hits.add((building_index, edge))
        self.wall_sources[(building_index, edge)] = source
        h, w = tile.shape[:2]
        return cv2.remap(tile, np.float32(u * (w-1)), np.float32(v * (h-1)), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def fill(self, face, normal, building, u, v, fallback):
        reviewed = self.reviewed_wall(building, normal, u, v)
        if reviewed is not None:
            return reviewed
        if not self.patches and not self.walls and not self.roofs:
            return fallback
        axis = np.cross([0.0, 1.0, 0.0], normal)
        wall = abs(normal[1]) < 0.1
        if not wall:
            axis = face[1] - face[0]
            axis = axis / np.linalg.norm(axis)
        width, height = max(0.1, np.ptp(face @ axis)), max(0.1, np.ptp(face @ np.cross(axis, normal)))
        center = face.mean(0)

        def cost(p):
            return (
                1.9 * abs(math.log(height / p.height))
                + 0.75 * abs(math.log(width / p.width))
                + 0.013 * np.linalg.norm(center - p.center)
                + 9 / max(5, p.quality)
                - 0.8 * (p.building == building)
            )

        curated = self.walls if wall else self.roofs
        if curated:
            own = [
                p
                for p in self.patches
                if wall
                and p.building == building
                and p.quality >= 15
                and 0.58 < height / p.height < 1.55
                and 0.55 < width / p.width < 1.9
            ]
            target = self.house_tones.get((building, "wall" if wall else "roof"))

            def score(p):
                shape = abs(math.log(height / p.height)) + 0.45 * abs(math.log(width / p.width))
                tint = np.linalg.norm(p.tone - target) / 23 if target is not None else 0
                return shape + tint - 0.75 * (p.building == building)

            ranked = sorted(curated + own, key=score)
            side = int(round((math.atan2(normal[0], normal[2]) + math.pi) * 13)) if wall else 0
            suitable = [p for p in ranked[:4] if score(p) < score(ranked[0]) + 0.36]
            patch = suitable[(building * 7 + side) % len(suitable)]
        elif wall and self.patches:
            patch = min(self.patches, key=cost)
        else:
            return fallback
        # Keep window bays close to their original scale on long elevations.
        repeat = max(1, round(width / patch.width)) if width > 1.65 * patch.width else 1
        along = 1 - np.abs(1 - np.mod(u * repeat, 2))
        h, w = patch.image.shape[:2]
        color = cv2.remap(
            patch.image, np.float32(along * (w - 1)), np.float32(v * (h - 1)), cv2.INTER_LINEAR
        )
        return color
