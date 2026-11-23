"""Estimate continuous material colour on the unobserved side of a fused height surface."""

import cv2
import numpy as np


def unseen_texture(rgb, height, step):
    """Work in the XZ grid, at metre scales. The caller adds observed image detail afterwards."""
    source = np.ascontiguousarray(np.clip(rgb, 0, 255), dtype=np.float32)
    h = np.asarray(height, np.float32)
    sx, sz = np.maximum(step, 1e-3)

    def blur(image, metres):
        sigma_x = np.clip(metres / sx, 0.45, max(1, h.shape[1] / 3))
        sigma_z = np.clip(metres / sz, 0.45, max(1, h.shape[0] / 3))
        return cv2.GaussianBlur(image, (0, 0), sigma_x, sigmaY=sigma_z, borderType=cv2.BORDER_REPLICATE)

    def regional_color(weight, metres, fallback):
        weight = np.float32(weight)
        support = blur(weight, metres)
        # A few isolated pixels cannot lend their colour to an entire unseen wall.
        residual = np.maximum(0.12 - support, 0)
        colors = blur(source * weight[:, :, None], metres)
        return (colors + residual[:, :, None] * fallback) / np.maximum(0.12, support)[:, :, None]

    level = blur(h, 0.42)
    slope_z, slope_x = np.gradient(level, sz, sx)
    upward = 1 / np.sqrt(1 + slope_x**2 + slope_z**2)
    wall = np.clip((0.80 - upward) / 0.42, 0, 1)
    raised = np.clip((h - 0.45) / 1.05, 0, 1)
    red, green, blue = source.transpose(2, 0, 1)
    luma = 0.27 * red + 0.57 * green + 0.16 * blue
    # Yellow-grey masonry is common here. Require coherent green chroma and
    # dark foliage values, allowing browner pixels inside a crown to inherit it.
    green_cast = (green - 0.90 * red - 0.10 * blue) / (luma + 12)
    leaf_pixels = np.clip((green_cast - 0.012) * 16, 0, 1)
    leaf_pixels *= np.clip(((green - blue) / (luma + 12) - 0.085) * 7, 0, 1) * np.clip(
        (155 - luma) / 53, 0, 1
    )
    canopy = np.clip((blur(leaf_pixels, 1.05) - 0.18) / 0.45, 0, 1)
    canopy = np.maximum(canopy, 0.55 * blur(leaf_pixels, 0.42))

    base = blur(cv2.medianBlur(source, 3), 0.75)
    # Robust local wall albedo excludes dark window interiors and roofs.
    # Across a row of buildings it still follows changes in plaster/brick colour.
    plausible_wall = (luma > 56) & (luma < 220) & (red > 0.85 * blue) & (green > 0.82 * blue)
    wall_evidence = (1 - leaf_pixels) * raised * wall * plausible_wall
    good = wall_evidence > 0.4
    default_wall = np.median(source[good][::3], axis=0) if np.count_nonzero(good) >= 24 else base
    masonry = regional_color(wall_evidence, 3.3, default_wall)
    roof = regional_color((1 - canopy) * raised * (1 - wall), 1.55, base)
    leaves = regional_color(leaf_pixels, 1.12, base)

    # Small material variations follow world units and are shared across
    # neighbouring triangles. They contain no directional window or door motifs.
    rng = np.random.default_rng(4729)
    noise = rng.standard_normal(h.shape, dtype=np.float32)

    def grain(metres):
        kernel_area = max(1.0, 4 * np.pi * (metres / sx) * (metres / sz))
        return np.clip(blur(noise, metres) * np.sqrt(kernel_area), -2, 2)

    roof += (1.9 * grain(0.52))[:, :, None]
    masonry += (1.3 * grain(0.70))[:, :, None]
    leaves += (4.8 * grain(0.50) + 2.5 * grain(1.35))[:, :, None]
    surface = roof * (1 - wall[:, :, None]) + masonry * wall[:, :, None]
    surface = surface * (1 - canopy[:, :, None]) + leaves * canopy[:, :, None]
    result = source * (1 - raised[:, :, None]) + surface * raised[:, :, None]
    return np.uint8(np.clip(np.rint(result), 0, 255))
