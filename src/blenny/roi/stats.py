"""Per-ROI statistics: polygon → mask → area + colour measurements.

Pure numpy/skimage functions (no Streamlit) so they are unit-testable and
reusable outside the GUI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import color
from skimage.draw import polygon


def scale_points(points: list[list[float]], scale_x: float, scale_y: float) -> list[list[float]]:
    """Scale polygon vertices by ``(scale_x, scale_y)``.

    Used to map display-space canvas coordinates back to the full-resolution
    image before measurement.
    """
    return [[float(x) * scale_x, float(y) * scale_y] for x, y in points]


def polygon_mask(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    """Return a boolean mask of the polygon's interior in an ``(h, w)`` frame.

    Points are ``[x, y]`` pairs (matching the canvas coordinate order).
    Polygons with fewer than 3 vertices produce an all-False mask.
    """
    mask = np.zeros(shape, dtype=bool)
    if len(points) < 3:
        return mask
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    rr, cc = polygon(ys, xs, shape=shape)
    mask[rr, cc] = True
    return mask


def compute_roi_stats(
    image_rgb: np.ndarray, mask: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Compute area + colour statistics for the pixels under ``mask``.

    Returns ``(stats, rgb_pixels, hsv_pixels)`` where ``stats`` is a dict of
    scalar measurements and the two arrays are the raw per-pixel RGB (uint8,
    shape ``(N, 3)``) and HSV (float 0..1, shape ``(N, 3)``) values — the
    granular data used later for histograms / distributions.
    """
    h, w = mask.shape
    area_px = int(mask.sum())
    area_pct = 100.0 * area_px / (h * w) if h * w else 0.0

    rgb_px = np.asarray(image_rgb)[mask]
    hsv_img = color.rgb2hsv(image_rgb)
    hsv_px = hsv_img[mask]

    def _mean(arr: np.ndarray, channel: int) -> float:
        return float(arr[:, channel].mean()) if arr.shape[0] else 0.0

    stats: dict[str, Any] = {
        "area_px": area_px,
        "area_pct": round(area_pct, 4),
        "n_pixels": int(rgb_px.shape[0]),
        "mean_r": round(_mean(rgb_px, 0), 4),
        "mean_g": round(_mean(rgb_px, 1), 4),
        "mean_b": round(_mean(rgb_px, 2), 4),
        "mean_h": round(_mean(hsv_px, 0), 4),
        "mean_s": round(_mean(hsv_px, 1), 4),
        "mean_v": round(_mean(hsv_px, 2), 4),
    }
    return stats, rgb_px, hsv_px
