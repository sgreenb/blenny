"""Tests for ROI mode statistics (polygon → mask → area + colour)."""

from __future__ import annotations

import numpy as np

from blenny.roi.stats import compute_roi_stats, polygon_mask, scale_points


def test_polygon_mask_rectangle_area() -> None:
    """A rectangle spanning rows/cols 3..7 in a 10x10 frame covers 5x5 px.

    skimage's polygon rasterizer includes the vertex rows/cols themselves
    (the vertices are treated as pixel centers), so [3..7] is 5 pixels wide.
    """
    points = [[3, 3], [7, 3], [7, 7], [3, 7]]  # [x, y]
    mask = polygon_mask(points, (10, 10))
    assert mask.shape == (10, 10)
    assert mask.dtype == bool
    assert int(mask.sum()) == 25
    assert mask[4, 4]  # interior
    assert not mask[1, 1]  # outside


def test_polygon_mask_fewer_than_3_vertices_is_empty() -> None:
    mask = polygon_mask([[1, 1], [2, 2]], (10, 10))
    assert not mask.any()


def test_scale_points() -> None:
    pts = [[10.0, 20.0], [30.0, 40.0]]
    scaled = scale_points(pts, 2.0, 0.5)
    assert scaled == [[20.0, 10.0], [60.0, 20.0]]
    # Input not mutated
    assert pts == [[10.0, 20.0], [30.0, 40.0]]


def test_compute_roi_stats_known_colours() -> None:
    """Stats of a 2x2 red block inside a black image."""
    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[2:4, 2:4] = [200, 40, 20]  # red-dominant block

    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True

    stats, rgb_px, hsv_px = compute_roi_stats(img, mask)

    assert stats["area_px"] == 4
    assert stats["n_pixels"] == 4
    assert stats["area_pct"] == round(100.0 * 4 / 36, 4)
    assert stats["mean_r"] == 200.0
    assert stats["mean_g"] == 40.0
    assert stats["mean_b"] == 20.0

    # HSV means: hue near red (~0), saturation near 1, value ~200/255.
    assert abs(stats["mean_h"]) < 0.05
    assert stats["mean_s"] > 0.85
    assert abs(stats["mean_v"] - 200 / 255) < 0.01

    # Granular arrays carry exactly the masked pixels.
    assert rgb_px.shape == (4, 3)
    assert rgb_px.dtype == np.uint8
    assert hsv_px.shape == (4, 3)


def test_compute_roi_stats_empty_mask() -> None:
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    mask = np.zeros((5, 5), dtype=bool)
    stats, rgb_px, hsv_px = compute_roi_stats(img, mask)
    assert stats["area_px"] == 0
    assert stats["area_pct"] == 0.0
    assert stats["n_pixels"] == 0
    assert rgb_px.shape == (0, 3)
    assert hsv_px.shape == (0, 3)
