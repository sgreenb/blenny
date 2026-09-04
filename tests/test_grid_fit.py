"""Unit tests for the grid-fitting geometry in blenny.modules._grid_fit."""

from __future__ import annotations

import math
import random

import pytest

from blenny.modules._grid_fit import fit_grid_to_centers


def _slot(r: int, c: int, *, x0: float = 100, y0: float = 100, dx: float = 300, dy: float = 300):
    """Return the (x, y) centre of grid slot (row, col)."""
    return (x0 + c * dx, y0 + r * dy)


RADII = [100] * 8


def test_perfect_full_grid():
    pts = [_slot(0, 1), _slot(1, 1), _slot(0, 0), _slot(1, 0)]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert res.ok
    assert res.empty_slots == []
    assert set(res.assignments.values()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(r < 1.0 for r in res.residuals)


def test_missing_slot_is_reported_empty():
    # 2x2 with the bottom-right cell empty -> one empty slot, no phantom.
    pts = [_slot(0, 0), _slot(0, 1), _slot(1, 0)]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert res.ok
    assert res.empty_slots == [(1, 1)]
    assert len(res.assignments) == 3


def test_3x2_five_plates_one_empty():
    pts = [_slot(0, 0), _slot(0, 1), _slot(1, 0), _slot(1, 1), _slot(2, 0)]
    res = fit_grid_to_centers(pts, 3, 2, radii=RADII)
    assert res.ok
    assert res.empty_slots == [(2, 1)]


def test_accepts_placement_jitter():
    random.seed(0)
    pts = [
        (x + random.uniform(-18, 18), y + random.uniform(-18, 18))
        for x, y in [_slot(0, 0), _slot(0, 1), _slot(1, 0)]
    ]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert res.ok
    assert set(res.assignments.values()) == {(0, 0), (0, 1), (1, 0)}
    assert res.empty_slots == [(1, 1)]


@pytest.mark.parametrize("rows,cols,pts", [
    (1, 3, [_slot(0, 0), _slot(0, 1), _slot(0, 2)]),
    (3, 1, [_slot(0, 0), _slot(1, 0), _slot(2, 0)]),
])
def test_single_row_or_column_grid(rows, cols, pts):
    res = fit_grid_to_centers(pts, rows, cols, radii=RADII)
    assert res.ok
    assert res.empty_slots == []
    assert len(res.assignments) == len(pts)


def test_too_few_plates_fails():
    res = fit_grid_to_centers([_slot(0, 0)], 2, 2, radii=RADII)
    assert not res.ok
    assert "at least" in res.error


def test_grid_smaller_than_detections_fails():
    # 5 detections into a 2x2 grid -> two must share a slot -> conflict.
    pts = [_slot(0, 0), _slot(0, 1), _slot(1, 0), _slot(1, 1), _slot(0, 0)]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert not res.ok
    assert "same slot" in res.error


def test_scattered_layout_fails():
    # Four points that are not on a lattice: median residual is too large.
    pts = [(100, 100), (300, 430), (470, 60), (150, 420)]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert not res.ok


def test_rotated_layout_fails():
    # A visibly rotated grid does not lie on the axis-aligned lattice.
    cx = cy = 250
    a = math.radians(15)

    def rot(x, y):
        return (
            cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
            cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a),
        )

    pts = [rot(*_slot(r, c)) for r in range(2) for c in range(2)]
    res = fit_grid_to_centers(pts, 2, 2, radii=RADII)
    assert not res.ok


def test_no_plates_fails():
    res = fit_grid_to_centers([], 2, 2, radii=RADII)
    assert not res.ok
    assert "no plates" in res.error
