"""Integration tests for grid-mode multi-plate mapping."""

from __future__ import annotations

import numpy as np

from blenny.modules.detect_multi_plate import MultiPlateDetector
from blenny.pipeline.context import ImageData
from blenny.testing.synthetic import make_synthetic_plate


def _compose_grid(rows: int, cols: int, present: list[tuple[int, int]], cell: int = 500) -> np.ndarray:
    """Build a rows x cols image with a synthetic plate in each present cell."""
    bg = np.full((rows * cell, cols * cell, 3), 25, dtype=np.uint8)
    for seed, (r, c) in enumerate(present):
        plate = make_synthetic_plate(n_colonies=30, image_size=(cell, cell), seed=seed)
        bg[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell, :] = plate.image
    return bg


def _run(grid: list[int], image: np.ndarray, *, labels: list[list[str]] | None = None) -> ImageData:
    data = ImageData(source="grid")
    data.image = image
    data.original_image = image
    MultiPlateDetector(grid=grid, labels=labels).process(image, data)
    return data


def test_maps_plates_and_flags_empty_slot():
    # 2x2 grid, bottom-right empty.
    img = _compose_grid(2, 2, [(0, 0), (0, 1), (1, 0)])
    data = _run([2, 2], img, labels=[["A1", "A2"], ["B1", "B2"]])
    rows = data.metadata["rois"]
    assert [r["label"] for r in rows] == ["A1", "A2", "B1"]
    assert [r["grid_pos"] for r in rows] == [(0, 0), (0, 1), (1, 0)]
    codes = [f.code for f in data.quality_flags]
    assert codes.count("plate_not_found") == 1
    assert "B2" in next(f.message for f in data.quality_flags if f.code == "plate_not_found")
    assert data.metadata.get("grid_fit_failed") is None


def test_5_plates_in_3x2_reports_one_empty():
    img = _compose_grid(3, 2, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
    data = _run([3, 2], img)
    assert len(data.metadata["rois"]) == 5
    # The empty bottom-right slot is flagged.
    assert any(f.code == "plate_not_found" for f in data.quality_flags)


def test_too_few_plates_falls_back_to_auto_labels():
    # One plate in a 2x2 grid cannot define a grid -> fallback + error flag.
    img = _compose_grid(2, 2, [(0, 0)])
    data = _run([2, 2], img)
    assert data.metadata["grid_fit_failed"] is True
    assert [r["label"] for r in data.metadata["rois"]] == ["1"]
    codes = [f.code for f in data.quality_flags]
    assert "plate_grid_mapping_failed" in codes


def test_single_plate_1x1_grid_maps_to_the_slot():
    img = _compose_grid(1, 1, [(0, 0)])
    data = _run([1, 1], img)
    assert data.metadata.get("grid_fit_failed") is None
    rows = data.metadata["rois"]
    assert len(rows) == 1
    assert rows[0]["label"] == "1"
