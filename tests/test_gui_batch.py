"""Tests for the GUI batch-file helpers (batch_summary_text / batch_colonies_text).

The GUI previously only wrote batch files when more than one *input image* was
processed, so a single scan image holding several plates produced no batch at
all. These tests pin the "emit batch files whenever >1 plate is analysed"
behaviour shared by the CLI and GUI.
"""

from __future__ import annotations

import gui.app as ga

from blenny.pipeline.context import ImageData


def _multiplate_data() -> ImageData:
    """A single scan image reporting five plates (as sub_pipeline would)."""
    d = ImageData(source="/scan.png", image=None, original_image=None)
    d.metadata.update(
        {
            "rois": [{"label": str(i + 1)} for i in range(5)],
            "multi_plate_mode": True,
            "per_plate_counts": {str(i + 1): 10 + i for i in range(5)},
            "colony_count": 50,
            "stem": "scan",
        }
    )
    d.measurements = [
        {
            "plate_label": str(i + 1),
            "label": 1,
            "area_px": 10,
            "is_artifact": False,
            "source": "/scan.png",
        }
        for i in range(5)
    ]
    return d


def _single_plate_data(stem: str) -> ImageData:
    d = ImageData(source=f"/{stem}.png", image=None, original_image=None)
    d.metadata.update(
        {
            "rois": [{"label": stem}],
            "multi_plate_mode": True,
            "per_plate_counts": {stem: 5},
            "colony_count": 5,
            "stem": stem,
        }
    )
    d.measurements = [
        {
            "plate_label": stem,
            "label": 1,
            "area_px": 10,
            "is_artifact": False,
            "source": f"/{stem}.png",
        }
    ]
    return d


def test_batch_summary_text_single_multiplate_image() -> None:
    """One multi-plate scan (5 plates) must still yield a batch summary."""
    text = ga.batch_summary_text([_multiplate_data()])
    assert text is not None
    assert "plate_1_count" in text
    assert "plate_5_count" in text


def test_batch_colonies_text_single_multiplate_image() -> None:
    text = ga.batch_colonies_text([_multiplate_data()])
    assert text is not None
    assert "plate_label" in text


def test_batch_texts_empty_for_single_plate() -> None:
    """A single plate (one ROI) has nothing to batch."""
    d = _single_plate_data("A2")
    assert ga.batch_summary_text([d]) is None
    assert ga.batch_colonies_text([d]) is None


def test_batch_summary_text_multiple_single_plates_is_clean() -> None:
    """Three single-plate images produce a batch summary without sparse
    per-plate columns (each image's count is already colony_count)."""
    text = ga.batch_summary_text(
        [_single_plate_data("A2"), _single_plate_data("C4"), _single_plate_data("C5")]
    )
    assert text is not None
    header = text.splitlines()[0]
    # Single-plate images contribute no plate_*_count columns.
    assert "plate_" not in header
    assert "colony_count" in header
