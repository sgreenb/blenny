"""Tests for the shared batch-output writers (blenny.batch)."""

from __future__ import annotations

from pathlib import Path

from blenny.batch import write_batch_colonies_csv


def _row(**overrides: object) -> dict:
    row = {
        "plate_label": "1",
        "label": 1,
        "centroid_x": 10.0,
        "centroid_y": 20.0,
        "area_px": 120,
        "circularity": 0.9,
        "solidity": 0.95,
        "eccentricity": 0.3,
        "mean_r": 200.0,
        "mean_g": 190.0,
        "mean_b": 180.0,
        "mean_h": 0.55,
        "mean_s": 0.4,
        "mean_v": 0.78,
        "is_artifact": False,
        "artifact_reason": "",
        "source": "img [1]",
        "colony_count_estimate": 2,
        "segment_label": 7,
        "classification": "bright",
        "bbox_y0": 0,
        "bbox_x0": 0,
        "bbox_y1": 20,
        "bbox_x1": 20,
    }
    row.update(overrides)
    return row


def test_batch_colonies_csv_keeps_all_measurement_columns(tmp_path: Path) -> None:
    """The batch writer must keep every measurement column (regression: the
    GUI's old fixed-column writer silently dropped 10 of them)."""
    p = tmp_path / "batch_colonies.csv"
    write_batch_colonies_csv(p, [_row()])

    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    for col in [
        "mean_h",
        "mean_s",
        "mean_v",
        "colony_count_estimate",
        "segment_label",
        "classification",
        "bbox_y0",
        "bbox_x0",
        "bbox_y1",
        "bbox_x1",
    ]:
        assert col in header, f"column {col!r} missing from batch CSV"


def test_batch_colonies_csv_preferred_order_first(tmp_path: Path) -> None:
    """Preferred columns come first (in order), extra columns after."""
    p = tmp_path / "batch_colonies.csv"
    write_batch_colonies_csv(p, [_row()])

    header = p.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[0] == "plate_label"
    assert header[1] == "label"
    assert header.index("mean_h") > header.index("mean_b")
    assert header[-1] == "bbox_x1"  # extras appended at the end


def test_batch_colonies_csv_placeholder_when_empty(tmp_path: Path) -> None:
    p = tmp_path / "batch_colonies.csv"
    write_batch_colonies_csv(p, [])
    assert p.read_text(encoding="utf-8") == "# no measurements\n"
