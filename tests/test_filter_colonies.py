"""Tests for the post-detection colony size/shape filter."""

from __future__ import annotations

import numpy as np

from blenny import ImageData
from blenny.modules import ColonyFilter


def _row(
    label: int,
    area: float = 400.0,
    circularity: float = 0.9,
    solidity: float = 0.95,
    estimate: int = 1,
    is_artifact: bool = False,
) -> dict:
    return {
        "label": label,
        "centroid_x": 100.0,
        "centroid_y": 100.0,
        "area_px": area,
        "circularity": circularity,
        "solidity": solidity,
        "colony_count_estimate": estimate,
        "is_artifact": is_artifact,
        "source": "t.jpg",
    }


def _data(roi_area: float = 100_000.0) -> ImageData:
    data = ImageData(source="t.jpg")
    mask = np.zeros((400, 400), dtype=bool)
    mask[: int(roi_area) // 400, :400] = True  # roughly roi_area pixels
    data.masks["plate"] = mask
    return data


def test_area_filter_disabled_by_default_or_zero() -> None:
    """No params, or min_area_ppm=0 (the GUI default), means no filtering."""
    for kwargs in ({}, {"min_area_ppm": 0}):
        rows = [_row(1, area=1), _row(2, area=500)]
        out = ColonyFilter(**kwargs).classify(rows, _data())
        assert all(not r["is_artifact"] for r in out)


def test_min_area_filters_small_rows() -> None:
    """Both min_area_px and min_area_ppm drop small detections."""
    # Pixel-based threshold.
    data = _data()
    rows = [_row(1, area=50), _row(2, area=2000)]
    out = ColonyFilter(min_area_px=100).classify(rows, data)
    small = next(r for r in out if r["area_px"] == 50)
    big = next(r for r in out if r["area_px"] == 2000)
    assert small["is_artifact"]
    assert "filter_colonies" in small["artifact_reason"]
    assert not big["is_artifact"]
    # Counts updated to exclude the filtered row.
    assert data.metadata["colony_count"] == 1
    assert data.metadata["artifact_count"] == 1
    assert any(f.code == "colonies_filtered" for f in data.quality_flags)

    # PPM-based threshold against the ROI area (100 ppm of 100_000 px = 10 px).
    rows = [_row(1, area=5), _row(2, area=500)]
    out = ColonyFilter(min_area_ppm=100).classify(rows, _data(roi_area=100_000.0))
    assert next(r for r in out if r["area_px"] == 5)["is_artifact"]
    assert not next(r for r in out if r["area_px"] == 500)["is_artifact"]

    # And ppm falls back to the image area when no plate mask exists.
    data_img = ImageData(source="t.jpg")
    data_img.image = np.zeros((200, 200, 3), dtype=np.uint8)  # 40_000 px
    rows = [_row(1, area=1), _row(2, area=500)]
    out = ColonyFilter(min_area_ppm=100).classify(rows, data_img)  # 100 ppm -> 4 px
    assert next(r for r in out if r["area_px"] == 1)["is_artifact"]
    assert not next(r for r in out if r["area_px"] == 500)["is_artifact"]


def test_min_circularity_drops_elongated_rows() -> None:
    rows = [_row(1, circularity=0.4), _row(2, circularity=0.92)]
    out = ColonyFilter(min_circularity=0.75).classify(rows, _data())
    flat = next(r for r in out if r["circularity"] == 0.4)
    round_ = next(r for r in out if r["circularity"] == 0.92)
    assert flat["is_artifact"]
    assert "circularity" in flat["artifact_reason"]
    assert not round_["is_artifact"]


def test_merged_colonies_exempt_from_shape_filters() -> None:
    """Fused colonies are legitimately non-round; don't filter them."""
    rows = [_row(1, area=800, circularity=0.6, estimate=2)]
    out = ColonyFilter(min_circularity=0.75).classify(rows, _data())
    assert not out[0]["is_artifact"]


def test_artifacts_respected_and_labels_renumbered() -> None:
    """Pre-existing artifacts are untouched; surviving colonies are renumbered first."""
    data = _data()
    rows = [_row(1, area=50, is_artifact=True), _row(2, area=500)]
    out = ColonyFilter(min_area_px=100).classify(rows, data)
    # The pre-existing artifact keeps its artifact status and reason.
    pre_artifact = next(r for r in out if r["area_px"] == 50)
    assert pre_artifact["is_artifact"]
    assert "filter_colonies" not in pre_artifact.get("artifact_reason", "")
    # Colony gets label 1, artifact gets label 2.
    colony = next(r for r in out if not r["is_artifact"])
    assert colony["label"] == 1
    assert pre_artifact["label"] == 2
