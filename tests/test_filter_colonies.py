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


def test_no_params_is_noop() -> None:
    rows = [_row(1, area=10), _row(2, area=5000)]
    out = ColonyFilter().classify(rows, _data())
    assert all(not r["is_artifact"] for r in out)
    assert len(out) == 2


def test_min_area_px_drops_small_rows() -> None:
    data = _data()
    rows = [_row(1, area=50), _row(2, area=2000)]
    out = ColonyFilter(min_area_px=100).classify(rows, data)
    small = next(r for r in out if r["centroid_x"] == 100.0 and r["area_px"] == 50)
    big = next(r for r in out if r["area_px"] == 2000)
    assert small["is_artifact"]
    assert "filter_colonies" in small["artifact_reason"]
    assert not big["is_artifact"]
    # Counts updated to exclude the filtered row.
    assert data.metadata["colony_count"] == 1
    assert data.metadata["artifact_count"] == 1
    assert any(f.code == "colonies_filtered" for f in data.quality_flags)


def test_min_area_ppm_uses_roi_area() -> None:
    # ROI = 100_000 px; 100 ppm -> min area 10 px.
    data = _data(roi_area=100_000.0)
    rows = [_row(1, area=5), _row(2, area=500)]
    out = ColonyFilter(min_area_ppm=100).classify(rows, data)
    small = next(r for r in out if r["area_px"] == 5)
    big = next(r for r in out if r["area_px"] == 500)
    assert small["is_artifact"]
    assert not big["is_artifact"]


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


def test_already_artifact_rows_untouched() -> None:
    rows = [_row(1, area=50, is_artifact=True), _row(2, area=500)]
    out = ColonyFilter(min_area_px=100).classify(rows, _data())
    pre_artifact = next(r for r in out if r["area_px"] == 50)
    assert pre_artifact["is_artifact"]  # still artifact, reason untouched
    assert "filter_colonies" not in pre_artifact.get("artifact_reason", "")


def test_labels_renumbered_colonies_first() -> None:
    data = _data()
    rows = [_row(1, area=50), _row(2, area=500)]
    out = ColonyFilter(min_area_px=100).classify(rows, data)
    # Surviving colony gets label 1, artifact gets label 2.
    colony = next(r for r in out if not r["is_artifact"])
    artifact = next(r for r in out if r["is_artifact"])
    assert colony["label"] == 1
    assert artifact["label"] == 2


def test_roi_fallback_to_image_area() -> None:
    data = ImageData(source="t.jpg")
    data.image = np.zeros((200, 200, 3), dtype=np.uint8)  # 40_000 px
    rows = [_row(1, area=1), _row(2, area=500)]
    # 100 ppm of 40_000 px -> 4 px minimum; row 1 dropped.
    out = ColonyFilter(min_area_ppm=100).classify(rows, data)
    assert next(r for r in out if r["area_px"] == 1)["is_artifact"]
    assert not next(r for r in out if r["area_px"] == 500)["is_artifact"]
