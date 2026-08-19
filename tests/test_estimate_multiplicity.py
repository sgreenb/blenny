"""Tests for MultiplicityEstimator."""

from __future__ import annotations

from typing import Any

from blenny import ImageData
from blenny.modules import MultiplicityEstimator


def _row(
    label: int,
    *,
    area: float = 400.0,
    circularity: float = 0.95,
    solidity: float = 0.96,
    eccentricity: float = 0.15,
) -> dict[str, Any]:
    return {
        "label": label,
        "centroid_y": 100.0,
        "centroid_x": 100.0,
        "area_px": area,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "mean_intensity": 0.7,
        "colony_count_estimate": 1,
    }


def _ten_clean_singletons() -> list[dict[str, Any]]:
    return [_row(i + 1, area=400.0) for i in range(10)]


def test_clean_singletons_keep_count_estimate_of_one() -> None:
    rows = _ten_clean_singletons()
    MultiplicityEstimator().classify(rows, ImageData(source="t.jpg"))
    assert all(r["colony_count_estimate"] == 1 for r in rows)


def test_merged_blobs_get_estimated_multiplicity() -> None:
    """Bilobed and tri-lobed blobs get count estimates of 2 and 3."""
    rows = _ten_clean_singletons()
    bilobed = _row(99, area=820.0, circularity=0.65, solidity=0.91, eccentricity=0.7)
    rows.append(bilobed)
    MultiplicityEstimator().classify(rows, ImageData(source="t.jpg"))
    assert bilobed["colony_count_estimate"] == 2
    assert "merged-shape" in bilobed.get("multiplicity_reason", "")

    rows = _ten_clean_singletons()
    triple = _row(99, area=1200.0, circularity=0.55, solidity=0.90, eccentricity=0.75)
    rows.append(triple)
    MultiplicityEstimator().classify(rows, ImageData(source="t.jpg"))
    assert triple["colony_count_estimate"] == 3


def test_count_estimate_is_capped() -> None:
    rows = _ten_clean_singletons()
    huge = _row(99, area=20000.0, circularity=0.5, solidity=0.90)
    rows.append(huge)
    MultiplicityEstimator(max_count_estimate=4).classify(rows, ImageData(source="t.jpg"))
    assert huge["colony_count_estimate"] == 4


def test_non_merged_shapes_are_not_upgraded() -> None:
    """Lacy (low solidity) and large-but-round blobs are not merged colonies."""
    rows = _ten_clean_singletons()
    lacy = _row(99, area=820.0, circularity=0.6, solidity=0.70)
    rows.append(lacy)
    MultiplicityEstimator().classify(rows, ImageData(source="t.jpg"))
    assert lacy["colony_count_estimate"] == 1

    rows = _ten_clean_singletons()
    big_round = _row(99, area=820.0, circularity=0.92, solidity=0.95)
    rows.append(big_round)
    MultiplicityEstimator().classify(rows, ImageData(source="t.jpg"))
    assert big_round["colony_count_estimate"] == 1


def test_skipped_when_no_clean_singletons_available() -> None:
    rows = [_row(i + 1, area=400.0, circularity=0.6, solidity=0.8) for i in range(10)]
    data = ImageData(source="t.jpg")
    MultiplicityEstimator().classify(rows, data)
    codes = [f.code for f in data.quality_flags]
    assert "multiplicity_skipped_no_reference" in codes
    assert all(r["colony_count_estimate"] == 1 for r in rows)


def test_metadata_records_singleton_reference() -> None:
    rows = _ten_clean_singletons()
    data = ImageData(source="t.jpg")
    MultiplicityEstimator().classify(rows, data)
    assert data.metadata["singleton_n"] == 10
    assert data.metadata["singleton_median_area_px"] == 400.0
