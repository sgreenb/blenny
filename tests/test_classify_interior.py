"""Tests for InteriorColonyClassifier."""

from __future__ import annotations

import math

import numpy as np

from blenny import ImageData, Pipeline
from blenny.modules import InteriorColonyClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row(
    label: int,
    cy: float,
    cx: float,
    area: float = 400.0,
    intensity: float = 0.75,
    eccentricity: float = 0.15,
) -> dict:
    return {
        "label": label,
        "centroid_y": cy,
        "centroid_x": cx,
        "area_px": area,
        "mean_intensity": intensity,
        "eccentricity": eccentricity,
        "equivalent_diameter_px": 2 * math.sqrt(area / math.pi),
        "touches_edge": False,
        "source": "test.jpg",
    }


def _data_with_geometry(
    center: tuple[float, float] = (100.0, 100.0),
    radius: float = 100.0,
) -> ImageData:
    """Return an ImageData with plate geometry set in metadata (no crop)."""
    data = ImageData(source="test.jpg")
    data.metadata["plate_center"] = (int(center[0]), int(center[1]))
    data.metadata["plate_radius"] = int(radius)
    # No plate_bbox → geometry is in the current (uncropped) frame
    return data


# Interior rows: near the centre (normalised dist ~ 0.3)
_INTERIOR = [_row(i + 1, 100 + 25 * math.sin(i), 100 + 25 * math.cos(i)) for i in range(10)]

# Artifact rows: near the edge (normalised dist ~ 0.92), much smaller area
_ARTIFACTS = [
    _row(11 + i, 100 + 88 * math.sin(i), 100 + 88 * math.cos(i), area=15.0) for i in range(5)
]

# Real-looking edge rows that match the interior profile
_REAL_EDGE = [
    _row(20 + i, 100 + 80 * math.sin(i), 100 + 80 * math.cos(i), area=420.0) for i in range(4)
]


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------


def test_small_edge_objects_marked_as_artifacts() -> None:
    data = _data_with_geometry()
    rows = _INTERIOR + _ARTIFACTS
    result = InteriorColonyClassifier().classify(rows, data)

    interior_result = result[: len(_INTERIOR)]
    artifact_result = result[len(_INTERIOR) :]

    assert all(not r["is_artifact"] for r in interior_result)
    assert all(r["is_artifact"] for r in artifact_result)
    assert all(r["artifact_reason"] for r in artifact_result)


def test_matching_edge_objects_are_accepted() -> None:
    """Edge detections that match the interior profile should not be rejected."""
    data = _data_with_geometry()
    rows = _INTERIOR + _REAL_EDGE
    result = InteriorColonyClassifier().classify(rows, data)
    edge_result = result[len(_INTERIOR) :]
    assert all(not r["is_artifact"] for r in edge_result)


def test_interior_colonies_are_never_rejected() -> None:
    """The classifier only evaluates the edge zone; interior is always trusted."""
    data = _data_with_geometry()
    rows = list(_INTERIOR)
    result = InteriorColonyClassifier().classify(rows, data)
    assert all(not r["is_artifact"] for r in result)


def test_colony_count_updated_to_exclude_artifacts() -> None:
    data = _data_with_geometry()
    rows = _INTERIOR + _ARTIFACTS
    InteriorColonyClassifier().classify(rows, data)
    assert data.metadata["colony_count"] == len(_INTERIOR)
    assert data.metadata["artifact_count"] == len(_ARTIFACTS)


def test_zone_and_normalised_dist_set_on_all_rows() -> None:
    data = _data_with_geometry()
    rows = _INTERIOR + _ARTIFACTS
    result = InteriorColonyClassifier().classify(rows, data)
    for r in result:
        assert r["zone"] in ("interior", "edge")
        assert r["normalized_dist"] is not None
        assert r["normalized_dist"] >= 0.0


def test_artifacts_flag_raised_when_rejections_occur() -> None:
    data = _data_with_geometry()
    rows = _INTERIOR + _ARTIFACTS
    InteriorColonyClassifier().classify(rows, data)
    codes = [f.code for f in data.quality_flags]
    assert "artifacts_removed" in codes


def test_no_flag_when_all_edge_rows_accepted() -> None:
    data = _data_with_geometry()
    rows = _INTERIOR + _REAL_EDGE
    InteriorColonyClassifier().classify(rows, data)
    codes = [f.code for f in data.quality_flags]
    assert "artifacts_removed" not in codes


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


def test_falls_back_to_strict_filter_when_interior_too_small() -> None:
    """Fewer interior samples than min_interior_samples → strict shape fallback.

    Round detections survive; non-round ones are rejected. This replaces the
    earlier pass-through behaviour, which let blank plates report dozens of
    "colonies" coming from agar texture.
    """
    data = _data_with_geometry()
    sparse_interior = _INTERIOR[:3]  # only 3, default min is 5; ecc=0.15
    elongated_edge = [_row(50, 100.0, 190.0, area=40.0, eccentricity=0.9)]
    rows = sparse_interior + elongated_edge
    result = InteriorColonyClassifier().classify(rows, data)

    # Round interior rows survive, elongated edge row is rejected.
    assert all(not r["is_artifact"] for r in result[: len(sparse_interior)])
    assert result[-1]["is_artifact"]
    codes = [f.code for f in data.quality_flags]
    # Either flag is acceptable depending on whether the small sample also
    # tripped the degeneracy heuristic; both lead to the same fallback path.
    assert "interior_classifier_insufficient_samples" in codes or "plate_likely_empty" in codes


def test_blank_plate_triggers_plate_likely_empty() -> None:
    """A plate whose detections are mostly elongated noise should be flagged
    as likely empty and have its noise rejected, not reported as colonies."""
    data = _data_with_geometry()
    # 30 detections, all elongated (eccentricity ~0.85), wildly variable area.
    rng = np.random.default_rng(0)
    rows = [
        _row(
            i + 1,
            cy=100 + 60 * math.sin(i),
            cx=100 + 60 * math.cos(i),
            area=float(rng.integers(20, 800)),
            eccentricity=0.85,
        )
        for i in range(30)
    ]
    InteriorColonyClassifier().classify(rows, data)
    codes = [f.code for f in data.quality_flags]
    assert "plate_likely_empty" in codes
    # Almost everything should be marked as an artifact.
    n_artifacts = sum(1 for r in rows if r["is_artifact"])
    assert n_artifacts >= 0.9 * len(rows)
    assert data.metadata["colony_count"] <= 0.1 * len(rows)


def test_skips_classification_when_no_plate_geometry() -> None:
    """No metadata → warning flag, all rows pass through."""
    data = ImageData(source="test.jpg")  # no plate_center or plate_radius
    rows = _INTERIOR + _ARTIFACTS
    result = InteriorColonyClassifier().classify(rows, data)
    assert all(not r["is_artifact"] for r in result)
    codes = [f.code for f in data.quality_flags]
    assert "interior_classifier_no_geometry" in codes


def test_falls_back_to_plate_mask_for_geometry() -> None:
    """When metadata lacks plate_center, derive geometry from the plate mask."""
    data = ImageData(source="test.jpg")
    # Create a circular plate mask centred at (100, 100) with radius 100.
    mask = np.zeros((200, 200), dtype=bool)
    yy, xx = np.ogrid[:200, :200]
    mask[(yy - 100) ** 2 + (xx - 100) ** 2 <= 100**2] = True
    data.masks["plate"] = mask

    rows = list(_INTERIOR) + list(_ARTIFACTS)
    result = InteriorColonyClassifier().classify(rows, data)

    # Classification should have run (no geometry warning).
    codes = [f.code for f in data.quality_flags]
    assert "interior_classifier_no_geometry" not in codes
    # Interior rows not rejected.
    interior_result = result[: len(_INTERIOR)]
    assert all(not r["is_artifact"] for r in interior_result)


def test_handles_empty_measurements() -> None:
    data = _data_with_geometry()
    result = InteriorColonyClassifier().classify([], data)
    assert result == []


# ---------------------------------------------------------------------------
# IQR edge cases
# ---------------------------------------------------------------------------


def test_zero_iqr_does_not_reject_everything() -> None:
    """If all interior colonies have the same area, the IQR=0 guard applies
    and normal edge colonies still pass."""
    data = _data_with_geometry()
    # All interior rows have identical area.
    interior = [
        _row(i + 1, 100 + 20 * math.sin(i), 100 + 20 * math.cos(i), area=500.0) for i in range(8)
    ]
    edge_matching = [_row(20, 190.0, 100.0, area=500.0)]  # same area, at edge
    rows = interior + edge_matching
    result = InteriorColonyClassifier().classify(rows, data)
    edge_result = result[len(interior) :]
    assert not edge_result[0]["is_artifact"]


def test_iqr_multiplier_controls_strictness() -> None:
    """Lower multiplier → more rejections; higher → fewer."""
    data_strict = _data_with_geometry()
    data_lenient = _data_with_geometry()

    # Edge row with area slightly outside the normal IQR range.
    # Note: fresh dict copies per run — the two runs must not share rows,
    # because classify() resets is_artifact on every row at the start.
    rows_strict = _INTERIOR + [dict(_row(20, 190.0, 100.0, area=200.0))]
    rows_lenient = _INTERIOR + [dict(_row(20, 190.0, 100.0, area=200.0))]

    InteriorColonyClassifier(iqr_multiplier=0.5).classify(rows_strict, data_strict)
    InteriorColonyClassifier(iqr_multiplier=5.0).classify(rows_lenient, data_lenient)

    strict_rejected = rows_strict[-1]["is_artifact"]
    lenient_rejected = rows_lenient[-1]["is_artifact"]
    # The strict multiplier must reject the borderline edge row; the lenient
    # one must accept it (area 200 vs interior 400 ± 0.5·40 vs ± 5·40).
    assert strict_rejected, "iqr_multiplier=0.5 should reject the edge row"
    assert not lenient_rejected, "iqr_multiplier=5.0 should accept the edge row"


# ---------------------------------------------------------------------------
# Integration: full pipeline on synthetic plate
# ---------------------------------------------------------------------------


def test_classifier_in_full_pipeline_does_not_reject_real_colonies(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """On a clean synthetic plate the classifier should accept all or most colonies."""
    from PIL import Image

    from blenny.testing import make_synthetic_plate

    plate = make_synthetic_plate(n_colonies=20, image_size=(384, 384), seed=0)
    img_path = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(img_path)

    pipe = Pipeline.from_config(
        [
            {"name": "load_image"},
            {"name": "detect_plate", "params": {"crop": True}},
            {"name": "correct_illumination", "params": {"radius": 20}},
            {"name": "threshold_segment", "params": {"roi_mask_key": "plate"}},
            {"name": "measure_colonies"},
            {"name": "classify_by_interior"},
        ]
    )
    result = pipe.run(img_path)

    n_artifacts = result.metadata.get("artifact_count", 0)
    n_colonies = result.metadata.get("colony_count", 0)
    total = n_artifacts + n_colonies

    # Expect very few artifacts on a clean synthetic plate.
    assert total > 0
    assert n_artifacts / total < 0.2, (
        f"Too many artifacts rejected ({n_artifacts}/{total}); "
        "classifier is probably too strict for clean images."
    )
