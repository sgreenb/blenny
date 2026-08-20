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
    # Every row carries zone + normalized distance metadata, and the
    # rejection raised the expected info flag.
    for r in result:
        assert r["zone"] in ("interior", "edge")
        assert r["normalized_dist"] is not None
        assert r["normalized_dist"] >= 0.0
    assert "artifacts_removed" in [f.code for f in data.quality_flags]


def test_matching_edge_objects_are_accepted() -> None:
    """Edge detections that match the interior profile should not be rejected."""
    data = _data_with_geometry()
    rows = _INTERIOR + _REAL_EDGE
    result = InteriorColonyClassifier().classify(rows, data)
    edge_result = result[len(_INTERIOR) :]
    assert all(not r["is_artifact"] for r in edge_result)
    assert "artifacts_removed" not in [f.code for f in data.quality_flags]


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
    rows_strict = [*_INTERIOR, dict(_row(20, 190.0, 100.0, area=200.0))]
    rows_lenient = [*_INTERIOR, dict(_row(20, 190.0, 100.0, area=200.0))]

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


# ---------------------------------------------------------------------------
# Plate geometry frame handling (regression for the sub-pipeline double-shift)
# ---------------------------------------------------------------------------


def test_local_plate_center_with_bbox_is_not_shifted() -> None:
    """sub_pipeline writes plate_center in the LOCAL frame plus
    ``plate_center_local=True``; _plate_geometry must not subtract the
    (original-frame) plate_bbox in that case."""
    data = ImageData(source="scan.jpg")
    data.metadata["plate_center"] = (150, 120)
    data.metadata["plate_radius"] = 180
    data.metadata["plate_bbox"] = (500, 700, 860, 1060)
    data.metadata["plate_center_local"] = True
    cy, cx, r = InteriorColonyClassifier()._plate_geometry(data)
    assert (cy, cx) == (150.0, 120.0)
    assert r == 180.0


def test_original_frame_plate_center_is_shifted_by_bbox() -> None:
    """detect_plate / detect_facile crop mode: centre is in the ORIGINAL frame
    and the bbox offset must be applied to reach the cropped frame."""
    data = ImageData(source="plate.jpg")
    data.metadata["plate_center"] = (650, 820)
    data.metadata["plate_radius"] = 180
    data.metadata["plate_bbox"] = (500, 700, 860, 1060)
    cy, cx, _ = InteriorColonyClassifier()._plate_geometry(data)
    assert (cy, cx) == (150.0, 120.0)


def test_sub_pipeline_geometry_matches_local_center(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: detect_facile -> sub_pipeline must feed classify_by_interior
    a geometry centred on the true plate centre (regression for the
    double-shift bug that shifted it by the ROI bbox origin)."""
    import numpy as np
    from PIL import Image

    h, w = 800, 1500
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    plates = [(400, 420, 180), (400, 1050, 180)]  # (cy, cx, r)
    for cy, cx, r in plates:
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img[d <= r] = (130, 130, 130)
        img[(d > r) & (d <= r + 6)] = (215, 215, 215)
    img_path = tmp_path / "two_plate.png"
    Image.fromarray(img).save(img_path)

    pipe = Pipeline.from_config(
        [
            {"name": "load_image"},
            {"name": "detect_facile", "params": {"multi_plate": None}},
            {
                "name": "sub_pipeline",
                "params": {
                    "steps": [
                        {
                            "name": "add_manual_colonies",
                            "params": {
                                "coordinates": [[420, 400], [1050, 400]],
                                "radius": 12,
                            },
                        },
                        {"name": "measure_colonies", "params": {"roi_mask_key": "plate"}},
                        {"name": "classify_by_interior"},
                    ]
                },
            },
        ]
    )
    data = pipe.run(img_path)
    subs = data.metadata["multi_plate_results"]
    assert len(subs) == 2
    for sub in subs:
        cy_l, cx_l = sub.metadata["plate_center"]
        geom = sub.metadata["interior_classifier_geometry"]
        assert (geom["cy"], geom["cx"]) == (float(cy_l), float(cx_l))
        # A colony drawn exactly at the plate centre must be interior.
        for m in sub.measurements:
            if abs(m["centroid_y"] - cy_l) < 3 and abs(m["centroid_x"] - cx_l) < 3:
                assert m["zone"] == "interior"
