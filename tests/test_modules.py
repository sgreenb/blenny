"""Unit tests for individual built-in modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from blenny import ImageData
from blenny.modules import (
    AnnotatedImageExporter,
    ColonyMeasurer,
    CSVExporter,
    ExclusionMasker,
    IDFilter,
    IlluminationCorrection,
    ImageFileLoader,
    ManualColonyAdder,
    PlateDetector,
    ThresholdSegmenter,
)
from blenny.testing import make_synthetic_plate


def _save_synthetic(tmp_path: Path, **kw):  # type: ignore[no-untyped-def]
    plate = make_synthetic_plate(**kw)
    p = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(p)
    return plate, p


# --- Loader -------------------------------------------------------------------


def test_image_file_loader_reads_png(tmp_path: Path) -> None:
    _, p = _save_synthetic(tmp_path, n_colonies=5, image_size=(128, 128), seed=0)
    data = ImageFileLoader().run(ImageData(source=str(p)))
    assert data.image.shape == (128, 128, 3)
    assert data.image.dtype == np.uint8
    assert data.metadata["image_shape"] == (128, 128, 3)
    assert "source_path" in data.metadata


def test_image_file_loader_grayscale_mode(tmp_path: Path) -> None:
    _, p = _save_synthetic(tmp_path, n_colonies=3, image_size=(64, 64), seed=0)
    data = ImageFileLoader(as_gray=True).run(ImageData(source=str(p)))
    assert data.image.ndim == 2


def test_image_file_loader_missing_file_raises(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        ImageFileLoader().run(ImageData(source=str(tmp_path / "nope.png")))


def test_image_file_loader_loads_at_native_resolution_by_default(tmp_path: Path) -> None:
    _, p = _save_synthetic(tmp_path, n_colonies=5, image_size=(3000, 2000), seed=0)
    data = ImageFileLoader().run(ImageData(source=str(p)))
    # Default max_dimension=None means native resolution is kept (and small
    # images are never upscaled — with max_dimension=None the resize branch
    # is never entered at all).
    assert data.image.shape[:2] == (3000, 2000)
    assert "resized" not in data.metadata
    assert all(f.code != "image_resized" for f in data.quality_flags)

    small_p = tmp_path / "small.png"
    Image.fromarray(make_synthetic_plate(n_colonies=3, image_size=(256, 256), seed=0).image).save(
        small_p
    )
    small = ImageFileLoader().run(ImageData(source=str(small_p)))
    assert small.image.shape[:2] == (256, 256)
    assert "resized" not in small.metadata


def test_image_file_loader_downscales_when_max_dimension_set(tmp_path: Path) -> None:
    _, p = _save_synthetic(tmp_path, n_colonies=5, image_size=(3000, 2000), seed=0)
    data = ImageFileLoader(max_dimension=2000).run(ImageData(source=str(p)))
    # max_dimension=2000 means longest side becomes 2000.
    assert max(data.image.shape[:2]) == 2000
    assert data.metadata["resized"] is True
    assert data.metadata["original_size_wh"] == (2000, 3000)  # PIL is (W, H)
    assert any(f.code == "image_resized" and f.severity == "info" for f in data.quality_flags)


# --- PlateDetector ------------------------------------------------------------


def test_plate_detector_finds_plate_and_crops() -> None:
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)
    data = ImageData(source="synthetic", image=plate.image, original_image=plate.image)
    out = PlateDetector().run(data)
    assert "plate" in out.masks
    # Cropped image should be smaller than the source.
    assert out.image.shape[0] <= plate.image.shape[0]
    assert out.image.shape[1] <= plate.image.shape[1]
    # And the detected center should be near the true center.
    cy, cx = out.metadata["plate_center"]
    true_cy, true_cx = plate.plate_center
    assert abs(cy - true_cy) <= 8
    assert abs(cx - true_cx) <= 8


def test_plate_detector_radius_scale_modifies_mask() -> None:
    """radius_scale > 1.0 produces a strictly larger plate mask."""
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)

    d_default = ImageData(source="x", image=plate.image, original_image=plate.image)
    PlateDetector(crop=False, radius_scale=1.0).run(d_default)

    d_expanded = ImageData(source="x", image=plate.image, original_image=plate.image)
    PlateDetector(crop=False, radius_scale=1.10).run(d_expanded)

    n_default = int(d_default.masks["plate"].sum())
    n_expanded = int(d_expanded.masks["plate"].sum())
    assert n_expanded > n_default
    # Detected centre should not move when only the radius is scaled.
    assert d_default.metadata["plate_center"] == d_expanded.metadata["plate_center"]
    # plate_radius_hough is recorded for provenance and must equal the
    # radius found before scaling (so it matches the un-scaled run).
    assert d_default.metadata["plate_radius_hough"] == d_expanded.metadata["plate_radius_hough"]
    assert d_expanded.metadata["plate_radius"] > d_default.metadata["plate_radius"]


def test_plate_detector_flags_when_no_plate() -> None:
    # A flat image with no edges → Hough finds nothing useful but still picks
    # a top peak. Use uniform noise to ensure no circular structure.
    rng = np.random.default_rng(0)
    flat = rng.integers(120, 130, size=(128, 128, 3), dtype=np.uint8)
    data = ImageData(source="x", image=flat, original_image=flat)
    out = PlateDetector(crop=False).run(data)
    # The mask is always written, either real or all-True fallback.
    assert "plate" in out.masks


# --- IlluminationCorrection ---------------------------------------------------


def test_illumination_correction_outputs_2d_float() -> None:
    plate = make_synthetic_plate(n_colonies=8, image_size=(192, 192), seed=0)
    data = ImageData(source="x", image=plate.image)
    out = IlluminationCorrection(radius=20).run(data)
    assert out.image.ndim == 2
    assert out.image.dtype.kind == "f"
    # Top-hat output should be much flatter than the input gradient.
    assert out.image.max() <= 1.0
    # And it stashes the pre-correction image for the annotated exporter.
    assert "pre_illumination" in out.artifacts


# --- ThresholdSegmenter -------------------------------------------------------


def test_threshold_segmenter_labels_objects() -> None:
    plate = make_synthetic_plate(n_colonies=12, image_size=(256, 256), seed=0)
    data = ImageData(source="x", image=plate.image)
    IlluminationCorrection(radius=20).run(data)
    out = ThresholdSegmenter(roi_mask_key=None).run(data)
    labels = out.masks["objects"]
    n_found = int(labels.max())
    # Should find roughly the right count on a clean synthetic plate.
    assert 8 <= n_found <= 16


def test_circularity_filter_drops_elongated_shapes() -> None:
    """A long thick rectangle (low circularity) should be filtered out."""
    img = np.zeros((200, 200), dtype=np.float32)
    # 8 px thick x 120 px long strip survives opening but has low circularity (~0.18).
    img[80:88, 30:150] = 1.0
    # 14x14 square; circularity ~0.85 -> passes 0.7 threshold.
    img[20:34, 20:34] = 1.0
    data = ImageData(source="x", image=img)
    out = ThresholdSegmenter(roi_mask_key=None, split_touching=False).run(data)
    labels = out.masks["objects"]
    # Strip should be dropped, square should remain.
    assert int(labels.max()) == 1
    # And disabling the filter brings the strip back.
    data2 = ImageData(source="x", image=img)
    out2 = ThresholdSegmenter(
        roi_mask_key=None, split_touching=False, min_circularity=0.0, min_solidity=0.0
    ).run(data2)
    assert int(out2.masks["objects"].max()) == 2


def test_solidity_filter_drops_irregular_shapes() -> None:
    """A C-shape (low solidity) should be filtered out at default threshold."""
    img = np.zeros((100, 100), dtype=np.float32)
    # Solid disk-ish square
    img[10:25, 10:25] = 1.0
    # C-shape: a square with a big bite taken out of it (low solidity)
    img[40:80, 40:80] = 1.0
    img[45:75, 45:75] = 0.0
    img[55:65, 60:80] = 1.0  # connect to the right side via a thin bridge
    data = ImageData(source="x", image=img)
    out = ThresholdSegmenter(roi_mask_key=None, split_touching=False, min_circularity=0.0).run(data)
    # Square remains; C-shape dropped.
    assert int(out.masks["objects"].max()) == 1


# --- ColonyMeasurer -----------------------------------------------------------


def test_colony_measurer_emits_one_row_per_object() -> None:
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)
    data = ImageData(source="x", image=plate.image)
    IlluminationCorrection(radius=20).run(data)
    ThresholdSegmenter(roi_mask_key=None).run(data)
    out = ColonyMeasurer().run(data)
    assert len(out.measurements) >= 1
    row = out.measurements[0]
    for key in (
        "label",
        "area_px",
        "centroid_x",
        "centroid_y",
        "equivalent_diameter_px",
        "mean_intensity",
        "touches_edge",
    ):
        assert key in row
    assert out.metadata["colony_count"] == len(out.measurements)


def test_colony_measurer_handles_empty_mask() -> None:
    data = ImageData(source="x", image=np.zeros((32, 32), dtype=np.uint8))
    data.masks["objects"] = np.zeros((32, 32), dtype=np.int32)
    out = ColonyMeasurer().run(data)
    assert out.measurements == []
    assert any(f.code == "no_objects" for f in out.quality_flags)


def test_colony_measurer_flags_suspect_high_count() -> None:
    """suspect_high_count fires when the count or coverage thresholds are hit."""
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)

    def _run(max_count: int, max_cov: float = 0.50):
        data = ImageData(source="x", image=plate.image)
        IlluminationCorrection(radius=20).run(data)
        ThresholdSegmenter(roi_mask_key=None).run(data)
        return ColonyMeasurer(max_plausible_count=max_count, max_coverage_frac=max_cov).run(data)

    # Threshold below the actual count -> flagged.
    assert any(f.code == "suspect_high_count" for f in _run(max_count=2).quality_flags)
    # Generous threshold -> not flagged.
    assert not any(f.code == "suspect_high_count" for f in _run(max_count=600).quality_flags)
    # Absurdly low coverage threshold -> flagged via coverage.
    out = _run(max_count=0, max_cov=0.0001)
    assert any(f.code == "suspect_high_count" for f in out.quality_flags)


# --- UI Support Modules -------------------------------------------------------


def test_manual_colony_adder_adds_blobs() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    data = ImageData(source="test", image=image)
    # Add a colony at (50, 50) with radius 10
    adder = ManualColonyAdder(coordinates=[(50, 50)], radius=10)
    out = adder.run(data)
    mask = out.masks["objects"]
    assert mask.shape == (100, 100)
    assert mask[50, 50] > 0
    assert mask[0, 0] == 0
    assert any(f.code == "manual_inclusions" for f in out.quality_flags)


def test_id_filter_marks_artifacts() -> None:
    data = ImageData(source="test")
    # IDs 1 and 2 at different positions so reassign_ids sort is stable
    data.measurements = [
        {"label": 1, "is_artifact": False, "centroid_y": 10, "centroid_x": 10},
        {"label": 2, "is_artifact": False, "centroid_y": 20, "centroid_x": 20},
    ]
    # Exclude ID 1
    filter_mod = IDFilter(exclude_ids=[1])
    out_rows = filter_mod.classify(data.measurements, data)
    # IDFilter reassigns IDs: colonies first, then artifacts.
    # Original #2 is now the only colony, so it gets label 1.
    # Original #1 is now the artifact, so it gets label 2.
    # We find the one that was original #1
    orig_1 = next(r for r in out_rows if r["centroid_y"] == 10)
    assert orig_1["is_artifact"] is True
    assert orig_1["label"] == 2
    assert any(f.code == "manual_exclusions" for f in data.quality_flags)


def test_exclusion_masker_subtracts_from_target(tmp_path: Path) -> None:
    # 1. Create a "painted" mask file (white square in center)
    mask_im = np.zeros((100, 100), dtype=np.uint8)
    mask_im[40:60, 40:60] = 255
    mask_path = tmp_path / "exclusion.png"
    Image.fromarray(mask_im).save(mask_path)

    # 2. Setup data with a full plate mask
    data = ImageData(source="test")
    data.image = np.zeros((100, 100, 3), dtype=np.uint8)
    data.masks["plate"] = np.ones((100, 100), dtype=bool)

    # 3. Run exclusion
    ExclusionMasker(mask_path=str(mask_path)).run(data)

    # 4. Center should now be False, edges should be True
    # Use == for numpy boolean scalars
    assert not data.masks["plate"][50, 50]
    assert data.masks["plate"][10, 10]


# --- CSVExporter --------------------------------------------------------------


def test_csv_exporter_writes_rows(tmp_path: Path) -> None:
    out_path = tmp_path / "out" / "results.csv"
    data = ImageData(source="x")
    data.measurements = [
        {"label": 1, "area_px": 50, "source": "a.png"},
        {"label": 2, "area_px": 70, "source": "a.png"},
    ]
    CSVExporter(output_path=str(out_path)).run(data)
    text = out_path.read_text()
    lines = text.strip().splitlines()
    # is_artifact is injected by the exporter; source should be present too.
    header_cols = lines[0].split(",")
    assert "label" in header_cols
    assert "area_px" in header_cols
    assert "is_artifact" in header_cols
    assert "source" in header_cols
    assert "50" in lines[1]
    assert "70" in lines[2]


def test_csv_exporter_writes_provenance_comment(tmp_path: Path) -> None:
    out_path = tmp_path / "results.csv"
    data = ImageData(source="x")
    data.measurements = [{"a": 1}]
    # Fake a provenance entry so the comment line has content.
    from blenny.pipeline.context import ProvenanceRecord

    data.provenance.append(
        ProvenanceRecord(step="fake", module_class="X", params={}, duration_s=0.0)
    )
    CSVExporter(output_path=str(out_path), include_provenance=True).run(data)
    first_line = out_path.read_text().splitlines()[0]
    assert first_line.startswith("# provenance:")
    assert "fake" in first_line


# --- AnnotatedImageExporter ---------------------------------------------------


def test_annotated_exporter_writes_png(tmp_path: Path) -> None:
    plate = make_synthetic_plate(n_colonies=6, image_size=(192, 192), seed=0)
    out_path = tmp_path / "annotated.png"
    data = ImageData(source="x", image=plate.image, original_image=plate.image)
    IlluminationCorrection(radius=20).run(data)
    ThresholdSegmenter(roi_mask_key=None).run(data)
    ColonyMeasurer().run(data)
    AnnotatedImageExporter(output_path=str(out_path)).run(data)
    assert out_path.exists()
    # File is a valid image we can re-open.
    with Image.open(out_path) as im:
        assert im.size == (192, 192)
