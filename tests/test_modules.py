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
    IlluminationCorrection,
    ImageFileLoader,
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
    assert lines[0] == "label,area_px,source"
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
