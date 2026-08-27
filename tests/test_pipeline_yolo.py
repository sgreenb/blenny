"""Integration test: the YOLO detector composes inside a real Pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from blenny import Pipeline
from blenny.testing import make_synthetic_plate


def _mock_box(coords: np.ndarray) -> MagicMock:
    """A YOLO box whose ``xyxy[0].cpu().numpy().astype(int)`` chain works."""
    box = MagicMock()
    cpu = MagicMock()
    cpu.numpy.return_value.astype.return_value = coords
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value = cpu
    return box


def test_yolo_pipeline_integration(tmp_path: Path) -> None:
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)
    image_path = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(image_path)

    pipe_config = [
        {"name": "load_image"},
        {"name": "detect_plate", "params": {"crop": True}},
        {"name": "yolo_detector", "params": {"model_path": "dummy.pt", "refine_mask": True}},
        {"name": "measure_colonies"},
        {"name": "export_csv", "params": {"output_path": str(tmp_path / "results.csv")}},
    ]

    mock_model = MagicMock()
    mock_results = MagicMock()
    # Two known boxes so the count and mask are deterministic.
    mock_results.boxes = [
        _mock_box(np.array([110, 110, 120, 120])),
        _mock_box(np.array([130, 130, 140, 140])),
    ]
    mock_model.predict.return_value = [mock_results]

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        out = Pipeline.from_config(pipe_config).run(image_path)

    assert out.metadata["colony_count"] == 2
    assert len(out.measurements) == 2
    assert (tmp_path / "results.csv").exists()
    assert any(p.step == "yolo_detector" for p in out.provenance)
    assert out.masks["objects"].shape == out.image.shape[:2]
