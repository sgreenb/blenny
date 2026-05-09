"""Integration test for the YOLO-based colony counting pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from blenny import Pipeline
from blenny.testing import make_synthetic_plate

def _mock_box(coords_array: np.ndarray) -> MagicMock:
    box = MagicMock()
    mock_cpu = MagicMock()
    mock_numpy = MagicMock()
    mock_astype = MagicMock(return_value=coords_array)
    mock_numpy.astype = mock_astype
    mock_cpu.numpy.return_value = mock_numpy
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value = mock_cpu
    return box

def test_yolo_pipeline_integration(tmp_path: Path) -> None:
    # 1. Create a synthetic plate
    plate = make_synthetic_plate(n_colonies=10, image_size=(256, 256), seed=0)
    image_path = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(image_path)

    # 2. Setup a pipeline using names from the registry
    pipe_config = [
        {"name": "load_image"},
        {"name": "detect_plate", "params": {"crop": True}},
        {
            "name": "yolo_detector",
            "params": {
                "model_path": "dummy.pt",
                "refine_mask": True,
                "output_key": "objects"
            }
        },
        {"name": "measure_colonies"},
        {"name": "export_csv", "params": {"output_path": str(tmp_path / "results.csv")}}
    ]

    # 3. Mock YOLO to return detections corresponding to our synthetic colonies
    # (Since we can't easily predict where they are without reading synthetic code,
    # we'll just mock 5 detections at known spots)
    mock_model = MagicMock()
    mock_results = MagicMock()
    
    # Mock 5 boxes near the center of the 256x256 image
    boxes = []
    for i in range(5):
        # [x1, y1, x2, y2]
        # Start at (100, 100), offset by i*10
        base = 100 + i*15
        boxes.append(_mock_box(np.array([base, base, base+10, base+10])))
    
    mock_results.boxes = boxes
    mock_model.predict.return_value = [mock_results]

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        pipe = Pipeline.from_config(pipe_config)
        out = pipe.run(image_path)

        # 4. Verifications
        assert out.metadata["colony_count"] == 5
        assert len(out.measurements) == 5
        assert (tmp_path / "results.csv").exists()
        
        # Check that YOLO ran
        assert any(p.step == "yolo_detector" for p in out.provenance)
        
        # Verify that the mask was created
        assert "objects" in out.masks
        assert out.masks["objects"].shape == out.image.shape[:2]
