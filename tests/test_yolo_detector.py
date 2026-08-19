"""Tests for the YOLO-based colony detector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from blenny.modules.yolo_detector import YoloDetector
from blenny.pipeline import ImageData


def _mock_box(coords_array: np.ndarray) -> MagicMock:
    """Create a mock YOLO box that returns the given coordinates."""
    box = MagicMock()
    # Chain: box.xyxy[0].cpu().numpy().astype(int)
    mock_cpu = MagicMock()
    mock_numpy = MagicMock()
    mock_astype = MagicMock(return_value=coords_array)

    mock_numpy.astype = mock_astype
    mock_cpu.numpy.return_value = mock_numpy

    # box.xyxy must be subscriptable
    box.xyxy = [MagicMock()]
    box.xyxy[0].cpu.return_value = mock_cpu
    return box


def test_yolo_detector_basic_inference() -> None:
    """Test that YoloDetector correctly converts mocked YOLO results to a mask."""
    mock_model = MagicMock()
    mock_results = MagicMock()

    # Box: [x1, y1, x2, y2]
    mock_results.boxes = [_mock_box(np.array([10, 10, 20, 20]))]
    mock_model.predict.return_value = [mock_results]

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        detector = YoloDetector(refine_mask=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        data = ImageData(source="test", image=image)

        out_mask = detector.segment(image, data)

        assert out_mask.shape == (100, 100)
        assert out_mask.max() == 1
        # Ellipse proxy should label the center
        assert out_mask[15, 15] == 1


def test_yolo_detector_roi_filtering() -> None:
    """Test that YoloDetector respects the ROI (plate) mask."""
    mock_model = MagicMock()
    mock_results = MagicMock()

    # Two boxes: one inside ROI, one outside
    # Box 1: Center (15, 15) -> inside top-left 50x50
    # Box 2: Center (85, 85) -> outside
    mock_results.boxes = [
        _mock_box(np.array([10, 10, 20, 20])),
        _mock_box(np.array([80, 80, 90, 90])),
    ]
    mock_model.predict.return_value = [mock_results]

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        detector = YoloDetector(refine_mask=False)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # ROI mask: only top-left quadrant
        roi_mask = np.zeros((100, 100), dtype=bool)
        roi_mask[0:50, 0:50] = True

        data = ImageData(source="test", image=image)
        data.masks["plate"] = roi_mask

        out_mask = detector.segment(image, data)

        # Should only find one colony (the one inside the ROI)
        assert out_mask.max() == 1
        assert out_mask[15, 15] == 1
        assert out_mask[85, 85] == 0


def test_yolo_detector_predict_receives_bgr_input() -> None:
    """RGB (PIL) images must be converted to BGR before predict().

    Ultralytics treats numpy arrays as BGR; without the conversion the model
    would receive red/blue-swapped pixels.
    """
    mock_model = MagicMock()
    captured: list[np.ndarray] = []

    def _fake_predict(image, **kwargs):
        captured.append(np.asarray(image))
        res = MagicMock()
        res.boxes = []
        return [res]

    mock_model.predict.side_effect = _fake_predict

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        detector = YoloDetector(model_path="dummy.pt", refine_mask=False)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[32, 32] = [255, 0, 0]  # pure red in RGB
        detector.segment(image, ImageData(source="test", image=image))

    assert len(captured) == 1
    # Red (RGB 255,0,0) must arrive as BGR (0,0,255)
    assert captured[0][32, 32].tolist() == [0, 0, 255]


def test_yolo_detector_handles_float_images() -> None:
    """Test that YoloDetector handles float images correctly (Otsu refinement)."""
    mock_model = MagicMock()
    mock_results = MagicMock()

    mock_results.boxes = [_mock_box(np.array([10, 10, 20, 20]))]
    mock_model.predict.return_value = [mock_results]

    with patch("blenny.modules.yolo_detector.YOLO", return_value=mock_model):
        detector = YoloDetector(refine_mask=True)
        # Float image (64-bit)
        image = np.zeros((100, 100, 3), dtype=np.float64)
        # Add a bright spot for Otsu to find inside the box
        image[12:18, 12:18] = 1.0

        data = ImageData(source="test", image=image)

        # This would have failed with cv2.error before the fix
        out_mask = detector.segment(image, data)

        assert out_mask.shape == (100, 100)
        assert out_mask.max() == 1
        assert out_mask[15, 15] == 1
