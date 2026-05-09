"""YOLO-based colony detection module."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import draw

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from blenny.pipeline import BlennyParams, ImageData, Segmenter, register


@register("yolo_detector")
class YoloDetector(Segmenter):
    """Detect colonies using a YOLO model.

    This module runs a YOLO object detection model on the image and converts
    the resulting bounding boxes into a label mask. Each detected box is
    rendered as an ellipse into the mask.
    """

    class Params(BlennyParams):
        model_path: str = "models/colony_model.pt"
        """Path to the trained YOLO model file (.pt or .onnx)."""

        conf_threshold: float = 0.15
        """Confidence threshold for detections."""

        iou_threshold: float = 0.8
        """IOU threshold for Non-Maximum Suppression (NMS)."""

        imgsz: int = 1280
        """Image size for YOLO inference."""

        refine_mask: bool = True
        """If True, use Otsu thresholding within each box to refine the colony shape."""

        roi_mask_key: str | None = "plate"
        """If set and present in ``data.masks``, detections outside this mask are ignored."""

        output_key: str = "objects"
        """Key in ``data.masks`` for the resulting label image."""

    def segment(self, image: Any, data: ImageData) -> Any:
        if not HAS_ULTRALYTICS:
            raise ImportError(
                "The 'ultralytics' package is required for yolo_detector. "
                "Install it with: pip install ultralytics"
            )

        import cv2

        # Load model (cached by ultralytics)
        model = YOLO(self.params.model_path)  # type: ignore[attr-defined]

        # Perform inference
        results = model.predict(
            image,
            imgsz=self.params.imgsz,  # type: ignore[attr-defined]
            conf=self.params.conf_threshold,  # type: ignore[attr-defined]
            iou=self.params.iou_threshold,  # type: ignore[attr-defined]
            max_det=1000,
            verbose=False,
        )

        # Create an empty label mask
        h, w = image.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)

        # Get ROI mask if requested
        roi_mask = None
        roi_key = self.params.roi_mask_key  # type: ignore[attr-defined]
        if roi_key and roi_key in data.masks:
            roi_mask = np.asarray(data.masks[roi_key], dtype=bool)

        if len(results) > 0:
            boxes = results[0].boxes
            next_id = 1
            for _i, box in enumerate(boxes):
                # YOLO boxes are [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords

                # Boundary check
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # ROI Check: the center of the detection must be inside the ROI
                if roi_mask is not None:
                    cy_box, cx_box = (y1 + y2) // 2, (x1 + x2) // 2
                    if not roi_mask[cy_box, cx_box]:
                        continue

                label_id = next_id
                next_id += 1

                if self.params.refine_mask:  # type: ignore[attr-defined]
                    # Refine using Otsu thresholding within the box
                    crop = image[y1:y2, x1:x2]

                    # Convert to gray 8-bit for OpenCV. 
                    # OpenCV functions (cvtColor, threshold) often require uint8 or uint16.
                    if crop.dtype != np.uint8:
                        if crop.max() <= 1.1: # Likely 0-1 float
                            crop_ui8 = (crop * 255).astype(np.uint8)
                        else:
                            crop_ui8 = crop.astype(np.uint8)
                    else:
                        crop_ui8 = crop

                    if crop_ui8.ndim == 3:
                        gray = cv2.cvtColor(crop_ui8, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = crop_ui8

                    # Otsu threshold
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Circular mask to remove background corners
                    h_c, w_c = gray.shape
                    circ_mask = np.zeros((h_c, w_c), dtype=np.uint8)
                    cv2.ellipse(
                        circ_mask, (w_c // 2, h_c // 2), (w_c // 2, h_c // 2), 0, 0, 360, 255, -1
                    )

                    # Combine Otsu and circular mask
                    refined = cv2.bitwise_and(thresh, thresh, mask=circ_mask)

                    # Apply to global label mask
                    labels[y1:y2, x1:x2][refined > 0] = label_id
                else:
                    # Draw an ellipse as a proxy for the colony shape
                    center = ((y1 + y2) // 2, (x1 + x2) // 2)
                    axes = ((y2 - y1) // 2, (x2 - x1) // 2)
                    rr, cc = draw.ellipse(
                        center[0], center[1], axes[0], axes[1], shape=labels.shape
                    )
                    labels[rr, cc] = label_id

        return labels
