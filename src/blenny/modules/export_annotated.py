"""Write an annotated overlay image showing detected objects.

Draws the boundary of every labelled region on top of the working
image (or, if available, the pre-illumination image stashed by
:class:`IlluminationCorrection`) and optionally labels each object
with its index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology, segmentation, util

from blenny.pipeline import BlennyParams, Exporter, ImageData, register


@register("export_annotated")
class AnnotatedImageExporter(Exporter):
    class Params(BlennyParams):
        output_path: str
        mask_key: str = "objects"
        draw_numbers: bool = True
        outline_color: tuple[int, int, int] = (255, 64, 64)
        artifact_outline_color: tuple[int, int, int] = (255, 0, 255)
        """Outline color for detections marked ``is_artifact=True``.
        Drawn without numbers so they're visually distinct from counted
        colonies but remain auditable. Default is magenta.
        """
        merged_outline_color: tuple[int, int, int] = (200, 0, 255)
        """Outline color for detections tagged by ``estimate_multiplicity``
        as merged colonies (``colony_count_estimate >= 2``). Their text
        label is suffixed with ``xN`` so it's clear they contribute more
        than one to the total count. Default is a deep purple/magenta.
        """
        text_color: tuple[int, int, int] = (255, 255, 0)
        draw_plate_boundary: bool = True
        """If True, draw the effective plate-area boundary (after rim margin
        exclusion) as a blue ring. Useful for diagnosing plate detection and
        seeing exactly which colonies fall inside the analysis region.
        """
        plate_boundary_color: tuple[int, int, int] = (0, 255, 0)
        """Color of the plate boundary ring (default: bright green)."""
        plate_mask_key: str = "plate"
        """Key in ``data.masks`` for the plate interior mask used to draw
        the boundary ring."""
        plate_boundary_thickness: int = 0
        """Half-thickness of the boundary ring in pixels (the ring is drawn
        by dilating the mask boundary by a disk of this radius).
        Default 0 = 1 px wide, matching the colony outline thickness.
        Increase to 1 (3 px) or 2 (5 px) if you need a more visible ring
        at lower display resolutions.
        """

    def export(self, data: ImageData) -> None:
        if self.params.mask_key not in data.masks:  # type: ignore[attr-defined]
            data.add_flag(
                "annotated_export_no_mask",
                f"AnnotatedImageExporter: mask {self.params.mask_key!r} not found.",  # type: ignore[attr-defined]
                severity="warning",
            )
            return

        labels = data.masks[self.params.mask_key]  # type: ignore[attr-defined]
        base = self._pick_base_image(data, labels.shape)
        if base is None:
            data.add_flag(
                "annotated_export_no_base",
                "AnnotatedImageExporter: no image of matching shape to annotate.",
                severity="warning",
            )
            return

        # Split label IDs into three groups for separate colouring:
        # artifacts (orange), merged colonies (magenta), and singletons (red).
        artifact_ids = {
            int(r["label"]) for r in data.measurements if r.get("is_artifact") and "label" in r
        }
        merged_ids = {
            int(r["label"])
            for r in data.measurements
            if not r.get("is_artifact")
            and int(r.get("colony_count_estimate", 1)) >= 2
            and "label" in r
        }

        # Convert to uint8 RGB for drawing.
        rgb = self._to_rgb_uint8(base)

        if artifact_ids or merged_ids:
            artifact_mask = (
                np.isin(labels, list(artifact_ids))
                if artifact_ids
                else np.zeros_like(labels, dtype=bool)
            )
            merged_mask = (
                np.isin(labels, list(merged_ids))
                if merged_ids
                else np.zeros_like(labels, dtype=bool)
            )
            
            # Multi-class support: handle colonies with custom class colors
            class_colored_rows = [
                r for r in data.measurements 
                if not r.get("is_artifact") 
                and "class_color" in r 
                and "label" in r
            ]
            
            normal_mask = (labels > 0) & ~artifact_mask & ~merged_mask
            
            # Remove custom colored ones from the normal mask
            for r in class_colored_rows:
                normal_mask &= (labels != int(r["label"]))

            normal_labels = np.where(normal_mask, labels, 0)
            artifact_labels_arr = np.where(artifact_mask, labels, 0)
            merged_labels_arr = np.where(merged_mask, labels, 0)
            
            normal_bounds = segmentation.find_boundaries(normal_labels, mode="outer")
            artifact_bounds = segmentation.find_boundaries(artifact_labels_arr, mode="outer")
            merged_bounds = segmentation.find_boundaries(merged_labels_arr, mode="outer")
            
            rgb[normal_bounds] = np.array(self.params.outline_color, dtype=np.uint8)  # type: ignore[attr-defined]
            rgb[artifact_bounds] = np.array(self.params.artifact_outline_color, dtype=np.uint8)  # type: ignore[attr-defined]
            rgb[merged_bounds] = np.array(self.params.merged_outline_color, dtype=np.uint8)  # type: ignore[attr-defined]
            
            # Draw each class-colored colony
            for r in class_colored_rows:
                m = (labels == int(r["label"]))
                b = segmentation.find_boundaries(m, mode="outer")
                rgb[b] = np.array(r["class_color"], dtype=np.uint8)
        else:
            boundaries = segmentation.find_boundaries(labels, mode="outer")
            rgb[boundaries] = np.array(self.params.outline_color, dtype=np.uint8)  # type: ignore[attr-defined]

        # Draw the plate boundary ring over the colony outlines so it's
        # always visible regardless of what's beneath it.
        if self.params.draw_plate_boundary:  # type: ignore[attr-defined]
            self._draw_plate_boundary(rgb, data, labels.shape)

        im = Image.fromarray(rgb)
        
        if self.params.draw_numbers:  # type: ignore[attr-defined]
            # Draw numbers for ALL detections now that artifacts have sequential IDs
            # following the colony IDs.
            self._draw_numbers(im, data.measurements)

        path = Path(self.params.output_path)  # type: ignore[attr-defined]
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path)

    @staticmethod
    def _pick_base_image(data: ImageData, target_shape: tuple[int, int]) -> Any:
        """Choose the most informative image whose 2D shape matches the labels."""
        candidates = [
            data.artifacts.get("pre_illumination"),
            data.image,
            data.original_image,
        ]
        for cand in candidates:
            if cand is None:
                continue
            shape2d = cand.shape[:2]
            if shape2d == target_shape:
                return cand
        return None

    @staticmethod
    def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            arr = util.img_as_ubyte(np.clip(image, 0, 1) if image.dtype.kind == "f" else image)
            return np.stack([arr] * 3, axis=-1)
        if image.dtype.kind == "f":
            return util.img_as_ubyte(np.clip(image, 0, 1))
        return image.astype(np.uint8, copy=True)

    def _draw_plate_boundary(
        self, rgb: np.ndarray, data: Any, target_shape: tuple[int, int]
    ) -> None:
        """Draw a thick ring at the inner edge of the plate mask."""
        key: str = self.params.plate_mask_key  # type: ignore[attr-defined]
        plate_mask = data.masks.get(key)
        if plate_mask is None:
            return
        arr = np.asarray(plate_mask, dtype=bool)
        if arr.shape[:2] != target_shape:
            return
        # Inner boundary: pixels inside the mask that touch the background.
        boundary = segmentation.find_boundaries(arr, mode="inner")
        t: int = self.params.plate_boundary_thickness  # type: ignore[attr-defined]
        if t > 0:
            boundary = morphology.dilation(boundary, morphology.disk(t))
        color = np.array(self.params.plate_boundary_color, dtype=np.uint8)  # type: ignore[attr-defined]
        rgb[boundary] = color

    def _draw_numbers(self, im: Image.Image, measurements: list[dict[str, Any]]) -> None:
        draw = ImageDraw.Draw(im)
        w, h = im.size
        # Heuristic for font size: roughly 1.5% of the image's min dimension.
        fs = max(10, int(min(w, h) * 0.015))
        try:
            font = ImageFont.load_default(size=fs)
        except TypeError:
            font = ImageFont.load_default()

        # Offset labels based on font size.
        off_x = int(fs * 0.4)
        off_y = -int(fs * 0.6)

        # Collision avoidance: keep track of occupied areas (rectangles).
        # We use a simple grid or list of bboxes. Since N is usually < 500,
        # a list is fine.
        occupied_bboxes: list[tuple[float, float, float, float]] = []

        for row in measurements:
            if "centroid_x" not in row or "centroid_y" not in row:
                continue
            
            x = float(row["centroid_x"])
            y = float(row["centroid_y"])
            label = str(row.get("label", "?"))
            est = int(row.get("colony_count_estimate", 1))
            if est >= 2:
                label = f"{label}x{est}"
            
            # Determine text size to calculate bounding box
            left, top, right, bottom = draw.textbbox((x + off_x, y + off_y), label, font=font)
            
            # Add a small buffer to the bbox
            buffer = fs * 0.2
            curr_bbox = (left - buffer, top - buffer, right + buffer, bottom + buffer)
            
            # Check for collisions with previously placed labels
            overlap = False
            for other_bbox in occupied_bboxes:
                # Standard AABB collision check
                if not (curr_bbox[2] < other_bbox[0] or 
                        curr_bbox[0] > other_bbox[2] or 
                        curr_bbox[3] < other_bbox[1] or 
                        curr_bbox[1] > other_bbox[3]):
                    overlap = True
                    break
            
            if not overlap:
                draw.text((x + off_x, y + off_y), label, fill=self.params.text_color, font=font)  # type: ignore[attr-defined]
                occupied_bboxes.append(curr_bbox)
            else:
                # If it overlaps, try a few alternative positions (above, below, left, right)
                found_alt = False
                # 8 compass directions to try shifting the label
                angles = [np.pi/2, -np.pi/2, 0, np.pi, np.pi/4, -np.pi/4, 3*np.pi/4, -3*np.pi/4]
                dist = fs * 1.2
                
                for angle in angles:
                    ax = x + np.cos(angle) * dist
                    ay = y + np.sin(angle) * dist
                    
                    a_left, a_top, a_right, a_bottom = draw.textbbox((ax, ay), label, font=font)
                    a_curr_bbox = (a_left - buffer, a_top - buffer, a_right + buffer, a_bottom + buffer)
                    
                    a_overlap = False
                    for other_bbox in occupied_bboxes:
                        if not (a_curr_bbox[2] < other_bbox[0] or 
                                a_curr_bbox[0] > other_bbox[2] or 
                                a_curr_bbox[3] < other_bbox[1] or 
                                a_curr_bbox[1] > other_bbox[3]):
                            a_overlap = True
                            break
                    
                    if not a_overlap:
                        draw.text((ax, ay), label, fill=self.params.text_color, font=font)  # type: ignore[attr-defined]
                        occupied_bboxes.append(a_curr_bbox)
                        found_alt = True
                        break
                
                # If still no room, we skip drawing this specific number to avoid
                # unreadable unholy messes, or we could draw it with high transparency.
                # For now, skipping is cleaner for professional output.
