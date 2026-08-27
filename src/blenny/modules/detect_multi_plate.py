"""Find multiple circular Petri-dish regions in a grid arrangement."""

from __future__ import annotations

from typing import Any

from blenny.modules.detect_facile import facile_detection
from blenny.pipeline import BlennyParams, Field, ImageData, Preprocessor, register


@register("detect_multi_plate")
class MultiPlateDetector(Preprocessor):
    class Params(BlennyParams):
        grid: list[int] = Field(default_factory=lambda: [1, 1])
        """Number of [rows, cols] in the image."""

        labels: list[list[str]] | None = None
        """Optional 2D list of labels for each grid cell."""

        radius_scale: float = 1.0
        """Scale factor for the detected plate radius."""

        min_points: int | None = None
        """Minimum number of edge points required to accept a circle."""

        max_error: float | None = None
        """Maximum allowed standard deviation for circle fitting."""

        min_confidence_score: float = 0.0
        """Legacy parameter for compatibility with Hough-based templates."""

    def process(self, image: Any, data: ImageData) -> Any:
        rows, cols = self.params.grid
        h, w = image.shape[:2]

        cell_h = h // rows
        cell_w = w // cols

        # Padding ensures plates on sector lines are still detected properly
        padding_h = int(cell_h * 0.15)
        padding_w = int(cell_w * 0.15)

        found_plates = []

        # 1. Localized Search in Padded Sectors
        for r in range(rows):
            for c in range(cols):
                y0 = max(0, r * cell_h - padding_h)
                y1 = min(h, (r + 1) * cell_h + padding_h)
                x0 = max(0, c * cell_w - padding_w)
                x1 = min(w, (c + 1) * cell_w + padding_w)

                cell_img = image[y0:y1, x0:x1]

                # Use the fast facile detection in each cell
                # We expect exactly one plate per cell in this mode
                circles = facile_detection(
                    cell_img,
                    petri_only=True,
                    min_points=self.params.min_points,
                    max_error=self.params.max_error,
                )

                label = self._get_label(r, c)

                if circles:
                    # facile sorts by radius descending, take the largest one in the cell
                    xc_c, yc_c, r_raw, _std, _n = circles[0]

                    found_plates.append(
                        {
                            "label": label,
                            "cy_global": yc_c + y0,
                            "cx_global": xc_c + x0,
                            "radius": r_raw,
                            "grid_pos": (r, c),
                        }
                    )
                else:
                    data.add_flag(
                        "plate_not_found",
                        f"No plate found in slot {label} at grid position ({r}, {c}).",
                        severity="warning",
                    )

        if not found_plates:
            data.add_flag(
                "no_plates_found", "MultiPlateDetector found zero plates.", severity="error"
            )
            return image

        # 3. Construct final ROIs
        rois = []
        for p in found_plates:
            r_raw = p["radius"]
            cy, cx = p["cy_global"], p["cx_global"]

            scale: float = self.params.radius_scale
            r_eff = max(1, round(r_raw * scale))
            r = max(r_raw, r_eff)

            buffer = int(r * 0.05) + 20
            y0_crop, y1_crop = max(0, int(cy - r - buffer)), min(h, int(cy + r + buffer + 1))
            x0_crop, x1_crop = max(0, int(cx - r - buffer)), min(w, int(cx + r + buffer + 1))

            rois.append(
                {
                    "label": p["label"],
                    "bbox": (int(y0_crop), int(x0_crop), int(y1_crop), int(x1_crop)),
                    "center_local": (int(cy - y0_crop), int(cx - x0_crop)),
                    "radius": int(r),
                    "radius_eff": int(r_eff),
                }
            )

        data.metadata["rois"] = rois
        data.metadata["multi_plate_mode"] = True
        return image

    def _get_label(self, row: int, col: int) -> str:
        if (
            self.params.labels
            and row < len(self.params.labels)
            and col < len(self.params.labels[row])
        ):
            return self.params.labels[row][col]

        # Default: 1, 2, 3...
        cols = self.params.grid[1]
        return str(row * cols + col + 1)
