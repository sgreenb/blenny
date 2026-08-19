"""Post-detection size/shape filtering for YOLO (and other) pipelines.

``threshold_segment`` applies its ``min_area`` / ``min_circularity`` filters
*during* segmentation, but the YOLO detector produces a label mask without
those filters. This classifier applies the same style of filters to existing
measurement rows, so small debris and elongated rim fragments can be excluded
from counts without re-segmenting.

Like :class:`~blenny.modules.classify_interior.InteriorColonyClassifier`,
detections are **marked** ``is_artifact=True`` (not deleted), an
``artifact_reason`` is recorded, and ``colony_count`` in metadata is updated
to reflect only the surviving detections.

Pipeline position: after ``measure_colonies``, before ``classify_by_interior``
(and after ``estimate_multiplicity`` if present — merged-colony detections are
exempt from the circularity filter because fused colonies are legitimately
non-round).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


@register("filter_colonies")
class ColonyFilter(Classifier):
    """Drop colonies that are too small or not round enough."""

    class Params(BlennyParams):
        min_area_ppm: int | None = None
        """Drop detections smaller than this many parts-per-million of the ROI
        (plate) area. ``None`` (the default) disables the area filter.
        A standard 90 mm plate is ~6300 mm²; 100 ppm ≈ 0.6 mm²."""

        min_area_px: int | None = None
        """Drop detections smaller than this many pixels. Overrides
        ``min_area_ppm`` when both are set."""

        min_circularity: float | None = None
        """Drop detections whose circularity (4π·area/perimeter², 1.0 = perfect
        circle) falls below this value. ``None`` or ``0`` disables the filter."""

        min_solidity: float | None = None
        """Drop detections whose solidity (area / convex-hull area) falls below
        this value. ``None`` or ``0`` disables the filter."""

        roi_mask_key: str = "plate"
        """Key in ``data.masks`` whose area is the denominator for the ppm
        calculation. Falls back to the image area if absent."""

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows:
            return rows

        min_a: int | None = self.params.min_area_px  # type: ignore[attr-defined]
        if min_a is None and self.params.min_area_ppm:  # type: ignore[attr-defined]
            roi_area = self._roi_area(data, rows)
            min_a = max(1, int((self.params.min_area_ppm * roi_area) / 1_000_000))  # type: ignore[attr-defined]

        min_circ: float | None = self.params.min_circularity or None  # type: ignore[attr-defined]
        min_sol: float | None = self.params.min_solidity or None  # type: ignore[attr-defined]

        if min_a is None and min_circ is None and min_sol is None:
            return rows  # nothing to filter

        n_filtered = 0
        for row in rows:
            if row.get("is_artifact"):
                continue  # already excluded by an earlier step

            reasons: list[str] = []

            if min_a is not None:
                area = row.get("area_px")
                if isinstance(area, (int, float)) and float(area) < min_a:
                    reasons.append(f"area {area:.0f} px < min {min_a} px")

            # Merged-colony detections are legitimately non-round; exempt them
            # from the shape filters so estimate_multiplicity isn't undone.
            is_merged = int(row.get("colony_count_estimate", 1)) >= 2
            if not is_merged:
                if min_circ is not None:
                    circ = row.get("circularity")
                    if isinstance(circ, (int, float)) and float(circ) < min_circ:
                        reasons.append(f"circularity {float(circ):.2f} < {min_circ}")
                if min_sol is not None:
                    sol = row.get("solidity")
                    if isinstance(sol, (int, float)) and float(sol) < min_sol:
                        reasons.append(f"solidity {float(sol):.2f} < {min_sol}")

            if reasons:
                row["is_artifact"] = True
                row["artifact_reason"] = "filter_colonies: " + "; ".join(reasons)
                n_filtered += 1

        if n_filtered:
            data.add_flag(
                "colonies_filtered",
                f"ColonyFilter removed {n_filtered} detection(s) as too small "
                "or not round enough. They remain in the CSV with "
                "is_artifact=True for inspection.",
                severity="info",
            )

        # Keep metadata counts and label numbering consistent (mirrors
        # classify_by_interior).
        from blenny.modules.classify_interior import InteriorColonyClassifier

        InteriorColonyClassifier.update_count(rows, data)
        return self._renumber(rows, data)

    def _roi_area(self, data: ImageData, rows: list[dict[str, Any]]) -> float:
        mask_key: str = self.params.roi_mask_key  # type: ignore[attr-defined]
        mask = data.masks.get(mask_key)
        if mask is not None:
            return float(np.asarray(mask, dtype=bool).sum())
        if data.image is not None:
            return float(np.asarray(data.image).shape[0] * np.asarray(data.image).shape[1])
        # Extremely defensive: derive from the first row's bbox area.
        for r in rows:
            if "bbox_y0" in r and "bbox_y1" in r and "bbox_x0" in r and "bbox_x1" in r:
                return float((r["bbox_y1"] - r["bbox_y0"]) * (r["bbox_x1"] - r["bbox_x0"]))
        return 1.0

    @staticmethod
    def _renumber(rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        """Reassign contiguous IDs: surviving colonies 1..N, artifacts after."""
        from blenny.modules.classify_interior import InteriorColonyClassifier

        return InteriorColonyClassifier.reassign_ids(rows, data)
