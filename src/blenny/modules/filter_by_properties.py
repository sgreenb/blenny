"""Filter measurement rows by morphological properties.

Post-measurement filter that marks colonies as artifacts if they fall below
circularity, solidity, or area thresholds. Unlike threshold_segment (which
removes objects from the mask before measurement), this filter operates on
already-measured rows, making it pipeline-agnostic — it works with YOLO,
classic CV, or any other segmenter.
"""

from __future__ import annotations

from typing import Any

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


@register("filter_by_properties")
class PropertyFilter(Classifier):
    """Filter measurement rows by circularity, solidity, and area thresholds.

    Colonies that fall below any enabled threshold are marked as artifacts
    (``is_artifact=True``) and excluded from the count. Set a threshold to
    ``0`` (or ``0.0``) to disable it.
    """

    class Params(BlennyParams):
        min_circularity: float = 0.9
        """Drop regions whose circularity (4π·area / perimeter²) falls below
        this. A perfect disk has circularity 1.0. Set to ``0`` to disable."""

        min_solidity: float = 0.7
        """Drop regions whose solidity (area / convex-hull-area) falls below
        this. Compact, round colonies score near 1.0. Set to ``0`` to disable."""

        min_area_px: int | None = None
        """Drop labelled regions smaller than this many pixels.
        If ``None``, ``min_area_ppm`` is used instead."""

        min_area_ppm: int = 0
        """Drop regions smaller than this many parts-per-million of the ROI
        area. Only used if ``min_area_px`` is ``None``."""

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows:
            return rows

        min_circ: float = self.params.min_circularity
        min_sol: float = self.params.min_solidity
        min_area_px_val: int | None = self.params.min_area_px
        min_area_ppm_val: int = self.params.min_area_ppm

        n_rejected = 0
        reasons: list[str] = []

        for row in rows:
            # Skip rows already marked as artifacts by earlier steps
            if row.get("is_artifact"):
                continue

            drop = False
            drop_reasons: list[str] = []

            # Circularity check
            if min_circ > 0:
                circ = row.get("circularity", 1.0)
                if circ < min_circ:
                    drop = True
                    drop_reasons.append(f"circularity {circ:.3f} < {min_circ}")

            # Solidity check
            if not drop and min_sol > 0:
                sol = row.get("solidity", 1.0)
                if sol < min_sol:
                    drop = True
                    drop_reasons.append(f"solidity {sol:.3f} < {min_sol}")

            # Area check
            if not drop:
                if min_area_px_val is not None:
                    area = row.get("area_px", 0)
                    if area < min_area_px_val:
                        drop = True
                        drop_reasons.append(f"area {area}px < {min_area_px_val}px")
                elif min_area_ppm_val > 0:
                    area_ppm = row.get("area_ppm", 0)
                    if area_ppm < min_area_ppm_val:
                        drop = True
                        drop_reasons.append(f"area {area_ppm}ppm < {min_area_ppm_val}ppm")

            if drop:
                row["is_artifact"] = True
                row["artifact_reason"] = "; ".join(drop_reasons)
                reasons.append(row["artifact_reason"])
                n_rejected += 1

        if n_rejected:
            from blenny.modules.classify_interior import InteriorColonyClassifier

            InteriorColonyClassifier.update_count(rows, data)
            InteriorColonyClassifier.reassign_ids(rows, data)

            data.add_flag(
                "property_filter_rejections",
                f"PropertyFilter rejected {n_rejected} object(s): "
                + "; ".join(reasons[:5])
                + (f" ... and {n_rejected - 5} more" if n_rejected > 5 else ""),
                severity="info",
            )

        return rows
