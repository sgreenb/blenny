"""Interior-anchored artifact rejection.

Every plate image contains its own ground truth: the colonies in the safe,
unambiguous interior of the plate. This classifier uses them to build a
per-image reference profile (size, intensity, eccentricity) and then
evaluates detections in the outer edge zone against it. Edge detections
that don't match the profile are marked as likely rim or crack artifacts.

Pipeline position: after ``measure_colonies``, before exporters.

Design notes
------------
* Detections are *marked*, not deleted. Every detection appears in the
  output CSV with ``is_artifact`` (bool) and ``artifact_reason`` (str)
  columns so researchers can audit what was rejected and why.
* ``colony_count`` in ``data.metadata`` is updated to reflect only
  non-artifact detections, so downstream summary tools see the corrected
  count automatically.
* The reference model is IQR-based rather than Gaussian because colony
  size distributions are right-skewed (roughly log-normal) and sample
  sizes are small (5-50). IQR with a configurable multiplier is robust
  and familiar to biologists (it's the boxplot rule).
* Interior colonies are *never* reclassified as artifacts — only edge-zone
  detections are evaluated. This means the classifier cannot over-reject
  on a plate where all colonies happen to be near the edge; instead it
  raises ``interior_classifier_insufficient_samples``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


@register("classify_by_interior")
class InteriorColonyClassifier(Classifier):
    """Validate edge-zone detections against interior colony characteristics."""

    class Params(BlennyParams):
        interior_radius_frac: float = 0.75
        """Detections within this fraction of the plate radius from its centre
        are treated as interior (trusted reference). Those beyond it form the
        edge zone and are scored against the reference.

        0.75 = inner 75% of the plate area. The outer 25% (radially) covers
        the rim band without being so conservative that dense plates lose
        many real edge colonies.
        """

        iqr_multiplier: float = 2.0
        """Acceptable range = [Q1 - k·IQR, Q3 + k·IQR] where k is this value.

        The standard Tukey boxplot rule is 1.5 (identifies mild outliers).
        2.0 is more permissive — only clear outliers are rejected. Raise to
        3.0 for maximum tolerance; lower to 1.5 to be stricter.
        """

        min_interior_samples: int = 5
        """Minimum interior detections needed to fit a reliable reference
        distribution. If fewer are found, classification is skipped and an
        ``interior_classifier_insufficient_samples`` info flag is raised.
        """

        features: list[str] = ["area_px", "mean_intensity", "eccentricity"]  # noqa: RUF012
        """Measurement columns to include in the reference model.

        ``area_px`` is the most discriminative: rim fragments are typically
        far smaller than real colonies. ``mean_intensity`` catches specular
        highlights. ``eccentricity`` catches elongated arc fragments.
        """

        plate_mask_key: str = "plate"
        """Key in ``data.masks`` used as a fallback for plate geometry when
        ``plate_center`` / ``plate_radius`` are absent from metadata.
        """

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows:
            return rows

        # --- 1. Plate geometry in the current (possibly-cropped) frame ---
        cy, cx, radius = self._plate_geometry(data)

        # Initialise / reset columns on every row so the CSV schema is always
        # consistent and the step is idempotent (safe to run twice).
        for row in rows:
            row["is_artifact"] = False  # always reset; set to True below if rejected
            row["artifact_reason"] = ""
            row.setdefault("normalized_dist", None)
            row.setdefault("zone", "unknown")

        if cy is None:
            data.add_flag(
                "interior_classifier_no_geometry",
                "InteriorColonyClassifier could not determine plate centre/radius; "
                "skipping edge classification. Ensure detect_plate ran before this step.",
                severity="warning",
            )
            return rows

        # --- 2. Compute radial position for every detection ---
        for row in rows:
            dy = (row.get("centroid_y") or 0.0) - cy
            dx = (row.get("centroid_x") or 0.0) - cx
            nd = math.sqrt(dy * dy + dx * dx) / radius
            row["normalized_dist"] = round(nd, 4)
            row["zone"] = (
                "interior"
                if nd <= self.params.interior_radius_frac  # type: ignore[attr-defined]
                else "edge"
            )

        interior = [r for r in rows if r["zone"] == "interior"]
        edge = [r for r in rows if r["zone"] == "edge"]

        # --- 3. Check we have enough interior samples ---
        min_n: int = self.params.min_interior_samples  # type: ignore[attr-defined]
        if len(interior) < min_n:
            data.add_flag(
                "interior_classifier_insufficient_samples",
                f"Only {len(interior)} interior detection(s) found "
                f"(need {min_n}); edge-zone classification skipped. "
                "All detections are kept as-is.",
                severity="info",
            )
            self._update_count(rows, data)
            return rows

        # --- 4. Fit reference distribution from interior colonies ---
        ref = self._fit_reference(interior)
        data.metadata["interior_reference_stats"] = {
            feat: {k: round(v, 4) for k, v in bounds.items()} for feat, bounds in ref.items()
        }
        data.metadata["interior_n"] = len(interior)
        data.metadata["edge_zone_n"] = len(edge)

        # --- 5. Score edge-zone detections ---
        n_rejected = 0
        for row in edge:
            reasons = self._score(row, ref)
            if reasons:
                row["is_artifact"] = True
                row["artifact_reason"] = "; ".join(reasons)
                n_rejected += 1

        # --- 6. Update colony count and raise info flag ---
        self._update_count(rows, data)

        if n_rejected:
            data.add_flag(
                "artifacts_removed",
                f"InteriorColonyClassifier removed {n_rejected} edge-zone "
                f"detection(s) as likely rim or crack artifacts based on "
                f"interior colony characteristics. They remain in the CSV "
                f"with is_artifact=True for inspection.",
                severity="info",
            )

        return rows

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plate_geometry(
        self, data: ImageData
    ) -> tuple[float, float, float] | tuple[None, None, None]:
        """Return (center_y, center_x, radius) in the current image frame."""
        plate_center = data.metadata.get("plate_center")
        plate_radius = data.metadata.get("plate_radius")
        plate_bbox = data.metadata.get("plate_bbox")  # (y0, x0, y1, x1)

        if plate_center is not None and plate_radius is not None:
            orig_cy, orig_cx = plate_center
            r = float(plate_radius)
            if plate_bbox is not None:
                y0, x0 = plate_bbox[0], plate_bbox[1]
                orig_cy = float(orig_cy) - y0
                orig_cx = float(orig_cx) - x0
            return float(orig_cy), float(orig_cx), r

        # Fallback: derive geometry from the plate mask pixel coordinates.
        mask_key: str = self.params.plate_mask_key  # type: ignore[attr-defined]
        plate_mask = data.masks.get(mask_key)
        if plate_mask is not None:
            arr = np.asarray(plate_mask, dtype=bool)
            if arr.any():
                ys, xs = np.where(arr)
                cy_f = float(ys.mean())
                cx_f = float(xs.mean())
                r_f = math.sqrt(float(arr.sum()) / math.pi)
                return cy_f, cx_f, r_f

        return None, None, None

    def _fit_reference(self, interior: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        """Build per-feature IQR-based bounds from interior detections."""
        ref: dict[str, dict[str, float]] = {}
        features: list[str] = self.params.features  # type: ignore[attr-defined]
        k: float = self.params.iqr_multiplier  # type: ignore[attr-defined]

        for feat in features:
            vals = [float(r[feat]) for r in interior if isinstance(r.get(feat), (int, float))]
            if len(vals) < 2:
                continue
            arr = np.array(vals)
            q1, med, q3 = (
                float(np.percentile(arr, 25)),
                float(np.median(arr)),
                float(np.percentile(arr, 75)),
            )
            iqr = q3 - q1
            # Guard against zero IQR: use 10% of median as a minimum spread
            # so that perfectly uniform interior detections don't trigger
            # rejection of every single edge colony.
            min_iqr = max(abs(med) * 0.10, 1e-6)
            iqr = max(iqr, min_iqr)
            ref[feat] = {
                "median": med,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lo": q1 - k * iqr,
                "hi": q3 + k * iqr,
            }
        return ref

    def _score(
        self,
        row: dict[str, Any],
        ref: dict[str, dict[str, float]],
    ) -> list[str]:
        """Return a list of human-readable rejection reasons, or [] if accepted."""
        reasons: list[str] = []
        for feat, bounds in ref.items():
            val = row.get(feat)
            if not isinstance(val, (int, float)):
                continue
            v = float(val)
            if v < bounds["lo"] or v > bounds["hi"]:
                reasons.append(f"{feat}={v:.2f} outside [{bounds['lo']:.2f}, {bounds['hi']:.2f}]")
        return reasons

    @staticmethod
    def _update_count(rows: list[dict[str, Any]], data: ImageData) -> None:
        """Rewrite colony_count in metadata to exclude artifacts."""
        n_colonies = sum(1 for r in rows if not r.get("is_artifact", False))
        n_artifacts = sum(1 for r in rows if r.get("is_artifact", False))
        data.metadata["colony_count"] = n_colonies
        data.metadata["artifact_count"] = n_artifacts
