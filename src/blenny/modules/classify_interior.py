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

        # ---- Blank-plate / noise-dominated guard ---------------------------
        # When the IQR reference can't be trusted (too few interior samples,
        # or the whole detection set looks like noise), we fall back to a
        # purely-geometric strict filter rather than passing everything
        # through. This is what keeps a blank negative control from being
        # reported as 100+ "colonies".
        degenerate_max_median_eccentricity: float = 0.55
        """If the median eccentricity of *all* detections exceeds this, the
        plate is treated as degenerate (likely empty / noise-dominated).
        Real colonies are nearly round (typical median eccentricity 0.2-0.4);
        agar texture and crack fragments are elongated.
        """

        degenerate_max_area_cv: float = 1.2
        """If the coefficient of variation (std/mean) of detection areas
        exceeds this, the plate is treated as degenerate. Real colonies
        cluster tightly in size on a given plate (CV typically 0.3-0.7);
        a CV above ~1 means the size distribution is dominated by
        outliers / noise.
        """

        strict_fallback_max_eccentricity: float = 0.55
        """In strict-fallback mode, every detection above this eccentricity
        is marked as an artifact. Bilobed colonies typically have
        eccentricity 0.6-0.85, so they are also rejected here — they will
        be recovered later by the multiplicity estimator (TODO).
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
            # Re-label anyway just in case measure_colonies left a mess
            return self._reassign_ids(rows, data)

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

        # --- 3. Check whether we can trust an IQR reference at all ---
        # If interior is too small OR the detection set as a whole looks
        # noise-dominated (most blobs elongated, sizes wildly variable),
        # using interior as a reference would just enshrine noise. Fall
        # back to a strict purely-geometric filter instead.
        min_n: int = self.params.min_interior_samples  # type: ignore[attr-defined]
        self._last_global_stats: dict[str, Any] = {}
        is_degenerate, degen_reason = self._plate_looks_degenerate(rows)
        if self._last_global_stats:
            data.metadata["detection_global_stats"] = self._last_global_stats
        insufficient = len(interior) < min_n

        if insufficient or is_degenerate:
            n_rejected = self._apply_strict_fallback(rows)
            if is_degenerate:
                data.add_flag(
                    "plate_likely_empty",
                    f"Plate appears empty or noise-dominated ({degen_reason}); "
                    f"applied strict shape fallback (eccentricity \u2264 "
                    f"{self.params.strict_fallback_max_eccentricity}). "  # type: ignore[attr-defined]
                    f"{n_rejected} detection(s) marked as artifacts.",
                    severity="warning",
                )
            else:  # insufficient interior, but plate doesn't look degenerate
                data.add_flag(
                    "interior_classifier_insufficient_samples",
                    f"Only {len(interior)} interior detection(s) found "
                    f"(need {min_n}); applied strict shape fallback. "
                    f"{n_rejected} detection(s) marked as artifacts.",
                    severity="info",
                )
            self._update_count(rows, data)
            return self._reassign_ids(rows, data)

        # --- 4. Fit reference distribution from interior colonies ---
        ref = self._fit_reference(interior)
        data.metadata["interior_reference_stats"] = {
            feat: {k: round(v, 4) for k, v in bounds.items()} for feat, bounds in ref.items()
        }
        data.metadata["interior_n"] = len(interior)
        data.metadata["edge_zone_n"] = len(edge)

        # --- 5. Score edge-zone detections ---
        # Detections already tagged as merged colonies (count_estimate >= 2)
        # bypass classification: an upstream geometric step has vouched for
        # them, and they would otherwise look like area outliers and be
        # rejected here.
        n_rejected = 0
        for row in edge:
            if int(row.get("colony_count_estimate", 1)) >= 2:
                continue
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

        return self._reassign_ids(rows, data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reassign_ids(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        """Re-assign IDs: colonies first (1..N), then artifacts (N+1..M).

        This ensures IDs in annotated images and tables match the count.
        """
        colonies = [r for r in rows if not r.get("is_artifact")]
        artifacts = [r for r in rows if r.get("is_artifact")]

        # Sort each group by their original label (or spatial position) to
        # maintain consistency across runs.
        def sort_key(row):
            return (row.get("centroid_y", 0), row.get("centroid_x", 0))

        colonies.sort(key=sort_key)
        artifacts.sort(key=sort_key)

        new_rows = []
        for i, row in enumerate(colonies, 1):
            row["label"] = i
            new_rows.append(row)

        n_colonies = len(colonies)
        for i, row in enumerate(artifacts, 1):
            row["label"] = n_colonies + i
            new_rows.append(row)

        # Replace data.measurements with the re-ordered and re-labeled list.
        data.measurements[:] = new_rows
        return data.measurements

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

    def _plate_looks_degenerate(self, rows: list[dict[str, Any]]) -> tuple[bool, str]:
        """Detect plates where the entire detection set looks like noise.

        Returns ``(is_degenerate, human_reason)``. The reason string is empty
        when the plate looks fine.
        """
        eccs = [
            float(r["eccentricity"])
            for r in rows
            if isinstance(r.get("eccentricity"), (int, float))
        ]
        areas = [float(r["area_px"]) for r in rows if isinstance(r.get("area_px"), (int, float))]
        if len(eccs) < 3 or len(areas) < 3:
            return True, f"only {len(eccs)} usable detection(s)"

        median_ecc = float(np.median(eccs))
        area_arr = np.array(areas, dtype=float)
        mean_a = float(area_arr.mean())
        area_cv = float(area_arr.std() / mean_a) if mean_a > 0 else float("inf")

        max_ecc: float = self.params.degenerate_max_median_eccentricity  # type: ignore[attr-defined]
        max_cv: float = self.params.degenerate_max_area_cv  # type: ignore[attr-defined]

        # Stash global stats so users (and the eval report) can see why
        # a plate was or wasn't flagged.
        self._last_global_stats = {
            "median_eccentricity": round(median_ecc, 4),
            "area_cv": round(area_cv, 4),
            "n_detections": len(eccs),
        }

        if median_ecc > max_ecc:
            return True, f"median eccentricity {median_ecc:.2f} > {max_ecc}"
        if area_cv > max_cv:
            return True, f"area coefficient-of-variation {area_cv:.2f} > {max_cv}"
        return False, ""

    def _apply_strict_fallback(self, rows: list[dict[str, Any]]) -> int:
        """Mark anything insufficiently round as an artifact. Returns count.

        Detections already tagged with ``colony_count_estimate >= 2`` are
        skipped: the multiplicity estimator has already vouched for them as
        merged colonies (which legitimately have eccentricity > 0.55).
        """
        max_ecc: float = self.params.strict_fallback_max_eccentricity  # type: ignore[attr-defined]
        n = 0
        for row in rows:
            if int(row.get("colony_count_estimate", 1)) >= 2:
                continue
            ecc = row.get("eccentricity")
            if isinstance(ecc, (int, float)) and float(ecc) > max_ecc:
                row["is_artifact"] = True
                row["artifact_reason"] = (
                    f"strict-fallback: eccentricity={float(ecc):.2f} > {max_ecc}"
                )
                n += 1
        return n

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
        """Rewrite colony_count in metadata to exclude artifacts.

        ``colony_count`` is the sum of ``colony_count_estimate`` across
        non-artifact rows so that merged-colony detections (tagged by
        ``estimate_multiplicity``) contribute their estimated multiplicity.
        ``detection_count`` is the raw row count for cross-reference.
        """
        kept = [r for r in rows if not r.get("is_artifact", False)]
        n_artifacts = sum(1 for r in rows if r.get("is_artifact", False))
        n_colonies = sum(int(r.get("colony_count_estimate", 1)) for r in kept)
        data.metadata["colony_count"] = n_colonies
        data.metadata["detection_count"] = len(kept)
        data.metadata["artifact_count"] = n_artifacts
