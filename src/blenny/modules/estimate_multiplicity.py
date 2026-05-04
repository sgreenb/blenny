"""Estimate how many colonies a single detection actually contains.

When two or more colonies grow into each other on a plate, classical
segmentation typically captures them as one elongated, peanut-shaped blob.
A human glancing at the plate would still count them as two (or three).
This module reproduces that intuition with a transparent, purely geometric
heuristic: a detection that is unusually large *and* shaped like a fused
cluster of round colonies (high solidity, low circularity) is recorded as
representing multiple colonies via the ``colony_count_estimate`` column.

Pipeline position: after ``measure_colonies``, before ``classify_by_interior``.

Why before classification? The interior classifier rejects area-outliers as
artifacts. Without this step, every bilobed colony would be marked as an
artifact and dropped. Tagging multiplicity first lets the classifier (and
the strict-fallback path) skip these rows so they aren't double-judged.

Design notes
------------
* The reference for "what one colony looks like" is built from *clean
  singletons* on the same plate: high circularity, high solidity, area in a
  reasonable band. This is robust to global size differences between
  plates / organisms / incubation times.
* Estimates are clamped to a small integer range (default ``[2, 6]``).
  Going beyond that means the heuristic is being asked to do real
  segmentation, which is the job of a future watershed / ML stage.
* Detections that don't meet the merged-shape criteria keep
  ``colony_count_estimate = 1`` (the default set by ``measure_colonies``),
  so this module is safe to omit or re-run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


@register("estimate_multiplicity")
class MultiplicityEstimator(Classifier):
    """Tag merged-colony detections with ``colony_count_estimate >= 2``."""

    class Params(BlennyParams):
        # ---- "Clean singleton" definition (used to build the reference) ----
        singleton_min_circularity: float = 0.85
        """A detection is considered a clean singleton (used to compute the
        median single-colony area) only if its circularity meets this bound.
        Defaults are deliberately strict so that touching colonies don't
        contaminate the reference.
        """

        singleton_min_solidity: float = 0.92
        """Solidity bound for clean singletons (see above)."""

        min_singletons: int = 5
        """Minimum number of clean singletons needed to build a reliable
        reference. Below this, multiplicity estimation is skipped and a
        ``multiplicity_skipped_no_reference`` info flag is raised.
        """

        # ---- "Looks like merged colonies" criteria ------------------------
        merge_min_solidity: float = 0.85
        """A merged-colony blob is still mostly convex (concavities live at
        the joins between component colonies). Lacy / fragmented noise
        falls below this and is ignored.
        """

        merge_max_circularity: float = 0.75
        """Merged-colony blobs are noticeably less round than singletons
        (peanut, trefoil shapes). Real singletons typically have circularity
        > 0.85, so the gap between this bound and ``singleton_min_circularity``
        gives a clear "intermediate" zone we don't claim either way.
        """

        merge_min_area_ratio: float = 1.5
        """Detection area must be at least this multiple of the median
        singleton area to be considered a merged blob. Below 1.5x, area
        differences are within normal colony-to-colony variation.
        """

        max_count_estimate: int = 6
        """Cap on the per-detection multiplicity. Larger clusters are flagged
        but capped, because the round-by-area heuristic becomes unreliable
        for high-density mergers (a future watershed/ML segmenter would do
        better).
        """

        merge_max_eccentricity: float = 0.90
        """Merged colonies are elongated, but not infinitely so. Fragments of
        the plate rim or long scratches can have eccentricity > 0.95.
        """

    # ------------------------------------------------------------------

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows:
            return rows

        # 1. Identify clean singletons and compute the per-plate reference.
        singletons = [r for r in rows if self._is_clean_singleton(r)]
        min_n: int = self.params.min_singletons  # type: ignore[attr-defined]
        if len(singletons) < min_n:
            data.add_flag(
                "multiplicity_skipped_no_reference",
                f"Only {len(singletons)} clean singleton(s) found "
                f"(need {min_n}); multiplicity estimation skipped. "
                "All detections keep colony_count_estimate=1.",
                severity="info",
            )
            return rows

        single_areas = np.array([float(r["area_px"]) for r in singletons], dtype=float)
        median_single_area = float(np.median(single_areas))
        data.metadata["singleton_median_area_px"] = round(median_single_area, 2)
        data.metadata["singleton_n"] = len(singletons)

        # 2. Walk all rows and upgrade merged-shape detections.
        min_ratio: float = self.params.merge_min_area_ratio  # type: ignore[attr-defined]
        max_circ: float = self.params.merge_max_circularity  # type: ignore[attr-defined]
        min_sol: float = self.params.merge_min_solidity  # type: ignore[attr-defined]
        cap: int = self.params.max_count_estimate  # type: ignore[attr-defined]

        max_ecc: float = self.params.merge_max_eccentricity  # type: ignore[attr-defined]
        n_upgraded = 0
        total_extra = 0
        for row in rows:
            area = row.get("area_px")
            sol = row.get("solidity")
            circ = row.get("circularity")
            ecc = row.get("eccentricity")
            if not all(isinstance(v, (int, float)) for v in (area, sol, circ, ecc)):
                continue

            # Already covered by a clean-singleton check? Nothing to upgrade.
            if self._is_clean_singleton(row):
                continue

            ratio = float(area) / median_single_area if median_single_area > 0 else 0.0
            if (
                ratio >= min_ratio
                and float(sol) >= min_sol
                and float(circ) <= max_circ
                and float(ecc) <= max_ecc
            ):
                # Bias toward smaller counts: a true bilobed pair has area
                # ~1.7-2.0x a singleton (overlap reduces total), so 2.4x
                # should still count as 2, not 3. Using int(ratio + 0.2)
                # means we only call N colonies once area is at least
                # (N - 0.2) singletons.
                est = int(ratio + 0.2)
                est = max(2, min(cap, est))
                row["colony_count_estimate"] = est
                row["multiplicity_reason"] = (
                    f"merged-shape: area={ratio:.2f}x singleton, "
                    f"solidity={float(sol):.2f}, circularity={float(circ):.2f}, "
                    f"eccentricity={float(ecc):.2f}"
                )
                n_upgraded += 1
                total_extra += est - 1

        if n_upgraded:
            data.add_flag(
                "multiplicity_estimated",
                f"MultiplicityEstimator tagged {n_upgraded} detection(s) as "
                f"merged colonies, contributing {total_extra} extra colonies "
                f"to the count beyond the per-detection default.",
                severity="info",
            )
            data.metadata["multiplicity_upgraded_n"] = n_upgraded
            data.metadata["multiplicity_extra_colonies"] = total_extra

        return rows

    # ------------------------------------------------------------------

    def _is_clean_singleton(self, row: dict[str, Any]) -> bool:
        circ = row.get("circularity")
        sol = row.get("solidity")
        if not isinstance(circ, (int, float)) or not isinstance(sol, (int, float)):
            return False
        return (
            float(circ) >= self.params.singleton_min_circularity  # type: ignore[attr-defined]
            and float(sol) >= self.params.singleton_min_solidity  # type: ignore[attr-defined]
        )
