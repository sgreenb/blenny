"""Filter out specific colonies by their ID (label).

Used for manual researcher-in-the-loop interventions where specific detections
can be excluded from the final count.
"""

from __future__ import annotations

from typing import Any

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


@register("filter_by_id")
class IDFilter(Classifier):
    """Exclude specific detections by their ID number."""

    class Params(BlennyParams):
        exclude_ids: list[int] = []
        """List of colony IDs (labels) to mark as artifacts."""

        reason: str = "Manual exclusion"
        """The reason string to put in the artifact_reason column."""

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows or not self.params.exclude_ids:
            return rows

        exclude_set = set(self.params.exclude_ids)
        n_filtered = 0

        for row in rows:
            label = row.get("label")
            if label in exclude_set:
                row["is_artifact"] = True
                row["artifact_reason"] = self.params.reason
                n_filtered += 1

        if n_filtered:
            from blenny.modules.classify_interior import InteriorColonyClassifier
            InteriorColonyClassifier.update_count(rows, data)
            InteriorColonyClassifier.reassign_ids(rows, data)

            data.add_flag(
                "manual_exclusions",
                f"Researcher manually excluded {n_filtered} colony/ies by ID.",
                severity="info",
            )

        return rows
