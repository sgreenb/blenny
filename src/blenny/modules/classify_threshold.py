"""Multi-class classification based on measurement thresholds.

This module allows users to define rules for categorizing colonies based on
any measurement (e.g., area, intensity, color). Categorized colonies can be
assigned a custom label and a custom outline color for the annotated export.

Example pipeline.yaml:
----------------------
- name: classify_by_threshold
  params:
    rules:
      - feature: "mean_v"
        min: 0.7
        label: "bright"
        color: [0, 255, 0]  # Green
      - feature: "mean_h"
        min: 0.5
        max: 0.7
        label: "blue"
        color: [0, 0, 255]  # Blue
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from blenny.pipeline import BlennyParams, Classifier, ImageData, register


class ThresholdRule(BlennyParams):
    feature: str
    """The measurement column to check (e.g., 'mean_v', 'area_px')."""

    min: float | None = None
    """Minimum value (inclusive). If None, no lower bound is applied."""

    max: float | None = None
    """Maximum value (inclusive). If None, no upper bound is applied."""

    label: str
    """The classification label to assign (e.g., 'high_expressor')."""

    color: list[int] | None = Field(default=None, min_length=3, max_length=3)
    """Optional [R, G, B] color for annotated outlines (0-255). Must be exactly 3 values."""


@register("classify_by_threshold")
class ThresholdClassifier(Classifier):
    """Categorize colonies using simple min/max thresholds on any feature."""

    class Params(BlennyParams):
        rules: list[ThresholdRule] = Field(default_factory=list)
        """List of threshold-based rules to apply in order."""

        default_label: str = "normal"
        """Label for colonies that don't match any rule."""

    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        if not rows:
            return rows

        for row in rows:
            # We add a 'classification' and 'class_color' column.
            # If multiple rules match, the FIRST one wins.
            match_label = self.params.default_label
            match_color = None

            for rule in self.params.rules:
                val = row.get(rule.feature)
                if val is None:
                    continue

                # Check bounds
                if rule.min is not None and val < rule.min:
                    continue
                if rule.max is not None and val > rule.max:
                    continue

                # Match found
                match_label = rule.label
                match_color = rule.color
                break

            row["classification"] = match_label
            if match_color:
                row["class_color"] = match_color

        # Update metadata summary with class counts
        counts: dict[str, int] = {}
        for row in rows:
            if not row.get("is_artifact", False):
                lbl = row.get("classification", self.params.default_label)
                counts[lbl] = counts.get(lbl, 0) + int(row.get("colony_count_estimate", 1))

        data.metadata["classification_counts"] = counts
        return rows
