"""Per-colony measurements from a label image."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import color, measure, util

from blenny.pipeline import BlennyParams, FeatureExtractor, ImageData, register


@register("measure_colonies")
class ColonyMeasurer(FeatureExtractor):
    class Params(BlennyParams):
        mask_key: str = "objects"
        edge_touch_flag_threshold: float = 0.10
        """Raise an ImageData-level flag if more than this fraction of objects touch the image edge."""

    def extract(self, image: Any, mask: Any, data: ImageData) -> list[dict[str, Any]]:
        labels = mask
        if labels is None or labels.max() == 0:
            data.add_flag(
                "no_objects",
                "ColonyMeasurer found zero labelled regions.",
                severity="warning",
            )
            return []

        # regionprops wants either no intensity image or one that matches
        # ``labels.shape``. Our image may be RGB and/or float.
        intensity = color.rgb2gray(image) if image.ndim == 3 else util.img_as_float(image)

        h, w = labels.shape
        rows: list[dict[str, Any]] = []
        edge_touches = 0
        for prop in measure.regionprops(labels, intensity_image=intensity):
            min_row, min_col, max_row, max_col = prop.bbox
            touches_edge = min_row == 0 or min_col == 0 or max_row >= h or max_col >= w
            if touches_edge:
                edge_touches += 1
            rows.append(
                {
                    "label": int(prop.label),
                    "area_px": int(prop.area),
                    "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]),
                    "equivalent_diameter_px": float(prop.equivalent_diameter_area),
                    "eccentricity": float(prop.eccentricity),
                    "mean_intensity": float(prop.intensity_mean),
                    "bbox_y0": int(min_row),
                    "bbox_x0": int(min_col),
                    "bbox_y1": int(max_row),
                    "bbox_x1": int(max_col),
                    "touches_edge": bool(touches_edge),
                }
            )

        if rows:
            frac = edge_touches / len(rows)
            if frac > self.params.edge_touch_flag_threshold:  # type: ignore[attr-defined]
                data.add_flag(
                    "many_edge_touches",
                    f"{edge_touches}/{len(rows)} objects touch the image edge "
                    f"({frac:.0%}); counts may be unreliable.",
                    severity="warning",
                )

        # Summary stats stashed in metadata for quick access by exporters.
        if rows:
            areas = np.array([r["area_px"] for r in rows], dtype=float)
            data.metadata["colony_count"] = len(rows)
            data.metadata["area_px_mean"] = float(areas.mean())
            data.metadata["area_px_median"] = float(np.median(areas))
        else:
            data.metadata["colony_count"] = 0

        return rows
