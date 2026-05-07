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

        max_plausible_count: int = 600
        """Raise ``suspect_high_count`` if the detected colony count exceeds this.

        600 is a practical upper bound for manually-countable plates; above
        this the plate would typically be reported as TNTC (too numerous to
        count). Set to ``0`` to disable.
        """

        max_coverage_frac: float = 0.50
        """Raise ``suspect_high_count`` if detected colonies cover more than this
        fraction of the plate (or image) area.

        A normally-countable plate rarely exceeds 30% coverage; values above
        50% almost always indicate large numbers of false positives. Set to
        ``0`` to disable.
        """

    def extract(self, image: Any, mask: Any, data: ImageData) -> list[dict[str, Any]]:
        labels = mask
        if labels is None or labels.max() == 0:
            data.add_flag(
                "no_objects",
                "ColonyMeasurer found zero labelled regions.",
                severity="warning",
            )
            return []

        # Intensity image for basic props (roundness, etc)
        # regionprops wants either no intensity image or one that matches
        # ``labels.shape``. Our image may be RGB and/or float.
        is_color = image.ndim == 3 and image.shape[-1] in (3, 4)
        if is_color:
            color_img = image[:, :, :3]
            intensity = color.rgb2gray(color_img)
            hsv = color.rgb2hsv(color_img)
        else:
            intensity = util.img_as_float(image)
            # Try to find a color version of this same-sized image for HSV props
            # (e.g. stashed by IlluminationCorrection before grayscale conversion)
            hsv = None
            color_img = None
            pre_illum = data.artifacts.get("pre_illumination")
            if pre_illum is not None and pre_illum.shape[:2] == image.shape[:2] and pre_illum.ndim == 3:
                color_img = pre_illum[:, :, :3]
                hsv = color.rgb2hsv(color_img)

        h, w = labels.shape
        rows: list[dict[str, Any]] = []
        edge_touches = 0

        # Pre-calculate region properties for intensity and color
        props = measure.regionprops(labels, intensity_image=intensity)
        if hsv is not None:
            # We also want mean H, S, V. regionprops only takes one intensity image,
            # so we'll grab mean intensity for the color channels separately.
            h_props = measure.regionprops(labels, intensity_image=hsv[:, :, 0])
            s_props = measure.regionprops(labels, intensity_image=hsv[:, :, 1])
            v_props = measure.regionprops(labels, intensity_image=hsv[:, :, 2])

            # Map labels to their HSV means for quick lookup
            h_means = {p.label: p.intensity_mean for p in h_props}
            s_means = {p.label: p.intensity_mean for p in s_props}
            v_means = {p.label: p.intensity_mean for p in v_props}

            # Also get RGB means (use the 3-channel color_img)
            r_means = {p.label: p.intensity_mean for p in measure.regionprops(labels, intensity_image=color_img[:, :, 0])}
            g_means = {p.label: p.intensity_mean for p in measure.regionprops(labels, intensity_image=color_img[:, :, 1])}
            b_means = {p.label: p.intensity_mean for p in measure.regionprops(labels, intensity_image=color_img[:, :, 2])}
        else:
            h_means = s_means = v_means = {}
            r_means = g_means = b_means = {}

        for prop in props:
            min_row, min_col, max_row, max_col = prop.bbox
            touches_edge = min_row == 0 or min_col == 0 or max_row >= h or max_col >= w
            if touches_edge:
                edge_touches += 1
            # Circularity = 4π·area / perimeter² (1.0 = perfect circle).
            perim = float(prop.perimeter)
            circularity = float(4 * np.pi * prop.area / (perim * perim)) if perim > 0 else 0.0

            row = {
                "label": int(prop.label),
                # Preserve the original segmentation label so downstream
                # steps that reassign "label" (e.g. classify_by_interior)
                # don't break exporters that need to look up pixels in the
                # mask array.
                "segment_label": int(prop.label),
                "area_px": int(prop.area),
                "centroid_y": float(prop.centroid[0]),
                "centroid_x": float(prop.centroid[1]),
                "equivalent_diameter_px": float(prop.equivalent_diameter_area),
                "eccentricity": float(prop.eccentricity),
                "solidity": float(prop.solidity),
                "circularity": round(circularity, 4),
                "perimeter_px": round(perim, 2),
                "mean_intensity": float(prop.intensity_mean),
                "bbox_y0": int(min_row),
                "bbox_x0": int(min_col),
                "bbox_y1": int(max_row),
                "bbox_x1": int(max_col),
                "touches_edge": bool(touches_edge),
                # Default multiplicity — the estimate_multiplicity module
                # may upgrade this for bilobed / merged colonies.
                "colony_count_estimate": 1,
            }

            # Add color features if available
            if hsv is not None:
                l_id = prop.label
                row.update({
                    "mean_r": round(float(r_means[l_id]), 4),
                    "mean_g": round(float(g_means[l_id]), 4),
                    "mean_b": round(float(b_means[l_id]), 4),
                    "mean_h": round(float(h_means[l_id]), 4),
                    "mean_s": round(float(s_means[l_id]), 4),
                    "mean_v": round(float(v_means[l_id]), 4),
                })

            rows.append(row)

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

        self._check_plausibility(rows, data)
        return rows

    def _check_plausibility(self, rows: list[dict], data: ImageData) -> None:
        """Raise ``suspect_high_count`` if either plausibility threshold is exceeded."""
        n = len(rows)
        max_count: int = self.params.max_plausible_count  # type: ignore[attr-defined]
        max_cov: float = self.params.max_coverage_frac  # type: ignore[attr-defined]

        # Absolute count threshold.
        if max_count > 0 and n > max_count:
            data.add_flag(
                "suspect_high_count",
                f"Detected {n} colonies, which exceeds max_plausible_count={max_count}. "
                "Results may contain many false positives (rim artifacts, noise). "
                "Check the annotated image and consider increasing margin_frac "
                "or the circularity/solidity filters.",
                severity="warning",
            )
            return  # one flag is enough; skip coverage check

        # Coverage fraction threshold.
        if max_cov > 0 and n > 0:
            total_colony_area = sum(r["area_px"] for r in rows)
            # Use the plate mask area if available, otherwise the image area.
            plate_mask = data.masks.get("plate")
            if plate_mask is not None:
                ref_area = float(np.asarray(plate_mask, dtype=bool).sum())
            elif data.image is not None:
                ref_area = float(np.asarray(data.image).shape[0] * np.asarray(data.image).shape[1])
            else:
                ref_area = 0.0
            if ref_area > 0:
                coverage = total_colony_area / ref_area
                data.metadata["colony_coverage_frac"] = round(coverage, 4)
                if coverage > max_cov:
                    data.add_flag(
                        "suspect_high_count",
                        f"Detected colonies cover {coverage:.0%} of the plate area, "
                        f"which exceeds max_coverage_frac={max_cov:.0%}. "
                        "Results may contain large numbers of false positives. "
                        "Check the annotated image.",
                        severity="warning",
                    )
