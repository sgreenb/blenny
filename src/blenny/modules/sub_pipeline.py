"""A container module that executes a sequence of steps for each ROI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from blenny.pipeline import BlennyParams, ImageData, Module, register
from blenny.pipeline.runner import Pipeline
import numpy as np
from skimage.draw import disk

@register("sub_pipeline")
class SubPipeline(Module):
    class Params(BlennyParams):
        steps: list[dict[str, Any]] = []
        """List of module configurations to run on each ROI."""

        roi_metadata_key: str = "rois"
        """Key in ``data.metadata`` where ROI definitions are stored."""

        max_subplate_dimension: int | None = None
        """If set, resize each sub-plate to this maximum dimension before analysis."""

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        # Pre-build the inner pipeline
        self._inner_pipeline = Pipeline.from_config(self.params.steps)

    def run(self, data: ImageData) -> ImageData:
        # Populate stem and output_dir if available from source or metadata
        if data.source and "stem" not in data.metadata:
            data.metadata["stem"] = Path(data.source).stem
        
        rois = data.metadata.get(self.params.roi_metadata_key, [])
        if not rois:
            return data

        all_measurements = []
        all_sub_results = []
        
        # Keep track of where sub-provenance starts in the parent list
        # Capture current provenance length to insert sub-provenance later
        from blenny.pipeline.context import ProvenanceRecord
        
        n_rois = len(rois)
        for i, roi in enumerate(rois):
            label = roi["label"]
            y0, x0, y1, x1 = roi["bbox"]
            
            # 1. Create a local context for this ROI
            source_img = data.original_image if data.original_image is not None else data.image
            h_orig, w_orig = source_img.shape[:2]
            h_work, w_work = data.image.shape[:2]
            scale_y, scale_x = h_orig / h_work, w_orig / w_work
            y0_hr, y1_hr = int(y0 * scale_y), int(y1 * scale_y)
            x0_hr, x1_hr = int(x0 * scale_x), int(x1 * scale_x)
            local_image = source_img[y0_hr:y1_hr, x0_hr:x1_hr]
            lh, lw = local_image.shape[:2]
            
            sub_scale_y, sub_scale_x = 1.0, 1.0
            max_dim: int | None = self.params.max_subplate_dimension # type: ignore[attr-defined]
            if max_dim and max(lh, lw) > max_dim:
                from skimage.transform import resize
                new_scale = max_dim / max(lh, lw)
                new_h, new_w = int(lh * new_scale), int(lw * new_scale)
                orig_dtype = local_image.dtype
                if orig_dtype == np.uint8:
                    local_image = (resize(local_image, (new_h, new_w), anti_aliasing=True) * 255).astype(np.uint8)
                else:
                    local_image = resize(local_image, (new_h, new_w), anti_aliasing=True)
                sub_scale_y, sub_scale_x = new_h / lh, new_w / lw
            
            lh_final, lw_final = local_image.shape[:2]
            local_mask = np.zeros((lh_final, lw_final), dtype=bool)
            cy_l, cx_l = int(roi["center_local"][0] * scale_y * sub_scale_y), int(roi["center_local"][1] * scale_x * sub_scale_x)
            r_eff = int(roi["radius_eff"] * max(scale_x, scale_y) * max(sub_scale_x, sub_scale_y))
            rr, cc = disk((cy_l, cx_l), r_eff, shape=(lh_final, lw_final))
            local_mask[rr, cc] = True
            
            sub_data = ImageData(
                source=f"{data.source} [{label}]",
                image=local_image,
                original_image=local_image,
            )
            sub_data.masks["plate"] = local_mask
            sub_data.metadata["plate_label"] = label
            sub_data.metadata["plate_center"] = (cy_l, cx_l)
            sub_data.metadata["plate_radius"] = int(roi["radius"] * max(scale_x, scale_y) * max(sub_scale_x, sub_scale_y))
            
            for key in ["output_dir", "stem"]:
                if key in data.metadata:
                    sub_data.metadata[key] = data.metadata[key]
            
            # 2. Run inner pipeline
            sub_data = self._inner_pipeline.run(sub_data)
            all_sub_results.append(sub_data)
            
            # 3. Map measurements back to global space
            for row in sub_data.measurements:
                row["plate_label"] = label
                if "centroid_row" in row:
                    row["centroid_row_global"] = (row["centroid_row"] / sub_scale_y) + y0_hr
                if "centroid_col" in row:
                    row["centroid_col_global"] = (row["centroid_col"] / sub_scale_x) + x0_hr
                if "y" in row:
                    row["y_global"] = (row["y"] / sub_scale_y) + y0_hr
                if "x" in row:
                    row["x_global"] = (row["x"] / sub_scale_x) + x0_hr
                all_measurements.append(row)
            
            # 4. Bubble up quality flags
            for flag in sub_data.quality_flags:
                flag.message = f"[{label}] {flag.message}"
                data.quality_flags.append(flag)

        # Harvesting sub-provenance for profiling
        for sub_res in all_sub_results:
            label = sub_res.metadata.get("plate_label", "unknown")
            for rec in sub_res.provenance:
                data.provenance.append(
                    ProvenanceRecord(
                        step=f"[{label}] {rec.step}",
                        module_class=rec.module_class,
                        params=rec.params,
                        duration_s=rec.duration_s
                    )
                )

        # Update global metadata for summary exporters
        data.metadata["colony_count"] = len([m for m in all_measurements if not m.get("is_artifact")])
        data.metadata["artifact_count"] = len([m for m in all_measurements if m.get("is_artifact")])
        data.metadata["multi_plate_results"] = all_sub_results

        per_plate_counts = {}
        for roi in rois:
            per_plate_counts[roi["label"]] = 0
            
        for m in all_measurements:
            if not m.get("is_artifact"):
                pl = m.get("plate_label", "unknown")
                per_plate_counts[pl] = per_plate_counts.get(pl, 0) + 1
        
        expected_labels = []
        detector_params = None
        for step in data.provenance:
            if step.module_class == "MultiPlateDetector":
                detector_params = step.params
                break
        
        if detector_params:
            rows_p, cols_p = detector_params.get("grid", [1, 1])
            labels_config = detector_params.get("labels")
            for r_p in range(rows_p):
                for c_p in range(cols_p):
                    if labels_config and r_p < len(labels_config) and c_p < len(labels_config[r_p]):
                        expected_labels.append(labels_config[r_p][c_p])
                    else:
                        expected_labels.append(str(r_p * cols_p + c_p + 1))
        
        ordered_per_plate_counts = {}
        for el in expected_labels:
            if el in per_plate_counts:
                ordered_per_plate_counts[el] = per_plate_counts[el]
            else:
                ordered_per_plate_counts[el] = "NA"

        data.metadata["per_plate_counts"] = ordered_per_plate_counts
        data.measurements.extend(all_measurements)
        
        return data
