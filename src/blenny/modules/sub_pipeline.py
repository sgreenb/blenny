"""A container module that executes a sequence of steps for each ROI."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import disk

from blenny.pipeline import BlennyParams, Exporter, Field, ImageData, Module, register
from blenny.pipeline.runner import Pipeline


@register("sub_pipeline")
class SubPipeline(Module):
    class Params(BlennyParams):
        steps: list[dict[str, Any]] = Field(default_factory=list)
        """List of module configurations to run on each ROI."""

        roi_metadata_key: str = "rois"
        """Key in ``data.metadata`` where ROI definitions are stored."""

        max_subplate_dimension: int | None = None
        """If set, resize each sub-plate to this maximum dimension before analysis."""

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        # Pre-build the inner pipeline
        self._inner_pipeline = Pipeline.from_config(self.params.steps)

    def run(self, data: ImageData, **kwargs: Any) -> ImageData:
        rois = data.metadata.get(self.params.roi_metadata_key, [])
        if not rois:
            return data

        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            parent_cb = progress_callback

            def _sub_progress(current: int, total: int, name: str, depth: int = 0) -> None:
                # Nested steps are reported one level deeper than the parent
                # pipeline's, so UIs (e.g. the GUI status log) can indent
                # sub-pipeline progress beneath the sub_pipeline step itself.
                parent_cb(current, total, name, depth + 1)

            progress_callback = _sub_progress
        all_measurements = []
        all_sub_results = []

        # Create global masks for the parent image (working resolution)
        h_work, w_work = data.image.shape[:2]
        global_labels = np.zeros((h_work, w_work), dtype=np.int32)
        global_plate_mask = np.zeros((h_work, w_work), dtype=bool)
        next_global_id = 1

        # Keep track of where sub-provenance starts in the parent list
        # Capture current provenance length to insert sub-provenance later
        from blenny.pipeline.context import ProvenanceRecord

        for _i, roi in enumerate(rois):
            label = roi["label"]
            y0, x0, y1, x1 = [int(v) for v in roi["bbox"]]
            target_h, target_w = y1 - y0, x1 - x0

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
            max_dim: int | None = self.params.max_subplate_dimension  # type: ignore[attr-defined]
            if max_dim and max(lh, lw) > max_dim:
                from skimage.transform import resize

                new_scale = max_dim / max(lh, lw)
                new_h, new_w = int(lh * new_scale), int(lw * new_scale)
                orig_dtype = local_image.dtype
                if orig_dtype == np.uint8:
                    local_image = (
                        resize(local_image, (new_h, new_w), anti_aliasing=True) * 255
                    ).astype(np.uint8)
                else:
                    local_image = resize(local_image, (new_h, new_w), anti_aliasing=True)
                sub_scale_y, sub_scale_x = new_h / lh, new_w / lw

            lh_final, lw_final = local_image.shape[:2]
            cy_l, cx_l = (
                int(roi["center_local"][0] * scale_y * sub_scale_y),
                int(roi["center_local"][1] * scale_x * sub_scale_x),
            )
            # Prefer the parent's plate mask (cropped to this ROI's bbox) so
            # manual polygon plates keep their drawn shape; fall back to a
            # synthetic disk for ROIs without a matching parent mask (e.g.
            # auto-detected multi-plate runs).
            parent_plate = data.masks.get("plate")
            if parent_plate is not None and parent_plate.shape[:2] == (h_work, w_work):
                roi_plate = parent_plate[y0:y1, x0:x1]
                if roi_plate.shape != (lh_final, lw_final):
                    from skimage.transform import resize

                    local_mask = (
                        resize(
                            roi_plate.astype(float),
                            (lh_final, lw_final),
                            order=0,
                            preserve_range=True,
                            anti_aliasing=False,
                        )
                        > 0.5
                    )
                else:
                    local_mask = roi_plate.copy()
            else:
                local_mask = np.zeros((lh_final, lw_final), dtype=bool)
                r_eff = int(
                    roi["radius_eff"] * max(scale_x, scale_y) * max(sub_scale_x, sub_scale_y)
                )
                rr_m, cc_m = disk((cy_l, cx_l), r_eff, shape=(lh_final, lw_final))
                local_mask[rr_m, cc_m] = True

            # Map sub-plate mask into global plate mask
            if (lh_final, lw_final) != (target_h, target_w):
                from skimage.transform import resize

                plate_mask_orig = resize(
                    local_mask.astype(float),
                    (target_h, target_w),
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(bool)
            else:
                plate_mask_orig = local_mask
            global_plate_mask[y0:y1, x0:x1] |= plate_mask_orig

            sub_data = ImageData(
                source=f"{data.source} [{label}]",
                image=local_image,
                original_image=local_image,
            )
            sub_data.masks["plate"] = local_mask
            sub_data.metadata["plate_label"] = label
            sub_data.metadata["plate_center"] = (cy_l, cx_l)
            # The centre/radius above are already expressed in the sub-image's
            # own (local) frame. Downstream modules (e.g. classify_by_interior)
            # must NOT subtract ``plate_bbox`` from them -- that offset belongs
            # to the ORIGINAL image frame (see the comment below on plate_bbox)
            # and is only applied when a detector writes the centre in the
            # original frame (detect_plate / detect_facile crop mode).
            sub_data.metadata["plate_center_local"] = True
            sub_data.metadata["plate_radius"] = int(
                roi["radius"] * max(scale_x, scale_y) * max(sub_scale_x, sub_scale_y)
            )

            for key in ["output_dir", "stem", "original_size_wh", "plate_shape"]:
                if key in data.metadata:
                    sub_data.metadata[key] = data.metadata[key]

            # Record the ROI's bounding box in the original image coordinate frame.
            # This allows modules like ExclusionMasker to correctly crop global masks.
            # NOTE: plate_center/plate_radius are LOCAL (see plate_center_local
            # above), so plate_bbox must only be used for original-frame masks.
            sub_data.metadata["plate_bbox"] = (y0_hr, x0_hr, y1_hr, x1_hr)

            # 2. Run inner pipeline. Give each sub-plate its own debug subdir
            # so per-plate step images don't overwrite each other.
            sub_debug_dir = None
            if kwargs.get("debug_dir"):
                sub_debug_dir = Path(kwargs["debug_dir"]) / str(label)
            sub_data = self._inner_pipeline.run(
                sub_data,
                progress_callback=progress_callback,
                output_dir=kwargs.get("output_dir"),
                debug_dir=sub_debug_dir,
            )
            all_sub_results.append(sub_data)

            # Map sub-labels into the global mask if present
            sub_mask_key = "objects"  # Default, could be parameterized
            local_to_global = {}
            if sub_mask_key in sub_data.masks:
                sub_labels = sub_data.masks[sub_mask_key]
                # Map local sub_labels to global labels at the correct offset
                sub_h, sub_w = sub_labels.shape

                if (sub_h, sub_w) != (target_h, target_w):
                    from skimage.transform import resize

                    sub_labels_orig = resize(
                        sub_labels.astype(float),
                        (target_h, target_w),
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False,
                    ).astype(np.int32)
                else:
                    sub_labels_orig = sub_labels

                unique_labels = np.unique(sub_labels_orig)
                unique_labels = unique_labels[unique_labels > 0]

                # Shift IDs and write to global mask
                for sub_id in unique_labels:
                    global_labels[y0:y1, x0:x1][sub_labels_orig == sub_id] = next_global_id
                    local_to_global[sub_id] = next_global_id
                    next_global_id += 1

            # 3. Map measurements back to global space (original image resolution)
            for row in sub_data.measurements:
                # We deepcopy the row to ensure no shared mutable state (like dicts/lists)
                # between the sub-plate results and the global combined result.
                global_row = copy.deepcopy(row)
                global_row["plate_label"] = label

                # Update segment_label to match the global mask we just built
                # We find the global ID by matching the local ID.
                # (We could have stored a map, but this is fine for N < 1000)
                sub_id = row.get("segment_label") or row.get("label")
                if sub_id in local_to_global:
                    global_row["segment_label"] = local_to_global[sub_id]

                # Map centroids
                for key in ["centroid_y", "centroid_x"]:
                    if key in global_row:
                        s = sub_scale_y if "y" in key else sub_scale_x
                        p = scale_y if "y" in key else scale_x
                        o = y0 if "y" in key else x0
                        global_row[key] = (global_row[key] / s) / p + o

                        # Also map _global centroids
                        offset_hr = y0_hr if "y" in key else x0_hr
                        global_row[f"{key}_global"] = (row[key] / s) + offset_hr

                # Map bounding boxes
                for key in ["bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"]:
                    if key in global_row:
                        s = sub_scale_y if "y" in key else sub_scale_x
                        p = scale_y if "y" in key else scale_x
                        o = y0 if "y" in key else x0
                        global_row[key] = (global_row[key] / s) / p + o

                        # Also map _global bounding boxes
                        offset_hr = y0_hr if "y" in key else x0_hr
                        global_row[f"{key}_global"] = (row[key] / s) + offset_hr

                all_measurements.append(global_row)

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
                        duration_s=rec.duration_s,
                    )
                )

        # Update global metadata for summary exporters. Counts sum
        # colony_count_estimate so merged-colony detections (tagged by
        # estimate_multiplicity inside the sub-pipeline) contribute their
        # estimated multiplicity, matching single-plate behaviour.
        data.metadata["colony_count"] = sum(
            int(m.get("colony_count_estimate", 1))
            for m in all_measurements
            if not m.get("is_artifact")
        )
        data.metadata["artifact_count"] = len([m for m in all_measurements if m.get("is_artifact")])
        data.metadata["multi_plate_results"] = all_sub_results

        # Write the combined masks to the parent data
        data.masks["objects"] = global_labels
        data.masks["plate"] = global_plate_mask

        per_plate_counts = {}
        for roi in rois:
            per_plate_counts[roi["label"]] = 0

        for m in all_measurements:
            if not m.get("is_artifact"):
                pl = m.get("plate_label", "unknown")
                per_plate_counts[pl] = per_plate_counts.get(pl, 0) + int(
                    m.get("colony_count_estimate", 1)
                )

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
        else:
            # Plates were found by a non-grid detector (e.g. detect_facile):
            # order by the ROI labels so per_plate_counts is still reported.
            expected_labels = [str(r["label"]) for r in rois]

        ordered_per_plate_counts = {}
        for el in expected_labels:
            ordered_per_plate_counts[el] = per_plate_counts.get(el, "NA")

        data.metadata["per_plate_counts"] = ordered_per_plate_counts
        data.measurements.extend(all_measurements)

        # Write the combined mask to the parent data
        data.masks["objects"] = global_labels

        return data

    def export(self, data: ImageData) -> None:
        """Re-run exporters in the inner pipeline for all sub-results.

        The GUI's "Save/Update All results" button swaps the parent's
        ``output_dir`` before calling this; re-stamp it onto every sub-result
        so per-plate exports land in the requested folder rather than the
        original analysis folder.
        """
        all_sub_results = data.metadata.get("multi_plate_results", [])
        parent_out = data.metadata.get("output_dir")
        for sub_data in all_sub_results:
            if parent_out is not None:
                sub_data.metadata["output_dir"] = parent_out
            for step in self._inner_pipeline.steps:
                if isinstance(step, Exporter):
                    step.export(sub_data)
