"""Threshold + (optional) watershed segmentation for blob-like objects.

Otsu is the default — robust for bimodal histograms and parameter-free.
For uneven contrast where Otsu fails, switch to ``method="local"``.
Touching colonies are split via distance-transform watershed when
``split_touching=True`` (the default).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import ndimage as ndi
from skimage import color, feature, filters, measure, morphology, segmentation, util

from blenny.pipeline import BlennyParams, ImageData, Segmenter, register


@register("threshold_segment")
class ThresholdSegmenter(Segmenter):
    class Params(BlennyParams):
        output_key: str = "objects"
        """Key in ``data.masks`` for the resulting label image."""

        roi_mask_key: str | None = "plate"
        """If set and present in ``data.masks``, restrict segmentation to this region."""

        method: Literal["otsu", "local"] = "otsu"
        block_size: int = 51
        """Window size for ``method='local'``. Must be odd."""

        min_area: int = 10
        """Drop labelled regions smaller than this many pixels."""

        max_area_frac: float = 0.25
        """Drop regions covering more than this fraction of the ROI (likely artefacts)."""

        split_touching: bool = True
        """Apply distance-transform watershed to split touching objects."""

        peak_min_distance: int = 5
        """Minimum separation (px) between watershed seeds."""

    def segment(self, image: Any, data: ImageData) -> Any:
        work = color.rgb2gray(image) if image.ndim == 3 else util.img_as_float(image)

        # Threshold.
        if self.params.method == "otsu":  # type: ignore[attr-defined]
            t = filters.threshold_otsu(work)
            binary = work > t
        else:
            block = self.params.block_size  # type: ignore[attr-defined]
            if block % 2 == 0:
                block += 1
            t = filters.threshold_local(work, block_size=block, offset=0)
            binary = work > t

        # Restrict to ROI if requested.
        roi_key = self.params.roi_mask_key  # type: ignore[attr-defined]
        if roi_key and roi_key in data.masks:
            roi = data.masks[roi_key].astype(bool)
            if roi.shape == binary.shape:
                binary = binary & roi

        # Clean small specks.
        binary = morphology.remove_small_objects(binary, min_size=self.params.min_area)  # type: ignore[attr-defined]
        binary = morphology.opening(binary, morphology.disk(1))

        # Label, optionally splitting touching objects via watershed.
        if self.params.split_touching:  # type: ignore[attr-defined]
            distance = ndi.distance_transform_edt(binary)
            assert distance is not None  # for type-checkers
            coords = feature.peak_local_max(
                distance,
                min_distance=self.params.peak_min_distance,  # type: ignore[attr-defined]
                labels=binary,
            )
            seed_mask = np.zeros(distance.shape, dtype=bool)
            if len(coords) > 0:
                seed_mask[tuple(coords.T)] = True
            markers = measure.label(seed_mask)
            labels = segmentation.watershed(-distance, markers, mask=binary)
        else:
            labels = measure.label(binary)

        # Drop oversized blobs (likely the plate edge or a smudge).
        if labels.max() > 0:
            roi_area = float(binary.sum()) if binary.any() else float(binary.size)
            max_area = self.params.max_area_frac * roi_area  # type: ignore[attr-defined]
            for prop in measure.regionprops(labels):
                if prop.area > max_area:
                    labels[labels == prop.label] = 0
            # Re-pack labels so they're contiguous 1..N.
            labels = measure.label(labels > 0)

        return labels.astype(np.int32)
