"""Threshold + (optional) watershed segmentation for blob-like objects.

Otsu is the default — robust for bimodal histograms and parameter-free.
For uneven contrast where Otsu fails, switch to ``method="local"``.
Touching colonies are split via distance-transform watershed when
``split_touching=True`` (the default).
"""

from __future__ import annotations

import math
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

        min_area: int | None = None
        """Drop labelled regions smaller than this many pixels.
        If ``None``, ``min_area_ppm`` is used instead.
        """

        min_area_ppm: int = 15
        """Drop regions smaller than this many parts-per-million of the ROI area.
        Only used if ``min_area`` is ``None``. A standard 90mm plate is ~6300mm2;
        100ppm is ~0.6mm2 (a small but clear colony).
        """

        max_area_frac: float = 0.25
        """Drop regions covering more than this fraction of the ROI (likely artefacts)."""

        min_circularity: float = 0.75
        """Drop regions whose 4π·area / perimeter² falls below this.

        A perfect disk has circularity 1.0. The plate-rim arcs and pen scribbles
        that plague real plate photos are long-and-thin and score well below 0.5.
        Set to ``0`` to disable the filter (e.g. when measuring elongated yeast
        cells or filaments).
        """

        min_solidity: float = 0.90
        """Drop regions whose area / convex-hull-area falls below this.

        Compact, round colonies have solidity near 1.0. Irregular pen marks,
        scratches, and merged colony chains have lower solidity. Set to ``0``
        to disable.
        """

        split_touching: bool = True
        """Apply distance-transform watershed to split touching objects."""

        peak_min_distance: int | None = None
        """Minimum separation (px) between watershed seeds.

        If ``None`` (the default), it scales with the image size:
        ``max(2, min(H, W) // 400)``. This is a conservative heuristic
        intended to prevent over-splitting in high-res images while remaining
        effective for small images.
        """

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
        roi_area = float(binary.size)
        if roi_key and roi_key in data.masks:
            roi = data.masks[roi_key].astype(bool)
            if roi.shape == binary.shape:
                binary = binary & roi
                roi_area = float(roi.sum())

        # Determine min_area threshold.
        min_a: int = self.params.min_area  # type: ignore[attr-defined]
        if min_a is None:
            # ppm = (px / total_px) * 1e6  => px = (ppm * total_px) / 1e6
            min_a = int((self.params.min_area_ppm * roi_area) / 1_000_000)  # type: ignore[attr-defined]
            min_a = max(1, min_a)  # Never 0.

        # Clean small specks.
        # skimage 0.26+ deprecated min_size in favor of max_size (same meaning:
        # the maximum size of an object that is still considered "small" and
        # thus removed).
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_a)
        except TypeError:
            # Fallback for skimage < 0.26
            binary = morphology.remove_small_objects(binary, min_size=min_a)
        binary = morphology.opening(binary, morphology.disk(1))

        # Label, optionally splitting touching objects via watershed.
        if self.params.split_touching:  # type: ignore[attr-defined]
            distance = ndi.distance_transform_edt(binary)
            assert distance is not None  # for type-checkers

            # Determine peak_min_distance: explicit param > scale-aware default.
            pmd = self.params.peak_min_distance  # type: ignore[attr-defined]
            if pmd is None:
                pmd = max(2, min(binary.shape) // 400)

            coords = feature.peak_local_max(
                distance,
                min_distance=pmd,
                labels=binary,
            )
            seed_mask = np.zeros(distance.shape, dtype=bool)
            if len(coords) > 0:
                seed_mask[tuple(coords.T)] = True
            markers = measure.label(seed_mask)
            labels = segmentation.watershed(-distance, markers, mask=binary)
        else:
            labels = measure.label(binary)

        # Drop oversized / non-circular / irregular blobs (rim arcs, smudges,
        # pen marks). Each filter is independently disable-able by setting its
        # threshold to 0.
        if labels.max() > 0:
            max_area = self.params.max_area_frac * roi_area  # type: ignore[attr-defined]
            min_circ: float = self.params.min_circularity  # type: ignore[attr-defined]
            min_sol: float = self.params.min_solidity  # type: ignore[attr-defined]

            n_rejected_circ = 0
            for prop in measure.regionprops(labels):
                drop = False
                if prop.area > max_area or prop.area < min_a:
                    drop = True
                elif min_circ > 0 and prop.perimeter > 0:
                    circ = 4.0 * math.pi * prop.area / (prop.perimeter * prop.perimeter)
                    if circ < min_circ:
                        drop = True
                        n_rejected_circ += 1
                if not drop and min_sol > 0 and prop.solidity < min_sol:
                    drop = True
                if drop:
                    labels[labels == prop.label] = 0

            if n_rejected_circ > 20:
                data.add_flag(
                    "many_low_circularity_rejected",
                    f"ThresholdSegmenter rejected {n_rejected_circ} objects due to low circularity. "
                    "This often indicates plate-rim contamination or high noise.",
                    severity="warning",
                )

            # Re-pack labels so they're contiguous 1..N.
            labels = measure.label(labels > 0)

        return labels.astype(np.int32)
