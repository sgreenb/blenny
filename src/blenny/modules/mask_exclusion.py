"""Subtract a manual exclusion mask from an existing ROI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from blenny.pipeline import BlennyParams, ImageData, Preprocessor, register


@register("apply_exclusion_mask")
class ExclusionMasker(Preprocessor):
    """Subtract a manually-painted exclusion mask from a target mask (e.g. 'plate')."""

    class Params(BlennyParams):
        mask_path: str | None = None
        """Path to a binary or grayscale image where non-zero pixels define
        areas to be EXCLUDED from the analysis.
        """

        target_mask_key: str = "plate"
        """The existing mask in ``data.masks`` to be modified. Usually 'plate'."""

    def process(self, image: Any, data: ImageData) -> Any:
        path_str = self.params.mask_path  # type: ignore[attr-defined]
        if path_str is None:
            return image

        path = Path(path_str)
        if not path.exists():
            data.add_flag(
                "exclusion_mask_missing",
                f"ExclusionMasker: could not find mask file at {path}",
                severity="warning",
            )
            return image

        # Load mask and convert to boolean (True = areas to EXCLUDE)
        with Image.open(path) as im:
            # The mask from the GUI is typically drawn on a resized version
            # of the original image. We first stretch it back to the
            # original image's dimensions to recover the global coordinate frame.
            orig = data.original_image
            if orig is None:
                orig = image  # Fallback

            orig_h, orig_w = orig.shape[:2]
            if im.size != (orig_w, orig_h):
                im = im.resize((orig_w, orig_h), Image.Resampling.NEAREST)

            mask_full = np.asarray(im.convert("L")) > 0

        # Now, if the image has been CROPPED (by detect_plate), we must crop
        # the mask using the same bounding box.
        bbox = data.metadata.get("plate_bbox")  # (y0, x0, y1, x1)
        if bbox is not None:
            y0, x0, y1, x1 = bbox
            mask_full = mask_full[y0:y1, x0:x1]

        # Finally, if the image has been RESIZED (by load_image max_dimension),
        # we must resize the (possibly cropped) mask to match the current 'image'.
        cur_h, cur_w = image.shape[:2]
        if mask_full.shape != (cur_h, cur_w):
            from skimage.transform import resize

            mask_arr = resize(mask_full, (cur_h, cur_w), order=0, anti_aliasing=False) > 0.5
        else:
            mask_arr = mask_full

        target_key = self.params.target_mask_key  # type: ignore[attr-defined]
        if target_key not in data.masks:
            data.add_flag(
                "exclusion_mask_no_target",
                f"ExclusionMasker: target mask '{target_key}' not found in ImageData.",
                severity="warning",
            )
            return image

        # The target mask (e.g. 'plate') defines where we WANT to count.
        # We logical-AND it with the INVERSE of the exclusion mask.
        data.masks[target_key] = data.masks[target_key].astype(bool) & ~mask_arr

        return image
