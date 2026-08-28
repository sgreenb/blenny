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
            # 1. Determine the reference frame of the mask.
            # In the GUI, masks are saved at the resolution of the original image.
            # We use 'original_size_wh' from metadata if available.
            full_w, full_h = data.metadata.get("original_size_wh", (None, None))

            if full_w is not None and (im.size != (full_w, full_h)):
                # Resize the global mask to match the full image frame
                im = im.resize((full_w, full_h), Image.Resampling.NEAREST)

            mask_array = np.asarray(im.convert("L")) > 0

        # 2. If we are in a ROI (cropped by detect_plate or sub_pipeline),
        # crop the global mask to match the ROI's bounding box.
        bbox = data.metadata.get("plate_bbox")  # (y0, x0, y1, x1)
        if bbox is not None:
            y0, x0, y1, x1 = [int(v) for v in bbox]
            # Boundary safety
            mh, mw = mask_array.shape
            y0, y1 = max(0, y0), min(mh, y1)
            x0, x1 = max(0, x0), min(mw, x1)
            mask_array = mask_array[y0:y1, x0:x1]

        # 3. Finally, resize the (possibly cropped) mask to match the current 'image'.
        # This handles cases where load_image or sub_pipeline performed resizing.
        cur_h, cur_w = image.shape[:2]
        if mask_array.shape != (cur_h, cur_w):
            from skimage.transform import resize

            mask_array = resize(mask_array, (cur_h, cur_w), order=0, anti_aliasing=False) > 0.5

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
        target_mask = data.masks[target_key].astype(bool)

        # Final safety check before bitwise operation
        if target_mask.shape != mask_array.shape:
            from skimage.transform import resize

            mask_array = resize(mask_array, target_mask.shape, order=0, anti_aliasing=False) > 0.5

        data.masks[target_key] = target_mask & ~mask_array

        # Keep the exclusion shape around (in the same frame as the target
        # mask) so exporters can draw its boundary on annotated output -- the
        # carved plate mask alone loses the exclusion geometry.
        data.masks["exclusion"] = mask_array

        return image
