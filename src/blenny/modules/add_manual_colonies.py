"""Manually add colony detections to a mask.

Used for researcher-in-the-loop intervention where colonies missed by the 
auto-segmenter can be force-added by coordinate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.draw import disk
from skimage.measure import label

from blenny.pipeline import BlennyParams, ImageData, Preprocessor, register


@register("add_manual_colonies")
class ManualColonyAdder(Preprocessor):
    """Add circular colonies at specific coordinates to the objects mask."""

    class Params(BlennyParams):
        coordinates: list[tuple[int, int]] = []
        """List of (x, y) pixel coordinates where colonies should be added."""

        radius: int = 15
        """The radius (in pixels) of the circular colonies to draw."""

        mask_key: str = "objects"
        """The mask in ``data.masks`` to modify. Usually 'objects'."""

    def process(self, image: Any, data: ImageData) -> Any:
        if not self.params.coordinates:
            return image

        mask_key: str = self.params.mask_key  # type: ignore[attr-defined]
        h, w = image.shape[:2]

        if mask_key not in data.masks:
            # Create a blank mask if it doesn't exist
            mask = np.zeros((h, w), dtype=int)
        else:
            mask = data.masks[mask_key].copy()

        # Scale coordinates if the image was resized
        scale = data.metadata.get("resize_scale", 1.0)
        
        # We need to be careful with coordinate frames.
        # If the image was cropped, coordinates from the GUI (original scale) 
        # need to be shifted.
        bbox = data.metadata.get("plate_bbox") # (y0, x0, y1, x1)
        y_off, x_off = 0, 0
        if bbox is not None:
            y_off, x_off = bbox[0], bbox[1]

        added_count = 0
        for x, y in self.params.coordinates:
            # 1. Scale from original to current load scale
            sx, sy = x * scale, y * scale
            # 2. Shift if cropped
            sx, sy = sx - x_off, sy - y_off
            
            rr, cc = disk((int(sy), int(sx)), self.params.radius, shape=(h, w))
            mask[rr, cc] = 1 # Mark as foreground
            added_count += 1

        # If it was a label image, we should re-label it to ensure 
        # the new blobs get their own IDs.
        if mask.max() > 1 or mask.dtype.kind in ('i', 'u'):
            mask = label(mask > 0)
        else:
            mask = mask > 0

        data.masks[mask_key] = mask
        
        if added_count:
            data.add_flag(
                "manual_inclusions",
                f"Researcher manually added {added_count} colony/ies.",
                severity="info",
            )

        return image
