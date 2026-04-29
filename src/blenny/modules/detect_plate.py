"""Find the circular Petri-dish region in a plate photo.

This is a Preprocessor (it can crop/mask ``data.image``) that *also*
writes a binary plate mask into ``data.masks``. Most preprocessors only
touch the image, but coupling crop+mask here keeps geometry consistent:
if we crop the image, the mask is in the cropped frame too.

Algorithm: Canny edges → Hough circle transform across a radius range
derived from image size. The single highest-accumulator circle wins.
If detection fails, a quality flag is raised and the image is left
untouched (downstream steps then operate on the whole frame).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import color, feature, transform
from skimage.draw import disk

from blenny.pipeline import BlennyParams, ImageData, Preprocessor, register


@register("detect_plate")
class PlateDetector(Preprocessor):
    class Params(BlennyParams):
        min_radius_frac: float = 0.20
        """Minimum plate radius as a fraction of ``min(H, W)``."""

        max_radius_frac: float = 0.50
        """Maximum plate radius as a fraction of ``min(H, W)``."""

        n_radii: int = 16
        """How many radii to sample within the search range."""

        canny_sigma: float = 2.0
        """Gaussian sigma for the Canny edge detector."""

        crop: bool = True
        """If True, crop ``data.image`` to the plate's bounding box."""

        mask_key: str = "plate"
        """Key under which the plate mask is stored in ``data.masks``."""

        margin: int = 4
        """Pixels to shrink the plate mask by, to avoid edge artefacts."""

    def process(self, image: Any, data: ImageData) -> Any:
        gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float) / 255.0
        h, w = gray.shape
        edges = feature.canny(gray, sigma=self.params.canny_sigma)  # type: ignore[attr-defined]

        rmin = int(self.params.min_radius_frac * min(h, w))  # type: ignore[attr-defined]
        rmax = int(self.params.max_radius_frac * min(h, w))  # type: ignore[attr-defined]
        radii = np.linspace(rmin, rmax, self.params.n_radii).astype(int)  # type: ignore[attr-defined]
        radii = np.unique(radii)

        hough = transform.hough_circle(edges, radii)
        accums, cxs, cys, rs = transform.hough_circle_peaks(hough, radii, total_num_peaks=1)

        if len(accums) == 0:
            data.add_flag(
                "plate_not_found",
                "PlateDetector could not locate a circular plate; "
                "downstream steps will run on the full image.",
                severity="warning",
            )
            # Still write a "everything" mask so FeatureExtractors don't break.
            data.masks[self.params.mask_key] = np.ones((h, w), dtype=bool)  # type: ignore[attr-defined]
            return image

        cy, cx, r = int(cys[0]), int(cxs[0]), int(rs[0])
        r_eff = max(1, r - self.params.margin)  # type: ignore[attr-defined]

        # Build mask in the original frame.
        mask = np.zeros((h, w), dtype=bool)
        rr, cc = disk((cy, cx), r_eff, shape=(h, w))
        mask[rr, cc] = True

        if self.params.crop:  # type: ignore[attr-defined]
            y0, y1 = max(0, cy - r), min(h, cy + r + 1)
            x0, x1 = max(0, cx - r), min(w, cx + r + 1)
            cropped = image[y0:y1, x0:x1]
            cropped_mask = mask[y0:y1, x0:x1]
            data.masks[self.params.mask_key] = cropped_mask  # type: ignore[attr-defined]
            data.metadata["plate_bbox"] = (int(y0), int(x0), int(y1), int(x1))
            data.metadata["plate_center"] = (int(cy), int(cx))
            data.metadata["plate_radius"] = int(r)
            return cropped

        data.masks[self.params.mask_key] = mask  # type: ignore[attr-defined]
        data.metadata["plate_center"] = (int(cy), int(cx))
        data.metadata["plate_radius"] = int(r)
        return image
