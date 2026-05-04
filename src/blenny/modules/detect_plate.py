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

import math
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

        margin_frac: float = 0.08
        """Shrink the plate mask by this fraction of the (expanded) radius.

        Real plate photos almost always include a bright reflective rim that
        otherwise gets segmented as a chain of small false-positive arcs.
        8% is calibrated for the wide translucent plastic rims common in
        phone photos; reduce to ``0.03``-``0.05`` for flatbed scans or
        photos where colonies grow close to the rim, raise to ``0.12`` if
        rim contamination persists.
        """

        radius_expand_frac: float = 0.0
        """Expand the detected plate radius by this fraction before applying
        ``margin_frac``.

        Off-axis camera angles make the plate appear slightly elliptical, and
        Hough circle detection then fits a circle smaller than the true plate.
        Expanding the radius before margin trimming recovers colonies near the
        far edge that would otherwise fall outside the analysis region. The
        ``InteriorColonyClassifier`` (if present in the pipeline) cleans up
        any extra rim artifacts that the expansion introduces.

        Default ``0.0`` preserves backward compatibility. Set to ``0.05``
        for typical hand-held phone photos; ``0.10`` for clearly off-axis
        shots where the plate ellipse is visible.
        """

        force_cy: int | None = None
        """Force the plate center Y coordinate (bypasses detection)."""

        force_cx: int | None = None
        """Force the plate center X coordinate (bypasses detection)."""

        force_r: int | None = None
        """Force the plate radius (bypasses detection)."""

        force_mask_path: str | None = None
        """Path to a binary mask file to use as the plate ROI (bypasses detection)."""

    def process(self, image: Any, data: ImageData) -> Any:
        gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float) / 255.0
        h, w = gray.shape

        # Scale forced coordinates if the image was resized during loading.
        # This ensures GUI-derived coords (from full-res display) match the
        # current 'image' frame.
        scale = data.metadata.get("resize_scale", 1.0)

        # --- Case A: Manual Mask File ---
        if self.params.force_mask_path:
            from PIL import Image
            mask_im = Image.open(self.params.force_mask_path).convert("L")
            mask = np.asarray(mask_im) > 127
            if mask.shape != (h, w):
                # Resize if needed (e.g. if mask was drawn on a thumb)
                from skimage.transform import resize
                mask = resize(mask, (h, w), order=0, anti_aliasing=False)
            
            data.masks[self.params.mask_key] = mask
            # For non-circular plates, we estimate a "center" and "radius" 
            # so downstream radial modules (like InteriorClassifier) still work
            # somewhat logically, though they are optimized for circles.
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                r = int(math.sqrt(mask.sum() / math.pi))
                data.metadata["plate_center"] = (cy, cx)
                data.metadata["plate_radius"] = r
            return image

        # --- Case B: Forced Circle Coords ---
        if self.params.force_cy is not None and self.params.force_cx is not None and self.params.force_r is not None:
            cy = int(round(self.params.force_cy * scale))
            cx = int(round(self.params.force_cx * scale))
            r_hough = int(round(self.params.force_r * scale))
            score = 1.0
        else:
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

            cy, cx, r_hough = int(cys[0]), int(cxs[0]), int(rs[0])
            score = float(accums[0])

            if score < 0.35:
                data.add_flag(
                    "low_plate_confidence",
                    f"Plate detection confidence is low (score={score:.2f}). "
                    "The detected region might not be the actual plate.",
                    severity="warning",
                )

        # Apply optional expansion *before* margin trimming. The bbox, mask,
        # and stored ``plate_radius`` all use the expanded radius so downstream
        # geometry (e.g. InteriorColonyClassifier) sees a single consistent
        # "plate radius" derived from the user-tunable knobs.
        expand: float = self.params.radius_expand_frac  # type: ignore[attr-defined]
        r = r_hough + round(r_hough * expand)
        r_eff = max(1, r - round(r * self.params.margin_frac))  # type: ignore[attr-defined]

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
            data.metadata["plate_radius_hough"] = int(r_hough)
            data.metadata["plate_hough_score"] = score
            return cropped

        data.masks[self.params.mask_key] = mask  # type: ignore[attr-defined]
        data.metadata["plate_center"] = (int(cy), int(cx))
        data.metadata["plate_radius"] = int(r)
        data.metadata["plate_radius_hough"] = int(r_hough)
        data.metadata["plate_hough_score"] = score
        return image
