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

import math
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

        detection_dimension: int = 512
        """Internal resolution for plate detection.

        Finding the plate doesn't require full resolution. Downscaling the
        working image to this dimension (long side) makes the Hough transform
        orders of magnitude faster on large images.
        """

        crop: bool = True
        """If True, crop ``data.image`` to the plate's bounding box."""

        mask_key: str = "plate"
        """Key under which the plate mask is stored in ``data.masks``."""

        radius_scale: float = 1.0
        """Scale factor for the detected plate radius.

        Values < 1.0 act as a margin (shrink); values > 1.0 expand the area
        (useful for tilted camera angles).
        """

        force_cy: int | None = None
        """Force the plate center Y coordinate (bypasses detection)."""

        force_cx: int | None = None
        """Force the plate center X coordinate (bypasses detection)."""

        force_r: int | None = None
        """Force the plate radius (bypasses detection)."""

        force_mask_path: str | None = None
        """Path to a binary mask file to use as the plate ROI (bypasses detection)."""

        min_confidence_score: float = 0.25
        """Minimum Hough accumulator score to avoid a ``low_plate_confidence`` flag.

        1.0 means every pixel on the circle circumference was an edge. Real
        plates in variable lighting typically score 0.25-0.50.
        """

    def process(self, image: Any, data: ImageData) -> Any:
        # 1. Prepare grayscale image for processing
        gray_full = color.rgb2gray(image) if image.ndim == 3 else image.astype(float) / 255.0
        h_full, w_full = gray_full.shape

        # 2. Downsample for detection speed (if enabled)
        det_dim: int = self.params.detection_dimension  # type: ignore[attr-defined]
        if det_dim > 0 and max(h_full, w_full) > det_dim:
            # Calculate scale such that max dimension is det_dim
            det_scale = det_dim / max(h_full, w_full)
            new_h, new_w = int(h_full * det_scale), int(w_full * det_scale)
            # Use order=1 (bilinear) for speed; anti-aliasing is usually overkill for Hough
            gray = transform.resize(gray_full, (new_h, new_w), order=1, anti_aliasing=False)
        else:
            gray = gray_full
            det_scale = 1.0

        h, w = gray.shape

        # Scale forced coordinates if the image was resized during loading.
        # This ensures GUI-derived coords (from full-res display) match the
        # current 'image' frame.
        load_scale = data.metadata.get("resize_scale", 1.0)

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
        if (
            self.params.force_cy is not None
            and self.params.force_cx is not None
            and self.params.force_r is not None
        ):
            cy = round(self.params.force_cy * load_scale)
            cx = round(self.params.force_cx * load_scale)
            r_eff = round(self.params.force_r * load_scale)
            r = r_eff
            score = 1.0
            r_hough = r_eff
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
                data.masks[self.params.mask_key] = np.ones((h_full, w_full), dtype=bool)  # type: ignore[attr-defined]
                return image

            # Scale coordinates back to the input 'image' resolution
            cy, cx, r_hough = (
                int(round(cys[0] / det_scale)),
                int(round(cxs[0] / det_scale)),
                int(round(rs[0] / det_scale)),
            )
            score = float(accums[0])

            # --- Sub-pixel Refinement via Least Squares ---
            # We use the FULL RESOLUTION edges for refinement if available.
            # This recovers precision lost during the 512px downsampling.
            try:
                from scipy import optimize

                # Find edges on the original grayscale image
                edges_full = feature.canny(gray_full, sigma=self.params.canny_sigma)
                ys_f, xs_f = np.where(edges_full)

                # Narrow band around the scaled hough radius to find rim pixels
                dists = np.sqrt((ys_f - cy)**2 + (xs_f - cx)**2)
                rim_pixels_mask = (dists > r_hough * 0.8) & (dists < r_hough * 1.2)

                if np.sum(rim_pixels_mask) > 50:
                    pts_y, pts_x = ys_f[rim_pixels_mask], xs_f[rim_pixels_mask]

                    def f_circle(coords):
                        xc, yc = coords
                        ri = np.sqrt((pts_x - xc)**2 + (pts_y - yc)**2)
                        return ri - ri.mean()

                    (cx_ref, cy_ref), _ = optimize.leastsq(f_circle, (float(cx), float(cy)))
                    r_ref = np.sqrt((pts_x - cx_ref)**2 + (pts_y - cy_ref)**2).mean()

                    # Update with refined coordinates
                    cy, cx, r_hough = int(round(cy_ref)), int(round(cx_ref)), int(round(r_ref))
            except Exception:
                # Fallback to Hough if refinement fails (e.g. scipy not installed or no edges)
                pass

            if score < self.params.min_confidence_score:
                data.add_flag(
                    "low_plate_confidence",
                    f"Plate detection confidence is low (score={score:.2f}). "
                    "The detected region might not be the actual plate.",
                    severity="warning",
                )

            # Apply radius scale factor to the detected radius.
            # Scale < 1.0 acts as a margin (shrinking the analysis area).
            # Scale > 1.0 acts as expansion (recovering edges in off-axis shots).
            scale: float = self.params.radius_scale  # type: ignore[attr-defined]
            r_eff = max(1, round(r_hough * scale))

            # The bounding box for cropping uses the larger of the two to ensure
            # we don't lose image context when shrinking.
            r = max(r_hough, r_eff)

        # Build mask in the original frame.
        mask = np.zeros((h_full, w_full), dtype=bool)
        rr, cc = disk((cy, cx), r_eff, shape=(h_full, w_full))
        mask[rr, cc] = True

        if self.params.crop:  # type: ignore[attr-defined]
            y0, y1 = max(0, cy - r), min(h_full, cy + r + 1)
            x0, x1 = max(0, cx - r), min(w_full, cx + r + 1)
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
