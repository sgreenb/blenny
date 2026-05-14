"""Find multiple circular Petri-dish regions in a grid arrangement."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage import color, feature, transform

from blenny.pipeline import BlennyParams, Field, ImageData, Preprocessor, register


@register("detect_multi_plate")
class MultiPlateDetector(Preprocessor):
    class Params(BlennyParams):
        grid: list[int] = Field(default_factory=lambda: [1, 1])
        """Number of [rows, cols] in the image."""

        labels: list[list[str]] | None = None
        """Optional 2D list of labels for each grid cell."""

        min_radius_frac: float = 0.05
        """Minimum plate radius as a fraction of the grid cell's minimum dimension."""

        max_radius_frac: float = 0.8
        """Maximum plate radius as a fraction of the grid cell's minimum dimension."""

        n_radii: int = 32
        """How many radii to sample within the search range. Higher values improve centering."""

        canny_sigma: float = 1.5
        """Gaussian sigma for the Canny edge detector."""

        radius_scale: float = 1.0
        """Scale factor for the detected plate radius.
        Values < 1.0 act as a margin (shrink); values > 1.0 expand the area.
        """

        min_confidence_score: float = 0.20
        """Minimum Hough accumulator score to accept a plate."""

        size_consistency_limit: float = 0.20
        """Max allowed deviation from the median radius (0.20 = 20%)."""

        min_diameter_image_frac: float = 0.10
        """Minimum plate diameter as a fraction of the shortest image dimension (H or W).
        Helps filter out small artifacts or large colonies being misidentified as plates."""

        detection_dimension: int = 512
        """Internal resolution for plate detection in each grid cell.

        Finding the plate doesn't require full resolution. Downscaling each
        grid cell to this dimension (long side) makes the Hough transform
        much faster.
        """

        refine: bool = False
        """If True, perform a high-resolution circle refinement after the
        initial Hough detection. Improves centering but increases runtime
        significantly on high-res scans.
        """

    def process(self, image: Any, data: ImageData) -> Any:
        rows, cols = self.params.grid
        gray = color.rgb2gray(image) if image.ndim == 3 else image.astype(float) / 255.0
        h, w = gray.shape

        cell_h = h // rows
        cell_w = w // cols

        # Padding ensures plates on sector lines are still detected properly
        padding_h = int(cell_h * 0.15)
        padding_w = int(cell_w * 0.15)

        found_plates = []

        # 1. Localized Search in Padded Sectors
        for r in range(rows):
            for c in range(cols):
                # Calculate padded sector boundaries
                y0 = max(0, r * cell_h - padding_h)
                y1 = min(h, (r + 1) * cell_h + padding_h)
                x0 = max(0, c * cell_w - padding_w)
                x1 = min(w, (c + 1) * cell_w + padding_w)

                cell_gray_full = gray[y0:y1, x0:x1]
                ch_full, cw_full = cell_gray_full.shape

                # --- Downsample cell for detection speed ---
                det_dim: int = self.params.detection_dimension  # type: ignore[attr-defined]
                if det_dim > 0 and max(ch_full, cw_full) > det_dim:
                    det_scale = det_dim / max(ch_full, cw_full)
                    new_ch, new_cw = int(ch_full * det_scale), int(cw_full * det_scale)
                    cell_gray = transform.resize(
                        cell_gray_full, (new_ch, new_cw), order=1, anti_aliasing=False
                    )
                else:
                    cell_gray = cell_gray_full
                    det_scale = 1.0

                ch, cw = cell_gray.shape

                # Radius range relative to cell size
                rmin_abs = int((self.params.min_diameter_image_frac * min(h, w)) / 2 * det_scale)
                rmin_cell = int(self.params.min_radius_frac * min(ch, cw))
                rmin = max(rmin_abs, rmin_cell)
                rmax = int(self.params.max_radius_frac * min(ch, cw))

                radii = np.linspace(rmin, rmax, self.params.n_radii).astype(int)
                radii = np.unique(radii)

                edges = feature.canny(cell_gray, sigma=self.params.canny_sigma)
                hough = transform.hough_circle(edges, radii)

                # --- OPTIMIZED PEAK FINDING ---
                # Since we only ever want the single best circle per sector, we
                # can bypass the expensive skimage.transform.hough_circle_peaks
                # (which does non-maximum suppression) and just take the argmax.
                idx = np.argmax(hough)
                r_idx, cy_idx, cx_idx = np.unravel_index(idx, hough.shape)
                accum = hough[r_idx, cy_idx, cx_idx]

                label = self._get_label(r, c)

                if accum >= self.params.min_confidence_score:
                    # Found a potential plate via Hough. Map back to cell coords.
                    cy_h, cx_h, r_h = (
                        round(cy_idx / det_scale),
                        round(cx_idx / det_scale),
                        round(radii[r_idx] / det_scale),
                    )

                    # --- Sub-pixel Refinement via Least Squares ---
                    # Only performed if refine=True
                    if self.params.refine:
                        try:
                            from scipy import optimize

                            edges_full = feature.canny(
                                cell_gray_full, sigma=self.params.canny_sigma
                            )
                            ys_f, xs_f = np.where(edges_full)
                            dists = np.sqrt((ys_f - cy_h) ** 2 + (xs_f - cx_h) ** 2)
                            # Narrow band around the scaled hough radius
                            mask = (dists > r_h * 0.8) & (dists < r_h * 1.2)
                            if np.sum(mask) > 50:
                                pts_y, pts_x = ys_f[mask], xs_f[mask]

                                def f_circle(coords, pts_x=pts_x, pts_y=pts_y):
                                    xc, yc = coords
                                    ri = np.sqrt((pts_x - xc) ** 2 + (pts_y - yc) ** 2)
                                    return ri - ri.mean()

                                (cx_ref, cy_ref), _ = optimize.leastsq(
                                    f_circle, (float(cx_h), float(cy_h))
                                )
                                r_ref = np.sqrt(
                                    (pts_x - cx_ref) ** 2 + (pts_y - cy_ref) ** 2
                                ).mean()

                                # Update with refined coordinates
                                cy_h, cx_h, r_h = int(cy_ref), int(cx_ref), int(r_ref)
                        except Exception:
                            # Fallback to Hough if refinement fails
                            pass

                    found_plates.append(
                        {
                            "label": label,
                            "cy_global": cy_h + y0,
                            "cx_global": cx_h + x0,
                            "radius": r_h,
                            "score": float(accum),
                            "grid_pos": (r, c),
                        }
                    )
                else:
                    data.add_flag(
                        "plate_not_found",
                        f"No plate found in slot {label} at grid position ({r}, {c}).",
                        severity="warning",
                    )

        if not found_plates:
            data.add_flag(
                "no_plates_found", "MultiPlateDetector found zero plates.", severity="error"
            )
            return image

        # 2. Size consistency check
        radii = [p["radius"] for p in found_plates]
        median_r = np.median(radii)

        valid_plates = []
        for p in found_plates:
            deviation = abs(p["radius"] - median_r) / median_r
            if deviation <= self.params.size_consistency_limit:
                valid_plates.append(p)
            else:
                data.add_flag(
                    "plate_size_outlier",
                    f"Plate {p['label']} rejected: radius {p['radius']} deviates from median {median_r:.1f}.",
                    severity="warning",
                )

        # 3. Construct final ROIs
        rois = []
        for p in valid_plates:
            r_raw = p["radius"]
            cy, cx = p["cy_global"], p["cx_global"]

            # Apply radius scale factor
            scale: float = self.params.radius_scale  # type: ignore[attr-defined]
            r_eff = max(1, round(r_raw * scale))

            # Use larger of raw or effective radius for cropping to keep context
            r = max(r_raw, r_eff)

            # Global bounding box for cropping with a safety buffer
            # Use a slightly larger relative buffer
            buffer = int(r * 0.05) + 20
            y0_crop, y1_crop = max(0, cy - r - buffer), min(h, cy + r + buffer + 1)
            x0_crop, x1_crop = max(0, cx - r - buffer), min(w, cx + r + buffer + 1)

            rois.append(
                {
                    "label": p["label"],
                    "bbox": (int(y0_crop), int(x0_crop), int(y1_crop), int(x1_crop)),
                    "center_local": (int(cy - y0_crop), int(cx - x0_crop)),
                    "radius": int(r),
                    "radius_eff": int(r_eff),
                }
            )

        data.metadata["rois"] = rois
        data.metadata["multi_plate_mode"] = True

        return image

    def _get_label(self, row: int, col: int) -> str:
        if (
            self.params.labels
            and row < len(self.params.labels)
            and col < len(self.params.labels[row])
        ):
            return self.params.labels[row][col]

        # Default: 1, 2, 3...
        cols = self.params.grid[1]
        return str(row * cols + col + 1)
