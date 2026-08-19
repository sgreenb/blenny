"""Plate detection using the 'facile' algorithm.

This algorithm is faster than Hough Circles and can detect multiple plates
without requiring a grid specification.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from skimage.draw import disk

from blenny.pipeline import BlennyParams, ImageData, Preprocessor, register


def kasa_fit(points):
    """
    Kasa's method for Direct Least Squares Fitting of a circle.
    """
    n = points.shape[0]
    if n < 3:
        return None

    x, y = points[:, 0].astype(np.float64), points[:, 1].astype(np.float64)
    x2, y2 = x**2, y**2
    x2y2 = x2 + y2

    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_x2, sum_y2 = np.sum(x2), np.sum(y2)
    sum_xy = np.sum(x * y)

    A = np.array([[sum_x2, sum_xy, sum_x], [sum_xy, sum_y2, sum_y], [sum_x, sum_y, n]])

    B = np.array([[-np.sum(x * x2y2)], [-np.sum(y * x2y2)], [-np.sum(x2y2)]])

    try:
        sol = np.linalg.solve(A, B)
        a, b, c = sol.flatten()
        xc, yc = -a / 2, -b / 2
        R2 = xc**2 + yc**2 - c
        if R2 <= 0:
            return None
        R = np.sqrt(R2)
        dist_sq = (x - xc)**2 + (y - yc)**2
        std_dev = np.sqrt(np.mean((dist_sq - R**2)**2)) / (2 * R)
        return xc, yc, R, std_dev, n
    except np.linalg.LinAlgError:
        return None


def refine_circle(xc, yc, r, edges, angles, delta_r=12, angle_threshold=8):
    """
    Refine circle parameters with tight thresholds to lock onto a single edge.
    """
    h, w = edges.shape
    x_min, x_max = max(0, int(xc - r - delta_r)), min(w - 1, int(xc + r + delta_r))
    y_min, y_max = max(0, int(yc - r - delta_r)), min(h - 1, int(yc + r + delta_r))

    roi_edges = edges[y_min : y_max + 1, x_min : x_max + 1]
    edge_y, edge_x = np.where(roi_edges > 0)
    edge_y, edge_x = edge_y + y_min, edge_x + x_min

    dist = np.sqrt((edge_x - xc)**2 + (edge_y - yc)**2)
    in_annulus = (dist >= r - delta_r) & (dist <= r + delta_r)
    edge_x, edge_y = edge_x[in_annulus], edge_y[in_annulus]

    if len(edge_x) < 40:
        return None

    radial_angles = np.rad2deg(np.arctan2(edge_y - yc, edge_x - xc)) % 180
    edge_angles = angles[edge_y, edge_x]
    angle_diff = np.abs(radial_angles - edge_angles)
    angle_diff = np.minimum(angle_diff, 180 - angle_diff)

    matching_gradient = angle_diff <= angle_threshold
    final_points = np.column_stack((edge_x[matching_gradient], edge_y[matching_gradient]))

    if len(final_points) < 40:
        return None
    return kasa_fit(final_points)


def facile_detection(image, petri_only=True, min_points=None, max_error=None, suppression_dist_frac=0.45):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 110)

    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.rad2deg(np.arctan2(gy, gx)) % 180

    masks = []
    masks.append(((angles >= 0) & (angles <= 45)) | ((angles >= 135) & (angles <= 180)))
    masks.append((angles >= 0) & (angles <= 90))
    masks.append((angles >= 45) & (angles <= 135))
    masks.append((angles >= 90) & (angles <= 180))

    raw_arcs = []
    img_h, img_w = image.shape[:2]
    img_diagonal = np.sqrt(img_h**2 + img_w**2)

    for mask in masks:
        seg_edges = (mask & (edges > 0)).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            seg_edges, connectivity=8
        )

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 50:
                continue

            roi_labels = labels[y : y + h, x : x + w]
            ly, lx = np.where(roi_labels == i)
            points = np.column_stack((lx + x, ly + y))

            if w > h:
                a_idx, b_idx = np.argmin(points[:, 0]), np.argmax(points[:, 0])
            else:
                a_idx, b_idx = np.argmin(points[:, 1]), np.argmax(points[:, 1])

            pA, pB, pC = points[a_idx], points[b_idx], points[len(points) // 2]
            vAC, vBC = pA - pC, pB - pC
            dot = np.sum(vAC * vBC)
            magAC2, magBC2 = np.sum(vAC**2), np.sum(vBC**2)

            if magAC2 * magBC2 == 0:
                continue
            if (dot**2) / (magAC2 * magBC2) >= 0.90:
                continue

            fit = kasa_fit(points)
            if fit:
                xc, yc, r, std_dev, n = fit
                if 20 < r < img_diagonal * 0.75:
                    error_limit = max_error if max_error is not None else (1.2 if r > 400 else 1.0)
                    if std_dev < error_limit:
                        raw_arcs.append((xc, yc, r))

    refined_circles = []
    for xc, yc, r in raw_arcs:
        refined = refine_circle(xc, yc, r, edges, angles)
        if refined:
            refined_circles.append(refined)

    # DUPLICATE SUPPRESSION
    final_circles = []
    # Sort by radius DESCENDING - we want the OUTERmost rim first
    refined_circles.sort(key=lambda x: x[2], reverse=True)

    for xc, yc, r, std, n in refined_circles:
        is_duplicate = False
        for i, (fxc, fyc, fr, fstd, fn) in enumerate(final_circles):
            dist = np.sqrt((xc - fxc)**2 + (yc - fyc)**2)
            # If centers are within suppression_dist_frac of the radius, they are the same plate.
            if dist < fr * suppression_dist_frac:
                is_duplicate = True
                # Only replace the existing larger circle if this one is
                # significantly more "solid" (more rim points).
                if n > fn * 1.5:
                    final_circles[i] = (xc, yc, r, std, n)
                break

        if not is_duplicate:
            # Minimum points to be considered a real petri plate rim
            min_pts = min_points if min_points is not None else (200 if r < 200 else 800)
            if n > min_pts:
                final_circles.append((xc, yc, r, std, n))

    if petri_only:
        if not final_circles:
            return []
        max_r = max(c[2] for c in final_circles)
        # Return circles that are plate-sized
        return [c for c in final_circles if c[2] > 150 or c[2] >= max_r * 0.5]

    return final_circles


@register("detect_facile")
class FacileDetector(Preprocessor):
    class Params(BlennyParams):
        petri_only: bool = True
        """Only return circles that look like Petri dishes (size filter)."""

        radius_scale: float = 1.0
        """Scale factor for the detected plate radius."""

        crop: bool = False
        """If True, crop the image to the detected plate (single-plate mode only)."""

        mask_key: str = "plate"
        """Key under which the plate mask is stored in ``data.masks``."""

        multi_plate: bool | None = None
        """If True, detect multiple plates. If False, detect one.
        If None (default), automatically switch based on number found."""

        size_consistency_limit: float | None = 0.15
        """Max allowed deviation from the median radius (0.15 = 15%). 
        Only applied when multiple plates are found. Set to None to disable."""

        suppression_dist_frac: float = 0.45
        """Fraction of radius used as the threshold for center-to-center suppression."""

        min_points: int | None = None
        """Minimum number of edge points required to accept a circle. 
        If None, automatically scaled based on radius."""

        max_error: float | None = None
        """Maximum allowed standard deviation for circle fitting.
        If None, automatically scaled based on radius."""

        force_cy: int | None = None
        """Force the plate center Y coordinate (bypasses detection)."""

        force_cx: int | None = None
        """Force the plate center X coordinate (bypasses detection)."""

        force_r: int | None = None
        """Force the plate radius (bypasses detection)."""

        force_mask_path: str | None = None
        """Path to a binary mask file to use as the plate ROI (bypasses detection)."""

    def process(self, image: Any, data: ImageData) -> Any:
        h_orig, w_orig = image.shape[:2]

        # --- Manual override mode (forced circle / forced mask) -----------
        # The GUI's "Manual Circle" and "Manual Shape" modes inject these
        # params. Bypass detection and build a single ROI from the supplied
        # geometry so downstream sub_pipeline steps work unchanged.
        force_mask_path: str | None = self.params.force_mask_path
        force_cy: int | None = self.params.force_cy
        force_cx: int | None = self.params.force_cx
        force_r: int | None = self.params.force_r
        if force_mask_path is not None or (
            force_cy is not None and force_cx is not None and force_r is not None
        ):
            return self._apply_forced(
                image, data, h_orig, w_orig, force_mask_path, force_cy, force_cx, force_r
            )

        # facile works best on images around 1000-2000px.
        # If the image is very large, we downsample it for detection.
        det_dim = 1500
        if max(h_orig, w_orig) > det_dim:
            det_scale = det_dim / max(h_orig, w_orig)
            new_h, new_w = int(h_orig * det_scale), int(w_orig * det_scale)
            # Use a fast resize for detection
            det_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            det_img = image
            det_scale = 1.0

        circles = facile_detection(
            det_img, 
            petri_only=self.params.petri_only,
            min_points=self.params.min_points,
            max_error=self.params.max_error,
            suppression_dist_frac=self.params.suppression_dist_frac
        )

        if not circles:
            data.add_flag(
                "plate_not_found",
                "FacileDetector could not locate any circular plates.",
                severity="warning",
            )
            h, w = image.shape[:2]
            data.masks[self.params.mask_key] = np.ones((h, w), dtype=bool)
            return image

        # Scale circles back to original resolution
        rescaled_circles = []
        for xc, yc, r, std, n in circles:
            rescaled_circles.append((xc / det_scale, yc / det_scale, r / det_scale, std, n))
        circles = rescaled_circles

        # Determination of multi-plate mode.
        use_roi_mode = self.params.multi_plate is not False

        # Apply Size Consistency Filter if in multi-plate mode and >1 found
        if use_roi_mode and len(circles) > 1 and self.params.size_consistency_limit is not None:
            radii = np.array([c[2] for c in circles])
            median_r = np.median(radii)
            limit = self.params.size_consistency_limit
            
            filtered_circles = []
            for c in circles:
                deviation = abs(c[2] - median_r) / median_r
                if deviation <= limit:
                    filtered_circles.append(c)
                else:
                    data.add_flag(
                        "plate_size_outlier",
                        f"A circle with radius {c[2]:.1f} was rejected as an outlier (median={median_r:.1f}).",
                        severity="info"
                    )
            circles = filtered_circles

        if not circles: # Edge case: all outliers
             data.masks[self.params.mask_key] = np.ones((h_orig, w_orig), dtype=bool)
             return image

        if use_roi_mode and len(circles) > 0:
            rois = []
            # Sort circles by position (top-to-bottom, then left-to-right) for consistent naming
            circles.sort(key=lambda c: (c[1], c[0]))
            
            for i, (xc, yc, r_raw, std, n) in enumerate(circles):
                r_eff = max(1, round(r_raw * self.params.radius_scale))
                r = max(r_raw, r_eff)

                buffer = int(r * 0.05) + 20
                h_img, w_img = image.shape[:2]
                y0_crop, y1_crop = max(0, int(yc - r - buffer)), min(h_img, int(yc + r + buffer + 1))
                x0_crop, x1_crop = max(0, int(xc - r - buffer)), min(w_img, int(xc + r + buffer + 1))

                rois.append(
                    {
                        "label": str(i + 1),
                        "bbox": (y0_crop, x0_crop, y1_crop, x1_crop),
                        "center_local": (int(yc - y0_crop), int(xc - x0_crop)),
                        "radius": int(r),
                        "radius_eff": int(r_eff),
                    }
                )
            data.metadata["rois"] = rois
            data.metadata["multi_plate_mode"] = True
            return image
        else:
            # Single plate mode (multi_plate=False or only 1 found): pick the largest one
            circles.sort(key=lambda x: x[2], reverse=True)
            xc, yc, r_raw, std, n = circles[0]
            r_eff = max(1, round(r_raw * self.params.radius_scale))
            r = max(r_raw, r_eff)

            h_img, w_img = image.shape[:2]
            mask = np.zeros((h_img, w_img), dtype=bool)
            rr, cc = disk((int(yc), int(xc)), r_eff, shape=(h_img, w_img))
            mask[rr, cc] = True

            data.metadata["plate_center"] = (int(yc), int(xc))
            data.metadata["plate_radius"] = int(r)
            data.metadata["plate_radius_raw"] = int(r_raw)

            if self.params.crop:
                y0, y1 = max(0, int(yc - r)), min(h_img, int(yc + r + 1))
                x0, x1 = max(0, int(xc - r)), min(w_img, int(xc + r + 1))
                cropped = image[y0:y1, x0:x1]
                data.masks[self.params.mask_key] = mask[y0:y1, x0:x1]
                data.metadata["plate_bbox"] = (y0, x0, y1, x1)
                return cropped

            data.masks[self.params.mask_key] = mask
            return image

    def _apply_forced(
        self,
        image: Any,
        data: ImageData,
        h_orig: int,
        w_orig: int,
        mask_path: str | None,
        fcy: int | None,
        fcx: int | None,
        fr: int | None,
    ) -> Any:
        """Handle Manual Circle / Manual Shape overrides.

        Produces a single ROI (same shape as the normal ROI branch) plus a
        plate mask, so ``sub_pipeline`` and the GUI review flow work exactly
        as in auto-detection mode.
        """
        mask = None
        if mask_path is not None:
            from PIL import Image

            mask_im = Image.open(mask_path).convert("L")
            mask = np.asarray(mask_im) > 127
            if mask.shape != (h_orig, w_orig):
                from skimage.transform import resize

                mask = resize(mask, (h_orig, w_orig), order=0, anti_aliasing=False) > 0.5
            ys, xs = np.where(mask)
            if len(ys) == 0:
                data.add_flag(
                    "plate_not_found",
                    "FacileDetector: the forced plate mask is empty; "
                    "downstream steps will run on the full image.",
                    severity="warning",
                )
                data.masks[self.params.mask_key] = np.ones((h_orig, w_orig), dtype=bool)
                return image
            cy, cx = int(ys.mean()), int(xs.mean())
            r_raw = int(math.sqrt(mask.sum() / math.pi))
        else:
            # The GUI draws coordinates on the full-resolution image; scale
            # them if the working image was downscaled at load time.
            load_scale = data.metadata.get("resize_scale", 1.0)
            cy = round(fcy * load_scale)
            cx = round(fcx * load_scale)
            r_raw = round(fr * load_scale)

        r_eff = max(1, round(r_raw * self.params.radius_scale))
        r = max(r_raw, r_eff)

        buffer = int(r * 0.05) + 20
        y0, y1 = max(0, int(cy - r - buffer)), min(h_orig, int(cy + r + buffer + 1))
        x0, x1 = max(0, int(cx - r - buffer)), min(w_orig, int(cx + r + buffer + 1))
        data.metadata["rois"] = [
            {
                "label": "1",
                "bbox": (y0, x0, y1, x1),
                "center_local": (int(cy - y0), int(cx - x0)),
                "radius": int(r),
                "radius_eff": int(r_eff),
            }
        ]
        data.metadata["multi_plate_mode"] = True
        data.metadata["plate_center"] = (cy, cx)
        data.metadata["plate_radius"] = int(r)

        if mask is None:
            mask = np.zeros((h_orig, w_orig), dtype=bool)
            rr, cc = disk((cy, cx), r_eff, shape=(h_orig, w_orig))
            mask[rr, cc] = True
        data.masks[self.params.mask_key] = mask
        return image
