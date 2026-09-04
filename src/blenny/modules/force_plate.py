"""Use a manually-specified plate (GUI circle or drawn polygon) as ground truth.

The GUI's Manual Circle / Manual Polygon modes don't detect anything: the user
supplies the plate geometry directly, so a circle finder is never needed. This
module replaces the detector step in those modes -- it builds the plate mask
and the ROI metadata that :class:`~blenny.modules.sub_pipeline.SubPipeline`
consumes straight from the forced geometry, with no circle fitting.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from skimage.draw import disk

from blenny.pipeline import BlennyParams, ImageData, Module, register


def build_forced_plate(
    data: ImageData,
    *,
    mask_path: str | None,
    force_cy: int | None,
    force_cx: int | None,
    force_r: int | None,
    radius_scale: float,
    mask_key: str,
) -> bool:
    """Build the plate mask + ROI metadata from manual geometry.

    The supplied polygon mask (``mask_path``) or circle (``force_cy/cx/r``)
    IS the plate area, at its exact size and position -- no detection, no
    circle fitting, and the ROI spans the full image so nothing is cropped.
    Returns ``False`` when a forced mask is empty (the caller should flag it
    and fall back to the full image).
    """
    h_orig, w_orig = data.image.shape[:2]
    mask: np.ndarray | None = None
    if mask_path is not None:
        from PIL import Image

        mask_im = Image.open(mask_path).convert("L")
        mask = np.asarray(mask_im) > 127
        if mask.shape != (h_orig, w_orig):
            from skimage.transform import resize

            mask = resize(mask, (h_orig, w_orig), order=0, anti_aliasing=False) > 0.5
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return False
        cy, cx = int(ys.mean()), int(xs.mean())
        r_raw = int(math.sqrt(mask.sum() / math.pi))
    else:
        # The GUI draws coordinates on the full-resolution image; scale them
        # if the working image was downscaled at load time.
        load_scale = data.metadata.get("resize_scale", 1.0)
        cy = round(force_cy * load_scale)  # type: ignore[operator]
        cx = round(force_cx * load_scale)  # type: ignore[operator]
        r_raw = round(force_r * load_scale)  # type: ignore[operator]

    r_eff = max(1, round(r_raw * radius_scale))
    r = max(r_raw, r_eff)

    # Manual modes analyse the FULL image: the forced polygon/circle is the
    # exact analysis area, so the ROI spans the whole frame and sub_pipeline
    # passes the image through uncropped. The centre/radius are kept as
    # metadata only (e.g. for the interior-artifact reference).
    #
    # Record how the plate geometry was supplied so downstream consumers can
    # tell a true circle (detected or user-drawn) from a polygon: for a
    # polygon, the centroid + equivalent radius above are metadata only and
    # must NOT be visualised as a circle on the annotated output.
    data.metadata["plate_shape"] = "manual_polygon" if mask_path is not None else "manual_circle"
    y0, y1, x0, x1 = 0, h_orig, 0, w_orig

    # Label the single manual plate by its source stem (like auto single-plate
    # mode) so a batch of manual plates is distinguishable, rather than "1".
    label = Path(data.source).stem if data.source else "1"

    data.metadata["rois"] = [
        {
            "label": label,
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
    data.masks[mask_key] = mask
    return True


@register("force_plate")
class ForcePlate(Module):
    """Apply the user's manual circle or polygon as the exact plate area.

    Used by the GUI in Manual Circle / Manual Polygon modes in place of a
    detector step: no circle finding is performed. Params mirror the forced
    overrides of :class:`~blenny.modules.detect_facile.FacileDetector` so a
    config can switch between them without changing the rest of the pipeline.
    """

    class Params(BlennyParams):
        mask_key: str = "plate"
        """Key under which the plate mask is stored in ``data.masks``."""

        radius_scale: float = 1.0
        """Scale factor applied to the equivalent plate radius (metadata only;
        used for the interior-artifact reference in classify_by_interior)."""

        force_cy: int | None = None
        """Forced plate centre Y (Manual Circle)."""

        force_cx: int | None = None
        """Forced plate centre X (Manual Circle)."""

        force_r: int | None = None
        """Forced plate radius (Manual Circle)."""

        force_mask_path: str | None = None
        """Path to a binary mask file to use as the exact plate area
        (Manual Polygon). Bypasses all detection."""

    def run(self, data: ImageData, **kwargs: Any) -> ImageData:
        ok = build_forced_plate(
            data,
            mask_path=self.params.force_mask_path,  # type: ignore[attr-defined]
            force_cy=self.params.force_cy,  # type: ignore[attr-defined]
            force_cx=self.params.force_cx,  # type: ignore[attr-defined]
            force_r=self.params.force_r,  # type: ignore[attr-defined]
            radius_scale=self.params.radius_scale,  # type: ignore[attr-defined]
            mask_key=self.params.mask_key,  # type: ignore[attr-defined]
        )
        if not ok:
            data.add_flag(
                "plate_not_found",
                "ForcePlate: the forced plate mask is empty; "
                "downstream steps will run on the full image.",
                severity="warning",
            )
            h, w = data.image.shape[:2]
            data.masks[self.params.mask_key] = np.ones((h, w), dtype=bool)  # type: ignore[attr-defined]
        return data
