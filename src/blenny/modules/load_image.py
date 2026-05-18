"""Read images from disk into an :class:`ImageData`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from blenny.pipeline import BlennyParams, Loader, register

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@register("load_image")
class ImageFileLoader(Loader):
    """Load JPEG/PNG/TIFF images via Pillow.

    Honours EXIF orientation (important for phone photos), optionally
    downscales large images for tractable processing time, and stores
    pixel dimensions in ``data.metadata`` so downstream modules can
    convert pixel measurements to physical units.
    """

    class Params(BlennyParams):
        as_gray: bool = False
        """If True, convert to a single-channel grayscale image on load."""

        max_dimension: int | None = None
        """If set, downscale so the longest side equals this many pixels (with
        Lanczos antialiasing). ``None`` (the default) loads at native resolution.

        Set to a value such as 2000 to speed up processing of large phone photos.
        Morphological operations scale roughly with image area, so a 12 MP phone
        photo is ~6x slower than a 2 MP image. Only use this if processing time
        is a concern and you don't need precise pixel-level measurements.
        """

    def load(self, source: str) -> Any:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            # Stash pre-resize size so run() can record provenance.
            self._last_original_size: tuple[int, int] = im.size  # (W, H)
            md: int | None = self.params.max_dimension  # type: ignore[attr-defined]
            if md is not None and max(im.size) > md:
                im.thumbnail((md, md), Image.Resampling.LANCZOS)
            im = im.convert("L") if self.params.as_gray else im.convert("RGB")  # type: ignore[attr-defined]
            arr = np.asarray(im).copy()
        return arr

    def run(self, data, **kwargs: Any):  # type: ignore[override, no-untyped-def]
        data = super().run(data, **kwargs)
        data.metadata.setdefault("source_path", str(Path(data.source).resolve()))
        data.metadata["image_shape"] = tuple(data.image.shape)
        data.metadata["image_dtype"] = str(data.image.dtype)
        original_w, original_h = self._last_original_size
        new_h, new_w = data.image.shape[:2]
        data.metadata["original_size_wh"] = (original_w, original_h)
        if (new_w, new_h) != (original_w, original_h):
            scale = new_w / original_w
            data.metadata["resized"] = True
            data.metadata["resize_scale"] = scale
            data.add_flag(
                "image_resized",
                f"Image downscaled from {original_w}x{original_h} to "
                f"{new_w}x{new_h} (scale={scale:.2f}). Pass max_dimension=None "
                "to load at native resolution.",
                severity="info",
            )
        return data
