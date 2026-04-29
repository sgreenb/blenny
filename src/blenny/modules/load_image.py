"""Read images from disk into an :class:`ImageData`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from blenny.pipeline import BlennyParams, Loader, register


@register("load_image")
class ImageFileLoader(Loader):
    """Load JPEG/PNG/TIFF images via Pillow.

    Honours EXIF orientation (important for phone photos) and stores
    pixel dimensions in ``data.metadata`` so downstream modules can
    convert pixel measurements to physical units.
    """

    class Params(BlennyParams):
        as_gray: bool = False
        """If True, convert to a single-channel grayscale image on load."""

    def load(self, source: str) -> Any:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("L") if self.params.as_gray else im.convert("RGB")  # type: ignore[attr-defined]
            arr = np.asarray(im).copy()
        return arr

    def run(self, data):  # type: ignore[override, no-untyped-def]
        data = super().run(data)
        data.metadata.setdefault("source_path", str(Path(data.source).resolve()))
        data.metadata["image_shape"] = tuple(data.image.shape)
        data.metadata["image_dtype"] = str(data.image.dtype)
        return data
