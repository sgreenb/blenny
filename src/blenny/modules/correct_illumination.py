"""Flatten uneven lighting via morphological top-hat.

A white top-hat extracts bright features smaller than a structuring
element from a slowly-varying background — exactly the situation for
bright bacterial colonies on an unevenly-lit plate. This is the
classical-CV alternative to subtracting a fitted illumination model.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from skimage import color, morphology, util

from blenny.pipeline import BlennyParams, ImageData, Preprocessor, register


@register("correct_illumination")
class IlluminationCorrection(Preprocessor):
    class Params(BlennyParams):
        method: Literal["white_tophat", "black_tophat"] = "white_tophat"
        """``white_tophat`` for bright objects on dark background, and vice versa."""

        radius: int | None = None
        """Structuring-element radius in pixels. Should exceed the largest object.

        If ``None`` (the default), the radius is automatically set to
        ``plate_radius / 15`` if a plate was detected, falling back to 25.
        """

        to_gray: bool = True
        """If True, output a single-channel float image. Most segmenters want this."""

    def process(self, image: Any, data: ImageData) -> Any:
        if image.ndim == 3 and self.params.to_gray:  # type: ignore[attr-defined]
            work = color.rgb2gray(image)
        elif image.ndim == 3:
            work = util.img_as_float(image)
        else:
            work = util.img_as_float(image)

        # Determine radius: explicit param > scale-aware default > fallback.
        radius = self.params.radius  # type: ignore[attr-defined]
        if radius is None:
            plate_r = data.metadata.get("plate_radius")
            radius = max(5, round(float(plate_r) / 15.0)) if plate_r is not None else 25

        selem = morphology.disk(radius)

        if work.ndim == 2:
            corrected = self._tophat(work, selem)
        else:
            # Apply per-channel.
            corrected = np.stack(
                [self._tophat(work[..., c], selem) for c in range(work.shape[-1])],
                axis=-1,
            )

        # Stash the original (possibly cropped) image for the annotated exporter.
        data.artifacts.setdefault("pre_illumination", image)
        return corrected

    def _tophat(self, im: np.ndarray, selem: np.ndarray) -> np.ndarray:
        if self.params.method == "white_tophat":  # type: ignore[attr-defined]
            return morphology.white_tophat(im, selem)
        return morphology.black_tophat(im, selem)
