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

        radius: int = 25
        """Structuring-element radius in pixels. Should exceed the largest object."""

        to_gray: bool = True
        """If True, output a single-channel float image. Most segmenters want this."""

    def process(self, image: Any, data: ImageData) -> Any:
        if image.ndim == 3 and self.params.to_gray:  # type: ignore[attr-defined]
            work = color.rgb2gray(image)
        elif image.ndim == 3:
            work = util.img_as_float(image)
        else:
            work = util.img_as_float(image)

        selem = morphology.disk(self.params.radius)  # type: ignore[attr-defined]

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
