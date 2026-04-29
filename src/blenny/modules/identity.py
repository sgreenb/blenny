"""A no-op preprocessor — the 'hello world' of pipeline steps.

Useful as:
  - a template for new module authors,
  - a placeholder in tests and example pipelines,
  - a minimal demonstration of the registry + Params pattern.
"""

from __future__ import annotations

from typing import Any

from blenny.pipeline import ImageData, Preprocessor, register


@register("identity")
class IdentityPreprocessor(Preprocessor):
    """Returns the image unchanged."""

    def process(self, image: Any, data: ImageData) -> Any:
        return image
