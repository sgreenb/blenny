"""Abstract base classes for pipeline modules.

Every pipeline step is a `Module`: an object with a typed `Params` schema
and a `run(ImageData) -> ImageData` method. The six semantic subclasses
below give beginner module authors a guiding shape for the most common
roles, without preventing power users from subclassing `Module` directly
for unusual cases.

The semantic subclasses do not enforce hard contracts beyond what the
`run` method does — they exist to:
  1. Make a pipeline's *intent* readable (you can see at a glance which
     step segments and which classifies).
  2. Provide narrower abstract methods (`segment`, `extract`, `export`)
     so authors only override what's relevant to their role.
  3. Let future tooling (the GUI in Phase 2, validators, docs generators)
     reason about pipeline shape — e.g., "this pipeline has no Loader,
     did you mean to start from an image?".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from blenny.pipeline.context import ImageData


class BlennyParams(BaseModel):
    """Base class for all module parameter schemas.

    Subclass this in each module's nested ``Params`` class. Unknown fields
    are rejected so typos in YAML configs (Step 3) fail loudly instead of
    silently using defaults.
    """

    model_config = ConfigDict(extra="forbid", frozen=False)


class Module(ABC):
    """Abstract base for every pipeline step.

    Subclasses typically declare a nested ``Params`` class extending
    :class:`BlennyParams` and override :meth:`run`. The default
    constructor accepts keyword arguments matching the ``Params`` schema;
    pass ``name=...`` to override the instance name used in provenance
    logs (useful when the same module appears twice in a pipeline).
    """

    class Params(BlennyParams):
        """Default empty params. Subclasses override with real fields."""

    #: Set by the registry when a module is registered with @register("name").
    registry_name: str | None = None

    def __init__(self, name: str | None = None, **kwargs: Any) -> None:
        self.params: BlennyParams = type(self).Params(**kwargs)
        self.name: str = name or self.registry_name or type(self).__name__

    def params_dict(self) -> dict[str, Any]:
        """Serializable dict of this module's parameters (for provenance)."""
        return self.params.model_dump()

    @abstractmethod
    def run(self, data: ImageData) -> ImageData:
        """Transform `data` and return it. May mutate in place."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# --- Semantic subclasses -----------------------------------------------------
#
# Each subclass narrows `run` to a more specific abstract method, but
# still satisfies the `Module.run` contract. Authors override the narrow
# method; the base `run` here handles the bookkeeping.


class Loader(Module):
    """Reads an input source into an :class:`ImageData`.

    Loaders are unusual: they don't transform an existing image, they
    create one. They expect ``data.source`` to be set (or accept it via
    params) and populate ``data.image`` and ``data.original_image``.
    """

    @abstractmethod
    def load(self, source: str) -> Any:
        """Return an image array for the given source identifier."""

    def run(self, data: ImageData) -> ImageData:
        if data.source is None:
            raise ValueError(f"{self.name}: ImageData.source is not set; a Loader needs a source.")
        image = self.load(data.source)
        data.image = image
        if data.original_image is None:
            data.original_image = image
        return data


class Preprocessor(Module):
    """Transforms ``data.image`` in place (denoise, illumination, crop, ...)."""

    @abstractmethod
    def process(self, image: Any, data: ImageData) -> Any:
        """Return the new image. ``data`` is provided for context (masks, metadata)."""

    def run(self, data: ImageData) -> ImageData:
        if data.image is None:
            raise ValueError(f"{self.name}: no image to preprocess; did a Loader run?")
        data.image = self.process(data.image, data)
        return data


class Segmenter(Module):
    """Produces a mask from ``data.image`` and stores it in ``data.masks``."""

    class Params(BlennyParams):
        output_key: str = "default"
        """Key under which the produced mask is stored in ``data.masks``."""

    @abstractmethod
    def segment(self, image: Any, data: ImageData) -> Any:
        """Return a mask array (binary or label) for ``image``."""

    def run(self, data: ImageData) -> ImageData:
        if data.image is None:
            raise ValueError(f"{self.name}: no image to segment; did a Loader run?")
        mask = self.segment(data.image, data)
        key = self.params.output_key  # type: ignore[attr-defined]
        data.masks[key] = mask
        return data


class FeatureExtractor(Module):
    """Computes per-object measurements from an image + mask."""

    class Params(BlennyParams):
        mask_key: str = "default"
        """Key in ``data.masks`` to read the mask from."""

    @abstractmethod
    def extract(self, image: Any, mask: Any, data: ImageData) -> list[dict[str, Any]]:
        """Return a list of per-object measurement rows."""

    def run(self, data: ImageData) -> ImageData:
        mask_key: str = self.params.mask_key  # type: ignore[attr-defined]
        if mask_key not in data.masks:
            raise ValueError(
                f"{self.name}: no mask {mask_key!r} in ImageData; available: {sorted(data.masks)}"
            )
        rows = self.extract(data.image, data.masks[mask_key], data)
        # Stamp each row with the source so batch CSVs are joinable.
        for row in rows:
            row.setdefault("source", data.source)
        data.measurements.extend(rows)
        return data


class Classifier(Module):
    """Annotates existing measurement rows with class labels or scores."""

    @abstractmethod
    def classify(self, rows: list[dict[str, Any]], data: ImageData) -> list[dict[str, Any]]:
        """Return updated rows (typically the same list, with new keys added)."""

    def run(self, data: ImageData) -> ImageData:
        data.measurements = self.classify(data.measurements, data)
        return data


class Exporter(Module):
    """Writes results to disk. Should not modify ``data`` in surprising ways."""

    @abstractmethod
    def export(self, data: ImageData) -> None:
        """Write outputs (CSVs, annotated images, ...) for ``data``."""

    def run(self, data: ImageData) -> ImageData:
        self.export(data)
        return data
