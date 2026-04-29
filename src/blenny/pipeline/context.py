"""The `ImageData` context object that flows through a pipeline.

Every pipeline step receives an `ImageData` and returns an `ImageData`
(usually the same instance, mutated). It carries the current image,
named masks, per-object measurements, arbitrary metadata, intermediate
debug artifacts, a provenance log, and quality flags.

Design notes:
  - We use a plain dataclass rather than a Pydantic model because the
    payload contains numpy arrays of arbitrary shape and dtype, and
    Pydantic's validation overhead is unhelpful for hot paths.
  - Image arrays are typed `Any` to avoid leaking numpy generics through
    the public API. Conventions (HxW or HxWxC, dtype expectations) are
    documented per module.
  - `measurements` is a list of dicts, not a DataFrame, to keep pandas
    out of the hot path. Conversion to a DataFrame is a one-liner at the
    edges (e.g., in a CSV exporter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Severity = Literal["info", "warning", "error"]


@dataclass
class QualityFlag:
    """A structured warning attached to an analysis result.

    Surfacing uncertainty is principle 4.7 of the design doc: a wrong
    answer presented confidently is worse than no answer.
    """

    code: str
    """Short machine-readable identifier, e.g. ``"low_contrast"``."""

    message: str
    """Human-readable explanation suitable for surfacing in a report."""

    severity: Severity = "warning"
    step: str = ""
    """Name of the pipeline step that raised the flag (filled by the runner)."""


@dataclass
class ProvenanceRecord:
    """One entry in the pipeline's execution log.

    Step 4 surfaces these to users; Step 3 serializes them alongside
    results for lightweight reproducibility.
    """

    step: str
    module_class: str
    params: dict[str, Any]
    duration_s: float


@dataclass
class ImageData:
    """The context object passed between pipeline steps.

    Fields are filled in progressively: a fresh `ImageData` from a source
    path has only `source` set; a `Loader` populates `image` and
    `original_image`; a `Segmenter` adds entries to `masks`; a
    `FeatureExtractor` appends rows to `measurements`; and so on.
    """

    source: str | None = None
    """Identifier for the input (file path, URI, or synthetic name)."""

    image: Any = None
    """The current working image (may be modified by preprocessors)."""

    original_image: Any = None
    """The image as first loaded, preserved for overlays and debugging."""

    masks: dict[str, Any] = field(default_factory=dict)
    """Named binary or label masks, e.g. ``{"plate": ..., "colonies": ...}``."""

    measurements: list[dict[str, Any]] = field(default_factory=list)
    """Per-object rows produced by feature extractors."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form metadata (pixel size, channel info, capture device, ...)."""

    artifacts: dict[str, Any] = field(default_factory=dict)
    """Intermediate images keyed by step name; used by the debug exporter."""

    provenance: list[ProvenanceRecord] = field(default_factory=list)
    """Append-only log written by the pipeline runner."""

    quality_flags: list[QualityFlag] = field(default_factory=list)
    """Warnings raised by steps about data or result reliability."""

    def add_flag(
        self,
        code: str,
        message: str,
        severity: Severity = "warning",
        step: str = "",
    ) -> None:
        """Convenience for steps to attach a `QualityFlag`."""
        self.quality_flags.append(
            QualityFlag(code=code, message=message, severity=severity, step=step)
        )
