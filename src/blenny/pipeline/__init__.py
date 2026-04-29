"""Core pipeline abstractions for Blenny.

Public API:
    - :class:`ImageData`, :class:`QualityFlag`, :class:`ProvenanceRecord`
    - :class:`Module` and the six semantic subclasses (Loader, Preprocessor,
      Segmenter, FeatureExtractor, Classifier, Exporter)
    - :class:`BlennyParams` for declaring module parameter schemas
    - :data:`MODULES` (the global registry) and :func:`register`
    - :class:`Pipeline` (the runner)
"""

from blenny.pipeline.base import (
    BlennyParams,
    Classifier,
    Exporter,
    FeatureExtractor,
    Loader,
    Module,
    Preprocessor,
    Segmenter,
)
from blenny.pipeline.context import ImageData, ProvenanceRecord, QualityFlag
from blenny.pipeline.registry import MODULES, ModuleRegistry, register
from blenny.pipeline.runner import Pipeline

__all__ = [
    "MODULES",
    "BlennyParams",
    "Classifier",
    "Exporter",
    "FeatureExtractor",
    "ImageData",
    "Loader",
    "Module",
    "ModuleRegistry",
    "Pipeline",
    "Preprocessor",
    "ProvenanceRecord",
    "QualityFlag",
    "Segmenter",
    "register",
]
