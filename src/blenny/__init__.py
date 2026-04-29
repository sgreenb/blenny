"""Blenny: a toolkit for analyzing plates and microscopy images."""

# Importing built-in modules registers them with MODULES as a side effect.
from blenny import modules as _modules  # noqa: F401
from blenny.pipeline import (
    MODULES,
    BlennyParams,
    Classifier,
    Exporter,
    FeatureExtractor,
    ImageData,
    Loader,
    Module,
    Pipeline,
    Preprocessor,
    QualityFlag,
    Segmenter,
    register,
)

__version__ = "0.0.1"

__all__ = [
    "MODULES",
    "BlennyParams",
    "Classifier",
    "Exporter",
    "FeatureExtractor",
    "ImageData",
    "Loader",
    "Module",
    "Pipeline",
    "Preprocessor",
    "QualityFlag",
    "Segmenter",
    "__version__",
    "register",
]
