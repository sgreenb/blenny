"""Core pipeline abstractions.

This module is intentionally minimal in Step 0; the real interfaces
(Loader, Preprocessor, Segmenter, FeatureExtractor, Classifier, Exporter)
will be designed in Step 1.

The shapes here exist only so the package is importable, has something
to test, and gives downstream code a stable import path.
"""

from blenny.pipeline.core import Pipeline, PipelineStep

__all__ = ["Pipeline", "PipelineStep"]
