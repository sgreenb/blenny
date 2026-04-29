"""Built-in pipeline modules.

Real modules (loaders, preprocessors, segmenters, ...) arrive in Step 2.
For Step 0 we ship only a trivial `IdentityPreprocessor` to exercise the
pipeline plumbing in tests and examples.
"""

from blenny.modules.identity import IdentityPreprocessor

__all__ = ["IdentityPreprocessor"]
