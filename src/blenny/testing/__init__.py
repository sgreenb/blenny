"""Helpers for testing and demos.

Anything in this subpackage is fair game for users to use in notebooks
and tutorials. The synthetic plate generator in particular doubles as
"example data" so we don't need to ship binary fixtures in git.
"""

from blenny.testing.synthetic import make_synthetic_plate

__all__ = ["make_synthetic_plate"]
