"""Shipped pipeline templates.

Loaded via :mod:`importlib.resources` so they work the same whether Blenny
is installed via pip, run from a checkout, or zipped into a wheel.
"""

from importlib.resources import files

__all__ = ["available", "load_text"]


def available() -> list[str]:
    """Return the names of every shipped template (without the ``.yaml`` suffix)."""
    return sorted(
        p.name.removesuffix(".yaml") for p in files(__name__).iterdir() if p.name.endswith(".yaml")
    )


def load_text(name: str) -> str:
    """Return the YAML text of a shipped template by name."""
    candidates = [f"{name}.yaml", f"{name.replace('-', '_')}.yaml"]
    pkg = files(__name__)
    for candidate in candidates:
        path = pkg / candidate
        if path.is_file():
            return path.read_text(encoding="utf-8")
    raise KeyError(f"No template named {name!r}. Available: {available()}")
