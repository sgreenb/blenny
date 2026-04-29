"""Module registry — maps short string names to :class:`Module` classes.

Used by:
  - YAML/JSON pipeline configs (Step 3): ``{"name": "threshold", "params": ...}``
  - Future plugin discovery: third-party packages can register modules
    via Python entry points and have them appear here.

The registry is intentionally a plain dict + decorator. We avoid metaclass
magic so that authors can grep for ``@register(`` and find every module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from blenny.pipeline.base import Module

ModuleT = TypeVar("ModuleT", bound=Module)


class ModuleRegistry:
    """A name → module-class registry."""

    def __init__(self) -> None:
        self._modules: dict[str, type[Module]] = {}

    def register(self, name: str) -> Callable[[type[ModuleT]], type[ModuleT]]:
        """Decorator: ``@MODULES.register("threshold")``."""

        def decorator(cls: type[ModuleT]) -> type[ModuleT]:
            if name in self._modules and self._modules[name] is not cls:
                raise ValueError(
                    f"Module name {name!r} already registered to "
                    f"{self._modules[name].__name__}; cannot re-register {cls.__name__}."
                )
            cls.registry_name = name
            self._modules[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type[Module]:
        try:
            return self._modules[name]
        except KeyError:
            raise KeyError(
                f"No module registered as {name!r}. Available: {sorted(self._modules)}"
            ) from None

    def create(self, registry_name: str, /, **params: object) -> Module:
        """Instantiate a registered module by name with the given params.

        The first argument is positional-only so that callers can still
        pass ``name=...`` as a module-level instance name without
        colliding with the registry-lookup name.
        """
        return self.get(registry_name)(**params)

    def names(self) -> list[str]:
        return sorted(self._modules)

    def __contains__(self, name: object) -> bool:
        return name in self._modules


#: The global registry. Most code should use this directly.
MODULES = ModuleRegistry()

#: Convenience alias so module files can ``from blenny.pipeline import register``.
register = MODULES.register
