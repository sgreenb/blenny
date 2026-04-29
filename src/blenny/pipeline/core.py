"""Minimal pipeline scaffolding.

This is a placeholder implementation for Step 0. It establishes:
  - a uniform `dict`-like context that flows between steps,
  - an abstract `PipelineStep` base class,
  - a `Pipeline` runner that records each step's name in the context's
    provenance log.

Step 1 will replace this with proper typed interfaces (Loader, Preprocessor,
Segmenter, FeatureExtractor, Classifier, Exporter), a richer `ImageData`
context object, parameter schemas via Pydantic, and a module registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

# The context is a free-form dict for now. In Step 1 it becomes a typed
# object carrying image arrays, masks, measurements, and provenance.
Context = dict[str, Any]


class PipelineStep(ABC):
    """Abstract base for any step that transforms a `Context`.

    Subclasses implement `run(context) -> context`. They should be pure
    with respect to their declared inputs/outputs where possible; any
    side effects (file writes, model downloads) should be obvious from
    the step's name.
    """

    #: Human-readable step name used in logs and provenance.
    name: str = ""

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        if not self.name:
            self.name = self.__class__.__name__

    @abstractmethod
    def run(self, context: Context) -> Context:
        """Transform the context and return it (may mutate in place)."""


class Pipeline:
    """Runs an ordered sequence of `PipelineStep`s over a context.

    Records the name of each executed step in `context["_provenance"]`
    so that downstream code (and users) can see exactly what ran.
    """

    def __init__(self, steps: Iterable[PipelineStep] | None = None) -> None:
        self.steps: list[PipelineStep] = list(steps) if steps is not None else []

    def add(self, step: PipelineStep) -> Pipeline:
        self.steps.append(step)
        return self

    def run(self, context: Context | None = None) -> Context:
        ctx: Context = dict(context) if context is not None else {}
        provenance: list[str] = list(ctx.get("_provenance", []))
        for step in self.steps:
            ctx = step.run(ctx)
            provenance.append(step.name)
        ctx["_provenance"] = provenance
        return ctx

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self.steps)
        return f"Pipeline([{names}])"
