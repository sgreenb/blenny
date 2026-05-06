"""The `Pipeline` runner: executes an ordered list of modules over an `ImageData`."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from blenny.pipeline.base import Module
from blenny.pipeline.context import ImageData, ProvenanceRecord
from blenny.pipeline.registry import MODULES


class Pipeline:
    """An ordered sequence of :class:`Module` instances.

    The runner is deliberately thin: it iterates the steps, times each
    one, and appends a :class:`ProvenanceRecord` to the context after
    every step. All real work lives in modules.
    """

    def __init__(self, steps: Iterable[Module] | None = None) -> None:
        self.steps: list[Module] = list(steps) if steps is not None else []

    # --- Construction --------------------------------------------------------

    def add(self, step: Module) -> Pipeline:
        self.steps.append(step)
        return self

    @classmethod
    def from_config(cls, steps_config: list[dict[str, Any]]) -> Pipeline:
        """Build a pipeline from a list of ``{"name": ..., "params": {...}}`` dicts.

        This is the in-memory shape that YAML/JSON configs deserialize to.
        ``Pipeline.from_yaml`` is a thin wrapper that adds parsing + path
        placeholder substitution.
        """
        steps: list[Module] = []
        for entry in steps_config:
            if "name" not in entry:
                raise ValueError(f"Pipeline config entry is missing 'name': {entry!r}")
            params = dict(entry.get("params", {}))
            if "instance_name" in entry:
                params["name"] = entry["instance_name"]
            steps.append(MODULES.create(entry["name"], **params))
        return cls(steps)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        input_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Pipeline:
        """Build a pipeline from a YAML file, with optional path placeholder
        substitution (``{stem}``, ``{input}``, ``{output_dir}``, ``{name}``).

        The placeholders are filled in inside ``params`` strings only; the
        module ``name`` field is never rewritten.
        """
        # Local import to avoid a circular dependency at module load time.
        from blenny.config import extract_steps, load_yaml, substitute_paths

        config = load_yaml(path)
        steps = extract_steps(config)
        steps = substitute_paths(steps, input_path=input_path, output_dir=output_dir)
        return cls.from_config(steps)

    # --- Execution -----------------------------------------------------------

    def run(
        self,
        data: ImageData | str | Path | None = None,
        *,
        debug_dir: str | Path | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ImageData:
        """Run the pipeline. Accepts an `ImageData`, a source path/string, or nothing.

        If ``debug_dir`` is provided, intermediate images and masks are saved
        after each step into that directory.

        The ``progress_callback`` (if provided) is called after each step with
        ``(step_index, total_steps, step_name)``.
        """
        ctx = self._coerce_input(data)
        dwriter = None
        if debug_dir is not None:
            from blenny.pipeline.debug import DebugWriter

            dwriter = DebugWriter(Path(debug_dir))

        n_steps = len(self.steps)
        for i, step in enumerate(self.steps):
            if progress_callback:
                progress_callback(i + 1, n_steps, step.name)
            
            t0 = time.perf_counter()
            ctx = step.run(ctx)
            duration = time.perf_counter() - t0
            ctx.provenance.append(
                ProvenanceRecord(
                    step=step.name,
                    module_class=type(step).__name__,
                    params=step.params_dict(),
                    duration_s=duration,
                )
            )
            # Stamp the step name onto any flags the step forgot to attribute.
            for flag in ctx.quality_flags:
                if not flag.step:
                    flag.step = step.name

            if dwriter is not None:
                dwriter.write_step(step.name, ctx)

        if dwriter is not None:
            dwriter.write_summary(ctx)

        return ctx

    @staticmethod
    def _coerce_input(data: ImageData | str | Path | None) -> ImageData:
        if data is None:
            return ImageData()
        if isinstance(data, ImageData):
            return data
        if isinstance(data, (str, Path)):
            return ImageData(source=str(data))
        raise TypeError(
            f"Pipeline.run accepts ImageData | str | Path | None, got {type(data).__name__}"
        )

    # --- Dunder --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.steps)

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self.steps)
        return f"Pipeline([{names}])"
