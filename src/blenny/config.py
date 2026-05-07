"""YAML pipeline configs and runtime path substitution.

A pipeline YAML looks like:

    steps:
      - name: load_image
      - name: detect_plate
        params:
          margin_frac: 0.05
      - name: export_csv
        params:
          output_path: "{output_dir}/{stem}/colonies.csv"

The ``{stem}``, ``{input}``, ``{output_dir}``, and ``{name}`` placeholders are
substituted per input image at run time. This is what lets one config drive a
batch job: each input gets its own subdirectory of outputs, derived from the
input's filename, without the user having to write a config per image.

Substitution is restricted to string values inside ``params`` (we don't want
to rewrite the registered module names accidentally), and uses Python's
``str.format_map`` so unknown placeholders raise a clear KeyError instead of
silently leaving the string unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and return the top-level mapping.

    Raises ``ValueError`` if the file is not a mapping (e.g. a bare list).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, Mapping):
        raise ValueError(f"{p}: expected a YAML mapping at top level, got {type(data).__name__}")
    return dict(data)


def extract_steps(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pull the ``steps:`` list out of a parsed config and validate its shape."""
    if "steps" not in config:
        raise ValueError("Pipeline config is missing required 'steps' key")
    steps = config["steps"]
    if not isinstance(steps, list):
        raise ValueError(
            f"'steps' must be a list of {{name, params}} mappings; got {type(steps).__name__}"
        )
    out: list[dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"steps[{i}] is not a mapping: {step!r}")
        if "name" not in step:
            raise ValueError(f"steps[{i}] is missing 'name': {step!r}")
        out.append(dict(step))
    return out


def substitute_paths(
    steps: list[dict[str, Any]],
    *,
    input_path: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return a deep-copied steps list with ``{stem}``, ``{input}``,
    ``{output_dir}``, and ``{name}`` substituted in every params string value.

    Substitution is intentionally only applied inside ``params``: the step
    ``name`` field selects which module to instantiate and is never rewritten.
    """
    variables: dict[str, str] = {}
    if input_path is not None:
        ip = Path(input_path)
        variables["input"] = str(ip)
        variables["stem"] = ip.stem
        variables["name"] = ip.stem
    if output_dir is not None:
        variables["output_dir"] = str(Path(output_dir))

    def _sub(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return value.format_map(variables)
            except KeyError as e:
                key = e.args[0]
                raise KeyError(
                    f"Unknown placeholder {{{key}}} in config string {value!r}. "
                    f"Available: {sorted(variables)}"
                ) from None
        if isinstance(value, Mapping):
            return {k: _sub(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sub(v) for v in value]
        return value

    out: list[dict[str, Any]] = []
    for step in steps:
        new_step = dict(step)
        if "params" in new_step:
            new_step["params"] = _sub(new_step["params"])
        out.append(new_step)
    return out


def dump_resolved_config(
    steps: list[dict[str, Any]],
    path: Path | str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write a YAML file capturing the exact steps used.

    Lightweight reproducibility: the resolved config sits
    next to the results so any analysis can be re-run with one command.
    """
    payload: dict[str, Any] = {"steps": steps}
    if extra:
        payload.update(dict(extra))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
