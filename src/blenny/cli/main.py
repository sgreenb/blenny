"""The ``blenny`` command-line interface.

Subcommands:
    blenny run PIPELINE.yaml --input INPUT --output OUTPUT_DIR
    blenny modules [--json]
    blenny init [TEMPLATE] [--out PATH]
    blenny --version

The ``run`` subcommand is the workhorse: it accepts a single file or a glob
pattern as ``--input``, runs the YAML-defined pipeline against each match,
and writes per-image outputs into subdirectories of ``--output``. It also
saves a resolved ``config.yaml`` (with all defaults filled in) and a
``provenance.json`` per image, satisfying design.md \u00a74.5.
"""

from __future__ import annotations

import dataclasses
import glob as glob_module
import json
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from blenny import __version__
from blenny.config import dump_resolved_config, extract_steps, load_yaml, substitute_paths
from blenny.pipeline import MODULES, Pipeline

app = typer.Typer(
    name="blenny",
    help="Blenny: a toolkit for analyzing plates and microscopy images.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"blenny {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """Top-level options. Subcommands do the real work."""


# --- run ---------------------------------------------------------------------


@app.command(help="Run a YAML pipeline on one image or a batch.")
def run(
    pipeline_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to the YAML pipeline config.",
        ),
    ],
    input_pattern: Annotated[
        str,
        typer.Option(
            "--input",
            "-i",
            help="Input image path, or glob pattern (quote it!) like 'plates/*.jpg'.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to write per-image outputs into. Created if missing.",
        ),
    ],
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--keep-going",
            help="Stop on first error vs. log and continue (default: keep going).",
        ),
    ] = False,
) -> None:
    inputs = sorted(_expand_input(input_pattern))
    if not inputs:
        typer.echo(f"No files matched {input_pattern!r}", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = load_yaml(pipeline_path)
    raw_steps = extract_steps(raw_config)

    # Sanity-check that the pipeline makes a Pipeline at all (no inputs yet).
    # Substitution leaves placeholders untouched if no values are provided,
    # which would later fail to format. Catch unknown registry names early.
    _ = Pipeline.from_config(_resolve_for_validation(raw_steps))

    typer.echo(f"Running pipeline {pipeline_path.name} on {len(inputs)} image(s) -> {output_dir}/")

    summary_rows: list[dict[str, Any]] = []
    failures = 0
    t_batch = time.perf_counter()
    first_resolved: list[dict[str, Any]] | None = None

    for img in inputs:
        stem = img.stem
        per_image_dir = output_dir / stem
        per_image_dir.mkdir(parents=True, exist_ok=True)
        resolved = substitute_paths(raw_steps, input_path=img, output_dir=output_dir)
        if first_resolved is None:
            first_resolved = resolved

        t0 = time.perf_counter()
        try:
            pipe = Pipeline.from_config(resolved)
            data = pipe.run(img)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            typer.echo(f"  [FAIL] {img.name}: {type(e).__name__}: {e}", err=True)
            failures += 1
            summary_rows.append(
                {
                    "input": str(img),
                    "stem": stem,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}",
                    "duration_s": round(elapsed, 3),
                }
            )
            if fail_fast:
                raise typer.Exit(code=1) from e
            continue

        elapsed = time.perf_counter() - t0
        _write_provenance(per_image_dir / "provenance.json", data, img)
        summary_rows.append(
            {
                "input": str(img),
                "stem": stem,
                "status": "ok",
                "colony_count": data.metadata.get("colony_count"),
                "n_quality_flags": len(data.quality_flags),
                "duration_s": round(elapsed, 3),
            }
        )
        count = data.metadata.get("colony_count", "?")
        typer.echo(f"  [OK]   {img.name}  colonies={count}  ({elapsed:.1f}s)")

    if first_resolved is not None:
        dump_resolved_config(
            first_resolved,
            output_dir / "config.yaml",
            extra={
                "_blenny_version": __version__,
                "_pipeline_source": str(pipeline_path.resolve()),
                "_run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
    _write_summary_csv(output_dir / "summary.csv", summary_rows)

    total = time.perf_counter() - t_batch
    succ = len(inputs) - failures
    typer.echo(f"Done: {succ}/{len(inputs)} succeeded in {total:.1f}s")
    if failures:
        raise typer.Exit(code=1)


# --- modules -----------------------------------------------------------------


@app.command(help="List every registered module and the parameters it accepts.")
def modules(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit JSON instead of human-readable text."),
    ] = False,
) -> None:
    import sys as _sys

    info = []
    for name in MODULES.names():
        cls = MODULES.get(name)
        params_schema = cls.Params.model_json_schema()
        # Prefer the class docstring; fall back to the module's first line.
        doc = cls.__doc__
        if not doc:
            mod = _sys.modules.get(cls.__module__)
            doc = getattr(mod, "__doc__", None)
        first_line = doc.strip().splitlines()[0] if doc else ""
        info.append(
            {
                "name": name,
                "class": cls.__name__,
                "doc": first_line,
                "params": _summarise_params(params_schema),
            }
        )
    if as_json:
        typer.echo(json.dumps(info, indent=2, default=str))
        return
    for entry in info:
        typer.echo(f"{entry['name']}  ({entry['class']})")
        if entry["doc"]:
            typer.echo(f"    {entry['doc']}")
        if entry["params"]:
            for pname, pinfo in entry["params"].items():
                default = pinfo.get("default", "<required>")
                typer.echo(f"    - {pname}: {pinfo.get('type', '?')} = {default!r}")
        typer.echo("")


# --- init --------------------------------------------------------------------


@app.command(help="Write a starter pipeline YAML.")
def init(
    template: Annotated[
        str,
        typer.Argument(help="Template name (run `blenny init --list` to see options)."),
    ] = "count-colonies",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Where to write the YAML. Defaults to stdout.",
        ),
    ] = None,
    list_templates: Annotated[
        bool,
        typer.Option("--list", help="List the available templates and exit."),
    ] = False,
) -> None:
    from blenny import templates

    if list_templates:
        for name in templates.available():
            typer.echo(name)
        return
    try:
        text = templates.load_text(template)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None
    if out is None:
        typer.echo(text, nl=False)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(f"Wrote {template} template to {out}")


# --- helpers -----------------------------------------------------------------


def _expand_input(pattern: str) -> list[Path]:
    """Resolve a single path or a glob pattern to a list of existing files."""
    p = Path(pattern)
    if any(c in pattern for c in "*?["):
        return [Path(m) for m in glob_module.glob(pattern, recursive=True) if Path(m).is_file()]
    if p.is_file():
        return [p]
    return []


def _resolve_for_validation(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Substitute placeholders with harmless dummies for a one-off validation pass.

    We can't validate a config that uses ``{stem}`` etc. without actual values,
    so we plug in placeholders here just to confirm every module name is real
    and required params are present.
    """
    return substitute_paths(
        steps,
        input_path=Path("/_validate/_dummy.png"),
        output_dir=Path("/_validate"),
    )


def _summarise_params(schema: dict[str, Any]) -> dict[str, Any]:
    """Pull a flat name->{type,default} mapping out of a Pydantic JSON schema."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    out: dict[str, Any] = {}
    for pname, pinfo in props.items():
        entry: dict[str, Any] = {}
        if "type" in pinfo:
            entry["type"] = pinfo["type"]
        elif "anyOf" in pinfo:
            entry["type"] = " | ".join(t.get("type", "?") for t in pinfo["anyOf"])
        else:
            entry["type"] = pinfo.get("title", "?")
        if pname not in required:
            entry["default"] = pinfo.get("default")
        out[pname] = entry
    return out


def _write_provenance(path: Path, data: Any, source: Path) -> None:
    """Dump a per-image provenance.json file."""
    payload = {
        "source": str(source),
        "metadata": _to_jsonable(dict(data.metadata)),
        "colony_count": data.metadata.get("colony_count"),
        "quality_flags": [dataclasses.asdict(f) for f in data.quality_flags],
        "provenance": [dataclasses.asdict(p) for p in data.provenance],
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of common scientific-Python types to JSON-friendly values."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):  # numpy arrays / scalars
        return obj.tolist()
    return obj


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("# no images processed\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- entry point -------------------------------------------------------------


def main() -> None:
    """Console-script entry point. ``pyproject.toml`` points ``blenny`` here."""
    app()


if __name__ == "__main__":
    main()
