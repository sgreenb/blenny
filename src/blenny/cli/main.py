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
    help="Blenny: A toolkit for analyzing plates and microscopy images.\n\n"
    "Documentation: https://github.com/your-org/blenny\n\n"
    "CORE WORKFLOW:\n"
    "  1. Initialize a pipeline:  blenny init\n"
    "  2. Run the analysis:       blenny run pipeline.yaml --input plate.jpg --output results/\n"
    "  3. Inspect modules:        blenny modules\n"
    "  4. Launch the GUI:         blenny gui\n\n"
    "Single-image outputs: annotated.png, colonies.csv, log.txt\n"
    "Batch outputs add:    summary.csv, batch_log.txt\n"
    "Optional:             --provenance (provenance.json)  --debug-dir (step images)",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
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


@app.command(
    help="Run a YAML pipeline on one image or a batch.\n\n"
    "Outputs per image: annotated.png, colonies.csv, log.txt\n"
    "Batch outputs (auto or --summary): summary.csv, batch_log.txt\n"
    "Optional: --provenance for provenance.json, --debug-dir for step images"
)
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
    debug_dir: Annotated[
        Path | None,
        typer.Option(
            "--debug-dir",
            help="If set, write intermediate artifacts to per-image subdirs here.",
        ),
    ] = None,
    write_provenance: Annotated[
        bool,
        typer.Option(
            "--provenance/--no-provenance",
            help="Save per-image provenance.json for reproducibility.",
        ),
    ] = False,
    write_summary: Annotated[
        bool | None,
        typer.Option(
            "--summary/--no-summary",
            help="Save batch summary.csv and batch_log.txt. Defaults to True if multiple images are processed.",
        ),
    ] = None,
    multiplicity: Annotated[
        bool,
        typer.Option(
            "--multiplicity/--no-multiplicity",
            help="Enable or disable merged-colony multiplicity estimation (default: enabled).",
        ),
    ] = True,
    fail_fast: Annotated[
        bool,
        typer.Option(
            "--fail-fast/--keep-going",
            help="Stop on first error vs. log and continue (default: keep going).",
        ),
    ] = False,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the batch summary as JSON to stdout."),
    ] = False,
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "-v",
            help="Override module parameters (e.g., -v threshold_segment.min_area=20).",
        ),
    ] = None,
) -> None:
    inputs = sorted(_expand_input(input_pattern))
    if not inputs:
        typer.echo(f"No files matched {input_pattern!r}", err=True)
        raise typer.Exit(code=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_config = load_yaml(pipeline_path)
    raw_steps = extract_steps(raw_config)

    # Apply command-line overrides to the raw steps
    if not multiplicity:
        for step in raw_steps:
            if step["name"] == "estimate_multiplicity":
                step.setdefault("params", {})["enabled"] = False

    if override:
        for item in override:
            try:
                key, value = item.split("=", 1)
                mod_name, param_name = key.split(".", 1)

                # Try to parse value as JSON, float, int, or bool
                parsed_val: Any
                if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
                    try:
                        parsed_val = json.loads(value)
                    except json.JSONDecodeError:
                        parsed_val = value
                elif value.lower() in ("null", "none"):
                    parsed_val = None
                elif value.lower() == "true":
                    parsed_val = True
                elif value.lower() == "false":
                    parsed_val = False
                else:
                    try:
                        parsed_val = float(value) if "." in value else int(value)
                    except ValueError:
                        parsed_val = value

                # Find the module in the steps and update it
                found = False
                for step in raw_steps:
                    if step["name"] == mod_name:
                        step.setdefault("params", {})[param_name] = parsed_val
                        found = True
                if not found:
                    typer.echo(f"Warning: Module '{mod_name}' not found in pipeline.", err=True)
            except ValueError as e:
                typer.echo(f"Error: Invalid override format '{item}'. Use mod.param=val", err=True)
                raise typer.Exit(1) from e

    # Sanity-check that the pipeline makes a Pipeline at all (no inputs yet).
    # Substitution leaves placeholders untouched if no values are provided,
    # which would later fail to format. Catch unknown registry names early.
    _ = Pipeline.from_config(_resolve_for_validation(raw_steps))

    if not as_json:
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
            img_debug_dir = debug_dir / stem if debug_dir else None
            data = pipe.run(img, debug_dir=img_debug_dir)
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
        if write_provenance:
            _write_provenance(per_image_dir / "provenance.json", data, img)
        summary_rows.append(
            {
                "input": str(img),
                "stem": stem,
                "status": "ok",
                "colony_count": data.metadata.get("colony_count"),
                "n_quality_flags": len(data.quality_flags),
                "flag_codes": "|".join(f.code for f in data.quality_flags),
                "duration_s": round(elapsed, 3),
            }
        )
        if not as_json:
            count = data.metadata.get("colony_count", "?")
            typer.echo(f"  [OK]   {img.name}  colonies={count}  ({elapsed:.1f}s)")

    if first_resolved is not None:
        # For the reproducible config, we want a "template" version where
        # per-image placeholders like {stem} are preserved, but global
        # parameters (including CLI overrides) are baked in.
        reproducible_steps = substitute_paths(
            raw_steps,
            output_dir="{output_dir}",
            input_path="{input_path_placeholder}"
        )
        # Fix the dummy input path back to placeholders
        for step in reproducible_steps:
            if "params" in step:
                for k, v in step["params"].items():
                    if isinstance(v, str) and "{input_path_placeholder}" in v:
                        step["params"][k] = v.replace("{input_path_placeholder}.stem", "{stem}").replace("{input_path_placeholder}", "{input}")

        dump_resolved_config(
            raw_steps, # raw_steps already has CLI overrides but preserves placeholders
            output_dir / "reproducible_config.yaml",
            extra={
                "_blenny_version": __version__,
                "_pipeline_source": str(pipeline_path.resolve()),
                "_run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    # Only write batch summary files if requested OR if there are multiple images
    # and the user hasn't explicitly disabled them.
    if write_summary is True or (write_summary is None and len(inputs) > 1):
        _write_summary_csv(output_dir / "summary.csv", summary_rows)
        _write_batch_log_txt(output_dir / "batch_log.txt", summary_rows, time.perf_counter() - t_batch)

    total = time.perf_counter() - t_batch
    succ = len(inputs) - failures

    if as_json:
        # Emit a clean machine-readable summary for GUI/script consumption
        report = {
            "total_images": len(inputs),
            "succeeded": succ,
            "failed": failures,
            "duration_s": round(total, 3),
            "results": summary_rows
        }
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Done: {succ}/{len(inputs)} succeeded in {total:.1f}s")

    if failures:
        raise typer.Exit(code=1)


@app.command(help="Launch the Blenny GUI.")
def gui() -> None:
    """Launch the Streamlit GUI.

    This command finds the ``gui/app.py`` file relative to the package
    installation and executes it via ``streamlit run``.
    """
    import subprocess
    import sys

    # Find the app.py file relative to this script
    gui_dir = Path(__file__).parent.parent.parent.parent / "gui"
    app_path = gui_dir / "app.py"

    if not app_path.exists():
        # Try local dev path
        app_path = Path(__file__).parent.parent.parent.parent / "gui" / "app.py"

    if not app_path.exists():
        typer.echo(f"Error: Could not find GUI source at {app_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Launching Blenny GUI from {app_path}...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        typer.echo(f"Error launching GUI: {e}", err=True)
        raise typer.Exit(code=1) from e


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


@app.command(help="Write starter pipeline YAML configs (Classic and YOLO).")
def init(
    template: Annotated[
        str | None,
        typer.Argument(help="Template name (run `blenny init --list` to see options)."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Where to write the YAML.",
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

    # Default behavior: Write both core templates to their standard filenames
    if template is None and out is None:
        try:
            classic_text = templates.load_text("count-colonies")
            yolo_text = templates.load_text("count-colonies-yolo")

            Path("pipeline_classic.yaml").write_text(classic_text, encoding="utf-8")
            Path("pipeline_yolo.yaml").write_text(yolo_text, encoding="utf-8")

            typer.echo("Wrote Classic CV template to pipeline_classic.yaml")
            typer.echo("Wrote YOLO ML template to pipeline_yolo.yaml")
            return
        except KeyError as e:
            typer.echo(f"Error loading default templates: {e}", err=True)
            raise typer.Exit(code=1) from None

    # Specific template or output path requested
    template_name = template or "count-colonies"
    out_path = out or Path("pipeline.yaml")

    try:
        text = templates.load_text(template_name)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    typer.echo(f"Wrote {template_name} template to {out_path}")


# --- helpers -----------------------------------------------------------------


def _expand_input(pattern: str) -> list[Path]:
    """Resolve a single path, a directory, or a glob pattern to a list of images."""
    from blenny.modules.load_image import IMAGE_EXTENSIONS

    p = Path(pattern)
    if any(c in pattern for c in "*?["):
        return [
            Path(m) for m in glob_module.glob(pattern, recursive=True)
            if Path(m).is_file() and Path(m).suffix.lower() in IMAGE_EXTENSIONS
        ]
    if p.is_dir():
        return sorted([
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ])
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
        "measurements": _to_jsonable(data.measurements),
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


def _write_batch_log_txt(path: Path, rows: list[dict[str, Any]], duration: float) -> None:
    """Dump a human-readable batch_log.txt for the entire batch."""
    if not rows:
        path.write_text("No images processed.\n", encoding="utf-8")
        return

    lines = [
        "=== Blenny Batch Processing Log ===",
        f"Total images:    {len(rows)}",
        f"Total duration:  {duration:.1f}s",
        "",
        f"{'Image':<30} {'Status':<10} {'Count':<10} {'Flags':<10}",
        "-" * 65
    ]
    for r in rows:
        name = r.get("stem", "unknown")[:30]
        status = r.get("status", "failed")
        count = str(r.get("colony_count", "-"))
        flags = str(r.get("n_quality_flags", "-"))
        lines.append(f"{name:<30} {status:<10} {count:<10} {flags:<10}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
