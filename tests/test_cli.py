"""CLI tests (run, modules, version)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from blenny.cli.main import app
from blenny.testing import make_synthetic_plate

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "blenny" in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # Typer's no_args_is_help returns exit code 2 (Click convention).
    assert result.exit_code == 2
    assert "Usage" in result.stdout or "Usage" in (result.output or "")


# --- modules -----------------------------------------------------------------


def test_modules_command_lists_built_ins() -> None:
    result = runner.invoke(app, ["modules"])
    assert result.exit_code == 0
    for name in ("load_image", "detect_plate", "threshold_segment", "measure_colonies"):
        assert name in result.stdout


def test_modules_command_json_output() -> None:
    result = runner.invoke(app, ["modules", "--json"])
    assert result.exit_code == 0
    info = json.loads(result.stdout)
    assert isinstance(info, list)
    names = {entry["name"] for entry in info}
    assert {"load_image", "detect_plate", "export_csv"} <= names


# --- run end-to-end ----------------------------------------------------------


_MINIMAL_PIPELINE = """
steps:
  - name: load_image
    params:
      max_dimension: null
  - name: detect_plate
    params:
      radius_scale: 1.0
      crop: false
  - name: threshold_segment
    params:
      method: otsu
      roi_mask_key: plate
  - name: measure_colonies
  - name: export_csv
    params:
      output_path: "{output_dir}/{stem}/colonies.csv"
  - name: export_summary
    params:
      output_path: "{output_dir}/{stem}/log.txt"
  - name: export_annotated
    params:
      output_path: "{output_dir}/{stem}/annotated.png"
"""


def _save_synthetic(tmp_path: Path, *, n: int = 25, seed: int = 0) -> Path:
    plate = make_synthetic_plate(n_colonies=n, image_size=(512, 512), seed=seed)
    p = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(p)
    return p


def test_run_single_image_with_template(tmp_path: Path) -> None:
    img = _save_synthetic(tmp_path)
    pipe_yaml = tmp_path / "pipe.yaml"
    pipe_yaml.write_text(_MINIMAL_PIPELINE)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(pipe_yaml), "--input", str(img), "--output", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout
    # Per-image outputs landed where expected.
    assert (out_dir / "plate" / "colonies.csv").exists()
    assert (out_dir / "plate" / "annotated.png").exists()
    assert (out_dir / "plate" / "log.txt").exists()
    # provenance.json is opt-in; should NOT be present by default.
    assert not (out_dir / "plate" / "provenance.json").exists()
    # Resolved config at the root; summary.csv only for batches.
    assert (out_dir / "reproducible_config.yaml").exists()
    assert not (out_dir / "summary.csv").exists()


def test_run_batch_with_glob(tmp_path: Path) -> None:
    plates_dir = tmp_path / "plates"
    plates_dir.mkdir()
    paths = []
    for i, n in enumerate([10, 20, 30]):
        plate = make_synthetic_plate(n_colonies=n, image_size=(384, 384), seed=i)
        p = plates_dir / f"plate_{i}.png"
        Image.fromarray(plate.image).save(p)
        paths.append(p)

    pipe_yaml = tmp_path / "pipe.yaml"
    pipe_yaml.write_text(_MINIMAL_PIPELINE)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            str(pipe_yaml),
            "--input",
            str(plates_dir / "*.png"),
            "--output",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    for p in paths:
        assert (out_dir / p.stem / "colonies.csv").exists()
        assert (out_dir / p.stem / "annotated.png").exists()
        assert (out_dir / p.stem / "log.txt").exists()
        # provenance.json is opt-in; should NOT be present by default.
        assert not (out_dir / p.stem / "provenance.json").exists()
    # summary.csv is auto-generated for batch runs.
    summary_csv = (out_dir / "summary.csv").read_text()
    assert summary_csv.count("\n") >= 4  # header + 3 data rows


def test_run_no_match_exits_with_error(tmp_path: Path) -> None:
    pipe_yaml = tmp_path / "pipe.yaml"
    pipe_yaml.write_text(_MINIMAL_PIPELINE)
    result = runner.invoke(
        app,
        [
            "run",
            str(pipe_yaml),
            "--input",
            str(tmp_path / "no-such-file.jpg"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0


def test_run_keeps_going_after_per_image_failure(tmp_path: Path) -> None:
    """Bad image (corrupt file) should be logged and skipped, good images succeed."""
    img_good = _save_synthetic(tmp_path, n=10, seed=0)
    img_bad = tmp_path / "broken.png"
    img_bad.write_bytes(b"not-a-png")

    pipe_yaml = tmp_path / "pipe.yaml"
    pipe_yaml.write_text(_MINIMAL_PIPELINE)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            str(pipe_yaml),
            "--input",
            str(tmp_path / "*.png"),
            "--output",
            str(out_dir),
        ],
    )
    # Returns nonzero because there was a failure, but the good image still ran.
    assert result.exit_code != 0
    assert (out_dir / img_good.stem / "colonies.csv").exists()
    summary_csv = (out_dir / "summary.csv").read_text()
    assert "failed" in summary_csv
