"""CLI tests (run, modules, init, version)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from typer.testing import CliRunner

from blenny.cli.main import app
from blenny.config import load_yaml
from blenny.pipeline import ImageData
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


# --- init --------------------------------------------------------------------


def test_init_writes_to_default_file(tmp_path: Path) -> None:
    import os

    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Wrote Classic CV template to pipeline_classic.yaml" in result.stdout
        assert "Wrote YOLO ML (Auto) template to pipeline_yolo.yaml" in result.stdout
        assert Path("pipeline_classic.yaml").exists()
        assert Path("pipeline_yolo.yaml").exists()
        assert "steps:" in Path("pipeline_classic.yaml").read_text()
    finally:
        os.chdir(orig_cwd)


def test_init_writes_to_file(tmp_path: Path) -> None:
    out = tmp_path / "pipe.yaml"
    result = runner.invoke(app, ["init", "count-colonies", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    text = out.read_text()
    assert "load_image" in text


def test_init_unknown_template_errors() -> None:
    result = runner.invoke(app, ["init", "nope-not-a-template"])
    assert result.exit_code != 0


def test_init_list_templates() -> None:
    result = runner.invoke(app, ["init", "--list"])
    assert result.exit_code == 0
    assert "count_colonies" in result.stdout or "count-colonies" in result.stdout


# --- run end-to-end ----------------------------------------------------------


def _save_synthetic(tmp_path: Path, *, n: int = 25, seed: int = 0) -> Path:
    plate = make_synthetic_plate(n_colonies=n, image_size=(512, 512), seed=seed)
    p = tmp_path / "plate.png"
    Image.fromarray(plate.image).save(p)
    return p


def test_run_single_image_with_template(tmp_path: Path) -> None:
    img = _save_synthetic(tmp_path)
    pipe_yaml = tmp_path / "pipe.yaml"
    runner.invoke(app, ["init", "count-colonies", "--out", str(pipe_yaml)])

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        ["run", str(pipe_yaml), "--input", str(img), "--output", str(out_dir)],
    )
    assert result.exit_code == 0, result.stdout
    # Per-image outputs landed where expected.
    assert (out_dir / "plate" / "colonies.csv").exists()
    assert (out_dir / "plate" / "annotated.png").exists()
    assert (out_dir / "plate" / "plate_run_summary.txt").exists()
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
    runner.invoke(app, ["init", "--out", str(pipe_yaml)])

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
        assert (out_dir / p.stem / f"{p.stem}_run_summary.txt").exists()
        # provenance.json is opt-in; should NOT be present by default.
        assert not (out_dir / p.stem / "provenance.json").exists()
    # batch_summary.csv and batch_colonies.csv are auto-generated for batches.
    summary_csv = (out_dir / "batch_summary.csv").read_text()
    assert summary_csv.count("\n") >= 4  # header + 3 data rows
    assert (out_dir / "batch_colonies.csv").exists()


def test_run_no_match_exits_with_error(tmp_path: Path) -> None:
    pipe_yaml = tmp_path / "pipe.yaml"
    runner.invoke(app, ["init", "--out", str(pipe_yaml)])
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
    runner.invoke(app, ["init", "--out", str(pipe_yaml)])

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
    summary_csv = (out_dir / "batch_summary.csv").read_text()
    assert "failed" in summary_csv


# --- confidence / nested overrides -------------------------------------------


class _FakePipeline:
    """Pipeline stand-in that skips real execution for config-level tests."""

    def __init__(self, steps: list[dict]) -> None:
        self.steps = steps

    @classmethod
    def from_config(cls, steps: list[dict]) -> _FakePipeline:
        return cls(steps)

    def run(self, *args, **kwargs) -> ImageData:
        return ImageData(source=str(args[0]))


def _write_nested_yolo_pipeline(path: Path) -> None:
    """Write a pipeline whose yolo_detector step sits inside sub_pipeline."""
    path.write_text(
        "steps:\n"
        "  - name: load_image\n"
        "  - name: sub_pipeline\n"
        "    params:\n"
        "      steps:\n"
        "        - name: yolo_detector\n"
        "          params:\n"
        "            conf_threshold: 0.15\n"
        "        - name: measure_colonies\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "args,expected",
    [
        (["--confidence", "0.5"], 0.5),  # dedicated flag
        (["-v", "yolo_detector.conf_threshold=0.7"], 0.7),  # generic override
    ],
)
def test_nested_yolo_conf_is_overridable(tmp_path: Path, args: list[str], expected: float) -> None:
    """Both --confidence and -v reach a yolo_detector inside sub_pipeline."""
    img = _save_synthetic(tmp_path)
    pipe_yaml = tmp_path / "pipe.yaml"
    _write_nested_yolo_pipeline(pipe_yaml)

    out_dir = tmp_path / "out"
    with patch("blenny.cli.main.Pipeline", _FakePipeline):
        result = runner.invoke(
            app,
            ["run", str(pipe_yaml), "--input", str(img), "--output", str(out_dir), *args],
        )
    assert result.exit_code == 0, result.stdout

    config = load_yaml(out_dir / "reproducible_config.yaml")
    inner_steps = config["steps"][1]["params"]["steps"]
    assert inner_steps[0]["name"] == "yolo_detector"
    assert inner_steps[0]["params"]["conf_threshold"] == expected
