"""Tests for blenny.config: YAML loading and path-placeholder substitution."""

from __future__ import annotations

from pathlib import Path

import pytest

from blenny.config import (
    dump_resolved_config,
    extract_steps,
    load_yaml,
    substitute_paths,
)
from blenny.pipeline import Pipeline


def test_load_yaml_returns_mapping(tmp_path: Path) -> None:
    p = tmp_path / "p.yaml"
    p.write_text("steps:\n  - name: identity\n")
    data = load_yaml(p)
    assert data == {"steps": [{"name": "identity"}]}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


def test_extract_steps_validates_shape() -> None:
    with pytest.raises(ValueError, match="missing required 'steps'"):
        extract_steps({})
    with pytest.raises(ValueError, match="must be a list"):
        extract_steps({"steps": "nope"})
    with pytest.raises(ValueError, match="not a mapping"):
        extract_steps({"steps": ["nope"]})
    with pytest.raises(ValueError, match="missing 'name'"):
        extract_steps({"steps": [{"params": {}}]})


def test_substitute_paths_fills_placeholders(tmp_path: Path) -> None:
    steps = [
        {
            "name": "export_csv",
            "params": {"output_path": "{output_dir}/{stem}/x.csv"},
        }
    ]
    out = substitute_paths(
        steps,
        input_path=tmp_path / "plate1.jpg",
        output_dir=tmp_path / "results",
    )
    # Normalize path for cross-platform comparison
    result_path = Path(out[0]["params"]["output_path"])
    assert result_path == tmp_path / "results" / "plate1" / "x.csv"
    # Original is untouched (deep copy semantics).
    assert steps[0]["params"]["output_path"] == "{output_dir}/{stem}/x.csv"

    # Substitution also recurses into nested lists and mappings.
    steps2 = [
        {
            "name": "weird",
            "params": {
                "list": ["{stem}.csv", "fixed"],
                "nested": {"file": "{output_dir}/x"},
            },
        }
    ]
    out2 = substitute_paths(steps2, input_path=Path("a/b/plate.jpg"), output_dir="out")
    assert out2[0]["params"]["list"] == ["plate.csv", "fixed"]
    assert out2[0]["params"]["nested"] == {"file": "out/x"}


def test_substitute_paths_unknown_placeholder_raises() -> None:
    steps = [{"name": "x", "params": {"p": "{not_a_var}"}}]
    with pytest.raises(KeyError, match="Unknown placeholder"):
        substitute_paths(steps, input_path="i", output_dir="o")


def test_substitute_paths_does_not_rewrite_step_name() -> None:
    """Module 'name' is the registry key and must never be substituted."""
    steps = [{"name": "load_image", "params": {"x": "{stem}"}}]
    out = substitute_paths(steps, input_path=Path("plate.jpg"), output_dir="o")
    assert out[0]["name"] == "load_image"
    assert out[0]["params"]["x"] == "plate"


def test_pipeline_from_yaml_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "pipe.yaml"
    p.write_text(
        "steps:\n  - name: identity\n  - name: load_image\n    params:\n      max_dimension: 1500\n"
    )
    pipe = Pipeline.from_yaml(p)
    assert len(pipe) == 2
    assert pipe.steps[0].name == "identity"
    assert pipe.steps[1].params.max_dimension == 1500  # type: ignore[attr-defined]


def test_dump_resolved_config_writes_yaml(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "config.yaml"
    dump_resolved_config(
        [{"name": "identity", "params": {}}],
        out,
        extra={"_blenny_version": "0.0.1"},
    )
    text = out.read_text()
    assert "steps:" in text
    assert "_blenny_version" in text
