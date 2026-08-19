"""Tests for the module registry and Pipeline.from_config."""

from __future__ import annotations

from typing import Any

import pytest

from blenny import MODULES, BlennyParams, ImageData, Pipeline, Preprocessor, register
from blenny.pipeline.registry import ModuleRegistry


def test_identity_module_is_registered_and_creatable() -> None:
    assert "identity" in MODULES
    cls = MODULES.get("identity")
    assert cls.registry_name == "identity"
    instance = MODULES.create("identity")
    assert instance.name == "identity"  # registry name becomes default instance name


def test_registry_rejects_duplicate_names() -> None:
    reg = ModuleRegistry()

    @reg.register("dup")
    class A(Preprocessor):
        def process(self, image: Any, data: ImageData) -> Any:
            return image

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        class B(Preprocessor):
            def process(self, image: Any, data: ImageData) -> Any:
                return image


def test_registry_unknown_name_raises_with_helpful_message() -> None:
    with pytest.raises(KeyError, match="No module registered"):
        MODULES.get("does-not-exist")


def test_pipeline_from_config_builds_modules_via_registry() -> None:
    # Register a tiny module just for this test, in an isolated registry-style
    # call: we use the global registry but with a unique name.
    @register("test-bump")
    class Bump(Preprocessor):
        class Params(BlennyParams):
            by: int = 1

        def process(self, image: Any, data: ImageData) -> Any:
            return image + self.params.by  # type: ignore[attr-defined]

    pipe = Pipeline.from_config(
        [
            {"name": "identity"},
            {"name": "test-bump", "params": {"by": 10}, "instance_name": "plus-ten"},
        ]
    )
    assert len(pipe) == 2
    assert pipe.steps[1].name == "plus-ten"
    # And it actually runs:
    data = ImageData(source="x", image=5)
    out = pipe.run(data)
    assert out.image == 15
    # Provenance preserves the instance name we asked for.
    assert [p.step for p in out.provenance] == ["identity", "plus-ten"]


def test_pipeline_from_config_requires_name_field() -> None:
    with pytest.raises(ValueError, match="missing 'name'"):
        Pipeline.from_config([{"params": {}}])
