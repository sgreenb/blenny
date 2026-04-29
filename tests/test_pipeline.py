"""Smoke tests for the Step 0 pipeline scaffolding."""

from __future__ import annotations

import pytest

from blenny import Pipeline, PipelineStep
from blenny.modules import IdentityPreprocessor


class _AddKey(PipelineStep):
    name = "AddKey"

    def __init__(self, key: str, value: int) -> None:
        super().__init__()
        self.key = key
        self.value = value

    def run(self, context):  # type: ignore[no-untyped-def]
        context[self.key] = self.value
        return context


def test_empty_pipeline_returns_context_with_provenance() -> None:
    out = Pipeline().run({"foo": 1})
    assert out["foo"] == 1
    assert out["_provenance"] == []


def test_identity_preprocessor_is_a_noop() -> None:
    pipe = Pipeline([IdentityPreprocessor()])
    out = pipe.run({"image": "fake"})
    assert out["image"] == "fake"
    assert out["_provenance"] == ["IdentityPreprocessor"]


def test_pipeline_runs_steps_in_order_and_records_provenance() -> None:
    pipe = Pipeline().add(_AddKey("a", 1)).add(_AddKey("b", 2)).add(IdentityPreprocessor())
    out = pipe.run()
    assert out["a"] == 1
    assert out["b"] == 2
    assert out["_provenance"] == ["AddKey", "AddKey", "IdentityPreprocessor"]


def test_pipeline_does_not_mutate_caller_context() -> None:
    caller_ctx: dict = {"x": 0}
    Pipeline([_AddKey("y", 9)]).run(caller_ctx)
    assert "y" not in caller_ctx
    assert "_provenance" not in caller_ctx


def test_pipeline_step_requires_run_to_be_implemented() -> None:
    with pytest.raises(TypeError):
        PipelineStep()  # type: ignore[abstract]


def test_pipeline_repr_lists_step_names() -> None:
    pipe = Pipeline([IdentityPreprocessor()])
    assert "IdentityPreprocessor" in repr(pipe)
    assert len(pipe) == 1
