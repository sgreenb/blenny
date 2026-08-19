"""Tests for the Pipeline runner and the semantic Module subclasses."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from blenny import (
    BlennyParams,
    Exporter,
    FeatureExtractor,
    ImageData,
    Loader,
    Module,
    Pipeline,
    Preprocessor,
    Segmenter,
)

# --- Fakes --------------------------------------------------------------------


class _ConstLoader(Loader):
    class Params(BlennyParams):
        value: int = 7

    def load(self, source: str) -> Any:
        return [[self.params.value]]  # type: ignore[attr-defined]


class _AddOne(Preprocessor):
    def process(self, image: Any, data: ImageData) -> Any:
        return [[cell + 1 for cell in row] for row in image]


class _OnesSegmenter(Segmenter):
    def segment(self, image: Any, data: ImageData) -> Any:
        return [[1 for _ in row] for row in image]


class _CountExtractor(FeatureExtractor):
    def extract(self, image: Any, mask: Any, data: ImageData) -> list[dict[str, Any]]:
        n = sum(sum(row) for row in mask)
        return [{"count": n}]


class _RecordExporter(Exporter):
    def __init__(self, sink: list[ImageData], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sink = sink

    def export(self, data: ImageData) -> None:
        self.sink.append(data)


# --- Pipeline runner ----------------------------------------------------------


def test_empty_pipeline_returns_imagedata_with_no_provenance() -> None:
    out = Pipeline().run()
    assert isinstance(out, ImageData)
    assert out.provenance == []


def test_pipeline_accepts_source_string_or_path() -> None:
    from pathlib import Path

    for source in ("plate.jpg", Path("plate.jpg")):
        out = Pipeline([_ConstLoader()]).run(source)
        assert out.source == "plate.jpg"
        assert out.image == [[7]]
        assert out.original_image == [[7]]


def test_pipeline_rejects_unknown_input_type() -> None:
    with pytest.raises(TypeError):
        Pipeline().run(42)  # type: ignore[arg-type]


def test_full_vertical_slice_loader_to_exporter() -> None:
    sink: list[ImageData] = []
    pipe = Pipeline(
        [
            _ConstLoader(value=3),
            _AddOne(),
            _OnesSegmenter(),
            _CountExtractor(),
            _RecordExporter(sink=sink),
        ]
    )
    out = pipe.run("plate.jpg")
    assert out.image == [[4]]
    assert out.masks["default"] == [[1]]
    assert out.measurements == [{"count": 1, "source": "plate.jpg"}]
    assert sink == [out]
    # One provenance record per step, in order, with module class names.
    assert [p.module_class for p in out.provenance] == [
        "_ConstLoader",
        "_AddOne",
        "_OnesSegmenter",
        "_CountExtractor",
        "_RecordExporter",
    ]
    # Params are serializable; the loader records its value.
    assert out.provenance[0].params == {"value": 3}
    # Durations are numeric and non-negative.
    assert all(p.duration_s >= 0 for p in out.provenance)


def test_instance_name_overrides_class_name_in_provenance() -> None:
    pipe = Pipeline([_AddOne(name="bump"), _AddOne(name="bump-again")])
    pipe.steps.insert(0, _ConstLoader())
    out = pipe.run("x")
    assert [p.step for p in out.provenance] == ["_ConstLoader", "bump", "bump-again"]


# --- Subclass guardrails ------------------------------------------------------


def test_semantic_subclasses_raise_helpful_errors_without_input() -> None:
    with pytest.raises(ValueError, match="no image"):
        Pipeline([_AddOne()]).run()  # preprocessor
    with pytest.raises(ValueError, match="no image"):
        Pipeline([_OnesSegmenter()]).run()  # segmenter
    with pytest.raises(ValueError, match="source"):
        Pipeline([_ConstLoader()]).run()  # loader
    with pytest.raises(ValueError, match="no mask"):
        Pipeline([_ConstLoader(), _CountExtractor()]).run("x")  # feature extractor


def test_output_and_mask_keys_are_configurable() -> None:
    # A segmenter can store its mask under a custom output_key...
    pipe = Pipeline([_ConstLoader(), _OnesSegmenter(output_key="colonies")])
    out = pipe.run("x")
    assert "colonies" in out.masks
    assert "default" not in out.masks
    # ...and the extractor can read that key back.
    pipe2 = Pipeline(
        [
            _ConstLoader(),
            _OnesSegmenter(output_key="colonies"),
            _CountExtractor(mask_key="colonies"),
        ]
    )
    out2 = pipe2.run("plate.jpg")
    assert out2.measurements == [{"count": 1, "source": "plate.jpg"}]


# --- Quality flags ------------------------------------------------------------


class _FlaggingPreprocessor(Preprocessor):
    def process(self, image: Any, data: ImageData) -> Any:
        data.add_flag("low_contrast", "Image is flat")
        return image


def test_runner_stamps_unattributed_quality_flags_with_step_name() -> None:
    pipe = Pipeline([_ConstLoader(), _FlaggingPreprocessor(name="contrast-check")])
    out = pipe.run("x")
    assert out.quality_flags[0].step == "contrast-check"


# --- Params validation --------------------------------------------------------


def test_unknown_param_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _ConstLoader(value=1, bogus=2)


# --- Module ABC enforcement ---------------------------------------------------


def test_module_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Module()  # type: ignore[abstract]
