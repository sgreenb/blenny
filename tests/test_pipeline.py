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
from blenny.modules import IdentityPreprocessor

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


def test_pipeline_accepts_a_source_string() -> None:
    out = Pipeline([_ConstLoader()]).run("plate.jpg")
    assert out.source == "plate.jpg"
    assert out.image == [[7]]
    assert out.original_image == [[7]]


def test_pipeline_accepts_a_path() -> None:
    from pathlib import Path

    out = Pipeline([_ConstLoader()]).run(Path("plate.jpg"))
    assert out.source == "plate.jpg"


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
    # Durations are numeric and non-negative.
    assert all(p.duration_s >= 0 for p in out.provenance)


def test_provenance_records_serializable_params() -> None:
    out = Pipeline([_ConstLoader(value=5)]).run("x")
    rec = out.provenance[0]
    assert rec.params == {"value": 5}
    assert rec.step == "_ConstLoader"  # default name = class name


def test_instance_name_overrides_class_name_in_provenance() -> None:
    pipe = Pipeline([_AddOne(name="bump"), _AddOne(name="bump-again")])
    pipe.steps.insert(0, _ConstLoader())
    out = pipe.run("x")
    assert [p.step for p in out.provenance] == ["_ConstLoader", "bump", "bump-again"]


# --- Subclass guardrails ------------------------------------------------------


def test_preprocessor_errors_if_no_image_loaded() -> None:
    with pytest.raises(ValueError, match="no image"):
        Pipeline([_AddOne()]).run()


def test_segmenter_errors_if_no_image_loaded() -> None:
    with pytest.raises(ValueError, match="no image"):
        Pipeline([_OnesSegmenter()]).run()


def test_loader_errors_if_no_source() -> None:
    with pytest.raises(ValueError, match="source"):
        Pipeline([_ConstLoader()]).run()


def test_feature_extractor_errors_on_missing_mask() -> None:
    pipe = Pipeline([_ConstLoader(), _CountExtractor()])
    with pytest.raises(ValueError, match="no mask"):
        pipe.run("x")


def test_segmenter_output_key_is_configurable() -> None:
    pipe = Pipeline([_ConstLoader(), _OnesSegmenter(output_key="colonies")])
    out = pipe.run("x")
    assert "colonies" in out.masks
    assert "default" not in out.masks


def test_feature_extractor_reads_configured_mask_key() -> None:
    pipe = Pipeline(
        [
            _ConstLoader(),
            _OnesSegmenter(output_key="colonies"),
            _CountExtractor(mask_key="colonies"),
        ]
    )
    out = pipe.run("plate.jpg")
    assert out.measurements == [{"count": 1, "source": "plate.jpg"}]


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


# --- Identity module is wired through the new API ----------------------------


def test_identity_preprocessor_is_a_noop_under_new_api() -> None:
    pipe = Pipeline([_ConstLoader(value=2), IdentityPreprocessor()])
    out = pipe.run("x")
    assert out.image == [[2]]


# --- Module ABC enforcement ---------------------------------------------------


def test_module_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Module()  # type: ignore[abstract]
