"""Tests for the ImageData context object and its helpers."""

from __future__ import annotations

from blenny import ImageData, QualityFlag


def test_imagedata_defaults_are_independent_per_instance() -> None:
    a = ImageData()
    b = ImageData()
    a.masks["x"] = 1
    a.measurements.append({"k": 1})
    a.metadata["pixel_size_um"] = 0.65
    a.artifacts["debug"] = object()
    assert b.masks == {}
    assert b.measurements == []
    assert b.metadata == {}
    assert b.artifacts == {}


def test_add_flag_appends_a_quality_flag() -> None:
    data = ImageData()
    data.add_flag("low_contrast", "Image is very flat", severity="warning")
    assert len(data.quality_flags) == 1
    flag = data.quality_flags[0]
    assert isinstance(flag, QualityFlag)
    assert flag.code == "low_contrast"
    assert flag.severity == "warning"
    assert flag.step == ""  # filled by the runner
