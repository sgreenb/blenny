"""Tests for the facile plate detector, including manual override modes."""

from __future__ import annotations

import numpy as np
from PIL import Image

from blenny import ImageData
from blenny.modules import FacileDetector


def _data(image_size: tuple[int, int] = (256, 256)) -> ImageData:
    rng = np.random.default_rng(0)
    img = rng.integers(60, 80, size=(*image_size, 3), dtype=np.uint8)
    return ImageData(source="x", image=img, original_image=img)


def test_forced_circle_params_bypass_detection() -> None:
    data = _data()
    out = FacileDetector(force_cy=128, force_cx=128, force_r=100).run(data)
    rois = out.metadata["rois"]
    assert len(rois) == 1
    assert rois[0]["radius"] == 100
    # center_local == forced center relative to the bbox origin
    assert rois[0]["center_local"] == (128 - rois[0]["bbox"][0], 128 - rois[0]["bbox"][1])
    # plate mask is written and centred on the forced circle
    assert "plate" in out.masks
    assert out.masks["plate"][128, 128]
    assert not out.masks["plate"][0, 0]

    # Manual coordinates are scaled when the working image was resized at load.
    data_scaled = _data()
    data_scaled.metadata["resize_scale"] = 0.5
    out_scaled = FacileDetector(force_cy=200, force_cx=100, force_r=80).run(data_scaled)
    assert out_scaled.metadata["plate_center"] == (100, 50)
    assert out_scaled.metadata["plate_radius"] == 40


def test_forced_mask_path_bypasses_detection(tmp_path) -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[60:196, 60:196] = 255
    p = tmp_path / "mask.png"
    Image.fromarray(mask).save(p)

    data = _data()
    out = FacileDetector(force_mask_path=str(p)).run(data)
    rois = out.metadata["rois"]
    assert len(rois) == 1
    # Equivalent radius of a 136x136 square ≈ sqrt(136²/π) ≈ 77.
    assert 70 <= rois[0]["radius"] <= 85
    # Mask pixels inside the drawn square are plate; outside are not.
    assert out.masks["plate"][128, 128]
    assert not out.masks["plate"][10, 10]


def test_forced_empty_mask_raises_flag(tmp_path) -> None:
    p = tmp_path / "empty.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(p)

    data = _data()
    out = FacileDetector(force_mask_path=str(p)).run(data)
    codes = [f.code for f in out.quality_flags]
    assert "plate_not_found" in codes
    assert out.masks["plate"].all()  # all-True fallback
