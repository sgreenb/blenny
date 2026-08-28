"""Tests for ForcePlate, the GUI's manual Circle/Shape plate step."""

from __future__ import annotations

import numpy as np
from PIL import Image

from blenny import ImageData
from blenny.modules import ForcePlate


def _data(size: tuple[int, int] = (256, 256)) -> ImageData:
    rng = np.random.default_rng(0)
    img = rng.integers(60, 90, size=(*size, 3), dtype=np.uint8)
    return ImageData(source="x", image=img, original_image=img)


def test_force_circle_builds_full_image_roi() -> None:
    data = _data()
    out = ForcePlate(force_cy=128, force_cx=128, force_r=100).run(data)
    rois = out.metadata["rois"]
    assert len(rois) == 1
    # The ROI spans the full frame: manual modes are never cropped.
    assert rois[0]["bbox"] == (0, 0, 256, 256)
    assert rois[0]["center_local"] == (128, 128)
    assert rois[0]["radius"] == 100
    assert out.metadata["plate_center"] == (128, 128)
    # Plate mask is the circle, centred on the forced params.
    assert out.masks["plate"][128, 128]
    assert not out.masks["plate"][0, 0]


def test_force_circle_scales_with_load_resize() -> None:
    data = _data((128, 128))
    data.metadata["resize_scale"] = 0.5
    out = ForcePlate(force_cy=200, force_cx=100, force_r=80).run(data)
    assert out.metadata["plate_center"] == (100, 50)
    assert out.metadata["plate_radius"] == 40


def test_force_mask_path_uses_polygon_exactly(tmp_path) -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[60:196, 60:196] = 255  # a square, NOT a circle
    p = tmp_path / "mask.png"
    Image.fromarray(mask).save(p)

    data = _data()
    out = ForcePlate(force_mask_path=str(p)).run(data)
    rois = out.metadata["rois"]
    assert len(rois) == 1
    assert rois[0]["bbox"] == (0, 0, 256, 256)  # no cropping
    # The plate mask must be the polygon exactly (ground truth, no circle).
    assert np.array_equal(out.masks["plate"], mask > 0)
    # Equivalent radius is metadata only (for the interior reference).
    assert 70 <= rois[0]["radius"] <= 85


def test_force_empty_mask_flags_and_falls_back(tmp_path) -> None:
    p = tmp_path / "empty.png"
    Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).save(p)

    data = _data()
    out = ForcePlate(force_mask_path=str(p)).run(data)
    assert any(f.code == "plate_not_found" for f in out.quality_flags)
    assert out.masks["plate"].all()  # all-True fallback


def test_force_records_plate_shape(tmp_path) -> None:
    """Manual geometry must be tagged so exporters can tell a polygon from a
    circle (a polygon's centroid + equivalent radius must not be drawn as a
    circle on the annotated output)."""
    data = _data()
    assert (
        ForcePlate(force_cy=128, force_cx=128, force_r=100).run(data).metadata["plate_shape"]
        == "manual_circle"
    )

    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[60:196, 60:196] = 255  # a square, NOT a circle
    p = tmp_path / "mask.png"
    Image.fromarray(mask).save(p)
    data2 = _data()
    assert ForcePlate(force_mask_path=str(p)).run(data2).metadata["plate_shape"] == (
        "manual_polygon"
    )


def test_manual_polygon_annotated_has_no_interior_circle(tmp_path) -> None:
    """Manual Polygon mode must not draw the interior-boundary circle on the
    annotated image: the user's polygon is the analysis area, and the circle
    (built from the polygon centroid + equivalent radius) looks like a
    spurious circle detection. Circle-based modes still draw it."""
    from blenny import Pipeline
    from blenny.modules.export_annotated import AnnotatedImageExporter

    rng = np.random.default_rng(1)
    img = rng.integers(40, 70, size=(256, 256, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:256, :256]
    img[(yy - 128) ** 2 + (xx - 128) ** 2 <= 12**2] = 200  # one fake colony

    objects = np.zeros((256, 256), dtype=np.int32)
    objects[(yy - 128) ** 2 + (xx - 128) ** 2 <= 12**2] = 1

    steps = [
        {"name": "measure_colonies", "params": {"mask_key": "objects", "roi_mask_key": "plate"}},
        {"name": "classify_by_interior"},
        {
            "name": "export_annotated",
            "params": {"output_path": str(tmp_path / "x.png"), "draw_numbers": False},
        },
    ]

    def _yellow_count(plate) -> int:
        sub = Pipeline.from_config(steps).run(plate)
        ann = AnnotatedImageExporter(output_path="d.png", draw_numbers=False).render(sub)
        rgb = np.asarray(ann).astype(int)
        yellow = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 200) & (rgb[:, :, 2] < 100)
        return int(yellow.sum())

    # Manual polygon: no circle may appear.
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[40:216, 40:216] = 255  # square
    p = tmp_path / "poly.png"
    Image.fromarray(mask).save(p)
    data = _data((256, 256))
    data.masks["objects"] = objects
    data = ForcePlate(force_mask_path=str(p)).run(data)
    assert _yellow_count(data) == 0

    # Manual circle: the interior boundary circle is still a useful diagnostic.
    data2 = _data((256, 256))
    data2.masks["objects"] = objects
    data2 = ForcePlate(force_cy=128, force_cx=128, force_r=110).run(data2)
    assert _yellow_count(data2) > 200

    # Auto-detected circle (no force_plate): still drawn.
    data3 = _data((256, 256))
    data3.masks["objects"] = objects
    data3.metadata["plate_center"] = (128, 128)
    data3.metadata["plate_radius"] = 110
    data3.masks["plate"] = (yy - 128) ** 2 + (xx - 128) ** 2 <= 110**2
    assert _yellow_count(data3) > 200
