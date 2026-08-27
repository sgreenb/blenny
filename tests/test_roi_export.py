"""End-to-end tests for ROI mode outputs (run_roi_analysis + writers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from blenny.roi import run_roi_analysis
from blenny.roi.export import write_geometry_json, write_overlay, write_pixels_npz, write_rois_csv


def _make_test_image(path: Path) -> None:
    """64x64 image: black background, red block top-left, green block bottom-right."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[8:20, 8:20] = [200, 30, 20]  # red
    img[40:56, 40:56] = [20, 200, 30]  # green
    Image.fromarray(img).save(path)


def _two_rois() -> list[dict]:
    return [
        {
            "id": 1,
            "name": "red_zone",
            "color": "#e6194b",
            "points": [[8, 8], [19, 8], [19, 19], [8, 19]],
        },
        {
            "id": 2,
            "name": "green_zone",
            "color": "#3cb44b",
            "points": [[40, 40], [55, 40], [55, 55], [40, 55]],
        },
    ]


def test_run_roi_analysis_writes_all_outputs(tmp_path: Path) -> None:
    img = tmp_path / "plate.png"
    _make_test_image(img)
    out = tmp_path / "results"

    rows, paths = run_roi_analysis(img, _two_rois(), out, scale=(1.0, 1.0), stem="plate")

    assert len(rows) == 2
    assert rows[0]["name"] == "red_zone"
    assert rows[0]["area_px"] == 144  # rows/cols 8..19 inclusive (12x12)
    assert rows[0]["mean_r"] == 200.0
    assert rows[0]["mean_g"] == 30.0
    assert rows[0]["mean_b"] == 20.0
    assert rows[1]["name"] == "green_zone"
    assert rows[1]["mean_g"] == 200.0

    for kind in ("csv", "pixels", "geometry", "overlay"):
        assert paths[kind].exists(), f"{kind} output missing"

    # CSV summary
    csv_text = paths["csv"].read_text()
    assert (
        csv_text.splitlines()[0]
        == "name,area_px,area_pct,n_pixels,mean_r,mean_g,mean_b,mean_h,mean_s,mean_v"
    )
    assert "red_zone" in csv_text and "green_zone" in csv_text

    # Granular pixel data
    data = np.load(paths["pixels"])
    assert "roi_1_red_zone/rgb" in data
    assert "roi_1_red_zone/hsv" in data
    assert data["roi_1_red_zone/rgb"].shape == (144, 3)

    # Geometry (full-resolution coordinates)
    geom = json.loads(paths["geometry"].read_text())
    assert geom["rois"][0]["name"] == "red_zone"
    assert geom["rois"][0]["points"][0] == [8.0, 8.0]

    # Overlay is a valid image
    with Image.open(paths["overlay"]) as im:
        assert im.size == (64, 64)


def test_run_roi_analysis_scales_display_coords(tmp_path: Path) -> None:
    """Display-space ROIs must be scaled to full resolution before measuring."""
    img = tmp_path / "plate.png"
    _make_test_image(img)
    out = tmp_path / "results"

    # Simulate a canvas drawn at half resolution: ROI covers the whole frame.
    display_rois = [
        {
            "id": 1,
            "name": "everything",
            "color": "#4363d8",
            "points": [[0, 0], [31, 0], [31, 31], [0, 31]],
        }
    ]
    rows, paths = run_roi_analysis(img, display_rois, out, scale=(2.0, 2.0))

    # Scaled to rows/cols 0..62 inclusive -> 63x63 px.
    assert rows[0]["area_px"] == 63 * 63
    geom = json.loads(paths["geometry"].read_text())
    assert geom["rois"][0]["points"][2] == [62.0, 62.0]


def test_canvas_js_has_inlined_palette() -> None:
    """The canvas JS must inline the palette (regression for the right-click bug).

    The close-polygon action colours the new ROI from ``PALETTE``; when the
    palette was referenced but not defined inside the JS, right-click threw a
    ReferenceError and polygons never committed.
    """
    import importlib

    canvas = importlib.import_module("blenny.roi.canvas")
    js = canvas._JS  # type: ignore[attr-defined]
    assert "const PALETTE =" in js
    assert "__PALETTE__" not in js  # placeholder fully replaced
    assert "#e6194b" in js  # first palette colour present


def test_writers_standalone(tmp_path: Path) -> None:
    out = tmp_path / "w"
    csv_path = out / "a.csv"
    write_rois_csv(
        csv_path,
        [
            {
                "name": "r1",
                "area_px": 10,
                "area_pct": 1.0,
                "n_pixels": 10,
                "mean_r": 1.0,
                "mean_g": 2.0,
                "mean_b": 3.0,
                "mean_h": 0.1,
                "mean_s": 0.2,
                "mean_v": 0.3,
            }
        ],
    )
    assert csv_path.exists()

    npz_path = out / "p.npz"
    write_pixels_npz(
        npz_path, {"roi_1_x": {"rgb": np.zeros((3, 3), dtype=np.uint8), "hsv": np.zeros((3, 3))}}
    )
    d = np.load(npz_path)
    assert "roi_1_x/rgb" in d and "roi_keys" in d

    json_path = out / "g.json"
    write_geometry_json(
        json_path,
        [{"id": 1, "name": "x", "color": "#fff", "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]}],
    )
    assert json_path.exists()

    overlay_path = out / "o.png"
    write_overlay(
        overlay_path,
        np.zeros((32, 32, 3), dtype=np.uint8),
        [
            {
                "id": 1,
                "name": "x",
                "color": "#e6194b",
                "points": [[2.0, 2.0], [20.0, 2.0], [20.0, 20.0], [2.0, 20.0]],
            }
        ],
    )
    with Image.open(overlay_path) as im:
        assert im.size == (32, 32)
