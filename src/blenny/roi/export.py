"""Output writers for ROI mode: CSV summary, pixel arrays, geometry, overlay.

Each writer is a pure function taking explicit paths, so the GUI just calls
:func:`blenny.roi.run_roi_analysis` and everything lands in the output folder.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def write_rois_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the per-ROI summary CSV (name, area, colour means)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "area_px",
        "area_pct",
        "n_pixels",
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_h",
        "mean_s",
        "mean_v",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_pixels_npz(path: Path, pixel_data: dict[str, dict[str, np.ndarray]]) -> None:
    """Save the granular per-ROI pixel arrays for later analysis.

    Layout (accessible via ``np.load``):
        arr = np.load(path)
        arr["roi_1_control/rgb"]   -> (N, 3) uint8
        arr["roi_1_control/hsv"]   -> (N, 3) float [0, 1]
    A ``names`` array maps each key to the human-readable ROI name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    names: list[str] = []
    for key, entry in pixel_data.items():
        arrays[f"{key}/rgb"] = np.asarray(entry["rgb"])
        arrays[f"{key}/hsv"] = np.asarray(entry["hsv"])
        names.append(key)
    arrays["roi_keys"] = np.asarray(names)
    np.savez(path, **arrays)


def write_geometry_json(path: Path, rois: list[dict[str, Any]]) -> None:
    """Save ROI definitions (names, colours, polygon vertices) as JSON.

    Vertices are stored in full-resolution image coordinates so the regions
    can be re-applied to the same image later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rois": [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "color": r.get("color"),
                "points": [[float(x), float(y)] for x, y in r["points"]],
            }
            for r in rois
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hex_to_rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    hex_str = color.lstrip("#")
    r, g, b = (int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)


def write_overlay(path: Path, image_rgb: np.ndarray, rois: list[dict[str, Any]]) -> None:
    """Draw every ROI (filled + outlined + named) on the full-res image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(12, int(min(base.size) * 0.02))
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    for roi in rois:
        pts = [(float(p[0]), float(p[1])) for p in roi["points"]]
        if len(pts) < 3:
            continue
        color = roi.get("color", "#e6194b")
        draw.polygon(pts, fill=_hex_to_rgba(color, 60), outline=_hex_to_rgba(color, 255))
        # Re-stroke the outline with a thicker line for visibility.
        draw.line([*pts, pts[0]], fill=_hex_to_rgba(color, 255), width=3, joint="curve")

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        draw.text(
            (cx, cy),
            str(roi.get("name", "")),
            font=font,
            fill=(255, 255, 255, 255),
            anchor="mm",
            stroke_width=2,
            stroke_fill=(0, 0, 0, 180),
        )

    Image.alpha_composite(base, overlay).convert("RGB").save(path)
