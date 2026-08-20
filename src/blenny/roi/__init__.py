"""Region-of-interest (ROI) mode: GUI-only region statistics.

This package is intentionally independent of the colony-counting pipeline:
it computes per-region *area* and *colour* statistics for polygons a user
draws on an image, and writes small output files (CSV summary, raw pixel
arrays for later figure generation, geometry, and an annotated overlay).

It is called only from the GUI's "ROI Mode"; there is no CLI command and no
YAML pipeline involved. The drawing canvas lives in
:mod:`blenny.roi.canvas` (a Streamlit v2 component); the pure computation
lives here so it is unit-testable without Streamlit.

Two-stage flow used by the GUI
------------------------------
* :func:`analyze_rois` — in-memory per-ROI statistics + 256-bin pixel
  histograms for the dashboard. Writes nothing; never mutates the core data.
* :func:`run_roi_analysis` / the writers — persist the core outputs (CSV
  summary, pixel npz, geometry JSON, overlay PNG). ``run_roi_analysis``
  keeps its original signature for tests/back-compat; it is exactly
  ``analyze`` followed by ``write``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from blenny.roi.analyze import (
    roi_exact_stats,
    roi_histograms,
    write_analysis_outputs,
)
from blenny.roi.export import (
    write_geometry_json,
    write_overlay,
    write_pixels_npz,
    write_rois_csv,
)
from blenny.roi.stats import compute_roi_stats, polygon_mask, scale_points

__all__ = ["analyze_rois", "run_roi_analysis", "write_analysis_outputs"]


def _slug(name: str) -> str:
    """Make a filesystem/npz-friendly key from a ROI name."""
    cleaned = "".join(c if c.isalnum() else "_" for c in name).strip("_")
    return cleaned or "roi"


def _prepare(
    image_path: str | Path, rois: list[dict[str, Any]], scale: tuple[float, float]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load the full-resolution image and map ROIs into its coordinate frame."""
    with Image.open(image_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    arr = np.asarray(im)
    sx, sy = scale
    fullres_rois = [{**r, "points": scale_points(r["points"], sx, sy)} for r in rois]
    return arr, fullres_rois


def _analyze_impl(
    arr: np.ndarray, fullres_rois: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], dict, dict]:
    """Shared loop: per-ROI summary rows, pixel arrays, histograms, exact stats."""
    h, w = arr.shape[:2]
    rows: list[dict[str, Any]] = []
    pixel_data: dict[str, dict[str, np.ndarray]] = {}
    hists: dict[int, dict[str, np.ndarray]] = {}
    exact: dict[int, dict[str, tuple[int, float, float]]] = {}
    for roi in fullres_rois:
        mask = polygon_mask(roi["points"], (h, w))
        stats, rgb_px, hsv_px = compute_roi_stats(arr, mask)
        rows.append({"name": roi["name"], **stats})
        pixel_data[f"roi_{len(rows)}_{_slug(roi['name'])}"] = {
            "rgb": rgb_px,
            "hsv": hsv_px,
        }
        rid = roi.get("id")
        if rid is not None:
            hists[rid] = roi_histograms(rgb_px, hsv_px)
            exact[rid] = roi_exact_stats(rgb_px, hsv_px)
    return rows, pixel_data, hists, exact


def analyze_rois(
    image_path: str | Path,
    rois: list[dict[str, Any]],
    *,
    scale: tuple[float, float] = (1.0, 1.0),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute in-memory analysis data for the dashboard. Writes nothing.

    Returns ``(rows, hist_data)`` where ``rows`` is one summary dict per ROI
    (name, area, RGB/HSV means) and ``hist_data`` is ``{"hists", "exact"}``
    keyed by ROI id — 256-bin per-parameter histograms plus exact full-range
    ``(n, mean, std)`` stats, suitable for :func:`write_analysis_outputs`.
    """
    arr, fullres_rois = _prepare(image_path, rois, scale)
    rows, _, hists, exact = _analyze_impl(arr, fullres_rois)
    return rows, {
        "hists": hists,
        "exact": exact,
    }


def run_roi_analysis(
    image_path: str | Path,
    rois: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    scale: tuple[float, float] = (1.0, 1.0),
    stem: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    """Analyse every ROI and write the core outputs into ``output_dir``.

    ``rois`` are polygon definitions drawn on the *display* image (so
    coordinates are in display space); ``scale = (orig_w / display_w,
    orig_h / display_h)`` maps them back to the full-resolution image before
    any measurement is taken.

    Returns ``(rows, paths)`` where ``rows`` is one summary dict per ROI and
    ``paths`` maps output kind → file path: ``csv``, ``pixels`` (npz),
    ``geometry`` (json), ``overlay`` (png).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arr, fullres_rois = _prepare(image_path, rois, scale)
    rows, pixel_data, _, _ = _analyze_impl(arr, fullres_rois)

    stem = stem or Path(image_path).stem
    paths = {
        "csv": out / f"{stem}_rois.csv",
        "pixels": out / f"{stem}_roi_pixels.npz",
        "geometry": out / f"{stem}_roi_geometry.json",
        "overlay": out / f"{stem}_roi_overlay.png",
    }

    write_rois_csv(paths["csv"], rows)
    write_pixels_npz(paths["pixels"], pixel_data)
    write_geometry_json(paths["geometry"], fullres_rois)
    write_overlay(paths["overlay"], arr, fullres_rois)

    return rows, paths
