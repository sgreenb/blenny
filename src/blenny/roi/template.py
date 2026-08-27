"""ROI template loading: parse JSON templates and map them onto the canvas.

Pure functions (no Streamlit), so they are unit-testable and reusable outside
the GUI. The GUI uploads a JSON template in the sidebar, parses it here, and
maps the full-resolution coordinates onto the display-space canvas with
:func:`map_to_display` — the exact inverse of
:func:`blenny.roi.stats.scale_points` (display → full-res) — so a saved
``*_roi_geometry.json`` round-trips with pixel-perfect accuracy.

Template format
---------------
Full-resolution image coordinates (what :func:`write_geometry_json` emits)::

    {"rois": [{"id": 1, "name": "Well A1", "color": "#e6194b",
               "points": [[x, y], [x, y], ...]}]}

``id`` / ``name`` / ``color`` are optional: missing or duplicate ids are
renumbered, missing names default to ``ROI N``, and missing colours are
filled in by the GUI from the canvas palette. A bare top-level list of ROIs
is also accepted. Canvas-state JSON (with ``draft`` / ``nextId`` keys, i.e.
display-space coordinates) is rejected with a helpful message, since its
points are not in full-resolution space.
"""

from __future__ import annotations

import json
import math
from typing import Any


def parse_template(data: bytes | bytearray | str) -> list[dict[str, Any]]:
    """Parse and structurally validate a template into full-res ROI dicts.

    Accepts either ``{"rois": [...]}`` or a bare ``[...]`` list of ROI
    objects. Raises :class:`ValueError` with a human-readable message on any
    structural problem (bad JSON, canvas-state format, missing ROIs,
    non-numeric or non-finite vertices, fewer than 3 points per ROI).
    """
    text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg} at line {exc.lineno})") from exc

    if isinstance(payload, dict):
        if "draft" in payload or "nextId" in payload:
            raise ValueError(
                "this looks like canvas-state JSON (display-space coordinates). "
                "Use full-resolution coordinates, e.g. a saved *_roi_geometry.json."
            )
        rois_raw = payload.get("rois")
        if rois_raw is None:
            raise ValueError('expected a "rois" list (e.g. {"rois": [...]})')
    elif isinstance(payload, list):
        rois_raw = payload
    else:
        raise ValueError('expected a JSON object with a "rois" list, or a bare list')

    if not isinstance(rois_raw, list):
        raise ValueError('"rois" must be a list of ROI objects')

    rois: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for i, r in enumerate(rois_raw, start=1):
        if not isinstance(r, dict):
            raise ValueError(f"ROI #{i} is not a JSON object")
        pts = r.get("points")
        if not isinstance(pts, list) or len(pts) < 3:
            raise ValueError(f"ROI #{i} ({r.get('name', 'unnamed')}) needs at least 3 points")
        clean_pts: list[list[float]] = []
        for j, p in enumerate(pts, start=1):
            if (
                not isinstance(p, (list, tuple))
                or len(p) != 2
                or not all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in p
                )
            ):
                raise ValueError(f"ROI #{i} vertex {j} must be a pair of numbers (got {p!r})")
            clean_pts.append([float(p[0]), float(p[1])])

        rid = r.get("id")
        if not isinstance(rid, int) or isinstance(rid, bool) or rid in seen_ids:
            rid = i  # missing / duplicate / invalid id → renumber
        seen_ids.add(rid)
        rois.append(
            {
                "id": rid,
                "name": str(r.get("name") or f"ROI {i}"),
                "color": r.get("color"),
                "points": clean_pts,
            }
        )
    return rois


def map_to_display(
    rois: list[dict[str, Any]],
    full_w: int,
    full_h: int,
    disp_w: int,
    disp_h: int,
    *,
    tolerance: float = 1.0,
) -> list[dict[str, Any]]:
    """Validate full-res bounds and map template ROIs into display space.

    Every vertex must lie within the full-resolution image, plus a small
    ``tolerance`` in pixels that absorbs float round-trip error from a saved
    geometry file; otherwise a :class:`ValueError` naming every offending ROI
    and vertex is raised and nothing is returned. Mapped points are clamped
    into the display canvas. Returned dicts reuse the input ``id`` /
    ``name`` / ``color`` and hold display-space ``points``.
    """
    if full_w <= 0 or full_h <= 0:
        raise ValueError("the image has no size")
    sx = disp_w / full_w
    sy = disp_h / full_h

    violations: list[str] = []
    for r in rois:
        for j, (x, y) in enumerate(r["points"], start=1):
            if not (
                -tolerance <= x <= full_w + tolerance and -tolerance <= y <= full_h + tolerance
            ):
                violations.append(
                    f"ROI '{r.get('name', 'unnamed')}' vertex {j} "
                    f"({x:.1f}, {y:.1f}) is beyond the image "
                    f"({full_w} x {full_h} px)"
                )
    if violations:
        shown = (
            violations
            if len(violations) <= 5
            else [
                *violations[:5],
                f"... and {len(violations) - 5} more",
            ]
        )
        raise ValueError("; ".join(shown))

    out: list[dict[str, Any]] = []
    for r in rois:
        points = [
            [min(max(x * sx, 0.0), float(disp_w)), min(max(y * sy, 0.0), float(disp_h))]
            for x, y in r["points"]
        ]
        out.append({**r, "points": points})
    return out
