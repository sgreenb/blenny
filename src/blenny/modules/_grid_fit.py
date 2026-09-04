"""Grid-fitting geometry for multi-plate detection.

Given the centres of detected plates, this module infers the regular grid
(rows x cols) that best explains their positions and reports which grid slots
are occupied vs. empty. It is deliberately independent of the Blender/pipeline
data structures so the geometry can be unit-tested in isolation.

The algorithm fits an axis-aligned lattice

    x = x0 + col * dx
    y = y0 + row * dy

to the detected centres by iterating nearest-slot assignment and least-squares
refinement. Validation bounds decide whether the detected layout is "grid-like
enough" to be trusted, or whether the caller should fall back to auto-detection
(labels 1..N) because the layout is not a regular grid.

Known limitations (accepted for v1):
- A grid in which a *whole* interior row or column is missing cannot be
  distinguished from a tighter grid (the fitted spacing doubles). Detections in
  this ambiguous case map to the leftmost/topmost slots, which is usually wrong.
  The common case (a few scattered empty cells) is handled correctly.
- Rotation is not modelled: the image is treated as the true orientation. A
  genuinely rotated capture produces large residuals and fails validation, which
  is the intended "use auto mode" signal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GridFitResult:
    """Outcome of fitting a grid to detected plate centres."""

    ok: bool
    error: str = ""
    """Human-readable reason when ``ok`` is False (empty otherwise)."""

    assignments: dict[int, tuple[int, int]] = field(default_factory=dict)
    """detection index -> (row, col) slot the plate was mapped to."""

    residuals: list[float] = field(default_factory=list)
    """Euclidean distance (px) from each detection to its assigned slot."""

    empty_slots: list[tuple[int, int]] = field(default_factory=list)
    """Grid slots (row, col) that had no plate."""

    dx: float = 0.0
    """Fitted column spacing (px)."""

    dy: float = 0.0
    """Fitted row spacing (px)."""

    origin: tuple[float, float] = (0.0, 0.0)
    """Fitted (x0, y0) position of the slot (row 0, col 0)."""


def fit_grid_to_centers(
    centers: Sequence[tuple[float, float]],
    rows: int,
    cols: int,
    *,
    radii: Sequence[float] | None = None,
    min_detections: int = 2,
    max_residual_frac: float = 0.30,
    max_outlier_frac: float = 0.25,
    max_median_residual_frac: float = 0.15,
    min_spacing_radius_ratio: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> GridFitResult:
    """Fit a rows x cols axis-aligned grid to ``centers`` (list of [(x, y), ...]).

    Returns a :class:`GridFitResult`. ``ok`` is True only when every detected
    plate maps to a distinct slot, the fitted spacing is sensible, and the
    fraction of large-residual (outlier) detections stays under
    ``max_outlier_frac``.
    """
    n = len(centers)
    if n == 0:
        return GridFitResult(ok=False, error="no plates to fit a grid to")
    if n < min_detections:
        return GridFitResult(
            ok=False,
            error=(
                f"Only {n} plate(s) detected; at least {min_detections} are needed "
                "to infer a grid."
            ),
        )

    pts = np.asarray(centers, dtype=float)
    if pts.shape != (n, 2):
        raise ValueError(f"centers must be a sequence of (x, y) pairs; got shape {pts.shape}")

    (x0, y0), dx, dy = _initial_guess(pts, rows, cols, radii)

    # --- Iterate nearest-slot assignment + least-squares refinement ---------
    assignments: list[tuple[int, int]] = []
    for _ in range(max_iter):
        assignments, _residuals = _assign_to_slots(pts, (x0, y0), dx, dy, rows, cols)
        nx, ny, ndx, ndy = _solve_lattice(pts, assignments, (x0, y0), dx, dy)
        if (
            abs(nx - x0) < tol
            and abs(ny - y0) < tol
            and abs(ndx - dx) < tol
            and abs(ndy - dy) < tol
        ):
            x0, y0, dx, dy = nx, ny, ndx, ndy
            break
        x0, y0, dx, dy = nx, ny, ndx, ndy

    # --- Final assignment + residual ----------------------------------------
    assignments, residuals = _assign_to_slots(pts, (x0, y0), dx, dy, rows, cols)

    # --- Validation ---------------------------------------------------------
    # A single-row (rows==1) or single-column (cols==1) grid has no genuine
    # spacing along the degenerate axis. We keep positive sentinels there and
    # skip the spacing-dependent checks on that axis.
    real_dx = dx if cols > 1 else None
    real_dy = dy if rows > 1 else None
    spacings = [s for s in (real_dx, real_dy) if s is not None]

    # 1. Every slot must be unique (grid too small for the detection count).
    seen: set[tuple[int, int]] = set()
    for r, c in assignments:
        if (r, c) in seen:
            return GridFitResult(
                ok=False,
                error=(
                    f"{len(assignments)} plates were mapped to a {rows}x{cols} grid but "
                    f"two landed on the same slot. The grid has fewer cells than plates, "
                    "or detection returned duplicate circles for one plate; use a larger "
                    "grid or Auto mode."
                ),
                assignments={i: s for i, s in enumerate(assignments)},
                residuals=residuals,
                dx=dx,
                dy=dy,
                origin=(x0, y0),
            )
        seen.add((r, c))

    # 2. Real spacing must be positive on any axis that has more than one slot.
    if any(s <= 0 for s in spacings):
        return GridFitResult(
            ok=False,
            error=(
                "Could not infer sensible row/column spacing from the plate "
                "positions (detections may lie on a single line). Use Auto mode."
            ),
            assignments={i: s for i, s in enumerate(assignments)},
            residuals=residuals,
            dx=dx,
            dy=dy,
            origin=(x0, y0),
        )

    # 3. Plates must be meaningfully separated relative to their radius.
    if radii is not None and len(radii) == n and spacings:
        med_r = float(np.median(np.asarray(radii, dtype=float)))
        min_spacing = min_spacing_radius_ratio * med_r
        if min(spacings) < min_spacing:
            return GridFitResult(
                ok=False,
                error=(
                    f"Fitted spacing ({min(spacings):.0f}px) is smaller than "
                    f"{min_spacing_radius_ratio:.2f}x the median plate radius "
                    f"({med_r:.0f}px); plates may be overlapping. Use Auto mode."
                ),
                assignments={i: s for i, s in enumerate(assignments)},
                residuals=residuals,
                dx=dx,
                dy=dy,
                origin=(x0, y0),
            )

    # 4. Reject layouts that are not grid-like. The residual thresholds are
    #    relative to the smallest real spacing (never the sentinel).
    min_spacing_eff = min(spacings) if spacings else 1.0

    # 4a. Too many individual plates sit far from their assigned slot.
    max_residual = max_residual_frac * min_spacing_eff
    outliers = [i for i, r in enumerate(residuals) if r > max_residual]
    if len(outliers) > max_outlier_frac * n:
        return GridFitResult(
            ok=False,
            error=(
                f"{len(outliers)}/{n} plates do not lie on a regular grid "
                f"(max residual {max_residual:.0f}px). The layout is not a true "
                "grid; use Auto mode."
            ),
            assignments={i: s for i, s in enumerate(assignments)},
            residuals=residuals,
            dx=dx,
            dy=dy,
            origin=(x0, y0),
        )

    # 4b. Or the whole layout is systematically shifted (typical placement error
    #     too large), even if no single plate is an egregious outlier.
    med_residual = float(np.median(residuals)) if residuals else 0.0
    if med_residual > max_median_residual_frac * min_spacing_eff:
        return GridFitResult(
            ok=False,
            error=(
                f"Plates are scattered too far from a regular grid "
                f"(median placement error {med_residual:.1f}px vs cell spacing "
                f"{min_spacing_eff:.0f}px). Use Auto mode."
            ),
            assignments={i: s for i, s in enumerate(assignments)},
            residuals=residuals,
            dx=dx,
            dy=dy,
            origin=(x0, y0),
        )

    # --- Build result -------------------------------------------------------
    assignments_map: dict[int, tuple[int, int]] = {i: s for i, s in enumerate(assignments)}
    empty: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in seen:
                empty.append((r, c))

    return GridFitResult(
        ok=True,
        assignments=assignments_map,
        residuals=residuals,
        empty_slots=empty,
        dx=dx,
        dy=dy,
        origin=(x0, y0),
    )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _initial_guess(
    pts: np.ndarray,
    rows: int,
    cols: int,
    radii: Sequence[float] | None,
) -> tuple[tuple[float, float], float, float]:
    """Return ((x0, y0), dx, dy) starting the lattice fit.

    Spacing is estimated from the median gap between *distinct* column/row
    positions (near-duplicates are merged). The origin is anchored at the
    top-left detection, which is correct whenever any plate sits in column 0
    and any plate sits in row 0.
    """
    xs = pts[:, 0]
    ys = pts[:, 1]

    merge_tol = float(np.median(np.asarray(radii, dtype=float))) if radii else 1.0
    # A conservative fallback if radii are absurdly small.
    if merge_tol < 1.0:
        merge_tol = 1.0

    dx = _axis_spacing(xs, merge_tol)
    dy = _axis_spacing(ys, merge_tol)

    # If a direction collapsed (all plates on one line), fall back to a
    # spacing derived from the other axis so rounding doesn't divide by zero.
    if dx <= 0:
        dx = _axis_spacing(xs, merge_tol) or 1.0
    if dy <= 0:
        dy = _axis_spacing(ys, merge_tol) or 1.0

    # Fallback to a neutral spacing if still undefined (single column/row).
    if dx <= 0:
        dx = max(1.0, float(np.ptp(xs)) if xs.size else 1.0)
    if dy <= 0:
        dy = max(1.0, float(np.ptp(ys)) if ys.size else 1.0)

    x0 = float(np.min(xs)) if xs.size else 0.0
    y0 = float(np.min(ys)) if ys.size else 0.0
    return (x0, y0), dx, dy


def _axis_spacing(values: np.ndarray, merge_tol: float) -> float:
    """Estimate the grid spacing (gap between adjacent rows/cols) on one axis.

    Near-duplicate positions (same column / same row) are merged within
    ``merge_tol`` before the median gap is taken, so plate jitter within a
    slot doesn't collapse the spacing estimate.
    """
    uniq = _merge_close(values, merge_tol)
    if uniq.size < 2:
        return 0.0
    diffs = np.diff(uniq)
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs[diffs > 1e-9]))


def _merge_close(values: np.ndarray, tol: float) -> np.ndarray:
    """Sort ``values`` and collapse runs separated by less than ``tol`` into one."""
    v = np.sort(values)
    if v.size == 0:
        return v
    out = [v[0]]
    for x in v[1:]:
        if x - out[-1] > tol:
            out.append(x)
    return np.asarray(out, dtype=float)


def _assign_to_slots(
    pts: np.ndarray,
    origin: tuple[float, float],
    dx: float,
    dy: float,
    rows: int,
    cols: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Assign each point to its nearest (row, col) slot; return (slots, residuals)."""
    x0, y0 = origin
    assignments: list[tuple[int, int]] = []
    residuals: list[float] = []
    for x, y in pts:
        c = round((x - x0) / dx) if dx > 0 else 0
        r = round((y - y0) / dy) if dy > 0 else 0
        c = max(0, min(cols - 1, c))
        r = max(0, min(rows - 1, r))
        sx = x0 + c * dx
        sy = y0 + r * dy
        residuals.append(math.hypot(float(x - sx), float(y - sy)))
        assignments.append((r, c))
    return assignments, residuals


def _solve_lattice(
    pts: np.ndarray,
    assignments: list[tuple[int, int]],
    origin: tuple[float, float],
    dx: float,
    dy: float,
) -> tuple[float, float, float, float]:
    """Least-squares refine (x0, y0, dx, dy) from the given assignments.

    If a direction has fewer than two distinct slot indices, the corresponding
    spacing cannot be refined and the previous value is kept.
    """
    x0, y0 = origin
    x_arr = pts[:, 0]
    y_arr = pts[:, 1]
    r_arr = np.array([r for r, _ in assignments], dtype=float)
    c_arr = np.array([c for _, c in assignments], dtype=float)

    if np.unique(c_arr).size >= 2:
        A = np.column_stack([np.ones(len(c_arr)), c_arr])
        coef, *_ = np.linalg.lstsq(A, x_arr, rcond=None)
        x0, dx = float(coef[0]), float(coef[1])

    if np.unique(r_arr).size >= 2:
        A = np.column_stack([np.ones(len(r_arr)), r_arr])
        coef, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
        y0, dy = float(coef[0]), float(coef[1])

    return x0, y0, dx, dy
