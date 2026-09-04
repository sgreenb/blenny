"""Find multiple circular Petri-dish regions arranged in a regular grid.

Grid mode is a *naming* convenience on top of auto-detection. It runs the same
whole-image detection as auto mode (:func:`blenny.modules.detect_facile.detect_plates`)
to find the real plates, then fits a ``rows x cols`` grid to the detected
centre-points and maps each plate to the nearest slot. Cells the detection found
no plate in (empty slots) are reported via a ``plate_not_found`` flag and get no
ROI, so a run with fewer plates than grid cells no longer fabricates a phantom
plate to "fill out" the grid.

If the detected layout is too irregular to be a true grid, the module falls back
to auto-labeled ROIs (labels 1..N) and raises ``plate_grid_mapping_failed`` so
the user is told to use Auto mode instead. See :mod:`blenny.modules._grid_fit`
for the fit geometry and its tolerance bounds.
"""

from __future__ import annotations

from typing import Any

from blenny.modules._grid_fit import fit_grid_to_centers
from blenny.modules.detect_facile import detect_plates
from blenny.pipeline import BlennyParams, Field, ImageData, Preprocessor, register


@register("detect_multi_plate")
class MultiPlateDetector(Preprocessor):
    class Params(BlennyParams):
        grid: list[int] = Field(default_factory=lambda: [1, 1])
        """Number of [rows, cols] in the image."""

        labels: list[list[str]] | None = None
        """Optional 2D list of labels for each grid cell."""

        radius_scale: float = 1.0
        """Scale factor for the detected plate radius."""

        min_points: int | None = None
        """Minimum number of edge points required to accept a circle."""

        max_error: float | None = None
        """Maximum allowed standard deviation for circle fitting."""

        min_confidence_score: float = 0.0
        """Legacy parameter for compatibility with Hough-based templates."""

        suppression_dist_frac: float = 0.45
        """Fraction of a plate's radius used as the centre-to-centre threshold when
        deduplicating overlapping circle fits (matches facile)."""

        size_consistency_limit: float | None = 0.15
        """Max allowed deviation from the median radius (0.15 = 15%). Only applied
        when multiple plates are found. Set to None to disable."""

        # ---- Grid-fit tolerance bounds (see blenny.modules._grid_fit) -------
        max_residual_frac: float = 0.30
        """A detected plate whose distance from its assigned grid slot exceeds this
        fraction of the row/column spacing is treated as an outlier."""

        max_outlier_frac: float = 0.25
        """If more than this fraction of plates are outliers, the layout is treated
        as not a true grid and grid mapping fails."""

        max_median_residual_frac: float = 0.15
        """If the *median* placement error exceeds this fraction of the spacing, the
        whole layout is treated as scattered (not a grid) and mapping fails."""

        min_spacing_radius_ratio: float = 1.0
        """Reject a fit whose row/column spacing is smaller than this multiple of the
        median plate radius (i.e. plates appear to overlap)."""

    def process(self, image: Any, data: ImageData) -> Any:
        rows, cols = self.params.grid
        h, w = image.shape[:2]

        # 1. Auto-detect the real plates (identical logic to auto mode).
        circles, _n_raw = detect_plates(
            image,
            petri_only=True,
            min_points=self.params.min_points,
            max_error=self.params.max_error,
            suppression_dist_frac=self.params.suppression_dist_frac,  # type: ignore[attr-defined]
            size_consistency_limit=self.params.size_consistency_limit,  # type: ignore[attr-defined]
            data=data,
        )

        if not circles:
            data.add_flag(
                "no_plates_found",
                "MultiPlateDetector found zero plates in the image.",
                severity="error",
            )
            return image

        # --- 1x1 grid shortcut -------------------------------------------
        if rows == 1 and cols == 1:
            xc0, yc0, r0, _std0, _n0 = circles[0]
            single_rois = [self._build_roi(xc0, yc0, r0, label=self._get_label(0, 0), h=h, w=w)]
            data.metadata["rois"] = single_rois
            data.metadata["multi_plate_mode"] = True
            if len(circles) > 1:
                data.add_flag(
                    "no_plates_found",
                    f"{len(circles)} plates were detected but the grid is 1x1; "
                    "only one was mapped.",
                    severity="error",
                )
            return image

        # 2. Fit the user's grid to the detected centres and map each plate to a slot.
        centers: list[tuple[float, float]] = [(c[0], c[1]) for c in circles]
        radii: list[float] = [c[2] for c in circles]
        fit = fit_grid_to_centers(
            centers,
            rows,
            cols,
            radii=radii,
            max_residual_frac=self.params.max_residual_frac,  # type: ignore[attr-defined]
            max_outlier_frac=self.params.max_outlier_frac,  # type: ignore[attr-defined]
            max_median_residual_frac=self.params.max_median_residual_frac,  # type: ignore[attr-defined]
            min_spacing_radius_ratio=self.params.min_spacing_radius_ratio,  # type: ignore[attr-defined]
        )

        if not fit.ok:
            # 3a. Layout is not a real grid: fall back to auto-labeled ROIs so
            # the pipeline still produces results, but tell the user to use Auto.
            data.add_flag(
                "plate_grid_mapping_failed",
                f"Could not map {len(circles)} detected plate(s) to a {rows}x{cols} "
                f"grid: {fit.error}",
                severity="error",
            )
            rois = self._build_auto_rois(circles, h=h, w=w)
            data.metadata["rois"] = rois
            data.metadata["multi_plate_mode"] = True
            data.metadata["grid_fit_failed"] = True
            return image

        # 3b. Successful grid mapping: one ROI per occupied slot in row-major
        # (grid reading) order, plus a flag for every genuinely-empty slot.
        slot_to_idx: dict[tuple[int, int], int] = {s: i for i, s in fit.assignments.items()}
        rois: list[dict[str, Any]] = []
        for r in range(rows):
            for c in range(cols):
                if (r, c) not in slot_to_idx:
                    continue
                idx = slot_to_idx[(r, c)]
                xc, yc, r_raw, _std, _n = circles[idx]
                rois.append(
                    self._build_roi(xc, yc, r_raw, label=self._get_label(r, c), h=h, w=w, grid_pos=(r, c))
                )

        for r in range(rows):
            for c in range(cols):
                if (r, c) in fit.empty_slots:
                    data.add_flag(
                        "plate_not_found",
                        f"No plate found in slot {self._get_label(r, c)} at grid "
                        f"position ({r}, {c}).",
                        severity="warning",
                    )

        data.metadata["rois"] = rois
        data.metadata["multi_plate_mode"] = True
        return image

    def _build_roi(
        self,
        xc: float,
        yc: float,
        r_raw: float,
        *,
        label: str,
        h: int,
        w: int,
        grid_pos: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Build the standard ``rois`` entry from a detected plate circle.

        Output matches what :class:`~blenny.modules.detect_facile.FacileDetector`
        produces so the downstream ``sub_pipeline`` step works unchanged.
        """
        scale: float = self.params.radius_scale
        r_eff = max(1, round(r_raw * scale))
        r = max(r_raw, r_eff)

        buffer = int(r * 0.05) + 20
        y0_crop, y1_crop = max(0, int(yc - r - buffer)), min(h, int(yc + r + buffer + 1))
        x0_crop, x1_crop = max(0, int(xc - r - buffer)), min(w, int(xc + r + buffer + 1))

        roi: dict[str, Any] = {
            "label": label,
            "bbox": (int(y0_crop), int(x0_crop), int(y1_crop), int(x1_crop)),
            "center_local": (int(yc - y0_crop), int(xc - x0_crop)),
            "radius": int(r),
            "radius_eff": int(r_eff),
        }
        if grid_pos is not None:
            roi["grid_pos"] = grid_pos
        return roi

    def _build_auto_rois(
        self,
        circles: list[tuple[float, float, float, float, int]],
        *,
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        """Build auto-labeled ROIs (labels 1..N in reading order) for the fallback."""
        ordered = sorted(circles, key=lambda c: (c[1], c[0]))
        return [
            self._build_roi(xc, yc, r_raw, label=str(i + 1), h=h, w=w)
            for i, (xc, yc, r_raw, _std, _n) in enumerate(ordered)
        ]

    def _get_label(self, row: int, col: int) -> str:
        if (
            self.params.labels
            and row < len(self.params.labels)
            and col < len(self.params.labels[row])
        ):
            return self.params.labels[row][col]

        # Default: 1, 2, 3...
        cols = self.params.grid[1]
        return str(row * cols + col + 1)
