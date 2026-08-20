"""Pixel-distribution analysis for ROI mode: histograms, clipping, exports.

Pure numpy/matplotlib (no Streamlit), so it is unit-testable and reusable.
The dashboard in the GUI pools per-ROI 256-bin histograms, lets the user clip
each parameter's range with thresholds, and exports the resulting figures and
stats. The *core* ROI data (image, ROIs, per-ROI summary) is never modified
here — this layer only derives and exports analysis.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

#: Parameters exposed in the dashboard, in display order.
PARAMS: list[str] = ["R", "G", "B", "H", "S", "V"]

#: Display names for the dashboard dropdown and figure labels (keys stay short).
PARAM_LABELS: dict[str, str] = {
    "R": "Red",
    "G": "Green",
    "B": "Blue",
    "H": "Hue",
    "S": "Saturation",
    "V": "Value",
}

#: Theoretical value range for each parameter (also the histogram x-range).
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "R": (0.0, 255.0),
    "G": (0.0, 255.0),
    "B": (0.0, 255.0),
    "H": (0.0, 1.0),
    "S": (0.0, 1.0),
    "V": (0.0, 1.0),
}

N_BINS = 256

#: Column index of each parameter in the combined (RGB | HSV) pixel array.
_PARAM_CHANNEL: dict[str, int] = {"R": 0, "G": 1, "B": 2, "H": 3, "S": 4, "V": 5}

#: Fallback colours when a ROI carries no assigned colour (e.g. headless tests).
_DEFAULT_COLORS: list[str] = [
    "#4363d8",
    "#e6194b",
    "#3cb44b",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


def resolve_colors(colors: Sequence[str | None] | None, n: int) -> list[str]:
    """Return ``n`` display colours, substituting ``None`` with palette defaults."""
    if not colors:
        return [_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i in range(n)]
    return [
        c if c else _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)] for i, c in enumerate(colors)
    ]

_EDGES: dict[str, np.ndarray] = {
    p: np.linspace(lo, hi, N_BINS + 1) for p, (lo, hi) in PARAM_RANGES.items()
}
_CENTERS: dict[str, np.ndarray] = {p: (e[:-1] + e[1:]) / 2 for p, e in _EDGES.items()}


def combined_channels(rgb_px: np.ndarray, hsv_px: np.ndarray) -> np.ndarray:
    """Return an ``(N, 6)`` float array of R,G,B,H,S,V for the given pixels."""
    return np.concatenate([np.asarray(rgb_px, dtype=np.float64), np.asarray(hsv_px)], axis=1)


def roi_histograms(rgb_px: np.ndarray, hsv_px: np.ndarray) -> dict[str, np.ndarray]:
    """256-bin counts per parameter for one ROI's pixels (keys ``PARAMS``)."""
    ch = combined_channels(rgb_px, hsv_px)
    out: dict[str, np.ndarray] = {}
    for p in PARAMS:
        lo, hi = PARAM_RANGES[p]
        counts, _ = np.histogram(ch[:, _PARAM_CHANNEL[p]], bins=N_BINS, range=(lo, hi))
        out[p] = counts
    return out


def roi_exact_stats(rgb_px: np.ndarray, hsv_px: np.ndarray) -> dict[str, tuple[int, float, float]]:
    """Exact ``(n, mean, std)`` per parameter for one ROI's pixels."""
    ch = combined_channels(rgb_px, hsv_px)
    out: dict[str, tuple[int, float, float]] = {}
    for p in PARAMS:
        v = ch[:, _PARAM_CHANNEL[p]]
        n = v.shape[0]
        if n == 0:
            out[p] = (0, 0.0, 0.0)
        else:
            out[p] = (n, float(v.mean()), float(v.std()))
    return out


def pooled_counts(
    hists: dict[int, dict[str, np.ndarray]], roi_ids: list[int], param: str
) -> np.ndarray:
    """Sum the per-ROI histograms for ``param`` across the selected ROIs."""
    total = np.zeros(N_BINS, dtype=np.int64)
    for rid in roi_ids:
        if rid in hists:
            total += np.asarray(hists[rid][param])
    return total


def pooled_exact_stats(
    exact: dict[int, dict[str, tuple[int, float, float]]],
    roi_ids: list[int],
    param: str,
) -> tuple[int, float, float]:
    """Exact pooled ``(n, mean, std)`` via parallel (group) combination."""
    n, m, m2 = 0, 0.0, 0.0  # m2 = sum of squared deviations from the pooled mean
    for rid in roi_ids:
        if rid not in exact:
            continue
        ni, mi, si = exact[rid][param]
        if ni == 0:
            continue
        n_new = n + ni
        delta = mi - m
        m = m + delta * ni / n_new
        m2 = m2 + si * si * ni + delta * delta * n * ni / n_new
        n = n_new
    if n == 0:
        return 0, 0.0, 0.0
    return n, m, math.sqrt(m2 / n)


def stats_from_hist(
    counts: np.ndarray, param: str, lo: float, hi: float
) -> tuple[int, float, float]:
    """Approximate ``(n, mean, std)`` of the values clipped to ``[lo, hi]``.

    Computed from the 256-bin counts using bin centres, which is accurate to
    within half a bin (≈0.5 for RGB, ≈0.002 for HSV).
    """
    mask = (_CENTERS[param] >= lo) & (_CENTERS[param] <= hi)
    n = int(np.asarray(counts)[mask].sum())
    if n == 0:
        return 0, 0.0, 0.0
    centers = _CENTERS[param][mask]
    w = np.asarray(counts)[mask].astype(np.float64)
    mean = float((w * centers).sum() / n)
    var = float((w * (centers - mean) ** 2).sum() / n)
    return n, mean, math.sqrt(var)


def count_in_range(counts: np.ndarray, param: str, lo: float, hi: float) -> int:
    """Number of pixels whose value lies in ``[lo, hi]`` (from bin counts)."""
    mask = (_CENTERS[param] >= lo) & (_CENTERS[param] <= hi)
    return int(np.asarray(counts)[mask].sum())


def build_histogram_figure(
    param: str,
    roi_histograms: Sequence[np.ndarray],
    roi_stats: Sequence[tuple[int, float, float]],
    roi_names: Sequence[str],
    roi_colors: Sequence[str | None],
    lo: float,
    hi: float,
    combined: tuple[int, float, float],
    clipped_n: int,
    clipped_mean: float,
    clipped_std: float,
    normalize: bool = False,
) -> Any:
    """Render per-ROI overlaid histograms for one parameter.

    Each ROI is drawn as a semi-transparent bar histogram in its own colour so
    overlapping regions remain visible; legend labels carry each ROI's name and
    ``mean ± std`` for the parameter. A solid line marks the combined
    (pixel-weighted pooled) mean with a ±1 standard-deviation band, and dotted
    lines mark the current threshold range.

    ``roi_stats`` entries are ``(n, mean, std)`` per ROI (see
    :func:`roi_exact_stats`); ``combined`` is the pooled ``(n, mean, std)``
    across the same ROIs (see :func:`pooled_exact_stats`). With
    ``normalize=True`` each ROI's bar heights are scaled to sum to 1 so shapes
    are comparable regardless of ROI area; the y-axis becomes a pixel
    fraction. The legend mean ± std values are unaffected (they come from the
    exact stats).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = _EDGES[param]
    width = float(edges[1] - edges[0])
    centers = _CENTERS[param]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)

    colors = resolve_colors(roi_colors, len(roi_histograms))
    for counts, (_n, mean, std), name, color in zip(
        roi_histograms, roi_stats, roi_names, colors, strict=True
    ):
        display = counts
        if normalize:
            total = int(np.asarray(counts).sum())
            display = np.asarray(counts, dtype=np.float64) / total if total > 0 else counts
        ax.bar(
            centers,
            display,
            width=width,
            color=color,
            alpha=0.55,
            edgecolor="none",
            label=f"{name}: {mean:.3f} ± {std:.3f}",
        )

    n_all, mean_all, std_all = combined
    if n_all > 0:
        if std_all > 0:
            ax.axvspan(
                mean_all - std_all, mean_all + std_all, color="#555555", alpha=0.10, zorder=0
            )
        ax.axvline(
            mean_all,
            color="#333333",
            ls="-",
            lw=1.6,
            label=f"All ROIs: {mean_all:.3f} ± {std_all:.3f}",
        )

    xmin, xmax = PARAM_RANGES[param]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel("fraction of pixels" if normalize else "pixel count")

    if clipped_n and clipped_n != n_all:
        ax.set_title(
            f"{PARAM_LABELS.get(param, param)}: {clipped_n:,}/{n_all:,} px kept "
            f"({100.0 * clipped_n / max(int(n_all), 1):.1f}%) — "
            f"clipped mean {clipped_mean:.3f} ± {clipped_std:.3f}"
        )
    ax.axvline(lo, color="#e6194b", ls=":", lw=1.5)
    ax.axvline(hi, color="#e6194b", ls=":", lw=1.5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


def write_analysis_outputs(
    output_dir: str | Path,
    stem: str,
    hist_data: dict[str, Any],
    roi_ids: list[int],
    roi_names: list[str],
    thresholds: dict[str, tuple[float, float]],
    roi_colors: Sequence[str | None] | None = None,
    normalize: bool = False,
) -> dict[str, Path]:
    """Write one histogram figure per parameter + a per-parameter stats CSV.

    ``hist_data`` is the dict produced by :func:`analyze_rois` (keys
    ``hists`` / ``exact``). ``roi_colors`` (optional) gives each ROI its
    display colour in the exported figures; ``None`` entries fall back to a
    default palette. ``normalize`` scales each ROI's bars to sum to 1 (see
    :func:`build_histogram_figure`). Returns a mapping of kind → path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    hists = hist_data["hists"]
    exact = hist_data["exact"]
    # Align ids/names/colours with the ROIs that actually have data.
    pairs = [
        (rid, name, color)
        for rid, name, color in zip(
            roi_ids, roi_names, roi_colors if roi_colors else [None] * len(roi_ids), strict=True
        )
        if rid in hists
    ]
    pair_ids = [p[0] for p in pairs]

    csv_path = output_dir / f"{stem}_roi_analysis.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "parameter",
                "rois",
                "n_total",
                "mean",
                "std",
                "threshold_lo",
                "threshold_hi",
                "n_kept",
                "mean_kept",
                "std_kept",
                "pct_kept",
            ]
        )
        for p in PARAMS:
            per_roi_hists = [hists[rid][p] for rid, _, _ in pairs]
            per_roi_stats = [exact[rid][p] for rid, _, _ in pairs]
            per_roi_names = [name for _, name, _ in pairs]
            per_roi_colors = [color for _, _, color in pairs]
            combined = pooled_exact_stats(exact, pair_ids, p)
            counts = pooled_counts(hists, pair_ids, p)
            n_total, mean, std = combined
            lo, hi = thresholds.get(p, PARAM_RANGES[p])
            n_kept, mean_kept, std_kept = stats_from_hist(counts, p, lo, hi)
            pct = 100.0 * n_kept / n_total if n_total else 0.0

            fig = build_histogram_figure(
                p,
                per_roi_hists,
                per_roi_stats,
                per_roi_names,
                per_roi_colors,
                lo,
                hi,
                combined,
                n_kept,
                mean_kept,
                std_kept,
                normalize=normalize,
            )
            fig_path = output_dir / f"{stem}_histogram_{p}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
            paths[p] = fig_path

            writer.writerow(
                [
                    p,
                    "|".join(per_roi_names),
                    n_total,
                    round(mean, 4),
                    round(std, 4),
                    round(lo, 4),
                    round(hi, 4),
                    n_kept,
                    round(mean_kept, 4),
                    round(std_kept, 4),
                    round(pct, 2),
                ]
            )
    paths["csv"] = csv_path
    return paths
