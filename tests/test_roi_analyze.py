"""Tests for ROI mode pixel-distribution analysis (histograms, clipping, exports)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blenny.roi import analyze_rois
from blenny.roi.analyze import (
    PARAM_RANGES,
    PARAMS,
    build_histogram_figure,
    count_in_range,
    pooled_counts,
    pooled_exact_stats,
    roi_exact_stats,
    roi_histograms,
    stats_from_hist,
    write_analysis_outputs,
)


def _pixels(n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """RGB pixels with a known distribution + matching HSV."""
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
    hsv = np.stack([rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n)], axis=1)
    return rgb, hsv


def test_roi_histograms_counts_match_pixels() -> None:
    rgb, hsv = _pixels()
    hists = roi_histograms(rgb, hsv)
    assert set(hists) == set(PARAMS)
    for p in PARAMS:
        counts = hists[p]
        assert counts.shape == (256,)
        assert counts.sum() == rgb.shape[0]
        assert counts.dtype.kind in "iu"


def test_roi_exact_stats_matches_numpy() -> None:
    rgb, hsv = _pixels(500)
    stats = roi_exact_stats(rgb, hsv)
    n, mean, std = stats["R"]
    assert n == 500
    assert mean == pytest.approx(rgb[:, 0].mean())
    assert std == pytest.approx(rgb[:, 0].std())
    # HSV stats come from the HSV array
    _, s_mean, _ = stats["S"]
    assert s_mean == pytest.approx(hsv[:, 1].mean())


def test_stats_from_hist_unclipped_matches_exact() -> None:
    """With thresholds at the full range, bin-centred stats ≈ exact stats."""
    rgb, hsv = _pixels(2000)
    hists = roi_histograms(rgb, hsv)
    exact = roi_exact_stats(rgb, hsv)
    for p in PARAMS:
        lo, hi = PARAM_RANGES[p]
        n, mean, std = stats_from_hist(hists[p], p, lo, hi)
        assert n == exact[p][0]
        # Bin-centre quantization: RGB bins are 1 unit wide, HSV ~0.004.
        # Half a bin bounds the error, but observed error is far smaller.
        assert mean == pytest.approx(exact[p][1], abs=0.6)
        assert std == pytest.approx(exact[p][2], abs=0.6)


def test_stats_from_hist_clipping() -> None:
    """Clipping to a narrow band excludes pixels and shifts the mean."""
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, size=(5000, 3)).astype(np.uint8)
    hsv = np.zeros((5000, 3), dtype=np.float64)
    counts = roi_histograms(rgb, hsv)["R"]

    n_all, mean_all, _ = stats_from_hist(counts, "R", 0, 255)
    n_lo, mean_lo, _ = stats_from_hist(counts, "R", 0, 100)

    assert n_all == 5000
    assert n_lo < n_all  # high-value pixels excluded
    # mean of the clipped-left distribution <= overall mean
    assert mean_lo <= mean_all + 1e-9
    # thresholds are inclusive
    n_edge, _, _ = stats_from_hist(counts, "R", 100, 100)
    assert n_edge == count_in_range(counts, "R", 100, 100)


def test_pooled_exact_stats_matches_direct() -> None:
    rgb1, hsv1 = _pixels(300)
    rgb2, hsv2 = _pixels(700)
    stats1 = roi_exact_stats(rgb1, hsv1)
    stats2 = roi_exact_stats(rgb2, hsv2)

    n, mean, std = pooled_exact_stats({1: stats1, 2: stats2}, [1, 2], "G")
    all_g = np.concatenate([rgb1[:, 1], rgb2[:, 1]])
    assert n == 1000
    assert mean == pytest.approx(all_g.mean())
    assert std == pytest.approx(all_g.std())

    # subset selection
    n2, mean2, _ = pooled_exact_stats({1: stats1, 2: stats2}, [2], "G")
    assert n2 == 700
    assert mean2 == pytest.approx(rgb2[:, 1].mean())


def test_pooled_counts_sums_selected_rois() -> None:
    rgb1, hsv1 = _pixels(100)
    rgb2, hsv2 = _pixels(200)
    hists = {1: roi_histograms(rgb1, hsv1), 2: roi_histograms(rgb2, hsv2)}
    both = pooled_counts(hists, [1, 2], "V")
    only2 = pooled_counts(hists, [2], "V")
    assert both.sum() == 300
    assert only2.sum() == 200
    # pooling is exact: counts add up bin-wise
    assert (both == hists[1]["V"] + hists[2]["V"]).all()


def test_build_histogram_figure_returns_figure() -> None:
    rgb, hsv = _pixels(100)
    counts = roi_histograms(rgb, hsv)["R"]
    fig = build_histogram_figure(
        "R",
        [counts],
        [(100, 128.0, 40.0)],
        ["r1"],
        ["#e6194b"],
        0.0,
        255.0,
        (100, 128.0, 40.0),
        800,
        110.0,
        30.0,
    )
    assert fig is not None
    assert fig.axes
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_histogram_figure_overlays_each_roi() -> None:
    """Every ROI gets its own legend entry (name + mean ± std) plus a combined one."""
    rgb1, hsv1 = _pixels(100)
    rgb2, hsv2 = _pixels(200)
    c1 = roi_histograms(rgb1, hsv1)["R"]
    c2 = roi_histograms(rgb2, hsv2)["R"]
    fig = build_histogram_figure(
        "R",
        [c1, c2],
        [(100, 110.0, 20.0), (200, 130.0, 25.0)],
        ["control", "treated"],
        ["#e6194b", "#3cb44b"],
        0.0,
        255.0,
        (300, 123.0, 24.0),
        300,
        123.0,
        24.0,
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "control: 110.000 ± 20.000" in labels
    assert "treated: 130.000 ± 25.000" in labels
    assert any("All ROIs" in lbl for lbl in labels)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_histogram_figure_normalize() -> None:
    """Normalization scales each ROI's bars to sum to 1 (shape comparison)."""
    rgb1, hsv1 = _pixels(100)
    rgb2, hsv2 = _pixels(500)
    c1 = roi_histograms(rgb1, hsv1)["G"]
    c2 = roi_histograms(rgb2, hsv2)["G"]

    def bar_sums(fig) -> list[float]:
        ax = fig.axes[0]
        # One BarContainer per ROI (the axvspan band is not in containers).
        return [sum(p.get_height() for p in c.patches) for c in ax.containers]

    # Raw counts: each ROI sums to its pixel count.
    fig_raw = build_histogram_figure(
        "G",
        [c1, c2],
        [(100, 0.5, 0.1), (500, 0.6, 0.2)],
        ["a", "b"],
        ["#e6194b", "#3cb44b"],
        0.0,
        1.0,
        (600, 0.58, 0.19),
        600,
        0.58,
        0.19,
    )
    assert bar_sums(fig_raw) == pytest.approx([100.0, 500.0])
    assert fig_raw.axes[0].get_ylabel() == "pixel count"

    # Normalized: each ROI sums to 1, y-axis becomes a fraction.
    fig_norm = build_histogram_figure(
        "G",
        [c1, c2],
        [(100, 0.5, 0.1), (500, 0.6, 0.2)],
        ["a", "b"],
        ["#e6194b", "#3cb44b"],
        0.0,
        1.0,
        (600, 0.58, 0.19),
        600,
        0.58,
        0.19,
        normalize=True,
    )
    assert bar_sums(fig_norm) == pytest.approx([1.0, 1.0])
    assert fig_norm.axes[0].get_ylabel() == "fraction of pixels"

    import matplotlib.pyplot as plt

    plt.close(fig_raw)
    plt.close(fig_norm)


def test_analyze_rois_returns_hist_data(tmp_path: Path) -> None:
    from PIL import Image

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [200, 30, 20]
    path = tmp_path / "p.png"
    Image.fromarray(img).save(path)

    rois = [
        {"id": 1, "name": "r1", "color": "#e6194b", "points": [[8, 8], [23, 8], [23, 23], [8, 23]]},
        {"id": 2, "name": "r2", "color": "#3cb44b", "points": [[0, 0], [7, 0], [7, 7], [0, 7]]},
    ]
    rows, hist_data = analyze_rois(path, rois, scale=(1.0, 1.0))
    assert len(rows) == 2
    assert set(hist_data) == {"hists", "exact"}
    assert set(hist_data["hists"]) == {1, 2}
    assert hist_data["hists"][1]["R"].sum() == 16 * 16  # red block
    assert hist_data["exact"][1]["R"][1] == pytest.approx(200.0)
    # ROI 2 is black (0,0,0): red mean == 0
    assert hist_data["exact"][2]["R"][1] == pytest.approx(0.0)


def test_write_analysis_outputs_writes_figures_and_csv(tmp_path: Path) -> None:
    from PIL import Image

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[8:24, 8:24] = [120, 60, 200]
    path = tmp_path / "p.png"
    Image.fromarray(img).save(path)
    rois = [
        {"id": 1, "name": "r1", "color": "#e6194b", "points": [[8, 8], [23, 8], [23, 23], [8, 23]]},
    ]
    _, hist_data = analyze_rois(path, rois, scale=(1.0, 1.0))

    out = tmp_path / "out"
    thresholds = {"R": (50.0, 200.0)}  # only R clipped; others full range
    paths = write_analysis_outputs(out, "plate", hist_data, [1], ["r1"], thresholds)

    for p in PARAMS:
        assert paths[p].exists(), f"{p} histogram missing"
    csv_path = paths["csv"]
    assert csv_path.exists()
    text = csv_path.read_text()
    assert text.splitlines()[0] == (
        "parameter,rois,n_total,mean,std,threshold_lo,threshold_hi,"
        "n_kept,mean_kept,std_kept,pct_kept"
    )
    # R row reflects the clip
    r_line = next(line for line in text.splitlines()[1:] if line.startswith("R,"))
    assert "50.0" in r_line and "200.0" in r_line
    # every parameter has a row
    assert len(text.splitlines()) == 1 + len(PARAMS)
