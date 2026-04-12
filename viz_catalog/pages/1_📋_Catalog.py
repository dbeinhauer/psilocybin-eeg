"""Analysis Catalog page — browse every analysis type with sketches."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Analysis Catalog", page_icon="📋", layout="wide")

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------
_CATALOG_PATH = Path(__file__).parent.parent / "catalog.yaml"


@st.cache_data
def load_catalog() -> dict:
    with open(_CATALOG_PATH) as f:
        return yaml.safe_load(f)


catalog = load_catalog()
analyses: list[dict] = catalog.get("analyses", [])

GITHUB_BASE = "https://github.com/dbeinhauer/psilocybin-eeg/blob/develop"

# ---------------------------------------------------------------------------
# Sketch functions
# ---------------------------------------------------------------------------
# Each function takes no arguments and returns a matplotlib Figure.
# Keep figures small and schematic — they are orientation aids, not data plots.
# ---------------------------------------------------------------------------


def sketch_timeseries_multichannel() -> Figure:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    n_ch, n_t = 8, 200
    t = np.linspace(0, 4, n_t)
    for i in range(n_ch):
        signal = rng.standard_normal(n_t) * 0.4
        signal[80:100] += 3 - i * 0.3  # artefact burst
        ax.plot(t, signal + i * 1.2, lw=0.7, color="steelblue")
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Channels", fontsize=7)
    ax.set_title("Multichannel time series", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def sketch_psd() -> Figure:
    fig, ax = plt.subplots(figsize=(5, 2.5))
    freqs = np.linspace(1, 100, 300)
    raw_psd = 10 / freqs + 0.05 * np.random.default_rng(1).standard_normal(300)
    filt_psd = raw_psd.copy()
    filt_psd[(freqs > 48) & (freqs < 52)] *= 0.05  # notch
    filt_psd[freqs < 1] = 0
    ax.semilogy(freqs, raw_psd, label="Raw", color="tomato", lw=1)
    ax.semilogy(freqs, filt_psd, label="Filtered", color="steelblue", lw=1)
    ax.axvline(50, color="gray", ls="--", lw=0.8, label="50 Hz notch")
    ax.set_xlabel("Frequency (Hz)", fontsize=7)
    ax.set_ylabel("Power", fontsize=7)
    ax.set_title("PSD before vs after filtering", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_topomap() -> Figure:
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(3, 3))
    theta_vals = np.linspace(0, 2 * np.pi, 60)
    xs = np.concatenate(
        [
            np.cos(theta_vals),
            0.7 * np.cos(theta_vals),
            0.4 * np.cos(theta_vals),
            [0],
        ]
    )
    ys = np.concatenate(
        [
            np.sin(theta_vals),
            0.7 * np.sin(theta_vals),
            0.4 * np.sin(theta_vals),
            [0],
        ]
    )
    power = rng.uniform(0, 1, len(xs))
    ax.scatter(xs, ys, c=power, cmap="RdYlBu_r", s=14, vmin=0, vmax=1)
    circle = plt.Circle((0, 0), 1.0, fill=False, color="gray", lw=1)
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    ax.set_title("Topomap — mean power", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_timeseries_offset() -> Figure:
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    t = np.linspace(0, 6, 300)
    offsets = [0, 0.8, 1.7]
    colors = ["steelblue", "darkorange", "seagreen"]
    labels = ["P1", "P2", "P3"]
    for i, (off, col, lab) in enumerate(zip(offsets, colors, labels)):
        y = np.zeros(300)
        start = int(off / 6 * 300)
        y[start : start + 20] = 1.0  # marker pulse
        y += rng.standard_normal(300) * 0.05 + i * 0.2
        ax.plot(t, y, lw=0.8, color=col, label=lab)
        ax.axvline(off, color=col, ls="--", lw=0.7, alpha=0.6)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_title("TAG channel overlay (offsets)", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_matrix_heatmap() -> Figure:
    rng = np.random.default_rng(4)
    n = 6
    mat = rng.uniform(-0.3, 0.8, (n, n))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    fig, ax = plt.subplots(figsize=(3, 2.8))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.3, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"P{i + 1}" for i in range(n)], fontsize=5)
    ax.set_yticklabels([f"P{i + 1}" for i in range(n)], fontsize=5)
    ax.set_title("Pairwise lag / ISC matrix", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_timeseries_variance() -> Figure:
    rng = np.random.default_rng(5)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    t = np.linspace(0, 10, 500)
    var_trace = np.abs(rng.standard_normal(500)) * 0.5 + 0.5
    var_trace[100:150] *= 0.3
    var_trace[320:370] *= 0.25
    threshold = np.percentile(var_trace, 10)
    ax.plot(t, var_trace, lw=0.8, color="steelblue")
    ax.axhline(threshold, color="tomato", ls="--", lw=0.8, label="10th pct")
    low = var_trace < threshold
    ax.fill_between(
        t, 0, var_trace.max(), where=low, alpha=0.2, color="gold", label="Synchrony"
    )
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("Variance", fontsize=7)
    ax.set_title("Inter-subject variance trace", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_histogram() -> Figure:
    rng = np.random.default_rng(6)
    fig, ax = plt.subplots(figsize=(4, 2.5))
    data = rng.gamma(2, 0.3, 400)
    clip99 = np.percentile(data, 99)
    data_clipped = data[data <= clip99]
    ax.hist(data_clipped, bins=30, color="darkorange", edgecolor="white", lw=0.4)
    median = np.median(data_clipped)
    ax.axvline(median, color="steelblue", ls="-", lw=1.2, label="median")
    ax.axvline(clip99, color="tomato", ls="--", lw=1, label="99th pct")
    ax.set_xlabel("Intersubject variance", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_title("Variance distribution (channel × time)", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_timeseries_multisubject() -> Figure:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    t = np.linspace(0, 8, 400)
    n_subj = 6
    for i in range(n_subj):
        y = rng.standard_normal(400) * 0.3
        y[180:220] += 0.6  # converging epoch
        ax.plot(t, y + i * 0.05, lw=0.7, alpha=0.7)
    ax.axvspan(t[180], t[220], alpha=0.15, color="gold", label="Synchrony")
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_title("Per-subject channel-average traces", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_timeseries_multiband() -> Figure:
    rng = np.random.default_rng(8)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    t = np.linspace(0, 8, 300)
    fig, axes = plt.subplots(5, 2, figsize=(5.5, 3.5), sharex=True)
    for row, (band, col) in enumerate(zip(bands, colors)):
        # Left: mean signal
        n_subj = 4
        mean_y = rng.standard_normal(300) * 0.3
        ax_l = axes[row, 0]
        for _ in range(n_subj):
            ax_l.plot(
                t, mean_y + rng.standard_normal(300) * 0.1, lw=0.5, alpha=0.4, color=col
            )
        ax_l.plot(t, mean_y, lw=0.9, color=col)
        ax_l.set_ylabel(band, fontsize=6, rotation=0, labelpad=28)
        ax_l.tick_params(labelsize=4)
        ax_l.set_yticks([])
        # Right: variance
        var_y = np.abs(rng.standard_normal(300)) * 0.4 + 0.4
        threshold = np.percentile(var_y, 10)
        ax_r = axes[row, 1]
        ax_r.plot(t, var_y, lw=0.8, color=col)
        ax_r.axhline(threshold, color="gray", ls="--", lw=0.6)
        ax_r.tick_params(labelsize=4)
        ax_r.set_yticks([])
    axes[-1, 0].set_xlabel("Time (s)", fontsize=6)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=6)
    axes[0, 0].set_title("Mean signal", fontsize=7)
    axes[0, 1].set_title("Variance", fontsize=7)
    fig.tight_layout()
    return fig


def sketch_histogram_overlay() -> Figure:
    rng = np.random.default_rng(9)
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    d1 = rng.normal(0.15, 0.12, 300)
    d2 = rng.normal(0.22, 0.11, 300)
    bins = np.linspace(-0.2, 0.6, 30)
    ax.hist(
        d1, bins=bins, alpha=0.6, label="CLASSIC", color="steelblue", edgecolor="white"
    )
    ax.hist(
        d2,
        bins=bins,
        alpha=0.6,
        label="PSYTRANCE",
        color="darkorange",
        edgecolor="white",
    )
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("LOO-ISC (r)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.set_title("LOO-ISC distribution overlay", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_timeseries_heatmap() -> Figure:
    rng = np.random.default_rng(10)
    n_win, n_ch = 60, 40
    heatmap = rng.uniform(-0.2, 0.6, (n_ch, n_win))
    trace = heatmap.mean(axis=0)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5, 3), gridspec_kw={"height_ratios": [1, 2]}
    )
    ax_top.plot(trace, lw=0.9, color="steelblue")
    threshold = np.percentile(trace, 75)
    ax_top.axhline(threshold, color="gold", ls="--", lw=0.8, label="thresh")
    ax_top.fill_between(
        range(n_win), 0, trace.max(), where=trace > threshold, alpha=0.25, color="gold"
    )
    ax_top.set_ylabel("Mean ISC", fontsize=6)
    ax_top.tick_params(labelsize=5)
    ax_top.legend(fontsize=5)
    im = ax_bot.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", vmin=-0.2, vmax=0.6)
    ax_bot.set_ylabel("Channel", fontsize=6)
    ax_bot.set_xlabel("Window", fontsize=6)
    ax_bot.tick_params(labelsize=5)
    plt.colorbar(im, ax=ax_bot, fraction=0.03, pad=0.02)
    ax_top.set_title("Sliding-window ISC", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_multiscale_sw_isc() -> Figure:
    """Three-figure composite for multi-scale sliding-window LOO-ISC.

    Figure 1 — bar chart of mean Pearson ISC per medium window, colour-coded:
      green = above threshold, blue = positive below threshold, red = negative.
      Error bars show per-window variance; dashed line shows grand mean.
    Figure 2 — stair-wise multi-scale overlay (light blue = fine, dark blue =
      medium, dashed red = coarse) above a fine-resolution per-channel heatmap.
    Figure 3 — stepped Pearson vs Spearman comparison at medium resolution;
      area below zero shaded red.
    """
    rng = np.random.default_rng(24)
    n_med = 30  # medium-window count
    n_fine = 90  # fine-window count (3× denser)
    n_coarse = 10  # coarse-window count
    n_ch = 40

    # --- synthetic data ---
    med_isc_p = rng.normal(0.12, 0.09, n_med)
    med_isc_s = rng.normal(0.10, 0.10, n_med)
    med_var = np.abs(rng.normal(0.04, 0.02, n_med))
    grand_mean = med_isc_p.mean()
    threshold = 0.18

    fine_isc = rng.normal(0.11, 0.08, n_fine)
    coarse_isc = rng.normal(0.13, 0.07, n_coarse)
    heatmap = rng.uniform(-0.15, 0.45, (n_ch, n_fine))

    fig = plt.figure(figsize=(6, 8))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 1.0, 1.5, 1.0], hspace=0.55)

    # ── Figure 1: colour-coded bar chart ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title("Fig 1 — Mean Pearson LOO-ISC per medium window", fontsize=7, pad=3)
    x = np.arange(n_med)
    bar_colors = [
        "seagreen" if v >= threshold else ("tomato" if v < 0 else "steelblue")
        for v in med_isc_p
    ]
    ax1.bar(x, med_isc_p, color=bar_colors, width=0.8, zorder=2)
    ax1.errorbar(
        x,
        med_isc_p,
        yerr=med_var,
        fmt="none",
        ecolor="black",
        elinewidth=0.6,
        capsize=2,
        zorder=3,
    )
    ax1.axhline(
        grand_mean,
        color="black",
        ls="--",
        lw=0.9,
        label=f"grand mean ({grand_mean:.2f})",
    )
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axhline(threshold, color="seagreen", ls=":", lw=0.8, label="threshold")
    # legend proxies for bar colours
    from matplotlib.patches import Patch

    ax1.legend(
        handles=[
            Patch(facecolor="seagreen", label="above threshold"),
            Patch(facecolor="steelblue", label="positive"),
            Patch(facecolor="tomato", label="negative"),
        ],
        fontsize=5,
        loc="upper right",
        ncol=3,
    )
    ax1.set_ylabel("Mean ISC (Pearson)", fontsize=6)
    ax1.set_xlabel("Medium window index", fontsize=6)
    ax1.tick_params(labelsize=5)

    # ── Figure 2: multi-scale stair-wise overlay ──────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title(
        "Fig 2 — Multi-scale overlay + fine-resolution heatmap", fontsize=7, pad=3
    )

    fine_x = np.linspace(0, n_med, n_fine)
    med_x = np.linspace(0, n_med, n_med)
    coarse_x = np.linspace(0, n_med, n_coarse)

    ax2.step(
        fine_x,
        fine_isc,
        where="post",
        color="lightsteelblue",
        lw=0.8,
        label="fine window",
        alpha=0.9,
    )
    ax2.step(
        med_x, med_isc_p, where="post", color="steelblue", lw=1.3, label="medium window"
    )
    ax2.step(
        coarse_x,
        coarse_isc,
        where="post",
        color="tomato",
        lw=1.1,
        ls="--",
        label="coarse window",
    )
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("LOO-ISC", fontsize=6)
    ax2.legend(fontsize=5, loc="upper right")
    ax2.tick_params(labelsize=5)
    ax2.set_xticklabels([])

    # ── Figure 2 continued: fine-resolution heatmap ───────────────────────────
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(heatmap, aspect="auto", cmap="RdYlBu_r", vmin=-0.2, vmax=0.5)
    ax3.set_ylabel("Channel", fontsize=6)
    ax3.set_xlabel("Fine window index", fontsize=6)
    ax3.tick_params(labelsize=5)
    plt.colorbar(im, ax=ax3, fraction=0.025, pad=0.02)

    # ── Figure 3: Pearson vs Spearman stepped comparison ─────────────────────
    ax4 = fig.add_subplot(gs[3])
    ax4.set_title("Fig 3 — Pearson vs Spearman at medium resolution", fontsize=7, pad=3)
    ax4.step(med_x, med_isc_p, where="post", color="steelblue", lw=1.2, label="Pearson")
    ax4.step(
        med_x,
        med_isc_s,
        where="post",
        color="darkorange",
        lw=1.2,
        ls="--",
        label="Spearman",
    )
    # Red-shade the area below zero
    ax4.fill_between(
        med_x,
        np.minimum(med_isc_p, 0),
        0,
        step="post",
        color="tomato",
        alpha=0.35,
        label="below zero",
    )
    ax4.fill_between(
        med_x,
        np.minimum(med_isc_s, 0),
        0,
        step="post",
        color="tomato",
        alpha=0.20,
    )
    ax4.axhline(0, color="gray", lw=0.6)
    ax4.set_ylabel("LOO-ISC", fontsize=6)
    ax4.set_xlabel("Medium window index", fontsize=6)
    ax4.legend(fontsize=5, loc="upper right")
    ax4.tick_params(labelsize=5)

    return fig


def sketch_bar_grouped() -> Figure:
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    n_subj = 7
    x = np.arange(n_subj)
    w = 0.35
    pearson = rng.normal(0.2, 0.08, n_subj)
    spearman = rng.normal(0.18, 0.09, n_subj)
    ax.bar(x - w / 2, pearson, w, label="Pearson", color="steelblue")
    ax.bar(x + w / 2, spearman, w, label="Spearman", color="darkorange")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i + 1}" for i in range(n_subj)], fontsize=6)
    ax.set_ylabel("LOO-ISC", fontsize=7)
    ax.set_title("Mean-field LOO-ISC per subject", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_histogram_facet() -> Figure:
    rng = np.random.default_rng(12)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    fig, axes = plt.subplots(1, 5, figsize=(6, 2.2), sharey=True)
    for ax, band, col in zip(axes, bands, colors):
        d = rng.gamma(2, 0.3, 200)
        clip99 = np.percentile(d, 99)
        d_clipped = d[d <= clip99]
        ax.hist(d_clipped, bins=15, color=col, edgecolor="white", lw=0.3)
        median = np.median(d_clipped)
        ax.axvline(median, color="black", ls="-", lw=0.8)
        ax.axvline(clip99, color="gray", ls="--", lw=0.6)
        ax.set_title(band, fontsize=6)
        ax.tick_params(labelsize=4)
        ax.set_xlabel("variance", fontsize=5)
    axes[0].set_ylabel("Count", fontsize=6)
    fig.suptitle("Per-band variance distributions", fontsize=8, y=1.02)
    fig.tight_layout()
    return fig


def sketch_bar_grouped_bands() -> Figure:
    rng = np.random.default_rng(13)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    x = np.arange(len(bands))
    w = 0.35
    classic = rng.uniform(0.05, 0.3, len(bands))
    psytrance = rng.uniform(0.08, 0.35, len(bands))
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.bar(x - w / 2, classic, w, label="CLASSIC", color="steelblue")
    ax.bar(x + w / 2, psytrance, w, label="PSYTRANCE", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=7)
    ax.set_ylabel("Mean ISC ± SD", fontsize=7)
    ax.set_title("Band mean ISC comparison", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_grid_timeseries_heatmap() -> Figure:
    rng = np.random.default_rng(14)
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.2))
    titles = [
        ("alpha Classic", "steelblue"),
        ("alpha Psytrance", "darkorange"),
        ("beta Classic", "seagreen"),
        ("beta Psytrance", "tomato"),
    ]
    n_w, n_ch = 40, 20
    for ax, (title, col) in zip(axes.flat, titles):
        hm = rng.uniform(0, 0.5, (n_ch, n_w))
        ax.imshow(hm, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=0.5)
        ax.set_title(title, fontsize=7, color=col)
        ax.tick_params(labelsize=4)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Band sliding-window ISC grid", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_raster_overlap() -> Figure:
    rng = np.random.default_rng(15)
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    colors = ["steelblue", "darkorange", "seagreen", "tomato", "mediumpurple"]
    n_w = 60
    masks = (rng.uniform(0, 1, (5, n_w)) > 0.6).astype(float)
    fig, (ax_raster, ax_count) = plt.subplots(
        2, 1, figsize=(5.5, 3), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    for i, (band, col) in enumerate(zip(bands, colors)):
        for j in range(n_w):
            if masks[i, j]:
                ax_raster.barh(i, 1, left=j, height=0.7, color=col, edgecolor="none")
    ax_raster.set_yticks(range(5))
    ax_raster.set_yticklabels(bands, fontsize=7)
    ax_raster.set_title("Band overlap raster", fontsize=8)
    ax_raster.tick_params(labelsize=5)
    count = masks.sum(axis=0)
    ax_count.bar(range(n_w), count, color="slategray", width=1)
    ax_count.set_ylabel("# bands", fontsize=6)
    ax_count.set_xlabel("Window", fontsize=6)
    ax_count.tick_params(labelsize=5)
    # Mark all-5 windows
    for j in np.where(count == 5)[0]:
        ax_count.annotate("★", (j, 5.1), fontsize=6, ha="center", color="gold")
    fig.tight_layout()
    return fig


def sketch_polar_histogram() -> Figure:
    rng = np.random.default_rng(16)
    # Simulate approximately uniform phase with a slight bias
    phases = rng.uniform(-np.pi, np.pi, 2000)
    fig, (ax_lin, ax_pol) = plt.subplots(1, 2, figsize=(5, 2.5))
    # Linear histogram
    ax_lin.hist(phases, bins=30, color="steelblue", edgecolor="white", lw=0.3)
    ax_lin.set_xlabel("Phase (rad)", fontsize=6)
    ax_lin.set_ylabel("Count", fontsize=6)
    ax_lin.set_xticks([-np.pi, 0, np.pi])
    ax_lin.set_xticklabels(["-π", "0", "π"], fontsize=6)
    ax_lin.tick_params(labelsize=5)
    ax_lin.set_title("Linear histogram", fontsize=7)
    # Polar histogram
    ax_pol.remove()
    ax_pol = fig.add_subplot(1, 2, 2, projection="polar")
    n_bins = 24
    counts, bin_edges = np.histogram(phases, bins=n_bins, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = 2 * np.pi / n_bins
    ax_pol.bar(
        bin_centers,
        counts,
        width=width,
        bottom=0,
        color="steelblue",
        edgecolor="white",
        lw=0.3,
    )
    ax_pol.set_yticks([])
    ax_pol.tick_params(labelsize=5)
    ax_pol.set_title("Polar", fontsize=7)
    fig.suptitle("Phase distribution (uniform = healthy)", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_zscore_transform() -> Figure:
    """Before/after Z-score illustration."""
    rng = np.random.default_rng(17)
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    t = np.linspace(0, 4, 200)
    amps = [0.5, 1.2, 2.0, 0.8]
    offsets = [0.0, 1.0, -0.5, 1.5]
    colors = ["steelblue", "darkorange", "seagreen", "tomato"]
    # Left: raw (different amplitudes)
    ax_raw = axes[0]
    for amp, off, col in zip(amps, offsets, colors):
        y = amp * np.sin(2 * np.pi * t) + off + rng.standard_normal(200) * 0.1
        ax_raw.plot(t, y, lw=0.9, color=col, alpha=0.8)
    ax_raw.set_title("Raw (unscaled)", fontsize=8)
    ax_raw.set_xlabel("Time (s)", fontsize=7)
    ax_raw.set_ylabel("Amplitude", fontsize=7)
    ax_raw.tick_params(labelsize=6)
    # Right: z-scored (normalized)
    ax_z = axes[1]
    for amp, off, col in zip(amps, offsets, colors):
        y = amp * np.sin(2 * np.pi * t) + off + rng.standard_normal(200) * 0.1
        y_z = (y - y.mean()) / y.std()
        ax_z.plot(t, y_z, lw=0.9, color=col, alpha=0.8)
    ax_z.set_title("Z-scored (mean=0, std=1)", fontsize=8)
    ax_z.set_xlabel("Time (s)", fontsize=7)
    ax_z.set_ylabel("Z-score", fontsize=7)
    ax_z.axhline(0, color="gray", ls="--", lw=0.6)
    ax_z.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_timeseries_two_panel() -> Figure:
    """Two-panel: top = mean signal overlay (blue), bottom = variance trace (orange)."""
    rng = np.random.default_rng(18)
    t = np.linspace(0, 10, 500)
    n_subj = 6
    group_mean = np.sin(0.5 * t) * 0.3 + rng.standard_normal(500) * 0.1
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(5, 3.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]}
    )
    # Top: per-subject traces + group mean + ±1 SD band
    sd = np.zeros(500)
    for _ in range(n_subj):
        y = group_mean + rng.standard_normal(500) * 0.25
        ax_top.plot(t, y, lw=0.5, alpha=0.35, color="steelblue")
        sd += (y - group_mean) ** 2
    sd = np.sqrt(sd / n_subj)
    ax_top.plot(t, group_mean, lw=1.2, color="steelblue", label="group mean")
    ax_top.fill_between(
        t, group_mean - sd, group_mean + sd, alpha=0.2, color="steelblue", label="±1 SD"
    )
    ax_top.set_ylabel("Mean signal", fontsize=7)
    ax_top.legend(fontsize=6)
    ax_top.tick_params(labelsize=6)
    ax_top.set_title("Inter-subject time-series overview", fontsize=8)
    # Bottom: channel-averaged variance
    var_trace = np.abs(rng.standard_normal(500)) * 0.5 + 0.5
    var_trace[100:150] *= 0.3
    var_trace[330:380] *= 0.25
    ax_bot.plot(t, var_trace, lw=0.8, color="darkorange")
    ax_bot.set_xlabel("Time (s)", fontsize=7)
    ax_bot.set_ylabel("Variance", fontsize=7)
    ax_bot.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_windowed_mean_var_bars() -> Figure:
    """Two stacked bar charts: mean signal (blue) and variance (orange, green=sync)."""
    rng = np.random.default_rng(19)
    n_win = 20
    x = np.arange(n_win)
    mean_vals = rng.standard_normal(n_win) * 0.3
    var_vals = np.abs(rng.standard_normal(n_win)) * 0.4 + 0.3
    threshold = np.percentile(var_vals, 10)
    sync_mask = var_vals < threshold
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
    # Top: mean signal per window (blue)
    ax_top.bar(x, mean_vals, color="steelblue", width=0.8)
    ax_top.axhline(0, color="gray", lw=0.6)
    ax_top.set_ylabel("Mean signal", fontsize=7)
    ax_top.tick_params(labelsize=6)
    ax_top.set_title("Windowed synchrony — bar charts", fontsize=8)
    # Bottom: variance per window (orange, green=sync)
    bar_colors = ["seagreen" if s else "darkorange" for s in sync_mask]
    ax_bot.bar(x, var_vals, color=bar_colors, width=0.8)
    ax_bot.axhline(threshold, color="gray", ls="--", lw=0.8, label="10th pct")
    ax_bot.set_xlabel("Window index", fontsize=7)
    ax_bot.set_ylabel("Variance", fontsize=7)
    ax_bot.legend(fontsize=6)
    ax_bot.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


def sketch_windowed_overlay_panels() -> Figure:
    """Three-panel windowed overlay: mean signal, variance+windowed line, channel heatmap."""
    rng = np.random.default_rng(20)
    t = np.linspace(0, 10, 500)
    n_ch = 30
    fig, axes = plt.subplots(
        3, 1, figsize=(5, 4), gridspec_kw={"height_ratios": [1, 1, 1.5]}
    )
    # Panel 1: mean signal (blue) with per-subject traces
    group_mean = np.sin(0.4 * t) * 0.3 + rng.standard_normal(500) * 0.1
    for _ in range(4):
        axes[0].plot(
            t,
            group_mean + rng.standard_normal(500) * 0.2,
            lw=0.5,
            alpha=0.3,
            color="steelblue",
        )
    axes[0].plot(t, group_mean, lw=1.0, color="steelblue")
    axes[0].set_ylabel("Mean signal", fontsize=7)
    axes[0].tick_params(labelsize=5)
    axes[0].set_title("Windowed overlay", fontsize=8)
    # Panel 2: continuous variance (orange) + windowed mean line (red step)
    var_cont = np.abs(rng.standard_normal(500)) * 0.4 + 0.5
    n_win = 25
    win_edges = np.linspace(0, len(t), n_win + 1, dtype=int)
    win_means = [var_cont[win_edges[i] : win_edges[i + 1]].mean() for i in range(n_win)]
    win_t = [
        (t[win_edges[i]] + t[min(win_edges[i + 1], len(t) - 1)]) / 2
        for i in range(n_win)
    ]
    axes[1].plot(t, var_cont, lw=0.8, color="darkorange")
    axes[1].step(
        win_t, win_means, where="mid", lw=1.2, color="tomato", label="windowed mean"
    )
    axes[1].set_ylabel("Variance", fontsize=7)
    axes[1].legend(fontsize=5)
    axes[1].tick_params(labelsize=5)
    # Panel 3: heatmap (channels × time), reddish colormap, positive only
    heatmap = np.abs(rng.standard_normal((n_ch, 500))) * 0.5 + 0.1
    im = axes[2].imshow(heatmap, aspect="auto", cmap="Reds", vmin=0, vmax=1.5)
    axes[2].set_ylabel("Channel", fontsize=7)
    axes[2].set_xlabel("Time (s)", fontsize=7)
    axes[2].tick_params(labelsize=5)
    axes[2].set_xticks([])
    plt.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def sketch_histogram_violin() -> Figure:
    """Two separate histograms (Pearson/Spearman) with red negative tail + markers, plus violin."""
    rng = np.random.default_rng(21)
    pearson = rng.normal(0.18, 0.11, 300)
    spearman = rng.normal(0.20, 0.10, 300)
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    for ax, data, label, col in [
        (axes[0], pearson, "Pearson", "steelblue"),
        (axes[1], spearman, "Spearman", "darkorange"),
    ]:
        bins = np.linspace(-0.3, 0.55, 28)
        counts, _ = np.histogram(data, bins=bins)
        for i, (left, right, cnt) in enumerate(zip(bins[:-1], bins[1:], counts)):
            bar_col = "tomato" if left < 0 else col
            ax.bar(
                left,
                cnt,
                width=(right - left),
                color=bar_col,
                edgecolor="white",
                lw=0.3,
                align="edge",
            )
        ax.axvline(data.mean(), color="black", ls="-", lw=1.0, label="mean")
        ax.axvline(np.median(data), color="gray", ls="--", lw=0.9, label="median")
        ax.set_xlabel("LOO-ISC", fontsize=6)
        ax.set_ylabel("Count", fontsize=6)
        ax.set_title(label, fontsize=7)
        ax.legend(fontsize=5)
        ax.tick_params(labelsize=5)
    # Violin plot
    ax = axes[2]
    subj_means_p = [rng.normal(0.18, 0.08, 40).mean() for _ in range(8)]
    subj_means_s = [rng.normal(0.20, 0.07, 40).mean() for _ in range(8)]
    vp = ax.violinplot([subj_means_p, subj_means_s], positions=[1, 2], showmedians=True)
    for pc, col in zip(vp["bodies"], ["steelblue", "darkorange"]):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    ax.scatter([1] * 8, subj_means_p, color="steelblue", s=15, zorder=3)
    ax.scatter([2] * 8, subj_means_s, color="darkorange", s=15, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pearson", "Spearman"], fontsize=6)
    ax.set_ylabel("Mean per-channel ISC", fontsize=6)
    ax.set_title("Per-subject violin", fontsize=7)
    ax.tick_params(labelsize=5)
    fig.suptitle("LOO-ISC distribution", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_matrix_heatmap_no_diag() -> Figure:
    """Pairwise ISC matrix with blank diagonal, plus bar chart and overlaid histogram."""
    rng = np.random.default_rng(22)
    n = 6
    mat = rng.uniform(-0.1, 0.6, (n, n))
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    # Heatmap with blank diagonal
    ax = axes[0]
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-0.2, vmax=0.6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"S{i + 1}" for i in range(n)], fontsize=5)
    ax.set_yticklabels([f"S{i + 1}" for i in range(n)], fontsize=5)
    ax.set_title("ISC matrix (diag blank)", fontsize=7)
    ax.tick_params(labelsize=4)
    # Bar chart: mean off-diagonal per subject
    ax = axes[1]
    x = np.arange(n)
    w = 0.35
    pearson_means = [np.nanmean(rng.uniform(0.05, 0.4, n - 1)) for _ in range(n)]
    spearman_means = [np.nanmean(rng.uniform(0.04, 0.38, n - 1)) for _ in range(n)]
    ax.bar(x - w / 2, pearson_means, w, label="Pearson", color="steelblue")
    ax.bar(x + w / 2, spearman_means, w, label="Spearman", color="darkorange")
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i + 1}" for i in range(n)], fontsize=5)
    ax.set_ylabel("Mean off-diag ISC", fontsize=6)
    ax.set_title("Per-subject mean ISC", fontsize=7)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)
    # Overlaid histogram of off-diagonal entries
    ax = axes[2]
    off_p = rng.normal(0.22, 0.09, 120)
    off_s = rng.normal(0.24, 0.08, 120)
    bins = np.linspace(-0.1, 0.55, 25)
    ax.hist(
        off_p,
        bins=bins,
        alpha=0.6,
        label="Pearson",
        color="steelblue",
        edgecolor="white",
    )
    ax.hist(
        off_s,
        bins=bins,
        alpha=0.6,
        label="Spearman",
        color="darkorange",
        edgecolor="white",
    )
    ax.axvline(off_p.mean(), color="steelblue", ls="--", lw=1.0)
    ax.axvline(off_s.mean(), color="darkorange", ls="--", lw=1.0)
    ax.set_xlabel("Pairwise ISC", fontsize=6)
    ax.set_ylabel("Count", fontsize=6)
    ax.set_title("Off-diagonal distribution", fontsize=7)
    ax.legend(fontsize=5)
    ax.tick_params(labelsize=5)
    fig.suptitle("Pairwise ISC (Pearson vs Spearman)", fontsize=8)
    fig.tight_layout()
    return fig


def sketch_timeseries_comparison_overlay() -> Figure:
    """Stair-wise overlay: channel-average ISC (solid blue) vs mean-field ISC (dashed, colour-coded).

    Mean-field line is dashed green when >= 0, dashed red when < 0.
    Style mirrors Section 3 Figure 3 (Pearson vs Spearman stepped comparison).

    The mean-field staircase is built by drawing each window's horizontal dashed
    segment explicitly (x[i] → x[i+1]) then a solid thin vertical transition to
    the next value. This avoids gaps that occur when ax.step() is called in pieces.
    """
    rng = np.random.default_rng(23)
    n_win = 40
    x = np.arange(n_win + 1)  # x[0]…x[n_win]; x[i+1]-x[i] = 1 window width
    channel_avg = rng.normal(0.18, 0.06, n_win)
    mean_field = rng.normal(0.12, 0.10, n_win)

    fig, ax = plt.subplots(figsize=(5.5, 2.8))

    # Channel-average ISC: single solid blue step call — no gaps
    ax.step(
        x[:-1],
        channel_avg,
        where="post",
        color="steelblue",
        lw=1.4,
        label="channel-avg ISC",
    )
    # Extend final horizontal to x[n_win]
    ax.plot(
        [x[-2], x[-1]], [channel_avg[-1], channel_avg[-1]], color="steelblue", lw=1.4
    )

    # Mean-field ISC: draw window-by-window so each horizontal segment has the
    # right colour.  Horizontal bars are dashed; vertical transitions are thin
    # solid lines in the colour of the outgoing window so the staircase is gapless.
    for i in range(n_win):
        col = "seagreen" if mean_field[i] >= 0 else "tomato"
        # Horizontal dashed bar for this window
        ax.plot(
            [x[i], x[i + 1]], [mean_field[i], mean_field[i]], color=col, lw=1.5, ls="--"
        )
        # Vertical transition to next window (solid, same colour)
        if i < n_win - 1:
            ax.plot(
                [x[i + 1], x[i + 1]],
                [mean_field[i], mean_field[i + 1]],
                color=col,
                lw=1.0,
                ls="-",
            )

    # Legend proxies
    ax.plot([], [], color="seagreen", ls="--", lw=1.5, label="mean-field ISC (≥ 0)")
    ax.plot([], [], color="tomato", ls="--", lw=1.5, label="mean-field ISC (< 0)")

    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("Medium window index", fontsize=7)
    ax.set_ylabel("ISC", fontsize=7)
    ax.set_title("Mean-field vs channel-average ISC", fontsize=8)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    return fig


SKETCH_FUNCTIONS: dict[str, Callable[[], Figure]] = {
    "timeseries_multichannel": sketch_timeseries_multichannel,
    "psd": sketch_psd,
    "topomap": sketch_topomap,
    "timeseries_offset": sketch_timeseries_offset,
    "matrix_heatmap": sketch_matrix_heatmap,
    "timeseries_variance": sketch_timeseries_variance,
    "histogram": sketch_histogram,
    "timeseries_multisubject": sketch_timeseries_multisubject,
    "timeseries_multiband": sketch_timeseries_multiband,
    "histogram_overlay": sketch_histogram_overlay,
    "timeseries_heatmap": sketch_timeseries_heatmap,
    "bar_grouped": sketch_bar_grouped,
    "histogram_facet": sketch_histogram_facet,
    "bar_grouped_bands": sketch_bar_grouped_bands,
    "grid_timeseries_heatmap": sketch_grid_timeseries_heatmap,
    "raster_overlap": sketch_raster_overlap,
    "polar_histogram": sketch_polar_histogram,
    # 01-mean-variance specific sketches
    "zscore_transform": sketch_zscore_transform,
    "timeseries_two_panel": sketch_timeseries_two_panel,
    "windowed_mean_var_bars": sketch_windowed_mean_var_bars,
    "windowed_overlay_panels": sketch_windowed_overlay_panels,
    # 02-isc specific sketches
    "histogram_violin": sketch_histogram_violin,
    "matrix_heatmap_no_diag": sketch_matrix_heatmap_no_diag,
    "timeseries_comparison_overlay": sketch_timeseries_comparison_overlay,
    "multiscale_sw_isc": sketch_multiscale_sw_isc,
}


def render_sketch(sketch_type: str) -> None:
    """Render a sketch inline using st.pyplot, or show a warning if unknown."""
    func = SKETCH_FUNCTIONS.get(sketch_type)
    if func is None:
        st.warning(f"No sketch function registered for type: `{sketch_type}`")
        return
    fig = func()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    st.image(buf, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — analysis group selector
# ---------------------------------------------------------------------------
st.sidebar.header("Analysis Group")
group_options = {f"{a['id']} — {a['title']}": a for a in analyses}
selected_label = st.sidebar.selectbox("Select analysis:", list(group_options.keys()))
selected = group_options[selected_label]

# ---------------------------------------------------------------------------
# Main area — analysis header
# ---------------------------------------------------------------------------
st.header(selected["title"])
st.write(selected.get("description", ""))

# Notebooks
nb_list = selected.get("notebooks", [])
if nb_list:
    st.subheader("📓 Notebooks")
    for nb in nb_list:
        url = f"{GITHUB_BASE}/{nb['path']}"
        st.markdown(f"- [{nb['label']}]({url})")

# ---------------------------------------------------------------------------
# Plot cards grid
# ---------------------------------------------------------------------------
plots: list[dict] = selected.get("plots", [])
if not plots:
    st.info("No plots defined for this analysis yet.")
else:
    st.subheader("🖼 Plot Cards")

    current_group: str | None = None
    col_index = 0  # tracks position within the current 2-column row

    for plot in plots:
        plot_group = plot.get("group")

        # Render a group header whenever the group label changes
        if plot_group and plot_group != current_group:
            st.markdown(f"### {plot_group}")
            st.divider()
            current_group = plot_group
            col_index = 0  # reset column position for the new group
            cols = st.columns(2, gap="large")
        elif col_index == 0:
            # First plot of a groupless section — create initial columns
            cols = st.columns(2, gap="large")

        col = cols[col_index % 2]
        col_index += 1

        with col:
            with st.container(border=True):
                st.markdown(f"**{plot['title']}**")
                st.caption(
                    f"Notebook: `{plot.get('notebook', '')}` | "
                    f"Section: *{plot.get('section', '')}*"
                )

                # Data shape
                ds = plot.get("data_shape", {})
                if ds:
                    st.markdown("**Data shape**")
                    st.table(
                        {
                            "": ["Input", "Output"],
                            "Shape": [ds.get("input", ""), ds.get("output", "")],
                        }
                    )

                # Order of operations
                ops = plot.get("operation_order", [])
                if ops:
                    st.markdown("**Order of operations**")
                    for i, op in enumerate(ops, 1):
                        st.markdown(f"{i}. {op}")

                # Interpretation
                interp = plot.get("interpretation", "")
                if interp:
                    st.info(f"💡 {interp.strip()}")

                # Sketch
                sketch_type = plot.get("sketch_type", "")
                if sketch_type:
                    st.markdown("**Sketch**")
                    render_sketch(sketch_type)

        # When we've just filled the right column, create a fresh pair for the next row
        if col_index % 2 == 0:
            cols = st.columns(2, gap="large")
