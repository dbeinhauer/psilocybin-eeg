"""
Reusable visualisation functions for ISC analysis.

Every function accepts generic dicts of results keyed by a descriptive label
such as a music type or experimental condition
(e.g. ``{"CLASSIC": …, "PSYTRANCE": …}``) and produces publication-ready
matplotlib figures.  Axis labels, colours, thresholds and save paths are
fully configurable so that the same functions work for any data
representation (raw EEG, ICA, wavelet, mean response, …).
"""

from __future__ import annotations

from itertools import groupby
import logging
from operator import itemgetter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.analysis.isc import FREQUENCY_BANDS
from src.analysis.data_representations import AnalysisData
from src.definitions.fields import FrequencyBandNames

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE = [
    "steelblue",
    "darkorange",
    "seagreen",
    "crimson",
    "mediumpurple",
    "sienna",
]

_BAND_COLORS = {
    FrequencyBandNames.DELTA.value: "#4e79a7",
    FrequencyBandNames.THETA.value: "#f28e2b",
    FrequencyBandNames.ALPHA.value: "#59a14f",
    FrequencyBandNames.BETA.value: "#e15759",
    FrequencyBandNames.GAMMA.value: "#b07aa1",
}


def _resolve_colors(
    labels: list[str], colors: Optional[dict[str, str]] = None
) -> dict[str, str]:
    if colors is not None:
        return colors
    return {
        label: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)]
        for i, label in enumerate(labels)
    }


def _resolve_band_thresholds(
    band_names: list[str],
    isc_threshold: Union[float, dict[str, float]],
) -> dict[str, float]:
    if isinstance(isc_threshold, dict):
        return {b: isc_threshold.get(b, 0.035) for b in band_names}
    return {b: float(isc_threshold) for b in band_names}


def _save_fig(fig: Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. LOO-ISC distribution
# ---------------------------------------------------------------------------


def plot_loo_isc_distribution(
    mean_loo_iscs: dict[str, np.ndarray],
    *,
    title: str = "Distribution of mean LOO-ISC across features",
    xlabel: str = "Pearson r (LOO-ISC)",
    ylabel: str = "Number of features",
    colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (9, 4),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Histogram of mean LOO-ISC across features, overlaid for every condition.

    :param mean_loo_iscs: ``{label: mean_loo_isc}`` with shape ``(n_features,)``.
    """
    labels = list(mean_loo_iscs.keys())
    colors = _resolve_colors(labels, colors)

    fig, ax = plt.subplots(figsize=figsize)
    for label, mean_isc in mean_loo_iscs.items():
        ax.hist(
            mean_isc,
            bins=30,
            alpha=0.55,
            edgecolor="black",
            color=colors[label],
            label=label,
        )
        ax.axvline(
            mean_isc.mean(),
            color=colors[label],
            ls="--",
            lw=1.5,
            label=f"{label} mean = {mean_isc.mean():.4f}",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 2. Sliding-window ISC (time-resolved heatmap + mean trace)
# ---------------------------------------------------------------------------


def plot_sliding_window_isc(
    sw_results: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    isc_threshold: float = 0.035,
    feature_axis_label: str = "Channel index",
    title: str = "Time-resolved sliding-window ISC",
    subtitle_template: str = "{label} — Time-resolved ISC",
    colors: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    For each condition: mean ISC time course (top) + per-feature heatmap (bottom).

    :param sw_results: ``{label: (isc_timecourse, window_times)}``
        where shapes are ``(n_windows, n_features)`` and ``(n_windows,)``.
    """
    labels = list(sw_results.keys())
    n_conditions = len(labels)
    colors = _resolve_colors(labels, colors)

    if figsize is None:
        figsize = (16, 8 * n_conditions)

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(n_conditions, 1, figure=fig, hspace=0.35)

    inner_grids = []
    for row in range(n_conditions):
        gs = GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=outer_gs[row],
            height_ratios=[1, 3],
            hspace=0.05,
        )
        inner_grids.append(gs)

    # Shared vmax for comparable colourscales
    global_vmax = max(np.nanmax(np.abs(r[0])) for r in sw_results.values())

    for idx, label in enumerate(labels):
        ax_top = fig.add_subplot(inner_grids[idx][0])
        ax_bot = fig.add_subplot(inner_grids[idx][1])

        sw_isc, sw_times = sw_results[label]
        time_min = sw_times / 60
        t_start, t_end = time_min[0], time_min[-1]
        half_step = (time_min[1] - time_min[0]) / 2
        mean_sw_isc = sw_isc.mean(axis=1)
        sig_mask = mean_sw_isc > isc_threshold
        color = colors[label]

        # Shade significant windows
        for ax in (ax_top, ax_bot):
            for i, is_sig in enumerate(sig_mask):
                if is_sig:
                    ax.axvspan(
                        time_min[i] - half_step,
                        time_min[i] + half_step,
                        color="gold",
                        alpha=0.35,
                        linewidth=0,
                    )

        # Top: mean ISC trace
        ax_top.plot(time_min, mean_sw_isc, color=color, lw=1.2, zorder=3)
        ax_top.fill_between(time_min, mean_sw_isc, alpha=0.25, color=color, zorder=2)
        ax_top.axhline(0, color="grey", ls="--", lw=0.6, zorder=1)
        ax_top.axhline(
            isc_threshold,
            color="tomato",
            ls="--",
            lw=1.2,
            zorder=3,
            label=f"threshold r = {isc_threshold}",
        )
        ax_top.set_xlim(t_start, t_end)
        ax_top.set_ylabel("Mean LOO-ISC (r)")
        ax_top.set_title(subtitle_template.format(label=label))
        ax_top.tick_params(labelbottom=False)
        ax_top.legend(loc="upper right", fontsize=9)
        div_top = make_axes_locatable(ax_top)
        div_top.append_axes("right", size="2%", pad=0.05).set_visible(False)

        # Bottom: heatmap
        im = ax_bot.imshow(
            sw_isc.T,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            extent=(t_start, t_end, 0, sw_isc.shape[1]),
            vmin=-global_vmax,
            vmax=global_vmax,
            zorder=1,
        )
        ax_bot.set_xlim(t_start, t_end)
        ax_bot.set_xlabel("Time (min)")
        ax_bot.set_ylabel(feature_axis_label)
        div_bot = make_axes_locatable(ax_bot)
        cax = div_bot.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(im, cax=cax, label="LOO-ISC (r)")

        n_sig = sig_mask.sum()
        _logger.info(
            f"[{label}] Significant windows (r > {isc_threshold}): "
            f"{n_sig}/{len(sig_mask)} ({100 * n_sig / len(sig_mask):.1f} %)"
        )

    fig.suptitle(title, y=1.01, fontsize=14)
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 3. Significant intervals (text output)
# ---------------------------------------------------------------------------


def print_significant_intervals(
    sw_results: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    isc_threshold: float = 0.035,
) -> None:
    """
    Print contiguous significant intervals for each condition.

    :param sw_results: ``{label: (isc_timecourse, window_times)}``
    """
    header = (
        f"{'#':>3}  {'Start (min)':>11}  {'End (min)':>9}  "
        f"{'Duration (s)':>12}  {'Mean r':>8}  {'Peak r':>8}"
    )
    sep = "-" * 62

    for label in sw_results:
        sw_isc, sw_times = sw_results[label]
        time_min = sw_times / 60
        half_step = (time_min[1] - time_min[0]) / 2
        mean_sw_isc = sw_isc.mean(axis=1)
        sig_mask = mean_sw_isc > isc_threshold
        n_sig = sig_mask.sum()

        sig_indices = np.where(sig_mask)[0].tolist()
        intervals: list[tuple[float, float, float, float]] = []
        for _, group in groupby(enumerate(sig_indices), lambda x: x[1] - x[0]):
            run = list(map(itemgetter(1), group))
            t_begin = time_min[run[0]] - half_step
            t_finish = time_min[run[-1]] + half_step
            peak_idx = run[np.argmax(mean_sw_isc[run])]
            intervals.append(
                (t_begin, t_finish, mean_sw_isc[run].mean(), mean_sw_isc[peak_idx])
            )

        _logger.info(f"=== {label} ===")
        _logger.info(
            f"Threshold: r > {isc_threshold}   |   "
            f"Significant windows: {n_sig}/{len(sig_mask)} "
            f"({100 * n_sig / len(sig_mask):.1f} %)"
        )
        _logger.info(header)
        _logger.info(sep)
        for k, (tb, tf, mean_r, peak_r) in enumerate(intervals, 1):
            dur_s = (tf - tb) * 60
            _logger.info(
                f"{k:>3}  {tb:>11.3f}  {tf:>9.3f}  {dur_s:>12.1f}  "
                f"{mean_r:>8.4f}  {peak_r:>8.4f}"
            )
        _logger.info("")


# ---------------------------------------------------------------------------
# 4. Band ISC distributions
# ---------------------------------------------------------------------------


def plot_band_isc_distributions(
    band_iscs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    feature_axis_label: str = "Number of features",
    colors: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    One subplot per frequency band, conditions overlaid.

    :param band_iscs: ``{label: {band_name: (loo_isc, mean_loo_isc)}}``
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_iscs.keys())
    n_bands = len(band_names)
    colors = _resolve_colors(music_labels, colors)

    if figsize is None:
        figsize = (5 * n_bands, 4)

    fig, axes = plt.subplots(1, n_bands, figsize=figsize, sharey=True)
    if n_bands == 1:
        axes = [axes]

    for ax, band in zip(axes, band_names):
        l_freq, h_freq = bands[band]
        for label in music_labels:
            mean_isc = band_iscs[label][band][1]
            color = colors.get(label, None)
            ax.hist(
                mean_isc,
                bins=25,
                alpha=0.5,
                edgecolor="black",
                color=color,
                label=label,
            )
            ax.axvline(
                mean_isc.mean(),
                color=color,
                ls="--",
                lw=1.5,
                label=f"{label} \u03bc={mean_isc.mean():.4f}",
            )
        ax.axvline(0, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{band}\n({l_freq}\u2013{h_freq} Hz)", fontsize=10)
        ax.set_xlabel("Mean LOO-ISC (r)")
        if ax is axes[0]:
            ax.set_ylabel(feature_axis_label)
        ax.legend(fontsize=7)

    fig.suptitle("Distribution of mean LOO-ISC per frequency band", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 5. Band mean ISC bar chart
# ---------------------------------------------------------------------------


def plot_band_mean_isc_bar(
    band_iscs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (8, 4),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Bar chart: mean LOO-ISC (averaged over features) per band × condition.

    :param band_iscs: ``{label: {band_name: (loo_isc, mean_loo_isc)}}``
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_iscs.keys())
    colors = _resolve_colors(music_labels, colors)

    x = np.arange(len(band_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    for k, label in enumerate(music_labels):
        means = [band_iscs[label][b][1].mean() for b in band_names]
        stds = [band_iscs[label][b][1].std() for b in band_names]
        offset = (k - (len(music_labels) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            label=label,
            yerr=stds,
            capsize=4,
            color=colors.get(label, None),
            alpha=0.8,
            edgecolor="black",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n({bands[b][0]}\u2013{bands[b][1]} Hz)" for b in band_names]
    )
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_ylabel("Mean LOO-ISC (r) \u00b1 SD")
    ax.set_title("Mean feature-averaged LOO-ISC per frequency band")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 6. Band sliding-window ISC
# ---------------------------------------------------------------------------


def plot_band_sliding_window_isc(
    band_sw: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    isc_threshold: Union[float, dict[str, float]] = 0.035,
    feature_axis_label: str = "Channel",
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    For each condition (column) × frequency band (row): trace + heatmap.

    :param band_sw: ``{label: {band_name: (isc_timecourse, window_times)}}``
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_sw.keys())
    n_bands = len(band_names)
    n_music = len(music_labels)
    thresholds = _resolve_band_thresholds(band_names, isc_threshold)

    if figsize is None:
        figsize = (10 * n_music, 5 * n_bands)

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(1, n_music, figure=fig, wspace=0.3)

    for col, label in enumerate(music_labels):
        inner_gs = GridSpecFromSubplotSpec(
            n_bands, 1, subplot_spec=outer_gs[col], hspace=0.6
        )
        for row, band in enumerate(band_names):
            band_gs = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=inner_gs[row],
                height_ratios=[1, 2.5],
                hspace=0.05,
            )
            ax_top = fig.add_subplot(band_gs[0])
            ax_bot = fig.add_subplot(band_gs[1])

            tc, times = band_sw[label][band]
            time_min = times / 60
            t_start, t_end = time_min[0], time_min[-1]
            half_step = (time_min[1] - time_min[0]) / 2
            mean_tc = tc.mean(axis=1)
            thr = thresholds[band]
            sig_mask = mean_tc > thr
            vmax = np.nanmax(np.abs(tc))

            for ax in (ax_top, ax_bot):
                for i, is_sig in enumerate(sig_mask):
                    if is_sig:
                        ax.axvspan(
                            time_min[i] - half_step,
                            time_min[i] + half_step,
                            color="gold",
                            alpha=0.35,
                            linewidth=0,
                        )

            ax_top.plot(time_min, mean_tc, color="steelblue", lw=1.2, zorder=3)
            ax_top.fill_between(
                time_min, mean_tc, alpha=0.2, color="steelblue", zorder=2
            )
            ax_top.axhline(0, color="grey", ls="--", lw=0.6, zorder=1)
            ax_top.axhline(
                thr, color="tomato", ls="--", lw=1.0, zorder=3, label=f"thr r={thr}"
            )
            ax_top.set_xlim(t_start, t_end)
            ax_top.set_ylabel("r", fontsize=8)
            l_freq, h_freq = bands[band]
            band_title = f"{band} ({l_freq}\u2013{h_freq} Hz)"
            if row == 0:
                ax_top.set_title(
                    f"{label}\n{band_title}", fontsize=10, fontweight="bold"
                )
            else:
                ax_top.set_title(band_title, fontsize=9)
            ax_top.tick_params(labelbottom=False, labelsize=7)
            ax_top.yaxis.set_tick_params(labelsize=7)
            ax_top.legend(loc="upper right", fontsize=6)
            div = make_axes_locatable(ax_top)
            div.append_axes("right", size="2%", pad=0.05).set_visible(False)

            im = ax_bot.imshow(
                tc.T,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                extent=(t_start, t_end, 0, tc.shape[1]),
                vmin=-vmax,
                vmax=vmax,
                zorder=1,
            )
            ax_bot.set_xlim(t_start, t_end)
            ax_bot.set_xlabel("Time (min)", fontsize=8)
            ax_bot.set_ylabel(feature_axis_label, fontsize=8)
            ax_bot.tick_params(labelsize=7)
            div_bot = make_axes_locatable(ax_bot)
            cax = div_bot.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax, label="ISC (r)")
            cax.tick_params(labelsize=7)

    fig.suptitle(
        "Band-specific sliding-window ISC (5 s / 2.5 s step)",
        fontsize=13,
        y=1.01,
    )
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 7. Band significant intervals (text output)
# ---------------------------------------------------------------------------


def print_band_significant_intervals(
    band_sw: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    band_thresholds: Optional[Union[float, dict[str, float]]] = None,
    default_threshold: float = 0.035,
) -> None:
    """Print contiguous significant intervals per band × condition."""
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    if band_thresholds is None:
        band_thresholds = default_threshold
    thresholds = _resolve_band_thresholds(band_names, band_thresholds)

    for label in band_sw:
        _logger.info(f"{'=' * 70}")
        _logger.info(f"  {label}")
        _logger.info(f"{'=' * 70}")
        for band in band_names:
            tc, times = band_sw[label][band]
            time_min = times / 60
            half_step = (time_min[1] - time_min[0]) / 2
            mean_tc = tc.mean(axis=1)
            thr = thresholds[band]
            sig_mask = mean_tc > thr
            n_sig = sig_mask.sum()
            n_total = len(sig_mask)

            sig_indices = np.where(sig_mask)[0].tolist()
            intervals: list[tuple[float, float, float, float]] = []
            for _, grp in groupby(enumerate(sig_indices), lambda x: x[1] - x[0]):
                run = list(map(itemgetter(1), grp))
                t_b = time_min[run[0]] - half_step
                t_f = time_min[run[-1]] + half_step
                intervals.append((t_b, t_f, mean_tc[run].mean(), mean_tc[run].max()))

            l_freq, h_freq = bands[band]
            _logger.info(
                f"  {band} ({l_freq}\u2013{h_freq} Hz)  |  threshold r > {thr}  |  "
                f"sig windows: {n_sig}/{n_total} ({100 * n_sig / n_total:.1f} %)"
            )
            if intervals:
                header = (
                    f"  {'#':>3}  {'Start(min)':>10}  {'End(min)':>9}  "
                    f"{'Dur(s)':>7}  {'Mean r':>8}  {'Peak r':>8}"
                )
                _logger.info(header)
                _logger.info("  " + "-" * 56)
                for k, (tb, tf, mr, pr) in enumerate(intervals, 1):
                    dur_s = (tf - tb) * 60
                    _logger.info(
                        f"  {k:>3}  {tb:>10.3f}  {tf:>9.3f}  "
                        f"{dur_s:>7.1f}  {mr:>8.4f}  {pr:>8.4f}"
                    )
            else:
                _logger.info("  (no significant intervals)")
        _logger.info("")


# ---------------------------------------------------------------------------
# 8. Band overlap — consensus synchrony windows
# ---------------------------------------------------------------------------


def plot_band_overlap(
    band_sw: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    band_thresholds: Optional[Union[float, dict[str, float]]] = None,
    broadband_sw: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
    broadband_threshold: float = 0.035,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Binary raster of per-band significance + band-count curve.

    :param band_sw: ``{label: {band_name: (isc_timecourse, window_times)}}``
    :param broadband_sw: ``{label: (isc_timecourse, window_times)}``
        broadband (unfiltered) sliding-window results.  Added as an extra row.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_sw.keys())
    n_bands = len(band_names)
    n_music = len(music_labels)
    use_broadband = broadband_sw is not None
    n_rows = n_bands + (1 if use_broadband else 0)

    if band_thresholds is None:
        band_thresholds = 0.035
    thresholds = _resolve_band_thresholds(band_names, band_thresholds)
    bb_thr = broadband_threshold

    band_colors = dict(_BAND_COLORS)
    default_mpl_colors = cm.get_cmap("tab10").colors
    for i, b in enumerate(band_names):
        if b not in band_colors:
            band_colors[b] = default_mpl_colors[i % len(default_mpl_colors)]
    BB_COLOR = "#7f7f7f"
    ALL_COLOR = "#333333"

    if figsize is None:
        figsize = (14 * n_music, 8)

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(1, n_music, figure=fig, wspace=0.35)

    for col, label in enumerate(music_labels):
        first_band = band_names[0]
        _, ref_times = band_sw[label][first_band]
        time_min = ref_times / 60
        n_windows = len(time_min)
        half_step = (time_min[1] - time_min[0]) / 2
        t_start, t_end = time_min[0], time_min[-1]

        sig_matrix = np.zeros((n_rows, n_windows), dtype=bool)
        for bi, band in enumerate(band_names):
            tc, _ = band_sw[label][band]
            sig_matrix[bi] = tc.mean(axis=1) > thresholds[band]
        if use_broadband:
            bb_tc, _ = broadband_sw[label]
            sig_matrix[n_bands] = bb_tc.mean(axis=1) > bb_thr

        band_count = sig_matrix.sum(axis=0)
        all_sig = band_count == n_rows

        inner_gs = GridSpecFromSubplotSpec(
            3,
            1,
            subplot_spec=outer_gs[col],
            height_ratios=[n_rows, n_rows, 1.5],
            hspace=0.08,
        )
        ax_raster = fig.add_subplot(inner_gs[0])
        ax_stack = fig.add_subplot(inner_gs[1], sharex=ax_raster)
        ax_count = fig.add_subplot(inner_gs[2], sharex=ax_raster)

        # -- Raster ---------------------------------------------------------
        ax_raster.set_xlim(t_start - half_step, t_end + half_step)
        ax_raster.set_ylim(0, n_rows)
        ax_raster.set_facecolor("#f5f5f5")

        row_colors = [band_colors[b] for b in band_names]
        if use_broadband:
            row_colors.append(BB_COLOR)

        for ri in range(n_rows):
            for wi, is_sig in enumerate(sig_matrix[ri]):
                if is_sig:
                    ax_raster.add_patch(
                        Rectangle(
                            (time_min[wi] - half_step, ri),
                            2 * half_step,
                            1,
                            color=row_colors[ri],
                            alpha=0.85,
                            lw=0,
                        )
                    )

        row_labels = [
            f"{b}\n({bands[b][0]}\u2013{bands[b][1]} Hz)\nthr={thresholds[b]}"
            for b in band_names
        ]
        if use_broadband:
            row_labels.append(f"Broadband\n(no filter)\nthr={bb_thr}")
        ax_raster.set_yticks(np.arange(n_rows) + 0.5)
        ax_raster.set_yticklabels(row_labels, fontsize=7)
        ax_raster.set_ylabel("Band", fontsize=9)
        ax_raster.set_title(
            f"{label} \u2014 Band consensus synchrony",
            fontsize=11,
            fontweight="bold",
        )
        ax_raster.tick_params(labelbottom=False)
        for ri in range(1, n_rows):
            lw = 2.0 if (use_broadband and ri == n_bands) else 0.8
            ax_raster.axhline(ri, color="white", lw=lw)

        # -- Stacked area ---------------------------------------------------
        cumsum = np.zeros(n_windows)
        for ri in range(n_rows):
            contrib = sig_matrix[ri].astype(float)
            lbl_patch = band_names[ri] if ri < n_bands else "Broadband"
            ax_stack.fill_between(
                time_min,
                cumsum,
                cumsum + contrib,
                step="mid",
                color=row_colors[ri],
                alpha=0.85,
                label=lbl_patch,
            )
            cumsum += contrib
        ax_stack.set_xlim(t_start, t_end)
        ax_stack.set_ylim(0, n_rows)
        ax_stack.set_yticks(range(n_rows + 1))
        ax_stack.set_ylabel("# sig. rows", fontsize=9)
        ax_stack.tick_params(labelbottom=False, labelsize=8)
        ax_stack.axhline(n_rows, color="grey", ls=":", lw=0.7)
        legend_patches = [
            Patch(
                color=band_colors[b],
                label=f"{b} (thr={thresholds[b]})",
            )
            for b in band_names
        ]
        if use_broadband:
            legend_patches.append(
                Patch(color=BB_COLOR, label=f"Broadband (thr={bb_thr})")
            )
        ax_stack.legend(
            handles=legend_patches,
            fontsize=7,
            loc="upper right",
            ncol=len(legend_patches),
        )

        # -- Count bar ------------------------------------------------------
        cmap_count = cm.get_cmap("YlOrRd", n_rows + 1)
        for wi, (t, cnt) in enumerate(zip(time_min, band_count)):
            color = (
                ALL_COLOR
                if cnt == n_rows
                else (cmap_count(cnt / n_rows) if cnt > 0 else "lightgrey")
            )
            ax_count.bar(
                t, cnt, width=2 * half_step * 0.95, color=color, edgecolor="none"
            )
        ax_count.set_xlim(t_start, t_end)
        ax_count.set_ylim(0, n_rows + 0.3)
        ax_count.set_yticks(range(n_rows + 1))
        ax_count.set_ylabel("Count", fontsize=9)
        ax_count.set_xlabel("Time (min)", fontsize=9)
        ax_count.tick_params(labelsize=8)
        for wi in np.where(all_sig)[0]:
            ax_count.text(
                float(time_min[wi]),
                n_rows + 0.05,
                "\u2605",
                ha="center",
                va="bottom",
                fontsize=7,
                color="darkred",
            )

        n_all = int(all_sig.sum())
        _logger.info(
            f"[{label}] ALL-rows simultaneous windows: "
            f"{n_all}/{n_windows} ({100 * n_all / n_windows:.1f} %)"
        )

    fig.suptitle(
        "Band-overlap consensus: simultaneous synchrony across frequency bands",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 8. Mean-field ISC
# ---------------------------------------------------------------------------


def plot_mean_field_loo_isc(
    loo_pearson: dict[str, np.ndarray],
    loo_spearman: dict[str, np.ndarray],
    *,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Per-subject bar chart comparing Pearson and Spearman mean-field LOO-ISC.

    :param loo_pearson: ``{label: array(n_subjects,)}`` Pearson LOO-ISC per subject.
    :param loo_spearman: ``{label: array(n_subjects,)}`` Spearman LOO-ISC per subject.
    """
    labels = list(loo_pearson.keys())
    n_figs = len(labels)
    fig, axes = plt.subplots(1, n_figs, figsize=(max(8, n_figs * 9), 4), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0, col]
        lp = loo_pearson[label]
        ls = loo_spearman[label]
        n_subjects = len(lp)
        subj_labels = [f"S{i + 1:02d}" for i in range(n_subjects)]
        df = pd.DataFrame(
            {
                "subject": subj_labels * 2,
                "loo_isc": np.concatenate([lp, ls]),
                "method": ["Pearson"] * n_subjects + ["Spearman"] * n_subjects,
            }
        )
        sns.barplot(
            data=df,
            x="subject",
            y="loo_isc",
            hue="method",
            palette={"Pearson": "steelblue", "Spearman": "darkorange"},
            ax=ax,
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("Subject")
        ax.set_ylabel("LOO-ISC (r)")
        ax.set_title(f"[{label}]  Mean-field LOO-ISC per subject")
        ax.legend(frameon=False, fontsize=9, title="Method")
        sns.despine(ax=ax)

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


def plot_mean_field_pairwise_isc(
    pairwise: dict[str, np.ndarray],
    *,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Heatmap of mean-field pairwise Pearson ISC matrix.

    :param pairwise: ``{label: array(n_subjects, n_subjects)}``.
    """
    labels = list(pairwise.keys())
    n_figs = len(labels)
    fig, axes = plt.subplots(1, n_figs, figsize=(7 * n_figs, 6), squeeze=False)

    for col, label in enumerate(labels):
        ax = axes[0, col]
        mat = pairwise[label]
        n_subjects = mat.shape[0]
        mat_display = mat.copy()
        np.fill_diagonal(mat_display, np.nan)
        vmax = max(float(np.nanmax(np.abs(mat_display))), 0.01)
        off_diag_mask = ~np.eye(n_subjects, dtype=bool)
        mean_off = float(mat[off_diag_mask].mean())
        subj_labels = [f"S{i + 1:02d}" for i in range(n_subjects)]

        im = ax.imshow(mat_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="ISC (r)", shrink=0.85)
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels(subj_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(subj_labels, fontsize=7)
        ax.set_title(
            f"[{label}]  Mean-field pairwise ISC (Pearson)\n"
            f"mean off-diagonal r = {mean_off:.4f}"
        )

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


def plot_mean_field_vs_channel_avg_isc(
    mf_tc: dict[str, np.ndarray],
    channel_avg_tc: dict[str, np.ndarray],
    *,
    isc_threshold: float = 0.035,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Time course comparison: mean-field ISC vs channel-average ISC.

    :param mf_tc: ``{label: isc_timecourse}`` shape ``(n_windows,)`` — mean-field.
    :param channel_avg_tc: ``{label: isc_timecourse}`` shape ``(n_windows,)`` — channel average.
    :param isc_threshold: Significance threshold.
    :param window_sec: Window size used to compute the time courses (for axis label).
    :param step_sec: Step size used to compute the time courses (for time axis and label).
    """
    labels = list(mf_tc.keys())
    n_figs = len(labels)
    fig, axes = plt.subplots(1, n_figs, figsize=(14 * n_figs, 4), squeeze=False)

    C_NEG = "#e57373"
    C_THRESH = "tomato"

    for col, label in enumerate(labels):
        ax = axes[0, col]
        mf = mf_tc[label]
        ca = channel_avg_tc[label]
        n_len = min(len(mf), len(ca))
        t = np.arange(n_len) * step_sec + window_sec / 2

        ax.step(
            t,
            ca[:n_len],
            where="post",
            color="steelblue",
            lw=2.0,
            label=f"Channel-avg ISC ({window_sec:.0f} s / {step_sec:.1f} s step)",
        )
        ax.step(
            t,
            mf[:n_len],
            where="post",
            color="seagreen",
            lw=2.0,
            ls="--",
            label=f"Mean-field ISC ({window_sec:.0f} s / {step_sec:.1f} s step)",
        )
        ax.fill_between(
            t,
            0,
            np.clip(mf[:n_len], 0, None),
            step="post",
            alpha=0.12,
            color="seagreen",
        )
        ax.fill_between(
            t,
            np.clip(mf[:n_len], None, 0),
            0,
            step="post",
            alpha=0.22,
            color=C_NEG,
        )
        ax.axhline(
            isc_threshold,
            color=C_THRESH,
            ls="--",
            lw=1.0,
            label=f"Threshold r = {isc_threshold}",
        )
        ax.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("LOO-ISC (r)")
        ax.set_title(
            f"[{label}]  Mean-field vs. channel-average ISC  ({window_sec:.0f} s / {step_sec:.1f} s step)"
        )
        ax.legend(frameon=False, fontsize=9)
        sns.despine(ax=ax)

        corr_between = float(np.corrcoef(ca[:n_len], mf[:n_len])[0, 1])
        _logger.info(
            f"[{label}] Correlation channel-avg vs mean-field ISC: r = {corr_between:.4f}"
        )

    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 9. Broadband LOO-ISC: Pearson vs. Spearman histogram + per-subject violin
# ---------------------------------------------------------------------------

_C_NEG = "#e57373"
_C_THRESH = "tomato"


def _make_step_arr(
    mean_tc: np.ndarray, step_samples: int, n_times_full: int
) -> np.ndarray:
    """Repeat each window value for step_samples ticks, then trim/pad to n_times_full."""
    repeated = np.repeat(mean_tc, step_samples)
    pad = n_times_full - len(repeated)
    if pad > 0:
        return np.pad(repeated, (0, pad), mode="edge")
    return repeated[:n_times_full]


def plot_loo_isc_pearson_vs_spearman(
    label: str,
    loo_pearson: np.ndarray,
    mean_isc_pearson: np.ndarray,
    loo_spearman: np.ndarray,
    mean_isc_spearman: np.ndarray,
    *,
    plot_pct: int = 99,
    color_pearson: str = "steelblue",
    color_spearman: str = "darkorange",
    save_path_hist: Optional[Path] = None,
    save_path_violin: Optional[Path] = None,
) -> tuple[Figure, Figure]:
    """
    Section 1 broadband ISC: side-by-side Pearson/Spearman LOO-ISC histogram
    and per-subject violin+strip plot.

    :param label: Dataset label used in figure titles.
    :param loo_pearson: Pearson LOO-ISC array, shape ``(n_subjects, n_channels)``.
    :param mean_isc_pearson: Channel-wise mean Pearson LOO-ISC, shape ``(n_channels,)``.
    :param loo_spearman: Spearman LOO-ISC array, shape ``(n_subjects, n_channels_sub)``.
    :param mean_isc_spearman: Channel-wise mean Spearman LOO-ISC, shape ``(n_channels_sub,)``.
    :param plot_pct: Percentile used to clip histogram outliers.
    :param color_pearson: Colour for Pearson bars/lines.
    :param color_spearman: Colour for Spearman bars/lines.
    :param save_path_hist: Optional path to save the histogram figure.
    :param save_path_violin: Optional path to save the per-subject violin figure.
    :return: Tuple ``(fig_hist, fig_violin)``.
    """
    n_channels = mean_isc_pearson.shape[0]
    n_subjects = loo_pearson.shape[0]

    # ── Histogram: Pearson vs. Spearman side by side ──────────────────────
    fig_hist, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, mean_isc, color, method in zip(
        axes,
        [mean_isc_pearson, mean_isc_spearman],
        [color_pearson, color_spearman],
        ["Pearson", "Spearman"],
    ):
        clip = np.percentile(mean_isc, plot_pct)
        n_out = int((mean_isc > clip).sum())
        vals = mean_isc[mean_isc <= clip]
        counts, edges = np.histogram(vals, bins=30)
        n_channels_current = mean_isc.shape[0]
        pct_vals = counts / n_channels_current * 100
        centers = 0.5 * (edges[:-1] + edges[1:])
        _hist_colors = [color if c >= 0 else _C_NEG for c in centers]
        ax.bar(
            centers,
            pct_vals,
            width=(edges[1] - edges[0]) * 0.9,
            color=_hist_colors,
            alpha=0.75,
            edgecolor="none",
        )
        ax.axvline(
            np.mean(mean_isc),
            color=".2",
            ls="--",
            lw=1.3,
            label=f"Mean = {np.mean(mean_isc):.4f}",
        )
        ax.axvline(
            np.median(mean_isc),
            color="crimson",
            ls=":",
            lw=1.2,
            label=f"Median = {np.median(mean_isc):.4f}",
        )
        ax.axvline(0, color="gray", ls="-", lw=0.8, alpha=0.5, label="r = 0")
        ax.set_xlabel("LOO-ISC (r)")
        ax.set_ylabel("% of channels")
        ax.set_title(
            f"[{label}]  {method} LOO-ISC\n"
            f"({n_out} channel(s) clipped at {plot_pct}th pct)"
        )
        ax.legend(frameon=False, fontsize=9)
    sns.despine(fig=fig_hist)
    fig_hist.suptitle(
        f"{label} — Channel-wise mean LOO-ISC distribution", fontsize=13, y=1.02
    )
    fig_hist.tight_layout()
    _save_fig(fig_hist, save_path_hist)
    plt.show()

    # ── Per-subject violin + strip plot ──────────────────────────────────
    df_subj = pd.concat(
        [
            pd.DataFrame(
                {
                    "subject": np.arange(n_subjects),
                    "mean_isc": loo_pearson.mean(axis=1),
                    "method": "Pearson",
                }
            ),
            pd.DataFrame(
                {
                    "subject": np.arange(n_subjects),
                    "mean_isc": loo_spearman.mean(axis=1),
                    "method": "Spearman",
                }
            ),
        ]
    )
    fig_violin, ax2 = plt.subplots(figsize=(7, 4))
    sns.violinplot(
        data=df_subj,
        x="method",
        y="mean_isc",
        hue="method",
        palette={"Pearson": color_pearson, "Spearman": color_spearman},
        legend=False,
        inner="box",
        ax=ax2,
    )
    sns.stripplot(
        data=df_subj,
        x="method",
        y="mean_isc",
        color=".25",
        size=5,
        jitter=True,
        alpha=0.65,
        ax=ax2,
    )
    ax2.axhline(0, color="gray", ls="--", lw=0.8, label="r = 0")
    ax2.set_xlabel("Correlation method")
    ax2.set_ylabel("Mean LOO-ISC across channels (r)")
    ax2.set_title(f"[{label}]  Per-subject mean LOO-ISC (each dot = one subject)")
    ax2.legend(frameon=False, fontsize=9)
    sns.despine(fig=fig_violin)
    fig_violin.tight_layout()
    _save_fig(fig_violin, save_path_violin)
    plt.show()

    _logger.info(
        f"[{label}] Pearson  LOO-ISC — mean: {mean_isc_pearson.mean():.4f}  "
        f"% channels > 0: {(mean_isc_pearson > 0).mean() * 100:.1f}%"
    )
    _logger.info(
        f"[{label}] Spearman LOO-ISC — mean: {mean_isc_spearman.mean():.4f}  "
        f"% channels > 0: {(mean_isc_spearman > 0).mean() * 100:.1f}%"
    )

    return fig_hist, fig_violin


# ---------------------------------------------------------------------------
# 10. Broadband pairwise ISC: Pearson vs. Spearman heatmaps + distributions
# ---------------------------------------------------------------------------


def plot_pairwise_isc_pearson_vs_spearman(
    label: str,
    pair_pearson: np.ndarray,
    pair_spearman: np.ndarray,
    *,
    color_pearson: str = "steelblue",
    color_spearman: str = "darkorange",
    save_path_heatmaps: Optional[Path] = None,
    save_path_per_subject: Optional[Path] = None,
    save_path_distribution: Optional[Path] = None,
) -> tuple[Figure, Figure, Figure]:
    """
    Section 2 broadband ISC: side-by-side pairwise ISC heatmaps (Pearson vs.
    Spearman), per-subject mean off-diagonal bar chart, and off-diagonal
    distribution histogram.

    :param label: Dataset label used in figure titles.
    :param pair_pearson: Pearson pairwise ISC matrix, shape ``(n_subjects, n_subjects)``.
    :param pair_spearman: Spearman pairwise ISC matrix, shape ``(n_subjects, n_subjects)``.
    :param color_pearson: Colour for Pearson elements.
    :param color_spearman: Colour for Spearman elements.
    :param save_path_heatmaps: Optional path to save the heatmaps figure.
    :param save_path_per_subject: Optional path to save the per-subject bar figure.
    :param save_path_distribution: Optional path to save the distribution figure.
    :return: Tuple ``(fig_heatmaps, fig_per_subject, fig_distribution)``.
    """
    n_subjects = pair_pearson.shape[0]
    subj_labels = [f"S{i + 1:02d}" for i in range(n_subjects)]
    off_diag_mask = ~np.eye(n_subjects, dtype=bool)

    # ── Side-by-side heatmaps ─────────────────────────────────────────────
    fig_hm, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mat, method in zip(
        axes,
        [pair_pearson, pair_spearman],
        ["Pearson", "Spearman"],
    ):
        mat_display = mat.copy()
        np.fill_diagonal(mat_display, np.nan)
        vmax = max(float(np.nanmax(np.abs(mat_display))), 0.01)
        im = ax.imshow(mat_display, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="ISC (r)", shrink=0.82)
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels(subj_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(subj_labels, fontsize=7)
        mean_off = float(mat[off_diag_mask].mean())
        ax.set_title(
            f"[{label}]  {method} pairwise ISC\nmean off-diagonal r = {mean_off:.4f}"
        )
    sns.despine(fig=fig_hm, left=True, bottom=True)
    fig_hm.suptitle(f"{label} — Pairwise ISC matrices", fontsize=13)
    fig_hm.tight_layout()
    _save_fig(fig_hm, save_path_heatmaps)
    plt.show()

    # ── Per-subject mean off-diagonal ISC ─────────────────────────────────
    mean_off_p = np.array(
        [pair_pearson[i, off_diag_mask[i]].mean() for i in range(n_subjects)]
    )
    mean_off_s = np.array(
        [pair_spearman[i, off_diag_mask[i]].mean() for i in range(n_subjects)]
    )
    df_off = pd.DataFrame(
        {
            "subject": subj_labels * 2,
            "mean_pairwise_isc": np.concatenate([mean_off_p, mean_off_s]),
            "method": ["Pearson"] * n_subjects + ["Spearman"] * n_subjects,
        }
    )
    fig_subj, ax2 = plt.subplots(figsize=(max(8, n_subjects * 0.55), 4))
    sns.barplot(
        data=df_off,
        x="subject",
        y="mean_pairwise_isc",
        hue="method",
        palette={"Pearson": color_pearson, "Spearman": color_spearman},
        ax=ax2,
    )
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("Subject")
    ax2.set_ylabel("Mean pairwise ISC (r)")
    ax2.set_title(
        f"[{label}]  Per-subject mean pairwise ISC — low values = outlier candidate"
    )
    ax2.legend(frameon=False, fontsize=9, title="Method")
    sns.despine(fig=fig_subj)
    fig_subj.tight_layout()
    _save_fig(fig_subj, save_path_per_subject)
    plt.show()

    # ── Off-diagonal distribution ─────────────────────────────────────────
    off_p = pair_pearson[off_diag_mask]
    off_s = pair_spearman[off_diag_mask]
    n_pairs = n_subjects * (n_subjects - 1) // 2
    fig_dist, ax3 = plt.subplots(figsize=(8, 4))
    ax3.hist(
        off_p,
        bins=20,
        alpha=0.55,
        color=color_pearson,
        label=f"Pearson  (mean = {off_p.mean():.4f})",
    )
    ax3.hist(
        off_s,
        bins=20,
        alpha=0.55,
        color=color_spearman,
        label=f"Spearman (mean = {off_s.mean():.4f})",
    )
    ax3.axvline(off_p.mean(), color=color_pearson, ls="--", lw=1.2)
    ax3.axvline(off_s.mean(), color=color_spearman, ls="--", lw=1.2)
    ax3.axvline(0, color="gray", ls="-", lw=0.8, alpha=0.5, label="r = 0")
    ax3.set_xlabel("Pairwise ISC (r)")
    ax3.set_ylabel("Number of subject pairs")
    ax3.set_title(
        f"[{label}]  Off-diagonal pairwise ISC distribution ({n_pairs} pairs)"
    )
    ax3.legend(frameon=False)
    sns.despine(fig=fig_dist)
    fig_dist.tight_layout()
    _save_fig(fig_dist, save_path_distribution)
    plt.show()

    return fig_hm, fig_subj, fig_dist


# ---------------------------------------------------------------------------
# 11. Multi-scale sliding-window ISC: bar + 3-scale overlay + Pearson/Spearman
# ---------------------------------------------------------------------------


def plot_multiscale_sliding_window_isc(
    label: str,
    isc_fine: np.ndarray,
    times_fine: np.ndarray,
    isc_med: np.ndarray,
    times_med: np.ndarray,
    isc_large: np.ndarray,
    times_large: np.ndarray,
    isc_spearman_med: np.ndarray,
    sfreq: float,
    n_times: int,
    *,
    window_fine_sec: float = 1.0,
    window_med_sec: float = 5.0,
    window_large_sec: float = 15.0,
    isc_threshold: float = 0.035,
    n_ch_subsample: Optional[int] = None,
    save_path_bar: Optional[Path] = None,
    save_path_overlay: Optional[Path] = None,
    save_path_comparison: Optional[Path] = None,
) -> tuple[Figure, Figure, Figure]:
    """
    Section 3 broadband ISC: three figures — per-window bar chart (medium
    window), 3-scale stair-step overlay with per-channel heatmap, and
    Pearson vs. Spearman comparison at medium window.

    :param label: Dataset label used in figure titles.
    :param isc_fine: Fine-window LOO-ISC, shape ``(n_windows_fine, n_channels)``.
    :param times_fine: Fine-window centre times in seconds, shape ``(n_windows_fine,)``.
    :param isc_med: Medium-window LOO-ISC, shape ``(n_windows_med, n_channels)``.
    :param times_med: Medium-window centre times in seconds, shape ``(n_windows_med,)``.
    :param isc_large: Large-window LOO-ISC, shape ``(n_windows_large, n_channels)``.
    :param times_large: Large-window centre times in seconds, shape ``(n_windows_large,)``.
    :param isc_spearman_med: Spearman LOO-ISC at medium window,
        shape ``(n_windows_med, n_channels_sub)``.
    :param sfreq: Sampling frequency in Hz (used for native-resolution step arrays).
    :param n_times: Total number of time samples in the original recording.
    :param window_fine_sec: Fine window length (seconds).
    :param window_med_sec: Medium window length (seconds).
    :param window_large_sec: Large window length (seconds).
    :param isc_threshold: Significance highlight threshold (r).
    :param n_ch_subsample: Number of subsampled channels (for axis label only).
    :param save_path_bar: Optional path to save the bar-chart figure.
    :param save_path_overlay: Optional path to save the overlay+heatmap figure.
    :param save_path_comparison: Optional path to save the Pearson/Spearman figure.
    :return: Tuple ``(fig_bar, fig_overlay, fig_comparison)``.
    """
    time_s = np.arange(n_times) / sfreq
    ds = max(1, n_times // 8000)
    t_ds = time_s[::ds]

    step_fine = int(round(window_fine_sec / 2 * sfreq))
    step_med = int(round(window_med_sec / 2 * sfreq))
    step_large = int(round(window_large_sec / 2 * sfreq))

    mean_fine = isc_fine.mean(axis=1)
    mean_med = isc_med.mean(axis=1)
    mean_large = isc_large.mean(axis=1)
    mean_spear_med = isc_spearman_med.mean(axis=1)

    step_fine_arr = _make_step_arr(mean_fine, step_fine, n_times)
    step_med_arr = _make_step_arr(mean_med, step_med, n_times)
    step_large_arr = _make_step_arr(mean_large, step_large, n_times)
    step_spear_arr = _make_step_arr(mean_spear_med, step_med, n_times)

    sig_mask_med = mean_med > isc_threshold

    # ── Figure A — per-window bar chart (medium window, Pearson) ─────────
    win_centers_s = times_med
    std_per_win = isc_med.std(axis=1)
    bar_w = window_med_sec / 2 * 0.9
    bar_colors = [
        "#5cb85c" if v > isc_threshold else (_C_NEG if v < 0 else "steelblue")
        for v in mean_med
    ]

    fig_bar, ax_bar = plt.subplots(figsize=(max(14, len(win_centers_s) * 0.28), 4))
    ax_bar.bar(
        win_centers_s,
        mean_med,
        width=bar_w,
        color=bar_colors,
        alpha=0.85,
        edgecolor="none",
    )
    ax_bar.errorbar(
        win_centers_s,
        mean_med,
        yerr=std_per_win,
        fmt="none",
        color=".35",
        capsize=2,
        linewidth=0.9,
        label="±std across channels",
    )
    ax_bar.axhline(
        isc_threshold,
        color=_C_THRESH,
        ls="--",
        lw=1.2,
        label=f"Threshold r = {isc_threshold}",
    )
    ax_bar.axhline(
        float(mean_med.mean()),
        color=".2",
        ls=":",
        lw=1.1,
        label=f"Grand mean r = {mean_med.mean():.4f}",
    )
    ax_bar.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
    bar_patches = [
        Patch(
            color="#5cb85c",
            alpha=0.85,
            label=f"Sig. positive (r > {isc_threshold})",
        ),
        Patch(color="steelblue", alpha=0.85, label="Positive (r ≥ 0)"),
        Patch(color=_C_NEG, alpha=0.85, label="Negative (r < 0)"),
    ]
    extra_h, _ = ax_bar.get_legend_handles_labels()
    ax_bar.legend(
        handles=bar_patches + extra_h,
        labels=[p.get_label() for p in bar_patches] + [h.get_label() for h in extra_h],
        frameon=False,
        fontsize=9,
    )
    ax_bar.set_xlabel("Time (s)")
    ax_bar.set_ylabel("Mean LOO-ISC (r)")
    ax_bar.set_title(
        f"[{label}]  Per-window mean LOO-ISC  "
        f"(Pearson,  window = {window_med_sec:.0f} s,  "
        f"step = {window_med_sec / 2:.1f} s)"
    )
    sns.despine(fig=fig_bar)
    fig_bar.tight_layout()
    _save_fig(fig_bar, save_path_bar)
    plt.show()

    # Log significant intervals (medium window, Pearson, in seconds)
    step_half_sec = window_med_sec / 2
    sig_idx = np.where(sig_mask_med)[0].tolist()
    intervals: list[tuple[float, float, float, float]] = []
    for _, grp in groupby(enumerate(sig_idx), lambda x: x[1] - x[0]):
        run = list(map(itemgetter(1), grp))
        tb = float(times_med[run[0]]) - step_half_sec
        te = float(times_med[run[-1]]) + step_half_sec
        intervals.append(
            (tb, te, float(mean_med[run].mean()), float(mean_med[run].max()))
        )
    n_sig = int(sig_mask_med.sum())
    _logger.info(
        f"[{label}] Significant windows ({window_med_sec:.0f} s): "
        f"{n_sig}/{len(sig_mask_med)} ({100 * n_sig / len(sig_mask_med):.1f}%)  "
        f"threshold r = {isc_threshold}"
    )
    for k, (tb, te, mr, pr) in enumerate(intervals, 1):
        _logger.info(
            f"  {k:>3}  start={tb:>8.1f} s  end={te:>8.1f} s  "
            f"mean_r={mr:.4f}  peak_r={pr:.4f}"
        )

    # ── Figure B — 3-scale stair-step overlay + per-channel heatmap ──────
    n_ch = isc_fine.shape[1]
    ch_label_sw = (
        f"Channel index (subsampled {n_ch_subsample})"
        if n_ch_subsample is not None and n_ch_subsample < n_ch
        else "Channel index"
    )

    fig_ov = plt.figure(figsize=(16, 10))
    gs_ov = gridspec.GridSpec(
        2,
        2,
        height_ratios=[2, 3],
        width_ratios=[30, 1],
        hspace=0.10,
        wspace=0.04,
        figure=fig_ov,
    )
    ax_isc = fig_ov.add_subplot(gs_ov[0, 0])
    ax_hm = fig_ov.add_subplot(gs_ov[1, 0], sharex=ax_isc)
    cax = fig_ov.add_subplot(gs_ov[1, 1])

    first_span = True
    for w in np.where(sig_mask_med)[0]:
        t_start = float(times_med[w]) - window_med_sec / 2
        t_end = float(times_med[w]) + window_med_sec / 2
        for _ax in (ax_isc, ax_hm):
            _ax.axvspan(
                t_start,
                t_end,
                color="gold",
                alpha=0.22,
                label=(
                    "Sig. window (med)"
                    if (first_span and _ax is ax_isc)
                    else "_nolegend_"
                ),
            )
        first_span = False

    win_configs_plot = [
        ("fine", step_fine_arr, "lightsteelblue", "solid", 1.0, window_fine_sec),
        ("med", step_med_arr, "steelblue", "solid", 2.2, window_med_sec),
        ("large", step_large_arr, "saddlebrown", "dashed", 2.5, window_large_sec),
    ]
    label_map = {"fine": "Fine", "med": "Medium", "large": "Large"}
    for name, step_arr, color, ls, lw, win_sec in win_configs_plot:
        ax_isc.step(
            t_ds,
            step_arr[::ds],
            where="post",
            color=color,
            lw=lw,
            ls=ls,
            label=f"{label_map[name]} ({win_sec:.0f} s / {win_sec / 2:.1f} s step)",
        )

    _med_step_ds = step_med_arr[::ds]
    ax_isc.fill_between(
        t_ds,
        0,
        np.clip(_med_step_ds, 0, None),
        step="post",
        alpha=0.15,
        color="steelblue",
        label="_nolegend_",
    )
    ax_isc.fill_between(
        t_ds,
        np.clip(_med_step_ds, None, 0),
        0,
        step="post",
        alpha=0.25,
        color=_C_NEG,
        label="_nolegend_",
    )
    ax_isc.axhline(
        isc_threshold,
        color=_C_THRESH,
        ls="--",
        lw=1.1,
        label=f"Threshold r = {isc_threshold}",
    )
    ax_isc.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
    ax_isc.set_ylabel("Mean LOO-ISC (r)")
    ax_isc.set_title(
        f"[{label}]  Time-resolved LOO-ISC — 3-scale stair-step + heatmap  (Pearson)"
    )
    ax_isc.legend(frameon=False, fontsize=9)
    plt.setp(ax_isc.get_xticklabels(), visible=False)

    _dt_fine = times_fine[1] - times_fine[0] if len(times_fine) > 1 else window_fine_sec
    _t_edges = np.r_[times_fine - _dt_fine / 2, times_fine[-1] + _dt_fine / 2]
    _ch_edges = np.arange(n_ch + 1)
    vmax_hm = max(float(np.nanpercentile(np.abs(isc_fine), 98)), 1e-6)
    pcm = ax_hm.pcolormesh(
        _t_edges,
        _ch_edges,
        isc_fine.T,
        cmap="RdBu_r",
        vmin=-vmax_hm,
        vmax=vmax_hm,
        rasterized=True,
        shading="flat",
    )
    ax_hm.set_ylim(n_ch, 0)
    fig_ov.colorbar(pcm, cax=cax, label="LOO-ISC (r)")
    ax_hm.set_xlabel("Time (s)")
    ax_hm.set_ylabel(ch_label_sw)
    sns.despine(fig=fig_ov, left=False, bottom=False)
    fig_ov.tight_layout()
    _save_fig(fig_ov, save_path_overlay)
    plt.show()

    # ── Figure C — Pearson vs. Spearman comparison (medium window) ────────
    fig_cmp, ax_cmp = plt.subplots(figsize=(14, 4))
    ax_cmp.step(
        t_ds,
        step_med_arr[::ds],
        where="post",
        color="steelblue",
        lw=2.0,
        label=f"Pearson  ({window_med_sec:.0f} s / {window_med_sec / 2:.1f} s step)",
    )
    ax_cmp.step(
        t_ds,
        step_spear_arr[::ds],
        where="post",
        color="darkorange",
        lw=2.0,
        ls="--",
        label=f"Spearman ({window_med_sec:.0f} s / {window_med_sec / 2:.1f} s step)",
    )
    ax_cmp.fill_between(
        t_ds,
        step_med_arr[::ds],
        step_spear_arr[::ds],
        alpha=0.12,
        color="gray",
        step="post",
        label="Pearson − Spearman gap",
    )
    _pearson_ds = step_med_arr[::ds]
    _spear_ds = step_spear_arr[::ds]
    ax_cmp.fill_between(
        t_ds,
        0,
        _pearson_ds,
        where=_pearson_ds >= 0,
        step="post",
        alpha=0.10,
        color="steelblue",
    )
    ax_cmp.fill_between(
        t_ds,
        0,
        _pearson_ds,
        where=_pearson_ds < 0,
        step="post",
        alpha=0.18,
        color=_C_NEG,
    )
    ax_cmp.fill_between(
        t_ds,
        0,
        _spear_ds,
        where=_spear_ds >= 0,
        step="post",
        alpha=0.08,
        color="darkorange",
    )
    ax_cmp.fill_between(
        t_ds,
        0,
        _spear_ds,
        where=_spear_ds < 0,
        step="post",
        alpha=0.14,
        color=_C_NEG,
    )
    ax_cmp.axhline(
        isc_threshold,
        color=_C_THRESH,
        ls="--",
        lw=1.0,
        label=f"Threshold r = {isc_threshold}",
    )
    ax_cmp.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
    ax_cmp.set_xlabel("Time (s)")
    ax_cmp.set_ylabel("Mean LOO-ISC (r)")
    ax_cmp.set_title(
        f"[{label}]  Pearson vs. Spearman  ({window_med_sec:.0f} s / 50% overlap)"
    )
    ax_cmp.legend(frameon=False, fontsize=9)
    sns.despine(fig=fig_cmp)
    fig_cmp.tight_layout()
    _save_fig(fig_cmp, save_path_comparison)
    plt.show()

    n_len = min(len(mean_med), len(mean_spear_med))
    corr_between = float(np.corrcoef(mean_med[:n_len], mean_spear_med[:n_len])[0, 1])
    _logger.info(
        f"[{label}] Pearson vs. Spearman ISC time course correlation: r = {corr_between:.4f}"
    )

    return fig_bar, fig_ov, fig_cmp


# ---------------------------------------------------------------------------
# 12. Per-band LOO-ISC: Pearson vs. Spearman histograms + per-subject violins
# ---------------------------------------------------------------------------


def plot_band_loo_isc_pearson_vs_spearman(
    label: str,
    band_iscs: dict[str, tuple[np.ndarray, np.ndarray]],
    band_iscs_spearman: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    plot_pct: int = 99,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    band_colors: Optional[dict[str, str]] = None,
    save_path_dir: Optional[Path] = None,
) -> dict[str, tuple[Figure, Figure]]:
    """
    Section 1 per-band ISC: for each frequency band, produce the Pearson vs.
    Spearman LOO-ISC histogram (side-by-side) and per-subject violin+strip plot.

    :param label: Dataset label used in figure titles.
    :param band_iscs: ``{band: (loo_pearson, mean_pearson)}`` — Pearson LOO-ISC
        per band.  ``loo_pearson`` shape ``(n_subjects, n_channels)``.
    :param band_iscs_spearman: ``{band: (loo_spearman, mean_spearman)}`` — Spearman
        LOO-ISC per band (may use subsampled channels).
    :param plot_pct: Percentile used to clip histogram outliers.
    :param bands: Band definitions for axis labels.
    :param band_colors: Optional ``{band_name: colour}`` mapping (Pearson colour).
    :param save_path_dir: Directory in which to save per-band figures.  When
        provided, saves ``loo_isc_distribution_{band}_{label}.png`` and
        ``loo_isc_per_subject_{band}_{label}.png`` in this directory.
    :return: ``{band: (fig_hist, fig_violin)}``.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _BAND_COLORS
    n_subjects = next(iter(band_iscs.values()))[0].shape[0]

    results: dict[str, tuple[Figure, Figure]] = {}

    for band in bands:
        loo_p, mean_p = band_iscs[band]
        loo_s, mean_s = band_iscs_spearman[band]
        color = band_colors.get(band, "steelblue")

        # ── Histogram ───────────────────────────────────────────────────
        fig_hist, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, mean_isc, col, method in zip(
            axes,
            [mean_p, mean_s],
            [color, "darkorange"],
            ["Pearson", "Spearman"],
        ):
            clip = np.percentile(mean_isc, plot_pct)
            n_out = int((mean_isc > clip).sum())
            vals = mean_isc[mean_isc <= clip]
            counts, edges = np.histogram(vals, bins=30)
            pct_vals = counts / len(mean_isc) * 100
            centers = 0.5 * (edges[:-1] + edges[1:])
            _hist_colors = [col if c >= 0 else _C_NEG for c in centers]
            ax.bar(
                centers,
                pct_vals,
                width=(edges[1] - edges[0]) * 0.9,
                color=_hist_colors,
                alpha=0.75,
                edgecolor="none",
            )
            ax.axvline(
                np.mean(mean_isc),
                color=".2",
                ls="--",
                lw=1.3,
                label=f"Mean = {np.mean(mean_isc):.4f}",
            )
            ax.axvline(
                np.median(mean_isc),
                color="crimson",
                ls=":",
                lw=1.2,
                label=f"Median = {np.median(mean_isc):.4f}",
            )
            ax.axvline(0, color="gray", ls="-", lw=0.8, alpha=0.5, label="r = 0")
            l_freq, h_freq = bands[band]
            thr = _resolve_band_thresholds([band], 0.035)[band]
            ax.axvline(
                thr,
                color="tomato",
                ls="--",
                lw=0.9,
                alpha=0.6,
                label=f"Threshold = {thr}",
            )
            ax.set_xlabel("LOO-ISC (r)")
            ax.set_ylabel("% of channels")
            ax.set_title(
                f"[{label} / {band.upper()}]  {method} LOO-ISC\n"
                f"({n_out} channel(s) clipped at {plot_pct}th pct)"
            )
            ax.legend(frameon=False, fontsize=9)
        sns.despine(fig=fig_hist)
        fig_hist.suptitle(
            f"{label} / {band.upper()} — Channel-wise mean LOO-ISC distribution",
            fontsize=13,
            y=1.02,
        )
        fig_hist.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_hist,
                Path(save_path_dir) / f"loo_isc_distribution_{band}_{label}.png",
            )
        plt.show()

        # ── Per-subject violin + strip ────────────────────────────────────
        df_subj = pd.concat(
            [
                pd.DataFrame(
                    {
                        "subject": np.arange(n_subjects),
                        "mean_isc": loo_p.mean(axis=1),
                        "method": "Pearson",
                    }
                ),
                pd.DataFrame(
                    {
                        "subject": np.arange(n_subjects),
                        "mean_isc": loo_s.mean(axis=1),
                        "method": "Spearman",
                    }
                ),
            ]
        )
        fig_violin, ax2 = plt.subplots(figsize=(7, 4))
        sns.violinplot(
            data=df_subj,
            x="method",
            y="mean_isc",
            hue="method",
            palette={"Pearson": color, "Spearman": "darkorange"},
            legend=False,
            inner="box",
            ax=ax2,
        )
        sns.stripplot(
            data=df_subj,
            x="method",
            y="mean_isc",
            color=".25",
            size=5,
            jitter=True,
            alpha=0.65,
            ax=ax2,
        )
        ax2.axhline(0, color="gray", ls="--", lw=0.8, label="r = 0")
        ax2.set_xlabel("Correlation method")
        ax2.set_ylabel("Mean LOO-ISC across channels (r)")
        ax2.set_title(f"[{label} / {band.upper()}]  Per-subject mean LOO-ISC")
        ax2.legend(frameon=False, fontsize=9)
        sns.despine(fig=fig_violin)
        fig_violin.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_violin,
                Path(save_path_dir) / f"loo_isc_per_subject_{band}_{label}.png",
            )
        plt.show()

        results[band] = (fig_hist, fig_violin)

    return results


# ---------------------------------------------------------------------------
# 13. Per-band pairwise ISC: Pearson vs. Spearman heatmaps + distributions
# ---------------------------------------------------------------------------


def plot_band_pairwise_isc_pearson_vs_spearman(
    label: str,
    band_pair_pearson: dict[str, np.ndarray],
    band_pair_spearman: dict[str, np.ndarray],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    band_colors: Optional[dict[str, str]] = None,
    save_path_dir: Optional[Path] = None,
) -> dict[str, tuple[Figure, Figure, Figure]]:
    """
    Section 2 per-band ISC: for each frequency band, produce side-by-side
    Pearson/Spearman pairwise ISC heatmaps, per-subject mean off-diagonal bar
    chart, and off-diagonal distribution histogram.

    :param label: Dataset label used in figure titles.
    :param band_pair_pearson: ``{band: array(n_subjects, n_subjects)}`` Pearson pairwise ISC.
    :param band_pair_spearman: ``{band: array(n_subjects, n_subjects)}`` Spearman pairwise ISC.
    :param bands: Band definitions for axis labels.
    :param band_colors: Optional ``{band_name: colour}`` mapping (Pearson colour).
    :param save_path_dir: Directory in which to save per-band figures.
    :return: ``{band: (fig_heatmaps, fig_per_subject, fig_distribution)}``.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _BAND_COLORS

    results: dict[str, tuple[Figure, Figure, Figure]] = {}

    for band in bands:
        pair_p = band_pair_pearson[band]
        pair_s = band_pair_spearman[band]
        n_subjects = pair_p.shape[0]
        subj_labels = [f"S{i + 1:02d}" for i in range(n_subjects)]
        off_diag_mask = ~np.eye(n_subjects, dtype=bool)
        color = band_colors.get(band, "steelblue")

        # ── Side-by-side heatmaps ─────────────────────────────────────────
        fig_hm, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, mat, method in zip(axes, [pair_p, pair_s], ["Pearson", "Spearman"]):
            mat_display = mat.copy()
            np.fill_diagonal(mat_display, np.nan)
            vmax = max(float(np.nanmax(np.abs(mat_display))), 0.01)
            im = ax.imshow(
                mat_display,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )
            plt.colorbar(im, ax=ax, label="ISC (r)", shrink=0.82)
            ax.set_xticks(range(n_subjects))
            ax.set_yticks(range(n_subjects))
            ax.set_xticklabels(subj_labels, rotation=90, fontsize=7)
            ax.set_yticklabels(subj_labels, fontsize=7)
            mean_off = float(mat[off_diag_mask].mean())
            ax.set_title(
                f"[{label} / {band.upper()}]  {method} pairwise ISC\n"
                f"mean off-diagonal r = {mean_off:.4f}"
            )
        sns.despine(fig=fig_hm, left=True, bottom=True)
        fig_hm.suptitle(
            f"{label} / {band.upper()} — Pairwise ISC matrices", fontsize=13
        )
        fig_hm.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_hm,
                Path(save_path_dir) / f"pairwise_isc_matrix_{band}_{label}.png",
            )
        plt.show()

        # ── Per-subject mean off-diagonal ─────────────────────────────────
        mean_off_p = np.array(
            [pair_p[i, off_diag_mask[i]].mean() for i in range(n_subjects)]
        )
        mean_off_s = np.array(
            [pair_s[i, off_diag_mask[i]].mean() for i in range(n_subjects)]
        )
        df_off = pd.DataFrame(
            {
                "subject": subj_labels * 2,
                "mean_pairwise_isc": np.concatenate([mean_off_p, mean_off_s]),
                "method": ["Pearson"] * n_subjects + ["Spearman"] * n_subjects,
            }
        )
        fig_subj, ax2 = plt.subplots(figsize=(max(8, n_subjects * 0.55), 4))
        sns.barplot(
            data=df_off,
            x="subject",
            y="mean_pairwise_isc",
            hue="method",
            palette={"Pearson": color, "Spearman": "darkorange"},
            ax=ax2,
        )
        ax2.axhline(0, color="gray", ls="--", lw=0.8)
        ax2.set_xlabel("Subject")
        ax2.set_ylabel("Mean pairwise ISC (r)")
        ax2.set_title(
            f"[{label} / {band.upper()}]  Per-subject mean pairwise ISC"
            " — low = outlier candidate"
        )
        ax2.legend(frameon=False, fontsize=9, title="Method")
        sns.despine(fig=fig_subj)
        fig_subj.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_subj,
                Path(save_path_dir) / f"pairwise_isc_per_subject_{band}_{label}.png",
            )
        plt.show()

        # ── Off-diagonal distribution ─────────────────────────────────────
        off_p = pair_p[off_diag_mask]
        off_s = pair_s[off_diag_mask]
        n_pairs = n_subjects * (n_subjects - 1) // 2
        fig_dist, ax3 = plt.subplots(figsize=(8, 4))
        ax3.hist(
            off_p,
            bins=20,
            alpha=0.55,
            color=color,
            label=f"Pearson  (mean = {off_p.mean():.4f})",
        )
        ax3.hist(
            off_s,
            bins=20,
            alpha=0.55,
            color="darkorange",
            label=f"Spearman (mean = {off_s.mean():.4f})",
        )
        ax3.axvline(off_p.mean(), color=color, ls="--", lw=1.2)
        ax3.axvline(off_s.mean(), color="darkorange", ls="--", lw=1.2)
        ax3.axvline(0, color="gray", ls="-", lw=0.8, alpha=0.5, label="r = 0")
        ax3.set_xlabel("Pairwise ISC (r)")
        ax3.set_ylabel("Number of subject pairs")
        ax3.set_title(
            f"[{label} / {band.upper()}]  Off-diagonal pairwise ISC distribution"
            f" ({n_pairs} pairs)"
        )
        ax3.legend(frameon=False)
        sns.despine(fig=fig_dist)
        fig_dist.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_dist,
                Path(save_path_dir) / f"pairwise_isc_distribution_{band}_{label}.png",
            )
        plt.show()

        results[band] = (fig_hm, fig_subj, fig_dist)

    return results


# ---------------------------------------------------------------------------
# 14. Per-band multi-scale sliding-window ISC
# ---------------------------------------------------------------------------


def plot_band_multiscale_sliding_window_isc(
    label: str,
    band_sw_fine: dict[str, tuple[np.ndarray, np.ndarray]],
    band_sw_med: dict[str, tuple[np.ndarray, np.ndarray]],
    band_sw_large: dict[str, tuple[np.ndarray, np.ndarray]],
    band_sw_spearman_med: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float,
    n_times: int,
    *,
    window_fine_sec: float = 1.0,
    window_med_sec: float = 5.0,
    window_large_sec: float = 15.0,
    band_thresholds: Optional[dict[str, float]] = None,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    band_colors: Optional[dict[str, str]] = None,
    n_ch_subsample: Optional[int] = None,
    save_path_dir: Optional[Path] = None,
) -> dict[str, tuple[Figure, Figure, Figure]]:
    """
    Section 3 per-band ISC: for each frequency band, produce the bar chart
    (medium window, Pearson), 3-scale stair-step overlay with per-channel
    heatmap, and Pearson vs. Spearman comparison.

    :param label: Dataset label used in figure titles.
    :param band_sw_fine: ``{band: (isc_tc, times)}`` for the fine window.
    :param band_sw_med: ``{band: (isc_tc, times)}`` for the medium window.
    :param band_sw_large: ``{band: (isc_tc, times)}`` for the large window.
    :param band_sw_spearman_med: ``{band: (isc_tc, times)}`` Spearman at medium window.
    :param sfreq: Sampling frequency in Hz.
    :param n_times: Total number of time samples in the original recording.
    :param window_fine_sec: Fine window length (seconds).
    :param window_med_sec: Medium window length (seconds).
    :param window_large_sec: Large window length (seconds).
    :param band_thresholds: Per-band ISC significance thresholds.
    :param bands: Band definitions for axis labels.
    :param band_colors: Optional ``{band_name: colour}`` mapping (Pearson colour).
    :param n_ch_subsample: Number of subsampled channels (for axis labels).
    :param save_path_dir: Directory in which to save per-band figures.
    :return: ``{band: (fig_bar, fig_overlay, fig_comparison)}``.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _BAND_COLORS
    if band_thresholds is None:
        band_thresholds = {}

    time_s = np.arange(n_times) / sfreq
    ds = max(1, n_times // 8000)
    t_ds = time_s[::ds]

    step_fine_samp = int(round(window_fine_sec / 2 * sfreq))
    step_med_samp = int(round(window_med_sec / 2 * sfreq))
    step_large_samp = int(round(window_large_sec / 2 * sfreq))

    results: dict[str, tuple[Figure, Figure, Figure]] = {}

    for band in bands:
        color_band = band_colors.get(band, "steelblue")
        thr_band = band_thresholds.get(band, 0.035)

        isc_fine, times_fine = band_sw_fine[band]
        isc_med, times_med = band_sw_med[band]
        isc_large, _ = band_sw_large[band]
        isc_sp_med, _ = band_sw_spearman_med[band]

        mean_fine = isc_fine.mean(axis=1)
        mean_med = isc_med.mean(axis=1)
        mean_large = isc_large.mean(axis=1)
        mean_sp = isc_sp_med.mean(axis=1)

        step_fine_arr = _make_step_arr(mean_fine, step_fine_samp, n_times)
        step_med_arr = _make_step_arr(mean_med, step_med_samp, n_times)
        step_large_arr = _make_step_arr(mean_large, step_large_samp, n_times)
        step_spear_arr = _make_step_arr(mean_sp, step_med_samp, n_times)

        sig_mask_med = mean_med > thr_band
        n_ch = isc_fine.shape[1]

        # ── Figure A — bar chart (medium window, Pearson) ──────────────────
        win_centers_s = times_med
        std_per_win = isc_med.std(axis=1)
        bar_w = window_med_sec / 2 * 0.9
        bar_colors = [
            "#5cb85c" if v > thr_band else (_C_NEG if v < 0 else color_band)
            for v in mean_med
        ]

        fig_bar, ax_bar = plt.subplots(figsize=(max(14, len(win_centers_s) * 0.28), 4))
        ax_bar.bar(
            win_centers_s,
            mean_med,
            width=bar_w,
            color=bar_colors,
            alpha=0.85,
            edgecolor="none",
        )
        ax_bar.errorbar(
            win_centers_s,
            mean_med,
            yerr=std_per_win,
            fmt="none",
            color=".35",
            capsize=2,
            linewidth=0.9,
            label="±std across channels",
        )
        ax_bar.axhline(
            thr_band,
            color=_C_THRESH,
            ls="--",
            lw=1.2,
            label=f"Threshold r = {thr_band}",
        )
        ax_bar.axhline(
            float(mean_med.mean()),
            color=".2",
            ls=":",
            lw=1.1,
            label=f"Grand mean r = {mean_med.mean():.4f}",
        )
        ax_bar.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
        bar_patches = [
            Patch(
                color="#5cb85c",
                alpha=0.85,
                label=f"Sig. positive (r > {thr_band})",
            ),
            Patch(color=color_band, alpha=0.85, label="Positive (r ≥ 0)"),
            Patch(color=_C_NEG, alpha=0.85, label="Negative (r < 0)"),
        ]
        extra_h, _ = ax_bar.get_legend_handles_labels()
        ax_bar.legend(
            handles=bar_patches + extra_h,
            labels=[p.get_label() for p in bar_patches]
            + [h.get_label() for h in extra_h],
            frameon=False,
            fontsize=9,
        )
        ax_bar.set_xlabel("Time (s)")
        ax_bar.set_ylabel("Mean LOO-ISC (r)")
        l_freq, h_freq = bands[band]
        ax_bar.set_title(
            f"[{label} / {band.upper()} ({l_freq}-{h_freq} Hz)]  "
            f"Per-window mean LOO-ISC  "
            f"(Pearson,  window = {window_med_sec:.0f} s,  "
            f"step = {window_med_sec / 2:.1f} s)",
        )
        sns.despine(fig=fig_bar)
        fig_bar.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_bar,
                Path(save_path_dir) / f"sw_isc_bar_{band}_{label}.png",
            )
        plt.show()

        # Log significant intervals
        step_half_sec_band = window_med_sec / 2
        sig_idx = np.where(sig_mask_med)[0].tolist()
        intervals: list[tuple[float, float, float, float]] = []
        for _, grp in groupby(enumerate(sig_idx), lambda x: x[1] - x[0]):
            run = list(map(itemgetter(1), grp))
            tb = float(times_med[run[0]]) - step_half_sec_band
            te = float(times_med[run[-1]]) + step_half_sec_band
            intervals.append(
                (tb, te, float(mean_med[run].mean()), float(mean_med[run].max()))
            )
        n_sig = int(sig_mask_med.sum())
        _logger.info(
            f"[{label} / {band.upper()}] Significant windows ({window_med_sec:.0f} s): "
            f"{n_sig}/{len(sig_mask_med)}  threshold r = {thr_band}"
        )
        for k, (tb, te, mr, pr) in enumerate(intervals, 1):
            _logger.info(
                f"  {k:>3}  start={tb:>8.1f} s  end={te:>8.1f} s  "
                f"mean_r={mr:.4f}  peak_r={pr:.4f}"
            )

        # ── Figure B — 3-scale stair-step overlay + heatmap ──────────────
        ch_label_sw = (
            f"Channel index (subsampled {n_ch_subsample})"
            if n_ch_subsample is not None and n_ch_subsample < n_ch
            else "Channel index"
        )

        fig_ov = plt.figure(figsize=(16, 10))
        gs_ov = gridspec.GridSpec(
            2,
            2,
            height_ratios=[2, 3],
            width_ratios=[30, 1],
            hspace=0.10,
            wspace=0.04,
            figure=fig_ov,
        )
        ax_isc = fig_ov.add_subplot(gs_ov[0, 0])
        ax_hm = fig_ov.add_subplot(gs_ov[1, 0], sharex=ax_isc)
        cax_hm = fig_ov.add_subplot(gs_ov[1, 1])

        first_span = True
        for w in np.where(sig_mask_med)[0]:
            t_start = float(times_med[w]) - window_med_sec / 2
            t_end = float(times_med[w]) + window_med_sec / 2
            for _ax in (ax_isc, ax_hm):
                _ax.axvspan(
                    t_start,
                    t_end,
                    color="gold",
                    alpha=0.22,
                    label=(
                        "Sig. window (med)"
                        if (first_span and _ax is ax_isc)
                        else "_nolegend_"
                    ),
                )
            first_span = False

        win_configs_plot = [
            ("fine", step_fine_arr, "lightsteelblue", "solid", 1.0, window_fine_sec),
            ("med", step_med_arr, color_band, "solid", 2.2, window_med_sec),
            ("large", step_large_arr, "saddlebrown", "dashed", 2.5, window_large_sec),
        ]
        label_map = {"fine": "Fine", "med": "Medium", "large": "Large"}
        for name, step_arr, color_c, ls, lw, win_sec in win_configs_plot:
            ax_isc.step(
                t_ds,
                step_arr[::ds],
                where="post",
                color=color_c,
                lw=lw,
                ls=ls,
                label=f"{label_map[name]} ({win_sec:.0f} s / {win_sec / 2:.1f} s step)",
            )

        _med_step_ds = step_med_arr[::ds]
        ax_isc.fill_between(
            t_ds,
            0,
            np.clip(_med_step_ds, 0, None),
            step="post",
            alpha=0.15,
            color=color_band,
        )
        ax_isc.fill_between(
            t_ds,
            np.clip(_med_step_ds, None, 0),
            0,
            step="post",
            alpha=0.25,
            color=_C_NEG,
        )
        ax_isc.axhline(
            thr_band,
            color=_C_THRESH,
            ls="--",
            lw=1.1,
            label=f"Threshold r = {thr_band}",
        )
        ax_isc.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
        ax_isc.set_ylabel("Mean LOO-ISC (r)")
        ax_isc.set_title(
            f"[{label} / {band.upper()}]  Time-resolved LOO-ISC"
            " — 3-scale stair-step + heatmap  (Pearson)"
        )
        ax_isc.legend(frameon=False, fontsize=9)
        plt.setp(ax_isc.get_xticklabels(), visible=False)

        _dt_fine = (
            times_fine[1] - times_fine[0] if len(times_fine) > 1 else window_fine_sec
        )
        _t_edges = np.r_[times_fine - _dt_fine / 2, times_fine[-1] + _dt_fine / 2]
        _ch_edges = np.arange(n_ch + 1)
        vmax_hm = max(float(np.nanpercentile(np.abs(isc_fine), 98)), 1e-6)
        pcm = ax_hm.pcolormesh(
            _t_edges,
            _ch_edges,
            isc_fine.T,
            cmap="RdBu_r",
            vmin=-vmax_hm,
            vmax=vmax_hm,
            rasterized=True,
            shading="flat",
        )
        ax_hm.set_ylim(n_ch, 0)
        fig_ov.colorbar(pcm, cax=cax_hm, label="LOO-ISC (r)")
        ax_hm.set_xlabel("Time (s)")
        ax_hm.set_ylabel(ch_label_sw)
        sns.despine(fig=fig_ov, left=False, bottom=False)
        fig_ov.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_ov,
                Path(save_path_dir) / f"sw_isc_overlay_{band}_{label}.png",
            )
        plt.show()

        # ── Figure C — Pearson vs. Spearman comparison ────────────────────
        fig_cmp, ax_cmp = plt.subplots(figsize=(14, 4))
        ax_cmp.step(
            t_ds,
            step_med_arr[::ds],
            where="post",
            color="steelblue",
            lw=2.0,
            label=f"Pearson  ({window_med_sec:.0f} s / {window_med_sec / 2:.1f} s step)",
        )
        ax_cmp.step(
            t_ds,
            step_spear_arr[::ds],
            where="post",
            color="darkorange",
            lw=2.0,
            ls="--",
            label=f"Spearman ({window_med_sec:.0f} s / {window_med_sec / 2:.1f} s step)",
        )
        ax_cmp.fill_between(
            t_ds,
            step_med_arr[::ds],
            step_spear_arr[::ds],
            alpha=0.12,
            color="gray",
            step="post",
            label="Pearson − Spearman gap",
        )
        _pearson_ds = step_med_arr[::ds]
        _spear_ds = step_spear_arr[::ds]
        ax_cmp.fill_between(
            t_ds,
            0,
            _pearson_ds,
            where=_pearson_ds >= 0,
            step="post",
            alpha=0.10,
            color="steelblue",
        )
        ax_cmp.fill_between(
            t_ds,
            0,
            _pearson_ds,
            where=_pearson_ds < 0,
            step="post",
            alpha=0.18,
            color=_C_NEG,
        )
        ax_cmp.fill_between(
            t_ds,
            0,
            _spear_ds,
            where=_spear_ds >= 0,
            step="post",
            alpha=0.08,
            color="darkorange",
        )
        ax_cmp.fill_between(
            t_ds,
            0,
            _spear_ds,
            where=_spear_ds < 0,
            step="post",
            alpha=0.14,
            color=_C_NEG,
        )
        ax_cmp.axhline(
            thr_band,
            color=_C_THRESH,
            ls="--",
            lw=1.0,
            label=f"Threshold r = {thr_band}",
        )
        ax_cmp.axhline(0, color="gray", ls="-", lw=0.6, alpha=0.4)
        ax_cmp.set_xlabel("Time (s)")
        ax_cmp.set_ylabel("Mean LOO-ISC (r)")
        ax_cmp.set_title(
            f"[{label} / {band.upper()}]  Pearson vs. Spearman  "
            f"({window_med_sec:.0f} s / 50% overlap)"
        )
        ax_cmp.legend(frameon=False, fontsize=9)
        sns.despine(fig=fig_cmp)
        fig_cmp.tight_layout()
        if save_path_dir is not None:
            _save_fig(
                fig_cmp,
                Path(save_path_dir) / f"sw_isc_pearson_vs_spearman_{band}_{label}.png",
            )
        plt.show()

        n_len = min(len(mean_med), len(mean_sp))
        corr_between = float(np.corrcoef(mean_med[:n_len], mean_sp[:n_len])[0, 1])
        _logger.info(
            f"[{label} / {band.upper()}] Pearson vs. Spearman ISC correlation:"
            f" r = {corr_between:.4f}"
        )

        results[band] = (fig_bar, fig_ov, fig_cmp)

    return results


# ---------------------------------------------------------------------------
# 15. Data overview (text)
# ---------------------------------------------------------------------------


def print_data_overview(
    datasets: dict[str, "AnalysisData"],
) -> None:
    """
    Print a summary table for each dataset in the dict.

    :param datasets: ``{label: AnalysisData}``
    """
    for label, ad in datasets.items():
        n_items, n_features, n_samples = ad.data.shape
        duration_sec = n_samples / ad.sfreq
        _logger.info(f"--- {label} ({ad.representation.value}) ---")
        _logger.info(f"  {ad.item_axis_label}s : {n_items}")
        _logger.info(f"  {ad.feature_axis_label}s: {n_features}")
        _logger.info(f"  Samples      : {n_samples}")
        _logger.info(f"  Sfreq        : {ad.sfreq} Hz")
        _logger.info(
            f"  Duration     : {duration_sec:.1f} s  ({duration_sec / 60:.1f} min)"
        )
        _logger.info(f"  Data dtype   : {ad.data.dtype}")
        _logger.info(f"  Data range   : [{ad.data.min():.3f}, {ad.data.max():.3f}]")
        _logger.info("")
