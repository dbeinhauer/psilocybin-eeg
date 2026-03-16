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
from operator import itemgetter
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.analysis.isc import FREQUENCY_BANDS
from src.analysis.data_representations import AnalysisData
from src.definitions.fields import FrequencyBandNames

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
        print(
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

        print(f"=== {label} ===")
        print(
            f"Threshold: r > {isc_threshold}   |   "
            f"Significant windows: {n_sig}/{len(sig_mask)} "
            f"({100 * n_sig / len(sig_mask):.1f} %)\n"
        )
        print(header)
        print(sep)
        for k, (tb, tf, mean_r, peak_r) in enumerate(intervals, 1):
            dur_s = (tf - tb) * 60
            print(
                f"{k:>3}  {tb:>11.3f}  {tf:>9.3f}  {dur_s:>12.1f}  "
                f"{mean_r:>8.4f}  {peak_r:>8.4f}"
            )
        print()


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
        print(f"{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")
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
            print(
                f"\n  {band} ({l_freq}\u2013{h_freq} Hz)  |  threshold r > {thr}  |  "
                f"sig windows: {n_sig}/{n_total} ({100 * n_sig / n_total:.1f} %)"
            )
            if intervals:
                header = (
                    f"  {'#':>3}  {'Start(min)':>10}  {'End(min)':>9}  "
                    f"{'Dur(s)':>7}  {'Mean r':>8}  {'Peak r':>8}"
                )
                print(header)
                print("  " + "-" * 56)
                for k, (tb, tf, mr, pr) in enumerate(intervals, 1):
                    dur_s = (tf - tb) * 60
                    print(
                        f"  {k:>3}  {tb:>10.3f}  {tf:>9.3f}  "
                        f"{dur_s:>7.1f}  {mr:>8.4f}  {pr:>8.4f}"
                    )
            else:
                print("  (no significant intervals)")
        print()


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
        print(
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
# 9. Data overview (text)
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
        print(f"--- {label} ({ad.representation.value}) ---")
        print(f"  {ad.item_axis_label}s : {n_items}")
        print(f"  {ad.feature_axis_label}s: {n_features}")
        print(f"  Samples      : {n_samples}")
        print(f"  Sfreq        : {ad.sfreq} Hz")
        print(f"  Duration     : {duration_sec:.1f} s  ({duration_sec / 60:.1f} min)")
        print(f"  Data dtype   : {ad.data.dtype}")
        print(f"  Data range   : [{ad.data.min():.3f}, {ad.data.max():.3f}]")
        print()


# ---------------------------------------------------------------------------
# 10. Mean & Variance distribution
# ---------------------------------------------------------------------------


def plot_mean_variance_distribution(
    mean_var_results: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str = "Distribution of signal mean and variance across features",
    feature_axis_label: str = "Number of features",
    colors: Optional[dict[str, str]] = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Histograms of per-feature mean and variance, overlaid for each condition.

    :param mean_var_results: ``{label: (mean_per_feature, var_per_feature)}``
        where each array has shape ``(n_features,)``.
    """
    labels = list(mean_var_results.keys())
    colors = _resolve_colors(labels, colors)

    fig, (ax_mean, ax_var) = plt.subplots(1, 2, figsize=figsize)

    for label, (mean_vals, var_vals) in mean_var_results.items():
        color = colors[label]

        ax_mean.hist(
            mean_vals,
            bins=30,
            alpha=0.55,
            edgecolor="black",
            color=color,
            label=label,
        )
        ax_mean.axvline(
            mean_vals.mean(),
            color=color,
            ls="--",
            lw=1.5,
            label=f"{label} μ={mean_vals.mean():.4f}",
        )

        ax_var.hist(
            var_vals,
            bins=30,
            alpha=0.55,
            edgecolor="black",
            color=color,
            label=label,
        )
        ax_var.axvline(
            var_vals.mean(),
            color=color,
            ls="--",
            lw=1.5,
            label=f"{label} μ={var_vals.mean():.4f}",
        )

    ax_mean.set_xlabel("Mean of signal (across subjects)")
    ax_mean.set_ylabel(feature_axis_label)
    ax_mean.set_title("Signal mean per feature")
    ax_mean.legend(fontsize=8)

    ax_var.set_xlabel("Variance of signal (across subjects)")
    ax_var.set_ylabel(feature_axis_label)
    ax_var.set_title("Signal variance per feature")
    ax_var.legend(fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 11. Sliding-window mean & variance (time-resolved heatmap + mean trace)
# ---------------------------------------------------------------------------


def plot_sliding_window_mean_variance(
    sw_mv_results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    feature_axis_label: str = "Channel index",
    title: str = "Time-resolved sliding-window mean & variance",
    subtitle_template: str = "{label} — {stat}",
    colors: Optional[dict[str, str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    For each condition: mean + variance time courses (top) and per-feature
    heatmaps (bottom).

    :param sw_mv_results:
        ``{label: (mean_timecourse, var_timecourse, window_times)}``
        where shapes are ``(n_windows, n_features)``,
        ``(n_windows, n_features)`` and ``(n_windows,)``.
    """
    labels = list(sw_mv_results.keys())
    n_conditions = len(labels)
    colors = _resolve_colors(labels, colors)

    if figsize is None:
        figsize = (16, 8 * n_conditions)

    # Two stats (mean, variance) per condition -> 2*n_conditions panels
    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(n_conditions * 2, 1, figure=fig, hspace=0.45)

    panel_idx = 0
    for label in labels:
        mean_tc, var_tc, sw_times = sw_mv_results[label]
        time_min = sw_times / 60
        t_start, t_end = time_min[0], time_min[-1]
        color = colors[label]

        for stat_name, stat_tc in [("Mean", mean_tc), ("Variance", var_tc)]:
            gs = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=outer_gs[panel_idx],
                height_ratios=[1, 3],
                hspace=0.05,
            )
            ax_top = fig.add_subplot(gs[0])
            ax_bot = fig.add_subplot(gs[1])

            # Mean across features trace
            mean_trace = stat_tc.mean(axis=1)

            ax_top.plot(time_min, mean_trace, color=color, lw=1.2, zorder=3)
            ax_top.fill_between(time_min, mean_trace, alpha=0.25, color=color, zorder=2)
            ax_top.axhline(0, color="grey", ls="--", lw=0.6, zorder=1)
            ax_top.set_xlim(t_start, t_end)
            ax_top.set_ylabel(f"Mean {stat_name.lower()}")
            ax_top.set_title(subtitle_template.format(label=label, stat=stat_name))
            ax_top.tick_params(labelbottom=False)
            div_top = make_axes_locatable(ax_top)
            div_top.append_axes("right", size="2%", pad=0.05).set_visible(False)

            # Heatmap
            vmax = np.nanmax(np.abs(stat_tc))
            cmap = "RdBu_r" if stat_name == "Mean" else "viridis"
            vmin = -vmax if stat_name == "Mean" else 0

            im = ax_bot.imshow(
                stat_tc.T,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=(t_start, t_end, 0, stat_tc.shape[1]),
                vmin=vmin,
                vmax=vmax,
                zorder=1,
            )
            ax_bot.set_xlim(t_start, t_end)
            ax_bot.set_xlabel("Time (min)")
            ax_bot.set_ylabel(feature_axis_label)
            div_bot = make_axes_locatable(ax_bot)
            cax = div_bot.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax, label=stat_name)

            panel_idx += 1

    fig.suptitle(title, y=1.01, fontsize=14)
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 12. Band mean & variance distribution
# ---------------------------------------------------------------------------


def plot_band_mean_variance_distributions(
    band_mv: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    feature_axis_label: str = "Number of features",
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Histogram of per-feature mean and variance for each frequency band,
    conditions overlaid.

    Layout: rows = frequency bands, columns = [mean histogram, variance histogram].

    :param band_mv: ``{label: {band_name: (mean_per_feature, var_per_feature)}}``
    :param bands: Band definitions.  Defaults to :data:`FREQUENCY_BANDS`.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_mv.keys())
    n_bands = len(band_names)
    colors = _resolve_colors(music_labels)

    if figsize is None:
        figsize = (12, 4 * n_bands)

    fig, axes = plt.subplots(n_bands, 2, figsize=figsize, squeeze=False)

    for row, band in enumerate(band_names):
        l_freq, h_freq = bands[band]
        ax_mean = axes[row, 0]
        ax_var = axes[row, 1]

        for label in music_labels:
            mean_vals, var_vals = band_mv[label][band]
            color = colors[label]

            ax_mean.hist(
                mean_vals,
                bins=30,
                alpha=0.55,
                edgecolor="black",
                color=color,
                label=label,
            )
            ax_mean.axvline(
                mean_vals.mean(),
                color=color,
                ls="--",
                lw=1.5,
                label=f"{label} \u03bc={mean_vals.mean():.4f}",
            )

            ax_var.hist(
                var_vals,
                bins=30,
                alpha=0.55,
                edgecolor="black",
                color=color,
                label=label,
            )
            ax_var.axvline(
                var_vals.mean(),
                color=color,
                ls="--",
                lw=1.5,
                label=f"{label} \u03bc={var_vals.mean():.4f}",
            )

        band_label = f"{band} ({l_freq}\u2013{h_freq} Hz)"
        ax_mean.set_title(f"{band_label} \u2014 signal mean", fontsize=9)
        ax_mean.set_xlabel("Mean of signal (across subjects)")
        ax_mean.set_ylabel(feature_axis_label)
        ax_mean.legend(fontsize=7)

        ax_var.set_title(f"{band_label} \u2014 signal variance", fontsize=9)
        ax_var.set_xlabel("Variance of signal (across subjects)")
        ax_var.set_ylabel(feature_axis_label)
        ax_var.legend(fontsize=7)

    fig.suptitle(
        "Distribution of signal mean and variance per frequency band",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 13. Band sliding-window mean & variance
# ---------------------------------------------------------------------------


def plot_band_sliding_window_mean_variance(
    band_sw_mv: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    feature_axis_label: str = "Channel index",
    figsize: Optional[tuple[float, float]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    For each condition (column) x frequency band (row): mean & variance
    time-course traces (top) and per-feature heatmaps (bottom).

    :param band_sw_mv:
        ``{label: {band_name: (mean_timecourse, var_timecourse, window_times)}}``
    :param bands: Band definitions.  Defaults to :data:`FREQUENCY_BANDS`.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    band_names = list(bands.keys())
    music_labels = list(band_sw_mv.keys())
    n_bands = len(band_names)
    n_music = len(music_labels)

    if figsize is None:
        figsize = (10 * n_music, 8 * n_bands)

    fig = plt.figure(figsize=figsize)
    outer_gs = GridSpec(1, n_music, figure=fig, wspace=0.3)

    for col, label in enumerate(music_labels):
        inner_gs = GridSpecFromSubplotSpec(
            n_bands, 1, subplot_spec=outer_gs[col], hspace=0.7
        )
        for row, band in enumerate(band_names):
            l_freq, h_freq = bands[band]
            mean_tc, var_tc, times = band_sw_mv[label][band]
            time_min = times / 60
            t_start, t_end = time_min[0], time_min[-1]
            band_color = _BAND_COLORS.get(
                band, _DEFAULT_PALETTE[row % len(_DEFAULT_PALETTE)]
            )

            # Each band panel: mean trace (top) + variance trace (middle) + heatmap
            band_gs = GridSpecFromSubplotSpec(
                3,
                1,
                subplot_spec=inner_gs[row],
                height_ratios=[1, 1, 2.5],
                hspace=0.08,
            )
            ax_mean = fig.add_subplot(band_gs[0])
            ax_var = fig.add_subplot(band_gs[1])
            ax_heat = fig.add_subplot(band_gs[2])

            # Mean trace
            mean_trace = mean_tc.mean(axis=1)
            ax_mean.plot(time_min, mean_trace, color=band_color, lw=1.2, zorder=3)
            ax_mean.fill_between(
                time_min, mean_trace, alpha=0.2, color=band_color, zorder=2
            )
            ax_mean.axhline(0, color="grey", ls="--", lw=0.6, zorder=1)
            ax_mean.set_xlim(t_start, t_end)
            ax_mean.set_ylabel("Mean", fontsize=7)
            band_title = f"{band} ({l_freq}\u2013{h_freq} Hz)"
            if row == 0:
                ax_mean.set_title(
                    f"{label}\n{band_title}", fontsize=10, fontweight="bold"
                )
            else:
                ax_mean.set_title(band_title, fontsize=9)
            ax_mean.tick_params(labelbottom=False, labelsize=7)
            ax_mean.yaxis.set_tick_params(labelsize=7)
            div = make_axes_locatable(ax_mean)
            div.append_axes("right", size="2%", pad=0.05).set_visible(False)

            # Variance trace
            var_trace = var_tc.mean(axis=1)
            ax_var.plot(time_min, var_trace, color=band_color, lw=1.2, zorder=3)
            ax_var.fill_between(
                time_min, var_trace, alpha=0.2, color=band_color, zorder=2
            )
            ax_var.axhline(0, color="grey", ls="--", lw=0.6, zorder=1)
            ax_var.set_xlim(t_start, t_end)
            ax_var.set_ylabel("Var", fontsize=7)
            ax_var.tick_params(labelbottom=False, labelsize=7)
            ax_var.yaxis.set_tick_params(labelsize=7)
            div2 = make_axes_locatable(ax_var)
            div2.append_axes("right", size="2%", pad=0.05).set_visible(False)

            # Mean heatmap (spatial per-feature view)
            vmax = np.nanmax(np.abs(mean_tc))
            im = ax_heat.imshow(
                mean_tc.T,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                extent=(t_start, t_end, 0, mean_tc.shape[1]),
                vmin=-vmax,
                vmax=vmax,
                zorder=1,
            )
            ax_heat.set_xlim(t_start, t_end)
            ax_heat.set_xlabel("Time (min)", fontsize=8)
            ax_heat.set_ylabel(feature_axis_label, fontsize=8)
            ax_heat.tick_params(labelsize=7)
            div_bot = make_axes_locatable(ax_heat)
            cax = div_bot.append_axes("right", size="2%", pad=0.05)
            plt.colorbar(im, cax=cax, label="Mean")
            cax.tick_params(labelsize=7)

    fig.suptitle(
        "Band-specific sliding-window mean & variance",
        fontsize=13,
        y=1.01,
    )
    _save_fig(fig, save_path)
    plt.show()
    return fig
