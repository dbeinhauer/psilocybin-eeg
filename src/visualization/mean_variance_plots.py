"""
Visualisation functions for the intersubject mean-variance analysis.

Each function mirrors a section from
``notebooks/01-raw-mean-variance-analysis/mean_variance_broadband.ipynb``
or ``notebooks/01-raw-mean-variance-analysis/mean_variance_bands.ipynb``
and produces publication-ready Matplotlib/Seaborn figures.

All functions accept an optional *save_path*; when provided the figure is
saved before being displayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src.analysis.isc import FREQUENCY_BANDS
from src.definitions.fields import FrequencyBandNames

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_BAND_COLORS: dict[str, str] = {
    FrequencyBandNames.DELTA.value: "#4e79a7",
    FrequencyBandNames.THETA.value: "#f28e2b",
    FrequencyBandNames.ALPHA.value: "#59a14f",
    FrequencyBandNames.BETA.value: "#e15759",
    FrequencyBandNames.GAMMA.value: "#b07aa1",
}

_C_BLUE = "#4c72b0"
_C_ORANGE = "#dd8452"
_C_GREEN = "#5cb85c"
_C_RED = "#c44e52"


def _save_fig(fig: Figure, save_path: Optional[Path]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Raw notebook — Section 1: intersubject time series
# ---------------------------------------------------------------------------


def plot_timeseries(
    stats: dict[str, np.ndarray],
    sfreq: float,
    label: str,
    *,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Two-panel time-series overview (raw data, Section 1).

    Top panel: group-mean signal with per-subject traces and ±1 SD band.
    Bottom panel: channel-averaged intersubject variance over time.

    :param stats: Output of
        :func:`~src.analysis.mean_variance.compute_intersubject_stats`.
    :param sfreq: Sampling frequency in Hz.
    :param label: Dataset label used in figure titles.
    :param save_path: Optional path to save the figure.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    mean_t = stats["mean_t"]
    var_t = stats["var_t"]
    std_t = stats["std_t"]
    mean_over_ch = stats["mean_over_ch"]
    n_times = len(mean_t)
    n_subjects = mean_over_ch.shape[0]
    time = np.arange(n_times) / sfreq

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Panel 1: mean signal with per-subject traces
    ax = axes[0]
    for s in range(n_subjects):
        ax.plot(time, mean_over_ch[s], color="steelblue", alpha=0.2, linewidth=0.7)
    ax.plot(
        time,
        mean_t,
        color="black",
        linewidth=2.0,
        label="Group mean (across subjects & channels)",
    )
    ax.fill_between(
        time,
        mean_t - std_t,
        mean_t + std_t,
        color="steelblue",
        alpha=0.22,
        label="±1 SD (intersubject)",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Signal (z-score)")
    ax.set_title(f"[{label}]  Intersubject mean signal over time (electrode-averaged)")
    ax.legend(frameon=False, fontsize=9)

    # Panel 2: channel-averaged intersubject variance
    ax = axes[1]
    ax.plot(
        time,
        var_t,
        color="darkorange",
        linewidth=1.8,
        label="Mean intersubject variance (across channels)",
    )
    ax.fill_between(time, 0, var_t, color="darkorange", alpha=0.18)
    ax.axhline(
        var_t.mean(),
        color="gray",
        linestyle="--",
        linewidth=0.8,
        label=f"Grand mean variance ({var_t.mean():.3f})",
    )
    ax.set_ylabel("Variance")
    ax.set_xlabel("Time (s)")
    ax.set_title("Intersubject variance over time — low = participants are in sync")
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Raw notebook — Section 2: variance distribution
# ---------------------------------------------------------------------------


def plot_variance_distribution(
    inter_var: np.ndarray,
    label: str,
    *,
    plot_pct: int = 99,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Histogram of intersubject variance values (raw data, Section 2).

    The histogram is clipped at *plot_pct* to suppress extreme outliers.

    :param inter_var: ``(n_channels, n_times)`` intersubject variance array
        from :func:`~src.analysis.mean_variance.compute_intersubject_stats`.
    :param label: Dataset label used in the figure title.
    :param plot_pct: Percentile at which to clip the x-axis.
    :param save_path: Optional path to save the figure.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    all_var = inter_var.ravel()
    total_samples = all_var.size
    p_clip = np.percentile(all_var, plot_pct)
    n_outlier = int((all_var > p_clip).sum())
    frac_out = 100.0 * n_outlier / total_samples

    clipped_vals = all_var[all_var <= p_clip]
    counts, bin_edges = np.histogram(clipped_vals, bins=80)
    pct_vals = counts / total_samples * 100
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        bin_centers,
        pct_vals,
        width=bar_width * 0.95,
        color="darkorange",
        alpha=0.85,
        edgecolor="none",
    )
    ax.axvline(
        np.median(all_var),
        color=".2",
        linestyle="--",
        linewidth=1.2,
        label=f"Median = {np.median(all_var):.3f}",
    )
    ax.axvline(
        p_clip,
        color="crimson",
        linestyle=":",
        linewidth=1.4,
        label=f"{plot_pct}th pct = {p_clip:.3f}",
    )
    ax.set_xlabel("Intersubject variance")
    ax.set_ylabel("% of all channel × time samples")
    ax.set_title(
        f"[{label}]  Variance distribution (clipped at {plot_pct}th pct)\n"
        f"⚠ {n_outlier:,} outlier samples ({frac_out:.2f}%) not shown"
    )
    ax.legend(frameon=False, fontsize=9)

    sns.despine(fig=fig)
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Raw notebook — Section 4: windowed analysis (bar + overlay)
# ---------------------------------------------------------------------------


def plot_windowed_analysis(
    stats: dict[str, np.ndarray],
    df_wins: pd.DataFrame,
    sfreq: float,
    label: str,
    window_sec: float,
    sync_percentile: float,
    *,
    save_path_bar: Optional[Path] = None,
    save_path_overlay: Optional[Path] = None,
) -> tuple[Figure, Figure]:
    """
    Windowed mean-variance analysis figures (raw data, Section 4).

    Produces two figures:

    1. **Bar charts** — per-window mean signal and mean variance.  Windows
       below the synchrony threshold are coloured green.
    2. **Overlay** — continuous signal/variance with windowed step functions
       and a per-electrode variance heatmap.

    :param stats: Output of
        :func:`~src.analysis.mean_variance.compute_intersubject_stats`.
    :param df_wins: Output of
        :func:`~src.analysis.mean_variance.compute_windowed_stats`.
    :param sfreq: Sampling frequency in Hz.
    :param label: Dataset label used in figure titles.
    :param window_sec: Window length in seconds (for axis labels).
    :param sync_percentile: Percentile used as sync threshold (for labels).
    :param save_path_bar: Optional path to save the bar-chart figure.
    :param save_path_overlay: Optional path to save the overlay figure.
    :return: ``(fig_bar, fig_overlay)``.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    mean_t = stats["mean_t"]
    var_t = stats["var_t"]
    std_t = stats["std_t"]
    inter_var = stats["inter_var"]
    n_times = len(mean_t)
    n_channels = inter_var.shape[0]
    time = np.arange(n_times) / sfreq

    sync_threshold = np.percentile(var_t, sync_percentile)
    win_samples = int(window_sec * sfreq)
    n_windows = len(df_wins)

    win_centers = df_wins["center"].values
    win_mean_sig = df_wins["mean_signal"].values
    win_var_sig = df_wins["var_signal"].values
    win_mean_var = df_wins["mean_variance"].values
    is_sync_win = df_wins["sync_candidate"].values

    # Step-function arrays padded to n_times
    _pad = n_times - n_windows * win_samples
    win_mean_sig_step = np.pad(
        np.repeat(win_mean_sig, win_samples), (0, _pad), mode="edge"
    )
    win_mean_var_step = np.pad(
        np.repeat(win_mean_var, win_samples), (0, _pad), mode="edge"
    )

    sync_label = f"Sync candidate (var < {sync_percentile}th pct)"
    norm_label = "Normal window"

    muted = sns.color_palette("muted")
    C_BLUE = muted[0]
    C_ORANGE = muted[1]

    sig_palette = {sync_label: _C_GREEN, norm_label: C_BLUE}
    var_palette = {sync_label: _C_GREEN, norm_label: C_ORANGE}

    bar_colors_sig = [_C_GREEN if s else C_BLUE for s in is_sync_win]
    bar_colors_var = [_C_GREEN if s else C_ORANGE for s in is_sync_win]
    bar_w = window_sec * 0.82

    # ── Figure 1: bar charts ─────────────────────────────────────
    fig1, axes = plt.subplots(
        2, 1, figsize=(max(14, n_windows * 0.18), 7), sharex=True
    )

    ax = axes[0]
    ax.bar(win_centers, win_mean_sig, width=bar_w, color=bar_colors_sig, alpha=0.85, edgecolor="none")
    ax.errorbar(
        win_centers,
        win_mean_sig,
        yerr=win_var_sig,
        fmt="none",
        color=".3",
        capsize=2,
        linewidth=0.8,
        label="±variance across subjects",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="Zero baseline")
    ax.axhline(
        win_mean_sig.mean(),
        color=".2",
        linestyle=":",
        linewidth=1.1,
        label=f"Grand mean ({win_mean_sig.mean():.3f})",
    )
    ax.set_ylabel("Mean signal (z-score)")
    ax.set_title(f"[{label}]  Per-window mean signal  (window = {window_sec:.1f} s)")
    bar_patches = [Patch(color=c, alpha=0.85, label=lbl) for lbl, c in sig_palette.items()]
    extra_h, extra_l = ax.get_legend_handles_labels()
    ax.legend(
        handles=bar_patches + extra_h,
        labels=[p.get_label() for p in bar_patches] + extra_l,
        frameon=False,
        fontsize=9,
    )

    ax = axes[1]
    ax.bar(win_centers, win_mean_var, width=bar_w, color=bar_colors_var, alpha=0.85, edgecolor="none")
    ax.axhline(
        sync_threshold,
        color=_C_GREEN,
        linestyle="--",
        linewidth=1.1,
        label=f"Sync threshold — {sync_percentile}th pct ({sync_threshold:.3f})",
    )
    ax.axhline(
        win_mean_var.mean(),
        color=".2",
        linestyle=":",
        linewidth=1.1,
        label=f"Grand mean variance ({win_mean_var.mean():.3f})",
    )
    ax.set_ylabel("Mean intersubject variance")
    ax.set_xlabel("Time (s)")
    ax.set_title("Per-window mean intersubject variance  |  green = sync candidates")
    var_patches = [Patch(color=c, alpha=0.85, label=lbl) for lbl, c in var_palette.items()]
    extra_h2, extra_l2 = ax.get_legend_handles_labels()
    ax.legend(
        handles=var_patches + extra_h2,
        labels=[p.get_label() for p in var_patches] + extra_l2,
        frameon=False,
        fontsize=9,
    )

    sns.despine(fig=fig1, left=False, bottom=False)
    fig1.tight_layout()
    _save_fig(fig1, save_path_bar)
    plt.show()

    # ── Figure 2: continuous overlay + heatmap ──────────────────
    ds = max(1, n_times // 8000)
    t_ds = time[::ds]

    df_sig = pd.DataFrame(
        {
            "time": t_ds,
            "mean": mean_t[::ds],
            "lower": (mean_t - std_t)[::ds],
            "upper": (mean_t + std_t)[::ds],
            "windowed": win_mean_sig_step[::ds],
        }
    )
    df_var = pd.DataFrame(
        {
            "time": t_ds,
            "variance": var_t[::ds],
            "windowed": win_mean_var_step[::ds],
        }
    )

    fig2 = plt.figure(figsize=(15, 11))
    gs2 = gridspec.GridSpec(
        3,
        2,
        height_ratios=[2, 2, 3],
        width_ratios=[30, 1],
        hspace=0.20,
        wspace=0.05,
        figure=fig2,
    )
    ax_s = fig2.add_subplot(gs2[0, 0])
    ax_v = fig2.add_subplot(gs2[1, 0], sharex=ax_s)
    ax_hm = fig2.add_subplot(gs2[2, 0], sharex=ax_s)
    cax = fig2.add_subplot(gs2[2, 1])

    # Signal overlay
    ax = ax_s
    ax.fill_between(
        df_sig["time"],
        df_sig["lower"],
        df_sig["upper"],
        color=C_BLUE,
        alpha=0.15,
        label="±1 SD (intersubject)",
    )
    sns.lineplot(
        data=df_sig,
        x="time",
        y="mean",
        ax=ax,
        errorbar=None,
        color=".2",
        linewidth=1.1,
        alpha=0.6,
        label="Continuous group mean",
    )
    ax.step(
        df_sig["time"],
        df_sig["windowed"],
        where="post",
        color=C_BLUE,
        linewidth=2.0,
        label=f"Windowed mean ({window_sec:.1f} s)",
    )
    _first_s = True
    for w in np.where(is_sync_win)[0]:
        t_s = time[w * win_samples]
        t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
        ax.axvspan(
            t_s,
            t_e,
            color=_C_GREEN,
            alpha=0.22,
            label="Sync candidate window" if _first_s else "_nolegend_",
        )
        _first_s = False
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, label="Zero baseline")
    ax.set_ylabel("Signal (z-score)")
    ax.set_title(
        f"[{label}]  Continuous vs. windowed mean signal / variance + "
        f"per-electrode heatmap  (window = {window_sec:.1f} s)"
    )
    ax.legend(frameon=False, fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Variance overlay
    ax = ax_v
    sns.lineplot(
        data=df_var,
        x="time",
        y="variance",
        ax=ax,
        errorbar=None,
        color=C_ORANGE,
        linewidth=1.0,
        alpha=0.55,
        label="Continuous intersubject variance",
    )
    ax.fill_between(df_var["time"], 0, df_var["variance"], color=C_ORANGE, alpha=0.10)
    ax.step(
        df_var["time"],
        df_var["windowed"],
        where="post",
        color=_C_RED,
        linewidth=2.0,
        label=f"Windowed mean variance ({window_sec:.1f} s)",
    )
    ax.axhline(
        sync_threshold,
        color=_C_GREEN,
        linestyle="--",
        linewidth=1.1,
        label=f"Sync threshold — {sync_percentile}th pct ({sync_threshold:.3f})",
    )
    _first_s = True
    for w in np.where(is_sync_win)[0]:
        t_s = time[w * win_samples]
        t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
        ax.axvspan(
            t_s,
            t_e,
            color=_C_GREEN,
            alpha=0.22,
            label="Sync candidate window" if _first_s else "_nolegend_",
        )
        _first_s = False
    ax.set_ylabel("Intersubject variance")
    ax.legend(frameon=False, fontsize=9)
    plt.setp(ax.get_xticklabels(), visible=False)

    # Per-electrode heatmap
    ds_hm = max(1, n_times // 2000)
    _time_hm = time[::ds_hm]
    _var_hm = inter_var[:, ::ds_hm]
    _dt = (_time_hm[1] - _time_hm[0]) if len(_time_hm) > 1 else 1.0 / sfreq
    _t_edges = np.r_[_time_hm - _dt / 2, _time_hm[-1] + _dt / 2]
    _ch_edges = np.arange(n_channels + 1)

    pcm = ax_hm.pcolormesh(
        _t_edges,
        _ch_edges,
        _var_hm,
        cmap="YlOrRd",
        vmin=0,
        vmax=np.percentile(inter_var, 98),
        rasterized=True,
        shading="flat",
    )
    ax_hm.set_ylim(n_channels, 0)
    fig2.colorbar(pcm, cax=cax, label="Intersubject variance")
    _first_s = True
    for w in np.where(is_sync_win)[0]:
        t_s = time[w * win_samples]
        t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
        ax_hm.axvspan(
            t_s,
            t_e,
            color=_C_GREEN,
            alpha=0.18,
            label="Sync candidate window" if _first_s else "_nolegend_",
        )
        _first_s = False
    ax_hm.set_xlabel("Time (s)")
    ax_hm.set_ylabel("Electrode (natural order)")

    sns.despine(fig=fig2, left=False, bottom=False)
    fig2.tight_layout()
    _save_fig(fig2, save_path_overlay)
    plt.show()

    return fig1, fig2


# ---------------------------------------------------------------------------
# Band notebook — Section 1: per-band time series
# ---------------------------------------------------------------------------


def plot_band_timeseries(
    band_stats: dict[str, dict[str, np.ndarray]],
    sfreq: float,
    label: str,
    *,
    sync_percentile: float = 10.0,
    band_colors: Optional[dict[str, str]] = None,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Five-row two-column per-band time-series overview (bands Section 1).

    Left column: group-mean signal with per-subject traces and ±1 SD.
    Right column: channel-averaged intersubject variance with sync threshold.

    :param band_stats: ``{band_name: stats_dict}`` — output of
        :func:`~src.analysis.mean_variance.compute_band_intersubject_stats`.
    :param sfreq: Sampling frequency in Hz.
    :param label: Dataset label used in figure titles.
    :param sync_percentile: Percentile for the synchrony threshold line.
    :param band_colors: Optional ``{band_name: colour_string}`` mapping.
        Defaults to :data:`_DEFAULT_BAND_COLORS`.
    :param bands: Band definitions for axis labels.
        Defaults to :data:`~src.analysis.isc.FREQUENCY_BANDS`.
    :param save_path: Optional path to save the figure.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _DEFAULT_BAND_COLORS

    band_names = list(band_stats.keys())
    n_bands = len(band_names)
    n_times = len(next(iter(band_stats.values()))["mean_t"])
    n_subjects = next(iter(band_stats.values()))["mean_over_ch"].shape[0]
    time = np.arange(n_times) / sfreq

    fig, axes = plt.subplots(
        n_bands, 2, figsize=(16, 3.5 * n_bands), sharex=True
    )
    if n_bands == 1:
        axes = np.array([axes])

    for row, band in enumerate(band_names):
        l_freq, h_freq = bands.get(band, (0, 0))
        st = band_stats[band]
        c = band_colors.get(band, "steelblue")

        # Left: mean signal
        ax = axes[row, 0]
        for s in range(n_subjects):
            ax.plot(time, st["mean_over_ch"][s], color=c, alpha=0.15, linewidth=0.5)
        ax.plot(time, st["mean_t"], color="black", linewidth=1.5, label="Group mean")
        ax.fill_between(
            time,
            st["mean_t"] - st["std_t"],
            st["mean_t"] + st["std_t"],
            color=c,
            alpha=0.25,
            label="±1 SD (intersubject)",
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel("Signal (z-score)", fontsize=8)
        if row == 0:
            ax.set_title("Mean signal per band", fontsize=10, fontweight="bold")
        ax.text(
            0.01,
            0.97,
            f"{band.upper()} ({l_freq:.0f}–{h_freq:.0f} Hz)",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            fontweight="bold",
            color=c,
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")

        # Right: intersubject variance
        ax = axes[row, 1]
        ax.plot(
            time,
            st["var_t"],
            color=c,
            linewidth=1.2,
            alpha=0.7,
            label="Mean intersubject variance",
        )
        ax.fill_between(time, 0, st["var_t"], color=c, alpha=0.18)
        ax.axhline(
            st["var_t"].mean(),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"Grand mean ({st['var_t'].mean():.3f})",
        )
        sync_thr = np.percentile(st["var_t"], sync_percentile)
        ax.axhline(
            sync_thr,
            color=_C_GREEN,
            linestyle=":",
            linewidth=1.1,
            label=f"Sync thr — {sync_percentile}th pct ({sync_thr:.3f})",
        )
        ax.set_ylabel("Variance", fontsize=8)
        if row == 0:
            ax.set_title(
                "Intersubject variance  (low = in sync)",
                fontsize=10,
                fontweight="bold",
            )
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")

    fig.suptitle(
        f"[{label}]  Intersubject mean signal & variance per frequency band",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Band notebook — Section 2: per-band variance distributions
# ---------------------------------------------------------------------------


def plot_band_variance_distributions(
    band_stats: dict[str, dict[str, np.ndarray]],
    label: str,
    *,
    plot_pct: int = 99,
    band_colors: Optional[dict[str, str]] = None,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Per-band intersubject variance histograms (bands Section 2).

    One histogram per band arranged in a 2-column grid, clipped at *plot_pct*.

    :param band_stats: ``{band_name: stats_dict}`` — output of
        :func:`~src.analysis.mean_variance.compute_band_intersubject_stats`.
    :param label: Dataset label used in the figure title.
    :param plot_pct: Clip percentile for the x-axis.
    :param band_colors: Optional colour mapping per band.
    :param bands: Band definitions for axis labels.
    :param save_path: Optional path to save the figure.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _DEFAULT_BAND_COLORS

    band_names = list(band_stats.keys())
    n_bands = len(band_names)
    ncols = 2
    nrows = (n_bands + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes_flat = axes.ravel()

    for idx, band in enumerate(band_names):
        l_freq, h_freq = bands.get(band, (0, 0))
        st = band_stats[band]
        c = band_colors.get(band, "steelblue")

        all_var = st["inter_var"].ravel()
        total = all_var.size
        p_clip = np.percentile(all_var, plot_pct)
        n_out = int((all_var > p_clip).sum())
        clipped = all_var[all_var <= p_clip]

        counts, bin_edges = np.histogram(clipped, bins=60)
        pct_vals = counts / total * 100
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width = bin_edges[1] - bin_edges[0]

        ax = axes_flat[idx]
        ax.bar(
            bin_centers,
            pct_vals,
            width=bar_width * 0.95,
            color=c,
            alpha=0.85,
            edgecolor="none",
        )
        ax.axvline(
            np.median(all_var),
            color=".2",
            linestyle="--",
            linewidth=1.2,
            label=f"Median = {np.median(all_var):.4f}",
        )
        ax.axvline(
            p_clip,
            color="crimson",
            linestyle=":",
            linewidth=1.2,
            label=f"{plot_pct}th pct = {p_clip:.4f}",
        )
        ax.set_xlabel("Intersubject variance")
        ax.set_ylabel("% of all ch × time samples")
        ax.set_title(
            f"{band.upper()} ({l_freq:.0f}–{h_freq:.0f} Hz)  "
            f"[{n_out:,} outliers clipped]",
            fontsize=9,
        )
        ax.legend(frameon=False, fontsize=8)

    # Hide any unused subplots
    for idx in range(n_bands, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"[{label}]  Intersubject variance distribution per frequency band "
        f"(clipped at {plot_pct}th pct)",
        fontsize=12,
        y=1.01,
    )
    sns.despine(fig=fig)
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Band notebook — Section 3: ISC matrices
# ---------------------------------------------------------------------------


def plot_isc_matrices(
    isc_matrices: dict[str, np.ndarray],
    n_subjects: int,
    label: str,
    *,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Pairwise ISC heatmaps per frequency band (bands Section 3).

    :param isc_matrices: ``{band_name: (n_subjects, n_subjects) matrix}``
        — output of
        :func:`~src.analysis.mean_variance.compute_pairwise_isc_matrices`.
    :param n_subjects: Number of subjects (used for axis labels).
    :param label: Dataset label used in the figure title.
    :param bands: Band definitions for axis labels.
    :param save_path: Optional path to save the figure.
    :return: The created :class:`~matplotlib.figure.Figure`.
    """
    if bands is None:
        bands = FREQUENCY_BANDS

    band_names = list(isc_matrices.keys())
    n_bands = len(band_names)
    subject_labels = [f"S{s + 1}" for s in range(n_subjects)]

    fig, axes = plt.subplots(1, n_bands, figsize=(4.5 * n_bands, 4.5))
    if n_bands == 1:
        axes = [axes]

    for ax, band in zip(axes, band_names):
        l_freq, h_freq = bands.get(band, (0, 0))
        mat = isc_matrices[band]
        vmax = np.abs(mat).max()

        im = ax.imshow(
            mat,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_xticks(range(n_subjects))
        ax.set_xticklabels(subject_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels(subject_labels, fontsize=7)
        ax.set_title(
            f"{band.upper()}\n({l_freq:.0f}–{h_freq:.0f} Hz)", fontsize=9
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean ISC")

    fig.suptitle(
        f"[{label}]  Pairwise inter-subject correlation (ISC) per band",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, save_path)
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Band notebook — Section 4: per-band windowed analysis
# ---------------------------------------------------------------------------


def plot_band_windowed_analysis(
    band_stats: dict[str, dict[str, np.ndarray]],
    sfreq: float,
    label: str,
    window_sec: float,
    sync_percentile: float,
    *,
    band_colors: Optional[dict[str, str]] = None,
    bands: Optional[dict[str, tuple[float, float]]] = None,
    save_path_summary: Optional[Path] = None,
    save_path_per_band_dir: Optional[Path] = None,
) -> tuple[Figure, list[Figure]]:
    """
    Windowed synchrony analysis per frequency band (bands Section 4).

    Produces:

    1. **Summary figure** — all bands in a single multi-row plot showing
       windowed mean variance and synchrony candidate spans.
    2. **Per-band detailed figures** (one per band) — continuous variance
       with windowed step, sync threshold, and per-electrode heatmap.

    :param band_stats: ``{band_name: stats_dict}`` — output of
        :func:`~src.analysis.mean_variance.compute_band_intersubject_stats`.
    :param sfreq: Sampling frequency in Hz.
    :param label: Dataset label used in figure titles.
    :param window_sec: Window length in seconds.
    :param sync_percentile: Percentile for synchrony detection.
    :param band_colors: Optional colour mapping per band.
    :param bands: Band definitions for axis labels.
    :param save_path_summary: Optional path to save the summary figure.
    :param save_path_per_band_dir: Optional directory in which to save each
        per-band figure as ``<band>.png``.
    :return: ``(fig_summary, per_band_figs)``.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    if bands is None:
        bands = FREQUENCY_BANDS
    if band_colors is None:
        band_colors = _DEFAULT_BAND_COLORS

    band_names = list(band_stats.keys())
    n_bands = len(band_names)
    n_times = len(next(iter(band_stats.values()))["mean_t"])
    time = np.arange(n_times) / sfreq

    win_samples = int(window_sec * sfreq)
    n_windows = n_times // win_samples

    # Pre-compute windowed stats for each band
    band_win_stats: dict[str, dict] = {}
    for band in band_names:
        st = band_stats[band]
        var_t_b = st["var_t"]
        sync_thr = np.percentile(var_t_b, sync_percentile)
        records = []
        for w in range(n_windows):
            sl = slice(w * win_samples, (w + 1) * win_samples)
            records.append(
                {
                    "window": w + 1,
                    "center": time[w * win_samples + win_samples // 2],
                    "t_start": time[w * win_samples],
                    "t_end": time[min((w + 1) * win_samples - 1, n_times - 1)],
                    "mean_variance": float(var_t_b[sl].mean()),
                }
            )
        df_w = pd.DataFrame(records)
        df_w["sync_candidate"] = df_w["mean_variance"] < sync_thr
        band_win_stats[band] = {
            "df": df_w,
            "sync_thr": sync_thr,
            "win_var_step": np.pad(
                np.repeat(df_w["mean_variance"].values, win_samples),
                (0, n_times - n_windows * win_samples),
                mode="edge",
            ),
        }

    # ── Summary figure ────────────────────────────────────────────────
    fig_sum, axes_sum = plt.subplots(
        n_bands, 1, figsize=(15, 3.0 * n_bands), sharex=True
    )
    if n_bands == 1:
        axes_sum = [axes_sum]

    for row, band in enumerate(band_names):
        l_freq, h_freq = bands.get(band, (0, 0))
        c = band_colors.get(band, "steelblue")
        bws = band_win_stats[band]
        df_w = bws["df"]
        sync_thr = bws["sync_thr"]
        win_var_step = bws["win_var_step"]
        is_sync_win = df_w["sync_candidate"].values

        ax = axes_sum[row]
        ax.step(time, win_var_step, where="post", color=c, linewidth=1.8, label=f"{band.upper()} windowed var")
        ax.axhline(sync_thr, color=_C_GREEN, linestyle="--", linewidth=1.0, label=f"Sync thr — {sync_percentile}th pct")
        _first = True
        for w in np.where(is_sync_win)[0]:
            t_s = time[w * win_samples]
            t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
            ax.axvspan(
                t_s,
                t_e,
                color=_C_GREEN,
                alpha=0.25,
                label="Sync candidate" if _first else "_nolegend_",
            )
            _first = False
        ax.set_ylabel("Mean var", fontsize=8)
        band_label = f"{band.upper()} ({l_freq:.0f}–{h_freq:.0f} Hz)"
        ax.set_title(band_label, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="upper right")

    axes_sum[-1].set_xlabel("Time (s)")
    fig_sum.suptitle(
        f"[{label}]  Windowed synchrony analysis per band  (window = {window_sec:.1f} s)",
        fontsize=12,
        y=1.01,
    )
    fig_sum.tight_layout()
    _save_fig(fig_sum, save_path_summary)
    plt.show()

    # ── Per-band detailed figures ─────────────────────────────────────
    per_band_figs: list[Figure] = []
    for band in band_names:
        l_freq, h_freq = bands.get(band, (0, 0))
        c = band_colors.get(band, "steelblue")
        st = band_stats[band]
        bws = band_win_stats[band]
        df_w = bws["df"]
        sync_thr = bws["sync_thr"]
        win_var_step = bws["win_var_step"]
        is_sync_win = df_w["sync_candidate"].values
        inter_var_b = st["inter_var"]
        n_channels = inter_var_b.shape[0]

        # Downsample for rendering
        ds = max(1, n_times // 8000)
        t_ds = time[::ds]

        fig_b = plt.figure(figsize=(15, 9))
        gs_b = gridspec.GridSpec(
            2,
            2,
            height_ratios=[2, 3],
            width_ratios=[30, 1],
            hspace=0.20,
            wspace=0.05,
            figure=fig_b,
        )
        ax_top = fig_b.add_subplot(gs_b[0, 0])
        ax_hm = fig_b.add_subplot(gs_b[1, 0], sharex=ax_top)
        cax_b = fig_b.add_subplot(gs_b[1, 1])

        # Continuous variance + windowed step
        sns.lineplot(
            x=t_ds,
            y=st["var_t"][::ds],
            ax=ax_top,
            errorbar=None,
            color=c,
            linewidth=1.0,
            alpha=0.6,
            label="Continuous intersubject variance",
        )
        ax_top.fill_between(t_ds, 0, st["var_t"][::ds], color=c, alpha=0.10)
        ax_top.step(
            t_ds,
            win_var_step[::ds],
            where="post",
            color=_C_RED,
            linewidth=2.0,
            label=f"Windowed mean variance ({window_sec:.1f} s)",
        )
        ax_top.axhline(
            sync_thr,
            color=_C_GREEN,
            linestyle="--",
            linewidth=1.1,
            label=f"Sync threshold — {sync_percentile}th pct ({sync_thr:.3f})",
        )
        _first = True
        for w in np.where(is_sync_win)[0]:
            t_s = time[w * win_samples]
            t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
            ax_top.axvspan(
                t_s,
                t_e,
                color=_C_GREEN,
                alpha=0.22,
                label="Sync candidate window" if _first else "_nolegend_",
            )
            _first = False
        ax_top.set_ylabel("Intersubject variance")
        ax_top.set_title(
            f"[{label}]  {band.upper()} ({l_freq:.0f}–{h_freq:.0f} Hz)  "
            f"windowed synchrony  (window = {window_sec:.1f} s)"
        )
        ax_top.legend(frameon=False, fontsize=9)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        # Per-electrode heatmap
        ds_hm = max(1, n_times // 2000)
        _time_hm = time[::ds_hm]
        _var_hm = inter_var_b[:, ::ds_hm]
        _dt = (_time_hm[1] - _time_hm[0]) if len(_time_hm) > 1 else 1.0 / sfreq
        _t_edges = np.r_[_time_hm - _dt / 2, _time_hm[-1] + _dt / 2]
        _ch_edges = np.arange(n_channels + 1)

        pcm = ax_hm.pcolormesh(
            _t_edges,
            _ch_edges,
            _var_hm,
            cmap="YlOrRd",
            vmin=0,
            vmax=np.percentile(inter_var_b, 98),
            rasterized=True,
            shading="flat",
        )
        ax_hm.set_ylim(n_channels, 0)
        fig_b.colorbar(pcm, cax=cax_b, label="Intersubject variance")
        _first = True
        for w in np.where(is_sync_win)[0]:
            t_s = time[w * win_samples]
            t_e = time[min((w + 1) * win_samples - 1, n_times - 1)]
            ax_hm.axvspan(
                t_s,
                t_e,
                color=_C_GREEN,
                alpha=0.18,
                label="Sync candidate window" if _first else "_nolegend_",
            )
            _first = False
        ax_hm.set_xlabel("Time (s)")
        ax_hm.set_ylabel("Electrode (natural order)")

        sns.despine(fig=fig_b, left=False, bottom=False)
        fig_b.tight_layout()

        if save_path_per_band_dir is not None:
            save_path_per_band_dir = Path(save_path_per_band_dir)
            _save_fig(fig_b, save_path_per_band_dir / f"{band}.png")
        plt.show()
        per_band_figs.append(fig_b)

    return fig_sum, per_band_figs
