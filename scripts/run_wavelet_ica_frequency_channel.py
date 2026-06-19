"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_frequency_channel.ipynb``
(Frequency-Channel approach: observations = S×T, features = F×C;
components live in frequency × channel — the transposed companion of the
Subject-Time variant, which has F×C observations and S×T features).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-frequency-channel-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/frequency_channel/                  # default (no --band)
            pca_scree_<label>.png
            ...
        bands/frequency_channel/                      # when --band <name> is given
            <band>_pca_scree_<label>.png
            ...

Pass ``--band <name>`` to slice the cached broadband wavelets down to a
single frequency band (alpha/beta/...) before the ICA step. The same
broadband cache is reused — no separate per-band cache is needed.

Usage::

    # broadband
    python scripts/run_wavelet_ica_frequency_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 50 --n_ica 10 --reuse_wavelets \\
        --sliding_variants 1.0:0.5

    # alpha-band only
    python scripts/run_wavelet_ica_frequency_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --band alpha --n_pca 30 --n_ica 10 --reuse_wavelets
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from mne.viz import plot_topomap  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import PCA, FastICA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    FREQUENCY_BANDS,
    _broadband_wavelet_4d,
    add_wavelet_grid_args,
    analyzers_to_datasets,
    load_analyzers,
)
from src.analysis.wavelet_ica import zscore_by_time  # noqa: E402
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    FrequencyBandNames,
    MusicTypeVariants,
)

_logger = logging.getLogger(__name__)

_STAGE_DIR = "04-frequency-channel-wavelet-ica-analysis"

# ---------------------------------------------------------------------------
# Cluster-analysis constants and helpers
# ---------------------------------------------------------------------------

# Pearson-r thresholds used for the cluster strip below each ISC matrix.
ISC_CLUSTER_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

# Cluster-agreement ratio thresholds used for the overall cluster strip.
OVERALL_RATIO_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

# Top-percent-by-mean-LOO-ISC windows highlighted in the sliding-window plot.
HIGHLIGHT_TOP_PERCENT: float = 10.0

# Discrete colormap: light gray for singletons (0) + tab10 for groups (1..10).
_GROUP_PALETTE = list(plt.colormaps["tab10"].colors)
_CLUSTER_CMAP = ListedColormap(["#dddddd"] + _GROUP_PALETTE)
_CLUSTER_NORM = BoundaryNorm(
    np.arange(-0.5, len(_GROUP_PALETTE) + 1.5, 1.0), _CLUSTER_CMAP.N
)


def _cluster_grid(corr_mat: np.ndarray, n_subjects: int) -> np.ndarray:
    """(T, S) grid of within-row cluster IDs. Singletons → 0, groups → 1, 2, ...."""
    grid = np.zeros((len(ISC_CLUSTER_THRESHOLDS), n_subjects), dtype=int)
    for t_idx, thr in enumerate(ISC_CLUSTER_THRESHOLDS):
        adj = (corr_mat >= thr) & ~np.eye(n_subjects, dtype=bool)
        _, comp_labels = connected_components(adj, directed=False)
        counts = Counter(comp_labels.tolist())
        next_group = 1
        group_map: dict[int, int] = {}
        for s in range(n_subjects):
            lab = int(comp_labels[s])
            if counts[lab] == 1:
                grid[t_idx, s] = 0
            else:
                if lab not in group_map:
                    group_map[lab] = next_group
                    next_group += 1
                grid[t_idx, s] = group_map[lab]
    return grid


def _overall_cluster_grid(ratio_mat: np.ndarray, n_subjects: int) -> np.ndarray:
    """(T, S) grid of overall cluster IDs at each agreement threshold."""
    grid = np.zeros((len(OVERALL_RATIO_THRESHOLDS), n_subjects), dtype=int)
    for t_idx, thr in enumerate(OVERALL_RATIO_THRESHOLDS):
        adj = (ratio_mat >= thr) & ~np.eye(n_subjects, dtype=bool)
        _, labels = connected_components(adj, directed=False)
        counts = Counter(labels.tolist())
        next_group = 1
        group_map: dict[int, int] = {}
        for s in range(n_subjects):
            lab = int(labels[s])
            if counts[lab] == 1:
                grid[t_idx, s] = 0
            else:
                if lab not in group_map:
                    group_map[lab] = next_group
                    next_group += 1
                grid[t_idx, s] = group_map[lab]
    return grid


def _annotate_cluster_grid(ax, grid: np.ndarray) -> None:
    """Overlay numeric cluster IDs (>0) on a cluster-strip imshow."""
    for ti in range(grid.shape[0]):
        for sj in range(grid.shape[1]):
            val = int(grid[ti, sj])
            if val > 0:
                ax.text(
                    sj,
                    ti,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    weight="bold",
                )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Frequency-Channel ICA analysis on preprocessed EEG "
            "wavelet power.  Produces the exact same plots as "
            "wavelet_ica_frequency_channel.ipynb, but for ALL ICA components."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=[e.value for e in ExperimentNames],
        default=ExperimentNames.PSILO_MUSIC.value,
        help="Which experiment dataset to analyse.",
    )
    parser.add_argument(
        "--condition",
        choices=[c.value for c in ConditionVariants],
        default=ConditionVariants.PLACEBO.value,
        help="Experimental condition.",
    )
    parser.add_argument(
        "--music_type",
        nargs="+",
        choices=[mt.value for mt in MusicTypeVariants],
        default=None,
        help=(
            "One or more music types to analyse. When omitted, defaults to "
            "CLASSIC + PSYTRANCE for the psilo_music experiment and ASSR for "
            "the assr experiment."
        ),
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=50,
        help="Number of PCA components to retain. Ignored when --skip_pca is set.",
    )
    parser.add_argument(
        "--skip_pca",
        action="store_true",
        help=(
            "Skip the PCA dimensionality reduction and run FastICA directly "
            "on the (S*T, F*C) z-scored matrix."
        ),
    )
    parser.add_argument(
        "--n_ica",
        type=int,
        default=10,
        help="Number of ICA components to extract.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed shared by PCA and FastICA.",
    )
    add_wavelet_grid_args(parser)
    parser.add_argument(
        "--wavelet_data_dir",
        type=Path,
        default=None,
        help="Directory for cached wavelet tensors.",
    )
    parser.add_argument(
        "--reuse_wavelets",
        action="store_true",
        help="Reuse cached wavelets from --wavelet_data_dir when available.",
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        help="Load raw .fif files, resample, stack, and save .npy caches.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for data loading.",
    )
    parser.add_argument(
        "--sliding_variants",
        nargs="+",
        default=["1.0:0.5"],
        help=(
            "Sliding-window LOO-ISC variants as 'window_sec:step_sec' strings. "
            "If multiple entries are supplied, only the one with the smallest "
            "step is plotted."
        ),
    )
    parser.add_argument(
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the cached broadband wavelet "
            "tensor is sliced to the band's frequency range before ICA, and "
            "plots are written to the 'bands/frequency_channel/' subdirectory "
            "with a '<band>_' filename prefix. Default: full broadband."
        ),
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=None,
        help="Base directory for output plots. Defaults to project plots/ root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Plot functions — exact replicas of the notebook cells, showing ALL ICs
#
# The helpers take their natural input shapes:
#   * ``subj_time_view``  has shape ``(K, S, T)`` — per-IC subject × time map.
#     In this variant it is ``scores_3d.transpose(2, 0, 1)``.
#   * ``freq_chan_view``  has shape ``(F, C, K)`` — per-IC freq × channel map.
#     In this variant it is ``components_2d.transpose(1, 2, 0)``.
# Helpers are byte-identical in structure to the Subject-Time script; the
# data sources differ only at the call sites below.
# ---------------------------------------------------------------------------


def _plot_pca_scree(
    explained: np.ndarray,
    *,
    label: str,
    save_path: Path,
) -> None:
    """PCA scree and cumulative variance plot (notebook Step 2)."""
    cumulative = np.cumsum(explained)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, len(explained) + 1), explained, color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance explained")
    axes[0].set_title(f"PCA Scree Plot — {label}")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, "o-", color="coral")
    axes[1].axhline(0.9, ls="--", color="gray", label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance — {label}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_isc_matrix(
    corr_mats: np.ndarray,
    *,
    label: str,
    suptitle: str,
    save_path: Path,
) -> None:
    """Analysis (a) — Per-IC subject × subject ISC matrices with a cluster strip."""
    n_ica, n_subjects, _ = corr_mats.shape
    fig, axes = plt.subplots(
        2,
        n_ica,
        figsize=(3.5 * n_ica, 5.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )
    if n_ica == 1:
        axes = axes.reshape(2, 1)

    im_corr = None
    for i in range(n_ica):
        corr_mat = corr_mats[i]

        ax_top = axes[0, i]
        im_corr = ax_top.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_top.set_xticks(range(n_subjects))
        ax_top.set_yticks(range(n_subjects))
        ax_top.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_top.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_top.set_title(f"IC {i + 1}", fontsize=10)

        ax_bot = axes[1, i]
        grid = _cluster_grid(corr_mat, n_subjects)
        ax_bot.imshow(grid, cmap=_CLUSTER_CMAP, norm=_CLUSTER_NORM, aspect="auto")
        _annotate_cluster_grid(ax_bot, grid)
        ax_bot.set_xticks(range(n_subjects))
        ax_bot.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax_bot.set_yticks(range(len(ISC_CLUSTER_THRESHOLDS)))
        ax_bot.set_yticklabels(
            [f"r≥{thr}" for thr in ISC_CLUSTER_THRESHOLDS], fontsize=8
        )
        if i == 0:
            ax_bot.set_ylabel("Threshold")

    fig.suptitle(f"{suptitle} — {label}", fontsize=12)
    if im_corr is not None:
        fig.colorbar(im_corr, ax=axes[0, -1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overall_cluster_matrix(
    corr_mats: np.ndarray,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Overall co-clustering ratio matrix + cluster strip aggregated over ICs."""
    n_ica, n_subjects, _ = corr_mats.shape
    n_total = n_ica * len(ISC_CLUSTER_THRESHOLDS)
    co_count = np.zeros((n_subjects, n_subjects), dtype=int)
    for k in range(n_ica):
        for thr in ISC_CLUSTER_THRESHOLDS:
            adj = (corr_mats[k] >= thr) & ~np.eye(n_subjects, dtype=bool)
            _, labels = connected_components(adj, directed=False)
            same = labels[:, None] == labels[None, :]
            co_count += same.astype(int)
    overall_ratio = co_count / n_total

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(5.0, 6.0),
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )

    im_ratio = ax_top.imshow(overall_ratio, vmin=0.0, vmax=1.0, cmap="viridis")
    ax_top.set_xticks(range(n_subjects))
    ax_top.set_yticks(range(n_subjects))
    ax_top.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
    ax_top.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
    ax_top.set_title(
        "Overall co-clustering ratio (mean over ICs × r-thresholds)",
        fontsize=10,
    )
    fig.colorbar(im_ratio, ax=ax_top, label="share of cases in same cluster")

    overall_grid = _overall_cluster_grid(overall_ratio, n_subjects)
    ax_bot.imshow(overall_grid, cmap=_CLUSTER_CMAP, norm=_CLUSTER_NORM, aspect="auto")
    _annotate_cluster_grid(ax_bot, overall_grid)
    ax_bot.set_xticks(range(n_subjects))
    ax_bot.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
    ax_bot.set_yticks(range(len(OVERALL_RATIO_THRESHOLDS)))
    ax_bot.set_yticklabels(
        [f"ratio≥{thr}" for thr in OVERALL_RATIO_THRESHOLDS], fontsize=8
    )
    ax_bot.set_ylabel("Threshold")

    fig.suptitle(f"Overall Subject Co-Clustering — {label}", fontsize=12)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _export_isc_clusters_csv(corr_mats: np.ndarray, save_path: Path) -> None:
    """Export per-(IC, threshold) clusters with two or more members to CSV."""
    n_ica, n_subjects, _ = corr_mats.shape
    rows = []
    for k in range(n_ica):
        for thr in ISC_CLUSTER_THRESHOLDS:
            adj = (corr_mats[k] >= thr) & ~np.eye(n_subjects, dtype=bool)
            _, comp_labels = connected_components(adj, directed=False)
            kept = 0
            for cid in np.unique(comp_labels):
                members = np.where(comp_labels == cid)[0]
                if len(members) < 2:
                    continue
                kept += 1
                rows.append(
                    {
                        "component": f"IC{k + 1}",
                        "threshold": thr,
                        "cluster_id": kept,
                        "cluster_size": int(len(members)),
                        "subject_ids": ",".join(f"S{s + 1}" for s in members),
                    }
                )
    pd.DataFrame(
        rows,
        columns=["component", "threshold", "cluster_id", "cluster_size", "subject_ids"],
    ).to_csv(save_path, index=False)


def _plot_subject_loadings(
    subj_time_view: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (b) — Per-subject mean loading bar plot for ALL ICs.

    Per-IC scalar per subject is ``mean_t |scores_3d[s, t, k]|``, equivalent to
    ``np.abs(subj_time_view).mean(axis=2).T`` since
    ``subj_time_view = scores_3d.transpose(2, 0, 1)`` has shape ``(K, S, T)``.
    """
    n_subjects = subj_time_view.shape[1]
    subject_loadings = np.abs(subj_time_view).mean(axis=2).T  # (S, K)

    n_show = n_ica
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.barh(range(n_subjects), subject_loadings[:, i], color="darkorange")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_xlabel("|score|")
        ax.set_title(f"IC {i + 1}", fontsize=10)

    axes[0].set_ylabel("Subject")
    fig.suptitle(
        f"Per-Subject Mean Loading per Component — {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_time_frequency(
    freq_chan_view: np.ndarray,
    subj_time_view: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (c) — Time × Frequency map per component (outer product) for ALL ICs.

    freq_profile[k] = mean_c components_2d[k, :, :]               (F,)
    time_profile[k] = mean_s scores_3d[:, :, k]                   (T,)
    tf_map[k]       = outer(freq_profile[k], time_profile[k])     (F, T)
    """
    # Frequency profile: collapse channels from ICA components — given
    # freq_chan_view shape (F, C, K), mean over C → (F, K)
    freq_profiles = freq_chan_view.mean(axis=1)  # (F, K)
    # Time profile: average subject scores — given subj_time_view (K, S, T),
    # mean over S → (K, T)
    time_profiles = subj_time_view.mean(axis=1)  # (K, T)
    # Outer product: (F, K) x (K, T) → (K, F, T)
    ft_maps = np.einsum("fk,kt->kft", freq_profiles, time_profiles)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = ft_maps[i]  # (F, T)
        vlim_i = float(np.percentile(np.abs(data_i), 99))
        mesh = ax.pcolormesh(
            time,
            freqs,
            data_i,
            cmap="RdBu_r",
            vmin=-vlim_i,
            vmax=vlim_i,
            shading="auto",
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(
            f"IC {i + 1} — Time × Frequency Map (outer product)",
            fontsize=10,
        )
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time × Frequency Maps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap_mean(
    freq_chan_view: np.ndarray,
    info,
    n_channels: int,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (d) — Mean component channel loading (topomap) for ALL ICs.

    chan_loading[c, k] = mean_f  components_2d[k, f, c]              (C, K)

    Given ``freq_chan_view`` of shape ``(F, C, K)``, this is the mean over F.
    Each component is plotted with its own symmetric color scale.
    """
    chan_loading = freq_chan_view.mean(axis=0)  # (C, K)

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(topo_info.ch_names):
        topo_info = mne.pick_info(topo_info, list(range(n_channels)))

    n_show = n_ica

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        vlim_i = np.percentile(np.abs(chan_loading[:, i]), 99)
        im, _ = plot_topomap(
            chan_loading[:, i],
            topo_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim_i, vlim_i),
        )
        ax.set_title(f"IC {i + 1}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Mean Component Channel Loading (topomap) — {label}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_mean_variance_over_time(
    subj_time_view: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (e) — Mean & variance of IC signal over time across subjects."""
    mean_temporal = subj_time_view.mean(axis=1)  # (K, T)
    var_temporal = subj_time_view.var(axis=1)  # (K, T)
    std_temporal = np.sqrt(var_temporal)  # (K, T)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, mean_temporal[i], lw=0.9, color="darkorange", label="mean")
        ax.fill_between(
            time,
            mean_temporal[i] - std_temporal[i],
            mean_temporal[i] + std_temporal[i],
            alpha=0.25,
            color="darkorange",
            label="± √variance",
        )
        ax.set_ylabel(f"IC {i + 1}")
        ax.set_title(
            f"Component {i + 1} — Mean & Variance Across Subjects",
            fontsize=10,
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Per-IC Mean and Variance of Activation Over Time — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_time_heatmap(
    subj_time_view: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (g) — Time × Subject activation heatmap per component."""
    n_subjects = subj_time_view.shape[1]
    n_show = n_ica

    vlim = float(np.percentile(np.abs(subj_time_view[:n_show]), 99))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    mesh = None
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time,
            np.arange(n_subjects),
            subj_time_view[i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(f"IC {i + 1} — Time × Subject Activation", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="activation")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time × Subject Activation Heatmaps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_freq_channel_heatmap(
    freq_chan_view: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (h) — Frequency × Channel heatmap of ICA components per IC.

    In this variant the ICA components themselves live in (F, C), so each
    IC's heatmap is read off ``components_2d`` directly (here via
    ``freq_chan_view`` of shape ``(F, C, K)``). Each component uses its own
    symmetric color scale.
    """
    n_freqs, n_channels, _ = freq_chan_view.shape
    n_show = n_ica
    channels = np.arange(n_channels)

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4.5), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # freq_chan_view[:, :, i] is (F, C); transpose so x = frequency, y = channel.
        data_i = freq_chan_view[:, :, i].T  # (C, F)
        vlim_i = float(np.percentile(np.abs(data_i), 99))
        mesh = ax.pcolormesh(
            freqs,
            channels,
            data_i,
            cmap="RdBu_r",
            vmin=-vlim_i,
            vmax=vlim_i,
            shading="auto",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"IC {i + 1}", fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="loading")

    axes[0].set_ylabel("Channel")
    fig.suptitle(
        f"Frequency × Channel ICA Component Heatmaps — {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_isc_bar(
    subj_time_view: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (i) — Bar plot of mean LOO-ISC across participants per IC.

    Per-subject vector is each subject's temporal score profile
    ``scores_3d[s, :, k]`` (length T). Bars whose across-subject mean
    LOO-ISC is negative are coloured red; positive bars are steel blue.
    """
    n_subjects = subj_time_view.shape[1]
    loo_isc_per_subject = np.zeros((n_ica, n_subjects))
    for k in range(n_ica):
        vecs = subj_time_view[k]  # (S, T)
        for s in range(n_subjects):
            others_mean = np.delete(vecs, s, axis=0).mean(axis=0)
            loo_isc_per_subject[k, s] = float(pearsonr(vecs[s], others_mean)[0])

    loo_isc_mean = loo_isc_per_subject.mean(axis=1)  # (K,)
    loo_isc_std = loo_isc_per_subject.std(axis=1)  # (K,)
    bar_colors = ["firebrick" if m < 0 else "steelblue" for m in loo_isc_mean]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_ica), 4.5))
    xs = np.arange(n_ica)
    ax.bar(xs, loo_isc_mean, yerr=loo_isc_std, color=bar_colors, capsize=4)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"IC {k + 1}" for k in range(n_ica)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Per-IC Mean LOO-ISC Across Participants — {label}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_sliding_variants(values: list[str]) -> list[tuple[float, float]]:
    """Parse 'window_sec:step_sec' strings into (window, step) float tuples."""
    parsed: list[tuple[float, float]] = []
    for raw in values:
        if ":" not in raw:
            raise ValueError(
                f"--sliding_variants entry {raw!r} must use 'window_sec:step_sec'."
            )
        win_s, step_s = raw.split(":", 1)
        parsed.append((float(win_s), float(step_s)))
    return parsed


def _compute_sliding_loo_isc(
    subj_time_view: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float,
) -> dict:
    """Per-subject and across-subject LOO-ISC for one (window, step) pair.

    subj_time_view: (K, S, T)
    Returns dict with per_subject (K, W, S), mean (K, W), std (K, W), edges (W,).
    """
    n_ica, n_subjects, n_times = subj_time_view.shape
    win_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    if win_samples < 2 or win_samples > n_times:
        raise ValueError(
            f"window_sec={window_sec}s -> {win_samples} samples; "
            f"must be in [2, {n_times}]."
        )
    if step_samples < 1:
        raise ValueError(
            f"step_sec={step_sec}s -> {step_samples} samples; must be ≥ 1."
        )

    starts = np.arange(0, n_times - win_samples + 1, step_samples)
    edges = starts / sfreq

    per_subject = np.zeros((n_ica, len(starts), n_subjects))
    for k in range(n_ica):
        comp = subj_time_view[k]  # (S, T)
        for w_idx, start in enumerate(starts):
            win = comp[:, start : start + win_samples]
            for s in range(n_subjects):
                others_mean = np.delete(win, s, axis=0).mean(axis=0)
                per_subject[k, w_idx, s] = float(pearsonr(win[s], others_mean)[0])

    return {
        "per_subject": per_subject,
        "mean": per_subject.mean(axis=2),
        "std": per_subject.std(axis=2),
        "edges": edges,
    }


def _close_to_end(
    edges: np.ndarray, vals: np.ndarray, t_end: float
) -> tuple[np.ndarray, np.ndarray]:
    """Append t_end and duplicate last value so steps-post extends to t_end."""
    if edges[-1] >= t_end:
        return edges, vals
    return np.append(edges, t_end), np.append(vals, vals[-1])


def _plot_sliding_window_loo_isc(
    subj_time_view: np.ndarray,
    sfreq: float,
    n_ica: int,
    sliding_variants: list[tuple[float, float]],
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (f) — Sliding-window LOO-ISC per IC for a single (win, step).

    If multiple variants are supplied, the one with the smallest step is used.
    """
    win, step = min(sliding_variants, key=lambda v: v[1])

    n_times = subj_time_view.shape[2]
    t_end = n_times / sfreq

    v = _compute_sliding_loo_isc(subj_time_view, sfreq, win, step)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    edges_orig = v["edges"]
    edges_end = np.append(edges_orig[1:], t_end)

    for i, ax in enumerate(axes):
        mean_orig = v["mean"][i]
        edges_m, mean_m = _close_to_end(edges_orig, mean_orig, t_end)
        _, std_m = _close_to_end(edges_orig, v["std"][i], t_end)

        thr_i = float(np.percentile(mean_orig, 100.0 - HIGHLIGHT_TOP_PERCENT))
        sig_mask = mean_orig >= thr_i
        for w_idx, is_sig in enumerate(sig_mask):
            if is_sig:
                ax.axvspan(
                    edges_orig[w_idx],
                    edges_end[w_idx],
                    color="gold",
                    alpha=0.3,
                    linewidth=0,
                    zorder=0,
                )

        ax.plot(
            edges_m,
            mean_m,
            lw=1.2,
            color="steelblue",
            drawstyle="steps-post",
            label="mean" if i == 0 else None,
            zorder=3,
        )
        ax.fill_between(
            edges_m,
            mean_m - std_m,
            mean_m + std_m,
            alpha=0.2,
            color="steelblue",
            step="post",
            label="± √variance" if i == 0 else None,
            zorder=2,
        )
        ax.axhline(0.0, ls="--", lw=0.6, color="gray")
        ax.axhline(
            thr_i,
            ls="--",
            lw=0.9,
            color="tomato",
            zorder=3,
            label=(
                f"top {HIGHLIGHT_TOP_PERCENT:.0f}% cutoff (r={thr_i:.2f})"
                if i == 0
                else None
            ),
        )
        ax.set_xlim(0.0, t_end)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"IC {i + 1}\nLOO-ISC")
        ax.set_title(f"Component {i + 1} — Sliding-Window LOO-ISC", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Per-IC Sliding-Window LOO-ISC (win={win:.1f}s, step={step:.1f}s) — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def _run_frequency_channel(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    sliding_variants: list[tuple[float, float]],
    save_dir: Path,
    band: str | None = None,
    skip_pca: bool = False,
) -> None:
    """Run the full frequency–channel ICA pipeline and save all plots.

    When ``band`` is given the outputs land in ``bands/frequency_channel/`` with
    filenames prefixed by ``<band>_``. Otherwise the broadband layout is used.
    """
    if band is None:
        out_dir = save_dir / "broadband" / "frequency_channel"
        prefix = ""
    else:
        out_dir = save_dir / "bands" / "frequency_channel"
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] Frequency-Channel: {data_4d.shape}  sfreq={sfreq} Hz")

    # Step 1 — Z-score and reshape: (S,C,F,T) → (S,T,F,C) → (S*T, F*C)
    bb_z = zscore_by_time(data_4d)
    bb_z_st = bb_z.transpose(0, 3, 2, 1)  # (S, T, F, C)
    X_st = bb_z_st.reshape(n_subjects * n_times, n_freqs * n_channels)
    _logger.info(f"[{label}] Reshaped: {X_st.shape}  (S*T, F*C)")

    # Step 2 — (optional) PCA + ICA
    ica = FastICA(
        n_components=n_ica,
        random_state=random_state,
        max_iter=500,
        whiten="unit-variance",
    )

    if skip_pca:
        _logger.info(f"[{label}] Skipping PCA; running FastICA directly on X_st.")
        ica_scores = ica.fit_transform(X_st)  # (S*T, K)
        ica_components = ica.components_  # (K, F*C)
    else:
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca_scores = pca.fit_transform(X_st)
        _logger.info(
            f"[{label}] PCA: {pca_scores.shape}, "
            f"explained={np.cumsum(pca.explained_variance_ratio_)[-1] * 100:.1f}%"
        )
        ica_scores = ica.fit_transform(pca_scores)
        ica_components = ica.components_ @ pca.components_  # (K, F*C)

    # Reshape to natural per-axis layouts
    scores_3d = ica_scores.reshape(n_subjects, n_times, n_ica)  # (S, T, K)
    components_2d = ica_components.reshape(n_ica, n_freqs, n_channels)  # (K, F, C)
    _logger.info(
        f"[{label}] ICA: scores_3d={scores_3d.shape}, "
        f"components_2d={components_2d.shape}"
    )

    # Views in the shapes expected by the shared plot helpers.
    # subj_time_view: per-IC subject × time map (used by ISC, sliding LOO-ISC,
    # loadings bar, mean/var-over-time, time × subject heatmap, LOO-ISC bar).
    subj_time_view = scores_3d.transpose(2, 0, 1)  # (K, S, T)
    # freq_chan_view: per-IC freq × channel map (used by topomap, freq × chan
    # heatmap, time × frequency outer product).
    freq_chan_view = components_2d.transpose(1, 2, 0)  # (F, C, K)

    # Plot 1 — PCA scree (skipped when PCA is not run)
    if not skip_pca:
        _plot_pca_scree(
            pca.explained_variance_ratio_,
            label=label,
            save_path=out_dir / f"{prefix}pca_scree_{label}.png",
        )

    # Per-IC subject × subject ISC matrices from the temporal score profiles.
    corr_mats = np.zeros((n_ica, n_subjects, n_subjects))
    for k in range(n_ica):
        corr_mats[k] = np.corrcoef(subj_time_view[k])  # (S, T) → (S, S)

    # Plot 2 — (a) ISC matrix
    _plot_isc_matrix(
        corr_mats,
        label=label,
        suptitle="Intersubject Correlation of IC Temporal Profiles",
        save_path=out_dir / f"{prefix}isc_component_matrix_{label}.png",
    )

    # Plot 2a — Overall co-clustering matrix aggregated over ICs and thresholds
    _plot_overall_cluster_matrix(
        corr_mats,
        label=label,
        save_path=out_dir / f"{prefix}isc_overall_cluster_matrix_{label}.png",
    )

    # CSV export of per-(IC, threshold) clusters with ≥2 members
    _export_isc_clusters_csv(
        corr_mats,
        save_path=out_dir / f"{prefix}ica_isc_clusters_{label}.csv",
    )

    # Plot 3 — (b) Subject loadings bar plot
    _plot_subject_loadings(
        subj_time_view,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_loadings_{label}.png",
    )

    # Plot 4 — (c) Time × Frequency map (outer product)
    _plot_time_frequency(
        freq_chan_view,
        subj_time_view,
        time,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_time_frequency_{label}.png",
    )

    # Plot 5 — (d) Mean component channel loading topomap
    _plot_topomap_mean(
        freq_chan_view,
        info,
        n_channels,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_topomap_mean_{label}.png",
    )

    # Plot 6 — (e) Mean & variance of IC signal over time across subjects
    _plot_mean_variance_over_time(
        subj_time_view,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_mean_variance_over_time_{label}.png",
    )

    # Plot 7 — (f) Sliding-window LOO-ISC per IC, smallest-step variant
    _plot_sliding_window_loo_isc(
        subj_time_view,
        sfreq,
        n_ica,
        sliding_variants,
        label=label,
        save_path=out_dir / f"{prefix}ica_sliding_window_loo_isc_{label}.png",
    )

    # Plot 8 — (g) Time × Subject activation heatmap per IC
    _plot_subject_time_heatmap(
        subj_time_view,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_time_heatmap_{label}.png",
    )

    # Plot 9 — (h) Frequency × Channel heatmap of ICA components per IC
    _plot_freq_channel_heatmap(
        freq_chan_view,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_freq_channel_heatmap_{label}.png",
    )

    # Plot 10 — (i) Mean LOO-ISC across participants per IC (bar plot)
    _plot_loo_isc_bar(
        subj_time_view,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_loo_isc_bar_{label}.png",
    )

    n_plots = 11 if not skip_pca else 10
    _logger.info(
        f"[{label}] Frequency-Channel: {n_plots} plots + cluster CSV saved to {out_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    experiment_name = ExperimentNames(args.experiment)
    condition = ConditionVariants(args.condition)
    if args.music_type is not None:
        music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    elif experiment_name == ExperimentNames.ASSR:
        # ASSR has no music dimension; uses a single placeholder "music type".
        music_types = [MusicTypeVariants.ASSR]
    else:
        music_types = [MusicTypeVariants.CLASSICAL, MusicTypeVariants.PSYTRANCE]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    wavelet_dir = (
        Path(args.wavelet_data_dir)
        if args.wavelet_data_dir is not None
        else ProjectPaths.PROCESSED_DATA_DIR / experiment_name.value / "wavelets"
    )
    freqs = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )
    sliding_variants = _parse_sliding_variants(args.sliding_variants)
    band_name = args.band  # None or e.g. "alpha"

    _logger.info(
        f"Frequency-Channel ICA: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"n_pca={args.n_pca}, n_ica={args.n_ica}, "
        f"band={band_name or 'broadband'}"
    )

    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=False,
        experiment_name=experiment_name,
    )
    datasets = analyzers_to_datasets(analyzers)

    for mt in music_types:
        mt_label = mt.value
        dataset_key = f"{condition.value}_{mt_label}"
        if dataset_key not in datasets:
            _logger.warning(f"No data for {dataset_key!r}; skipping.")
            continue

        ad = datasets[dataset_key]
        analyzer = analyzers.get(dataset_key)
        info = getattr(analyzer, "info", None) if analyzer is not None else None

        _logger.info(
            f"Dataset [{dataset_key}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz"
        )

        wd = _broadband_wavelet_4d(
            ad,
            dataset_key,
            representation="power",
            freqs=freqs,
            wavelet_dir=(wavelet_dir / "broadband"),
            reuse_wavelets=args.reuse_wavelets,
        )

        if band_name is None:
            data_4d = wd.data
            ica_freqs = freqs
        else:
            band_lo, band_hi = FREQUENCY_BANDS[band_name]
            band_mask = (freqs >= band_lo) & (freqs <= band_hi)
            if not band_mask.any():
                raise ValueError(
                    f"No broadband frequencies fall in {band_name} range "
                    f"[{band_lo}, {band_hi}] Hz; got freqs={freqs}."
                )
            data_4d = wd.data[:, :, band_mask, :]
            ica_freqs = freqs[band_mask]
            _logger.info(
                f"[{dataset_key}] band={band_name}: sliced "
                f"{wd.data.shape} -> {data_4d.shape} "
                f"(freqs {ica_freqs[0]:.1f}-{ica_freqs[-1]:.1f} Hz)"
            )

        save_dir = save_root / _STAGE_DIR / dataset_key
        _run_frequency_channel(
            data_4d,
            wd.sfreq,
            ica_freqs,
            info,
            label=dataset_key,
            n_pca=args.n_pca,
            n_ica=args.n_ica,
            random_state=args.random_state,
            sliding_variants=sliding_variants,
            save_dir=save_dir,
            band=band_name,
            skip_pca=args.skip_pca,
        )

    _logger.info("Frequency-Channel ICA analysis complete.")
