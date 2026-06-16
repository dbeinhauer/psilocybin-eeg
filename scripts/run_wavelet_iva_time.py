"""
CLI script reproducing the analyses from
``notebooks/05-wavelet-iva-analysis/wavelet_iva_time.ipynb``
(subjects-as-datasets, time is the independent dimension:
K = S, features = T, samples = C × F; per-subject PCA → IVA-G).
Transposed companion of run_wavelet_iva_frequency_channel.py.

Produces **all** IVA components (not just the first 10 shown in the
notebook) and writes every plot into the canonical per-condition
layout under::

    plots/05-time-wavelet-iva-analysis/<Condition>_<MusicType>/
        broadband/iva_time/pca_<n_pca>/               # default (no --band)
            pca_scree_<label>.png
            iva_component_ranking_<label>.png
            iva_component_isc_time_<label>_top.png
            iva_component_isc_time_<label>_bottom.png
            ...
        bands/iva_time/pca_<n_pca>/                   # when --band <name> is given
            <band>_pca_scree_<label>.png
            ...

The ``pca_<n_pca>/`` sub-folder isolates outputs by the per-subject PCA
dimensionality so multiple ``--n_pca`` sweeps don't overwrite each other.

Components are ranked by the mean off-diagonal value of the per-component
``Sigma_N`` correlation matrix returned by ``iva_g`` (the IVA model's own
estimate of how strongly the kth SCV couples across subjects); the top
--n_top and bottom --n_bottom are then visualised separately so the
bottom set serves as a noise / no-alignment contrast.

Pass ``--band <name>`` to slice the cached broadband wavelets down to
a single frequency band (alpha/beta/...) before the IVA step — the
same broadband cache is reused, no separate per-band cache is needed.

Usage::

    # broadband
    python scripts/run_wavelet_iva_time.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 50 --n_top 10 --n_bottom 5 --reuse_wavelets

    # alpha-band only
    python scripts/run_wavelet_iva_time.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --band alpha --n_pca 20 --n_top 10 --n_bottom 5 --reuse_wavelets
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
from independent_vector_analysis import iva_g  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from mne.viz import plot_topomap  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    FREQUENCY_BANDS,
    _broadband_wavelet_4d,
    analyzers_to_datasets,
    load_analyzers,
)
from src.analysis.wavelet_ica import (  # noqa: E402
    align_iva_component_signs,
    zscore_by_time,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    FrequencyBandNames,
    MusicTypeVariants,
)

_logger = logging.getLogger(__name__)

_STAGE_DIR = "05-time-wavelet-iva-analysis"

# ---------------------------------------------------------------------------
# Cluster-analysis constants and helpers (shared with the ICA scripts)
# ---------------------------------------------------------------------------

ISC_CLUSTER_THRESHOLDS: tuple[float, ...] = (0.3, 0.5, 0.7)

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
            "Run the time-as-independent IVA decomposition on wavelet "
            "power. Produces the same plots as wavelet_iva_subjects.ipynb, "
            "with all the analyses from the notebook (PCA scree, ISC matrices + clusters, dual LOO-ISC bars, topomaps, time × frequency map, score timecourse mean ± √variance, pair-space heatmaps, subject loadings)."
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
        help="Per-subject PCA dim before IVA-G (square mixing requirement).",
    )
    parser.add_argument(
        "--n_top",
        type=int,
        default=10,
        help=(
            "Number of top-ranked IVA components (by mean off-diagonal "
            "Sigma_N correlation) to visualise."
        ),
    )
    parser.add_argument(
        "--n_bottom",
        type=int,
        default=5,
        help=(
            "Number of bottom-ranked IVA components to visualise alongside "
            "the top components as a noise contrast."
        ),
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for per-subject PCA and IVA W_init.",
    )
    parser.add_argument(
        "--iva_opt_approach",
        choices=["gradient", "newton", "quasi"],
        default="newton",
        help="IVA-G optimisation method.",
    )
    parser.add_argument(
        "--iva_max_iter",
        type=int,
        default=1024,
        help="Maximum IVA-G iterations.",
    )
    parser.add_argument(
        "--iva_w_diff_stop",
        type=float,
        default=1e-6,
        help="IVA-G convergence threshold on |ΔW|.",
    )
    parser.add_argument(
        "--wavelet_freq_min",
        type=float,
        default=1.0,
        help="Minimum Morlet frequency (Hz).",
    )
    parser.add_argument(
        "--wavelet_freq_max",
        type=float,
        default=40.0,
        help="Maximum Morlet frequency (Hz). Capped at 40 Hz to match the "
        "Stage-04 ICA upper bound (higher frequencies are not computationally "
        "feasible here).",
    )
    parser.add_argument(
        "--wavelet_n_freqs",
        type=int,
        default=40,
        help="Number of Morlet frequency steps (≈ 1 Hz resolution by default).",
    )
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
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the cached broadband wavelet "
            "tensor is sliced to the band's frequency range before IVA, and "
            "plots are written to 'bands/iva/' with a '<band>_' filename prefix."
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
# Plot helpers — exact replicas of the notebook analyses, for ALL components
# ---------------------------------------------------------------------------


def _plot_pca_scree(
    pca_evr: np.ndarray,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Across-subject per-subject PCA scree + cumulative variance."""
    pca_evr_mean = pca_evr.mean(axis=0)
    pca_evr_std = pca_evr.std(axis=0)
    pca_evr_cum_mean = np.cumsum(pca_evr_mean)
    n_subjects, n_pca = pca_evr.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(1, n_pca + 1)
    axes[0].bar(xs, pca_evr_mean, yerr=pca_evr_std, color="steelblue", capsize=2)
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"PCA Scree (mean ± std across subjects) — {label}")

    for k in range(n_subjects):
        axes[1].plot(xs, np.cumsum(pca_evr[k]), lw=0.6, alpha=0.5, color="gray")
    axes[1].plot(xs, pca_evr_cum_mean, "o-", color="coral", label="mean cumulative")
    axes[1].axhline(0.9, ls="--", color="gray", label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance — {label}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_ranking(
    rank_score: np.ndarray,
    top_indices: list[int],
    bottom_indices: list[int],
    *,
    label: str,
    save_path: Path,
) -> None:
    """Bar chart of all per-IC ranking scores with top (blue) / bottom (red)."""
    n_pca = len(rank_score)
    colors = ["lightgray"] * n_pca
    for k in top_indices:
        colors[k] = "steelblue"
    for k in bottom_indices:
        colors[k] = "firebrick"

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * n_pca), 4.5))
    xs = np.arange(n_pca)
    ax.bar(xs, rank_score, color=colors)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{k + 1}" for k in range(n_pca)], fontsize=7)
    ax.set_xlabel("IVA component index")
    ax.set_ylabel("Mean off-diagonal Sigma_N correlation")
    ax.set_title(
        f"Component Ranking by Sigma_N Mean Off-Diagonal Correlation — "
        f"{label}  "
        f"(top {len(top_indices)} = blue, bottom {len(bottom_indices)} = red)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_isc_grid(
    corr_per_comp: np.ndarray,
    *,
    labels: list[str],
    fig_title: str,
    save_path: Path,
) -> None:
    """Per-IC subject × subject correlation grid + cluster strip beneath each."""
    n_show, n_subjects, _ = corr_per_comp.shape
    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(2.6 * n_show, 5.5),
        gridspec_kw={"height_ratios": [3, 1.2]},
        constrained_layout=True,
    )
    if n_show == 1:
        axes = axes.reshape(2, 1)

    im_corr = None
    for i in range(n_show):
        corr_mat = corr_per_comp[i]
        ax_top = axes[0, i]
        im_corr = ax_top.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax_top.set_xticks(range(n_subjects))
        ax_top.set_yticks(range(n_subjects))
        ax_top.set_xticklabels(
            [f"S{s + 1}" for s in range(n_subjects)], fontsize=7
        )
        ax_top.set_yticklabels(
            [f"S{s + 1}" for s in range(n_subjects)], fontsize=7
        )
        ax_top.set_title(labels[i], fontsize=8)

        ax_bot = axes[1, i]
        grid = _cluster_grid(corr_mat, n_subjects)
        ax_bot.imshow(grid, cmap=_CLUSTER_CMAP, norm=_CLUSTER_NORM, aspect="auto")
        _annotate_cluster_grid(ax_bot, grid)
        ax_bot.set_xticks(range(n_subjects))
        ax_bot.set_xticklabels(
            [f"S{s + 1}" for s in range(n_subjects)], fontsize=7
        )
        ax_bot.set_yticks(range(len(ISC_CLUSTER_THRESHOLDS)))
        ax_bot.set_yticklabels(
            [f"r≥{thr}" for thr in ISC_CLUSTER_THRESHOLDS], fontsize=8
        )
        if i == 0:
            ax_bot.set_ylabel("Threshold")

    fig.suptitle(fig_title, fontsize=12)
    if im_corr is not None:
        fig.colorbar(im_corr, ax=axes[0, -1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_isc_bar(
    subj_time_view: np.ndarray,
    *,
    labels: list[str],
    n_top: int,
    label: str,
    save_path: Path,
) -> None:
    """Whole-recording LOO-ISC per IC over a (K, S, T) view.

    Bars cover top + bottom in one figure with a separator at ``n_top``;
    top bars get strong colours, bottom bars get muted colours.
    """
    n_show, n_subjects, _ = subj_time_view.shape
    loo_isc = np.zeros((n_show, n_subjects))
    for k in range(n_show):
        vecs = subj_time_view[k]  # (S, T)
        for s in range(n_subjects):
            others_mean = np.delete(vecs, s, axis=0).mean(axis=0)
            loo_isc[k, s] = float(pearsonr(vecs[s], others_mean)[0])

    loo_mean = loo_isc.mean(axis=1)
    loo_std = loo_isc.std(axis=1)
    bar_colors: list[str] = []
    for i, m in enumerate(loo_mean):
        if i < n_top:
            bar_colors.append("firebrick" if m < 0 else "steelblue")
        else:
            bar_colors.append("salmon" if m < 0 else "lightsteelblue")

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * n_show), 5.0))
    xs = np.arange(n_show)
    ax.bar(xs, loo_mean, yerr=loo_std, color=bar_colors, capsize=4)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    if 0 < n_top < n_show:
        ax.axvline(
            n_top - 0.5,
            ls="--",
            lw=0.9,
            color="black",
            alpha=0.5,
            label=f"top {n_top} | bottom {n_show - n_top}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"Per-IC Mean LOO-ISC — Time Dimension (component timecourses) — {label}"
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap_mean_var(
    chan_loading: np.ndarray,
    info,
    n_channels: int,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC mean (signed) and variance (non-negative) topomaps across subjects.

    ``chan_loading`` shape ``(S, n_show, C)`` — already collapsed over freq.
    """
    chan_mean = chan_loading.mean(axis=0)  # (n_show, C)
    chan_var = chan_loading.var(axis=0)  # (n_show, C)
    n_show = len(labels)

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(topo_info.ch_names):
        topo_info = mne.pick_info(topo_info, list(range(n_channels)))

    fig, axes = plt.subplots(2, n_show, figsize=(3.0 * n_show, 7.0))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        vlim_m = float(np.percentile(np.abs(chan_mean[i]), 99))
        if vlim_m == 0.0:
            vlim_m = 1e-12
        im_m, _ = plot_topomap(
            chan_mean[i],
            topo_info,
            axes=axes[0, i],
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim_m, vlim_m),
        )
        axes[0, i].set_title(labels[i], fontsize=8)
        fig.colorbar(im_m, ax=axes[0, i], fraction=0.046, pad=0.04)

        vlim_v = float(np.percentile(chan_var[i], 99))
        if vlim_v == 0.0:
            vlim_v = 1e-12
        im_v, _ = plot_topomap(
            chan_var[i],
            topo_info,
            axes=axes[1, i],
            show=False,
            cmap="viridis",
            vlim=(0.0, vlim_v),
        )
        fig.colorbar(im_v, ax=axes[1, i], fraction=0.046, pad=0.04)

    fig.text(
        0.01, 0.75, "Mean across subjects", rotation=90, va="center",
        fontsize=11, fontweight="bold",
    )
    fig.text(
        0.01, 0.25, "Variance across subjects", rotation=90, va="center",
        fontsize=11, fontweight="bold",
    )
    fig.suptitle(
        f"Mean and Variance Topomaps Across Subjects — {label}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_tf_map(
    freq_profiles: np.ndarray,
    time_profiles: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC outer-product time × frequency map (rank-1).

    ``freq_profiles`` shape ``(F, K)``; ``time_profiles`` shape ``(K, T)``;
    K = len(labels) (the selected subset, pre-sliced by the caller).
    """
    ft_maps = np.einsum("fk,kt->kft", freq_profiles, time_profiles)
    n_show = len(labels)

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = ft_maps[i]
        vlim_i = max(float(np.percentile(np.abs(data_i), 99)), 1e-12)
        mesh = ax.pcolormesh(
            time, freqs, data_i, cmap="RdBu_r",
            vmin=-vlim_i, vmax=vlim_i, shading="auto",
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(
            f"{labels[i]} — Time × Frequency (outer product)", fontsize=10,
        )
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Per-IC Time × Frequency Maps — {label}", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_timecourse_mean_var(
    sources_view: np.ndarray,
    time: np.ndarray,
    *,
    labels: list[str],
    title_suffix: str,
    label: str,
    save_path: Path,
) -> None:
    """Per-IC mean ± √variance source timecourse over a (datasets, K, T) view."""
    mean_temporal = sources_view.mean(axis=0)  # (K, T)
    var_temporal = sources_view.var(axis=0)
    std_temporal = np.sqrt(var_temporal)
    n_show = len(labels)

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
        ax.set_ylabel(labels[i], fontsize=8)
        ax.set_title(
            f"{labels[i]} — Mean & Variance {title_suffix}",
            fontsize=10,
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Per-IC Component Timecourse — Mean & Variance {title_suffix} — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_isc_bar_fc(
    flat_patterns_per_comp: np.ndarray,
    *,
    labels: list[str],
    n_top: int,
    label: str,
    save_path: Path,
) -> None:
    """Whole-recording LOO-ISC over flattened (F × C) per IC.

    Same layout as ``_plot_loo_isc_bar`` but on the F × C axis.
    """
    n_show, n_subjects, _ = flat_patterns_per_comp.shape
    loo_isc = np.zeros((n_show, n_subjects))
    for k in range(n_show):
        vecs = flat_patterns_per_comp[k]
        for s in range(n_subjects):
            others_mean = np.delete(vecs, s, axis=0).mean(axis=0)
            loo_isc[k, s] = float(pearsonr(vecs[s], others_mean)[0])

    loo_mean = loo_isc.mean(axis=1)
    loo_std = loo_isc.std(axis=1)
    bar_colors: list[str] = []
    for i, m in enumerate(loo_mean):
        if i < n_top:
            bar_colors.append("firebrick" if m < 0 else "steelblue")
        else:
            bar_colors.append("salmon" if m < 0 else "lightsteelblue")

    fig, ax = plt.subplots(figsize=(max(10, 0.9 * n_show), 5.0))
    xs = np.arange(n_show)
    ax.bar(xs, loo_mean, yerr=loo_std, color=bar_colors, capsize=4)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    if 0 < n_top < n_show:
        ax.axvline(
            n_top - 0.5, ls="--", lw=0.9, color="black", alpha=0.5,
            label=f"top {n_top} | bottom {n_show - n_top}",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f"Per-IC Mean LOO-ISC — Frequency × Channel Dimension "
        f"(component patterns) — {label}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _safe_vlim(arr: np.ndarray) -> float:
    return max(float(np.percentile(np.abs(arr), 99)), 1e-12)


def _plot_pairmap_time_subject(
    time_subj: np.ndarray,
    time: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC Time × Subject heatmap stack."""
    n_show, n_subjects, _ = time_subj.shape
    vlim = _safe_vlim(time_subj)
    subject_labels = [f"S{s + 1}" for s in range(n_subjects)]

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time, np.arange(n_subjects), time_subj[i],
            cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels(subject_labels, fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(f"{labels[i]} — Time × Subject", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="component")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time × Subject per IC (from components) — {label}",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairmap_subject_x(
    data_per_comp: np.ndarray,
    y_values: np.ndarray,
    *,
    y_label: str,
    variant_name: str,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC heatmap stack with subject on x and ``y_values`` on y."""
    n_show, _, n_subjects = data_per_comp.shape
    vlim = _safe_vlim(data_per_comp)
    subjects_idx = np.arange(n_subjects)
    subject_labels = [f"S{s + 1}" for s in subjects_idx]

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 5.0), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            subjects_idx, y_values, data_per_comp[i],
            cmap="RdBu_r", vmin=-vlim, vmax=vlim, shading="auto",
        )
        ax.set_xticks(subjects_idx)
        ax.set_xticklabels(subject_labels, fontsize=8)
        ax.set_xlabel("Subject")
        ax.set_title(labels[i], fontsize=8)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="loading")
    axes[0].set_ylabel(y_label)
    fig.suptitle(f"{variant_name} per IC — {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pairmap_frequency_channel(
    freq_chan: np.ndarray,
    freqs: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC Frequency × Channel heatmap row."""
    n_show, n_freqs, n_channels = freq_chan.shape
    channels = np.arange(n_channels)

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4.5), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        data_i = freq_chan[i].T
        vlim_i = _safe_vlim(data_i)
        mesh = ax.pcolormesh(
            freqs, channels, data_i, cmap="RdBu_r",
            vmin=-vlim_i, vmax=vlim_i, shading="auto",
        )
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(labels[i], fontsize=8)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="loading")
    axes[0].set_ylabel("Channel")
    fig.suptitle(
        f"Frequency × Channel per IC (subject-avg scores) — {label}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_loadings(
    subject_loadings: np.ndarray,
    *,
    labels: list[str],
    label: str,
    save_path: Path,
) -> None:
    """Per-IC horizontal bar of mean |value| over time, one bar per subject."""
    n_subjects, n_show = subject_loadings.shape
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.barh(range(n_subjects), subject_loadings[:, i], color="darkorange")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels(
            [f"S{s + 1}" for s in range(n_subjects)], fontsize=8
        )
        ax.set_xlabel("|component|")
        ax.set_title(labels[i], fontsize=8)
    axes[0].set_ylabel("Subject")
    fig.suptitle(
        f"Per-Subject Mean Loading per Component — {label}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main IVA pipeline (subjects-as-datasets; time as the independent axis)
# ---------------------------------------------------------------------------


def _run_iva(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    label: str,
    n_pca: int,
    n_top: int,
    n_bottom: int,
    random_state: int,
    iva_opt_approach: str,
    iva_max_iter: int,
    iva_w_diff_stop: float,
    save_dir: Path,
    band: str | None,
) -> None:
    """Run the subjects-as-datasets IVA pipeline and save all plots."""
    pca_subdir = f"pca_{n_pca}"
    if band is None:
        out_dir = save_dir / "broadband" / "iva_time" / pca_subdir
        prefix = ""
    else:
        out_dir = save_dir / "bands" / "iva_time" / pca_subdir
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] IVA-time: {data_4d.shape}  sfreq={sfreq} Hz")

    n_show = n_top + n_bottom
    if n_show > n_pca:
        raise ValueError(
            f"--n_top + --n_bottom ({n_show}) must be ≤ --n_pca ({n_pca}); "
            f"IVA outputs exactly N_PCA components."
        )

    # Step 1 — Z-score along time, then per-subject transpose + reshape
    # so that time becomes the feature axis: (S, C, F, T) → (S, T, C, F) →
    # (S, T, C*F).
    bb_z = zscore_by_time(data_4d)
    n_samp = n_channels * n_freqs
    X_subjects = bb_z.transpose(0, 3, 1, 2).reshape(
        n_subjects, n_times, n_samp
    )
    _logger.info(
        f"[{label}] Per-subject reshape: {X_subjects.shape}  (S, T, C*F)"
    )

    # Step 2 — Per-subject PCA reduces the T (feature) axis → IVA input
    # (N_PCA, C*F, S).
    pcas: list[PCA] = []
    pca_scores = np.zeros((n_subjects, n_pca, n_samp))
    pca_evr = np.zeros((n_subjects, n_pca))
    for k in range(n_subjects):
        subj_matrix = X_subjects[k].T  # (C*F, T) — samples × features for sklearn
        pca = PCA(n_components=n_pca, random_state=random_state)
        scores = pca.fit_transform(subj_matrix)  # (C*F, n_pca)
        pcas.append(pca)
        pca_scores[k] = scores.T  # (n_pca, C*F)
        pca_evr[k] = pca.explained_variance_ratio_
    X_pca = np.ascontiguousarray(pca_scores.transpose(1, 2, 0))  # (n_pca, C*F, S)
    _logger.info(
        f"[{label}] PCA: X_pca={X_pca.shape}, "
        f"mean retained variance = {pca_evr.sum(axis=1).mean() * 100:.1f}%"
    )

    # Step 3 — Run IVA-G.
    rng = np.random.default_rng(random_state)
    W_init = rng.standard_normal((n_pca, n_pca, n_subjects))
    W, cost, Sigma_N, _isi = iva_g(
        X_pca,
        opt_approach=iva_opt_approach,
        whiten=True,
        verbose=False,
        W_init=W_init,
        max_iter=iva_max_iter,
        W_diff_stop=iva_w_diff_stop,
    )
    _logger.info(
        f"[{label}] IVA-G: iterations={len(cost)}  final cost={cost[-1]:.6f}"
    )

    # Step 3b — Resolve per-subject sign ambiguity. IVA recovers each component
    # only up to a per-subject sign; flip mismatched subjects (using the leading
    # eigenvector of each component's Sigma_N correlation matrix) so that W — and
    # everything recovered from it below — is sign-aligned across subjects.
    sigma_corr, W, sign_flips = align_iva_component_signs(Sigma_N, W)
    n_flipped = int((sign_flips < 0).sum())
    _logger.info(
        f"[{label}] Sign alignment: flipped {n_flipped} (component, subject) "
        f"pairs across {n_pca} components."
    )

    # Step 4 — Recover scores in the (F, C) sample subspace and components
    # in the original T feature space.
    iva_components = np.zeros((n_subjects, n_pca, n_times))
    iva_scores = np.zeros((n_subjects, n_pca, n_freqs, n_channels))
    for k in range(n_subjects):
        W_k = W[:, :, k]
        scores_flat = W_k @ X_pca[:, :, k]  # (n_pca, C*F)
        iva_scores[k] = scores_flat.reshape(
            n_pca, n_channels, n_freqs
        ).transpose(0, 2, 1)
        iva_components[k] = W_k @ pcas[k].components_  # (n_pca, T)
    _logger.info(
        f"[{label}] Recover: iva_components={iva_components.shape}, "
        f"iva_scores={iva_scores.shape}"
    )

    # Step 5 — Rank components by mean off-diagonal value of the per-component
    # Sigma_N correlation matrix (the IVA model's own estimate of how strongly
    # the kth SCV couples across subjects), then select top + bottom. ``sigma_corr``
    # is the sign-aligned correlation stack from Step 3b.
    off_diag_mask = ~np.eye(n_subjects, dtype=bool)
    rank_score = np.array(
        [sigma_corr[k][off_diag_mask].mean() for k in range(n_pca)]
    )

    order = np.argsort(rank_score)[::-1]
    top_indices = order[:n_top].tolist()
    bottom_indices = order[-n_bottom:][::-1].tolist() if n_bottom > 0 else []
    selected_indices = top_indices + bottom_indices

    def _ic_title(i: int) -> str:
        k = selected_indices[i]
        tag = f"TOP {i + 1}" if i < n_top else f"BOT {i - n_top + 1}"
        return f"{tag} (IC {k + 1}, r={rank_score[k]:+.2f})"

    selected_labels = [_ic_title(i) for i in range(len(selected_indices))]
    top_labels = selected_labels[:n_top]
    bot_labels = selected_labels[n_top:]
    _logger.info(
        f"[{label}] Top IC indices: {top_indices}  "
        f"(r={[round(rank_score[k], 3) for k in top_indices]})"
    )
    _logger.info(
        f"[{label}] Bottom IC indices: {bottom_indices}  "
        f"(r={[round(rank_score[k], 3) for k in bottom_indices]})"
    )

    # ---------- Plots ----------

    # (a) Component-ranking summary
    _plot_component_ranking(
        rank_score, top_indices, bottom_indices,
        label=label,
        save_path=out_dir / f"{prefix}iva_component_ranking_{label}.png",
    )

    # (b) PCA scree
    _plot_pca_scree(
        pca_evr,
        label=label,
        save_path=out_dir / f"{prefix}pca_scree_{label}.png",
    )

    # (c) Source ISC matrix per component + cluster strip — TOP / BOTTOM
    for indices, labels_grp, group_name, suffix in (
        (top_indices, top_labels, f"TOP {n_top}", "top"),
        (bottom_indices, bot_labels, f"BOTTOM {n_bottom}", "bottom"),
    ):
        source_corr = np.stack(
            [np.corrcoef(iva_components[:, k, :]) for k in indices]
        )
        _plot_isc_grid(
            source_corr,
            labels=labels_grp,
            fig_title=(
                f"Intersubject Correlation of IVA Component Timecourses — "
                f"{group_name} — {label}"
            ),
            save_path=out_dir / f"{prefix}iva_component_isc_time_{label}_{suffix}.png",
        )

    # (d) Pattern ISC matrix per component + cluster strip — TOP / BOTTOM
    for indices, labels_grp, group_name, suffix in (
        (top_indices, top_labels, f"TOP {n_top}", "top"),
        (bottom_indices, bot_labels, f"BOTTOM {n_bottom}", "bottom"),
    ):
        pattern_corr = np.zeros((len(indices), n_subjects, n_subjects))
        for i, k in enumerate(indices):
            flat = iva_scores[:, k, :, :].reshape(n_subjects, -1)
            pattern_corr[i] = np.corrcoef(flat)
        _plot_isc_grid(
            pattern_corr,
            labels=labels_grp,
            fig_title=(
                f"Intersubject Correlation of IVA Score (F × C) Patterns — "
                f"{group_name} — {label}"
            ),
            save_path=out_dir / f"{prefix}iva_score_isc_freq_channel_{label}_{suffix}.png",
        )

    # (d2) Sigma_N correlation matrix per component + cluster strip — TOP / BOTTOM.
    # Direct visualisation of the same numbers that drove the ranking in Step 5.
    for indices, labels_grp, group_name, suffix in (
        (top_indices, top_labels, f"TOP {n_top}", "top"),
        (bottom_indices, bot_labels, f"BOTTOM {n_bottom}", "bottom"),
    ):
        _plot_isc_grid(
            np.stack([sigma_corr[k] for k in indices]),
            labels=labels_grp,
            fig_title=(
                f"Subject × Subject Correlation from Sigma_N — "
                f"{group_name} — {label}"
            ),
            save_path=out_dir / f"{prefix}iva_sigma_n_corr_{label}_{suffix}.png",
        )

    # (e) Per-IC LOO-ISC bar (time axis) — single figure, top + bottom
    subj_time_view = iva_components[:, selected_indices, :].transpose(1, 0, 2)
    _plot_loo_isc_bar(
        subj_time_view,
        labels=selected_labels,
        n_top=n_top,
        label=label,
        save_path=out_dir / f"{prefix}iva_loo_isc_bar_time_{label}.png",
    )

    # (f) Mean & variance topomap (freq-collapsed channel loading) — TOP / BOTTOM
    chan_loading_all = iva_scores.mean(axis=2)  # (S, N_PCA, C)
    if info is not None:
        for indices, labels_grp, suffix in (
            (top_indices, top_labels, "top"),
            (bottom_indices, bot_labels, "bottom"),
        ):
            _plot_topomap_mean_var(
                chan_loading_all[:, indices, :],
                info,
                n_channels,
                labels=labels_grp,
                label=label,
                save_path=out_dir / f"{prefix}iva_topomap_mean_var_{label}_{suffix}.png",
            )
    else:
        _logger.warning(f"[{label}] No info available; skipping topomap plot.")

    # (g) Time × frequency outer-product map — TOP / BOTTOM
    mean_components_all = iva_scores.mean(axis=0)  # (N_PCA, F, C)
    time_profiles_all = iva_components.mean(axis=0)  # (N_PCA, T)
    for indices, labels_grp, suffix in (
        (top_indices, top_labels, "top"),
        (bottom_indices, bot_labels, "bottom"),
    ):
        freq_profiles_sel = (
            mean_components_all[indices].mean(axis=2).T  # (F, K_sel)
        )
        time_profiles_sel = time_profiles_all[indices]  # (K_sel, T)
        _plot_tf_map(
            freq_profiles_sel,
            time_profiles_sel,
            time,
            freqs,
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_time_frequency_{label}_{suffix}.png",
        )

    # (h) Source timecourse mean ± √variance across subjects — TOP / BOTTOM
    for indices, labels_grp, suffix in (
        (top_indices, top_labels, "top"),
        (bottom_indices, bot_labels, "bottom"),
    ):
        _plot_timecourse_mean_var(
            iva_components[:, indices, :],
            time,
            labels=labels_grp,
            title_suffix="Across Subjects",
            label=label,
            save_path=out_dir / f"{prefix}iva_mean_variance_over_time_{label}_{suffix}.png",
        )

    # (i) Mean LOO-ISC bar in the F × C dimension — single figure, top + bottom
    flat_patterns = iva_scores[:, selected_indices, :, :].reshape(
        n_subjects, n_show, -1
    ).transpose(1, 0, 2)  # (n_show, S, F*C)
    _plot_loo_isc_bar_fc(
        flat_patterns,
        labels=selected_labels,
        n_top=n_top,
        label=label,
        save_path=out_dir / f"{prefix}iva_loo_isc_bar_freq_channel_{label}.png",
    )

    # (j) Pair-space heatmaps — Time × Subject, Subject × Freq/Channel, Freq × Channel.
    for indices, labels_grp, suffix in (
        (top_indices, top_labels, "top"),
        (bottom_indices, bot_labels, "bottom"),
    ):
        time_subj = iva_components.transpose(1, 0, 2)[indices]  # (K, S, T)
        _plot_pairmap_time_subject(
            time_subj,
            time,
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_pairmap_time_subject_{label}_{suffix}.png",
        )

        subj_freq = iva_scores.mean(axis=-1).transpose(1, 2, 0)[indices]
        _plot_pairmap_subject_x(
            subj_freq,
            freqs,
            y_label="Frequency (Hz)",
            variant_name=f"Subject × Frequency (channel-avg scores)",
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_pairmap_subject_frequency_{label}_{suffix}.png",
        )

        subj_chan = iva_scores.mean(axis=-2).transpose(1, 2, 0)[indices]
        _plot_pairmap_subject_x(
            subj_chan,
            np.arange(n_channels),
            y_label="Channel",
            variant_name=f"Subject × Channel (frequency-avg scores)",
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_pairmap_subject_channel_{label}_{suffix}.png",
        )

        freq_chan = iva_scores.mean(axis=0)[indices]  # (K, F, C)
        _plot_pairmap_frequency_channel(
            freq_chan,
            freqs,
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_pairmap_frequency_channel_{label}_{suffix}.png",
        )

    # (k) Mean subject loading per component: mean |value| over time per subject
    subject_loadings_all = np.abs(iva_components).mean(axis=2)  # (S, N_PCA) — mean |component| over T
    for indices, labels_grp, suffix in (
        (top_indices, top_labels, "top"),
        (bottom_indices, bot_labels, "bottom"),
    ):
        _plot_subject_loadings(
            subject_loadings_all[:, indices],
            labels=labels_grp,
            label=label,
            save_path=out_dir / f"{prefix}iva_subject_loadings_{label}_{suffix}.png",
        )

    _logger.info(
        f"[{label}] IVA-time: 20+ plot files saved to {out_dir}"
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
    freqs_full = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )
    band_name = args.band

    _logger.info(
        f"IVA-time: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"n_pca={args.n_pca}, n_top={args.n_top}, n_bottom={args.n_bottom}, "
        f"band={band_name or 'broadband'}, "
        f"iva_opt={args.iva_opt_approach}"
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
            freqs=freqs_full,
            wavelet_dir=(wavelet_dir / "broadband"),
            reuse_wavelets=args.reuse_wavelets,
        )

        if band_name is None:
            data_4d = wd.data
            iva_freqs = freqs_full
        else:
            band_lo, band_hi = FREQUENCY_BANDS[band_name]
            band_mask = (freqs_full >= band_lo) & (freqs_full <= band_hi)
            if not band_mask.any():
                raise ValueError(
                    f"No broadband frequencies fall in {band_name} range "
                    f"[{band_lo}, {band_hi}] Hz; got freqs={freqs_full}."
                )
            data_4d = wd.data[:, :, band_mask, :]
            iva_freqs = freqs_full[band_mask]
            _logger.info(
                f"[{dataset_key}] band={band_name}: sliced "
                f"{wd.data.shape} -> {data_4d.shape} "
                f"(freqs {iva_freqs[0]:.1f}-{iva_freqs[-1]:.1f} Hz)"
            )

        save_dir = save_root / _STAGE_DIR / dataset_key
        _run_iva(
            data_4d,
            wd.sfreq,
            iva_freqs,
            info,
            label=dataset_key,
            n_pca=args.n_pca,
            n_top=args.n_top,
            n_bottom=args.n_bottom,
            random_state=args.random_state,
            iva_opt_approach=args.iva_opt_approach,
            iva_max_iter=args.iva_max_iter,
            iva_w_diff_stop=args.iva_w_diff_stop,
            save_dir=save_dir,
            band=band_name,
        )

    _logger.info("IVA-subjects analysis complete.")
