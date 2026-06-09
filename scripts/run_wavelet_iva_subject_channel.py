"""
CLI script reproducing the analyses from
``notebooks/05-wavelet-iva-analysis/wavelet_iva_subject_channel.ipynb``
where the **datasets passed to IVA are individual ``(subject, channel)``
pairs** rather than whole subjects: K = S × C, features = F, samples = T.

Produces the first ``--n_show`` IVA components and writes every plot
into the canonical per-condition layout::

    plots/05-wavelet-iva-analysis/<Condition>_<MusicType>/
        broadband/iva_subject_channel/
            pca_scree_<label>.png      (only when --use_pca)
            iva_source_isc_matrix_<label>.png
            iva_pattern_isc_matrix_<label>.png
            iva_loo_isc_bar_<label>.png
            iva_topomap_mean_var_<label>.png
            iva_time_frequency_<label>.png
            iva_mean_variance_over_time_<label>.png

The downstream summaries channel-average per subject first (sources and
patterns), so the ISC, LOO-ISC and topomap plots remain subject × subject
comparable with the subjects-as-datasets variant. The timecourse plot
summarises across **all `(S × C)` datasets** (stronger consistency
criterion than the subjects-only band).

Usage::

    python scripts/run_wavelet_iva_subject_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_show 10 --reuse_wavelets
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
    MusicTypeVariants,
)

_logger = logging.getLogger(__name__)

_STAGE_DIR = "05-wavelet-iva-analysis"

# ---------------------------------------------------------------------------
# Cluster-analysis constants and helpers
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
            "Run the (subject × channel)-as-datasets IVA decomposition on EEG "
            "wavelet power. Each (s, c) pair is one IVA dataset; features "
            "are frequencies, samples are time points."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        default=[
            MusicTypeVariants.CLASSICAL.value,
            MusicTypeVariants.PSYTRANCE.value,
        ],
        help="One or more music types to analyse.",
    )
    parser.add_argument(
        "--use_pca",
        action="store_true",
        help=(
            "Enable per-(s,c) PCA before IVA. Defaults off because the F "
            "feature dim (≈70 with 1 Hz resolution) is already small."
        ),
    )
    parser.add_argument(
        "--n_pca",
        type=int,
        default=30,
        help="Per-(s,c) PCA dim when --use_pca is set.",
    )
    parser.add_argument(
        "--n_show",
        type=int,
        default=10,
        help="Number of leading IVA components to plot.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Seed for per-(s,c) PCA and IVA W_init.",
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
        default=(
            ProjectPaths.PROCESSED_DATA_DIR
            / ExperimentNames.PSILO_MUSIC.value
            / "wavelets"
        ),
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
# Plot helpers
# ---------------------------------------------------------------------------


def _plot_pca_scree(
    pca_evr: np.ndarray,
    *,
    n_show: int,
    n_datasets: int,
    label: str,
    save_path: Path,
) -> None:
    """Per-(s,c) PCA scree + cumulative variance — only when --use_pca is set."""
    pca_evr_mean = pca_evr.mean(axis=0)
    pca_evr_std = pca_evr.std(axis=0)
    pca_evr_cum_mean = np.cumsum(pca_evr_mean)
    n_pca = pca_evr.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = np.arange(1, n_pca + 1)
    axes[0].bar(xs, pca_evr_mean, yerr=pca_evr_std, color="steelblue", capsize=2)
    axes[0].axvline(
        n_show + 0.5, ls="--", color="firebrick", label=f"first {n_show}"
    )
    axes[0].set_xlabel("PCA component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(
        f"PCA Scree (mean ± std over {n_datasets} (s,c)) — {label}"
    )
    axes[0].legend()

    for d in range(pca_evr.shape[0]):
        axes[1].plot(xs, np.cumsum(pca_evr[d]), lw=0.4, alpha=0.2, color="gray")
    axes[1].plot(xs, pca_evr_cum_mean, "o-", color="coral", label="mean cumulative")
    axes[1].axhline(0.9, ls="--", color="gray", label="90%")
    axes[1].axvline(
        n_show + 0.5, ls="--", color="firebrick", label=f"first {n_show}"
    )
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance — {label}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_isc_grid(
    corr_per_comp: np.ndarray,
    *,
    fig_title: str,
    save_path: Path,
) -> None:
    """Per-IC subject × subject correlation grid + cluster strip beneath each."""
    n_show, n_subjects, _ = corr_per_comp.shape
    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(2.8 * n_show, 5.5),
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
        ax_top.set_title(f"IC {i + 1}", fontsize=10)

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
    per_subject_source: np.ndarray,
    *,
    n_show: int,
    label: str,
    save_path: Path,
) -> None:
    """Whole-recording LOO-ISC per IC, computed on channel-averaged sources.

    ``per_subject_source`` shape ``(S, K, T)``.
    """
    n_subjects = per_subject_source.shape[0]
    loo_isc = np.zeros((n_show, n_subjects))
    for k in range(n_show):
        vecs = per_subject_source[:, k, :]  # (S, T)
        for s in range(n_subjects):
            others_mean = np.delete(vecs, s, axis=0).mean(axis=0)
            loo_isc[k, s] = float(pearsonr(vecs[s], others_mean)[0])

    loo_isc_mean = loo_isc.mean(axis=1)
    loo_isc_std = loo_isc.std(axis=1)
    bar_colors = ["firebrick" if m < 0 else "steelblue" for m in loo_isc_mean]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * n_show), 4.5))
    xs = np.arange(n_show)
    ax.bar(xs, loo_isc_mean, yerr=loo_isc_std, color=bar_colors, capsize=4)
    ax.axhline(0.0, ls="--", lw=0.6, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"IC {k + 1}" for k in range(n_show)])
    ax.set_xlabel("Component")
    ax.set_ylabel("Mean LOO-ISC across subjects")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"Per-IC Mean LOO-ISC (channel-averaged sources) — {label}")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap_mean_var(
    chan_loading: np.ndarray,
    info,
    n_channels: int,
    *,
    n_show: int,
    label: str,
    save_path: Path,
) -> None:
    """Mean / variance topomaps across subjects.

    ``chan_loading`` shape ``(S, C, K)`` — already collapsed over freq.
    """
    chan_mean = chan_loading[:, :, :n_show].mean(axis=0).T  # (n_show, C)
    chan_var = chan_loading[:, :, :n_show].var(axis=0).T  # (n_show, C)

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
        axes[0, i].set_title(f"IC {i + 1}", fontsize=10)
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
        0.01,
        0.75,
        "Mean across subjects",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.25,
        "Variance across subjects",
        rotation=90,
        va="center",
        fontsize=11,
        fontweight="bold",
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
    n_show: int,
    label: str,
    save_path: Path,
) -> None:
    """Per-IC outer-product time × frequency map (rank-1).

    ``freq_profiles`` shape ``(F, K)``; ``time_profiles`` shape ``(K, T)``.
    """
    ft_maps = np.einsum("fk,kt->kft", freq_profiles, time_profiles)

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = ft_maps[i]
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
            f"IC {i + 1} — Time × Frequency (outer product)",
            fontsize=10,
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
    n_show: int,
    title_suffix: str,
    label: str,
    save_path: Path,
) -> None:
    """Per-IC mean ± √variance source timecourse over a (datasets, K, T) view."""
    mean_temporal = sources_view.mean(axis=0)
    var_temporal = sources_view.var(axis=0)
    std_temporal = np.sqrt(var_temporal)

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
            f"Component {i + 1} — Mean & Variance {title_suffix}",
            fontsize=10,
        )
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Per-IC Source Timecourse — Mean & Variance {title_suffix} — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main IVA pipeline — (subject × channel) datasets
# ---------------------------------------------------------------------------


def _run_iva_subject_channel(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    label: str,
    use_pca: bool,
    n_pca: int,
    n_show: int,
    random_state: int,
    iva_opt_approach: str,
    iva_max_iter: int,
    iva_w_diff_stop: float,
    save_dir: Path,
) -> None:
    """Run the (subject × channel)-as-datasets IVA pipeline."""
    out_dir = save_dir / "broadband" / "iva_subject_channel"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = ""

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    n_datasets = n_subjects * n_channels
    _logger.info(
        f"[{label}] IVA-(S×C): data={data_4d.shape}  sfreq={sfreq} Hz  "
        f"K=S*C={n_datasets}"
    )

    # Step 1 — Z-score along time and flatten (S, C) → datasets axis.
    bb_z = zscore_by_time(data_4d)
    X_datasets = bb_z.reshape(n_datasets, n_freqs, n_times)
    _logger.info(
        f"[{label}] Per-(s,c) reshape: {X_datasets.shape}  ((S*C), F, T)"
    )

    # Step 2 — Optional per-dataset PCA → IVA input layout (N_IVA, T, S*C).
    pcas: list[PCA] = []
    if use_pca:
        if n_show > n_pca:
            raise ValueError(
                f"--n_show ({n_show}) must be ≤ --n_pca ({n_pca}) when --use_pca."
            )
        pca_scores = np.zeros((n_datasets, n_pca, n_times))
        pca_evr = np.zeros((n_datasets, n_pca))
        for d in range(n_datasets):
            mat = X_datasets[d].T  # (T, F)
            pca = PCA(n_components=n_pca, random_state=random_state)
            scores = pca.fit_transform(mat)
            pcas.append(pca)
            pca_scores[d] = scores.T
            pca_evr[d] = pca.explained_variance_ratio_
        X_iva = np.ascontiguousarray(pca_scores.transpose(1, 2, 0))
        n_iva = n_pca
        _logger.info(
            f"[{label}] Per-(s,c) PCA: F={n_freqs} → N_IVA={n_iva}, "
            f"mean retained variance = {pca_evr.sum(axis=1).mean() * 100:.1f}%"
        )
    else:
        if n_show > n_freqs:
            raise ValueError(
                f"--n_show ({n_show}) must be ≤ n_freqs ({n_freqs}) without --use_pca."
            )
        pca_evr = np.zeros((n_datasets, n_freqs))
        X_iva = np.ascontiguousarray(X_datasets.transpose(1, 2, 0))
        n_iva = n_freqs
        _logger.info(
            f"[{label}] PCA skipped — IVA on {n_freqs} frequency features per dataset."
        )

    # Step 3 — Run IVA-G.
    rng = np.random.default_rng(random_state)
    W_init = rng.standard_normal((n_iva, n_iva, n_datasets))
    W, cost, Sigma_N, _isi = iva_g(
        X_iva,
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

    # Step 3b — Resolve per-dataset sign ambiguity. IVA recovers each component
    # only up to a per-dataset sign; flip mismatched datasets (using the leading
    # eigenvector of each component's Sigma_N correlation matrix) so that W — and
    # everything recovered from it below — is sign-aligned across datasets.
    _sigma_corr, W, sign_flips = align_iva_component_signs(Sigma_N, W)
    n_flipped = int((sign_flips < 0).sum())
    _logger.info(
        f"[{label}] Sign alignment: flipped {n_flipped} (component, dataset) "
        f"pairs across {n_iva} components."
    )

    # Step 4 — Recover sources and frequency-pattern components.
    iva_sources_flat = np.zeros((n_datasets, n_iva, n_times))
    iva_components_flat = np.zeros((n_datasets, n_iva, n_freqs))
    for d in range(n_datasets):
        W_d = W[:, :, d]
        iva_sources_flat[d] = W_d @ X_iva[:, :, d]
        if use_pca:
            iva_components_flat[d] = W_d @ pcas[d].components_
        else:
            iva_components_flat[d] = W_d

    # Reshape K-axis (S*C,) → (S, C, …) for indexing by (subject, channel).
    iva_sources = iva_sources_flat.reshape(
        n_subjects, n_channels, n_iva, n_times
    )
    iva_components = iva_components_flat.reshape(
        n_subjects, n_channels, n_iva, n_freqs
    )
    _logger.info(
        f"[{label}] Recover: iva_sources={iva_sources.shape}  "
        f"iva_components={iva_components.shape}"
    )

    # ---------- Plots ----------

    # (a) Per-dataset PCA scree — only when PCA was applied.
    if use_pca:
        _plot_pca_scree(
            pca_evr,
            n_show=n_show,
            n_datasets=n_datasets,
            label=label,
            save_path=out_dir / f"{prefix}pca_scree_{label}.png",
        )
    else:
        _logger.info(f"[{label}] --use_pca not set; PCA scree skipped.")

    # Channel-averaged per-subject views.
    per_subject_source = iva_sources.mean(axis=1)  # (S, N_IVA, T)
    per_subject_pattern = iva_components.mean(axis=1)  # (S, N_IVA, F)

    # (b) Source ISC matrix per component (channel-averaged sources)
    source_corr = np.stack(
        [np.corrcoef(per_subject_source[:, k, :]) for k in range(n_show)]
    )
    _plot_isc_grid(
        source_corr,
        fig_title=(
            f"Intersubject Correlation of Channel-Averaged IVA Sources — {label}"
        ),
        save_path=out_dir / f"{prefix}iva_source_isc_matrix_{label}.png",
    )

    # (c) Pattern ISC matrix per component (channel-averaged patterns in F)
    pattern_corr = np.zeros((n_show, n_subjects, n_subjects))
    for k in range(n_show):
        pattern_corr[k] = np.corrcoef(per_subject_pattern[:, k, :])
    _plot_isc_grid(
        pattern_corr,
        fig_title=(
            f"Intersubject Correlation of Channel-Averaged IVA Patterns — {label}"
        ),
        save_path=out_dir / f"{prefix}iva_pattern_isc_matrix_{label}.png",
    )

    # (d) LOO-ISC bar on channel-averaged per-subject sources.
    _plot_loo_isc_bar(
        per_subject_source,
        n_show=n_show,
        label=label,
        save_path=out_dir / f"{prefix}iva_loo_isc_bar_{label}.png",
    )

    # (e) Mean/var topomap from per-(s,c) channel loadings (freq-collapsed).
    if info is not None:
        chan_loading = iva_components.mean(axis=-1)  # (S, C, N_IVA)
        _plot_topomap_mean_var(
            chan_loading,
            info,
            n_channels,
            n_show=n_show,
            label=label,
            save_path=out_dir / f"{prefix}iva_topomap_mean_var_{label}.png",
        )
    else:
        _logger.warning(f"[{label}] No info available; skipping topomap plot.")

    # (f) Time × frequency outer-product map (averaged over both S and C).
    mean_components_full = iva_components.mean(axis=(0, 1))  # (N_IVA, F)
    freq_profiles = mean_components_full[:n_show].T  # (F, n_show)
    time_profiles = iva_sources.mean(axis=(0, 1))[:n_show]  # (n_show, T)
    _plot_tf_map(
        freq_profiles,
        time_profiles,
        time,
        freqs,
        n_show=n_show,
        label=label,
        save_path=out_dir / f"{prefix}iva_time_frequency_{label}.png",
    )

    # (g) Source timecourse mean ± √variance across (S × C).
    sources_flat_show = iva_sources_flat[:, :n_show, :]  # (S*C, n_show, T)
    _plot_timecourse_mean_var(
        sources_flat_show,
        time,
        n_show=n_show,
        title_suffix="Across (S × C) Datasets",
        label=label,
        save_path=out_dir / f"{prefix}iva_mean_variance_over_time_{label}.png",
    )

    n_plots = 7 if use_pca else 6
    _logger.info(
        f"[{label}] IVA-(S×C): {n_plots} plot files saved to {out_dir}"
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

    condition = ConditionVariants(args.condition)
    music_types = [MusicTypeVariants(mt) for mt in args.music_type]
    exclusion_categories = [
        ExclusionCategories.BAD_MUSIC,
        ExclusionCategories.ARTIFACTS,
    ]
    save_root = args.save_dir if args.save_dir is not None else ProjectPaths.PLOTS_PATH
    wavelet_dir = Path(args.wavelet_data_dir)
    freqs_full = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )

    _logger.info(
        f"IVA-(S×C): condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"use_pca={args.use_pca}, n_pca={args.n_pca}, "
        f"n_show={args.n_show}, iva_opt={args.iva_opt_approach}"
    )

    analyzers = load_analyzers(
        music_types,
        condition,
        exclusion_categories,
        args.process_and_save,
        n_jobs=args.n_jobs,
        normalize_data=False,
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

        save_dir = save_root / _STAGE_DIR / dataset_key
        _run_iva_subject_channel(
            wd.data,
            wd.sfreq,
            freqs_full,
            info,
            label=dataset_key,
            use_pca=args.use_pca,
            n_pca=args.n_pca,
            n_show=args.n_show,
            random_state=args.random_state,
            iva_opt_approach=args.iva_opt_approach,
            iva_max_iter=args.iva_max_iter,
            iva_w_diff_stop=args.iva_w_diff_stop,
            save_dir=save_dir,
        )

    _logger.info("IVA-(S×C) analysis complete.")
