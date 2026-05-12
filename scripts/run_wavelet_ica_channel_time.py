"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_channel_time.ipynb``
(Channel-Time approach: observations = S×F, features = C×T;
components live in channel × time, scores in subject × frequency).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-channel-time-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/channel_time/                       # default (no --band)
            pca_scree_<label>.png
            ...
        bands/channel_time/                           # when --band <name> is given
            <band>_pca_scree_<label>.png
            ...

Pass ``--band <name>`` to slice the cached broadband wavelets down to a
single frequency band (alpha/beta/...) before the ICA step. The same
broadband cache is reused — no separate per-band cache is needed.

Usage::

    # broadband
    python scripts/run_wavelet_ica_channel_time.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 50 --n_ica 10 --reuse_wavelets

    # alpha-band only
    python scripts/run_wavelet_ica_channel_time.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --band alpha --n_pca 50 --n_ica 10 --reuse_wavelets
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402
from mne.viz import plot_topomap  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.decomposition import PCA, FastICA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    FREQUENCY_BANDS,
    _broadband_wavelet_4d,
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

_STAGE_DIR = "04-channel-time-wavelet-ica-analysis"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Channel-Time ICA analysis on preprocessed EEG wavelet "
            "power. Produces the exact same plots as "
            "wavelet_ica_channel_time.ipynb, but for ALL ICA components."
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
            "on the z-scored observation matrix."
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
        help="Maximum Morlet frequency (Hz).",
    )
    parser.add_argument(
        "--wavelet_n_freqs",
        type=int,
        default=20,
        help="Number of Morlet frequency steps.",
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
        "--band",
        choices=[b.value for b in FrequencyBandNames],
        default=None,
        help=(
            "Optional frequency band. When set, the cached broadband wavelet "
            "tensor is sliced to the band's frequency range before ICA, and "
            "plots are written to the 'bands/channel_time/' subdirectory with "
            "a '<band>_' filename prefix. Default: full broadband."
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
    scores_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (a) — Per-IC subject × subject correlation of frequency profiles."""
    n_subjects = scores_2d.shape[0]
    n_show = n_ica
    fig, axes = plt.subplots(
        1, n_show, figsize=(3.5 * n_show, 3.5), constrained_layout=True
    )
    if n_show == 1:
        axes = [axes]

    im = None
    for i, ax in enumerate(axes):
        # Each row of scores_2d[:, :, i] is one subject's frequency profile (length F);
        # np.corrcoef rows-as-variables gives the (S, S) inter-subject correlation.
        corr_mat = np.corrcoef(scores_2d[:, :, i])
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig.suptitle(
        f"Intersubject Correlation of IC Frequency Profiles — {label}",
        fontsize=12,
    )
    if im is not None:
        plt.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_loadings(
    scores_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (b) — Per-subject mean |score| over frequencies per component."""
    n_subjects = scores_2d.shape[0]
    subject_loadings = np.abs(scores_2d).mean(axis=1)  # (S, K)

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
    scores_2d: np.ndarray,
    components_2d: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (c) — Frequency × Time outer-product map per component."""
    # Frequency profile: collapse subjects from ICA scores  (S, F, K) → (F, K)
    freq_profiles = scores_2d.mean(axis=0)
    # Time profile: average channels from ICA components  (K, C, T) → (K, T)
    time_profiles = components_2d.mean(axis=1)
    ft_maps = np.einsum("fk,kt->kft", freq_profiles, time_profiles)  # (K, F, T)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
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
        ax.set_title(f"IC {i + 1} — Freq × Time Map (outer product)", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Frequency × Time Maps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomap_mean_variance(
    scores_2d: np.ndarray,
    components_2d: np.ndarray,
    info,
    n_channels: int,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (d) — Two-row topomap: mean (top) and across-subject variance (bottom)."""
    # Time-averaged channel profile from components → (C, K)
    chan_profile = components_2d.mean(axis=2).T  # (C, K)
    # Per-subject score per IC (mean over frequencies) → (S, K)
    subj_score = scores_2d.mean(axis=1)
    # Per-subject topomap via outer product → (K, S, C)
    topo_per_subj = np.einsum("sk,ck->ksc", subj_score, chan_profile)

    ica_ch_mean = topo_per_subj.mean(axis=1).T  # (C, K)
    ica_ch_var = topo_per_subj.var(axis=1).T  # (C, K)

    info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(info.ch_names):
        info = mne.pick_info(info, list(range(n_channels)))

    n_show = n_ica

    fig, axes = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7.5))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        # Per-component symmetric color scale around zero for the mean row
        vlim_mean_i = float(np.percentile(np.abs(ica_ch_mean[:, i]), 99))
        # Per-component sequential color scale (variance is non-negative)
        vmax_var_i = float(np.percentile(ica_ch_var[:, i], 99))

        im_mean, _ = plot_topomap(
            ica_ch_mean[:, i],
            info,
            axes=axes[0, i],
            show=False,
            cmap="RdBu_r",
            vlim=(-vlim_mean_i, vlim_mean_i),
        )
        axes[0, i].set_title(f"IC {i + 1}", fontsize=10)
        fig.colorbar(im_mean, ax=axes[0, i], fraction=0.046, pad=0.04)

        im_var, _ = plot_topomap(
            ica_ch_var[:, i],
            info,
            axes=axes[1, i],
            show=False,
            cmap="viridis",
            vlim=(0, vmax_var_i),
        )
        fig.colorbar(im_var, ax=axes[1, i], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Mean", fontsize=11)
    axes[1, 0].set_ylabel("Variance", fontsize=11)

    fig.suptitle(
        f"Component Channel Loading (topomap) — Mean & Across-Subject Variance — {label}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_frequency_heatmap(
    scores_2d: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (e) — Subject × Frequency loading heatmap per component."""
    n_subjects = scores_2d.shape[0]
    # scores_2d already has shape (S, F, K) — no axis to collapse
    sf_loadings = scores_2d

    n_show = n_ica
    vlim = float(np.percentile(np.abs(sf_loadings[:, :, :n_show]), 99))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            freqs,
            np.arange(n_subjects),
            sf_loadings[:, :, i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(f"IC {i + 1} — Subject × Frequency Loading", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="loading")

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"Subject × Frequency Loading Heatmaps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_channel_time_heatmap(
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (f) — Channel × Time activation heatmap per component.

    components_2d[k] : (C, T) — plotted directly without averaging.
    """
    n_channels = components_2d.shape[1]
    n_show = n_ica
    vlim = float(np.percentile(np.abs(components_2d[:n_show]), 99))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time,
            np.arange(n_channels),
            components_2d[i],  # (C, T)
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_ylabel("Channel")
        ax.set_title(f"IC {i + 1} — Channel × Time Activation", fontsize=10)
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="activation")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Channel × Time Activation Heatmaps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_component_timecourses(
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (g) — Channel-averaged temporal pattern of each ICA component."""
    time_profiles = components_2d.mean(axis=1)  # (K, T)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.2 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, time_profiles[i], lw=0.8, color="darkorange")
        ax.axhline(0.0, color="gray", lw=0.5, ls="--")
        ax.set_ylabel(f"IC {i + 1}")
        ax.set_title(
            f"Component {i + 1} — Channel-Averaged Temporal Pattern", fontsize=10
        )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"ICA Component Time Courses (channel-averaged) — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_time_heatmap(
    scores_2d: np.ndarray,
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (g) — Subject × Time activation heatmap per component (outer product)."""
    n_subjects = scores_2d.shape[0]

    # Subject profile: collapse frequencies  (S, F, K) → (S, K)
    subj_profile = scores_2d.mean(axis=1)
    # Time profile: channel-averaged components  (K, C, T) → (K, T)
    time_profile = components_2d.mean(axis=1)
    # Outer product per component: (S, K) x (K, T) → (K, S, T)
    st_maps = np.einsum("sk,kt->kst", subj_profile, time_profile)

    n_show = n_ica
    vlim = float(np.percentile(np.abs(st_maps[:n_show]), 99))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time,
            np.arange(n_subjects),
            st_maps[i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(
            f"IC {i + 1} — Subject × Time Activation (outer product)", fontsize=10
        )
        fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025, label="activation")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Subject × Time Activation Heatmaps per IC — {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_isc_bar(
    scores_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (h) — Bar plot of mean LOO-ISC across participants per IC.

    Per-subject vector is the subject's frequency-loading profile
    ``scores_2d[s, :, k]`` (length F). Bars whose across-subject mean
    LOO-ISC is negative are coloured red; positive bars are steel blue.
    """
    n_subjects = scores_2d.shape[0]
    loo_isc_per_subject = np.zeros((n_ica, n_subjects))
    for k in range(n_ica):
        vecs = scores_2d[:, :, k]  # (S, F)
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


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def _run_channel_time(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    save_dir: Path,
    band: str | None = None,
    skip_pca: bool = False,
) -> None:
    """Run the full Channel-Time ICA pipeline and save all plots.

    When ``band`` is given the outputs land in ``bands/channel_time/`` with
    filenames prefixed by ``<band>_``. Otherwise the broadband layout is used.
    """
    if band is None:
        out_dir = save_dir / "broadband" / "channel_time"
        prefix = ""
    else:
        out_dir = save_dir / "bands" / "channel_time"
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] Channel-Time: {data_4d.shape}  sfreq={sfreq} Hz")

    # Step 1 — Z-score and reshape: (S, C, F, T) → (S, F, C, T) → (S*F, C*T)
    bb_z = zscore_by_time(data_4d)
    bb_z_sf = bb_z.transpose(0, 2, 1, 3)  # (S, F, C, T)
    n_obs = n_subjects * n_freqs
    n_feat = n_channels * n_times
    X = bb_z_sf.reshape(n_obs, n_feat)
    _logger.info(f"[{label}] Reshaped: {X.shape}  (S*F, C*T)")

    # Step 2 — (optional) PCA + ICA
    ica = FastICA(
        n_components=n_ica,
        random_state=random_state,
        max_iter=500,
        whiten="unit-variance",
    )

    if skip_pca:
        _logger.info(f"[{label}] Skipping PCA; running FastICA directly on X.")
        ica_scores = ica.fit_transform(X)  # (S*F, K)
        ica_components = ica.components_  # (K, C*T)
    else:
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca_scores = pca.fit_transform(X)
        _logger.info(
            f"[{label}] PCA: {pca_scores.shape}, "
            f"explained={np.cumsum(pca.explained_variance_ratio_)[-1] * 100:.1f}%"
        )
        ica_scores = ica.fit_transform(pca_scores)  # (S*F, K)
        ica_components = ica.components_ @ pca.components_  # (K, C*T)

    # Reshape ICA scores to (S, F, K) and components to (K, C, T)
    scores_2d = ica_scores.reshape(n_subjects, n_freqs, n_ica)
    components_2d = ica_components.reshape(n_ica, n_channels, n_times)
    _logger.info(
        f"[{label}] ICA: scores_2d={scores_2d.shape}, "
        f"components_2d={components_2d.shape}"
    )

    # Plot 1 — PCA scree (skipped when PCA is not run)
    if not skip_pca:
        _plot_pca_scree(
            pca.explained_variance_ratio_,
            label=label,
            save_path=out_dir / f"{prefix}pca_scree_{label}.png",
        )

    # Plot 2 — (a) ISC matrix from per-subject frequency profiles
    _plot_isc_matrix(
        scores_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}isc_component_matrix_{label}.png",
    )

    # Plot 3 — (b) Subject loadings bar plot
    _plot_subject_loadings(
        scores_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_loadings_{label}.png",
    )

    # Plot 4 — (c) Freq × Time outer-product map
    _plot_time_frequency(
        scores_2d,
        components_2d,
        time,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_time_frequency_{label}.png",
    )

    # Plot 5 — (d) Mean + variance topomap
    _plot_topomap_mean_variance(
        scores_2d,
        components_2d,
        info,
        n_channels,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_topomap_mean_variance_{label}.png",
    )

    # Plot 6 — (e) Subject × Frequency loading heatmap
    _plot_subject_frequency_heatmap(
        scores_2d,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_frequency_heatmap_{label}.png",
    )

    # Plot 7 — (f) Channel × Time activation heatmap
    _plot_channel_time_heatmap(
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_channel_time_heatmap_{label}.png",
    )

    # Plot 8 — (g) Component time courses (channel-averaged)
    _plot_component_timecourses(
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_component_timecourses_{label}.png",
    )

    # Plot 9 — (h) Subject × Time activation heatmap
    _plot_subject_time_heatmap(
        scores_2d,
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_time_heatmap_{label}.png",
    )

    # Plot 10 — (i) Mean LOO-ISC across participants per IC (bar plot)
    _plot_loo_isc_bar(
        scores_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_loo_isc_bar_{label}.png",
    )

    n_plots = 10 if not skip_pca else 9
    _logger.info(f"[{label}] Channel-Time: {n_plots} plots saved to {out_dir}")


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
    freqs = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )
    band_name = args.band  # None or e.g. "alpha"

    _logger.info(
        f"Channel-Time ICA: condition={condition.value}, "
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

        # Compute / load 4D wavelet power (broadband cache; band slice is in-memory)
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
        _run_channel_time(
            data_4d,
            wd.sfreq,
            ica_freqs,
            info,
            label=dataset_key,
            n_pca=args.n_pca,
            n_ica=args.n_ica,
            random_state=args.random_state,
            save_dir=save_dir,
            band=band_name,
            skip_pca=args.skip_pca,
        )

    _logger.info("Channel-Time ICA analysis complete.")
