"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_subject_time.ipynb``
(Subject-Time approach: observations = F×C, features = S×T;
components live in subject × time).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-subject-time-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/subject_time/                       # default (no --band)
            pca_scree_<label>.png
            ...
        bands/subject_time/                           # when --band <name> is given
            <band>_pca_scree_<label>.png
            ...

Pass ``--band <name>`` to slice the cached broadband wavelets down to a
single frequency band (alpha/beta/...) before the ICA step. The same
broadband cache is reused — no separate per-band cache is needed.

Usage::

    # broadband
    python scripts/run_wavelet_ica_subject_time.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 50 --n_ica 10 --reuse_wavelets \\
        --sliding_variants 1.0:0.5

    # alpha-band only
    python scripts/run_wavelet_ica_subject_time.py \\
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

_STAGE_DIR = "04-subject-time-wavelet-ica-analysis"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Subject-Time ICA analysis on preprocessed EEG wavelet "
            "power.  Produces the exact same plots as "
            "wavelet_ica_subject_time.ipynb, but for ALL ICA components."
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
            "on the (F*C, S*T) z-scored matrix."
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
            "plots are written to the 'bands/subject_time/' subdirectory with "
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
    components_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (a) — Intersubject correlation matrix for ALL ICA components.

    Each row of components_2d[k] is one subject's temporal profile (length T);
    np.corrcoef gives the (S, S) inter-subject correlation per component.
    """
    n_subjects = components_2d.shape[1]
    n_show = n_ica

    fig, axes = plt.subplots(
        1,
        n_show,
        figsize=(3.5 * n_show, 3.5),
        constrained_layout=True,
    )
    if n_show == 1:
        axes = [axes]

    im = None
    for i, ax in enumerate(axes):
        corr_mat = np.corrcoef(components_2d[i])  # (S, S)
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig.suptitle(
        f"Intersubject Correlation of IC Temporal Profiles — {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_subject_loadings(
    components_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (b) — Per-subject mean loading bar plot for ALL ICs."""
    n_subjects = components_2d.shape[1]
    subject_loadings = np.abs(components_2d).mean(axis=2).T  # (S, K)

    n_show = n_ica
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.barh(range(n_subjects), subject_loadings[:, i], color="darkorange")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_xlabel("|activation|")
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
    """Analysis (c) — Time × Frequency map per component (outer product) for ALL ICs.

    freq_profile[k] = mean_c scores_2d[:, :, k]              (F,)
    time_profile[k] = mean_s components_2d[k, :, :]          (T,)
    tf_map[k]       = outer(freq_profile[k], time_profile[k])  (F, T)
    """
    # Frequency profile: collapse channels from ICA scores  (F, C, K) → (F, K)
    freq_profiles = scores_2d.mean(axis=1)  # (F, K)
    # Time profile: average subject activations from ICA components  (K, S, T) → (K, T)
    time_profiles = components_2d.mean(axis=1)  # (K, T)
    # Outer product: (F, K) x (K, T) → (K, F, T)
    ft_maps = np.einsum("fk,kt->kft", freq_profiles, time_profiles)  # (K, F, T)

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
            f"IC {i + 1} — Freq × Time Map (outer product)",
            fontsize=10,
        )
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


def _plot_topomap_mean(
    scores_2d: np.ndarray,
    info,
    n_channels: int,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (d) — Mean component channel loading (topomap) for ALL ICs.

    chan_loading[c, k] = mean_f  scores_2d[f, c, k]                  (C, K)

    Each component is plotted with its own symmetric color scale.
    """
    chan_loading = scores_2d.mean(axis=0)  # (C, K)

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
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (e) — Mean & variance of IC signal over time across subjects."""
    mean_temporal = components_2d.mean(axis=1)  # (K, T)
    var_temporal = components_2d.var(axis=1)  # (K, T)
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
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (g) — Subject × Time activation heatmap per component."""
    n_subjects = components_2d.shape[1]
    n_show = n_ica

    vlim = float(np.percentile(np.abs(components_2d[:n_show]), 99))

    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    mesh = None
    for i, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            time,
            np.arange(n_subjects),
            components_2d[i],
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            shading="auto",
        )
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_ylabel("Subject")
        ax.set_title(f"IC {i + 1} — Subject × Time Activation", fontsize=10)
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


def _plot_freq_channel_heatmap(
    scores_2d: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (h) — Frequency × Channel heatmap of ICA scores per component.

    Companion to the topomap in (d): the ICA scores `(F, C, K)` are shown
    without collapsing the frequency axis, with each component on its own
    symmetric color scale.
    """
    n_freqs, n_channels, _ = scores_2d.shape
    n_show = n_ica
    channels = np.arange(n_channels)

    fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4.5), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = scores_2d[:, :, i]  # (F, C)
        vlim_i = float(np.percentile(np.abs(data_i), 99))
        mesh = ax.pcolormesh(
            channels,
            freqs,
            data_i,
            cmap="RdBu_r",
            vmin=-vlim_i,
            vmax=vlim_i,
            shading="auto",
        )
        ax.set_xlabel("Channel")
        ax.set_title(f"IC {i + 1}", fontsize=10)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="score")

    axes[0].set_ylabel("Frequency (Hz)")
    fig.suptitle(
        f"Frequency × Channel ICA Score Heatmaps — {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_loo_isc_bar(
    components_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (i) — Bar plot of mean LOO-ISC across participants per IC.

    Per-subject vector is each subject's temporal profile
    ``components_2d[k, s, :]`` (length T). Bars whose across-subject mean
    LOO-ISC is negative are coloured red; positive bars are steel blue.
    """
    n_subjects = components_2d.shape[1]
    loo_isc_per_subject = np.zeros((n_ica, n_subjects))
    for k in range(n_ica):
        vecs = components_2d[k]  # (S, T)
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
    components_2d: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float,
) -> dict:
    """Per-subject and across-subject LOO-ISC for one (window, step) pair.

    components_2d: (K, S, T)
    Returns dict with per_subject (K, W, S), mean (K, W), std (K, W), edges (W,).
    """
    n_ica, n_subjects, n_times = components_2d.shape
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
    edges = starts / sfreq  # window-start times → first stair begins at t=0

    per_subject = np.zeros((n_ica, len(starts), n_subjects))
    for k in range(n_ica):
        comp = components_2d[k]  # (S, T)
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
    components_2d: np.ndarray,
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

    n_times = components_2d.shape[2]
    t_end = n_times / sfreq

    v = _compute_sliding_loo_isc(components_2d, sfreq, win, step)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.8 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        edges_m, mean_m = _close_to_end(v["edges"], v["mean"][i], t_end)
        _, std_m = _close_to_end(v["edges"], v["std"][i], t_end)
        ax.plot(
            edges_m,
            mean_m,
            lw=1.2,
            color="steelblue",
            drawstyle="steps-post",
            label="mean" if i == 0 else None,
        )
        ax.fill_between(
            edges_m,
            mean_m - std_m,
            mean_m + std_m,
            alpha=0.2,
            color="steelblue",
            step="post",
            label="± √variance" if i == 0 else None,
        )
        ax.axhline(0.0, ls="--", lw=0.6, color="gray")
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


def _run_subject_time(
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
    """Run the full subject-time ICA pipeline and save all plots.

    When ``band`` is given the outputs land in ``bands/subject_time/`` with
    filenames prefixed by ``<band>_``. Otherwise the broadband layout is used.
    """
    if band is None:
        out_dir = save_dir / "broadband" / "subject_time"
        prefix = ""
    else:
        out_dir = save_dir / "bands" / "subject_time"
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] Subject-Time: {data_4d.shape}  sfreq={sfreq} Hz")

    # Step 1 — Z-score and reshape: (S,C,F,T) → (F,C,S,T) → (F*C, S*T)
    bb_z = zscore_by_time(data_4d)
    bb_z_fc = bb_z.transpose(2, 1, 0, 3)  # (F, C, S, T)
    X_fc = bb_z_fc.reshape(n_freqs * n_channels, n_subjects * n_times)
    _logger.info(f"[{label}] Reshaped: {X_fc.shape}  (F*C, S*T)")

    # Step 2 — (optional) PCA + ICA
    ica = FastICA(
        n_components=n_ica,
        random_state=random_state,
        max_iter=500,
        whiten="unit-variance",
    )

    if skip_pca:
        _logger.info(f"[{label}] Skipping PCA; running FastICA directly on X_fc.")
        ica_scores = ica.fit_transform(X_fc)  # (F*C, K)
        ica_components = ica.components_  # (K, S*T)
    else:
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca_scores = pca.fit_transform(X_fc)
        _logger.info(
            f"[{label}] PCA: {pca_scores.shape}, "
            f"explained={np.cumsum(pca.explained_variance_ratio_)[-1] * 100:.1f}%"
        )
        ica_scores = ica.fit_transform(pca_scores)
        ica_components = ica.components_ @ pca.components_  # (K, S*T)

    # Reshape
    scores_2d = ica_scores.reshape(n_freqs, n_channels, n_ica)  # (F, C, K)
    components_2d = ica_components.reshape(n_ica, n_subjects, n_times)  # (K, S, T)
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

    # Plot 2 — (a) ISC matrix
    _plot_isc_matrix(
        components_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}isc_component_matrix_{label}.png",
    )

    # Plot 3 — (b) Subject loadings bar plot
    _plot_subject_loadings(
        components_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_loadings_{label}.png",
    )

    # Plot 4 — (c) Freq × Time map (outer product)
    _plot_time_frequency(
        scores_2d,
        components_2d,
        time,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_time_frequency_{label}.png",
    )

    # Plot 5 — (d) Mean component channel loading topomap
    _plot_topomap_mean(
        scores_2d,
        info,
        n_channels,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_topomap_mean_{label}.png",
    )

    # Plot 6 — (e) Mean & variance of IC signal over time across subjects
    _plot_mean_variance_over_time(
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_mean_variance_over_time_{label}.png",
    )

    # Plot 7 — (f) Sliding-window LOO-ISC per IC, smallest-step variant
    _plot_sliding_window_loo_isc(
        components_2d,
        sfreq,
        n_ica,
        sliding_variants,
        label=label,
        save_path=out_dir / f"{prefix}ica_sliding_window_loo_isc_{label}.png",
    )

    # Plot 8 — (g) Subject × Time activation heatmap per IC
    _plot_subject_time_heatmap(
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_subject_time_heatmap_{label}.png",
    )

    # Plot 9 — (h) Frequency × Channel heatmap of ICA scores per IC
    _plot_freq_channel_heatmap(
        scores_2d,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_freq_channel_heatmap_{label}.png",
    )

    # Plot 10 — (i) Mean LOO-ISC across participants per IC (bar plot)
    _plot_loo_isc_bar(
        components_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_loo_isc_bar_{label}.png",
    )

    n_plots = 10 if not skip_pca else 9
    _logger.info(f"[{label}] Subject-Time: {n_plots} plots saved to {out_dir}")


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
    sliding_variants = _parse_sliding_variants(args.sliding_variants)
    band_name = args.band  # None or e.g. "alpha"

    _logger.info(
        f"Subject-Time ICA: condition={condition.value}, "
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
        _run_subject_time(
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

    _logger.info("Subject-Time ICA analysis complete.")
