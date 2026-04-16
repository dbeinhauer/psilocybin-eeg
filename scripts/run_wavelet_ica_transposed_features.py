"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_transposed_features.ipynb``
(Approach 7 — Transposed Features: observations = T, features = S×C×F).

This is the **exact inversion** of Approach 6 (combined features).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-transposed-features-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/transposed_features/
            pca_scree_<label>.png
            isc_component_matrix_<label>.png
            ica_temporal_activations_<label>.png
            ica_time_frequency_<label>.png
            ica_topomap_mean_<label>.png
            ica_topomap_variance_<label>.png
            ica_subject_mixing_<label>.png

Usage::

    python scripts/run_wavelet_ica_transposed_features.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 20 --n_ica 10 --reuse_wavelets
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
from scipy.signal import spectrogram as sp_spectrogram  # noqa: E402
from sklearn.decomposition import PCA, FastICA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
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
    MusicTypeVariants,
)

_logger = logging.getLogger(__name__)

_STAGE_DIR = "04-transposed-features-wavelet-ica-analysis"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Transposed-Features ICA analysis (Approach 7) on "
            "preprocessed EEG wavelet power.  Produces the exact same "
            "plots as wavelet_ica_transposed_features.ipynb, but for ALL "
            "ICA components."
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
        default=20,
        help="Number of PCA components to retain.",
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
    """PCA scree and cumulative variance plot (notebook cell 13, first half)."""
    cumulative = np.cumsum(explained)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, len(explained) + 1), explained, color="steelblue")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance explained")
    axes[0].set_title(f"PCA Scree Plot \u2014 {label}")

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, "o-", color="coral")
    axes[1].axhline(0.9, ls="--", color="gray", label="90%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative Variance \u2014 {label}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_isc_matrix(
    mixing_4d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Intersubject correlation matrix for ALL ICA components (cell 15).

    Uses per-subject channel×frequency mixing vectors: mixing_4d shape (S, C, F, K).
    """
    n_subjects, n_channels, n_freqs, _ = mixing_4d.shape
    subject_cf_mixing = mixing_4d.reshape(
        n_subjects, n_channels * n_freqs, n_ica
    )

    n_show = n_ica  # ALL components
    fig, axes = plt.subplots(
        1,
        n_show,
        figsize=(3.5 * n_show, 3.5),
        constrained_layout=True,
    )
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        corr_mat = np.corrcoef(subject_cf_mixing[:, :, i])
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig.suptitle(
        f"Intersubject Correlation of IC Mixing Vectors \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_temporal_activations(
    mixing_4d: np.ndarray,
    bb_z: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Per-subject temporal activations for ALL ICs (cell 17)."""
    n_subjects, n_channels, n_freqs, _ = mixing_4d.shape
    subject_temporal = np.einsum("scfk,scft->skt", mixing_4d, bb_z) / (
        n_channels * n_freqs
    )
    mean_temporal = subject_temporal.mean(axis=0)
    std_temporal = subject_temporal.std(axis=0)

    n_show = n_ica  # ALL components
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, mean_temporal[i], lw=0.8, color="teal", label="mean")
        ax.fill_between(
            time,
            mean_temporal[i] - std_temporal[i],
            mean_temporal[i] + std_temporal[i],
            alpha=0.25,
            color="teal",
            label="\u00b1 1 std",
        )
        ax.set_ylabel(f"IC {i + 1}")
        ax.set_title(f"Component {i + 1} \u2014 Temporal Activation", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"ICA Mode Time Courses (mean \u00b1 std across subjects) "
        f"\u2014 {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_time_frequency(
    ica_sources: np.ndarray,
    sfreq: float,
    freqs_max: float,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Time-frequency spectrograms of ICA source time courses for ALL ICs (cell 19)."""
    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        f_spec, t_spec, Sxx = sp_spectrogram(
            ica_sources[:, i],
            fs=sfreq,
            nperseg=int(sfreq * 2),
            noverlap=int(sfreq),
        )
        freq_mask = f_spec <= freqs_max
        Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-12)
        vmin_s, vmax_s = np.percentile(Sxx_db, 1), np.percentile(Sxx_db, 99)
        ax.pcolormesh(
            t_spec,
            f_spec[freq_mask],
            Sxx_db,
            cmap="inferno",
            vmin=vmin_s,
            vmax=vmax_s,
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"IC {i + 1} \u2014 Source Spectrogram", fontsize=10)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Time\u2013Frequency of ICA Source Time Courses \u2014 {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomaps(
    mixing_4d: np.ndarray,
    info,
    n_channels: int,
    n_ica: int,
    *,
    label: str,
    save_path_mean: Path,
    save_path_var: Path,
) -> None:
    """Mean and variance topomaps for ALL ICs (cell 21)."""
    ica_channel_mixing = mixing_4d.mean(axis=2)  # (S, C, K)
    ica_ch_mean = ica_channel_mixing.mean(axis=0)  # (C, K)
    ica_ch_var = ica_channel_mixing.var(axis=0)  # (C, K)

    topo_info = mne.pick_info(info, mne.pick_types(info, eeg=True))
    if n_channels < len(topo_info.ch_names):
        topo_info = mne.pick_info(topo_info, list(range(n_channels)))

    n_show = n_ica

    # --- Mean topomaps ---
    _vlim_mean = np.percentile(np.abs(ica_ch_mean[:, :n_show]), 99)
    fig_mean, axes_mean = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
    if n_show == 1:
        axes_mean = [axes_mean]

    for i, ax in enumerate(axes_mean):
        im, _ = plot_topomap(
            ica_ch_mean[:, i],
            topo_info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-_vlim_mean, _vlim_mean),
        )
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig_mean.suptitle(
        f"Mean Component Mixing (topomap) \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes_mean[-1], label="mean mixing weight")
    fig_mean.tight_layout()
    fig_mean.savefig(save_path_mean, dpi=150, bbox_inches="tight")
    plt.close(fig_mean)

    # --- Variance topomaps ---
    _vmax_var = np.percentile(ica_ch_var[:, :n_show], 99)
    fig_var, axes_var = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
    if n_show == 1:
        axes_var = [axes_var]

    for i, ax in enumerate(axes_var):
        im, _ = plot_topomap(
            ica_ch_var[:, i],
            topo_info,
            axes=ax,
            show=False,
            cmap="YlOrRd",
            vlim=(0, _vmax_var),
        )
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig_var.suptitle(
        f"Variance of Component Mixing (topomap) \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes_var[-1], label="variance")
    fig_var.tight_layout()
    fig_var.savefig(save_path_var, dpi=150, bbox_inches="tight")
    plt.close(fig_var)


def _plot_subject_mixing(
    mixing_4d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Per-subject mixing-weight bar plot for ALL ICs (cell 23)."""
    n_subjects = mixing_4d.shape[0]
    subject_mixing = np.abs(mixing_4d).mean(axis=(1, 2))  # (S, K)

    n_show = n_ica
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.barh(range(n_subjects), subject_mixing[:, i], color="teal")
        ax.set_yticks(range(n_subjects))
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=8)
        ax.set_xlabel("|mixing|")
        ax.set_title(f"IC {i + 1}", fontsize=10)

    axes[0].set_ylabel("Subject")
    fig.suptitle(
        f"Per-Subject Mixing Weight per Component \u2014 {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def _run_transposed_features(
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
) -> None:
    """Run the full transposed-features pipeline and save all plots."""
    out_dir = save_dir / "broadband" / "transposed_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] Transposed Features: {data_4d.shape}  sfreq={sfreq} Hz")

    # Step 1 — Z-score and reshape → transpose: (T, S×C×F)
    bb_z = zscore_by_time(data_4d)
    X_t = bb_z.reshape(-1, n_times).T  # (T, S*C*F)
    _logger.info(f"[{label}] Transposed: {X_t.shape}  (T, S*C*F)")

    # Step 2 — PCA + ICA
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_scores = pca.fit_transform(X_t)
    _logger.info(
        f"[{label}] PCA: {pca_scores.shape}, "
        f"explained={np.cumsum(pca.explained_variance_ratio_)[-1] * 100:.1f}%"
    )

    ica = FastICA(
        n_components=n_ica,
        random_state=random_state,
        max_iter=500,
        whiten="unit-variance",
    )
    ica_sources = ica.fit_transform(pca_scores)  # (T, K_ica)

    # Full feature-space mixing matrix: (S*C*F, K_ica)
    ica_full_mixing = pca.components_.T @ ica.mixing_  # (S*C*F, K_ica)
    mixing_4d = ica_full_mixing.reshape(
        n_subjects, n_channels, n_freqs, n_ica
    )  # (S, C, F, K)
    _logger.info(
        f"[{label}] ICA: sources={ica_sources.shape}, "
        f"mixing_4d={mixing_4d.shape}"
    )

    # Plot 1 — PCA scree
    _plot_pca_scree(
        pca.explained_variance_ratio_,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )

    # Plot 2 — ISC matrix
    _plot_isc_matrix(
        mixing_4d,
        n_ica,
        label=label,
        save_path=out_dir / f"isc_component_matrix_{label}.png",
    )

    # Plot 3 — Temporal activations
    _plot_temporal_activations(
        mixing_4d,
        bb_z,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"ica_temporal_activations_{label}.png",
    )

    # Plot 4 — Time-frequency spectrograms
    _plot_time_frequency(
        ica_sources,
        sfreq,
        float(freqs[-1]),
        n_ica,
        label=label,
        save_path=out_dir / f"ica_time_frequency_{label}.png",
    )

    # Plot 5+6 — Mean and variance topomaps
    _plot_topomaps(
        mixing_4d,
        info,
        n_channels,
        n_ica,
        label=label,
        save_path_mean=out_dir / f"ica_topomap_mean_{label}.png",
        save_path_var=out_dir / f"ica_topomap_variance_{label}.png",
    )

    # Plot 7 — Per-subject mixing bars
    _plot_subject_mixing(
        mixing_4d,
        n_ica,
        label=label,
        save_path=out_dir / f"ica_subject_mixing_{label}.png",
    )

    _logger.info(f"[{label}] Transposed Features: 7 plots saved to {out_dir}")


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

    _logger.info(
        f"Transposed Features ICA: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"n_pca={args.n_pca}, n_ica={args.n_ica}"
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

        # Compute / load 4D wavelet power
        wd = _broadband_wavelet_4d(
            ad,
            dataset_key,
            representation="power",
            freqs=freqs,
            wavelet_dir=(wavelet_dir / "broadband"),
            reuse_wavelets=args.reuse_wavelets,
        )

        save_dir = save_root / _STAGE_DIR / dataset_key
        _run_transposed_features(
            wd.data,
            wd.sfreq,
            freqs,
            info,
            label=dataset_key,
            n_pca=args.n_pca,
            n_ica=args.n_ica,
            random_state=args.random_state,
            save_dir=save_dir,
        )

    _logger.info("Transposed Features ICA analysis complete.")
