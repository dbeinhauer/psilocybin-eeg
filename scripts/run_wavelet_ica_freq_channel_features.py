"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_freq_channel_features.ipynb``
(Approach 10 — Freq-Channel Features: observations = F×C, features = S×T).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-freq-channel-features-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/freq_channel_features/
            pca_scree_<label>.png
            isc_component_matrix_<label>.png
            ica_temporal_profiles_<label>.png
            ica_time_frequency_<label>.png
            ica_topomap_mean_<label>.png
            ica_topomap_variance_<label>.png
            ica_subject_loadings_<label>.png

Usage::

    python scripts/run_wavelet_ica_freq_channel_features.py \\
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

_STAGE_DIR = "04-freq-channel-features-wavelet-ica-analysis"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Freq-Channel Features ICA analysis (Approach 10) on "
            "preprocessed EEG wavelet power.  Produces the exact same plots "
            "as wavelet_ica_freq_channel_features.ipynb, but for ALL ICA "
            "components."
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
    components_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Intersubject correlation matrix for ALL ICA components (cell 15).

    Uses per-subject temporal profiles: components_2d shape (K, S, T).
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

    for i, ax in enumerate(axes):
        corr_mat = np.corrcoef(components_2d[i])
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig.suptitle(
        f"Intersubject Correlation of IC Temporal Profiles \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_temporal_profiles(
    components_2d: np.ndarray,
    time: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Component temporal profiles (mean ± std across subjects) for ALL ICs (cell 17)."""
    mean_temporal = components_2d.mean(axis=1)  # (K, T)
    std_temporal = components_2d.std(axis=1)  # (K, T)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 2.5 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(time, mean_temporal[i], lw=0.8, color="darkorange", label="mean")
        ax.fill_between(
            time,
            mean_temporal[i] - std_temporal[i],
            mean_temporal[i] + std_temporal[i],
            alpha=0.25,
            color="darkorange",
            label="\u00b1 1 std",
        )
        ax.set_ylabel(f"IC {i + 1}")
        ax.set_title(f"Component {i + 1} \u2014 Temporal Profile", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"ICA Component Temporal Profiles (mean \u00b1 std across subjects) "
        f"\u2014 {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_time_frequency(
    scores_2d: np.ndarray,
    components_2d: np.ndarray,
    bb_z: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Freq × Time mean-loading heatmap for ALL ICs (cell 19)."""
    n_subjects = bb_z.shape[0]
    # Frequency weights from scores: mean over channels
    freq_weights = scores_2d.mean(axis=1)  # (F, K)
    # Subject weights from components: mean over time
    sub_weights = components_2d.mean(axis=2)  # (K, S)
    # Weighted data: mean over channels, then weight by subject weights
    bb_z_chan_avg = bb_z.mean(axis=1)  # (S, F, T)
    # weighted_sub(k, f, t) = mean_s[ sub_weights(k,s) * bb_z_chan_avg(s,f,t) ]
    weighted_sub = np.einsum("ks,sft->kft", sub_weights, bb_z_chan_avg) / n_subjects
    # Final: loading(k, f, t) = freq_weights(f, k) * weighted_sub(k, f, t)
    ft_loading = np.einsum("fk,kft->kft", freq_weights, weighted_sub)

    n_show = n_ica
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = ft_loading[i]
        vmin_s, vmax_s = np.percentile(data_i, 1), np.percentile(data_i, 99)
        ax.pcolormesh(
            time,
            freqs,
            data_i,
            cmap="inferno",
            vmin=vmin_s,
            vmax=vmax_s,
        )
        ax.set_ylabel("Freq (Hz)")
        ax.set_title(f"IC {i + 1} \u2014 Freq \u00d7 Time Mean Loading", fontsize=10)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Frequency \u00d7 Time Mean Loading per IC \u2014 {label}",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_topomaps(
    scores_2d: np.ndarray,
    components_2d: np.ndarray,
    info,
    n_channels: int,
    n_ica: int,
    *,
    label: str,
    save_path_mean: Path,
    save_path_var: Path,
) -> None:
    """Mean and variance topomaps for ALL ICs (cell 21)."""
    score_channel_loadings = scores_2d.mean(axis=0)  # (C, K)
    subject_mean_activation = components_2d.mean(axis=2).T  # (S, K)
    ica_channel_loadings = np.einsum(
        "sk,ck->sck",
        subject_mean_activation,
        score_channel_loadings,
    )  # (S, C, K)

    ica_ch_mean = ica_channel_loadings.mean(axis=0)  # (C, K)
    ica_ch_var = ica_channel_loadings.var(axis=0)  # (C, K)

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
        f"Mean Component Channel Loading (topomap) \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes_mean[-1], label="mean loading")
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
        f"Variance of Component Channel Loading (topomap) \u2014 {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes_var[-1], label="variance")
    fig_var.tight_layout()
    fig_var.savefig(save_path_var, dpi=150, bbox_inches="tight")
    plt.close(fig_var)


def _plot_subject_loadings(
    components_2d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Per-subject loading bar plot for ALL ICs (cell 23)."""
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
        f"Per-Subject Loading per Component \u2014 {label}",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def _run_freq_channel_features(
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
    """Run the full freq-channel-features pipeline and save all plots."""
    out_dir = save_dir / "broadband" / "freq_channel_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(f"[{label}] Freq-Channel Features: {data_4d.shape}  sfreq={sfreq} Hz")

    # Step 1 — Z-score and reshape: (S,C,F,T) → (F,C,S,T) → (F*C, S*T)
    bb_z = zscore_by_time(data_4d)
    bb_z_fc = bb_z.transpose(2, 1, 0, 3)  # (F, C, S, T)
    X_fc = bb_z_fc.reshape(n_freqs * n_channels, n_subjects * n_times)
    _logger.info(f"[{label}] Reshaped: {X_fc.shape}  (F*C, S*T)")

    # Step 2 — PCA + ICA
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_scores = pca.fit_transform(X_fc)
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
    ica_scores = ica.fit_transform(pca_scores)
    ica_components = ica.components_ @ pca.components_  # (K, S*T)

    # Reshape
    scores_2d = ica_scores.reshape(n_freqs, n_channels, n_ica)  # (F, C, K)
    components_2d = ica_components.reshape(n_ica, n_subjects, n_times)  # (K, S, T)
    _logger.info(
        f"[{label}] ICA: scores_2d={scores_2d.shape}, "
        f"components_2d={components_2d.shape}"
    )

    # Plot 1 — PCA scree
    _plot_pca_scree(
        pca.explained_variance_ratio_,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )

    # Plot 2 — ISC matrix
    _plot_isc_matrix(
        components_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"isc_component_matrix_{label}.png",
    )

    # Plot 3 — Temporal profiles
    _plot_temporal_profiles(
        components_2d,
        time,
        n_ica,
        label=label,
        save_path=out_dir / f"ica_temporal_profiles_{label}.png",
    )

    # Plot 4 — Freq × Time mean-loading heatmap
    _plot_time_frequency(
        scores_2d,
        components_2d,
        bb_z,
        time,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"ica_time_frequency_{label}.png",
    )

    # Plot 5+6 — Mean and variance topomaps
    _plot_topomaps(
        scores_2d,
        components_2d,
        info,
        n_channels,
        n_ica,
        label=label,
        save_path_mean=out_dir / f"ica_topomap_mean_{label}.png",
        save_path_var=out_dir / f"ica_topomap_variance_{label}.png",
    )

    # Plot 7 — Per-subject loading bars
    _plot_subject_loadings(
        components_2d,
        n_ica,
        label=label,
        save_path=out_dir / f"ica_subject_loadings_{label}.png",
    )

    _logger.info(f"[{label}] Freq-Channel Features: 7 plots saved to {out_dir}")


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
        f"Freq-Channel Features ICA: condition={condition.value}, "
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
        _run_freq_channel_features(
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

    _logger.info("Freq-Channel Features ICA analysis complete.")
