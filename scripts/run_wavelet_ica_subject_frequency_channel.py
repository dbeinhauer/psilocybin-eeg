"""
CLI script reproducing the exact analyses from
``notebooks/04-wavelet-ica-analysis/wavelet_ica_subject_frequency_channel.ipynb``
(Subject–Frequency–Channel approach: observations = T, features = F×C×S;
components live in frequency × channel × subject, scores in time).

Produces **all** ICA components (not just the first 6 shown in the notebook)
and writes every plot into the canonical per-condition layout under::

    plots/04-subject-frequency-channel-wavelet-ica-analysis/<Condition>_<MusicType>/
        broadband/subject_frequency_channel/        # default (no --band)
            pca_scree_<label>.png
            ...
        bands/subject_frequency_channel/            # when --band <name> is given
            <band>_pca_scree_<label>.png
            ...

Pass ``--band <name>`` to slice the cached broadband wavelets down to a
single frequency band (alpha/beta/...) before the ICA step. The same
broadband cache is reused — no separate per-band cache is needed.

Usage::

    # broadband
    python scripts/run_wavelet_ica_subject_frequency_channel.py \\
        --condition Placebo --music_type CLASSIC PSYTRANCE \\
        --n_pca 50 --n_ica 10 --reuse_wavelets

    # alpha-band only
    python scripts/run_wavelet_ica_subject_frequency_channel.py \\
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
import numpy as np  # noqa: E402
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

_STAGE_DIR = "04-subject-frequency-channel-wavelet-ica-analysis"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Subject–Frequency–Channel ICA analysis on preprocessed "
            "EEG wavelet power. Produces the exact same plots as "
            "wavelet_ica_subject_frequency_channel.ipynb, but for ALL ICA "
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
        default=50,
        help="Number of PCA components to retain. Ignored when --skip_pca is set.",
    )
    parser.add_argument(
        "--skip_pca",
        action="store_true",
        help=(
            "Skip the PCA dimensionality reduction and run FastICA directly "
            "on the (T, F*C*S) z-scored matrix."
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
            "plots are written to the 'bands/subject_frequency_channel/' "
            "subdirectory with a '<band>_' filename prefix. Default: full "
            "broadband."
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
    components_3d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (a) — Intersubject correlation of (F×C) loading maps per IC.

    For each IC the per-subject vector is the flattened ``(F, C)`` slice of
    the component, of length ``F*C``; ``np.corrcoef`` gives the ``(S, S)``
    inter-subject correlation per component.
    """
    n_subjects = components_3d.shape[3]
    n_show = n_ica

    fig, axes = plt.subplots(
        1, n_show, figsize=(3.5 * n_show, 3.5), constrained_layout=True
    )
    if n_show == 1:
        axes = [axes]

    im = None
    for i, ax in enumerate(axes):
        subj_maps = components_3d[i].transpose(2, 0, 1).reshape(n_subjects, -1)
        corr_mat = np.corrcoef(subj_maps)  # (S, S)
        im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(n_subjects))
        ax.set_yticks(range(n_subjects))
        ax.set_xticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_yticklabels([f"S{s + 1}" for s in range(n_subjects)], fontsize=7)
        ax.set_title(f"IC {i + 1}", fontsize=10)

    fig.suptitle(
        f"Intersubject Correlation of IC Loading Maps (F×C) — {label}",
        fontsize=12,
    )
    plt.colorbar(im, ax=axes[-1], label="Pearson r", shrink=0.8)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_time_frequency(
    ica_scores: np.ndarray,
    components_3d: np.ndarray,
    time: np.ndarray,
    freqs: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (b) — Time × Frequency map per component (outer product).

    freq_profile[k] = mean_(c, s)  components_3d[k]    (K, F)
    time_profile[k] = ica_scores[:, k]                 (K, T)
    tf_map[k]       = outer(freq_profile[k], time_profile[k])  (K, F, T)
    """
    freq_profiles = components_3d.mean(axis=(2, 3))  # (K, F)
    time_profiles = ica_scores.T  # (K, T)
    ft_maps = np.einsum("kf,kt->kft", freq_profiles, time_profiles)  # (K, F, T)

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


def _plot_loo_isc_bar(
    components_3d: np.ndarray,
    n_ica: int,
    *,
    label: str,
    save_path: Path,
) -> None:
    """Analysis (c) — Mean LOO-ISC across participants per IC (bar plot).

    Per-subject vector for each IC is the flattened ``(F, C)`` loading map
    of length ``F*C``; matches the subject vectors used in Analysis (a).
    """
    n_subjects = components_3d.shape[3]
    loo_isc_per_subject = np.zeros((n_ica, n_subjects))
    for k in range(n_ica):
        subj_vectors = components_3d[k].transpose(2, 0, 1).reshape(n_subjects, -1)
        for s in range(n_subjects):
            others_mean = np.delete(subj_vectors, s, axis=0).mean(axis=0)
            loo_isc_per_subject[k, s] = float(
                pearsonr(subj_vectors[s], others_mean)[0]
            )

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


def _plot_pairwise_heatmap(
    maps: np.ndarray,
    n_ica: int,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    extent: list[float],
    xticks: np.ndarray | None,
    save_path: Path,
) -> None:
    """Generic 1×K row of heatmaps with per-panel symmetric color scale.

    Used for Analysis (d): one figure per pairwise view of ``components_3d``
    averaged along the third dimension.
    """
    n_show = n_ica
    fig, axes = plt.subplots(
        1, n_show, figsize=(3.2 * n_show, 3.6), constrained_layout=True
    )
    if n_show == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        data_i = maps[i]
        lim_i = float(np.percentile(np.abs(data_i), 99)) or 1e-12
        im = ax.imshow(
            data_i,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-lim_i,
            vmax=lim_i,
            origin="lower",
            extent=extent,
        )
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        ax.set_title(f"IC {i + 1}", fontsize=10)
        if i == 0:
            ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle(title, fontsize=13)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------


def _run_subject_frequency_channel(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    *,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    save_dir: Path,
    band: str | None = None,
    skip_pca: bool = False,
) -> None:
    """Run the full Subject–Frequency–Channel ICA pipeline and save all plots.

    When ``band`` is given the outputs land in
    ``bands/subject_frequency_channel/`` with filenames prefixed by
    ``<band>_``. Otherwise the broadband layout is used.
    """
    if band is None:
        out_dir = save_dir / "broadband" / "subject_frequency_channel"
        prefix = ""
    else:
        out_dir = save_dir / "bands" / "subject_frequency_channel"
        prefix = f"{band}_"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subjects, n_channels, n_freqs, n_times = data_4d.shape
    time = np.arange(n_times) / sfreq
    _logger.info(
        f"[{label}] Subject-Frequency-Channel: {data_4d.shape}  sfreq={sfreq} Hz"
    )

    # Step 1 — Z-score and reshape: (S, C, F, T) → (T, F, C, S) → (T, F*C*S)
    bb_z = zscore_by_time(data_4d)
    bb_z_t = bb_z.transpose(3, 2, 1, 0)  # (T, F, C, S)
    n_feat = n_freqs * n_channels * n_subjects
    X_t = bb_z_t.reshape(n_times, n_feat)
    _logger.info(f"[{label}] Reshaped: {X_t.shape}  (T, F*C*S)")

    # Step 2 — (optional) PCA + ICA
    ica = FastICA(
        n_components=n_ica,
        random_state=random_state,
        max_iter=500,
        whiten="unit-variance",
    )

    if skip_pca:
        _logger.info(f"[{label}] Skipping PCA; running FastICA directly on X_t.")
        ica_scores = ica.fit_transform(X_t)  # (T, K)
        ica_components = ica.components_  # (K, F*C*S)
    else:
        pca = PCA(n_components=n_pca, random_state=random_state)
        pca_scores = pca.fit_transform(X_t)
        _logger.info(
            f"[{label}] PCA: {pca_scores.shape}, "
            f"explained={np.cumsum(pca.explained_variance_ratio_)[-1] * 100:.1f}%"
        )
        ica_scores = ica.fit_transform(pca_scores)  # (T, K)
        ica_components = ica.components_ @ pca.components_  # (K, F*C*S)

    # Reshape ICA components to (K, F, C, S)
    components_3d = ica_components.reshape(n_ica, n_freqs, n_channels, n_subjects)
    _logger.info(
        f"[{label}] ICA: scores={ica_scores.shape}, components_3d={components_3d.shape}"
    )

    # Plot 1 — PCA scree (skipped when PCA is not run)
    if not skip_pca:
        _plot_pca_scree(
            pca.explained_variance_ratio_,
            label=label,
            save_path=out_dir / f"{prefix}pca_scree_{label}.png",
        )

    # Plot 2 — (a) Intersubject correlation matrix of (F×C) loading maps
    _plot_isc_matrix(
        components_3d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}isc_component_matrix_{label}.png",
    )

    # Plot 3 — (b) Frequency × Time outer-product map
    _plot_time_frequency(
        ica_scores,
        components_3d,
        time,
        freqs,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_time_frequency_{label}.png",
    )

    # Plot 4 — (c) Mean LOO-ISC across participants per IC
    _plot_loo_isc_bar(
        components_3d,
        n_ica,
        label=label,
        save_path=out_dir / f"{prefix}ica_loo_isc_bar_{label}.png",
    )

    # Plot 5–7 — (d) Pairwise component heatmaps, one figure per pair
    sf_maps = components_3d.mean(axis=2)  # (K, F, S) — mean over channels
    sc_maps = components_3d.mean(axis=1)  # (K, C, S) — mean over frequencies
    fc_maps = components_3d.mean(axis=3)  # (K, F, C) — mean over subjects
    subject_ticks = np.arange(1, n_subjects + 1)

    _plot_pairwise_heatmap(
        sf_maps,
        n_ica,
        title=f"Subject × Frequency (mean over channels) — {label}",
        xlabel="Subject",
        ylabel="Frequency (Hz)",
        extent=[0.5, n_subjects + 0.5, float(freqs[0]), float(freqs[-1])],
        xticks=subject_ticks,
        save_path=out_dir / f"{prefix}ica_pairwise_subject_frequency_{label}.png",
    )

    _plot_pairwise_heatmap(
        sc_maps,
        n_ica,
        title=f"Subject × Channel (mean over frequencies) — {label}",
        xlabel="Subject",
        ylabel="Channel",
        extent=[0.5, n_subjects + 0.5, 0.5, n_channels + 0.5],
        xticks=subject_ticks,
        save_path=out_dir / f"{prefix}ica_pairwise_subject_channel_{label}.png",
    )

    _plot_pairwise_heatmap(
        fc_maps,
        n_ica,
        title=f"Frequency × Channel (mean over subjects) — {label}",
        xlabel="Channel",
        ylabel="Frequency (Hz)",
        extent=[0.5, n_channels + 0.5, float(freqs[0]), float(freqs[-1])],
        xticks=None,
        save_path=out_dir / f"{prefix}ica_pairwise_frequency_channel_{label}.png",
    )

    n_plots = 7 if not skip_pca else 6
    _logger.info(
        f"[{label}] Subject-Frequency-Channel: {n_plots} plots saved to {out_dir}"
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
    freqs = np.linspace(
        args.wavelet_freq_min,
        args.wavelet_freq_max,
        args.wavelet_n_freqs,
    )
    band_name = args.band  # None or e.g. "alpha"

    _logger.info(
        f"Subject-Frequency-Channel ICA: condition={condition.value}, "
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
        _run_subject_frequency_channel(
            data_4d,
            wd.sfreq,
            ica_freqs,
            label=dataset_key,
            n_pca=args.n_pca,
            n_ica=args.n_ica,
            random_state=args.random_state,
            save_dir=save_dir,
            band=band_name,
            skip_pca=args.skip_pca,
        )

    _logger.info("Subject-Frequency-Channel ICA analysis complete.")
