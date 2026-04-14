"""
Standalone script to run the Stage-04 wavelet-ICA / PCA analysis.

Implements the production version of the workflows demonstrated in
``notebooks/04-wavelet-ica-analysis/``:

* ``wavelet_ica_superbrain.ipynb``
* ``wavelet_ica_intersubject.ipynb``
* ``wavelet_ica_temporal.ipynb``
* ``wavelet_ica_inverted_superbrain.ipynb``

For each requested ``(condition, music_type)`` the script:

1. Loads preprocessed EEG data via
   :func:`scripts.analysis_common.load_analyzers` and converts it to
   :class:`~src.analysis.data_representations.AnalysisData` objects.
2. Computes the shared 4-D wavelet-power tensor
   ``(n_subjects, n_channels, n_freqs, n_times)`` via
   :func:`scripts.analysis_common._broadband_wavelet_4d`.
3. Runs each of the four PCA/ICA decomposition strategies from
   :mod:`src.analysis.wavelet_ica` and saves every plot from
   :mod:`src.visualization.wavelet_ica_plots` into the canonical layout.
   Each decomposition type lives under its own stage directory so the
   Results Browser can filter by approach independently::

       plots/
           04-superbrain-wavelet-ica-analysis/<Condition>_<MusicType>/
               broadband/superbrain/*.png
               bands/superbrain/<band>_*.png
               bands/cross_band_scree/*.png
           04-intersubject-wavelet-ica-analysis/<Condition>_<MusicType>/
               broadband/intersubject/*.png
               bands/intersubject/<band>_*.png
               bands/cross_band_scree/*.png
           04-temporal-wavelet-ica-analysis/<Condition>_<MusicType>/
               broadband/temporal/*.png
               bands/temporal/<band>_*.png
               bands/cross_band_scree/*.png
           04-inverted-superbrain-wavelet-ica-analysis/<Condition>_<MusicType>/
               broadband/inverted_superbrain/*.png
               bands/inverted_superbrain/<band>_*.png
               bands/cross_band_scree/*.png
           04-wavelet-ica-cross-band-summary/<Condition>_<MusicType>/
               bands/variance_summary/variance_summary_*.png

Usage examples::

    # Default: Placebo condition, both music types, broadband + bands
    python scripts/run_wavelet_ica.py

    # Broadband only
    python scripts/run_wavelet_ica.py --skip_bands

    # Bands only, Psytrance, reuse cached wavelets
    python scripts/run_wavelet_ica.py --skip_broadband --music_type PSYTRANCE --reuse_wavelets

    # Custom PCA / ICA dimensionality
    python scripts/run_wavelet_ica.py --n_pca 30 --n_ica 15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analysis_common import (  # noqa: E402
    _broadband_wavelet_4d,
    analyzers_to_datasets,
    load_analyzers,
)
from src.analysis.data_representations import AnalysisData  # noqa: E402
from src.analysis.isc import FREQUENCY_BANDS, compute_loo_isc  # noqa: E402
from src.analysis.wavelet_ica import (  # noqa: E402
    InterSubjectResult,
    InvertedSuperBrainResult,
    SuperBrainResult,
    TemporalResult,
    decompose_intersubject,
    decompose_inverted_superbrain,
    decompose_superbrain,
    decompose_temporal,
)
from src.definitions.constants import ProjectPaths  # noqa: E402
from src.definitions.fields import (  # noqa: E402
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    MusicTypeVariants,
)
from src.visualization.wavelet_ica_plots import (  # noqa: E402
    plot_cross_band_scree_comparison,
    plot_cross_band_variance_summary,
    plot_intersubject_band_resolved_loadings,
    plot_intersubject_cross_component_correlation,
    plot_intersubject_group_mean_activations,
    plot_intersubject_ica_channel_freq_maps,
    plot_intersubject_ica_interindividual_correlation,
    plot_intersubject_ica_topomap_mean,
    plot_intersubject_ica_topomap_variance,
    plot_intersubject_pca_channel_freq_maps,
    plot_intersubject_pca_scree,
    plot_intersubject_pca_subject_activations,
    plot_intersubject_pca_topomaps,
    plot_intersubject_similarity,
    plot_inverted_cross_component_correlation,
    plot_inverted_ica_channel_topomap,
    plot_inverted_ica_freq_loadings,
    plot_inverted_ica_interindividual_correlation,
    plot_inverted_ica_spectrograms,
    plot_inverted_ica_subject_loadings,
    plot_inverted_ica_temporal_patterns,
    plot_inverted_ica_topomap_mean,
    plot_inverted_ica_topomap_variance,
    plot_inverted_pca_scree,
    plot_superbrain_cross_component_correlation,
    plot_superbrain_ica_interindividual_correlation,
    plot_superbrain_ica_timecourses,
    plot_superbrain_ica_topomap_mean,
    plot_superbrain_ica_topomap_variance,
    plot_superbrain_pca_channel_topomap,
    plot_superbrain_pca_component_isc,
    plot_superbrain_pca_freq_loadings,
    plot_superbrain_pca_scree,
    plot_superbrain_pca_spectrograms,
    plot_superbrain_pca_subject_loadings,
    plot_superbrain_pca_timecourses,
    plot_temporal_ica_band_energy,
    plot_temporal_ica_component_correlation,
    plot_temporal_ica_freq_time_maps,
    plot_temporal_ica_intersubject_correlation,
    plot_temporal_ica_per_subject_topomaps,
    plot_temporal_ica_spectral_profiles,
    plot_temporal_ica_timecourses,
    plot_temporal_ica_topomap_mean,
    plot_temporal_ica_topomap_variance,
    plot_temporal_pca_scree,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-approach output stage directory names
# ---------------------------------------------------------------------------

#: Maps approach key → top-level stage directory name under the plots root.
#: Each approach gets its own Results-Browser-compatible stage directory so
#: users can filter by approach independently.
_APPROACH_STAGE_DIRS: dict[str, str] = {
    "superbrain": "04-superbrain-wavelet-ica-analysis",
    "intersubject": "04-intersubject-wavelet-ica-analysis",
    "temporal": "04-temporal-wavelet-ica-analysis",
    "inverted_superbrain": "04-inverted-superbrain-wavelet-ica-analysis",
}

#: Maps approach key → human-readable display name used in plot titles and
#: EVR accumulator keys.  Keeps ``_APPROACH_STAGE_DIRS`` as the single source
#: of truth for approach enumeration so both dicts stay in sync.
_APPROACH_DISPLAY_NAMES: dict[str, str] = {
    "superbrain": "Super-Brain",
    "intersubject": "Inter-Subject",
    "temporal": "Temporal",
    "inverted_superbrain": "Inverted Super-Brain",
}

#: Stage directory for cross-approach band comparison plots.
_CROSS_BAND_SUMMARY_STAGE = "04-wavelet-ica-cross-band-summary"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Stage-04 wavelet-ICA / PCA analysis on preprocessed EEG "
            "data. Produces all four decomposition approaches (Super-Brain, "
            "Inter-Subject, Temporal, Inverted Super-Brain) and writes every "
            "plot into the canonical per-condition layout."
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
        help=(
            "Load raw .fif files, resample, stack, and save .npy caches before "
            "running the analysis. By default cached files are used."
        ),
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
        help=(
            "Base directory for output plots. "
            "Each approach is written under its own subdirectory: "
            "04-{approach}-wavelet-ica-analysis/. "
            "Defaults to the project plots/ root."
        ),
    )
    parser.add_argument(
        "--skip_broadband",
        action="store_true",
        help="Skip the broadband (full-spectrum) decomposition.",
    )
    parser.add_argument(
        "--skip_bands",
        action="store_true",
        help="Skip the per-frequency-band decomposition.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_4d_wavelet(
    ad: AnalysisData,
    label: str,
    freqs: np.ndarray,
    wavelet_dir: Path | None,
    reuse_wavelets: bool,
) -> AnalysisData:
    """Compute / load the shared 4D wavelet-power tensor for a dataset."""
    wd = _broadband_wavelet_4d(
        ad,
        label,
        representation="power",
        freqs=freqs,
        wavelet_dir=(wavelet_dir / "broadband") if wavelet_dir is not None else None,
        reuse_wavelets=reuse_wavelets,
    )
    _logger.info(f"[{label}] wavelet 4D shape: {wd.data.shape}   sfreq={wd.sfreq} Hz")
    return wd


# ---------------------------------------------------------------------------
# Super-Brain approach
# ---------------------------------------------------------------------------


def _run_superbrain(
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
) -> SuperBrainResult:
    """Super-Brain decomposition + 12 plots."""
    _logger.info(f"[{label}] === Super-Brain decomposition ===")
    out_dir = save_dir / "broadband" / "superbrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = decompose_superbrain(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F = r.n_subjects, r.n_channels, r.n_freqs

    # (K, S, C, F) loadings for per-axis marginalisations.
    loadings_4d = r.pca_components.reshape(n_pca, S, C, F)

    # ICA channel loadings per subject: (S, C, n_ica).
    #   ica_mixing: (S*C*F, n_ica) -> (S, C, F, n_ica) -> mean over F -> (S, C, n_ica)
    ica_mix_4d = r.ica_mixing.reshape(S, C, F, n_ica)
    ica_channel_loadings = ica_mix_4d.mean(axis=2)  # (S, C, n_ica)
    ica_ch_mean = ica_channel_loadings.mean(axis=0)  # (C, n_ica)
    ica_ch_var = ica_channel_loadings.var(axis=0)  # (C, n_ica)

    # LOO-ISC on per-subject PCA component time courses.
    #   For each subject s, project their (C, F, T) data onto each component's
    #   per-subject loading slice (C, F) to yield a (K, T) signal per subject,
    #   then stack to (S, K, T) and feed into compute_loo_isc (which takes
    #   ``(n_items, n_features, n_samples)``).
    pca_scores_per_subj = np.einsum("kscf,scft->skt", loadings_4d, data_4d)
    _, mean_loo_isc = compute_loo_isc(pca_scores_per_subj)

    # Plot 1 — PCA scree.
    plot_superbrain_pca_scree(
        r.pca_explained_variance_ratio,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )
    # Plot 2 — PCA time courses.
    plot_superbrain_pca_timecourses(
        r.pca_scores,
        sfreq,
        label=label,
        save_path=out_dir / f"pca_timecourses_{label}.png",
    )
    # Plot 3 — PCA subject loadings (bar).
    plot_superbrain_pca_subject_loadings(
        loadings_4d,
        label=label,
        save_path=out_dir / f"pca_subject_loadings_{label}.png",
    )
    # Plot 4 — PCA channel topomap.
    plot_superbrain_pca_channel_topomap(
        loadings_4d,
        info,
        label=label,
        save_path=out_dir / f"pca_channel_topomap_{label}.png",
    )
    # Plot 5 — PCA frequency loadings.
    plot_superbrain_pca_freq_loadings(
        loadings_4d,
        freqs,
        bands=FREQUENCY_BANDS,
        label=label,
        save_path=out_dir / f"pca_freq_loadings_{label}.png",
    )
    # Plot 6 — PCA spectrograms.
    plot_superbrain_pca_spectrograms(
        r.pca_scores,
        sfreq,
        freqs_max=float(freqs[-1]),
        label=label,
        save_path=out_dir / f"pca_spectrograms_{label}.png",
    )
    # Plot 7 — PCA component ISC bar chart.
    plot_superbrain_pca_component_isc(
        mean_loo_isc,
        label=label,
        save_path=out_dir / f"pca_component_isc_{label}.png",
    )
    # Plot 8 — Cross-component correlation.
    plot_superbrain_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=label,
        save_path=out_dir / f"cross_component_correlation_{label}.png",
    )
    # Plot 9 — ICA time courses.
    plot_superbrain_ica_timecourses(
        r.ica_sources,
        sfreq,
        label=label,
        save_path=out_dir / f"ica_timecourses_{label}.png",
    )
    # Plot 10 — ICA topomap (mean).
    plot_superbrain_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_mean_{label}.png",
    )
    # Plot 11 — ICA topomap (variance).
    plot_superbrain_ica_topomap_variance(
        ica_ch_var,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_variance_{label}.png",
    )
    # Plot 12 — Inter-individual IC correlations.
    plot_superbrain_ica_interindividual_correlation(
        ica_channel_loadings,
        label=label,
        save_path=out_dir / f"ica_interindividual_correlation_{label}.png",
    )
    return r


# ---------------------------------------------------------------------------
# Inter-Subject approach
# ---------------------------------------------------------------------------


def _run_intersubject(
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
) -> InterSubjectResult:
    """Inter-Subject decomposition + 12 plots."""
    _logger.info(f"[{label}] === Inter-Subject decomposition ===")
    out_dir = save_dir / "broadband" / "intersubject"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = decompose_intersubject(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F, T = r.n_subjects, r.n_channels, r.n_freqs, r.n_times

    # PCA components in (K, C, F) — channel × frequency heatmaps.
    pca_components_cf = r.pca_components.reshape(n_pca, C, F)
    # Scores reshape: (S*T, K) -> (S, T, K).
    scores_3d = r.pca_scores.reshape(S, T, n_pca)
    # Channel marginal across frequencies: (K, C).
    channel_marginal = pca_components_cf.mean(axis=2)

    # Band-resolved PCA loadings: {band: (K,)} mean |loading| per band.
    band_loadings: dict[str, np.ndarray] = {}
    for band, (lo, hi) in FREQUENCY_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            continue
        band_loadings[band] = np.abs(pca_components_cf[:, :, mask]).mean(axis=(1, 2))

    # ICA mixing in (C, F, n_ica).
    ica_mixing_cf = r.ica_mixing.reshape(C, F, n_ica)
    # ICA channel loadings per subject for inter-individual correlation:
    #   per-subject score sequences (S, T, n_ica) from ica_sources.
    ica_scores_3d = r.ica_sources.reshape(S, T, n_ica)
    # Channel-averaged ICA loadings (freq-averaged) for topomap.
    ica_ch_mean = ica_mixing_cf.mean(axis=1)  # (C, n_ica)
    ica_ch_var = ica_mixing_cf.var(axis=1)  # (C, n_ica)

    # Plot 1 — PCA scree.
    plot_intersubject_pca_scree(
        r.pca_explained_variance_ratio,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )
    # Plot 2 — PCA channel × frequency maps.
    plot_intersubject_pca_channel_freq_maps(
        pca_components_cf,
        freqs,
        label=label,
        save_path=out_dir / f"pca_channel_freq_maps_{label}.png",
    )
    # Plot 3 — per-subject PCA activations.
    plot_intersubject_pca_subject_activations(
        scores_3d,
        sfreq,
        label=label,
        save_path=out_dir / f"pca_subject_activations_{label}.png",
    )
    # Plot 4 — PCA topomaps (freq-averaged).
    plot_intersubject_pca_topomaps(
        channel_marginal,
        info,
        label=label,
        save_path=out_dir / f"pca_topomaps_{label}.png",
    )
    # Plot 5 — inter-subject similarity (S×S per component).
    plot_intersubject_similarity(
        scores_3d,
        label=label,
        save_path=out_dir / f"intersubject_similarity_{label}.png",
    )
    # Plot 6 — band-resolved loadings.
    if band_loadings:
        plot_intersubject_band_resolved_loadings(
            band_loadings,
            label=label,
            save_path=out_dir / f"band_resolved_loadings_{label}.png",
        )
    # Plot 7 — group-mean component activations.
    plot_intersubject_group_mean_activations(
        scores_3d,
        sfreq,
        label=label,
        save_path=out_dir / f"group_mean_activations_{label}.png",
    )
    # Plot 8 — cross-component correlation.
    plot_intersubject_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=label,
        save_path=out_dir / f"cross_component_correlation_{label}.png",
    )
    # Plot 9 — ICA channel × frequency maps.
    plot_intersubject_ica_channel_freq_maps(
        ica_mixing_cf,
        freqs,
        label=label,
        save_path=out_dir / f"ica_channel_freq_maps_{label}.png",
    )
    # Plot 10 — ICA topomap (mean).
    plot_intersubject_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_mean_{label}.png",
    )
    # Plot 11 — ICA topomap (variance).
    plot_intersubject_ica_topomap_variance(
        ica_ch_var,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_variance_{label}.png",
    )
    # Plot 12 — inter-individual IC correlations (time-course based).
    plot_intersubject_ica_interindividual_correlation(
        ica_scores_3d,
        label=label,
        save_path=out_dir / f"ica_interindividual_correlation_{label}.png",
    )
    return r


# ---------------------------------------------------------------------------
# Temporal approach
# ---------------------------------------------------------------------------


def _run_temporal(
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
) -> TemporalResult:
    """Temporal decomposition + 10 plots."""
    _logger.info(f"[{label}] === Temporal decomposition ===")
    out_dir = save_dir / "broadband" / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = decompose_temporal(data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state)
    S, C, F, T = r.n_subjects, r.n_channels, r.n_freqs, r.n_times

    # ICA mixing in (F, T, n_ica) — freq × time component maps.
    ica_mixing_ft = r.ica_mixing.reshape(F, T, n_ica)
    #   For plot_temporal_ica_freq_time_maps we need (n_ica, F, T).
    ica_components_ft = ica_mixing_ft.transpose(2, 0, 1)

    # ICA scores in (S, C, n_ica).
    ica_scores_3d = r.ica_sources.reshape(S, C, n_ica)
    ica_channel_mean = ica_scores_3d.mean(axis=0)  # (C, n_ica)
    ica_channel_var = ica_scores_3d.var(axis=0)  # (C, n_ica)

    # ICA time profiles (freq-averaged): (n_ica, T)
    ica_time_profiles = ica_components_ft.mean(axis=1)  # (n_ica, T)
    # ICA frequency profiles (time-averaged): (n_ica, F)
    ica_freq_profiles = ica_components_ft.mean(axis=2)  # (n_ica, F)

    # Band energy: {band: (n_ica,)} mean squared weight per band per IC.
    band_energy: dict[str, np.ndarray] = {}
    for band, (lo, hi) in FREQUENCY_BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            continue
        # Mean over (F_in_band, T) of squared weights.
        band_energy[band] = (ica_components_ft[:, mask, :] ** 2).mean(axis=(1, 2))

    # Plot 1 — PCA scree.
    plot_temporal_pca_scree(
        r.pca_explained_variance_ratio,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )
    # Plot 2 — ICA freq×time maps.
    plot_temporal_ica_freq_time_maps(
        ica_components_ft,
        freqs,
        sfreq,
        label=label,
        save_path=out_dir / f"ica_freq_time_maps_{label}.png",
    )
    # Plot 3 — ICA topomap (mean).
    plot_temporal_ica_topomap_mean(
        ica_channel_mean,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_mean_{label}.png",
    )
    # Plot 4 — ICA topomap (variance).
    plot_temporal_ica_topomap_variance(
        ica_channel_var,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_variance_{label}.png",
    )
    # Plot 5 — ICA per-subject topomaps.
    plot_temporal_ica_per_subject_topomaps(
        ica_scores_3d,
        info,
        label=label,
        save_path=out_dir / f"ica_per_subject_topomaps_{label}.png",
    )
    # Plot 6 — ICA time courses (freq-averaged).
    plot_temporal_ica_timecourses(
        ica_time_profiles,
        sfreq,
        label=label,
        save_path=out_dir / f"ica_timecourses_{label}.png",
    )
    # Plot 7 — ICA spectral profiles (time-averaged).
    plot_temporal_ica_spectral_profiles(
        ica_freq_profiles,
        freqs,
        bands=FREQUENCY_BANDS,
        label=label,
        save_path=out_dir / f"ica_spectral_profiles_{label}.png",
    )
    # Plot 8 — Inter-subject IC correlations (channel loadings).
    plot_temporal_ica_intersubject_correlation(
        ica_scores_3d,
        label=label,
        save_path=out_dir / f"ica_intersubject_correlation_{label}.png",
    )
    # Plot 9 — Band-resolved IC energy.
    if band_energy:
        plot_temporal_ica_band_energy(
            band_energy,
            label=label,
            save_path=out_dir / f"ica_band_energy_{label}.png",
        )
    # Plot 10 — ICA component correlation (temporal profiles).
    plot_temporal_ica_component_correlation(
        ica_time_profiles,
        label=label,
        save_path=out_dir / f"ica_component_correlation_{label}.png",
    )
    return r


# ---------------------------------------------------------------------------
# Inverted Super-Brain approach
# ---------------------------------------------------------------------------


def _run_inverted_superbrain(
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
) -> InvertedSuperBrainResult:
    """Inverted Super-Brain decomposition + 10 plots."""
    _logger.info(f"[{label}] === Inverted Super-Brain decomposition ===")
    out_dir = save_dir / "broadband" / "inverted_superbrain"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = decompose_inverted_superbrain(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F = r.n_subjects, r.n_channels, r.n_freqs

    # ICA temporal patterns: mixing in (T, n_ica) -> (n_ica, T).
    ica_components = r.ica_mixing.T  # (n_ica, T)

    # ICA scores: (S*C*F, n_ica) -> (S, C, F, n_ica).
    ica_scores_4d = r.ica_sources.reshape(S, C, F, n_ica)

    # Subject loadings: (S, n_ica) mean |score| over C and F.
    subject_loadings = np.abs(ica_scores_4d).mean(axis=(1, 2))
    # Channel loadings: (C, n_ica) mean |score| over S and F.
    channel_loadings = np.abs(ica_scores_4d).mean(axis=(0, 2))
    # Channel-mean / -var (signed) for the shared topomap helpers.
    ica_ch_mean = ica_scores_4d.mean(axis=(0, 2))  # (C, n_ica)
    ica_ch_var = ica_scores_4d.mean(axis=2).var(axis=0)  # (C, n_ica)
    # Frequency loadings: (F, n_ica) mean |score| over S and C.
    freq_loadings = np.abs(ica_scores_4d).mean(axis=(0, 1))
    # Inter-individual channel loadings: (S, C, n_ica).
    ica_channel_loadings = np.abs(ica_scores_4d).mean(axis=2)  # (S, C, n_ica)

    # Plot 1 — PCA scree.
    plot_inverted_pca_scree(
        r.pca_explained_variance_ratio,
        label=label,
        save_path=out_dir / f"pca_scree_{label}.png",
    )
    # Plot 2 — ICA temporal patterns.
    plot_inverted_ica_temporal_patterns(
        ica_components,
        sfreq,
        label=label,
        save_path=out_dir / f"ica_temporal_patterns_{label}.png",
    )
    # Plot 3 — ICA subject loadings.
    plot_inverted_ica_subject_loadings(
        subject_loadings,
        label=label,
        save_path=out_dir / f"ica_subject_loadings_{label}.png",
    )
    # Plot 4 — ICA channel topomap (|score|).
    plot_inverted_ica_channel_topomap(
        channel_loadings,
        info,
        label=label,
        save_path=out_dir / f"ica_channel_topomap_{label}.png",
    )
    # Plot 5 — ICA topomap (mean, signed).
    plot_inverted_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_mean_{label}.png",
    )
    # Plot 6 — ICA topomap (variance).
    plot_inverted_ica_topomap_variance(
        ica_ch_var,
        info,
        label=label,
        save_path=out_dir / f"ica_topomap_variance_{label}.png",
    )
    # Plot 7 — ICA frequency loadings.
    plot_inverted_ica_freq_loadings(
        freq_loadings,
        freqs,
        label=label,
        save_path=out_dir / f"ica_freq_loadings_{label}.png",
    )
    # Plot 8 — ICA spectrograms.
    plot_inverted_ica_spectrograms(
        ica_components,
        sfreq,
        freqs_max=float(freqs[-1]),
        label=label,
        save_path=out_dir / f"ica_spectrograms_{label}.png",
    )
    # Plot 9 — Inter-individual IC correlations (channel loadings).
    plot_inverted_ica_interindividual_correlation(
        ica_channel_loadings,
        label=label,
        save_path=out_dir / f"ica_interindividual_correlation_{label}.png",
    )
    # Plot 10 — Cross-component correlation.
    plot_inverted_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=label,
        save_path=out_dir / f"cross_component_correlation_{label}.png",
    )
    return r


# ---------------------------------------------------------------------------
# Per-band helpers — each _run_<approach>_band writes into
#   save_dir / "bands" / <approach> / <band>_<plot>.png
# following the canonical layout.
# ---------------------------------------------------------------------------


def _run_superbrain_band(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    band: str,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    out_dir: Path,
) -> SuperBrainResult:
    """Super-Brain decomposition for a single band — same plots, band-prefixed."""
    band_label = f"{label} [{band}]"
    _logger.info(f"[{band_label}] === Super-Brain (band) ===")

    r = decompose_superbrain(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F = r.n_subjects, r.n_channels, r.n_freqs
    n_ica_val = r.ica_sources.shape[1]

    loadings_4d = r.pca_components.reshape(n_pca, S, C, F)
    ica_mix_4d = r.ica_mixing.reshape(S, C, F, n_ica_val)
    ica_channel_loadings = ica_mix_4d.mean(axis=2)
    ica_ch_mean = ica_channel_loadings.mean(axis=0)
    ica_ch_var = ica_channel_loadings.var(axis=0)

    pca_scores_per_subj = np.einsum("kscf,scft->skt", loadings_4d, data_4d)
    _, mean_loo_isc = compute_loo_isc(pca_scores_per_subj)

    plot_superbrain_pca_scree(
        r.pca_explained_variance_ratio,
        label=band_label,
        save_path=out_dir / f"{band}_pca_scree_{label}.png",
    )
    plot_superbrain_pca_timecourses(
        r.pca_scores,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_pca_timecourses_{label}.png",
    )
    plot_superbrain_pca_subject_loadings(
        loadings_4d,
        label=band_label,
        save_path=out_dir / f"{band}_pca_subject_loadings_{label}.png",
    )
    plot_superbrain_pca_channel_topomap(
        loadings_4d,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_pca_channel_topomap_{label}.png",
    )
    plot_superbrain_pca_freq_loadings(
        loadings_4d,
        freqs,
        bands=FREQUENCY_BANDS,
        label=band_label,
        save_path=out_dir / f"{band}_pca_freq_loadings_{label}.png",
    )
    plot_superbrain_pca_spectrograms(
        r.pca_scores,
        sfreq,
        freqs_max=float(freqs[-1]),
        label=band_label,
        save_path=out_dir / f"{band}_pca_spectrograms_{label}.png",
    )
    plot_superbrain_pca_component_isc(
        mean_loo_isc,
        label=band_label,
        save_path=out_dir / f"{band}_pca_component_isc_{label}.png",
    )
    plot_superbrain_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=band_label,
        save_path=out_dir / f"{band}_cross_component_correlation_{label}.png",
    )
    plot_superbrain_ica_timecourses(
        r.ica_sources,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_ica_timecourses_{label}.png",
    )
    plot_superbrain_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_mean_{label}.png",
    )
    plot_superbrain_ica_topomap_variance(
        ica_ch_var,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_variance_{label}.png",
    )
    plot_superbrain_ica_interindividual_correlation(
        ica_channel_loadings,
        label=band_label,
        save_path=out_dir / f"{band}_ica_interindividual_correlation_{label}.png",
    )
    return r


def _run_intersubject_band(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    band: str,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    out_dir: Path,
) -> InterSubjectResult:
    """Inter-Subject decomposition for a single band — band-prefixed plots."""
    band_label = f"{label} [{band}]"
    _logger.info(f"[{band_label}] === Inter-Subject (band) ===")

    r = decompose_intersubject(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F, T = r.n_subjects, r.n_channels, r.n_freqs, r.n_times

    pca_components_cf = r.pca_components.reshape(n_pca, C, F)
    scores_3d = r.pca_scores.reshape(S, T, n_pca)
    channel_marginal = pca_components_cf.mean(axis=2)

    ica_mixing_cf = r.ica_mixing.reshape(C, F, r.ica_sources.shape[1])
    ica_scores_3d = r.ica_sources.reshape(S, T, r.ica_sources.shape[1])
    ica_ch_mean = ica_mixing_cf.mean(axis=1)
    ica_ch_var = ica_mixing_cf.var(axis=1)

    plot_intersubject_pca_scree(
        r.pca_explained_variance_ratio,
        label=band_label,
        save_path=out_dir / f"{band}_pca_scree_{label}.png",
    )
    plot_intersubject_pca_channel_freq_maps(
        pca_components_cf,
        freqs,
        label=band_label,
        save_path=out_dir / f"{band}_pca_channel_freq_maps_{label}.png",
    )
    plot_intersubject_pca_subject_activations(
        scores_3d,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_pca_subject_activations_{label}.png",
    )
    plot_intersubject_pca_topomaps(
        channel_marginal,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_pca_topomaps_{label}.png",
    )
    plot_intersubject_similarity(
        scores_3d,
        label=band_label,
        save_path=out_dir / f"{band}_intersubject_similarity_{label}.png",
    )
    plot_intersubject_group_mean_activations(
        scores_3d,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_group_mean_activations_{label}.png",
    )
    plot_intersubject_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=band_label,
        save_path=out_dir / f"{band}_cross_component_correlation_{label}.png",
    )
    plot_intersubject_ica_channel_freq_maps(
        ica_mixing_cf,
        freqs,
        label=band_label,
        save_path=out_dir / f"{band}_ica_channel_freq_maps_{label}.png",
    )
    plot_intersubject_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_mean_{label}.png",
    )
    plot_intersubject_ica_topomap_variance(
        ica_ch_var,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_variance_{label}.png",
    )
    plot_intersubject_ica_interindividual_correlation(
        ica_scores_3d,
        label=band_label,
        save_path=out_dir / f"{band}_ica_interindividual_correlation_{label}.png",
    )
    return r


def _run_temporal_band(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    band: str,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    out_dir: Path,
) -> TemporalResult:
    """Temporal decomposition for a single band — band-prefixed plots."""
    band_label = f"{label} [{band}]"
    _logger.info(f"[{band_label}] === Temporal (band) ===")

    r = decompose_temporal(data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state)
    S, C, F, T = r.n_subjects, r.n_channels, r.n_freqs, r.n_times

    ica_mixing_ft = r.ica_mixing.reshape(F, T, r.ica_sources.shape[1])
    ica_components_ft = ica_mixing_ft.transpose(2, 0, 1)
    ica_scores_3d = r.ica_sources.reshape(S, C, r.ica_sources.shape[1])
    ica_channel_mean = ica_scores_3d.mean(axis=0)
    ica_channel_var = ica_scores_3d.var(axis=0)
    ica_time_profiles = ica_components_ft.mean(axis=1)
    ica_freq_profiles = ica_components_ft.mean(axis=2)

    plot_temporal_pca_scree(
        r.pca_explained_variance_ratio,
        label=band_label,
        save_path=out_dir / f"{band}_pca_scree_{label}.png",
    )
    plot_temporal_ica_freq_time_maps(
        ica_components_ft,
        freqs,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_ica_freq_time_maps_{label}.png",
    )
    plot_temporal_ica_topomap_mean(
        ica_channel_mean,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_mean_{label}.png",
    )
    plot_temporal_ica_topomap_variance(
        ica_channel_var,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_variance_{label}.png",
    )
    plot_temporal_ica_per_subject_topomaps(
        ica_scores_3d,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_per_subject_topomaps_{label}.png",
    )
    plot_temporal_ica_timecourses(
        ica_time_profiles,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_ica_timecourses_{label}.png",
    )
    plot_temporal_ica_spectral_profiles(
        ica_freq_profiles,
        freqs,
        bands=FREQUENCY_BANDS,
        label=band_label,
        save_path=out_dir / f"{band}_ica_spectral_profiles_{label}.png",
    )
    plot_temporal_ica_intersubject_correlation(
        ica_scores_3d,
        label=band_label,
        save_path=out_dir / f"{band}_ica_intersubject_correlation_{label}.png",
    )
    plot_temporal_ica_component_correlation(
        ica_time_profiles,
        label=band_label,
        save_path=out_dir / f"{band}_ica_component_correlation_{label}.png",
    )
    return r


def _run_inverted_superbrain_band(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    band: str,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    out_dir: Path,
) -> InvertedSuperBrainResult:
    """Inverted Super-Brain decomposition for a single band — band-prefixed."""
    band_label = f"{label} [{band}]"
    _logger.info(f"[{band_label}] === Inverted Super-Brain (band) ===")

    r = decompose_inverted_superbrain(
        data_4d, n_pca=n_pca, n_ica=n_ica, random_state=random_state
    )
    S, C, F = r.n_subjects, r.n_channels, r.n_freqs
    n_ica_val = r.ica_sources.shape[1]

    ica_components = r.ica_mixing.T
    ica_scores_4d = r.ica_sources.reshape(S, C, F, n_ica_val)
    subject_loadings = np.abs(ica_scores_4d).mean(axis=(1, 2))
    channel_loadings = np.abs(ica_scores_4d).mean(axis=(0, 2))
    ica_ch_mean = ica_scores_4d.mean(axis=(0, 2))
    ica_ch_var = ica_scores_4d.mean(axis=2).var(axis=0)
    freq_loadings = np.abs(ica_scores_4d).mean(axis=(0, 1))
    ica_channel_loadings = np.abs(ica_scores_4d).mean(axis=2)

    plot_inverted_pca_scree(
        r.pca_explained_variance_ratio,
        label=band_label,
        save_path=out_dir / f"{band}_pca_scree_{label}.png",
    )
    plot_inverted_ica_temporal_patterns(
        ica_components,
        sfreq,
        label=band_label,
        save_path=out_dir / f"{band}_ica_temporal_patterns_{label}.png",
    )
    plot_inverted_ica_subject_loadings(
        subject_loadings,
        label=band_label,
        save_path=out_dir / f"{band}_ica_subject_loadings_{label}.png",
    )
    plot_inverted_ica_channel_topomap(
        channel_loadings,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_channel_topomap_{label}.png",
    )
    plot_inverted_ica_topomap_mean(
        ica_ch_mean,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_mean_{label}.png",
    )
    plot_inverted_ica_topomap_variance(
        ica_ch_var,
        info,
        label=band_label,
        save_path=out_dir / f"{band}_ica_topomap_variance_{label}.png",
    )
    plot_inverted_ica_freq_loadings(
        freq_loadings,
        freqs,
        label=band_label,
        save_path=out_dir / f"{band}_ica_freq_loadings_{label}.png",
    )
    plot_inverted_ica_spectrograms(
        ica_components,
        sfreq,
        freqs_max=float(freqs[-1]),
        label=band_label,
        save_path=out_dir / f"{band}_ica_spectrograms_{label}.png",
    )
    plot_inverted_ica_interindividual_correlation(
        ica_channel_loadings,
        label=band_label,
        save_path=out_dir / f"{band}_ica_interindividual_correlation_{label}.png",
    )
    plot_inverted_cross_component_correlation(
        r.pca_scores,
        r.ica_sources,
        label=band_label,
        save_path=out_dir / f"{band}_cross_component_correlation_{label}.png",
    )
    return r


# ---------------------------------------------------------------------------
# Orchestrator: run all four approaches for a single band
# ---------------------------------------------------------------------------


def _run_all_approaches_for_band(
    data_4d: np.ndarray,
    sfreq: float,
    freqs: np.ndarray,
    info,
    *,
    band: str,
    label: str,
    n_pca: int,
    n_ica: int,
    random_state: int,
    approach_save_dirs: dict[str, Path],
) -> dict[str, np.ndarray]:
    """Run all four decompositions for a single frequency band.

    Returns ``{approach: explained_variance_ratio}`` for cross-band summaries.

    :param approach_save_dirs: Mapping from approach key (``"superbrain"``,
        ``"intersubject"``, ``"temporal"``, ``"inverted_superbrain"``) to the
        per-approach dataset root directory.  Each approach writes its band
        plots under ``<approach_save_dirs[approach]>/bands/<approach>/``.
    """
    approaches_dir = {
        approach: approach_save_dirs[approach] / "bands" / approach
        for approach in _APPROACH_STAGE_DIRS
    }
    for d in approaches_dir.values():
        d.mkdir(parents=True, exist_ok=True)

    evr: dict[str, np.ndarray] = {}

    r_sb = _run_superbrain_band(
        data_4d,
        sfreq,
        freqs,
        info,
        band=band,
        label=label,
        n_pca=n_pca,
        n_ica=n_ica,
        random_state=random_state,
        out_dir=approaches_dir["superbrain"],
    )
    evr[_APPROACH_DISPLAY_NAMES["superbrain"]] = r_sb.pca_explained_variance_ratio

    r_is = _run_intersubject_band(
        data_4d,
        sfreq,
        freqs,
        info,
        band=band,
        label=label,
        n_pca=n_pca,
        n_ica=n_ica,
        random_state=random_state,
        out_dir=approaches_dir["intersubject"],
    )
    evr[_APPROACH_DISPLAY_NAMES["intersubject"]] = r_is.pca_explained_variance_ratio

    r_tm = _run_temporal_band(
        data_4d,
        sfreq,
        freqs,
        info,
        band=band,
        label=label,
        n_pca=n_pca,
        n_ica=n_ica,
        random_state=random_state,
        out_dir=approaches_dir["temporal"],
    )
    evr[_APPROACH_DISPLAY_NAMES["temporal"]] = r_tm.pca_explained_variance_ratio

    r_inv = _run_inverted_superbrain_band(
        data_4d,
        sfreq,
        freqs,
        info,
        band=band,
        label=label,
        n_pca=n_pca,
        n_ica=n_ica,
        random_state=random_state,
        out_dir=approaches_dir["inverted_superbrain"],
    )
    evr[_APPROACH_DISPLAY_NAMES["inverted_superbrain"]] = (
        r_inv.pca_explained_variance_ratio
    )

    return evr


def _run_band_analysis(
    ad: AnalysisData,
    label: str,
    freqs: np.ndarray,
    info,
    *,
    wavelet_dir: Path | None,
    reuse_wavelets: bool,
    n_pca: int,
    n_ica: int,
    random_state: int,
    approach_save_dirs: dict[str, Path],
    cross_band_summary_dir: Path,
) -> None:
    """Run the full per-band wavelet-ICA analysis for one dataset.

    For each band in ``FREQUENCY_BANDS``:

    1. Band-pass filter the raw EEG data.
    2. Compute 4-D wavelet-power on the filtered data.
    3. Run all four decomposition approaches.

    Then produce per-approach cross-band scree plots (written into each
    approach's own directory) and the cross-approach variance summary
    (written into ``cross_band_summary_dir``).

    :param approach_save_dirs: Mapping from approach key to that approach's
        dataset root directory (``plots/<stage>/<dataset_key>``).
    :param cross_band_summary_dir: Directory for cross-approach variance
        summary plots (``plots/04-wavelet-ica-cross-band-summary/<dataset_key>``).
    """
    _logger.info(f"[{label}] ===== Per-band wavelet-ICA analysis =====")

    # Accumulate per-approach, per-band explained-variance ratios for
    # cross-band summary plots.  Keyed by display name so plot titles match.
    approach_band_evr: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in _APPROACH_DISPLAY_NAMES.values()
    }

    for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
        _logger.info(f"[{label}] ── Band: {band} ({l_freq}–{h_freq} Hz) ──")
        filtered = ad.filter_to_band(l_freq, h_freq)

        band_wavelet_dir = (
            (wavelet_dir / "bands" / band) if wavelet_dir is not None else None
        )
        wd_4d = _compute_4d_wavelet(
            filtered, f"{label}_{band}", freqs, band_wavelet_dir, reuse_wavelets
        )
        data_4d = wd_4d.data
        sfreq = wd_4d.sfreq

        evr = _run_all_approaches_for_band(
            data_4d,
            sfreq,
            freqs,
            info,
            band=band,
            label=label,
            n_pca=n_pca,
            n_ica=n_ica,
            random_state=random_state,
            approach_save_dirs=approach_save_dirs,
        )
        for approach, ratios in evr.items():
            approach_band_evr[approach][band] = ratios

    # ── Per-approach cross-band scree comparison ─────────────────────────
    _logger.info(f"[{label}] Per-approach cross-band scree plots …")
    for approach_key, display_name in _APPROACH_DISPLAY_NAMES.items():
        slug = display_name.lower().replace("-", "").replace(" ", "_")
        band_evr = approach_band_evr[display_name]
        scree_dir = approach_save_dirs[approach_key] / "bands" / "cross_band_scree"
        scree_dir.mkdir(parents=True, exist_ok=True)
        plot_cross_band_scree_comparison(
            band_evr,
            approach=display_name,
            label=label,
            save_path=scree_dir / f"{slug}_scree_comparison_{label}.png",
        )

    # ── Cross-approach variance summary ─────────────────────────────────
    _logger.info(f"[{label}] Cross-approach variance summary …")
    summary_dir = cross_band_summary_dir / "bands" / "variance_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_band_variance_summary(
        approach_band_evr,
        label=label,
        save_path=summary_dir / f"variance_summary_{label}.png",
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
        args.wavelet_freq_min, args.wavelet_freq_max, args.wavelet_n_freqs
    )

    _logger.info(
        f"Starting wavelet-ICA analysis: condition={condition.value}, "
        f"music_types={[mt.value for mt in music_types]}, "
        f"n_pca={args.n_pca}, n_ica={args.n_ica}, "
        f"broadband={not args.skip_broadband}, bands={not args.skip_bands}"
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
        label = mt.value
        dataset_key = f"{condition.value}_{label}"
        if dataset_key not in datasets:
            _logger.warning(f"No data found for music type {label!r}; skipping.")
            continue

        ad = datasets[dataset_key]
        analyzer = analyzers.get(dataset_key)
        info = getattr(analyzer, "info", None) if analyzer is not None else None

        _logger.info(
            f"Dataset [{dataset_key}]: shape={ad.data.shape}  sfreq={ad.sfreq} Hz"
        )

        # Per-approach save directories (each approach lives under its own stage).
        approach_save_dirs = {
            approach: save_root / stage_dir / dataset_key
            for approach, stage_dir in _APPROACH_STAGE_DIRS.items()
        }
        # Directory for the cross-approach variance summary.
        cross_band_summary_dir = save_root / _CROSS_BAND_SUMMARY_STAGE / dataset_key

        # ── Broadband decomposition ──────────────────────────────────────
        if not args.skip_broadband:
            wd_4d = _compute_4d_wavelet(
                ad, dataset_key, freqs, wavelet_dir, args.reuse_wavelets
            )
            data_4d = wd_4d.data
            sfreq = wd_4d.sfreq

            _run_superbrain(
                data_4d,
                sfreq,
                freqs,
                info,
                label=dataset_key,
                n_pca=args.n_pca,
                n_ica=args.n_ica,
                random_state=args.random_state,
                save_dir=approach_save_dirs["superbrain"],
            )
            _run_intersubject(
                data_4d,
                sfreq,
                freqs,
                info,
                label=dataset_key,
                n_pca=args.n_pca,
                n_ica=args.n_ica,
                random_state=args.random_state,
                save_dir=approach_save_dirs["intersubject"],
            )
            _run_temporal(
                data_4d,
                sfreq,
                freqs,
                info,
                label=dataset_key,
                n_pca=args.n_pca,
                n_ica=args.n_ica,
                random_state=args.random_state,
                save_dir=approach_save_dirs["temporal"],
            )
            _run_inverted_superbrain(
                data_4d,
                sfreq,
                freqs,
                info,
                label=dataset_key,
                n_pca=args.n_pca,
                n_ica=args.n_ica,
                random_state=args.random_state,
                save_dir=approach_save_dirs["inverted_superbrain"],
            )

        # ── Per-band decomposition ───────────────────────────────────────
        if not args.skip_bands:
            _run_band_analysis(
                ad,
                dataset_key,
                freqs,
                info,
                wavelet_dir=wavelet_dir,
                reuse_wavelets=args.reuse_wavelets,
                n_pca=args.n_pca,
                n_ica=args.n_ica,
                random_state=args.random_state,
                approach_save_dirs=approach_save_dirs,
                cross_band_summary_dir=cross_band_summary_dir,
            )

    _logger.info("Wavelet-ICA analysis complete.")
