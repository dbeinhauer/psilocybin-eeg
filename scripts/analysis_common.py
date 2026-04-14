"""
Shared helpers and workflow functions for the unified analysis script.

Centralises data-loading, argument-parsing, display-setup and the actual
analysis workflows (ISC and mean/variance) so they can be driven by a single
entry-point script or from a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

from src.definitions.fields import (
    MusicTypeVariants,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    CoordinateSystems,
    SingleDataMetadata,
    FrequencyBandNames,
    AnalysisVariants,
)
from src.definitions.constants import ProjectPaths
import numpy as np
from scipy.stats import circmean

from src.analysis.data_representations import (
    AnalysisData,
    DataRepresentation,
    to_wavelet_phase,
    to_wavelet_power,
)
from src.analysis.isc import (
    compute_loo_isc,
    compute_sliding_window_isc,
    FREQUENCY_BANDS,
)
from src.visualization.isc_plots import (
    plot_loo_isc_distribution,
    plot_sliding_window_isc,
    print_significant_intervals,
    plot_band_isc_distributions,
    plot_band_mean_isc_bar,
    plot_band_sliding_window_isc,
    print_band_significant_intervals,
    plot_band_overlap,
    print_data_overview,
)
from src.visualization.wavelet_plots import (
    compute_itpc,
    compute_phase_band_loo_iscs,
    plot_band_itpc_timecourse,
    plot_band_power_timecourse,
    plot_cross_frequency_coupling,
    plot_intersubject_variance,
    plot_itpc_spectrum,
    plot_itpc_vs_isc,
    plot_phase_distribution,
    plot_power_phase_joint,
    plot_spectral_profile,
    plot_tf_isc,
    plot_tf_itpc_map,
    plot_tf_map,
    plot_wavelet_loo_isc_bar,
    plot_wavelet_loo_isc_distributions,
    plot_wavelet_topomap_isc,
)

if TYPE_CHECKING:
    from src.analysis.summary import EEGSummarizedAnalyzer

_logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

#: Default broadband ISC significance thresholds for each frequency band.
BAND_ISC_THRESHOLDS: dict[str, float] = {
    FrequencyBandNames.DELTA.value: 0.1,
    FrequencyBandNames.THETA.value: 0.07,
    FrequencyBandNames.ALPHA.value: 0.035,
    FrequencyBandNames.BETA.value: 0.02,
    FrequencyBandNames.GAMMA.value: 0.01,
}


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared across all analysis scripts."""
    parser.add_argument(
        "--analysis",
        type=str,
        nargs="+",
        default=[AnalysisVariants.WAVELET_POWER.value],
        choices=[
            analysis.value
            for analysis in AnalysisVariants
            # ISC is handled by scripts/run_isc.py; mean_variance by run_mean_variance.py
            if analysis not in (AnalysisVariants.MEAN_VARIANCE, AnalysisVariants.ISC)
        ],
        help=(
            "Which analyses to run. Defaults to wavelet_power. "
            "Use wavelet_power or wavelet_phase to run wavelet-based analyses. "
            "For ISC analysis use scripts/run_isc.py. "
            "For mean-variance analysis use scripts/run_mean_variance.py."
        ),
    )
    parser.add_argument(
        "--condition",
        type=str,
        default=ConditionVariants.PLACEBO.value,
        choices=[cond.value for cond in ConditionVariants],
        help=(
            "The condition to process, should be one of "
            f"{[cond.value for cond in ConditionVariants]}."
        ),
    )
    parser.add_argument(
        "--music_type",
        type=str,
        nargs="+",
        default=[mt.value for mt in MusicTypeVariants],
        choices=[mt.value for mt in MusicTypeVariants],
        help="One or more music types to analyse. Defaults to all available types.",
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        default=False,
        help="When set, load raw files, resample, stack and save before analysis.",
    )
    parser.add_argument(
        "--isc_threshold",
        type=float,
        default=0.035,
        help="Broadband ISC significance threshold (default: 0.035).",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=5.0,
        help="Sliding-window length in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=2.5,
        help="Sliding-window step size in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help=(
            "Number of parallel jobs for data loading/resampling. "
            "Use -1 to use all available CPUs (default: 1)."
        ),
    )
    parser.add_argument(
        "--wavelet_freq_min",
        type=float,
        default=1.0,
        help="Minimum frequency (Hz) for wavelet analysis (default: 1.0).",
    )
    parser.add_argument(
        "--wavelet_freq_max",
        type=float,
        default=40.0,
        help="Maximum frequency (Hz) for wavelet analysis (default: 40.0).",
    )
    parser.add_argument(
        "--wavelet_n_freqs",
        type=int,
        default=20,
        help="Number of frequency steps for wavelet analysis (default: 20).",
    )
    parser.add_argument(
        "--wavelet_bands",
        type=str,
        nargs="+",
        default=None,
        choices=list(FREQUENCY_BANDS.keys()),
        help=(
            "Optional subset of EEG bands for wavelet per-band analysis. "
            "Defaults to all bands."
        ),
    )
    parser.add_argument(
        "--skip_wavelet_broadband",
        action="store_true",
        default=False,
        help=(
            "Skip the broadband wavelet stage and run only selected per-band "
            "wavelet analyses."
        ),
    )
    parser.add_argument(
        "--wavelet_data_dir",
        type=str,
        default=str(
            ProjectPaths.PROCESSED_DATA_DIR
            / ExperimentNames.PSILO_MUSIC.value
            / "wavelets"
        ),
        help=(
            "Directory for storing wavelet-transformed datasets for future use "
            "(default: data/processed/psilo_music/wavelets)."
        ),
    )
    parser.add_argument(
        "--reuse_wavelets",
        action="store_true",
        default=False,
        help="Reuse stored wavelets from --wavelet_data_dir when available.",
    )
    parser.add_argument(
        "--wavelet_cache_dir",
        type=str,
        dest="wavelet_data_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reuse_wavelet_cache",
        action="store_true",
        dest="reuse_wavelets",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wavelet_keep_frequency_dim",
        action="store_true",
        default=False,
        help=(
            "Keep frequency dimension in wavelet output "
            "(flattened as feature×frequency instead of averaging across "
            "frequencies)."
        ),
    )
    parser.add_argument(
        "--wavelet_reshape_frequency_dim",
        action="store_true",
        default=False,
        help=(
            "When --wavelet_keep_frequency_dim is enabled, reshape wavelet output "
            "to [n_individuals, n_channels, n_frequencies, n_times]."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────


def load_analyzers(
    music_types: Sequence[MusicTypeVariants],
    condition: ConditionVariants,
    exclusion_categories: Sequence[ExclusionCategories],
    process_and_save: bool,
    n_jobs: int = -1,
    normalize_data: bool = True,
) -> dict[str, EEGSummarizedAnalyzer]:
    """Load (or process & save) and normalise analysers for each music type.

    Returns a dict keyed by ``"{condition}_{music_type}"``
    (e.g. ``"Placebo_CLASSIC"``), matching the on-disk data file naming
    convention used by :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.save_data`.
    """
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzers: dict[str, EEGSummarizedAnalyzer] = {}
    for mt in music_types:
        label = f"{condition.value}_{mt.value}"
        analyzer = EEGSummarizedAnalyzer(
            experiment_name=ExperimentNames.PSILO_MUSIC,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[mt],
            conditions=[condition],
            exclusion_categories=list(exclusion_categories),
        )

        if process_and_save:
            analyzer.load_and_prepare_data(resample_freq=250.0, n_jobs=n_jobs)
            _logger.info(f"[{label}] data shape: {analyzer.data.shape}")
            analyzer.save_data()
        else:
            analyzer.load_data(
                info_filename=analyzer.filtered_df[SingleDataMetadata.FILENAME].iloc[0],
            )
            _logger.info(f"[{label}] Loaded data shape: {analyzer.data.shape}")

        if normalize_data:
            analyzer.normalize()
        analyzers[label] = analyzer

    return analyzers


def analyzers_to_datasets(analyzers: dict) -> dict[str, AnalysisData]:
    """Convert loaded analysers to AnalysisData instances."""
    datasets = {
        label: a.to_analysis_data(label=label) for label, a in analyzers.items()
    }
    for label, ad in datasets.items():
        _logger.info(f"[{label}] {ad}")
    return datasets


# ──────────────────────────────────────────────────────────────────────
# ISC workflow
# ──────────────────────────────────────────────────────────────────────


def run_isc_workflow(
    datasets: dict[str, AnalysisData],
    *,
    save_dir: Path,
    isc_threshold: float = 0.035,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
) -> None:
    """Run the full ISC analysis and save figures.

    Steps:
    1. Data overview
    2. Broadband LOO-ISC — distribution plot
    3. Broadband sliding-window ISC — time-resolved plot & significant intervals
    4. Per-band LOO-ISC — distribution & bar-chart plots
    5. Per-band sliding-window ISC — time-resolved plots & significant intervals
    6. Band-overlap analysis — raster plot
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"ISC figures will be saved to: {save_dir}")

    print_data_overview(datasets)
    _first_ad = next(iter(datasets.values()))

    # ── Broadband LOO-ISC ─────────────────────────────────────────
    _logger.info("=== Broadband LOO-ISC ===")
    loo_iscs = {}
    mean_loo_iscs = {}
    for label, ad in datasets.items():
        loo_isc, mean_loo_isc = compute_loo_isc(ad.data)
        loo_iscs[label] = loo_isc
        mean_loo_iscs[label] = mean_loo_isc
        _logger.info(
            f"[{label}]  loo_isc: {loo_isc.shape}   mean_loo_isc: {mean_loo_isc.shape}"
        )

    plot_loo_isc_distribution(
        mean_loo_iscs,
        ylabel=f"Number of {_first_ad.feature_axis_label.lower()}s",
        save_path=save_dir / "loo_isc_distribution.png",
    )

    # ── Broadband sliding-window ISC ──────────────────────────────
    _logger.info("=== Broadband Sliding-Window ISC ===")
    sw_results = {}
    for label, ad in datasets.items():
        sw_isc, sw_times = compute_sliding_window_isc(
            ad.data, window_sec=window_sec, step_sec=step_sec, sfreq=ad.sfreq
        )
        sw_results[label] = (sw_isc, sw_times)
        _logger.info(f"[{label}]  sw_isc: {sw_isc.shape}   sw_times: {sw_times.shape}")

    plot_sliding_window_isc(
        sw_results,
        isc_threshold=isc_threshold,
        feature_axis_label=f"{_first_ad.feature_axis_label} index",
        save_path=save_dir / "sliding_window_isc.png",
    )
    print_significant_intervals(sw_results, isc_threshold=isc_threshold)

    # ── Per-band LOO-ISC ──────────────────────────────────────────
    _logger.info("=== Per-Band LOO-ISC ===")
    band_iscs: dict = {}
    for label, ad in datasets.items():
        _logger.info(f"Computing band ISC for {label} …")
        band_iscs[label] = {}
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
            filtered = ad.filter_to_band(l_freq, h_freq)
            loo, mean_isc = compute_loo_isc(filtered.data)
            band_iscs[label][band] = (loo, mean_isc)
            _logger.info(
                f"  {band:6s}  loo_isc={loo.shape}  mean={mean_isc.mean():.4f}"
            )

    plot_band_isc_distributions(
        band_iscs,
        bands=FREQUENCY_BANDS,
        feature_axis_label=(f"Number of {_first_ad.feature_axis_label.lower()}s"),
        save_path=save_dir / "band_isc_distributions.png",
    )
    plot_band_mean_isc_bar(
        band_iscs,
        bands=FREQUENCY_BANDS,
        save_path=save_dir / "band_isc_mean_bar.png",
    )

    # ── Per-band sliding-window ISC ───────────────────────────────
    _logger.info("=== Per-Band Sliding-Window ISC ===")
    band_sw: dict = {}
    for label, ad in datasets.items():
        _logger.info(f"Computing band sliding-window ISC for {label} …")
        band_sw[label] = {}
        for band, (l_freq, h_freq) in FREQUENCY_BANDS.items():
            filtered = ad.filter_to_band(l_freq, h_freq)
            tc, times = compute_sliding_window_isc(
                filtered.data,
                window_sec=window_sec,
                step_sec=step_sec,
                sfreq=ad.sfreq,
            )
            band_sw[label][band] = (tc, times)
            _logger.info(f"  {band:6s}  isc_tc={tc.shape}  times={times.shape}")

    plot_band_sliding_window_isc(
        band_sw,
        bands=FREQUENCY_BANDS,
        isc_threshold=BAND_ISC_THRESHOLDS,
        feature_axis_label=_first_ad.feature_axis_label,
        save_path=save_dir / "band_sliding_window_isc.png",
    )
    print_band_significant_intervals(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        default_threshold=isc_threshold,
    )

    # ── Band overlap ──────────────────────────────────────────────
    _logger.info("=== Band Overlap ===")
    plot_band_overlap(
        band_sw,
        bands=FREQUENCY_BANDS,
        band_thresholds=BAND_ISC_THRESHOLDS,
        broadband_sw=sw_results,
        broadband_threshold=isc_threshold,
        save_path=save_dir / "band_overlap.png",
    )

    _logger.info("ISC analysis complete.")


# ──────────────────────────────────────────────────────────────────────
# Wavelet workflow helpers
# ──────────────────────────────────────────────────────────────────────

#: Frequency resolution (Hz) used when building per-band Morlet frequencies.
WAVELET_BAND_FREQ_RESOLUTION_HZ: float = 1.0
#: Alias retained for backward compatibility with older call-sites.
_WAVELET_BAND_FREQ_RESOLUTION_HZ: float = WAVELET_BAND_FREQ_RESOLUTION_HZ


def wavelet_transform(
    datasets: dict[str, AnalysisData],
    freqs: np.ndarray,
    representation: str = "power",
    *,
    keep_frequency_dim: bool = False,
    reshape_frequency_dim: bool = False,
    wavelet_dir: Path | None = None,
    reuse_wavelets: bool = False,
) -> dict[str, AnalysisData]:
    """Apply a wavelet *representation* to every dataset.

    :param datasets: Source :class:`AnalysisData` objects keyed by label.
    :param freqs: Morlet frequencies (Hz).
    :param representation: Wavelet representation (``"power"`` or ``"phase"``).
    :param keep_frequency_dim: Keep frequency dimension instead of averaging it.
    :param reshape_frequency_dim: When ``keep_frequency_dim`` is true, reshape
        flattened ``(feature×frequency)`` output into
        ``(n_items, n_features, n_freqs, n_samples)``.
    :param wavelet_dir: Optional directory for persisted wavelet outputs.
    :param reuse_wavelets: If true, reuse stored wavelet files when present.
    :returns: New dict of transformed datasets with the same keys.
    """
    if representation not in ("power", "phase"):
        raise ValueError(
            f"Only 'power' and 'phase' wavelet representations are supported, got {representation!r}"
        )
    if reshape_frequency_dim and not keep_frequency_dim:
        raise ValueError("reshape_frequency_dim=True requires keep_frequency_dim=True.")

    if wavelet_dir is not None:
        wavelet_dir.mkdir(parents=True, exist_ok=True)

    transformed: dict[str, AnalysisData] = {}
    freq_sig = f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{len(freqs)}"

    def _reduce_frequency_dimension(
        wavelet_ad: AnalysisData,
        *,
        representation_kind: str,
        base_feature_names: list[str] | None,
    ) -> AnalysisData:
        """Reduce a frequency-resolved wavelet AnalysisData along frequency axis."""
        n_freqs = int(np.asarray(wavelet_ad.metadata["freqs"]).shape[0])
        n_items, n_features_flat, n_samples = wavelet_ad.data.shape
        if n_features_flat % n_freqs != 0:
            raise ValueError(
                "Frequency-resolved wavelet data has invalid shape for "
                f"reduction: n_features={n_features_flat}, n_freqs={n_freqs}"
            )

        n_features = n_features_flat // n_freqs
        reshaped = wavelet_ad.data.reshape(n_items, n_features, n_freqs, n_samples)
        if representation_kind == "power":
            reduced = reshaped.mean(axis=2)
        else:
            reduced = circmean(reshaped, high=np.pi, low=-np.pi, axis=2)

        return AnalysisData(
            data=reduced,
            sfreq=wavelet_ad.sfreq,
            representation=wavelet_ad.representation,
            label=wavelet_ad.label,
            feature_names=base_feature_names,
            info=wavelet_ad.info,
            metadata={**wavelet_ad.metadata, "keep_frequency_dim": False},
        )

    def _reshape_frequency_dimension(
        wavelet_ad: AnalysisData,
        *,
        base_feature_names: list[str] | None,
    ) -> AnalysisData:
        """Reshape flattened frequency-resolved wavelet output to 4D."""
        n_freqs = len(wavelet_ad.metadata["freqs"])
        n_items, n_features_flat, n_samples = wavelet_ad.data.shape
        if n_features_flat % n_freqs != 0:
            raise ValueError(
                "Frequency-resolved wavelet data has invalid shape for "
                f"reshape: n_features={n_features_flat}, n_freqs={n_freqs}"
            )

        n_features = n_features_flat // n_freqs
        reshaped = wavelet_ad.data.reshape(n_items, n_features, n_freqs, n_samples)
        return AnalysisData(
            data=reshaped,
            sfreq=wavelet_ad.sfreq,
            representation=wavelet_ad.representation,
            label=wavelet_ad.label,
            feature_names=base_feature_names,
            info=wavelet_ad.info,
            metadata={**wavelet_ad.metadata, "keep_frequency_dim": True},
        )

    for label, ad in datasets.items():
        wavelet_file: Path | None = None
        legacy_wavelet_file: Path | None = None
        if wavelet_dir is not None:
            safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
            if not safe_label:
                safe_label = f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"
            wavelet_file = wavelet_dir / (
                f"{safe_label}__wavelet_{representation}__{freq_sig}__freqdim1.npz"
            )
            legacy_wavelet_file = wavelet_dir / (
                f"{safe_label}__wavelet_{representation}__"
                f"{freq_sig}__freqdim{int(keep_frequency_dim)}.npz"
            )
            if reuse_wavelets and wavelet_file.exists():
                loaded = np.load(wavelet_file)
                has_feature_names = loaded["has_feature_names"].item()
                loaded_feature_names = loaded["feature_names"]
                feature_names = (
                    loaded_feature_names.tolist()
                    if has_feature_names and loaded_feature_names.size > 0
                    else None
                )
                transformed[label] = AnalysisData(
                    data=loaded["data"],
                    sfreq=float(loaded["sfreq"]),
                    representation=(
                        DataRepresentation.WAVELET_PHASE
                        if representation == "phase"
                        else DataRepresentation.WAVELET_POWER
                    ),
                    label=str(loaded["label"]),
                    feature_names=feature_names,
                    info=ad.info,
                    metadata={
                        **ad.metadata,
                        "freqs": loaded["freqs"],
                        "n_cycles": loaded["n_cycles"],
                        "keep_frequency_dim": bool(loaded["keep_frequency_dim"]),
                        "loaded_from_wavelet_file": str(wavelet_file),
                    },
                )
                if not keep_frequency_dim:
                    transformed[label] = _reduce_frequency_dimension(
                        transformed[label],
                        representation_kind=representation,
                        base_feature_names=ad.feature_names,
                    )
                elif reshape_frequency_dim:
                    transformed[label] = _reshape_frequency_dimension(
                        transformed[label],
                        base_feature_names=ad.feature_names,
                    )
                continue

            if (
                reuse_wavelets
                and legacy_wavelet_file is not None
                and legacy_wavelet_file.exists()
            ):
                loaded = np.load(legacy_wavelet_file)
                has_feature_names = loaded["has_feature_names"].item()
                loaded_feature_names = loaded["feature_names"]
                feature_names = (
                    loaded_feature_names.tolist()
                    if has_feature_names and loaded_feature_names.size > 0
                    else None
                )
                transformed[label] = AnalysisData(
                    data=loaded["data"],
                    sfreq=float(loaded["sfreq"]),
                    representation=(
                        DataRepresentation.WAVELET_PHASE
                        if representation == "phase"
                        else DataRepresentation.WAVELET_POWER
                    ),
                    label=str(loaded["label"]),
                    feature_names=feature_names,
                    info=ad.info,
                    metadata={
                        **ad.metadata,
                        "freqs": loaded["freqs"],
                        "n_cycles": loaded["n_cycles"],
                        "keep_frequency_dim": bool(loaded["keep_frequency_dim"]),
                        "loaded_from_wavelet_file": str(legacy_wavelet_file),
                    },
                )
                if not keep_frequency_dim:
                    transformed[label] = _reduce_frequency_dimension(
                        transformed[label],
                        representation_kind=representation,
                        base_feature_names=ad.feature_names,
                    )
                elif reshape_frequency_dim:
                    transformed[label] = _reshape_frequency_dimension(
                        transformed[label],
                        base_feature_names=ad.feature_names,
                    )
                continue

        if representation == "power":
            wd_freq = to_wavelet_power(ad, freqs, keep_frequency_dim=True)
        else:
            wd_freq = to_wavelet_phase(ad, freqs, keep_frequency_dim=True)
        if keep_frequency_dim and reshape_frequency_dim:
            transformed[label] = _reshape_frequency_dimension(
                wd_freq,
                base_feature_names=ad.feature_names,
            )
        elif keep_frequency_dim:
            transformed[label] = wd_freq
        else:
            transformed[label] = _reduce_frequency_dimension(
                wd_freq,
                representation_kind=representation,
                base_feature_names=ad.feature_names,
            )
        if wavelet_file is not None:
            feature_names = (
                wd_freq.feature_names if wd_freq.feature_names is not None else []
            )
            np.savez_compressed(
                wavelet_file,
                data=wd_freq.data,
                sfreq=wd_freq.sfreq,
                label=wd_freq.label,
                feature_names=np.asarray(feature_names, dtype=str),
                has_feature_names=np.asarray(
                    int(wd_freq.feature_names is not None), dtype=np.int8
                ),
                freqs=freqs,
                n_cycles=np.asarray(wd_freq.metadata.get("n_cycles")),
                keep_frequency_dim=np.asarray(1, dtype=np.int8),
            )
    return transformed


#: Alias retained for backward compatibility with older call-sites.
_wavelet_transform = wavelet_transform


def _wavelet_isc_for_label(
    wd: AnalysisData,
    label: str,
    *,
    loo_save_path: Path,
    sw_save_path: Path,
    isc_threshold: float,
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute & plot LOO-ISC and sliding-window ISC for a single label.

    Unlike :func:`run_isc_workflow`, this helper does **not** attempt
    per-band splitting via :meth:`~AnalysisData.filter_to_band`, which
    would be semantically incorrect on frequency-decomposed data.

    :returns: ``(loo_isc, mean_loo_isc, sw_isc, sw_times)``
    """
    loo_save_path.parent.mkdir(parents=True, exist_ok=True)
    sw_save_path.parent.mkdir(parents=True, exist_ok=True)

    loo, mean_loo = compute_loo_isc(wd.data)
    _logger.info(f"[{label}]  loo_isc={loo.shape}  mean_loo_isc={mean_loo.shape}")
    plot_loo_isc_distribution(
        {label: mean_loo},
        ylabel=f"Number of {wd.feature_axis_label.lower()}s",
        save_path=loo_save_path,
    )

    sw_isc, sw_times = compute_sliding_window_isc(
        wd.data, window_sec=window_sec, step_sec=step_sec, sfreq=wd.sfreq
    )
    _logger.info(f"[{label}]  sw_isc={sw_isc.shape}  sw_times={sw_times.shape}")
    plot_sliding_window_isc(
        {label: (sw_isc, sw_times)},
        isc_threshold=isc_threshold,
        feature_axis_label=f"{wd.feature_axis_label} index",
        save_path=sw_save_path,
    )
    print_significant_intervals(
        {label: (sw_isc, sw_times)}, isc_threshold=isc_threshold
    )

    return loo, mean_loo, sw_isc, sw_times


# ──────────────────────────────────────────────────────────────────────
# Wavelet workflow
# ──────────────────────────────────────────────────────────────────────


def _broadband_wavelet_4d(
    ad: AnalysisData,
    label: str,
    *,
    representation: str,
    freqs: np.ndarray,
    wavelet_dir: Path | None,
    reuse_wavelets: bool,
) -> AnalysisData:
    """Return a 4D wavelet ``AnalysisData`` for ``ad``.

    Forces ``keep_frequency_dim=True`` and ``reshape_frequency_dim=True`` so
    callers can index the result as ``(n_subjects, n_channels, n_freqs, n_times)``
    for the notebook-parity broadband plots.
    """
    transformed = wavelet_transform(
        {label: ad},
        freqs,
        representation,
        keep_frequency_dim=True,
        reshape_frequency_dim=True,
        wavelet_dir=wavelet_dir,
        reuse_wavelets=reuse_wavelets,
    )
    return transformed[label]


def _wavelet_4d_to_3d(
    wd_4d: AnalysisData,
    *,
    representation: str,
    base_feature_names: list[str] | None,
) -> AnalysisData:
    """Reduce a 4D wavelet ``AnalysisData`` to 3D for ISC computation.

    Power → arithmetic mean across the frequency axis.
    Phase → circular mean across the frequency axis.
    """
    data_4d = wd_4d.data
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D wavelet data, got shape {data_4d.shape}")
    if representation == "power":
        reduced = data_4d.mean(axis=2)
    else:
        reduced = circmean(data_4d, high=np.pi, low=-np.pi, axis=2)
    return AnalysisData(
        data=reduced,
        sfreq=wd_4d.sfreq,
        representation=wd_4d.representation,
        label=wd_4d.label,
        feature_names=base_feature_names,
        info=wd_4d.info,
        metadata={**wd_4d.metadata, "keep_frequency_dim": False},
    )


def _try_load_phase_band_iscs(
    ad: AnalysisData,
    label: str,
    *,
    wavelet_dir: Path | None,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Load cached per-band phase wavelets and compute LOO-ISC(cos φ).

    Used by the wavelet-power workflow to render the optional power vs.
    phase joint plot. Returns ``{}`` (and logs a warning) when no per-band
    phase cache exists.
    """
    if wavelet_dir is None:
        return {}
    out: dict[str, np.ndarray] = {}
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
    if not safe_label:
        safe_label = f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    for band, (lo, hi) in bands.items():
        band_dir = wavelet_dir / f"band_{band}"
        n_freqs_band = max(
            2,
            int(round((hi - lo) / _WAVELET_BAND_FREQ_RESOLUTION_HZ)) + 1,
        )
        band_freqs = np.linspace(lo, hi, n_freqs_band)
        freq_sig = f"{band_freqs[0]:.3f}_{band_freqs[-1]:.3f}_{len(band_freqs)}"
        cache_file = band_dir / f"{safe_label}__wavelet_phase__{freq_sig}__freqdim1.npz"
        if not cache_file.exists():
            _logger.info(
                f"[{label}] No phase wavelet cache for band {band!r} "
                f"(expected {cache_file.name}); "
                "skipping power_phase_joint contribution for this band."
            )
            continue
        try:
            phase_band = wavelet_transform(
                {label: ad},
                band_freqs,
                representation="phase",
                keep_frequency_dim=False,
                reshape_frequency_dim=False,
                wavelet_dir=band_dir,
                reuse_wavelets=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            _logger.warning(
                f"[{label}] Failed to load phase cache for band {band!r}: {exc}"
            )
            continue
        cos_phase = np.cos(phase_band[label].data)
        _, mean_loo = compute_loo_isc(cos_phase)
        out[band] = mean_loo
    return out


def _run_wavelet_workflow_for_label(
    ad: AnalysisData,
    label: str,
    *,
    representation: str,
    freqs: np.ndarray,
    save_dir: Path,
    selected_bands: dict[str, tuple[float, float]],
    selected_band_thresholds: dict[str, float],
    include_broadband: bool,
    wavelet_dir: Path | None,
    reuse_wavelets: bool,
    isc_threshold: float,
    window_sec: float,
    step_sec: float,
    info,
    cross_representation_wavelet_dir: Path | None,
) -> None:
    """Process a single ``(condition, music_type)`` label.

    Writes the canonical layout::

        <save_dir>/
            broadband/<analysis_type>/<filename>_<label>.png
            bands/<analysis_type>/<band>_<filename>_<label>.png
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"=== Processing {label} → {save_dir} (rep={representation}) ===")

    bb_dir = save_dir / "broadband"
    bands_dir = save_dir / "bands"

    # ── Broadband ────────────────────────────────────────────────
    bb_sw_for_overlap: tuple[np.ndarray, np.ndarray] | None = None
    band_mean_iscs_from_bb: dict[str, np.ndarray] = {}

    if include_broadband:
        wd_4d = _broadband_wavelet_4d(
            ad,
            label,
            representation=representation,
            freqs=freqs,
            wavelet_dir=(wavelet_dir / "broadband") if wavelet_dir else None,
            reuse_wavelets=reuse_wavelets,
        )
        bb_4d_data = wd_4d.data  # (n_subj, n_ch, n_freqs, n_times)
        _logger.info(f"[{label}] broadband 4D shape: {bb_4d_data.shape}")

        # Reduce to 3D for the existing ISC distribution / sliding-window plots.
        wd_3d = _wavelet_4d_to_3d(
            wd_4d,
            representation=representation,
            base_feature_names=ad.feature_names,
        )

        loo_3d, mean_loo_3d, sw_isc_3d, sw_times_3d = _wavelet_isc_for_label(
            wd_3d,
            label,
            loo_save_path=(bb_dir / "loo_isc" / f"loo_isc_distribution_{label}.png"),
            sw_save_path=(
                bb_dir / "sliding_window" / f"sliding_window_isc_{label}.png"
            ),
            isc_threshold=isc_threshold,
            window_sec=window_sec,
            step_sec=step_sec,
        )
        bb_sw_for_overlap = (sw_isc_3d, sw_times_3d)

        # ── Notebook-parity broadband plots ──
        if representation == "power":
            plot_spectral_profile(
                bb_4d_data,
                freqs,
                label=label,
                bands=selected_bands,
                save_path=bb_dir / "spectral_profile" / f"spectral_profile_{label}.png",
            )
            plot_tf_map(
                bb_4d_data,
                freqs,
                wd_4d.sfreq,
                label=label,
                save_path=bb_dir / "tf_map" / f"tf_map_{label}.png",
            )
            plot_band_power_timecourse(
                bb_4d_data,
                freqs,
                wd_4d.sfreq,
                label=label,
                bands=selected_bands,
                save_path=(bb_dir / "band_power_tc" / f"band_power_tc_{label}.png"),
            )
            plot_intersubject_variance(
                bb_4d_data,
                freqs,
                wd_4d.sfreq,
                label=label,
                bands=selected_bands,
                save_path=(
                    bb_dir
                    / "intersubject_variance"
                    / f"intersubject_variance_{label}.png"
                ),
            )

            # Per-band LOO-ISC computed from broadband 4D, used by bar /
            # distribution / topomap / power-phase-joint plots.
            for band, (lo, hi) in selected_bands.items():
                band_mask = (freqs >= lo) & (freqs <= hi)
                if not band_mask.any():
                    continue
                band_power_3d = bb_4d_data[:, :, band_mask, :].mean(axis=2)
                _, mean_loo_band = compute_loo_isc(band_power_3d)
                band_mean_iscs_from_bb[band] = mean_loo_band
                _logger.info(
                    f"[{label}] {band:6s} broadband-derived "
                    f"mean LOO-ISC = {mean_loo_band.mean():.4f}"
                )

            plot_wavelet_loo_isc_bar(
                band_mean_iscs_from_bb,
                label=label,
                save_path=(bb_dir / "wavelet_loo_isc" / f"wavelet_loo_isc_{label}.png"),
            )
            plot_wavelet_loo_isc_distributions(
                band_mean_iscs_from_bb,
                label=label,
                save_path=(
                    bb_dir
                    / "wavelet_loo_isc"
                    / f"wavelet_loo_isc_distributions_{label}.png"
                ),
            )
            plot_wavelet_topomap_isc(
                band_mean_iscs_from_bb,
                info,
                label=label,
                save_path=bb_dir / "topomap_isc" / f"topomap_isc_{label}.png",
            )
            plot_tf_isc(
                bb_4d_data,
                freqs,
                label=label,
                save_path=bb_dir / "tf_isc" / f"tf_isc_{label}.png",
            )
            plot_cross_frequency_coupling(
                bb_4d_data,
                freqs,
                label=label,
                bands=selected_bands,
                save_path=(
                    bb_dir / "cross_freq_coupling" / f"cross_freq_coupling_{label}.png"
                ),
            )

            # Power vs Phase joint plot — only when phase cache is present.
            phase_band_iscs = _try_load_phase_band_iscs(
                ad,
                label,
                wavelet_dir=cross_representation_wavelet_dir,
                bands=selected_bands,
            )
            if phase_band_iscs:
                try:
                    plot_power_phase_joint(
                        band_mean_iscs_from_bb,
                        phase_band_iscs,
                        label=label,
                        save_path=(
                            bb_dir
                            / "power_phase_joint"
                            / f"power_phase_isc_{label}.png"
                        ),
                    )
                except ValueError as exc:
                    _logger.warning(f"[{label}] Skipping power_phase_joint plot: {exc}")

        else:  # representation == "phase"
            plot_phase_distribution(
                bb_4d_data,
                label=label,
                save_path=(
                    bb_dir / "phase_distribution" / f"phase_distribution_{label}.png"
                ),
            )
            itpc_full = compute_itpc(bb_4d_data)
            plot_itpc_spectrum(
                itpc_full,
                freqs,
                label=label,
                bands=selected_bands,
                save_path=bb_dir / "itpc_spectrum" / f"itpc_spectrum_{label}.png",
            )
            plot_tf_itpc_map(
                itpc_full,
                freqs,
                wd_4d.sfreq,
                label=label,
                save_path=bb_dir / "tf_itpc_map" / f"tf_itpc_map_{label}.png",
            )
            plot_band_itpc_timecourse(
                itpc_full,
                freqs,
                wd_4d.sfreq,
                label=label,
                bands=selected_bands,
                save_path=bb_dir / "band_itpc_tc" / f"band_itpc_tc_{label}.png",
            )

            # Phase LOO-ISC per band: cos(circmean(phase)) → compute_loo_isc.
            band_mean_iscs_from_bb = compute_phase_band_loo_iscs(
                bb_4d_data, freqs, bands=selected_bands
            )
            plot_wavelet_loo_isc_bar(
                band_mean_iscs_from_bb,
                label=label,
                color="darkgreen",
                save_path=(bb_dir / "phase_loo_isc" / f"phase_loo_isc_{label}.png"),
            )
            plot_wavelet_loo_isc_distributions(
                band_mean_iscs_from_bb,
                label=label,
                color="darkgreen",
                save_path=(
                    bb_dir
                    / "phase_loo_isc"
                    / f"phase_loo_isc_distributions_{label}.png"
                ),
            )
            plot_wavelet_topomap_isc(
                band_mean_iscs_from_bb,
                info,
                label=label,
                title_prefix="Topographic phase LOO-ISC(cos φ) per band",
                save_path=(
                    bb_dir / "topomap_phase_isc" / f"topomap_phase_isc_{label}.png"
                ),
            )
            plot_itpc_vs_isc(
                itpc_full,
                band_mean_iscs_from_bb,
                freqs,
                label=label,
                bands=selected_bands,
                save_path=bb_dir / "itpc_vs_isc" / f"itpc_vs_isc_{label}.png",
            )

        del wd_4d, wd_3d, bb_4d_data

    # ── Per-band wavelet ISC ─────────────────────────────────────
    band_loo_iscs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    band_sw_iscs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_axis_label: str | None = None

    for band, (l_freq, h_freq) in selected_bands.items():
        n_freqs_band = max(
            2,
            int(round((h_freq - l_freq) / _WAVELET_BAND_FREQ_RESOLUTION_HZ)) + 1,
        )
        _logger.info(
            f"  [{label}] band {band} ({l_freq:.1f}–{h_freq:.1f} Hz, "
            f"{n_freqs_band} steps)"
        )
        band_freqs = np.linspace(l_freq, h_freq, n_freqs_band)
        band_ds = wavelet_transform(
            {label: ad},
            band_freqs,
            representation,
            keep_frequency_dim=False,
            reshape_frequency_dim=False,
            wavelet_dir=(wavelet_dir / f"band_{band}") if wavelet_dir else None,
            reuse_wavelets=reuse_wavelets,
        )
        wd = band_ds[label]
        if feature_axis_label is None:
            feature_axis_label = wd.feature_axis_label

        loo, mean_loo, sw_isc, sw_times = _wavelet_isc_for_label(
            wd,
            label,
            loo_save_path=(
                bands_dir / "loo_isc" / f"{band}_loo_isc_distribution_{label}.png"
            ),
            sw_save_path=(
                bands_dir / "sliding_window" / f"{band}_sliding_window_isc_{label}.png"
            ),
            isc_threshold=selected_band_thresholds[band],
            window_sec=window_sec,
            step_sec=step_sec,
        )
        band_loo_iscs[band] = (loo, mean_loo)
        band_sw_iscs[band] = (sw_isc, sw_times)
        del band_ds, wd

    if not band_loo_iscs:
        raise ValueError(
            f"[{label}] No valid bands were processed; ensure at least one "
            "band from FREQUENCY_BANDS is selected."
        )

    # ── Per-label cross-band aggregate plots (live alongside per-band ones) ─
    feature_axis_label = feature_axis_label or "Channel"
    plot_band_isc_distributions(
        {label: band_loo_iscs},
        bands=selected_bands,
        feature_axis_label=f"Number of {feature_axis_label.lower()}s",
        save_path=(bands_dir / "loo_isc" / f"band_isc_distributions_{label}.png"),
    )
    plot_band_mean_isc_bar(
        {label: band_loo_iscs},
        bands=selected_bands,
        save_path=bands_dir / "loo_isc" / f"band_isc_mean_bar_{label}.png",
    )
    plot_band_sliding_window_isc(
        {label: band_sw_iscs},
        bands=selected_bands,
        isc_threshold=selected_band_thresholds,
        feature_axis_label=feature_axis_label,
        save_path=(
            bands_dir / "sliding_window" / f"band_sliding_window_isc_{label}.png"
        ),
    )
    print_band_significant_intervals(
        {label: band_sw_iscs},
        bands=selected_bands,
        band_thresholds=selected_band_thresholds,
        default_threshold=isc_threshold,
    )
    if bb_sw_for_overlap is not None:
        plot_band_overlap(
            {label: band_sw_iscs},
            bands=selected_bands,
            band_thresholds=selected_band_thresholds,
            broadband_sw={label: bb_sw_for_overlap},
            broadband_threshold=isc_threshold,
            save_path=(bands_dir / "band_overlap" / f"band_overlap_{label}.png"),
        )


def run_wavelet_workflow(
    datasets: dict[str, AnalysisData],
    analyzers: dict[str, "EEGSummarizedAnalyzer"],
    *,
    representation: str = "power",
    freqs: np.ndarray,
    save_dir: Path,
    bands: Sequence[str] | None = None,
    include_broadband: bool = True,
    wavelet_dir: Path | None = None,
    reuse_wavelets: bool = False,
    keep_frequency_dim: bool = False,
    reshape_frequency_dim: bool = False,
    isc_threshold: float = 0.035,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    cross_representation_wavelet_dir: Path | None = None,
) -> None:
    """Run wavelet-domain ISC analyses and notebook-parity plots.

    Drives one workflow per ``(condition, music_type)`` label in *datasets*,
    writing all outputs under the canonical layout::

        <save_dir>/<label>/broadband/<analysis_type>/<filename>_<label>.png
        <save_dir>/<label>/bands/<analysis_type>/<band>_<filename>_<label>.png

    The broadband stage reproduces the figures from the exploratory wavelet
    notebooks (spectral profile, time–frequency map, per-band power time
    course, intersubject variance, wavelet LOO-ISC bar / distribution /
    topomap, time–frequency ISC, cross-frequency coupling, and — for the
    power workflow — an optional power vs. phase joint comparison).

    The per-band stage computes Morlet-wavelet-restricted LOO-ISC and
    sliding-window ISC for each canonical band; bandpass filtering of
    wavelet data is explicitly avoided as it would be semantically incorrect
    on frequency-decomposed data.

    :param datasets: Time-domain :class:`AnalysisData` objects keyed by label
        (e.g. ``"Placebo_CLASSIC"``).
    :param analyzers: Loaded :class:`EEGSummarizedAnalyzer` instances keyed
        by the same label as *datasets*. Used only for the topomap MNE
        ``Info`` object — pass an empty dict to skip topomaps.
    :param representation: Wavelet representation; ``"power"`` or ``"phase"``.
    :param freqs: Morlet frequencies for the broadband analysis (Hz).
    :param save_dir: Root output directory. Per-label subdirectories are
        created automatically.
    :param bands: Optional subset of band names to run in per-band stage.
        Defaults to all keys from :data:`FREQUENCY_BANDS`.
    :param include_broadband: Whether to run the broadband stage.
    :param wavelet_dir: Optional directory holding persisted wavelet caches.
    :param reuse_wavelets: Reuse cached wavelets from *wavelet_dir* when
        available.
    :param keep_frequency_dim: Reserved for backwards compatibility — the
        broadband stage always uses 4D wavelet tensors internally; per-band
        ISC always reduces to 3D.
    :param reshape_frequency_dim: Must be ``False`` — the per-band ISC stage
        requires 3D arrays.
    :param isc_threshold: Broadband ISC significance threshold.
    :param window_sec: Sliding-window length in seconds.
    :param step_sec: Sliding-window step size in seconds.
    :param cross_representation_wavelet_dir: For the wavelet-power workflow,
        the directory of cached *phase* wavelets used by the optional power
        vs. phase joint plot. Skipped silently when missing.
    :raises ValueError: If *representation* is not ``"power"`` or ``"phase"``,
        or *reshape_frequency_dim* is ``True``.
    """
    if representation not in ("power", "phase"):
        raise ValueError(
            f"representation must be 'power' or 'phase', got {representation!r}"
        )

    if reshape_frequency_dim:
        raise ValueError(
            "reshape_frequency_dim=True is not supported in run_wavelet_workflow. "
            "The ISC analysis functions require 3-D (n_items, n_features, n_samples) "
            "arrays and will break with 4-D wavelet tensors. "
            "Call wavelet_transform directly if you need a 4-D output."
        )

    # The keep_frequency_dim flag is preserved for CLI compatibility but is
    # not honoured here: the broadband stage internally toggles between 4D
    # (for notebook-parity plots) and 3D (for ISC) as needed, and the
    # per-band stage always operates on 3D data. Warn loudly if a caller
    # tries to override this.
    if keep_frequency_dim:
        _logger.info(
            "keep_frequency_dim=True is ignored by run_wavelet_workflow; "
            "broadband plots use 4D internally and per-band ISC uses 3D."
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Wavelet {representation} figures root: {save_dir}")

    selected_band_names = list(FREQUENCY_BANDS.keys()) if bands is None else list(bands)
    selected_bands = {band: FREQUENCY_BANDS[band] for band in selected_band_names}
    selected_band_thresholds = {
        band: BAND_ISC_THRESHOLDS[band] for band in selected_band_names
    }

    for label, ad in datasets.items():
        analyzer = analyzers.get(label) if analyzers else None
        info = getattr(analyzer, "info", None) if analyzer is not None else None
        label_save_dir = save_dir / label

        _run_wavelet_workflow_for_label(
            ad,
            label,
            representation=representation,
            freqs=freqs,
            save_dir=label_save_dir,
            selected_bands=selected_bands,
            selected_band_thresholds=selected_band_thresholds,
            include_broadband=include_broadband,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=reuse_wavelets,
            isc_threshold=isc_threshold,
            window_sec=window_sec,
            step_sec=step_sec,
            info=info,
            cross_representation_wavelet_dir=cross_representation_wavelet_dir,
        )

    _logger.info(f"Wavelet {representation} analysis complete.")
