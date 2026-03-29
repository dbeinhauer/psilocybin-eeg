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

    Returns a dict keyed by the music-type *value* (e.g. ``"CLASSIC"``).
    """
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzers: dict[str, EEGSummarizedAnalyzer] = {}
    for mt in music_types:
        label = mt.value
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
_WAVELET_BAND_FREQ_RESOLUTION_HZ: float = 1.0


def _wavelet_transform(
    datasets: dict[str, AnalysisData],
    freqs: np.ndarray,
    representation: str = "power",
    *,
    keep_frequency_dim: bool = False,
    wavelet_dir: Path | None = None,
    reuse_wavelets: bool = False,
) -> dict[str, AnalysisData]:
    """Apply a wavelet *representation* to every dataset.

    :param datasets: Source :class:`AnalysisData` objects keyed by label.
    :param freqs: Morlet frequencies (Hz).
    :param representation: Wavelet representation (``"power"`` or ``"phase"``).
    :param keep_frequency_dim: Keep frequency dimension instead of averaging it.
    :param wavelet_dir: Optional directory for persisted wavelet outputs.
    :param reuse_wavelets: If true, reuse stored wavelet files when present.
    :returns: New dict of transformed datasets with the same keys.
    """
    if representation not in ("power", "phase"):
        raise ValueError(
            f"Only 'power' and 'phase' wavelet representations are supported, got {representation!r}"
        )

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
                continue

        if representation == "power":
            wd_freq = to_wavelet_power(ad, freqs, keep_frequency_dim=True)
        else:
            wd_freq = to_wavelet_phase(ad, freqs, keep_frequency_dim=True)
        transformed[label] = (
            wd_freq
            if keep_frequency_dim
            else _reduce_frequency_dimension(
                wd_freq,
                representation_kind=representation,
                base_feature_names=ad.feature_names,
            )
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


def _run_wavelet_isc(
    wavelet_ds: dict[str, AnalysisData],
    save_dir: Path,
    *,
    isc_threshold: float,
    window_sec: float,
    step_sec: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    """LOO-ISC and sliding-window ISC on wavelet-transformed data.

    Unlike :func:`run_isc_workflow`, this function does **not** attempt
    per-band splitting via :meth:`~AnalysisData.filter_to_band`, which
    would be semantically incorrect on frequency-decomposed data.

    :returns: ``(loo_iscs, mean_loo_iscs, sw_results)`` — raw result dicts
        keyed by label, suitable for building aggregated band-comparison
        plots in :func:`run_wavelet_workflow`.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    _first_wd = next(iter(wavelet_ds.values()))

    # ── LOO-ISC ───────────────────────────────────────────────────
    loo_iscs: dict[str, np.ndarray] = {}
    mean_loo_iscs: dict[str, np.ndarray] = {}
    for label, wd in wavelet_ds.items():
        loo, mean_loo = compute_loo_isc(wd.data)
        loo_iscs[label] = loo
        mean_loo_iscs[label] = mean_loo
        _logger.info(f"[{label}]  loo_isc={loo.shape}  mean_loo_isc={mean_loo.shape}")

    plot_loo_isc_distribution(
        mean_loo_iscs,
        ylabel=f"Number of {_first_wd.feature_axis_label.lower()}s",
        save_path=save_dir / "loo_isc_distribution.png",
    )

    # ── Sliding-window ISC ────────────────────────────────────────
    sw_results: dict = {}
    for label, wd in wavelet_ds.items():
        sw_isc, sw_times = compute_sliding_window_isc(
            wd.data, window_sec=window_sec, step_sec=step_sec, sfreq=wd.sfreq
        )
        sw_results[label] = (sw_isc, sw_times)
        _logger.info(f"[{label}]  sw_isc={sw_isc.shape}  sw_times={sw_times.shape}")

    plot_sliding_window_isc(
        sw_results,
        isc_threshold=isc_threshold,
        feature_axis_label=f"{_first_wd.feature_axis_label} index",
        save_path=save_dir / "sliding_window_isc.png",
    )
    print_significant_intervals(sw_results, isc_threshold=isc_threshold)

    return loo_iscs, mean_loo_iscs, sw_results


# ──────────────────────────────────────────────────────────────────────
# Wavelet workflow
# ──────────────────────────────────────────────────────────────────────


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
    isc_threshold: float = 0.035,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
) -> None:
    """Run ISC and mean/variance analyses on wavelet-transformed data.

    Runs two stages:

    1. **Broadband** — transform with the full *freqs* range; saves plots
       under ``<save_dir>/broadband/``.
    2. **Per-band** — transform each EEG band independently using
       :data:`FREQUENCY_BANDS` (1 Hz frequency resolution); saves per-band
       plots under ``<save_dir>/band_<name>/`` and aggregated cross-band
       comparison plots under ``<save_dir>/band_comparison/``.

    The per-band ISC and mean/variance use Morlet-wavelet-transformed data
    restricted to each band's own frequency range; bandpass filtering of the
    wavelet data is explicitly avoided as it would be semantically incorrect
    on frequency-decomposed data.

    :param datasets: Time-domain :class:`AnalysisData` objects keyed by label.
    :param analyzers: Reserved for future extension.  Not used at present.
    :param representation: Wavelet representation; ``"power"`` or ``"phase"``.
    :param freqs: Morlet frequencies for the broadband analysis (Hz).
    :param save_dir: Root directory in which to save plots.
    :param bands: Optional subset of band names to run in per-band stage.
        Defaults to all keys from :data:`FREQUENCY_BANDS`.
    :param include_broadband: Whether to run the broadband wavelet stage.
    :param wavelet_dir: Optional directory for persisted wavelet transforms.
    :param reuse_wavelets: Reuse stored wavelets from *wavelet_dir* when available.
    :param keep_frequency_dim: Keep frequency dimension in wavelet outputs.
    :param isc_threshold: ISC significance threshold.
    :param window_sec: Sliding-window length in seconds.
    :param step_sec: Sliding-window step size in seconds.
    :raises ValueError: If *representation* is not ``"power"`` or ``"phase"``.
    """
    if representation not in ("power", "phase"):
        raise ValueError(
            f"representation must be 'power' or 'phase', got {representation!r}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Wavelet {representation} figures will be saved to: {save_dir}")

    # Shared keyword dicts to avoid repeating the same kwargs everywhere.
    _isc_kw: dict = dict(
        isc_threshold=isc_threshold, window_sec=window_sec, step_sec=step_sec
    )

    selected_band_names = list(FREQUENCY_BANDS.keys()) if bands is None else list(bands)
    selected_bands = {band: FREQUENCY_BANDS[band] for band in selected_band_names}
    selected_band_thresholds = {
        band: BAND_ISC_THRESHOLDS[band] for band in selected_band_names
    }

    # ── Broadband wavelet analysis ────────────────────────────────
    bb_sw_results: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    if include_broadband:
        _logger.info(f"=== Broadband Wavelet {representation.capitalize()} ===")
        broadband_ds = _wavelet_transform(
            datasets,
            freqs,
            representation,
            keep_frequency_dim=keep_frequency_dim,
            wavelet_dir=(wavelet_dir / "broadband") if wavelet_dir else None,
            reuse_wavelets=reuse_wavelets,
        )
        for label, wd in broadband_ds.items():
            _logger.info(f"  [{label}] {wd}")

        _, _, bb_sw_results = _run_wavelet_isc(
            broadband_ds, save_dir / "broadband" / "isc", **_isc_kw
        )
        del broadband_ds

    # ── Per-band wavelet analysis ─────────────────────────────────
    _logger.info(f"=== Per-Band Wavelet {representation.capitalize()} ===")

    # Accumulators for cross-band comparison plots.
    band_loo_iscs: dict[str, dict] = {label: {} for label in datasets}
    band_sw_iscs: dict[str, dict] = {label: {} for label in datasets}
    feature_axis_label: str | None = None

    for band, (l_freq, h_freq) in selected_bands.items():
        n_freqs = max(
            2,
            int(round((h_freq - l_freq) / _WAVELET_BAND_FREQ_RESOLUTION_HZ)) + 1,
        )
        _logger.info(
            f"  Processing band: {band} ({l_freq:.1f}–{h_freq:.1f} Hz, {n_freqs} steps)"
        )
        band_freqs = np.linspace(l_freq, h_freq, n_freqs)
        band_ds = _wavelet_transform(
            datasets,
            band_freqs,
            representation,
            keep_frequency_dim=keep_frequency_dim,
            wavelet_dir=(wavelet_dir / f"band_{band}") if wavelet_dir else None,
            reuse_wavelets=reuse_wavelets,
        )
        band_dir = save_dir / f"band_{band}"
        if feature_axis_label is None:
            feature_axis_label = next(iter(band_ds.values())).feature_axis_label

        loo_iscs, mean_loo_iscs, sw_iscs = _run_wavelet_isc(
            band_ds, band_dir / "isc", **_isc_kw
        )

        for label in datasets:
            band_loo_iscs[label][band] = (loo_iscs[label], mean_loo_iscs[label])
            band_sw_iscs[label][band] = sw_iscs[label]
        del band_ds

    # ── Cross-band comparison plots ───────────────────────────────
    _logger.info(f"=== Band Comparison Wavelet {representation.capitalize()} ===")
    cmp_dir_name = "band_comparison"
    if len(selected_bands) != len(FREQUENCY_BANDS):
        band_tag = "_".join(selected_band_names)
        if len(band_tag) > 40:
            band_tag = f"{len(selected_band_names)}_bands"
        cmp_dir_name = f"band_comparison_{band_tag}"
    cmp_dir = save_dir / cmp_dir_name
    cmp_dir.mkdir(parents=True, exist_ok=True)
    if feature_axis_label is None:
        raise ValueError(
            "No valid bands were processed in wavelet per-band analysis. "
            "Ensure that at least one band from FREQUENCY_BANDS is selected."
        )

    plot_band_isc_distributions(
        band_loo_iscs,
        bands=selected_bands,
        feature_axis_label=(f"Number of {feature_axis_label.lower()}s"),
        save_path=cmp_dir / "band_isc_distributions.png",
    )
    plot_band_mean_isc_bar(
        band_loo_iscs,
        bands=selected_bands,
        save_path=cmp_dir / "band_isc_mean_bar.png",
    )
    plot_band_sliding_window_isc(
        band_sw_iscs,
        bands=selected_bands,
        isc_threshold=selected_band_thresholds,
        feature_axis_label=feature_axis_label,
        save_path=cmp_dir / "band_sliding_window_isc.png",
    )
    print_band_significant_intervals(
        band_sw_iscs,
        bands=selected_bands,
        band_thresholds=selected_band_thresholds,
        default_threshold=isc_threshold,
    )
    if bb_sw_results is not None:
        plot_band_overlap(
            band_sw_iscs,
            bands=selected_bands,
            band_thresholds=selected_band_thresholds,
            broadband_sw=bb_sw_results,
            broadband_threshold=isc_threshold,
            save_path=cmp_dir / "band_overlap.png",
        )

    _logger.info(f"Wavelet {representation} analysis complete.")
