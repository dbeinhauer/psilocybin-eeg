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
    SpectrumTypeVariants,
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
from src.definitions.frequency import (
    WAVELET_FREQ_MAX,
    WAVELET_FREQ_MIN,
    WAVELET_N_FREQS,
)
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
    compute_sliding_window_isc_spearman,
    FREQUENCY_BANDS,
)
from src.analysis.mean_variance import (
    compute_intersubject_stats,
    compute_windowed_stats,
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
    plot_multiscale_sliding_window_isc,
    plot_band_multiscale_sliding_window_isc,
)
from src.visualization.mean_variance_plots import (
    plot_timeseries,
    plot_variance_distribution,
    plot_windowed_analysis,
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

from src.preprocessing.stimulus_alignment import apply_keep_segments_to_array

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
# Path resolution
# ──────────────────────────────────────────────────────────────────────


def resolve_wavelet_dir(
    base: str | Path | None,
    experiment_name: ExperimentNames,
) -> Path:
    """Resolve the wavelet cache directory for a given experiment.

    The ``<experiment>/wavelets`` suffix is *always* part of the returned
    path, so the wavelets of one experiment can never be written into another
    experiment's directory — even when an explicit base directory is supplied.

    :param base: Optional base *data* directory. When ``None``, defaults to
        :attr:`ProjectPaths.PROCESSED_DATA_DIR`. It must point at a data root,
        not at an experiment-specific or ``wavelets`` directory; the
        ``<experiment>/wavelets`` suffix is appended automatically.
    :param experiment_name: Experiment whose wavelets are being stored/loaded.
    :returns: ``<base>/<experiment>/wavelets``.
    :raises ValueError: If ``base`` already contains an experiment-name
        segment (e.g. a stale ``.../psilo_music/wavelets`` path), which would
        nest or mis-route the cache across experiments.
    """
    root = Path(base) if base is not None else ProjectPaths.PROCESSED_DATA_DIR
    experiment_values = {e.value for e in ExperimentNames}
    offending = experiment_values.intersection(root.parts)
    if offending:
        raise ValueError(
            "--wavelet_data_dir must be a base data directory, not an "
            f"experiment-specific path (found experiment segment(s) "
            f"{sorted(offending)} in '{root}'). The '<experiment>/wavelets' "
            "suffix is added automatically — pass e.g. 'data/processed'."
        )
    return root / experiment_name.value / "wavelets"


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────


def add_wavelet_grid_args(parser: argparse.ArgumentParser) -> None:
    """Add the three wavelet-grid CLI arguments shared across wavelet scripts."""
    parser.add_argument(
        "--wavelet_freq_min",
        type=float,
        default=WAVELET_FREQ_MIN,
        help="Minimum Morlet frequency (Hz). Defaults to the project-wide "
        "wavelet grid minimum (src.definitions.frequency.WAVELET_FREQ_MIN).",
    )
    parser.add_argument(
        "--wavelet_freq_max",
        type=float,
        default=WAVELET_FREQ_MAX,
        help="Maximum Morlet frequency (Hz). Defaults to the project-wide "
        "wavelet grid maximum (src.definitions.frequency.WAVELET_FREQ_MAX).",
    )
    parser.add_argument(
        "--wavelet_n_freqs",
        type=int,
        default=WAVELET_N_FREQS,
        help="Number of Morlet frequency steps. Defaults to the project-wide "
        "wavelet grid size (src.definitions.frequency.WAVELET_N_FREQS), giving "
        "≈ 1 Hz resolution over the default range.",
    )


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
        "--experiment",
        type=str,
        default=ExperimentNames.PSILO_MUSIC.value,
        choices=[e.value for e in ExperimentNames],
        help="Which experiment dataset to analyse.",
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
        default=None,
        choices=[mt.value for mt in MusicTypeVariants],
        help=(
            "One or more music types to analyse. When omitted, defaults to "
            "all music types for the psilo_music experiment and ASSR for the "
            "assr experiment."
        ),
    )
    parser.add_argument(
        "--process_and_save",
        action="store_true",
        default=False,
        help=(
            "Force a rebuild of the concatenated time-domain array from "
            "RAW_CROPPED (load, resample, stack, save), overwriting any cached "
            "copy. Normally unnecessary: when omitted, the saved array is "
            "loaded if present and built automatically on first use."
        ),
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
        help="Medium sliding-window length in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--window_fine_sec",
        type=float,
        default=1.0,
        help="Fine sliding-window length in seconds for multi-scale ISC (default: 1.0).",
    )
    parser.add_argument(
        "--window_large_sec",
        type=float,
        default=15.0,
        help="Large sliding-window length in seconds for multi-scale ISC (default: 15.0).",
    )
    parser.add_argument(
        "--step_sec",
        type=float,
        default=2.5,
        help="Sliding-window step size in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--n_ch_subsample",
        type=int,
        default=64,
        help=(
            "Number of channels randomly subsampled for Spearman ISC computation. "
            "Set to 0 to use all channels (much slower, default: 64)."
        ),
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
    add_wavelet_grid_args(parser)
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
        default=None,
        help=(
            "Base data directory under which wavelet-transformed datasets are "
            "stored. The '<experiment>/wavelets' suffix is appended "
            "automatically, so pass a data root (e.g. data/processed), NOT an "
            "experiment-specific path. When omitted, defaults to "
            "data/processed. Resolves to <base>/<experiment>/wavelets."
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
    experiment_name: ExperimentNames = ExperimentNames.PSILO_MUSIC,
) -> dict[str, EEGSummarizedAnalyzer]:
    """Load (or process & save) and normalise analysers for each music type.

    Returns a dict keyed by ``"{condition}_{music_type}"``
    (e.g. ``"Placebo_CLASSIC"``), matching the on-disk data file naming
    convention used by :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.save_data`.

    :param experiment_name: Which experiment dataset to load. Defaults to the
        psilocybin music-listening experiment; pass
        :attr:`~src.definitions.fields.ExperimentNames.ASSR` for the
        auditory steady-state response experiment.
    """
    from src.analysis.summary import EEGSummarizedAnalyzer

    analyzers: dict[str, EEGSummarizedAnalyzer] = {}
    for mt in music_types:
        label = f"{condition.value}_{mt.value}"
        analyzer = EEGSummarizedAnalyzer(
            experiment_name=experiment_name,
            coordinate_system=CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS,
            music_types=[mt],
            conditions=[condition],
            exclusion_categories=list(exclusion_categories),
        )

        if process_and_save:
            # Forced (re)build of the concatenated time-domain array from
            # RAW_CROPPED — use to refresh a stale cache or change resampling.
            analyzer.load_and_prepare_data(resample_freq=250.0, n_jobs=n_jobs)
            _logger.info(f"[{label}] data shape: {analyzer.data.shape}")
            analyzer.save_data()
        else:
            # Default: load the previously-saved concatenated array. If none
            # exists yet, build it from RAW_CROPPED and save it once, so no
            # explicit --process_and_save flag is ever needed on a first run.
            # This keeps invocations consistent across every experiment: the
            # time-domain scaffolding is managed automatically and the only
            # wavelet-cache control the caller needs is --reuse_wavelets.
            #
            # Caveat for stimulus-aligned experiments (e.g. ASSR): the wavelet
            # cache itself is computed from the continuous RAW_AFTER_ICA
            # recordings (see precompute_pre_alignment_wavelet_cache) and does
            # NOT conceptually depend on this concatenated time-domain array.
            # The surrounding workflow still needs it for MNE Info / array-shape
            # scaffolding, so this load — or the RAW_CROPPED-based rebuild it
            # falls back to — must succeed. In other words, storing the
            # stimulus-aligned wavelets will still fail if neither the saved
            # concatenated array nor the RAW_CROPPED data is available, even
            # though that data is not used to compute the wavelets themselves.
            try:
                analyzer.load_data(
                    info_filename=analyzer.filtered_df[
                        SingleDataMetadata.FILENAME
                    ].iloc[0],
                )
                _logger.info(f"[{label}] Loaded data shape: {analyzer.data.shape}")
            except FileNotFoundError:
                _logger.info(
                    f"[{label}] No saved concatenated array found; building it "
                    "from RAW_CROPPED and saving it for reuse."
                )
                analyzer.load_and_prepare_data(resample_freq=250.0, n_jobs=n_jobs)
                _logger.info(f"[{label}] data shape: {analyzer.data.shape}")
                analyzer.save_data()

        if normalize_data:
            analyzer.normalize()
        analyzers[label] = analyzer

    return analyzers


def participant_label(participant_id) -> str:
    """Format a metadata participant ID as its zero-padded 3-digit label.

    The sidecar CSV round-trip coerces ``PARTICIPANT_ID`` to an integer (``31``),
    losing the zero padding of the original ``"031"``, so the digits are
    re-padded here. A ``PSI`` prefix on the input is stripped; plot labels carry
    the bare number.

    :param participant_id: Participant ID from the dataset metadata.
    :return: Label of the form ``031``.
    """
    digits = "".join(ch for ch in str(participant_id) if ch.isdigit())
    return (digits[-3:] if digits else "").zfill(3)


def participant_labels(filtered_df, n_subjects: int) -> list[str]:
    """Map each concatenated subject index to its 3-digit participant label.

    The subject axis of a concatenated array is ordered by
    :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX`,
    which
    :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.load_and_prepare_data`
    writes into the metadata sidecar. When an analyser was constructed but its
    sidecar was never loaded, that column is absent — the rows are still in
    concatenation order, so positional order is used as the fallback.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param n_subjects: Number of subjects on the data's first axis. May be
        smaller than ``len(filtered_df)`` when a subject subset is in use.
    :return: ``n_subjects`` labels, ordered by subject index.
    :raises ValueError: If the metadata is missing, empty, carries no
        participant IDs, or covers fewer than ``n_subjects`` rows.
    """
    if filtered_df is None or len(filtered_df) == 0:
        raise ValueError("No participant metadata available (filtered_df is empty).")
    if SingleDataMetadata.PARTICIPANT_ID not in filtered_df.columns:
        raise ValueError(
            "Participant metadata has no PARTICIPANT_ID column; available: "
            f"{list(filtered_df.columns)}"
        )
    if len(filtered_df) < n_subjects:
        raise ValueError(
            f"Participant metadata covers {len(filtered_df)} recording(s) but the "
            f"data has {n_subjects} subject(s)."
        )

    if SingleDataMetadata.CONCATENATED_PERSON_INDEX in filtered_df.columns:
        by_index = dict(
            zip(
                filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX],
                filtered_df[SingleDataMetadata.PARTICIPANT_ID],
            )
        )
        missing = [s for s in range(n_subjects) if s not in by_index]
        if missing:
            raise ValueError(
                f"CONCATENATED_PERSON_INDEX is missing subject index/indices "
                f"{missing}; cannot map them to participants."
            )
        return [participant_label(by_index[s]) for s in range(n_subjects)]

    # No sidecar mapping — rows are still in concatenation order.
    _logger.warning(
        "Participant metadata has no CONCATENATED_PERSON_INDEX column; falling "
        "back to metadata row order (the concatenation order) for subject labels."
    )
    ids = filtered_df[SingleDataMetadata.PARTICIPANT_ID].tolist()
    return [participant_label(pid) for pid in ids[:n_subjects]]


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


def precompute_pre_alignment_wavelet_cache(
    analyzers: dict[str, "EEGSummarizedAnalyzer"],
    freqs: np.ndarray,
    representations: list[str],
    wavelet_dir: Path,
    resample_freq: float = 250.0,
    n_jobs: int = -1,
) -> None:
    """Compute per-subject wavelet caches from continuous ``RAW_AFTER_ICA`` data.

    For stimulus-based experiments (ASSR), computing wavelets on the
    stimulus-spliced ``RAW_CROPPED`` signal introduces edge artifacts at every
    splice point.  This function loads each subject's full, unspliced
    ``RAW_AFTER_ICA`` recording, computes the wavelet transform on the
    continuous signal, then trims the wavelet time axis to the aligned segments
    using :func:`~src.preprocessing.stimulus_alignment.apply_keep_segments_to_array`
    — reproducing the same splice that ``align_raws`` would apply, but after
    the transform instead of before.

    Saves the result to the same ``.npz`` cache format that
    :func:`wavelet_transform` produces, so downstream steps (Stage-04 ICA,
    Stage-05 IVA) transparently reuse it via ``--reuse_wavelets``.

    Skips any label whose cache file already exists (idempotent).

    :param analyzers: Loaded :class:`~src.analysis.summary.EEGSummarizedAnalyzer`
        instances keyed by label (e.g. ``"Placebo_ASSR"``). Each must have a
        registered stimulus label (e.g. ASSR).
    :param freqs: Morlet wavelet frequencies (Hz).
    :param representations: Wavelet representations to compute and cache;
        ``"power"`` and/or ``"phase"``.
    :param wavelet_dir: Directory where cache files are written (the same
        directory passed as ``wavelet_dir`` to :func:`wavelet_transform`).
    :param resample_freq: Target sampling frequency in Hz (default 250).
    :param n_jobs: Parallel jobs for resampling (``-1`` = all CPUs).
    """
    wavelet_dir = Path(wavelet_dir)
    freq_sig = f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{len(freqs)}"
    n_cycles = freqs / 2.0

    for label, analyzer in analyzers.items():
        safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
        if not safe_label:
            safe_label = f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"

        for representation in representations:
            cache_file = wavelet_dir / (
                f"{safe_label}__wavelet_{representation}__{freq_sig}__freqdim1.npz"
            )
            if cache_file.exists():
                _logger.info(
                    f"[{label}] Pre-alignment wavelet cache already exists, "
                    f"skipping: {cache_file.name}"
                )
                continue

            _logger.info(
                f"[{label}] Pre-alignment wavelet ({representation}): "
                "loading RAW_AFTER_ICA per subject …"
            )

            arrays, aligner, info = analyzer.load_pre_alignment_data(
                resample_freq=resample_freq,
                n_jobs=n_jobs,
            )
            ch_names = list(info["ch_names"]) if info is not None else None

            wavelet_aligned: list[np.ndarray] = []
            cache_label: str | None = None
            cache_feature_names: list[str] | None = None

            for subj_idx, (subj_arr, segments) in enumerate(
                zip(arrays, aligner.keep_segments)
            ):
                _logger.info(
                    f"  [{label}] subject {subj_idx + 1}/{len(arrays)} "
                    f"({subj_arr.shape[-1]} samples) …"
                )
                ad_subj = AnalysisData(
                    data=subj_arr[np.newaxis].astype(float),
                    sfreq=resample_freq,
                    representation=DataRepresentation.TIME_DOMAIN,
                    label=label,
                    feature_names=ch_names,
                    info=info,
                )
                if representation == "power":
                    wd = to_wavelet_power(ad_subj, freqs, keep_frequency_dim=True)
                else:
                    wd = to_wavelet_phase(ad_subj, freqs, keep_frequency_dim=True)

                # wd.data: (1, n_ch*n_freqs, n_times_full) — trim the time axis.
                trimmed = apply_keep_segments_to_array(wd.data[0], segments)
                # trimmed: (n_ch*n_freqs, n_times_aligned)
                wavelet_aligned.append(trimmed)

                if cache_label is None:
                    cache_label = wd.label
                    cache_feature_names = wd.feature_names

            stacked = np.stack(
                wavelet_aligned
            )  # (n_subj, n_ch*n_freqs, n_times_aligned)
            _logger.info(
                f"[{label}] stacked pre-alignment wavelet shape: {stacked.shape}"
            )

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_file,
                data=stacked,
                sfreq=np.float64(resample_freq),
                label=cache_label if cache_label is not None else label,
                feature_names=np.asarray(
                    cache_feature_names if cache_feature_names is not None else [],
                    dtype=str,
                ),
                has_feature_names=np.asarray(
                    int(cache_feature_names is not None),
                    dtype=np.int8,
                ),
                freqs=freqs,
                n_cycles=np.asarray(n_cycles),
                keep_frequency_dim=np.asarray(1, dtype=np.int8),
            )
            _logger.info(
                f"[{label}] saved pre-alignment wavelet cache "
                f"({representation}): {cache_file.name}"
            )


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
    freqs: np.ndarray,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Compute per-band phase LOO-ISC(cos φ) from the broadband phase cache.

    Used by the wavelet-power workflow to render the optional power vs.
    phase joint plot. Slices the cached *broadband* phase wavelet tensor per
    band (matching the broadband-only caching scheme) rather than reading
    separate per-band caches. Returns ``{}`` (and logs) when no broadband
    phase cache is present.
    """
    if wavelet_dir is None:
        return {}
    bb_phase_dir = wavelet_dir / SpectrumTypeVariants.BROADBAND.value
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
    if not safe_label:
        safe_label = f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    freq_sig = f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{len(freqs)}"
    cache_file = bb_phase_dir / f"{safe_label}__wavelet_phase__{freq_sig}__freqdim1.npz"
    if not cache_file.exists():
        _logger.info(
            f"[{label}] No broadband phase wavelet cache "
            f"(expected {cache_file.name}); skipping power_phase_joint plot."
        )
        return {}
    try:
        phase_4d = _broadband_wavelet_4d(
            ad,
            label,
            representation="phase",
            freqs=freqs,
            wavelet_dir=bb_phase_dir,
            reuse_wavelets=True,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _logger.warning(f"[{label}] Failed to load broadband phase cache: {exc}")
        return {}
    return compute_phase_band_loo_iscs(phase_4d.data, freqs, bands=bands)


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
    window_fine_sec: float,
    window_large_sec: float,
    n_ch_subsample: int,
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

    bb_dir = save_dir / SpectrumTypeVariants.BROADBAND.value
    bands_dir = save_dir / SpectrumTypeVariants.BANDS.value

    # ── Broadband ────────────────────────────────────────────────
    bb_sw_for_overlap: tuple[np.ndarray, np.ndarray] | None = None
    band_mean_iscs_from_bb: dict[str, np.ndarray] = {}

    # Compute (or load) the broadband 4D wavelet tensor once. Both the
    # broadband stage and the per-band stage slice this same tensor along the
    # frequency axis — no separate per-band wavelet is computed or cached.
    wd_4d = _broadband_wavelet_4d(
        ad,
        label,
        representation=representation,
        freqs=freqs,
        wavelet_dir=(wavelet_dir / SpectrumTypeVariants.BROADBAND.value)
        if wavelet_dir
        else None,
        reuse_wavelets=reuse_wavelets,
    )
    bb_4d_data = wd_4d.data  # (n_subj, n_ch, n_freqs, n_times)
    _logger.info(f"[{label}] broadband 4D shape: {bb_4d_data.shape}")

    if include_broadband:
        # Reduce to 3D for mean-variance and ISC plots.
        wd_3d = _wavelet_4d_to_3d(
            wd_4d,
            representation=representation,
            base_feature_names=ad.feature_names,
        )
        n_times_bb = wd_3d.data.shape[2]
        n_channels_bb = wd_3d.data.shape[1]
        sfreq_bb = wd_3d.sfreq

        # Channel subsampling for Spearman ISC
        if n_ch_subsample > 0 and n_ch_subsample < n_channels_bb:
            rng_bb = np.random.default_rng(42)
            ch_idx_bb = np.sort(
                rng_bb.choice(n_channels_bb, n_ch_subsample, replace=False)
            )
            data_sub_bb = wd_3d.data[:, ch_idx_bb, :]
            _logger.info(
                f"[{label}] Subsampling {n_ch_subsample}/{n_channels_bb} "
                "channels for Spearman (broadband)."
            )
        else:
            data_sub_bb = wd_3d.data
            ch_idx_bb = None

        # ── Broadband mean-variance (matching 01-raw-mean-variance-analysis) ──
        # Only meaningful for power; phase angles are circular and shouldn't be
        # passed through the linear intersubject-variance pipeline.
        if representation == "power":
            _logger.info(f"[{label}] Broadband mean-variance analysis …")
            bb_mv_dir_ts = bb_dir / "timeseries"
            bb_mv_dir_ts.mkdir(parents=True, exist_ok=True)
            bb_mv_dir_var = bb_dir / "variance"
            bb_mv_dir_var.mkdir(parents=True, exist_ok=True)
            bb_mv_dir_win = bb_dir / "windowed"
            bb_mv_dir_win.mkdir(parents=True, exist_ok=True)

            # Z-score per subject × channel along time to normalise amplitude
            # scale differences before computing intersubject variance (mirrors
            # the z-scoring in plot_intersubject_variance).
            bb_mean = wd_3d.data.mean(axis=2, keepdims=True)
            bb_std = wd_3d.data.std(axis=2, keepdims=True)
            bb_data_z = (wd_3d.data - bb_mean) / (bb_std + 1e-10)

            bb_stats = compute_intersubject_stats(bb_data_z)
            plot_timeseries(
                bb_stats,
                sfreq_bb,
                label,
                save_path=bb_mv_dir_ts / f"timeseries_{label}.png",
            )
            plot_variance_distribution(
                bb_stats["inter_var"],
                label,
                save_path=bb_mv_dir_var / f"variance_distribution_{label}.png",
            )
            df_wins_bb = compute_windowed_stats(
                bb_stats,
                n_times=n_times_bb,
                sfreq=sfreq_bb,
                window_sec=window_sec,
                step_sec=step_sec,
            )
            plot_windowed_analysis(
                bb_stats,
                df_wins_bb,
                sfreq_bb,
                label,
                window_sec,
                10.0,
                step_sec=step_sec,
                save_path_bar=bb_mv_dir_win / f"windowed_bar_{label}.png",
                save_path_overlay=bb_mv_dir_win / f"windowed_overlay_{label}.png",
            )

        # ── Broadband LOO-ISC and sliding-window ISC ──────────────
        # Pearson/Spearman ISC requires linear data; phase angles are circular
        # and are handled via the phase-specific workflow below (cos(circmean)).
        if representation == "power":
            _logger.info(f"[{label}] Broadband LOO-ISC distribution …")
            loo_3d, mean_loo_3d = compute_loo_isc(wd_3d.data)
            bb_loo_dir = bb_dir / "loo_isc"
            bb_loo_dir.mkdir(parents=True, exist_ok=True)
            plot_loo_isc_distribution(
                {label: mean_loo_3d},
                ylabel=f"Number of {wd_3d.feature_axis_label.lower()}s",
                save_path=bb_loo_dir / f"loo_isc_distribution_{label}.png",
            )

            _logger.info(f"[{label}] Broadband multi-scale sliding-window ISC …")
            step_fine = window_fine_sec / 2
            step_med = window_sec / 2
            step_large = window_large_sec / 2

            _logger.info(
                f"  Fine   ({window_fine_sec:.0f} s / {step_fine:.1f} s step) …"
            )
            sw_isc_fine, sw_times_fine = compute_sliding_window_isc(
                data_sub_bb, window_fine_sec, step_fine, sfreq_bb
            )
            _logger.info(f"  Medium ({window_sec:.0f} s / {step_med:.1f} s step) …")
            sw_isc_med, sw_times_med = compute_sliding_window_isc(
                data_sub_bb, window_sec, step_med, sfreq_bb
            )
            _logger.info(
                f"  Large  ({window_large_sec:.0f} s / {step_large:.1f} s step) …"
            )
            sw_isc_large, sw_times_large = compute_sliding_window_isc(
                data_sub_bb, window_large_sec, step_large, sfreq_bb
            )
            _logger.info(
                f"  Spearman medium ({window_sec:.0f} s / {step_med:.1f} s step) …"
            )
            sw_isc_spear, _ = compute_sliding_window_isc_spearman(
                data_sub_bb, window_sec, step_med, sfreq_bb
            )

            bb_sw_for_overlap = (sw_isc_med, sw_times_med)

            sw_dir_bb = bb_dir / "sliding_window"
            sw_dir_bb.mkdir(parents=True, exist_ok=True)
            plot_multiscale_sliding_window_isc(
                label,
                sw_isc_fine,
                sw_times_fine,
                sw_isc_med,
                sw_times_med,
                sw_isc_large,
                sw_times_large,
                sw_isc_spear,
                sfreq_bb,
                n_times_bb,
                window_fine_sec=window_fine_sec,
                window_med_sec=window_sec,
                window_large_sec=window_large_sec,
                isc_threshold=isc_threshold,
                n_ch_subsample=n_ch_subsample if ch_idx_bb is not None else None,
                save_path_bar=sw_dir_bb / f"sw_isc_bar_{label}.png",
                save_path_overlay=sw_dir_bb / f"sw_isc_overlay_{label}.png",
                save_path_comparison=sw_dir_bb
                / f"sw_isc_pearson_vs_spearman_{label}.png",
            )
            print_significant_intervals(
                {label: (sw_isc_med, sw_times_med)}, isc_threshold=isc_threshold
            )

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
                freqs=freqs,
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

        # Keep wd_4d / bb_4d_data — the per-band stage slices them below.
        del wd_3d, data_sub_bb

    # ── Per-band wavelet mean-variance and ISC ───────────────────
    band_loo_iscs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    band_sw_fine: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    band_sw_med: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    band_sw_large: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    band_sw_spearman_med: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    feature_axis_label: str | None = None

    band_mv_dir_ts = bands_dir / "timeseries"
    band_mv_dir_ts.mkdir(parents=True, exist_ok=True)
    band_mv_dir_var = bands_dir / "variance"
    band_mv_dir_var.mkdir(parents=True, exist_ok=True)
    band_mv_dir_win = bands_dir / "windowed"
    band_mv_dir_win.mkdir(parents=True, exist_ok=True)
    band_loo_dir = bands_dir / "loo_isc"
    band_loo_dir.mkdir(parents=True, exist_ok=True)

    for band, (l_freq, h_freq) in selected_bands.items():
        band_mask = (freqs >= l_freq) & (freqs <= h_freq)
        if not band_mask.any():
            _logger.warning(
                f"  [{label}] No broadband frequencies fall in {band} range "
                f"[{l_freq:.1f}, {h_freq:.1f}] Hz; skipping band."
            )
            continue
        _logger.info(
            f"  [{label}] band {band} ({l_freq:.1f}–{h_freq:.1f} Hz, "
            f"{int(band_mask.sum())} bins sliced from broadband)"
        )
        # Slice the broadband 4D tensor to this band, then reduce over the
        # frequency axis (mean for power, circular mean for phase). No separate
        # per-band wavelet is computed or cached.
        band_4d = AnalysisData(
            data=bb_4d_data[:, :, band_mask, :],
            sfreq=wd_4d.sfreq,
            representation=wd_4d.representation,
            label=wd_4d.label,
            feature_names=wd_4d.feature_names,
            info=wd_4d.info,
            metadata={**wd_4d.metadata, "freqs": freqs[band_mask]},
        )
        wd = _wavelet_4d_to_3d(
            band_4d,
            representation=representation,
            base_feature_names=ad.feature_names,
        )
        if feature_axis_label is None:
            feature_axis_label = wd.feature_axis_label

        # ── Per-band mean-variance (matching 01-raw-mean-variance-analysis) ──
        # Only meaningful for power; phase angles are circular and require
        # different normalisation.
        n_times_band = wd.data.shape[2]
        if representation == "power":
            _logger.info(f"  [{label}] {band} mean-variance analysis …")
            # Z-score per subject × channel along time (matches
            # plot_intersubject_variance in wavelet_plots.py).
            band_mean = wd.data.mean(axis=2, keepdims=True)
            band_std = wd.data.std(axis=2, keepdims=True)
            band_data_z = (wd.data - band_mean) / (band_std + 1e-10)
            band_stats = compute_intersubject_stats(band_data_z)
            plot_timeseries(
                band_stats,
                wd.sfreq,
                f"{label} / {band}",
                save_path=band_mv_dir_ts / f"{band}_timeseries_{label}.png",
            )
            plot_variance_distribution(
                band_stats["inter_var"],
                f"{label} / {band}",
                save_path=band_mv_dir_var / f"{band}_variance_distribution_{label}.png",
            )
            df_wins_band = compute_windowed_stats(
                band_stats,
                n_times=n_times_band,
                sfreq=wd.sfreq,
                window_sec=window_sec,
                step_sec=step_sec,
            )
            plot_windowed_analysis(
                band_stats,
                df_wins_band,
                wd.sfreq,
                f"{label} / {band}",
                window_sec,
                10.0,
                step_sec=step_sec,
                save_path_bar=band_mv_dir_win / f"{band}_windowed_bar_{label}.png",
                save_path_overlay=band_mv_dir_win
                / f"{band}_windowed_overlay_{label}.png",
            )

        # ── Per-band LOO-ISC distribution ──────────────────────────
        # For phase data use cos(circmean(phase)) to convert circular angles to
        # a linear-correlation-friendly representation (matching the pattern in
        # compute_phase_band_loo_iscs).
        _logger.info(f"  [{label}] {band} LOO-ISC …")
        if representation == "power":
            loo_data = wd.data
        else:
            loo_data = np.cos(wd.data)
        loo, mean_loo = compute_loo_isc(loo_data)
        plot_loo_isc_distribution(
            {f"{label} / {band}": mean_loo},
            ylabel=f"Number of {wd.feature_axis_label.lower()}s",
            save_path=band_loo_dir / f"{band}_loo_isc_distribution_{label}.png",
        )
        band_loo_iscs[band] = (loo, mean_loo)

        # ── Per-band multi-scale sliding-window ISC ────────────────
        # Pearson/Spearman ISC requires linear data; skip for phase.
        if representation == "power":
            n_ch_band = wd.data.shape[1]
            if n_ch_subsample > 0 and n_ch_subsample < n_ch_band:
                rng_band = np.random.default_rng(42)
                ch_idx_band = np.sort(
                    rng_band.choice(n_ch_band, n_ch_subsample, replace=False)
                )
                band_data_sub = wd.data[:, ch_idx_band, :]
            else:
                band_data_sub = wd.data
                ch_idx_band = None

            _logger.info(f"  [{label}] {band} multi-scale sliding-window ISC …")
            tc_fine, t_fine = compute_sliding_window_isc(
                band_data_sub, window_fine_sec, window_fine_sec / 2, wd.sfreq
            )
            tc_med, t_med = compute_sliding_window_isc(
                band_data_sub, window_sec, window_sec / 2, wd.sfreq
            )
            tc_large, t_large = compute_sliding_window_isc(
                band_data_sub, window_large_sec, window_large_sec / 2, wd.sfreq
            )
            tc_sp, _ = compute_sliding_window_isc_spearman(
                band_data_sub, window_sec, window_sec / 2, wd.sfreq
            )
            band_sw_fine[band] = (tc_fine, t_fine)
            band_sw_med[band] = (tc_med, t_med)
            band_sw_large[band] = (tc_large, t_large)
            band_sw_spearman_med[band] = (tc_sp, t_med)

        del band_4d, wd

    del wd_4d, bb_4d_data

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

    # Multi-scale sliding-window ISC per band (power only)
    if band_sw_med:
        sliding_window_dir = bands_dir / "sliding_window"
        sliding_window_dir.mkdir(parents=True, exist_ok=True)
        plot_band_multiscale_sliding_window_isc(
            label,
            band_sw_fine,
            band_sw_med,
            band_sw_large,
            band_sw_spearman_med,
            sfreq=ad.sfreq,
            n_times=ad.data.shape[2],
            window_fine_sec=window_fine_sec,
            window_med_sec=window_sec,
            window_large_sec=window_large_sec,
            band_thresholds=selected_band_thresholds,
            bands=selected_bands,
            n_ch_subsample=(
                n_ch_subsample
                if n_ch_subsample > 0 and n_ch_subsample < ad.data.shape[1]
                else None
            ),
            save_path_dir=sliding_window_dir,
        )
        print_band_significant_intervals(
            {label: band_sw_med},
            bands=selected_bands,
            band_thresholds=selected_band_thresholds,
            default_threshold=isc_threshold,
        )
        if bb_sw_for_overlap is not None:
            plot_band_overlap(
                {label: band_sw_med},
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
    window_fine_sec: float = 1.0,
    window_large_sec: float = 15.0,
    n_ch_subsample: int = 64,
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
    power workflow — an optional power vs. phase joint comparison).  It also
    runs the mean-variance analysis (matching the 01-raw-mean-variance-analysis
    conventions) and a multi-scale sliding-window ISC (fine / medium / large
    windows + per-channel heatmap).

    The per-band stage runs the same mean-variance and multi-scale ISC analyses
    for each canonical frequency band; bandpass filtering of wavelet data is
    explicitly avoided as it would be semantically incorrect on
    frequency-decomposed data.

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
    :param window_sec: Medium sliding-window length in seconds (also used as
        the window for mean-variance windowed analysis).
    :param step_sec: Step size in seconds for the mean-variance windowed
        analysis.  The multi-scale ISC steps are automatically set to half
        of the respective window length (50 % overlap).
    :param window_fine_sec: Fine sliding-window length for multi-scale ISC.
    :param window_large_sec: Large sliding-window length for multi-scale ISC.
    :param n_ch_subsample: Number of channels randomly subsampled for Spearman
        ISC computation.  Set to ``0`` to use all channels (slower).
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
            window_fine_sec=window_fine_sec,
            window_large_sec=window_large_sec,
            n_ch_subsample=n_ch_subsample,
            info=info,
            cross_representation_wavelet_dir=cross_representation_wavelet_dir,
        )

    _logger.info(f"Wavelet {representation} analysis complete.")
