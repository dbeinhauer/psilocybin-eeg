"""
Shared helpers and workflow functions for the unified analysis script.

Centralises data-loading, argument-parsing, display-setup and the actual
analysis workflows (ISC and mean/variance) so they can be driven by a single
entry-point script or from a Jupyter notebook.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")

from src.definitions.fields import (
    REAL_CONDITIONS,
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
    subset as data_subset,
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
    plot_band_power_timecourse,
    plot_cross_frequency_coupling,
    plot_intersubject_variance,
    plot_spectral_profile,
    plot_tf_isc,
    plot_tf_map,
    plot_wavelet_loo_isc_bar,
    plot_wavelet_loo_isc_distributions,
    plot_wavelet_topomap_isc,
)

from src.io.wavelet_store import (
    REPRESENTATIONS,
    WaveletSet,
    find_wavelet_set,
    load_wavelet_subjects,
    safe_label,
    wavelet_root,
    write_wavelet_subject,
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
    """Resolve the wavelet store root for a given experiment.

    The ``<experiment>/wavelets`` suffix is *always* part of the returned
    path, so the wavelets of one experiment can never be written into another
    experiment's directory — even when an explicit base directory is supplied.
    Each ``<Condition>_<MusicType>`` label has its own subdirectory below it, holding
    one file per participant (:mod:`src.io.wavelet_store`).

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
    return wavelet_root(root, experiment_name)


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


def add_iva_store_args(parser: argparse.ArgumentParser) -> None:
    """Add the CLI arguments controlling the IVA component store.

    Storing is opt-in rather than automatic because the dominant product is the
    ``(recordings, components, frequencies, times)`` time-frequency map: at the full
    time axis that is gigabytes per run, which is not something a plotting run
    should write to a shared filesystem without being asked.

    :param parser: Parser to extend.
    """
    parser.add_argument(
        "--store_components",
        action="store_true",
        default=False,
        help=(
            "Store the recovered IVA components (time-frequency maps, channel "
            "topographies and the marginals the variant produces) under "
            "data/processed/<experiment>/iva_results/<Condition>/, together with "
            "the participant label and condition of every row. Off by default: "
            "the maps are large. See src/io/iva_store.py."
        ),
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default=None,
        help=(
            "Processed-data root the component store is resolved against. The "
            "'<experiment>/iva_results/<Condition>' suffix is appended "
            "automatically, so pass a data root (e.g. data/processed), NOT an "
            "experiment-specific path. Defaults to the project's data/processed."
        ),
    )
    parser.add_argument(
        "--store_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64"],
        help=(
            "Floating precision the stored component arrays are cast to. "
            "'float32' (default) halves the store and is well past what any "
            "downstream read-out of these maps resolves; 'float64' keeps the "
            "computed values bit-exact."
        ),
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


def _ordered_metadata_column(
    filtered_df,
    column: SingleDataMetadata,
    n_subjects: int,
    description: str,
) -> list:
    """Read one metadata column in concatenated-subject order.

    Shares :func:`participant_labels`' contract: the subject axis is ordered by
    :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX` when
    the sidecar provides it, and by metadata row order (which *is* the concatenation
    order) otherwise.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param column: Metadata column to read.
    :param n_subjects: Number of subjects on the data's first axis.
    :param description: Human-readable column name for error messages.
    :return: ``n_subjects`` values, ordered by subject index.
    :raises ValueError: If the metadata is missing, empty, lacks *column*, or covers
        fewer than *n_subjects* rows.
    """
    if filtered_df is None or len(filtered_df) == 0:
        raise ValueError("No participant metadata available (filtered_df is empty).")
    if column not in filtered_df.columns:
        raise ValueError(
            f"Participant metadata has no {description} column; available: "
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
                filtered_df[column],
            )
        )
        missing = [s for s in range(n_subjects) if s not in by_index]
        if missing:
            raise ValueError(
                f"CONCATENATED_PERSON_INDEX is missing subject index/indices "
                f"{missing}; cannot map them to {description}."
            )
        return [by_index[s] for s in range(n_subjects)]

    return filtered_df[column].tolist()[:n_subjects]


def subject_conditions(filtered_df, n_subjects: int) -> list[str]:
    """Map each concatenated subject index to its condition.

    Needed whenever the dataset pools both conditions
    (:attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE`), where the subject axis
    holds every participant twice and the array alone cannot say which half is which.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param n_subjects: Number of subjects on the data's first axis.
    :return: ``n_subjects`` condition values (e.g. ``"Placebo"``), ordered by subject
        index.
    :raises ValueError: If the metadata cannot supply the mapping.
    """
    values = _ordered_metadata_column(
        filtered_df, SingleDataMetadata.CONDITION, n_subjects, "CONDITION"
    )
    return [v if isinstance(v, str) else v.value for v in values]


def condition_index_mask(
    filtered_df,
    n_subjects: int,
    condition: ConditionVariants,
) -> np.ndarray:
    """Boolean mask selecting the subjects recorded under *condition*.

    Prefer this over slicing by position: it keeps downstream code correct regardless
    of how the subject axis happens to be ordered.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param n_subjects: Number of subjects on the data's first axis.
    :param condition: Condition to select.
    :return: Boolean array of length *n_subjects*.
    :raises ValueError: If the metadata cannot supply the mapping.
    """
    conditions = subject_conditions(filtered_df, n_subjects)
    return np.asarray([c == condition.value for c in conditions], dtype=bool)


def participant_condition_labels(filtered_df, n_subjects: int) -> list[str]:
    """Label each concatenated subject index with participant *and* condition.

    In a pooled dataset every participant appears twice, so
    :func:`participant_labels` alone produces duplicate plot labels.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param n_subjects: Number of subjects on the data's first axis.
    :return: ``n_subjects`` labels of the form ``031 Placebo``.
    :raises ValueError: If the metadata cannot supply the mapping.
    """
    participants = participant_labels(filtered_df, n_subjects)
    conditions = subject_conditions(filtered_df, n_subjects)
    return [f"{p} {c}" for p, c in zip(participants, conditions)]


def paired_subject_index(filtered_df, n_subjects: int) -> np.ndarray:
    """Map each subject index to the index of the same participant's other condition.

    Lets paired (within-subject) statistics index directly into the subject axis
    without re-deriving the pairing. In a
    :attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE` dataset every participant
    contributes exactly two recordings, so every entry is a valid partner index.

    :param filtered_df: The analyser's ``filtered_df`` metadata table.
    :param n_subjects: Number of subjects on the data's first axis.
    :return: Integer array of length *n_subjects*; entry *k* is the partner of subject
        *k*, or ``-1`` where the participant has no second recording.
    :raises ValueError: If the metadata cannot supply the mapping, or a participant
        appears more than twice.
    """
    participants = participant_labels(filtered_df, n_subjects)
    partners = np.full(n_subjects, -1, dtype=int)

    positions: dict[str, list[int]] = {}
    for index, participant in enumerate(participants):
        positions.setdefault(participant, []).append(index)

    for participant, indices in positions.items():
        if len(indices) > 2:
            raise ValueError(
                f"Participant {participant} appears {len(indices)} times on the "
                "subject axis; a paired dataset allows at most two recordings."
            )
        if len(indices) == 2:
            first, second = indices
            partners[first], partners[second] = second, first
    return partners


def analyzers_to_datasets(analyzers: dict) -> dict[str, AnalysisData]:
    """Convert loaded analysers to AnalysisData instances.

    Each dataset's ``metadata["participants"]`` names the participant of every row,
    which is how its wavelets are found in the per-subject store
    (:mod:`src.io.wavelet_store`). An analyser without participant metadata yields a
    dataset without the list; it can still be transformed, but not stored.
    """
    datasets = {}
    for label, analyzer in analyzers.items():
        ad = analyzer.to_analysis_data(label=label)
        try:
            ad.metadata["participants"] = participant_labels(
                getattr(analyzer, "filtered_df", None), ad.data.shape[0]
            )
        except ValueError as error:
            _logger.warning(f"[{label}] rows carry no participant ids: {error}")
        datasets[label] = ad
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


def dataset_participants(ad: AnalysisData) -> list[str]:
    """The participant of every row of *ad*, as :func:`analyzers_to_datasets` records it.

    :raises ValueError: If *ad* names no participants, or one twice. The store holds
        one file per participant, so a pooled (joined) dataset cannot be stored as
        one: store each condition on its own and pool them afterwards
        (:func:`~scripts.notebook_helpers.load_joined_condition_wavelets`).
    """
    participants = ad.metadata.get("participants")
    if participants is None or len(participants) != ad.data.shape[0]:
        raise ValueError(
            f"[{ad.label}] the dataset does not name the participant of each of its "
            f"{ad.data.shape[0]} rows, so its wavelets cannot be stored or read per "
            "subject. Build it with analyzers_to_datasets."
        )
    duplicates = sorted({p for p in participants if list(participants).count(p) > 1})
    if duplicates:
        raise ValueError(
            f"[{ad.label}] participants {duplicates} appear more than once; store each "
            "condition on its own and pool them with load_joined_condition_wavelets."
        )
    return list(participants)


def _is_fragment(ad: AnalysisData) -> bool:
    """Whether *ad* was cut on its channel or time axis.

    A subject subset is not a fragment: each of its subjects is still whole.
    """
    original = ad.metadata.get("subset_of")
    return original is not None and tuple(original[1:]) != tuple(ad.data.shape[1:])


def _wavelet_output(
    data_4d: np.ndarray,
    ad: AnalysisData,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    representation: str,
    *,
    keep_frequency_dim: bool,
    reshape_frequency_dim: bool,
    source: Path | None,
) -> AnalysisData:
    """Wrap ``(subjects, channels, freqs, times)`` wavelets in the shape asked for.

    Power is averaged over frequency, phase circularly averaged, when the frequency
    axis is not kept; a kept axis is either reshaped out (4-D) or flattened into the
    feature axis channel-major, named ``<channel>@<freq>Hz``.
    """
    n_items, n_channels, n_freqs, n_samples = data_4d.shape
    channel_names = (
        list(ad.feature_names[:n_channels]) if ad.feature_names is not None else None
    )
    feature_names = channel_names
    if not keep_frequency_dim:
        data = (
            data_4d.mean(axis=2)
            if representation == "power"
            else circmean(data_4d, high=np.pi, low=-np.pi, axis=2)
        )
    elif reshape_frequency_dim:
        data = data_4d
    else:
        data = data_4d.reshape(n_items, n_channels * n_freqs, n_samples)
        if channel_names is not None:
            feature_names = [
                f"{name}@{freq:.1f}Hz" for name in channel_names for freq in freqs
            ]
    metadata = {
        **ad.metadata,
        "freqs": freqs,
        "n_cycles": n_cycles,
        "keep_frequency_dim": keep_frequency_dim,
    }
    if source is not None:
        metadata["loaded_from_wavelet_file"] = str(source)
    return AnalysisData(
        data=data,
        sfreq=ad.sfreq,
        representation=(
            DataRepresentation.WAVELET_PHASE
            if representation == "phase"
            else DataRepresentation.WAVELET_POWER
        ),
        label=(
            f"{ad.label} (wavelet {representation} "
            f"{freqs[0]:.0f}-{freqs[-1]:.0f} Hz)"
        ),
        feature_names=feature_names,
        info=ad.info,
        metadata=metadata,
    )


def _load_stored_wavelets(
    ad: AnalysisData,
    directory: Path,
    representation: str,
    freqs: np.ndarray,
) -> tuple[np.ndarray, WaveletSet] | None:
    """Read *ad*'s subjects from the store at *ad*'s own extent; ``None`` on a miss.

    The extent is the dataset's: its participants, its first channels and its first
    samples, so a subset run reads only what it keeps.

    :raises ValueError: If the stored channels are not the dataset's.
    """
    participants = dataset_participants(ad)
    try:
        wavelet_set = find_wavelet_set(directory, representation, freqs)
    except FileNotFoundError:
        return None
    missing = wavelet_set.missing(participants)
    if missing:
        _logger.info(f"[{ad.label}] {missing} not stored yet; computing.")
        return None

    n_channels, n_times = ad.data.shape[1], ad.data.shape[2]
    if ad.feature_names is not None and (
        wavelet_set.channel_names[:n_channels] != list(ad.feature_names[:n_channels])
    ):
        raise ValueError(
            f"[{ad.label}] the stored wavelets in {directory} have a different channel "
            "axis than the dataset; they were built from other data."
        )
    stored_times = wavelet_set.open(participants[0]).shape[-1]
    if stored_times != n_times and not _is_fragment(ad):
        _logger.warning(
            f"[{ad.label}] stored wavelets have {stored_times} samples, the dataset "
            f"{n_times}; reading {min(stored_times, n_times)}."
        )
    data = load_wavelet_subjects(
        wavelet_set,
        participants,
        n_channels=n_channels,
        n_times=min(stored_times, n_times),
    )
    _logger.info(f"[{ad.label}] read {data.shape} from {directory}")
    return data, wavelet_set


def _compute_wavelets(
    ad: AnalysisData,
    label: str,
    representation: str,
    freqs: np.ndarray,
    directory: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform *ad* one subject at a time, storing each subject under *directory*.

    A dataset cut on its channel or time axis is transformed but not stored: its
    arrays are fragments, and under a participant's name they would replace the
    whole array a later full run expects to find.

    :return: ``(subjects, channels, freqs, times)`` float32 and the ``n_cycles`` used.
    """
    store = directory is not None and not _is_fragment(ad)
    if directory is not None and not store:
        _logger.warning(
            f"[{label}] the dataset is cut on its channel or time axis; its wavelets "
            "are computed in memory and not stored."
        )
    participants = dataset_participants(ad) if store else None
    if store and ad.feature_names is None:
        raise ValueError(f"[{label}] wavelets cannot be stored without channel names.")

    transform = to_wavelet_power if representation == "power" else to_wavelet_phase
    n_items, n_channels, n_samples = ad.data.shape
    out = np.empty((n_items, n_channels, len(freqs), n_samples), np.float32)
    n_cycles = np.asarray(freqs, dtype=float) / 2.0
    for row in range(n_items):
        single = AnalysisData(
            data=ad.data[row : row + 1],
            sfreq=ad.sfreq,
            representation=ad.representation,
            label=ad.label,
            feature_names=ad.feature_names,
        )
        transformed = transform(single, freqs, keep_frequency_dim=True)
        n_cycles = np.asarray(transformed.metadata["n_cycles"], dtype=float)
        out[row] = transformed.data[0].reshape(n_channels, len(freqs), n_samples)
        if store:
            write_wavelet_subject(
                directory,
                participants[row],
                out[row],
                label=label,
                representation=representation,
                freqs=freqs,
                n_cycles=n_cycles,
                sfreq=ad.sfreq,
                channel_names=ad.feature_names,
            )
        _logger.info(f"[{label}] subject {row + 1}/{n_items} transformed")
    return out, n_cycles


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
    """Apply a wavelet *representation* to every dataset, through the per-subject store.

    With *wavelet_dir* (the experiment's ``wavelets`` root), each dataset's subjects
    live under ``<wavelet_dir>/<label>/`` — see :mod:`src.io.wavelet_store`. With
    *reuse_wavelets*, a dataset whose subjects are all stored is read back at its own
    extent and nothing is computed; otherwise every subject is transformed and stored.

    :param datasets: Source :class:`AnalysisData` objects keyed by label. Storing or
        reading needs ``metadata["participants"]`` (:func:`analyzers_to_datasets`).
    :param freqs: Morlet frequencies (Hz).
    :param representation: ``"power"`` or ``"phase"``.
    :param keep_frequency_dim: Keep frequency dimension instead of averaging it.
    :param reshape_frequency_dim: When ``keep_frequency_dim`` is true, return
        ``(n_items, n_features, n_freqs, n_samples)`` instead of flattening.
    :param wavelet_dir: The experiment's wavelet root, or ``None`` to neither read nor
        store.
    :param reuse_wavelets: Read stored subjects instead of recomputing them.
    :returns: New dict of transformed datasets with the same keys, float32.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Only 'power' and 'phase' wavelet representations are supported, got "
            f"{representation!r}"
        )
    if reshape_frequency_dim and not keep_frequency_dim:
        raise ValueError("reshape_frequency_dim=True requires keep_frequency_dim=True.")

    transformed: dict[str, AnalysisData] = {}
    for label, ad in datasets.items():
        directory = None if wavelet_dir is None else Path(wavelet_dir) / safe_label(label)
        stored = (
            _load_stored_wavelets(ad, directory, representation, freqs)
            if directory is not None and reuse_wavelets
            else None
        )
        if stored is not None:
            data_4d, wavelet_set = stored
            n_cycles, source = wavelet_set.n_cycles, directory
        else:
            data_4d, n_cycles = _compute_wavelets(
                ad, label, representation, freqs, directory
            )
            source = None
        transformed[label] = _wavelet_output(
            data_4d,
            ad,
            np.asarray(freqs, dtype=float),
            n_cycles,
            representation,
            keep_frequency_dim=keep_frequency_dim,
            reshape_frequency_dim=reshape_frequency_dim,
            source=source,
        )
    return transformed


def assemble_condition_track_wavelets(
    analyzer: "EEGSummarizedAnalyzer",
    music_type: MusicTypeVariants,
    freqs: np.ndarray,
    representation: str,
    wavelet_dir: Path,
) -> AnalysisData:
    """Assemble ``JoinedTracks`` wavelets from the per-condition stores, in memory.

    There is deliberately **no** ``JoinedTracks`` store on disk. Since the alignment is
    fitted once over every recording of both conditions, the two per-condition stores
    already share a time base and carry the same stimuli, so the joined dataset is just
    a selection plus a concatenation — cheap enough to redo per run. Only participants
    recorded under both conditions are read.

    Each track is z-scored on its own before concatenation. Both segments then have
    mean 0 and unit variance per ``(subject, channel, frequency)``, so the pooled
    series does too, which makes a downstream
    :func:`~src.analysis.wavelet_ica.zscore_by_time` an exact identity — every existing
    consumer works on this unchanged.

    :param analyzer: A ``JOINED_TRACKS`` analyser, used for the participant cohorts.
    :param music_type: Music type, needed to derive the per-condition labels.
    :param freqs: Morlet frequencies (Hz).
    :param representation: ``"power"`` or ``"phase"``.
    :param wavelet_dir: The experiment's wavelet root.
    :return: The concatenated 4-D ``(n_pairs, n_channels, n_freqs, n_times)`` dataset.
    :raises FileNotFoundError: If a per-condition store is missing. Build it first by
        running the same workflow for that single condition.
    """
    from src.analysis.condition_tracks import concatenate_condition_tracks

    joined_label = f"{ConditionVariants.JOINED_TRACKS.value}_{music_type.value}"
    cohorts = {c: _condition_participants(analyzer, c) for c in REAL_CONDITIONS}
    paired = [
        p for p in cohorts[REAL_CONDITIONS[0]] if all(p in cohorts[c] for c in cohorts)
    ]

    tracks: dict[ConditionVariants, AnalysisData] = {}
    for condition in REAL_CONDITIONS:
        label = f"{condition.value}_{music_type.value}"
        try:
            wavelet_set = find_wavelet_set(
                Path(wavelet_dir) / safe_label(label), representation, freqs
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"The {label} wavelets are missing, so the {joined_label} dataset "
                f"cannot be assembled. Run the same workflow with --condition "
                f"{condition.value} first."
            ) from error
        _logger.info(f"[{joined_label}] reading {label} for {len(paired)} pair(s) …")
        tracks[condition] = AnalysisData(
            data=load_wavelet_subjects(wavelet_set, paired),
            sfreq=wavelet_set.sfreq,
            representation=(
                DataRepresentation.WAVELET_PHASE
                if representation == "phase"
                else DataRepresentation.WAVELET_POWER
            ),
            label=label,
            feature_names=wavelet_set.channel_names,
            metadata={"freqs": wavelet_set.freqs, "n_cycles": wavelet_set.n_cycles},
        )

    paired_tracks = concatenate_condition_tracks(
        tracks,
        {condition: paired for condition in REAL_CONDITIONS},
        conditions=REAL_CONDITIONS,
        zscore_mode="per_condition",
        label=joined_label,
    )
    _logger.info(
        f"[{joined_label}] assembled {paired_tracks.n_pairs} pair(s): "
        f"{paired_tracks.data.data.shape}, segments {paired_tracks.segment_lengths}."
    )
    return paired_tracks.data


def _condition_participants(
    analyzer: "EEGSummarizedAnalyzer",
    condition: ConditionVariants,
) -> list[str]:
    """Participant labels of one condition's cohort under the analyser's exclusions.

    :param analyzer: A ``JOINED_TRACKS`` analyser, used for the dataset metadata,
        music types and exclusion categories.
    :param condition: Condition whose cohort is wanted.
    :return: Participant labels, in the cohort's metadata order.
    """
    from src.filtering.dataset_filter import DatasetFilter

    frame = DatasetFilter.filter_dataset_by_all_categories(
        analyzer.dataset_handler.dataset_metadata,
        analyzer.dataset_handler.excluded_participants_metadata,
        analyzer.music_types,
        [condition],
        analyzer.exclusion_categories,
    )
    return [participant_label(pid) for pid in frame[SingleDataMetadata.PARTICIPANT_ID]]


def precompute_pre_alignment_wavelet_cache(
    analyzers: dict[str, "EEGSummarizedAnalyzer"],
    freqs: np.ndarray,
    representations: list[str],
    wavelet_dir: Path,
    resample_freq: float = 250.0,
    n_jobs: int = -1,
) -> None:
    """Store per-subject wavelets computed from continuous ``RAW_AFTER_ICA`` data.

    For stimulus-based experiments (ASSR), computing wavelets on the
    stimulus-spliced ``RAW_CROPPED`` signal introduces edge artifacts at every
    splice point.  This function loads each subject's full, unspliced
    ``RAW_AFTER_ICA`` recording, computes the wavelet transform on the
    continuous signal, then trims the wavelet time axis to the aligned segments
    using :func:`~src.preprocessing.stimulus_alignment.apply_keep_segments_to_array`
    — reproducing the same splice that ``align_raws`` would apply, but after
    the transform instead of before.

    Each subject is written to the per-subject store
    (:func:`~src.io.wavelet_store.write_wavelet_subject`) as soon as it is
    transformed, so downstream steps reuse it via ``--reuse_wavelets``, and a run
    that stops part-way resumes at the first subject not yet stored.

    :param analyzers: Loaded :class:`~src.analysis.summary.EEGSummarizedAnalyzer`
        instances keyed by label (e.g. ``"Placebo_ASSR"``). Each must have a
        registered stimulus label (e.g. ASSR).
    :param freqs: Morlet wavelet frequencies (Hz).
    :param representations: Wavelet representations to compute and store;
        ``"power"`` and/or ``"phase"``.
    :param wavelet_dir: The experiment's wavelet root.
    :param resample_freq: Target sampling frequency in Hz (default 250).
    :param n_jobs: Parallel jobs for resampling (``-1`` = all CPUs).
    """
    n_cycles = np.asarray(freqs, dtype=float) / 2.0

    for label, analyzer in analyzers.items():
        directory = Path(wavelet_dir) / safe_label(label)
        # load_pre_alignment_data returns the recordings in the analyser's metadata
        # order, so the participants are read off that same table.
        participants = [
            participant_label(pid)
            for pid in analyzer.filtered_df[SingleDataMetadata.PARTICIPANT_ID]
        ]

        for representation in representations:
            try:
                pending = find_wavelet_set(directory, representation, freqs).missing(
                    participants
                )
            except FileNotFoundError:
                pending = list(participants)
            if not pending:
                _logger.info(
                    f"[{label}] all {len(participants)} subject(s) already stored "
                    f"({representation}); skipping."
                )
                continue

            _logger.info(
                f"[{label}] Pre-alignment wavelet ({representation}) for {pending}: "
                "loading RAW_AFTER_ICA per subject …"
            )
            # `segments` is indexed by this analyser's selection; `aligner` is fitted
            # over the whole alignment cohort, so its own keep_segments must not be
            # zipped with `arrays`.
            arrays, segments, aligner, info = analyzer.load_pre_alignment_data(
                resample_freq=resample_freq,
                n_jobs=n_jobs,
            )
            if len(arrays) != len(participants):
                raise ValueError(
                    f"[{label}] {len(arrays)} recording(s) loaded for "
                    f"{len(participants)} participant(s) in the metadata."
                )
            ch_names = list(info["ch_names"]) if info is not None else None

            for participant, subj_arr, keep in zip(participants, arrays, segments):
                if participant not in pending:
                    continue
                _logger.info(
                    f"  [{label}] {participant} ({subj_arr.shape[-1]} samples) …"
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
                trimmed = apply_keep_segments_to_array(wd.data[0], keep)
                write_wavelet_subject(
                    directory,
                    participant,
                    trimmed.reshape(len(ch_names), len(freqs), -1),
                    label=label,
                    representation=representation,
                    freqs=freqs,
                    n_cycles=n_cycles,
                    sfreq=resample_freq,
                    channel_names=ch_names,
                )
            _logger.info(f"[{label}] stored pre-alignment wavelets ({representation}).")


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
    analyzer: "EEGSummarizedAnalyzer | None" = None,
) -> AnalysisData:
    """Return a 4D wavelet ``AnalysisData`` for ``ad``.

    Forces ``keep_frequency_dim=True`` and ``reshape_frequency_dim=True`` so
    callers can index the result as ``(n_subjects, n_channels, n_freqs, n_times)``
    for the notebook-parity broadband plots.

    :param analyzer: The analyser *ad* came from, when available. A
        :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` dataset must
        have its cache assembled from the per-condition caches rather than transformed
        from its own spliced time-domain array, and this is how that is detected. Pass
        ``None`` for ordinary single-condition datasets.
    """
    if getattr(analyzer, "concatenates_condition_tracks", False):
        return assemble_condition_track_wavelets(
            analyzer, analyzer.music_types[0], freqs, representation, wavelet_dir
        )

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
        wavelet_dir=wavelet_dir,
        reuse_wavelets=reuse_wavelets,
    )
    bb_4d_data = wd_4d.data  # (n_subj, n_ch, n_freqs, n_times)
    _logger.info(f"[{label}] broadband 4D shape: {bb_4d_data.shape}")

    if representation == "phase":
        # Phase is extracted and cached, never analysed. Everything below treats its
        # input as a linear quantity -- means, variances, Pearson and Spearman ISC --
        # and phase angles are circular, so none of it is meaningful on them. The
        # extraction is kept because a later analysis may want the cache.
        _logger.info(f"[{label}] phase extracted and cached; no analysis is run on it.")
        return

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


def add_subset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the three axis limits every stage uses to run notebook-sized.

    Named the same everywhere so one habit carries across stages, and applied through
    :func:`~src.analysis.data_representations.subset` so they mean the same thing.

    :param parser: Parser to extend.
    """
    group = parser.add_argument_group("notebook-sized runs")
    group.add_argument(
        "--n_subjects",
        type=int,
        default=None,
        help="Keep only the first N subjects. Default: all.",
    )
    group.add_argument(
        "--n_channels",
        type=int,
        default=None,
        help="Keep only the first N channels. Default: all. Note a spatial filter or "
        "electrode mask built for the full montage will not match a subset.",
    )
    group.add_argument(
        "--n_times",
        type=int,
        default=None,
        help="Keep only the first N time samples. Default: all.",
    )


def apply_subset(datasets: dict, args: argparse.Namespace) -> dict:
    """Apply ``--n_subjects/--n_channels/--n_times`` to every dataset.

    :param datasets: Mapping of dataset key to :class:`AnalysisData`.
    :param args: Parsed arguments carrying the three limits.
    :return: The same mapping with each entry subset; unchanged when no limit is set.
    """
    limits = {
        "n_items": getattr(args, "n_subjects", None),
        "n_features": getattr(args, "n_channels", None),
        "n_samples": getattr(args, "n_times", None),
    }
    if all(value is None for value in limits.values()):
        return datasets
    out = {key: data_subset(value, **limits) for key, value in datasets.items()}
    shapes = {key: value.data.shape for key, value in out.items()}
    _logger.info(f"subset to {limits} -> {shapes}")
    return out
