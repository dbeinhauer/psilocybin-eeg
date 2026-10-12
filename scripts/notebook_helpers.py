"""
Notebook-facing helpers — a single, stable import surface for the exploration
notebooks under ``notebooks/``.

The notebooks reuse the shared analysis helpers in
:mod:`scripts.analysis_common`, but only a small, stable subset of them. This
module re-exports that subset and adds the notebook-only orchestration helper
:func:`compute_wavelet_datasets`, so a notebook imports everything from one
place and stays decoupled from the internal CLI/analysis API in
:mod:`scripts.analysis_common`.

Typical notebook usage::

    from scripts.notebook_helpers import (
        FREQUENCY_BANDS,
        WAVELET_FREQ_MAX,
        WAVELET_FREQ_MIN,
        WAVELET_N_FREQS,
        analyzers_to_datasets,
        compute_wavelet_datasets,
        load_analyzers,
        participant_labels,
    )

Both conditions are aligned together during preprocessing, so they share one time base
and carry the same stimuli. Selecting a condition — including the two virtual ones — is
therefore pure filtering, and needs no extra preprocessing or wavelet recomputation:

* :attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE` pools the conditions on the
  **subject** axis, so a participant appears twice. Build it from the existing
  per-condition stores with :func:`load_joined_condition_wavelets`.
* :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` pools them along
  **time**: each participant is one subject carrying their Placebo track followed by
  their Psilocybin track, assembled in memory from the per-condition stores
  (:func:`load_paired_condition_wavelets`).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Shared helpers re-exported so notebooks have a single import surface.
from scripts.analysis_common import (
    BAND_ISC_THRESHOLDS,
    FREQUENCY_BANDS,
    WAVELET_FREQ_MAX,
    WAVELET_FREQ_MIN,
    WAVELET_N_FREQS,
    analyzers_to_datasets,
    condition_index_mask,
    load_analyzers,
    paired_subject_index,
    participant_condition_labels,
    participant_label,
    participant_labels,
    assemble_condition_track_wavelets,
    precompute_pre_alignment_wavelet_cache,
    resolve_wavelet_dir,
    subject_conditions,
    wavelet_transform,
)
from src.analysis.condition_tracks import (
    PairedConditionTracks,
    PooledConditionSubjects,
    concatenate_condition_tracks,
    pool_condition_subjects,
)
from src.analysis.data_representations import AnalysisData
from src.analysis.data_representations import subset as data_subset
from src.definitions.fields import (
    REAL_CONDITIONS,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    MusicTypeVariants,
)
from src.preprocessing.stimulus_alignment import EXPERIMENT_STIMULUS_MARKERS

if TYPE_CHECKING:
    from src.analysis.summary import EEGSummarizedAnalyzer

_logger = logging.getLogger(__name__)

__all__ = [
    "BAND_ISC_THRESHOLDS",
    "FREQUENCY_BANDS",
    "WAVELET_FREQ_MAX",
    "WAVELET_FREQ_MIN",
    "WAVELET_N_FREQS",
    "analyzers_to_datasets",
    "compute_wavelet_datasets",
    "concatenate_condition_tracks",
    "condition_index_mask",
    "load_analyzers",
    "load_joined_condition_wavelets",
    "load_paired_condition_wavelets",
    "paired_subject_index",
    "pool_condition_subjects",
    "participant_condition_labels",
    "participant_label",
    "participant_labels",
    "resolve_wavelet_dir",
    "subject_conditions",
]


def compute_wavelet_datasets(
    datasets: dict[str, AnalysisData],
    analyzers: dict[str, "EEGSummarizedAnalyzer"],
    freqs: np.ndarray,
    representation: str,
    *,
    wavelet_dir: Path,
    reuse_wavelets: bool,
    experiment_name: ExperimentNames,
    keep_frequency_dim: bool = True,
    reshape_frequency_dim: bool = True,
    n_jobs: int = -1,
) -> dict[str, AnalysisData]:
    """Compute or load wavelet datasets, pre-aligning stimulus experiments.

    Use this instead of calling :func:`wavelet_transform` directly whenever the
    experiment may be stimulus-based (e.g. ASSR). For such experiments, Morlet
    wavelets computed on the stimulus-spliced ``RAW_CROPPED`` signal carry edge
    artifacts at every splice point. When the store is not being reused, this
    first computes the wavelet transform on the continuous ``RAW_AFTER_ICA``
    recordings and crops afterwards (via
    :func:`precompute_pre_alignment_wavelet_cache`), storing every subject under
    *wavelet_dir*, then reads them back. For non-stimulus experiments it simply
    runs :func:`wavelet_transform` on *datasets*.

    The pre-alignment transform always covers the analyser's **full** cohort and
    all channels, because the stimulus alignment is defined across the whole group.
    The read-back is at each dataset's own extent — its participants, first channels
    and first samples — so a subset run reads only what it keeps.

    :param datasets: Time-domain datasets keyed by label (possibly subset).
    :param analyzers: Loaded analysers keyed by the same label, used to load the
        continuous ``RAW_AFTER_ICA`` recordings for the pre-alignment precompute.
    :param freqs: Morlet frequencies (Hz).
    :param representation: ``"power"`` or ``"phase"``.
    :param wavelet_dir: The experiment's wavelet root (:func:`resolve_wavelet_dir`).
    :param reuse_wavelets: Reuse stored subjects when present. For stimulus
        experiments, ``False`` triggers the pre-alignment precompute.
    :param experiment_name: Experiment being analysed (selects the pre-alignment
        path for stimulus-based experiments).
    :param keep_frequency_dim: Forwarded to :func:`wavelet_transform`.
    :param reshape_frequency_dim: Forwarded to :func:`wavelet_transform`.
    :param n_jobs: Parallel jobs for the pre-alignment resampling.
    :returns: Wavelet datasets keyed by label, at the extent of each *datasets* entry.
    """
    # A JOINED_TRACKS dataset is assembled in memory from the per-condition stores
    # rather than transformed from its own (spliced) time-domain array. Nothing is
    # written: the per-condition stores already share a time base, so the join is a
    # selection plus a concatenation.
    track_analyzers = {
        label: analyzer
        for label, analyzer in analyzers.items()
        if getattr(analyzer, "concatenates_condition_tracks", False)
    }
    if track_analyzers:
        return {
            label: assemble_condition_track_wavelets(
                analyzer,
                analyzer.music_types[0],
                freqs,
                representation,
                wavelet_dir,
            )
            for label, analyzer in track_analyzers.items()
        }

    effective_reuse = reuse_wavelets
    if experiment_name in EXPERIMENT_STIMULUS_MARKERS and not reuse_wavelets:
        _logger.info(
            "Stimulus-based experiment '%s': computing the pre-alignment wavelets "
            "from continuous RAW_AFTER_ICA (full cohort) before loading.",
            experiment_name.value,
        )
        precompute_pre_alignment_wavelet_cache(
            analyzers,
            freqs=freqs,
            representations=[representation],
            wavelet_dir=wavelet_dir,
            n_jobs=n_jobs,
        )
        effective_reuse = True

    return wavelet_transform(
        datasets,
        freqs,
        representation,
        keep_frequency_dim=keep_frequency_dim,
        reshape_frequency_dim=reshape_frequency_dim,
        wavelet_dir=wavelet_dir,
        reuse_wavelets=effective_reuse,
    )


def _load_condition_tracks(
    music_type: MusicTypeVariants,
    exclusion_categories: Sequence[ExclusionCategories],
    freqs: np.ndarray,
    *,
    wavelet_dir: Path,
    experiment_name: ExperimentNames,
    representation: str,
    conditions: Sequence[ConditionVariants],
    n_channels: int | None,
    n_times: int | None,
    reuse_wavelets: bool,
    n_jobs: int,
) -> tuple[
    dict[ConditionVariants, AnalysisData],
    dict[ConditionVariants, list[str]],
    dict[ConditionVariants, np.ndarray | None],
    dict[ConditionVariants, "EEGSummarizedAnalyzer"],
]:
    """Load every condition's 4-D wavelets, one condition at a time.

    The subject axis is never subset: pairing across conditions needs every recording
    each condition has. Channels and time are, through
    :func:`~src.analysis.data_representations.subset`, so only that extent is read and
    nothing cut is ever stored.

    :return: Per condition: the wavelet track, its participants, its stimulus onsets
        and the analyser it came from.
    """
    tracks: dict[ConditionVariants, AnalysisData] = {}
    participants: dict[ConditionVariants, list[str]] = {}
    onsets: dict[ConditionVariants, np.ndarray | None] = {}
    analyzers_by_condition: dict[ConditionVariants, "EEGSummarizedAnalyzer"] = {}

    for condition in conditions:
        analyzers = load_analyzers(
            [music_type],
            condition,
            exclusion_categories,
            process_and_save=False,
            normalize_data=False,
            experiment_name=experiment_name,
            n_jobs=n_jobs,
        )
        label = f"{condition.value}_{music_type.value}"
        analyzer = analyzers[label]
        datasets = {
            key: data_subset(ad, n_features=n_channels, n_samples=n_times)
            for key, ad in analyzers_to_datasets(analyzers).items()
        }
        track = compute_wavelet_datasets(
            datasets,
            analyzers,
            freqs,
            representation,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=reuse_wavelets,
            experiment_name=experiment_name,
            keep_frequency_dim=True,
            reshape_frequency_dim=True,
            n_jobs=n_jobs,
        )[label]

        tracks[condition] = track
        participants[condition] = list(
            datasets[label].metadata.get("participants")
            or participant_labels(analyzer.filtered_df, track.data.shape[0])
        )
        onsets[condition] = analyzer.stimulus_onsets
        analyzers_by_condition[condition] = analyzer
        _logger.info(
            f"[{label}] {track.data.shape}, {len(participants[condition])} "
            "participant(s)."
        )
        del datasets
    return tracks, participants, onsets, analyzers_by_condition


def load_paired_condition_wavelets(
    music_type: MusicTypeVariants,
    exclusion_categories: Sequence[ExclusionCategories],
    freqs: np.ndarray,
    *,
    wavelet_dir: Path,
    experiment_name: ExperimentNames,
    representation: str = "power",
    conditions: Sequence[ConditionVariants] = REAL_CONDITIONS,
    zscore_mode: str = "per_condition",
    n_channels: int | None = None,
    n_times: int | None = None,
    reuse_wavelets: bool = True,
    n_jobs: int = -1,
) -> tuple[PairedConditionTracks, dict[ConditionVariants, "EEGSummarizedAnalyzer"]]:
    """Load each condition's wavelets and lay the participant-matched tracks end to end.

    The one-call path to the time-concatenated joined dataset: it loads each condition
    exactly as any single-condition workflow does — reusing the per-condition stores —
    then hands both to
    :func:`~src.analysis.condition_tracks.concatenate_condition_tracks`. Nothing
    upstream is re-run or re-aligned.

    Contrast with :func:`load_joined_condition_wavelets`
    (:attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE`), which pools the same
    cohort on the *subject* axis and therefore needs the two conditions to share one
    time base. Here they keep their own time bases and are pooled along *time* instead,
    so each participant is one subject carrying both tracks.

    :param music_type: Music type to load (``MusicTypeVariants.ASSR`` for ASSR).
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs: Morlet frequencies (Hz).
    :param wavelet_dir: The experiment's wavelet root (:func:`resolve_wavelet_dir`).
    :param experiment_name: Experiment being analysed.
    :param representation: ``"power"`` or ``"phase"``.
    :param conditions: Conditions to concatenate, in time-axis order.
    :param zscore_mode: See
        :data:`~src.analysis.condition_tracks.ZSCORE_MODES`. Defaults to
        ``"per_condition"``: each condition is z-scored on its own and the standardised
        tracks are then concatenated.
    :param n_channels: Keep only the first *n_channels* channels of each condition.
        ``None`` keeps all.
    :param n_times: Keep only the first *n_times* time samples **of each condition**,
        i.e. of each segment, not of the concatenation. ``None`` keeps all.
    :param reuse_wavelets: Reuse stored subjects when present.
    :param n_jobs: Parallel jobs for loading/resampling.
    :return: Tuple ``(paired, analyzers)`` where *paired* is the concatenated dataset
        with its segment bookkeeping and *analyzers* maps each condition to the
        analyser it was loaded from (for ``info``, montage and metadata).
    """
    tracks, participants, onsets, analyzers_by_condition = _load_condition_tracks(
        music_type,
        exclusion_categories,
        freqs,
        wavelet_dir=wavelet_dir,
        experiment_name=experiment_name,
        representation=representation,
        conditions=conditions,
        n_channels=n_channels,
        n_times=n_times,
        reuse_wavelets=reuse_wavelets,
        n_jobs=n_jobs,
    )
    paired = concatenate_condition_tracks(
        tracks,
        participants,
        conditions=conditions,
        zscore_mode=zscore_mode,
        onsets=onsets,
    )
    _logger.info(
        f"Paired {paired.n_pairs} participant(s) across "
        f"{[c.value for c in paired.conditions]}; concatenated shape "
        f"{paired.data.data.shape}, segment lengths {paired.segment_lengths}."
    )
    return paired, analyzers_by_condition


def load_joined_condition_wavelets(
    music_type: MusicTypeVariants,
    exclusion_categories: Sequence[ExclusionCategories],
    freqs: np.ndarray,
    *,
    wavelet_dir: Path,
    experiment_name: ExperimentNames,
    representation: str = "power",
    conditions: Sequence[ConditionVariants] = REAL_CONDITIONS,
    n_channels: int | None = None,
    n_times: int | None = None,
    reuse_wavelets: bool = True,
    n_jobs: int = -1,
) -> tuple[PooledConditionSubjects, dict[ConditionVariants, "EEGSummarizedAnalyzer"]]:
    """Load each condition's wavelets and stack the participant-matched recordings.

    The one-call path to the subject-axis joined dataset — the layout
    :attr:`~src.definitions.fields.ConditionVariants.JOINED_SEPARATE` names. It loads each
    condition exactly as any single-condition workflow does, **reusing the
    per-condition stores**, then hands both to
    :func:`~src.analysis.condition_tracks.pool_condition_subjects`. Nothing upstream is
    re-run or re-aligned, and nothing is stored for the pooled cohort.

    Contrast with :func:`load_paired_condition_wavelets`, which pools the same cohort
    along *time* instead, so each participant is one subject carrying both tracks.

    Pass *n_channels* / *n_times* to keep the pooled tensor small during exploration;
    there is no subject subset because dropping subjects before matching would break
    the pairing (slice
    :attr:`~src.analysis.condition_tracks.PooledConditionSubjects.data` afterwards
    instead, or use the participant bookkeeping to choose which pairs to keep).

    :param music_type: Music type to load (``MusicTypeVariants.ASSR`` for ASSR).
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs: Morlet frequencies (Hz).
    :param wavelet_dir: The experiment's wavelet root (:func:`resolve_wavelet_dir`).
    :param experiment_name: Experiment being analysed.
    :param representation: ``"power"`` or ``"phase"``.
    :param conditions: Conditions to pool, in subject-axis block order.
    :param n_channels: Keep only the first *n_channels* channels. ``None`` keeps all.
    :param n_times: Keep only the first *n_times* time samples. ``None`` keeps all.
    :param reuse_wavelets: Reuse stored subjects when present.
    :param n_jobs: Parallel jobs for loading/resampling.
    :return: Tuple ``(pooled, analyzers)`` where *pooled* is the subject-axis joined
        dataset with its per-subject participant/condition bookkeeping and *analyzers*
        maps each condition to the analyser it was loaded from (for ``info``, montage
        and metadata).
    """
    tracks, participants, onsets, analyzers_by_condition = _load_condition_tracks(
        music_type,
        exclusion_categories,
        freqs,
        wavelet_dir=wavelet_dir,
        experiment_name=experiment_name,
        representation=representation,
        conditions=conditions,
        n_channels=n_channels,
        n_times=n_times,
        reuse_wavelets=reuse_wavelets,
        n_jobs=n_jobs,
    )
    pooled = pool_condition_subjects(
        tracks,
        participants,
        conditions=conditions,
        onsets=onsets,
    )
    _logger.info(
        f"Pooled {pooled.n_pairs} participant(s) across "
        f"{[c.value for c in pooled.conditions]} onto {pooled.n_subjects} subject(s); "
        f"shape {pooled.data.data.shape}."
    )
    return pooled, analyzers_by_condition
