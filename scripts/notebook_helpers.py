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

* :attr:`~src.definitions.fields.ConditionVariants.JOINED` pools the conditions on the
  **subject** axis, so a participant appears twice. Build it from the existing
  per-condition caches with :func:`load_joined_condition_wavelets`.
* :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` pools them along
  **time**: each participant is one subject carrying their Placebo track followed by
  their Psilocybin track, assembled in memory from the per-condition caches
  (:func:`load_paired_condition_wavelets`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
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
from src.analysis.data_representations import AnalysisData, DataRepresentation
from src.definitions.constants import ProjectPaths
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
    "resolve_notebook_wavelet_cache_dir",
    "resolve_wavelet_dir",
    "subject_conditions",
]


def _slice_wavelet_to_extent(
    wavelet_ad: AnalysisData,
    reference_ad: AnalysisData | None,
) -> AnalysisData:
    """Trim a wavelet :class:`AnalysisData` to the extent of a reference dataset.

    The pre-alignment cache always spans the full subject cohort, all channels
    and the full aligned time axis (the stimulus alignment is defined across the
    whole group, so it cannot be computed on a subset). *reference_ad* is the
    possibly-subset time-domain dataset that drives the caller's exploration
    extent; this slices the leading subjects, channels and time samples of the
    wavelet tensor to match it, leaving the frequency axis untouched.

    :param wavelet_ad: Wavelet output, 4-D ``(items, channels, freqs, times)``
        or 3-D ``(items, channels, times)``.
    :param reference_ad: Time-domain dataset whose ``(items, channels, times)``
        extent should be matched. ``None`` returns *wavelet_ad* unchanged.
    :returns: A sliced copy of *wavelet_ad*, or the original when no trim is
        needed.
    """
    if reference_ad is None:
        return wavelet_ad
    n_items = reference_ad.data.shape[0]
    n_channels = reference_ad.data.shape[1]
    n_times = reference_ad.data.shape[2]
    data = wavelet_ad.data
    if data.ndim == 4:  # (items, channels, freqs, times)
        sliced = data[:n_items, :n_channels, :, :n_times]
    elif data.ndim == 3:  # (items, channels, times)
        sliced = data[:n_items, :n_channels, :n_times]
    else:
        return wavelet_ad
    if sliced.shape == data.shape:
        return wavelet_ad
    feature_names = wavelet_ad.feature_names
    if feature_names is not None and len(feature_names) > n_channels:
        feature_names = list(feature_names[:n_channels])
    return dataclasses.replace(wavelet_ad, data=sliced, feature_names=feature_names)


#: Where the notebook-level wavelet **subset** caches live, under the stage-03
#: notebook that owns the Morlet transform. One directory for every workflow: the
#: caches are keyed by dataset and extent, not by the notebook that wrote them, so a
#: subset computed in 03 is picked up unchanged by 04, 05 and 06.
NOTEBOOK_WAVELET_CACHE_SUBDIR = Path("03-wavelet-analysis") / "wavelet_cache"


def resolve_notebook_wavelet_cache_dir(experiment_name: ExperimentNames) -> Path:
    """Resolve the notebook-level wavelet subset-cache directory for an experiment.

    This is the companion of :func:`~scripts.analysis_common.resolve_wavelet_dir`, and
    the two are not interchangeable:

    * ``resolve_wavelet_dir`` points at the **source-of-truth** cache under
      ``data/processed/<experiment>/wavelets`` — one file per condition spanning the
      full cohort, all channels and the whole aligned time axis. That is the tens-of-GB
      artefact the preprocessing stage writes, and reading it means decompressing all
      of it even when only a subset is wanted.
    * this one points at ``notebooks/03-wavelet-analysis/wavelet_cache/<experiment>``,
      holding **per-extent** copies of that same data — the trimmed tensor a notebook
      actually works on. Pass it as ``subset_cache_dir`` to
      :func:`compute_wavelet_datasets` and the first run writes the trim, every later
      run with the same extent reads it back and never touches the big cache.

    It lives under the stage-03 notebook because that is where the Morlet transform is
    owned; nothing about the directory is stage-03-specific, and every wavelet workflow
    is meant to share it.

    :param experiment_name: Experiment whose subset caches are being stored/loaded.
    :returns: ``notebooks/03-wavelet-analysis/wavelet_cache/<experiment>``. Callers
        append the band subdirectory (e.g. ``"broadband"``), matching
        ``resolve_wavelet_dir``.
    """
    return (
        ProjectPaths.NOTEBOOKS_DIR
        / NOTEBOOK_WAVELET_CACHE_SUBDIR
        / experiment_name.value
    )


def _subset_cache_path(
    subset_cache_dir: Path,
    label: str,
    representation: str,
    freqs: np.ndarray,
    reference_ad: AnalysisData | None,
    *,
    keep_frequency_dim: bool,
    reshape_frequency_dim: bool,
) -> Path:
    """Cache filename for one label at one requested extent.

    The extent is part of the name rather than something checked after loading, so a
    lookup never reads a file it would then have to reject, and caches for different
    subsets of the same dataset coexist. Everything that changes the stored array is in
    the name: the dataset, the representation, the frequency grid, the
    subject/channel/time extent and the two frequency-axis flags.

    :param subset_cache_dir: Directory the caches live in.
    :param label: Dataset label, e.g. ``"Placebo_ASSR"``.
    :param representation: ``"power"`` or ``"phase"``.
    :param freqs: Morlet frequencies (Hz).
    :param reference_ad: The time-domain dataset defining the requested extent, or
        ``None`` when the full extent is wanted.
    :param keep_frequency_dim: Whether the frequency axis is kept.
    :param reshape_frequency_dim: Whether a kept frequency axis is reshaped out.
    :return: Path to the ``.npz`` cache file.
    """
    safe_label = re.sub(r"[^A-Za-z0-9_-]", "_", label).strip("_")
    if not safe_label:
        safe_label = f"dataset_{hashlib.sha256(label.encode()).hexdigest()[:8]}"
    freq_sig = f"{freqs[0]:.3f}_{freqs[-1]:.3f}_{len(freqs)}"
    if reference_ad is None:
        extent = "full"
    else:
        n_items, n_features, n_samples = reference_ad.data.shape[:3]
        extent = f"S{n_items}_C{n_features}_T{n_samples}"
    dims = f"freqdim{int(keep_frequency_dim)}{'r' if reshape_frequency_dim else ''}"
    return subset_cache_dir / (
        f"{safe_label}__wavelet_{representation}__{freq_sig}__{extent}__{dims}.npz"
    )


def _load_subset_cache(path: Path, reference_ad: AnalysisData | None) -> AnalysisData:
    """Read a wavelet subset cache back as an :class:`AnalysisData`.

    The array is stored in exactly the shape the caller asked for, so nothing is
    reduced or reshaped on the way out. ``info`` cannot be serialised into an ``.npz``
    and is taken from *reference_ad* instead.

    :param path: Cache file written by :func:`_save_subset_cache`.
    :param reference_ad: Time-domain dataset the extent was taken from; supplies
        ``info`` and the base metadata. May be ``None``.
    :return: The cached wavelet dataset.
    """
    loaded = np.load(path)
    feature_names = (
        loaded["feature_names"].tolist()
        if loaded["has_feature_names"].item() and loaded["feature_names"].size > 0
        else None
    )
    return AnalysisData(
        data=loaded["data"],
        sfreq=float(loaded["sfreq"]),
        representation=(
            DataRepresentation.WAVELET_PHASE
            if str(loaded["representation"]) == "phase"
            else DataRepresentation.WAVELET_POWER
        ),
        label=str(loaded["label"]),
        feature_names=feature_names,
        info=None if reference_ad is None else reference_ad.info,
        metadata={
            **({} if reference_ad is None else reference_ad.metadata),
            "freqs": loaded["freqs"],
            "n_cycles": loaded["n_cycles"],
            "keep_frequency_dim": bool(loaded["keep_frequency_dim"]),
            # Kept under the same key the per-condition cache uses, so the "where did
            # this come from" prints in the notebooks keep working unchanged.
            "loaded_from_wavelet_file": str(path),
            "loaded_from_subset_cache": str(path),
        },
    )


def _save_subset_cache(
    path: Path,
    wavelet_ad: AnalysisData,
    freqs: np.ndarray,
    representation: str,
) -> None:
    """Write a wavelet subset cache, atomically and uncompressed.

    **Uncompressed** on purpose: the point of this cache is to be fast, and
    decompressing the source-of-truth cache is exactly the cost being avoided. Wavelet
    power is dense float data that compresses poorly, so the space saved would not pay
    for the time spent.

    Written to a temporary file in the same directory and renamed into place, so an
    interrupted multi-GB write cannot leave a truncated cache that a later run would
    happily reuse.

    :param path: Destination path from :func:`_subset_cache_path`.
    :param wavelet_ad: Wavelet dataset to store, in the shape it should be read back.
    :param freqs: Morlet frequencies (Hz).
    :param representation: ``"power"`` or ``"phase"``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cycles = wavelet_ad.metadata.get("n_cycles")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                data=wavelet_ad.data,
                sfreq=np.float64(wavelet_ad.sfreq),
                label=np.asarray(wavelet_ad.label),
                representation=np.asarray(representation),
                feature_names=np.asarray(wavelet_ad.feature_names or [], dtype=str),
                has_feature_names=np.asarray(
                    int(wavelet_ad.feature_names is not None), dtype=np.int8
                ),
                freqs=np.asarray(freqs),
                n_cycles=np.asarray(
                    np.asarray(freqs) / 2.0 if n_cycles is None else n_cycles
                ),
                keep_frequency_dim=np.asarray(
                    int(bool(wavelet_ad.metadata.get("keep_frequency_dim", True))),
                    dtype=np.int8,
                ),
            )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


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
    subset_cache_dir: Path | None = None,
    reuse_subset_cache: bool = True,
    n_jobs: int = -1,
) -> dict[str, AnalysisData]:
    """Compute or load wavelet datasets, pre-aligning stimulus experiments.

    Use this instead of calling :func:`wavelet_transform` directly whenever the
    experiment may be stimulus-based (e.g. ASSR). For such experiments, Morlet
    wavelets computed on the stimulus-spliced ``RAW_CROPPED`` signal carry edge
    artifacts at every splice point. When the cache is not being reused, this
    first computes the wavelet transform on the continuous ``RAW_AFTER_ICA``
    recordings and crops afterwards (via
    :func:`precompute_pre_alignment_wavelet_cache`), writes the cache into
    *wavelet_dir*, then loads it back. For non-stimulus experiments it simply
    runs :func:`wavelet_transform` on *datasets*.

    The pre-alignment cache is always computed on the **full** subject cohort
    and all channels, because the stimulus alignment (common onset count, trim
    windows) is defined across the whole group; a subset would change the
    alignment and the time axis. Any subject/channel/time subsetting already
    applied to *datasets* is therefore re-applied to the loaded result instead
    (see :func:`_slice_wavelet_to_extent`), so downstream exploration still sees
    the requested subset. This means subsetting does not reduce the cost of the
    pre-alignment computation — only the size of the returned tensors.

    :param datasets: Time-domain datasets keyed by label (possibly subset).
    :param analyzers: Loaded analysers keyed by the same label, used to load the
        continuous ``RAW_AFTER_ICA`` recordings for the pre-alignment precompute.
    :param freqs: Morlet frequencies (Hz).
    :param representation: ``"power"`` or ``"phase"``.
    :param wavelet_dir: Cache directory (e.g. ``WAVELET_DIR / "broadband"``).
    :param reuse_wavelets: Reuse a cached transform when present. For stimulus
        experiments, ``False`` triggers the pre-alignment precompute.
    :param experiment_name: Experiment being analysed (selects the pre-alignment
        path for stimulus-based experiments).
    :param keep_frequency_dim: Forwarded to :func:`wavelet_transform`.
    :param reshape_frequency_dim: Forwarded to :func:`wavelet_transform`.
    :param subset_cache_dir: Optional directory for the notebook-level **subset**
        cache — see :func:`resolve_notebook_wavelet_cache_dir`. When given, the trimmed
        result is written there on the first run and read straight back on every later
        run that asks for the same dataset at the same extent, so the source-of-truth
        cache is never decompressed again. ``None`` (the default) disables it, leaving
        the behaviour of existing call-sites unchanged.
    :param reuse_subset_cache: Read an existing subset cache when one is present. Set
        ``False`` to recompute and overwrite it — the escape hatch for a stale cache.
        Ignored when *subset_cache_dir* is ``None``.
    :param n_jobs: Parallel jobs for the pre-alignment resampling.
    :returns: Wavelet datasets keyed by label, trimmed to the extent of the
        corresponding *datasets* entry for stimulus experiments.
    """
    # A JOINED_TRACKS dataset is assembled in memory from the per-condition caches
    # rather than transformed from its own (spliced) time-domain array. Nothing is
    # written: the per-condition caches already share a time base, so the join is a
    # selection plus a concatenation.
    track_analyzers = {
        label: analyzer
        for label, analyzer in analyzers.items()
        if getattr(analyzer, "concatenates_condition_tracks", False)
    }
    if track_analyzers:
        if subset_cache_dir is not None:
            # The extent key is read off `datasets`, which this path does not consume —
            # the assembly is driven by the analyser instead — so a cache written here
            # would be labelled with an extent it does not have. The per-condition
            # sources it reads can carry their own subset caches; this join cannot.
            _logger.info(
                "Condition-track assembly does not use the wavelet subset cache; "
                "cache its per-condition sources instead."
            )
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

    # Split the request into labels already cached at this extent and labels that must
    # be computed. A full hit returns without touching the source-of-truth cache at
    # all, which is the whole point: that read is the expensive step.
    subset_paths: dict[str, Path] = {}
    cached: dict[str, AnalysisData] = {}
    if subset_cache_dir is not None:
        for label, ad in datasets.items():
            subset_paths[label] = _subset_cache_path(
                subset_cache_dir,
                label,
                representation,
                freqs,
                ad,
                keep_frequency_dim=keep_frequency_dim,
                reshape_frequency_dim=reshape_frequency_dim,
            )
            if reuse_subset_cache and subset_paths[label].exists():
                _logger.info(
                    "[%s] Reusing wavelet subset cache: %s",
                    label,
                    subset_paths[label].name,
                )
                cached[label] = _load_subset_cache(subset_paths[label], ad)
        if len(cached) == len(datasets):
            return {label: cached[label] for label in datasets}
        datasets = {label: ad for label, ad in datasets.items() if label not in cached}

    is_stimulus = experiment_name in EXPERIMENT_STIMULUS_MARKERS
    effective_reuse = reuse_wavelets
    if is_stimulus and not reuse_wavelets:
        _logger.info(
            "Stimulus-based experiment '%s': computing the pre-alignment wavelet "
            "cache from continuous RAW_AFTER_ICA (full cohort) before loading.",
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

    transformed = wavelet_transform(
        datasets,
        freqs,
        representation,
        keep_frequency_dim=keep_frequency_dim,
        reshape_frequency_dim=reshape_frequency_dim,
        wavelet_dir=wavelet_dir,
        reuse_wavelets=effective_reuse,
    )

    if is_stimulus:
        transformed = {
            label: _slice_wavelet_to_extent(ad, datasets.get(label))
            for label, ad in transformed.items()
        }

    if subset_cache_dir is not None:
        for label, ad in transformed.items():
            _logger.info(
                "[%s] Writing wavelet subset cache: %s",
                label,
                subset_paths[label].name,
            )
            _save_subset_cache(subset_paths[label], ad, freqs, representation)
        merged = {**cached, **transformed}
        # Restore the caller's key order, which the hit/miss split broke.
        return {label: merged[label] for label in (*cached, *transformed)}
    return transformed


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
    subset_cache_dir: Path | None = None,
    reuse_subset_cache: bool = True,
    n_jobs: int = -1,
) -> tuple[PairedConditionTracks, dict[ConditionVariants, "EEGSummarizedAnalyzer"]]:
    """Load each condition's wavelets and lay the participant-matched tracks end to end.

    The one-call path to the time-concatenated joined dataset: it loads each condition
    exactly as any single-condition workflow does — reusing the existing per-condition
    wavelet caches — then hands both to
    :func:`~src.analysis.condition_tracks.concatenate_condition_tracks`. Nothing
    upstream is re-run or re-aligned.

    Contrast with :func:`load_joined_condition_wavelets`
    (:attr:`~src.definitions.fields.ConditionVariants.JOINED`), which pools the same
    cohort on the *subject* axis and therefore needs the two conditions to share one
    time base. Here they keep their own time bases and are pooled along *time* instead,
    so each participant is one subject carrying both tracks.

    :param music_type: Music type to load (``MusicTypeVariants.ASSR`` for ASSR).
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs: Morlet frequencies (Hz).
    :param wavelet_dir: Wavelet cache directory, e.g.
        ``resolve_wavelet_dir(None, experiment_name) / "broadband"``.
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
    :param reuse_wavelets: Reuse the cached transform when present.
    :param subset_cache_dir: Optional notebook-level subset cache, forwarded per
        condition to :func:`compute_wavelet_datasets`. See
        :func:`resolve_notebook_wavelet_cache_dir`. Pass *n_channels* / *n_times* with
        it: the cache is keyed by extent, so caching the full extent saves nothing.
    :param reuse_subset_cache: Read an existing subset cache when present.
    :param n_jobs: Parallel jobs for loading/resampling.
    :return: Tuple ``(paired, analyzers)`` where *paired* is the concatenated dataset
        with its segment bookkeeping and *analyzers* maps each condition to the
        analyser it was loaded from (for ``info``, montage and metadata).
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
        datasets = analyzers_to_datasets(analyzers)

        # Subsetting the time-domain dataset is what drives the trim applied to the
        # loaded wavelet cache (see _slice_wavelet_to_extent). The subject axis is left
        # alone: the concatenation matches participants across the conditions, so it
        # needs every recording each condition has.
        if n_channels is not None:
            datasets = {
                key: dataclasses.replace(ad, data=ad.data[:, :n_channels, :])
                for key, ad in datasets.items()
            }
        if n_times is not None:
            datasets = {
                key: dataclasses.replace(ad, data=ad.data[:, :, :n_times])
                for key, ad in datasets.items()
            }

        wavelets = compute_wavelet_datasets(
            datasets,
            analyzers,
            freqs,
            representation,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=reuse_wavelets,
            experiment_name=experiment_name,
            keep_frequency_dim=True,
            reshape_frequency_dim=True,
            subset_cache_dir=subset_cache_dir,
            reuse_subset_cache=reuse_subset_cache,
            n_jobs=n_jobs,
        )

        # When a subset was trimmed off the source-of-truth cache the result is a view
        # into that much larger tensor; copy it out so the big array is released before
        # the next condition is read. Tested on the base being *larger*, not merely
        # present: a subset-cache hit also arrives with a base.
        track = wavelets[label]
        base = track.data.base
        if base is not None and getattr(base, "size", 0) > track.data.size:
            track = dataclasses.replace(track, data=np.ascontiguousarray(track.data))

        tracks[condition] = track
        participants[condition] = participant_labels(
            analyzer.filtered_df, track.data.shape[0]
        )
        onsets[condition] = analyzer.stimulus_onsets
        analyzers_by_condition[condition] = analyzer
        _logger.info(
            f"[{label}] track {track.data.shape}, "
            f"{len(participants[condition])} participant(s)."
        )
        del wavelets, datasets

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
    subset_cache_dir: Path | None = None,
    reuse_subset_cache: bool = True,
    n_jobs: int = -1,
) -> tuple[PooledConditionSubjects, dict[ConditionVariants, "EEGSummarizedAnalyzer"]]:
    """Load each condition's wavelets and stack the participant-matched recordings.

    The one-call path to the subject-axis joined dataset — the layout
    :attr:`~src.definitions.fields.ConditionVariants.JOINED` names. It loads each
    condition exactly as any single-condition workflow does, **reusing the existing
    per-condition wavelet caches**, then hands both to
    :func:`~src.analysis.condition_tracks.pool_condition_subjects`. Nothing upstream is
    re-run or re-aligned, and — unlike asking
    :func:`compute_wavelet_datasets` for a ``Joined_*`` label directly — no second copy
    of the cache is written for the pooled cohort.

    Contrast with :func:`load_paired_condition_wavelets`, which pools the same cohort
    along *time* instead, so each participant is one subject carrying both tracks.

    The two conditions are loaded and trimmed **one at a time**, so peak memory stays at
    one condition's cache rather than both. Pass *n_channels* / *n_times* to keep the
    pooled tensor small during exploration; there is no subject subset because dropping
    subjects before matching would break the pairing (slice
    :attr:`~src.analysis.condition_tracks.PooledConditionSubjects.data` afterwards
    instead, or use the participant bookkeeping to choose which pairs to keep).

    :param music_type: Music type to load (``MusicTypeVariants.ASSR`` for ASSR).
    :param exclusion_categories: Exclusion categories applied to both conditions.
    :param freqs: Morlet frequencies (Hz).
    :param wavelet_dir: Wavelet cache directory, e.g.
        ``resolve_wavelet_dir(None, experiment_name) / "broadband"``.
    :param experiment_name: Experiment being analysed.
    :param representation: ``"power"`` or ``"phase"``.
    :param conditions: Conditions to pool, in subject-axis block order.
    :param n_channels: Keep only the first *n_channels* channels. ``None`` keeps all.
    :param n_times: Keep only the first *n_times* time samples. ``None`` keeps all.
    :param reuse_wavelets: Reuse the cached transform when present.
    :param subset_cache_dir: Optional notebook-level subset cache, forwarded per
        condition to :func:`compute_wavelet_datasets`. Strongly worth setting here: the
        two source-of-truth caches are the dominant cost of a joined load, and with a
        cache hit on both conditions neither is opened. Because the cache is keyed by
        condition and extent — not by the workflow that wrote it — the entries this
        fills in are the same ones a single-condition notebook reads. See
        :func:`resolve_notebook_wavelet_cache_dir`.
    :param reuse_subset_cache: Read an existing subset cache when present.
    :param n_jobs: Parallel jobs for loading/resampling.
    :return: Tuple ``(pooled, analyzers)`` where *pooled* is the subject-axis joined
        dataset with its per-subject participant/condition bookkeeping and *analyzers*
        maps each condition to the analyser it was loaded from (for ``info``, montage
        and metadata).
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
        datasets = analyzers_to_datasets(analyzers)

        # Subset the time-domain dataset, which is what drives the trim applied to the
        # loaded wavelet cache (see _slice_wavelet_to_extent). The subject axis is left
        # alone: the pooling matches participants across the conditions, so it needs
        # every recording each condition has.
        if n_channels is not None:
            datasets = {
                key: dataclasses.replace(ad, data=ad.data[:, :n_channels, :])
                for key, ad in datasets.items()
            }
        if n_times is not None:
            datasets = {
                key: dataclasses.replace(ad, data=ad.data[:, :, :n_times])
                for key, ad in datasets.items()
            }

        wavelets = compute_wavelet_datasets(
            datasets,
            analyzers,
            freqs,
            representation,
            wavelet_dir=wavelet_dir,
            reuse_wavelets=reuse_wavelets,
            experiment_name=experiment_name,
            keep_frequency_dim=True,
            reshape_frequency_dim=True,
            subset_cache_dir=subset_cache_dir,
            reuse_subset_cache=reuse_subset_cache,
            n_jobs=n_jobs,
        )

        # When a subset was trimmed off the source-of-truth cache the result is a
        # *view* into that (much larger) tensor; copy it out so the big array can be
        # released before the next condition is read. The test is deliberately on the
        # base being larger, not merely present: a subset-cache hit also arrives with a
        # base, and copying a full-extent tensor to no purpose would double the peak.
        block = wavelets[label]
        base = block.data.base
        if base is not None and getattr(base, "size", 0) > block.data.size:
            block = dataclasses.replace(block, data=np.ascontiguousarray(block.data))
        tracks[condition] = block
        participants[condition] = participant_labels(
            analyzer.filtered_df, tracks[condition].data.shape[0]
        )
        onsets[condition] = analyzer.stimulus_onsets
        analyzers_by_condition[condition] = analyzer
        _logger.info(
            f"[{label}] block {tracks[condition].data.shape}, "
            f"{len(participants[condition])} participant(s)."
        )
        del wavelets, datasets

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
