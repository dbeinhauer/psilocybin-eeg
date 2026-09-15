"""
This module contains the EEGSummarizedAnalyzer class which encapsulates
data loading, persistence and high-level analysis of preprocessed EEG data
across subjects.

Computation helpers delegate to :mod:`src.analysis.isc` (pure numpy
functions) and adapter helpers delegate to
:mod:`src.analysis.data_representations` so that the same analysis routines
work on any data representation (raw channels, ICA activations, wavelet
amplitudes, mean responses, …).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import mne
import pandas as pd
from scipy.stats import zscore

from src.preprocessing.pipeline import DatasetHandler
from src.preprocessing.stimulus_alignment import (
    resolve_stimulus_marker,
    StimulusAligner,
    StimulusMarker,
    get_stimulus_onset_samples,
    apply_keep_segments_to_array,  # noqa: F401 — re-exported for caller convenience
)
from src.filtering.dataset_filter import DatasetFilter
from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ALIGNMENT_EXCLUSIONS,
    REAL_CONDITIONS,
    CoordinateSystems,
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    MusicTypeVariants,
    PreprocessedDataVariants,
    SingleDataMetadata,
)
from src.definitions.frequency import FREQUENCY_BANDS
from src.analysis.isc import (
    compute_loo_isc as _compute_loo_isc,
    compute_pairwise_isc as _compute_pairwise_isc,
    compute_pairwise_isc_per_feature as _compute_pairwise_isc_per_feature,
    compute_sliding_window_isc as _compute_sliding_window_isc,
)
from src.utils.logging_config import LoggerMixin


class EEGSummarizedAnalyzer(LoggerMixin):
    """
    Manages loading, persisting, and analysing preprocessed EEG data.

    Typical workflow
    ----------------
    1. Instantiate with an experiment name and coordinate system.
    2. Call :meth:`load_and_prepare_data` (or :meth:`load_data`) to populate
       :attr:`data` (shape: ``n_subjects × n_channels × n_times``) and :attr:`info`.
    3. Run any of the analysis helpers, e.g. :meth:`compute_loo_isc`.

    Attributes
    ----------
    dataset_handler : DatasetHandler
        Underlying handler used for I/O and metadata access.
    data : np.ndarray or None
        EEG data array of shape ``(n_subjects, n_channels, n_times)`` after loading.
    info : mne.Info or None
        MNE Info object taken from the first loaded file.
    filtered_df : pd.DataFrame or None
        Metadata DataFrame of the last applied filter / load operation.
        Includes :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX`
        after :meth:`load_and_prepare_data`, mapping
        each row to its subject index in :attr:`data`.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        experiment_name: ExperimentNames,
        coordinate_system: CoordinateSystems,
        music_types: List[MusicTypeVariants],
        conditions: List[ConditionVariants],
        exclusion_categories: List[ExclusionCategories],
    ) -> None:
        """
        :param experiment_name: Which experiment dataset to use.
        :param coordinate_system: Electrode coordinate system to use for loading.
        :param music_types: Music types to include (e.g. MusicTypeVariants.CLASSICAL).
        :param conditions: Experimental conditions to include (e.g. ConditionVariants.PLACEBO).
        :param exclusion_categories: Exclusion categories to filter out bad recordings.
        """
        self.dataset_handler = DatasetHandler(experiment_name, coordinate_system)
        self._experiment_name = experiment_name

        # Assign the filter parameters to instance variables for potential later use (e.g. in default save path generation).
        self.music_types = music_types
        self.conditions = conditions
        self.exclusion_categories = exclusion_categories

        self.filtered_df = DatasetFilter.filter_dataset_by_all_categories(
            self.dataset_handler.dataset_metadata,
            self.dataset_handler.excluded_participants_metadata,
            self.music_types,
            self.conditions,
            self.exclusion_categories,
        )

        self.data: Optional[np.ndarray] = None
        self.info: Optional[mne.Info] = None
        self.resample_freq: Optional[float] = None

        # Stimulus marker for this experiment (label + marker→onset offset, e.g.
        # ``fam+`` for ASSR), or ``None`` for experiments without stimulus
        # annotations. When set, the onset sample positions (identical across
        # subjects by construction) are extracted during loading and saved next to
        # the concatenated data.
        self._stimulus_marker: Optional[StimulusMarker] = resolve_stimulus_marker(
            experiment_name
        )
        self.stimulus_onsets: Optional[np.ndarray] = None

        # Whether the two conditions are pooled along the time axis rather than the
        # subject axis: each participant becomes one subject whose recording is their
        # Placebo track followed by their Psilocybin track. Set for
        # ConditionVariants.JOINED_TRACKS only.
        self.concatenates_condition_tracks: bool = (
            ConditionVariants.JOINED_TRACKS in self.conditions
        )
        # For such a dataset, the sample index where each condition's segment starts,
        # plus the final end (len(REAL_CONDITIONS) + 1 entries). ``None`` otherwise.
        self.segment_boundaries: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Data loading                                                         #
    # ------------------------------------------------------------------ #

    def load_and_prepare_data(
        self,
        resample_freq: float = 250.0,
        n_jobs: int = -1,
    ) -> tuple[np.ndarray, mne.Info]:
        """
        Filter the dataset, load EEG files, resample, stack into a numpy array.

        Results are stored in :attr:`data` (``n_subjects × n_channels × n_times``)
        and :attr:`info`.

        For :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` the two
        conditions are pooled along the time axis instead, so each participant yields
        **one** subject whose recording is their Placebo track followed by their
        Psilocybin track (see :meth:`_prepare_condition_track_data`).

        :param resample_freq: Target sampling frequency in Hz (default 250).
        :param n_jobs: Number of parallel jobs for resampling (-1 = all CPUs).
        :return: Tuple ``(data, info)`` where *data* is the loaded EEG array and *info* is the MNE Info object.
        """

        self.logger.info(
            f"Loading {len(self.filtered_df)} recording(s) "
            f"(condition={[c.value for c in self.conditions]}, "
            f"music={[m.value for m in self.music_types]})."
        )

        if self.concatenates_condition_tracks:
            return self._prepare_condition_track_data(resample_freq, n_jobs)

        raws: list[mne.io.Raw] = []
        axis0_indices: list[int] = []
        for axis0_index, (_, row) in enumerate(self.filtered_df.iterrows()):
            filename = row[SingleDataMetadata.FILENAME]
            raw = self.dataset_handler.load_data_file(
                filename,
                is_processed=True,
                processed_data_type=PreprocessedDataVariants.RAW_CROPPED,
                preload=True,
            ).pick(["eeg"])
            raws.append(raw.resample(resample_freq, n_jobs=n_jobs))
            axis0_indices.append(axis0_index)

        self.filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = axis0_indices

        # Store MNE Info from first file (before any resampling changes it)
        self._refresh_info(raws[0].info)
        self.resample_freq = resample_freq

        # Stimulus onsets, on the same sample grid as the concatenated data. The
        # alignment makes the *onsets* identical across subjects, so one recording
        # is representative — but its annotations must be read with its own offset,
        # since that is what the alignment removed to put the onsets there.
        self.stimulus_onsets = self._extract_stimulus_onsets(
            raws[0], self.filtered_df.iloc[0][SingleDataMetadata.FILENAME]
        )

        self.data = np.array([r.get_data() for r in raws])  # (n_subj, n_ch, n_times)
        self.logger.info(f"Data array shape: {self.data.shape}")

        return self.data, self.info

    def _prepare_condition_track_data(
        self,
        resample_freq: float,
        n_jobs: int,
    ) -> tuple[np.ndarray, mne.Info]:
        """
        Build the time-concatenated array for
        :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS`.

        Each participant contributes one subject whose recording is their tracks laid
        end to end in :data:`~src.definitions.fields.REAL_CONDITIONS` order. The
        alignment was fitted once over every recording of both conditions, so the two
        tracks already share a time base and carry the same stimuli; this only selects
        and concatenates.

        :attr:`filtered_df` keeps one row per *recording* (so filenames stay
        traceable), but
        :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX`
        is the **subject** index, so the two rows of a participant share one value.
        :attr:`segment_boundaries` records where each condition's segment starts and
        ends, and :attr:`stimulus_onsets` holds the first segment's onsets — use
        :class:`~src.analysis.condition_tracks.PairedConditionTracks` for the
        per-condition view.

        :param resample_freq: Target sampling frequency in Hz.
        :param n_jobs: Number of parallel jobs for resampling.
        :return: Tuple ``(data, info)``.
        :raises RuntimeError: If a participant does not contribute exactly one
            recording per condition, which
            :meth:`~src.filtering.dataset_filter.DatasetFilter.restrict_to_complete_pairs`
            should already have guaranteed.
        """
        conditions = list(REAL_CONDITIONS)
        # Preserve the block ordering the filter produced: the first block's order is
        # the participant order of the concatenated subject axis.
        by_participant: dict[str, dict[ConditionVariants, str]] = {}
        order: list[str] = []
        for _, row in self.filtered_df.iterrows():
            participant = str(row[SingleDataMetadata.PARTICIPANT_ID])
            condition = row[SingleDataMetadata.CONDITION]
            if not isinstance(condition, ConditionVariants):
                condition = ConditionVariants(condition)
            if participant not in by_participant:
                by_participant[participant] = {}
                order.append(participant)
            by_participant[participant][condition] = row[SingleDataMetadata.FILENAME]

        incomplete = [
            participant
            for participant in order
            if any(c not in by_participant[participant] for c in conditions)
        ]
        if incomplete:
            raise RuntimeError(
                f"Participant(s) {incomplete} do not contribute a recording to every "
                f"condition {[c.value for c in conditions]}; a JOINED_TRACKS dataset "
                "requires complete pairs."
            )

        # Load condition by condition so every subject's segment for a given condition
        # is built the same way, then concatenate per participant.
        per_condition_arrays: dict[ConditionVariants, list[np.ndarray]] = {}
        first_raw: Optional[mne.io.Raw] = None
        for condition in conditions:
            arrays: list[np.ndarray] = []
            for participant in order:
                raw = self.dataset_handler.load_data_file(
                    by_participant[participant][condition],
                    is_processed=True,
                    processed_data_type=PreprocessedDataVariants.RAW_CROPPED,
                    preload=True,
                ).pick(["eeg"])
                raw = raw.resample(resample_freq, n_jobs=n_jobs)
                if first_raw is None:
                    first_raw = raw
                arrays.append(raw.get_data())
            per_condition_arrays[condition] = arrays

        # Every recording of one condition shares that condition's aligned length.
        segment_lengths = [
            per_condition_arrays[condition][0].shape[-1] for condition in conditions
        ]
        for condition, length in zip(conditions, segment_lengths):
            mismatched = [
                order[index]
                for index, array in enumerate(per_condition_arrays[condition])
                if array.shape[-1] != length
            ]
            if mismatched:
                raise RuntimeError(
                    f"Recordings of condition {condition.value} have differing lengths "
                    f"(participant(s) {mismatched} differ from {length} samples); the "
                    "within-condition alignment is inconsistent."
                )

        self.data = np.stack(
            [
                np.concatenate(
                    [per_condition_arrays[c][index] for c in conditions], axis=-1
                )
                for index in range(len(order))
            ]
        )

        edges = [0]
        for length in segment_lengths:
            edges.append(edges[-1] + int(length))
        self.segment_boundaries = np.asarray(edges, dtype=int)

        # Subject index is the participant's position, so a participant's two rows
        # share it.
        participant_index = {p: index for index, p in enumerate(order)}
        self.filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX] = [
            participant_index[str(row[SingleDataMetadata.PARTICIPANT_ID])]
            for _, row in self.filtered_df.iterrows()
        ]

        assert first_raw is not None
        self._refresh_info(first_raw.info)
        self.resample_freq = resample_freq

        # Onsets of the *first* segment, on the concatenated grid (its segment starts
        # at 0, so no shift is needed). Later segments have their own onset grids;
        # PairedConditionTracks resolves them per condition.
        self.stimulus_onsets = self._extract_stimulus_onsets(
            first_raw, by_participant[order[0]][conditions[0]]
        )

        self.logger.info(
            f"Concatenated {len(order)} participant-matched track pair(s): "
            f"shape {self.data.shape}, segment lengths {segment_lengths}, "
            f"boundaries {self.segment_boundaries.tolist()}."
        )
        return self.data, self.info

    def _extract_stimulus_onsets(
        self, raw: mne.io.Raw, filename: str
    ) -> Optional[np.ndarray]:
        """
        Extract the stimulus-onset sample positions mapping onto the time axis of
        :attr:`data`.

        The stimulus alignment splices every recording so that each onset lands at
        the same sample index in all participants, so a single (here resampled)
        recording is representative of the whole group.

        :param raw: A loaded (resampled) recording of the current group.
        :param filename: Filename *raw* was loaded from. The marker→onset offset is
            per recording, so reading another recording's annotations with this
            one's offset would place the onsets tens of milliseconds off.
        :return: Sorted array of onset sample indices, or ``None`` when the
            experiment has no stimulus annotations / none are present.
        """
        if self._stimulus_marker is None:
            return None

        onsets = get_stimulus_onset_samples(
            raw,
            self._stimulus_marker.label,
            self._stimulus_marker.onset_offset_for(filename),
        )
        if len(onsets) == 0:
            self.logger.warning(
                f"No '{self._stimulus_marker.label}' annotations found in the "
                "loaded recordings; skipping stimulus-onset extraction."
            )
            return None
        return onsets

    def load_pre_alignment_data(
        self,
        resample_freq: float = 250.0,
        n_jobs: int = -1,
        stimulus_label: Optional[str] = None,
        keep_tail_sec: float = 0.1,
        pre_window_sec: Optional[float] = None,
        post_window_sec: Optional[float] = None,
    ) -> tuple[
        list[np.ndarray],
        list[list[tuple[int, int]]],
        StimulusAligner,
        Optional[mne.Info],
    ]:
        """
        Load per-subject ``RAW_AFTER_ICA`` data for pre-wavelet stimulus alignment.

        For stimulus-based experiments (e.g. ASSR), wavelets computed on
        stimulus-trimmed data suffer from edge artifacts at every splice point.
        This method returns the full continuous recording per subject so that
        the wavelet transform can be applied *before* trimming.  After wavelet
        computation, pass each subject's wavelet array and the corresponding
        entry in :attr:`~src.preprocessing.stimulus_alignment.StimulusAligner.keep_segments`
        to :func:`~src.preprocessing.stimulus_alignment.apply_keep_segments_to_array`
        to reproduce the stimulus alignment on the wavelet output.

        Typical usage::

            arrays, segments, aligner, info = analyzer.load_pre_alignment_data(resample_freq=250)
            # arrays[i]: (n_channels, n_times_i)  — variable lengths
            wavelet_aligned = []
            for arr, keep in zip(arrays, segments):
                ad = from_array(arr[np.newaxis], sfreq=250, ...)
                wd = to_wavelet_power(ad, freqs, keep_frequency_dim=True)
                # wd.data: (1, n_ch, n_freqs, n_times_i)
                trimmed = apply_keep_segments_to_array(wd.data[0], keep)
                # trimmed: (n_ch, n_freqs, aligner.total_length)
                wavelet_aligned.append(trimmed)
            data_4d = np.stack(wavelet_aligned)  # (n_subj, n_ch, n_freqs, n_times_aligned)

        :param resample_freq: Target sampling frequency in Hz (default 250).
            Resampling is applied before constructing the aligner so that the
            keep-segments are expressed on the same sample grid as the returned
            arrays.
        :param n_jobs: Parallel jobs for resampling (``-1`` = all CPUs).
        :param stimulus_label: Annotation label marking stimulus onsets.
            Defaults to the experiment's registered label (e.g. ``"fam+"`` for
            ASSR).
        :param keep_tail_sec: Continuous data preserved before each onset.
            Forwarded to :class:`~src.preprocessing.stimulus_alignment.StimulusAligner`.
        :param pre_window_sec: Cap on the window kept before the first onset.
            ``None`` keeps the per-subject shortest available lead-in.
        :param post_window_sec: Cap on the window kept after the last onset.
            ``None`` keeps the per-subject shortest available lead-out.
        :return: Tuple ``(arrays, segments, aligner, info)`` where

            * ``arrays`` is a list of ``(n_channels, n_times_i)`` numpy arrays — one per
              recording **selected by this analyser**, in its metadata order, variable
              length before trimming.
            * ``segments`` is the matching list of keep-segments; pass ``segments[i]``
              with
              :func:`~src.preprocessing.stimulus_alignment.apply_keep_segments_to_array`
              to trim subject *i*'s wavelet output. Use these rather than
              ``aligner.keep_segments``, which is indexed by the **whole** alignment
              cohort, not by this analyser's selection.
            * ``aligner`` is the fitted :class:`~src.preprocessing.stimulus_alignment.StimulusAligner`,
              fitted over every recording of both conditions.
            * ``info`` is the MNE Info object from the first loaded recording.

        :raises ValueError: If the experiment has no registered stimulus label
            and none is supplied via *stimulus_label*.
        """
        registered_label = (
            self._stimulus_marker.label if self._stimulus_marker is not None else None
        )
        label = stimulus_label if stimulus_label is not None else registered_label
        if label is None:
            raise ValueError(
                f"Experiment '{self._experiment_name.value}' has no registered "
                "stimulus label. Pass stimulus_label explicitly."
            )

        # The aligner must be fitted on the SAME group the stored crops were fitted on
        # — every recording of both conditions — not on this analyser's selection.
        # Fitting it on a subset would give different group minima, so the wavelet
        # cache would describe a different splice than the RAW_CROPPED it is supposed
        # to match, and the two conditions would drift apart again.
        cohort_df = DatasetFilter.filter_dataset_by_all_categories(
            self.dataset_handler.dataset_metadata,
            self.dataset_handler.excluded_participants_metadata,
            self.music_types,
            list(REAL_CONDITIONS),
            list(ALIGNMENT_EXCLUSIONS.get(self._experiment_name, ())),
        )
        cohort_filenames = [
            row[SingleDataMetadata.FILENAME] for _, row in cohort_df.iterrows()
        ]
        # The offset always comes from the experiment's registered marker: an
        # explicit label override selects *which* annotation to read, not how it
        # relates in time to the stimulus. It is resolved per recording, since the
        # marker error differs between recordings.
        onset_offsets = (
            self._stimulus_marker.offsets_for(cohort_filenames)
            if self._stimulus_marker is not None
            else [0.0] * len(cohort_filenames)
        )

        self.logger.info(
            f"Loading {len(cohort_filenames)} RAW_AFTER_ICA recording(s) to fit the "
            f"alignment (resample → {resample_freq} Hz); "
            f"{len(self.filtered_df)} of them are in this analyser's selection."
        )

        raws: list[mne.io.Raw] = []
        for filename in cohort_filenames:
            raw = (
                self.dataset_handler.load_data_file(
                    filename,
                    is_processed=True,
                    processed_data_type=PreprocessedDataVariants.RAW_AFTER_ICA,
                    preload=True,
                )
                .pick(["eeg"])
                .resample(resample_freq, n_jobs=n_jobs)
            )
            raws.append(raw)

        # Build the alignment plan from the resampled recordings so that the
        # keep-segments are expressed on the resampled sample grid.
        onset_samples = [
            get_stimulus_onset_samples(raw, label, offset)
            for raw, offset in zip(raws, onset_offsets)
        ]
        recording_lengths = [raw.n_times for raw in raws]
        aligner = StimulusAligner(
            onset_samples,
            recording_lengths,
            sfreq=resample_freq,
            keep_tail_sec=keep_tail_sec,
            pre_window_sec=pre_window_sec,
            post_window_sec=post_window_sec,
        )

        # Return only this analyser's recordings, in ITS metadata order, together with
        # their keep-segments — the plan is group-wide, the data is not.
        cohort_index = {name: index for index, name in enumerate(cohort_filenames)}
        selected = []
        for _, row in self.filtered_df.iterrows():
            filename = row[SingleDataMetadata.FILENAME]
            if filename not in cohort_index:
                raise RuntimeError(
                    f"Recording {filename} is selected for analysis but is not in the "
                    "alignment cohort, so it was never aligned. Widen "
                    "ALIGNMENT_EXCLUSIONS or drop it from the analysis."
                )
            selected.append(cohort_index[filename])

        arrays = [raws[index].get_data() for index in selected]
        segments = [aligner.keep_segments[index] for index in selected]

        self._refresh_info(raws[0].info)
        self.resample_freq = resample_freq

        self.logger.info(
            f"Pre-alignment load complete: {len(arrays)} selected subject(s) of "
            f"{len(raws)} in the alignment cohort, "
            f"aligned total_length={aligner.total_length} samples "
            f"(common stimulus count {aligner.common_count})."
        )
        return arrays, segments, aligner, self.info

    def _refresh_info(self, info: mne.Info) -> None:
        """Store an mne.Info copy taken from the first loaded raw object."""
        self.info = info

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_data(
        self,
        save_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        overwrite: bool = True,
    ) -> Path:
        """
        Save :attr:`data` as a ``.npy`` file.

        When :attr:`filtered_df` is available, a metadata CSV is also stored
        alongside the concatenated data, preserving row ordering and the
        :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX` mapping.

        If *save_path* is not given the file is placed in the project's
        ``processed/<experiment>/concatenated/`` directory with an auto-generated
        name derived from the last filter applied.

        The metadata CSV is written to *metadata_path* when given.  When
        omitted it defaults to the project's concatenated directory next to
        the data array (see :meth:`_default_metadata_save_path`).

        :param save_path: Explicit destination path for the ``.npy`` file.
        :param metadata_path: Explicit destination path for the metadata CSV.
            Defaults to the project's concatenated directory.
        :param overwrite: Whether to overwrite existing files.
        :return: The path where the data file was written.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        if save_path is None:
            save_path = self._default_save_path()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.exists() and not overwrite:
            self.logger.warning(f"File already exists and overwrite=False: {save_path}")
            return save_path

        np.save(save_path, self.data)

        if self.stimulus_onsets is not None:
            onsets_path = self._stimulus_onsets_path(save_path)
            np.save(onsets_path, self.stimulus_onsets)
            self.logger.info(
                f"Stimulus onsets saved to {onsets_path} "
                f"({len(self.stimulus_onsets)} onsets)."
            )

        if self.segment_boundaries is not None:
            boundaries_path = self._segment_boundaries_path(save_path)
            np.save(boundaries_path, self.segment_boundaries)
            self.logger.info(
                f"Condition-track segment boundaries saved to {boundaries_path} "
                f"({self.segment_boundaries.tolist()})."
            )

        if self.filtered_df is not None:
            resolved_metadata_path = (
                Path(metadata_path)
                if metadata_path is not None
                else save_path.with_suffix(".metadata.csv")
            )
            resolved_metadata_path.parent.mkdir(parents=True, exist_ok=True)
            self.filtered_df.to_csv(resolved_metadata_path, index=True)
            self.logger.info(f"Metadata saved to {resolved_metadata_path}")

        self.logger.info(f"Data saved to {save_path}  (shape={self.data.shape})")

        return save_path

    def load_data(
        self,
        load_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        info_filename: Optional[str] = None,
        resample_freq: float = 250.0,
    ) -> tuple[np.ndarray, mne.Info]:
        """
        Load a previously saved ``.npy`` data array from disk.

        If a metadata CSV exists it is loaded into :attr:`filtered_df`,
        including the :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX` mapping.  The CSV is read from
        *metadata_path* when provided; otherwise the default path in the
        project's concatenated directory is tried (see
        :meth:`_default_metadata_save_path`).

        :param load_path: Path to the ``.npy`` file.
        :param metadata_path: Explicit path to the metadata CSV.  Defaults to
            the project's concatenated directory.
        :param info_filename: Optional filename from the dataset metadata to use for
            loading ``mne.Info``. When provided, the Info object is populated from
            the corresponding processed file.
        :param resample_freq: Resampling frequency of the stored data.
        :return: Tuple ``(data, info)`` where *data* is the loaded EEG array and *info* is the MNE Info object.
        :raises FileNotFoundError: If *load_path* does not exist.
        """
        if load_path is None:
            load_path = self._default_save_path()

        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Data file not found: {load_path}")

        self.data = np.load(load_path)
        self.logger.info(f"Data loaded from {load_path}  (shape={self.data.shape})")

        onsets_path = self._stimulus_onsets_path(load_path)
        if onsets_path.exists():
            self.stimulus_onsets = np.load(onsets_path)
            self.logger.info(
                f"Stimulus onsets loaded from {onsets_path} "
                f"({len(self.stimulus_onsets)} onsets)."
            )

        boundaries_path = self._segment_boundaries_path(load_path)
        if boundaries_path.exists():
            self.segment_boundaries = np.load(boundaries_path)
            self.logger.info(
                f"Condition-track segment boundaries loaded from {boundaries_path} "
                f"({self.segment_boundaries.tolist()})."
            )

        if metadata_path is not None:
            candidate_metadata_paths = [Path(metadata_path)]
        else:
            # Derive metadata path from the resolved load_path (same directory and
            # stem), mirroring :meth:`save_data`, which writes ``<stem>.metadata.csv``.
            # The bare ``<stem>.csv`` is kept as a fallback for sidecars written
            # before the suffixes were aligned.
            candidate_metadata_paths = [
                load_path.with_suffix(".metadata.csv"),
                load_path.with_suffix(".csv"),
            ]
        resolved_metadata_path = next(
            (p for p in candidate_metadata_paths if p.exists()), None
        )
        if resolved_metadata_path is not None:
            self.filtered_df = pd.read_csv(resolved_metadata_path, index_col=0)
            self._normalize_filtered_df_columns()
            self.logger.info(f"Metadata loaded from {resolved_metadata_path}")
        else:
            # Without the sidecar, ``filtered_df`` keeps the freshly-filtered rows
            # from ``__init__``, which carry no CONCATENATED_PERSON_INDEX — any
            # subject-index -> participant lookup would silently degrade.
            self.logger.warning(
                "No metadata sidecar found for "
                f"{load_path.name} (looked for "
                f"{', '.join(p.name for p in candidate_metadata_paths)}); "
                "filtered_df has no CONCATENATED_PERSON_INDEX mapping."
            )

        if info_filename is not None:
            raw = self.dataset_handler.load_data_file(
                info_filename,
                is_processed=True,
                processed_data_type=PreprocessedDataVariants.RAW_CROPPED,
                preload=False,
            )
            self._refresh_info(raw.info)

        # Store the resampling frequency.
        self.resample_freq = resample_freq

        return self.data, self.info

    @staticmethod
    def _stimulus_onsets_path(data_path: Path) -> Path:
        """
        Build the stimulus-onsets file path for a concatenated data array.

        Same prefix as the data array, with the
        :attr:`~src.definitions.constants.ProjectPaths.STIMULUS_ONSETS_SUFFIX` suffix
        (e.g. ``Placebo_ASSR.npy`` -> ``Placebo_ASSR.stimulus_onsets.npy``).
        """
        return data_path.parent / (data_path.stem + ProjectPaths.STIMULUS_ONSETS_SUFFIX)

    @staticmethod
    def _segment_boundaries_path(data_path: Path) -> Path:
        """
        Build the segment-boundaries file path for a concatenated data array.

        Same prefix as the data array, with the
        :attr:`~src.definitions.constants.ProjectPaths.SEGMENT_BOUNDARIES_SUFFIX`
        suffix (e.g. ``JoinedTracks_ASSR.npy`` ->
        ``JoinedTracks_ASSR.segment_boundaries.npy``). Only written for
        :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` datasets, and
        what lets the per-condition split be reconstructed from disk alone.
        """
        return data_path.parent / (
            data_path.stem + ProjectPaths.SEGMENT_BOUNDARIES_SUFFIX
        )

    def _default_metadata_save_path(self) -> Path:
        """
        Build the default path for the metadata CSV in the concatenated directory.

        The file is placed next to the default data array (see
        :meth:`_default_save_path`), with a ``.metadata.csv`` suffix.
        """
        data_path = self._default_save_path()
        return data_path.parent / f"{data_path.stem}.metadata.csv"

    def _normalize_filtered_df_columns(self) -> None:
        """
        Normalise loaded metadata column names back to enum keys when possible.

        Sidecar CSV round-trips can coerce enum column names to strings
        (e.g. ``SingleDataMetadata.FILENAME``), which breaks lookups expecting
        enum keys in existing workflows.
        """
        if self.filtered_df is None:
            return

        rename_map: dict[str, Enum] = {}
        for metadata_field in SingleDataMetadata:
            if metadata_field in self.filtered_df.columns:
                continue

            enum_repr = f"{metadata_field.__class__.__name__}.{metadata_field.name}"
            for column_variant in (enum_repr, metadata_field.value):
                if column_variant in self.filtered_df.columns:
                    rename_map[column_variant] = metadata_field
                    break

        if rename_map:
            self.filtered_df = self.filtered_df.rename(columns=rename_map)

    def _default_save_path(self) -> Path:
        """Build a default save path from the current filtered DataFrame."""
        if self.filtered_df is None or self.filtered_df.empty:
            return (
                ProjectPaths.PROCESSED_DATA_DIR
                / self._experiment_name.value
                / PreprocessedDataVariants.CONCATENATED.value
                / "eeg_data.npy"
            )

        # Try to derive a meaningful name from the first row.
        condition = self.conditions[0] if self.conditions else "unknown"
        music = self.music_types[0] if self.music_types else "unknown"
        if hasattr(condition, "value"):
            condition = condition.value
        if hasattr(music, "value"):
            music = music.value
        name = f"{condition}_{music}.npy"
        return (
            ProjectPaths.PROCESSED_DATA_DIR
            / self._experiment_name.value
            / PreprocessedDataVariants.CONCATENATED.value
            / name
        )

    # ------------------------------------------------------------------ #
    #  Pre-processing helpers                                               #
    # ------------------------------------------------------------------ #

    def normalize(self, axis: int = 2) -> None:
        """
        Apply z-score normalization to :attr:`data` in-place.

        :param axis: Axis along which to compute the z-score (default 2 = time axis).
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")
        self.data = zscore(self.data, axis=axis)
        self.logger.info(f"Z-score normalisation applied along axis={axis}.")

    # ------------------------------------------------------------------ #
    #  Conversion to AnalysisData                                           #
    # ------------------------------------------------------------------ #

    def to_analysis_data(self, label: Optional[str] = None) -> "AnalysisData":  # noqa: F821
        """
        Wrap the loaded data in an :class:`~src.analysis.data_representations.AnalysisData`
        container for use with the generic analysis and visualisation pipeline.

        :param label: Human-readable label.  Defaults to the first music type value.
        :return: ``AnalysisData`` with :attr:`DataRepresentation.TIME_DOMAIN`.
        :raises RuntimeError: If no data has been loaded yet.
        """
        from src.analysis.data_representations import AnalysisData, DataRepresentation

        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")
        if label is None:
            music = self.music_types[0].value if self.music_types else "unknown"
            label = music
        sfreq = self.resample_freq or (
            self.info["sfreq"] if self.info is not None else 250.0
        )
        ch_names = list(self.info["ch_names"]) if self.info is not None else None
        return AnalysisData(
            data=self.data.copy(),
            sfreq=sfreq,
            representation=DataRepresentation.TIME_DOMAIN,
            label=label,
            feature_names=ch_names,
            info=self.info,
        )

    def load_and_prepare_ica_data(
        self,
        resample_freq: float = 250.0,
        n_jobs: int = -1,
        label: Optional[str] = None,
    ) -> "AnalysisData":  # noqa: F821
        """
        Load ICA component activations for all filtered recordings and
        return as :class:`~src.analysis.data_representations.AnalysisData`.

        :param resample_freq: Target sampling frequency in Hz.
        :param n_jobs: Number of parallel jobs for resampling.
        :param label: Human-readable label for plots.
        :return: ``AnalysisData`` with ``DataRepresentation.ICA_ACTIVATIONS``.
        """
        from src.analysis.data_representations import (
            AnalysisData,
            DataRepresentation,
            extract_ica_activations,
        )

        self.logger.info(
            f"Loading ICA activations for {len(self.filtered_df)} recording(s)."
        )
        raws, icas = [], []
        for _, row in self.filtered_df.iterrows():
            filename = row[SingleDataMetadata.FILENAME]
            raw = (
                self.dataset_handler.load_data_file(
                    filename,
                    is_processed=True,
                    processed_data_type=PreprocessedDataVariants.RAW_CROPPED,
                    preload=True,
                )
                .pick(["eeg"])
                .resample(resample_freq, n_jobs=n_jobs)
            )
            ica = self.dataset_handler.load_data_file(
                filename,
                is_processed=True,
                processed_data_type=PreprocessedDataVariants.ICA_COMPONENTS,
            )
            raws.append(raw)
            icas.append(ica)

        data = extract_ica_activations(raws, icas, sfreq=resample_freq)
        n_components = data.shape[1]
        if label is None:
            music = self.music_types[0].value if self.music_types else "unknown"
            label = f"ICA ({music})"

        return AnalysisData(
            data=data,
            sfreq=resample_freq,
            representation=DataRepresentation.ICA_ACTIVATIONS,
            label=label,
            feature_names=[f"IC{i}" for i in range(n_components)],
        )

    # ------------------------------------------------------------------ #
    #  Analysis — internal helpers (delegate to src.analysis.isc)            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_loo_isc_from_data(
        data: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate to :func:`src.analysis.isc.compute_loo_isc`."""
        return _compute_loo_isc(data)

    @staticmethod
    def _compute_sliding_window_isc_from_data(
        data: np.ndarray,
        window_sec: float,
        step_sec: float,
        sfreq: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Delegate to :func:`src.analysis.isc.compute_sliding_window_isc`."""
        return _compute_sliding_window_isc(data, window_sec, step_sec, sfreq)

    # ------------------------------------------------------------------ #
    #  Analysis — public methods                                            #
    # ------------------------------------------------------------------ #

    def compute_loo_isc(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute leave-one-out (LOO) Inter-Subject Correlation (ISC) for every channel.

        For each subject *s* the mean signal of all *other* subjects is computed, and
        the Pearson correlation between subject *s* and that mean is recorded for each
        channel.

        :return: Tuple ``(loo_isc, mean_loo_isc)`` where

            * ``loo_isc`` has shape ``(n_subjects, n_channels)`` — per-subject ISC.
            * ``mean_loo_isc`` has shape ``(n_channels,)`` — mean ISC across subjects.

        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        n_subjects, n_channels, _ = self.data.shape
        self.logger.info(
            f"Computing LOO-ISC for {n_subjects} subject(s), {n_channels} channel(s)."
        )
        loo_isc, mean_loo_isc = self._compute_loo_isc_from_data(self.data)
        self.logger.info("LOO-ISC computation complete.")
        return loo_isc, mean_loo_isc

    def compute_pairwise_isc(self) -> np.ndarray:
        """
        Compute pairwise Inter-Subject Correlation averaged across all channels.

        For every pair of subjects (i, j) the Pearson correlation is computed
        per channel and then averaged, yielding a single scalar per pair.

        :return: Symmetric matrix of shape ``(n_subjects, n_subjects)`` where
            entry ``[i, j]`` is the mean-across-channels Pearson *r* between
            subject *i* and subject *j*.  Diagonal is 1.0.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        n_subjects, n_channels, _ = self.data.shape
        self.logger.info(
            f"Computing pairwise ISC for {n_subjects} subject(s), "
            f"{n_channels} channel(s)."
        )
        pairwise = _compute_pairwise_isc(self.data)
        self.logger.info("Pairwise ISC computation complete.")
        return pairwise

    def compute_pairwise_isc_per_channel(self) -> np.ndarray:
        """
        Compute pairwise Inter-Subject Correlation for every channel separately.

        :return: Array of shape ``(n_channels, n_subjects, n_subjects)``.
            For each channel *c*, entry ``[c, i, j]`` is the Pearson *r* between
            subject *i* and subject *j* on that channel.  Diagonal is 1.0.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        n_subjects, n_channels, _ = self.data.shape
        self.logger.info(
            f"Computing per-channel pairwise ISC "
            f"({n_subjects} subjects × {n_channels} channels)."
        )
        pairwise = _compute_pairwise_isc_per_feature(self.data)
        self.logger.info("Per-channel pairwise ISC computation complete.")
        return pairwise

    def compute_sliding_window_isc(
        self,
        window_sec: float = 5.0,
        step_sec: float = 2.5,
        sfreq: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute time-resolved leave-one-out ISC using a sliding window.

        For each window position, the LOO-ISC is computed identically to
        :meth:`compute_loo_isc` but restricted to the samples within that
        window.  Results are averaged across subjects to yield a time course
        of ISC per channel.

        :param window_sec: Window length in seconds.
        :param step_sec: Step size (hop) in seconds.
        :param sfreq: Sampling frequency.  If ``None``, taken from
            ``self.info['sfreq']``.
        :return: Tuple ``(isc_timecourse, window_times)`` where

            * ``isc_timecourse`` has shape ``(n_windows, n_channels)`` —
              mean LOO-ISC in each window.
            * ``window_times`` has shape ``(n_windows,)`` — centre time (in
              seconds) of each window.

        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Call load_and_prepare_data() first.")

        if sfreq is None:
            if self.info is None:
                raise RuntimeError("No MNE Info available; pass sfreq explicitly.")
            sfreq = self.info["sfreq"]

        n_windows = len(
            np.arange(
                0,
                self.data.shape[2] - int(round(window_sec * sfreq)) + 1,
                int(round(step_sec * sfreq)),
            )
        )
        self.logger.info(
            f"Sliding-window ISC: {n_windows} windows "
            f"(win={window_sec}s, step={step_sec}s, sfreq={sfreq} Hz)."
        )
        isc_timecourse, window_times = self._compute_sliding_window_isc_from_data(
            self.data, window_sec=window_sec, step_sec=step_sec, sfreq=sfreq
        )
        self.logger.info("Sliding-window ISC computation complete.")
        return isc_timecourse, window_times

    # ------------------------------------------------------------------ #
    #  Analysis — frequency-band helpers                                    #
    # ------------------------------------------------------------------ #

    def filter_to_band(
        self,
        l_freq: float,
        h_freq: float,
        sfreq: Optional[float] = None,
    ) -> np.ndarray:
        """
        Band-pass filter :attr:`data` and return the filtered copy.

        Uses MNE's FIR filter (Hamming window) applied independently to each
        subject's data matrix.

        :param l_freq: Low cutoff frequency in Hz.
        :param h_freq: High cutoff frequency in Hz.
        :param sfreq: Sampling frequency.  Falls back to :attr:`resample_freq`
            or ``self.info['sfreq']`` when ``None``.
        :return: Filtered copy of :attr:`data`, same shape
            ``(n_subjects, n_channels, n_times)``.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")
        if sfreq is None:
            sfreq = (
                self.resample_freq
                if self.resample_freq is not None
                else self.info["sfreq"]
            )

        filtered = np.empty_like(self.data, dtype=float)
        for s in range(self.data.shape[0]):
            filtered[s] = mne.filter.filter_data(
                self.data[s].astype(float),
                sfreq=sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                method="fir",
                fir_window="hamming",
                verbose=False,
            )
        return filtered

    def compute_band_isc(
        self,
        bands: Optional[dict[str, tuple[float, float]]] = None,
        sfreq: Optional[float] = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute LOO-ISC separately for each frequency band.

        For each band the data is band-pass filtered first, then LOO-ISC is
        computed identically to :meth:`compute_loo_isc`.

        :param bands: Mapping of band name to ``(l_freq, h_freq)`` in Hz.
            Defaults to :data:`FREQUENCY_BANDS`
            (delta / theta / alpha / beta / gamma).
        :param sfreq: Sampling frequency override.  Falls back to
            :attr:`resample_freq` or ``self.info['sfreq']``.
        :return: Dict mapping each band name to
            ``(loo_isc, mean_loo_isc)`` with shapes
            ``(n_subjects, n_channels)`` and ``(n_channels,)``.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")
        if bands is None:
            bands = FREQUENCY_BANDS

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (l_freq, h_freq) in bands.items():
            self.logger.info(f"Band ISC — {name} [{l_freq}–{h_freq} Hz].")
            filtered = self.filter_to_band(l_freq, h_freq, sfreq)
            results[name] = self._compute_loo_isc_from_data(filtered)
        self.logger.info("Band ISC computation complete.")
        return results

    def compute_band_sliding_window_isc(
        self,
        bands: Optional[dict[str, tuple[float, float]]] = None,
        window_sec: float = 5.0,
        step_sec: float = 2.5,
        sfreq: Optional[float] = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute time-resolved LOO-ISC via a sliding window, separated by
        frequency band.

        Each band is band-pass filtered first; then :meth:`compute_sliding_window_isc`
        logic is applied inside each window.

        :param bands: Mapping of band name to ``(l_freq, h_freq)`` in Hz.
            Defaults to :data:`FREQUENCY_BANDS`.
        :param window_sec: Window length in seconds.
        :param step_sec: Step size (hop) in seconds.
        :param sfreq: Sampling frequency override.  Falls back to
            :attr:`resample_freq` or ``self.info['sfreq']``.
        :return: Dict mapping each band name to
            ``(isc_timecourse, window_times)`` with shapes
            ``(n_windows, n_channels)`` and ``(n_windows,)``.
        :raises RuntimeError: If no data has been loaded yet.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")
        if bands is None:
            bands = FREQUENCY_BANDS
        if sfreq is None:
            sfreq = (
                self.resample_freq
                if self.resample_freq is not None
                else self.info["sfreq"]
            )

        results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (l_freq, h_freq) in bands.items():
            self.logger.info(
                f"Band sliding-window ISC — {name} [{l_freq}–{h_freq} Hz], "
                f"win={window_sec}s, step={step_sec}s."
            )
            filtered = self.filter_to_band(l_freq, h_freq, sfreq)
            isc_tc, times = self._compute_sliding_window_isc_from_data(
                filtered, window_sec=window_sec, step_sec=step_sec, sfreq=sfreq
            )
            results[name] = (isc_tc, times)
        self.logger.info("Band sliding-window ISC computation complete.")
        return results
