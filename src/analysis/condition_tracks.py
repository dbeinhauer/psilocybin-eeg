"""
Join the Placebo and Psilocybin recordings of the same participants into one dataset.

Both joins here are **participant-matched**: the cohort is restricted to participants
that contribute a recording to each condition, so the design is balanced. They differ
only in the axis the conditions are pooled along, which is exactly the distinction the
two virtual conditions in :class:`~src.definitions.fields.ConditionVariants` name:

* :func:`concatenate_condition_tracks` pools along **time**
  (:attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS`) — each participant
  is one subject whose recording is their Placebo track followed by their Psilocybin
  track:

  .. code-block:: text

      subject k  ->  [ ---- Placebo track ---- | ---- Psilocybin track ---- ]
                     0                     T_pl                  T_pl + T_ps

  Because the subject axis holds one entry per participant and the mixing is estimated
  over the whole concatenated recording, a decomposition returns a *single* set of
  components covering both conditions. The contrast is then read off by slicing the
  sources back into the two segments (:meth:`PairedConditionTracks.condition_track`),
  so the components are identical between conditions by construction and any difference
  is in the sources, not in a component-matching step. The conditions keep their own
  time bases here and need not even have the same length.

* :func:`pool_condition_subjects` pools along the **subject** axis
  (:attr:`~src.definitions.fields.ConditionVariants.JOINED`) — every recording is one
  subject, so a participant appears once per condition and the mixing is estimated per
  recording. Read either condition's half back out with
  :meth:`PooledConditionSubjects.condition_subjects`. This one *does* require the two
  conditions to share a time base, which holds when the alignment was fitted over every
  recording of both conditions.

**Nothing upstream changes for either.** Both consume the *already computed*
per-condition datasets — typically the wavelet caches loaded by
:func:`~scripts.notebook_helpers.compute_wavelet_datasets` — so there is no
re-alignment, no new crop stage and no wavelet recomputation, and no new cache is
written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from src.analysis.data_representations import AnalysisData
from src.analysis.wavelet_ica import zscore_by_time
from src.definitions.fields import (
    REAL_CONDITIONS,
    ConditionVariants,
    SingleDataMetadata,
)

#: How the two tracks are standardised before being concatenated.
#:
#: * ``"per_condition"`` — z-score each track separately, then concatenate. This is the
#:   **default**: it matches how a single-condition workflow standardises its input, so
#:   each track enters the decomposition on the same footing as it would on its own.
#:   Note the consequence: every segment then has zero mean and unit variance per
#:   ``(subject, channel, frequency)`` by construction, so an overall power difference
#:   between the conditions is normalised away. What the comparison then tests is the
#:   temporal and spectral *structure* of each condition, not its amplitude.
#: * ``"joint"`` — concatenate first, then z-score each
#:   ``(subject, channel, frequency)`` series over the **whole** recording. A condition
#:   difference in mean power survives as a mean offset between the two segments. Use
#:   this when the amplitude contrast itself is the effect of interest.
#: * ``"none"`` — concatenate verbatim; use when the caller already standardised.
ZSCORE_MODES = ("per_condition", "joint", "none")


@dataclass(frozen=True)
class PairedConditionTracks:
    """A participant-matched, time-concatenated two-condition dataset.

    :param data: The concatenated dataset. Same axes as the inputs, with the last
        (time) axis holding every segment end to end — e.g.
        ``(n_pairs, n_channels, n_freqs, sum(segment_lengths))`` for wavelet power.
    :param participants: Participant label per subject index, in subject order.
    :param conditions: Segment order along the time axis.
    :param segment_lengths: Time-axis length of each segment, aligned with
        :attr:`conditions`.
    :param onsets: Per condition, stimulus-onset sample indices **local to that
        condition's segment**, or ``None`` when the experiment has no stimulus
        annotations. Use :meth:`condition_onsets` to get them on the concatenated axis.
    :param zscore_mode: Which entry of :data:`ZSCORE_MODES` produced :attr:`data`.
        Recorded because the choice changes what a between-condition difference means:
        ``"per_condition"`` (the default) normalises the amplitude contrast away, while
        ``"joint"`` keeps it.
    """

    data: AnalysisData
    participants: tuple[str, ...]
    conditions: tuple[ConditionVariants, ...]
    segment_lengths: tuple[int, ...]
    onsets: Mapping[ConditionVariants, Optional[np.ndarray]] = field(
        default_factory=dict
    )
    zscore_mode: str = "per_condition"

    @property
    def n_pairs(self) -> int:
        """Number of participants contributing both conditions."""
        return len(self.participants)

    @property
    def total_length(self) -> int:
        """Length of the concatenated time axis, in samples."""
        return int(sum(self.segment_lengths))

    @property
    def boundaries(self) -> tuple[int, ...]:
        """Sample index where each segment starts, plus the final end.

        ``boundaries[i]`` and ``boundaries[i + 1]`` bracket segment *i*, so the tuple
        has ``len(conditions) + 1`` entries.
        """
        edges = [0]
        for length in self.segment_lengths:
            edges.append(edges[-1] + int(length))
        return tuple(edges)

    def segment(self, condition: ConditionVariants) -> slice:
        """Return the slice of the concatenated time axis belonging to *condition*.

        :param condition: Condition whose segment is wanted.
        :return: A ``slice`` into the last axis.
        :raises KeyError: If *condition* is not one of the concatenated segments.
        """
        index = self._condition_index(condition)
        edges = self.boundaries
        return slice(edges[index], edges[index + 1])

    def condition_track(
        self,
        array: np.ndarray,
        condition: ConditionVariants,
    ) -> np.ndarray:
        """Recover one condition's track from any array on the concatenated time axis.

        This is the reverse of the concatenation and the point of the whole exercise:
        run the decomposition once over the joined recording, then split the result to
        compare the conditions. It is deliberately shape-agnostic — the only
        requirement is that the **last** axis is the concatenated time axis — so it
        works on the data itself ``(S, C, F, T)``, on IVA sources ``(S, K, F, T)``, on
        a single component's TF map ``(F, T)``, and so on.

        :param array: Any array whose last axis spans :attr:`total_length` samples.
        :param condition: Condition to extract.
        :return: A view into *array* covering only that condition's segment.
        :raises ValueError: If the last axis does not match :attr:`total_length`.
        :raises KeyError: If *condition* is not one of the concatenated segments.
        """
        array = np.asarray(array)
        if array.shape[-1] != self.total_length:
            raise ValueError(
                f"Array's last axis is {array.shape[-1]} samples but the concatenated "
                f"time axis is {self.total_length}; it does not come from this "
                "concatenation."
            )
        return array[..., self.segment(condition)]

    def condition_onsets(
        self,
        condition: ConditionVariants,
        *,
        absolute: bool = False,
    ) -> Optional[np.ndarray]:
        """Stimulus-onset samples for one condition.

        :param condition: Condition whose onsets are wanted.
        :param absolute: When ``True``, return positions on the **concatenated** time
            axis (i.e. shifted by the segment start), suitable for indexing arrays that
            were not split with :meth:`condition_track`. When ``False`` (default),
            return positions local to that condition's segment, matching the output of
            :meth:`condition_track`.
        :return: Sorted onset sample indices, or ``None`` when the experiment has none.
        :raises KeyError: If *condition* is not one of the concatenated segments.
        """
        self._condition_index(condition)
        onsets = self.onsets.get(condition)
        if onsets is None:
            return None
        onsets = np.asarray(onsets)
        return onsets + self.segment(condition).start if absolute else onsets

    def _condition_index(self, condition: ConditionVariants) -> int:
        """Position of *condition* among the concatenated segments."""
        if condition not in self.conditions:
            raise KeyError(
                f"Condition {condition} is not part of this concatenation; it holds "
                f"{[c.value for c in self.conditions]}."
            )
        return self.conditions.index(condition)

    @classmethod
    def from_analyzer(
        cls,
        analyzer,
        *,
        data: Optional[AnalysisData] = None,
        conditions: Sequence[ConditionVariants] = REAL_CONDITIONS,
    ) -> "PairedConditionTracks":
        """Rebuild the segment bookkeeping from a loaded ``JOINED_TRACKS`` analyser.

        The segment boundaries are persisted next to the concatenated array (see
        :attr:`~src.definitions.constants.ProjectPaths.SEGMENT_BOUNDARIES_SUFFIX`), so
        the per-condition split is recoverable from disk alone — including in a bare
        CLI run that never called :func:`concatenate_condition_tracks`.

        :param analyzer: A loaded
            :class:`~src.analysis.summary.EEGSummarizedAnalyzer` constructed with
            :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS`.
        :param data: Dataset the returned object should describe. Defaults to the
            analyser's own time-domain data; pass the wavelet dataset when splitting
            wavelet-derived arrays. Only its ``label`` and shape are used, and the last
            axis must span the same number of samples as the analyser's array.
        :param conditions: Segment order, matching how the array was built.
        :return: A :class:`PairedConditionTracks` whose reverse accessors describe the
            analyser's array.
        :raises ValueError: If the analyser is not a ``JOINED_TRACKS`` dataset, has no
            stored boundaries, or the boundaries do not match *conditions*.
        """
        if not getattr(analyzer, "concatenates_condition_tracks", False):
            raise ValueError(
                "Analyser was not constructed with ConditionVariants.JOINED_TRACKS, so "
                "it has no per-condition time segments."
            )
        boundaries = getattr(analyzer, "segment_boundaries", None)
        if boundaries is None:
            raise ValueError(
                "Analyser has no segment boundaries; the "
                "'.segment_boundaries.npy' sidecar is missing next to the "
                "concatenated array. Rebuild it with load_and_prepare_data()."
            )
        boundaries = np.asarray(boundaries, dtype=int)
        conditions = tuple(conditions)
        if len(boundaries) != len(conditions) + 1:
            raise ValueError(
                f"Stored boundaries {boundaries.tolist()} describe "
                f"{len(boundaries) - 1} segment(s) but {len(conditions)} condition(s) "
                "were given."
            )

        resolved = data if data is not None else analyzer.to_analysis_data()
        segment_lengths = tuple(
            int(np.diff(boundaries)[i]) for i in range(len(conditions))
        )

        # participant_labels lives in scripts/, which src/ must not import; read the
        # sidecar mapping directly instead.
        participants = _participants_from_metadata(
            analyzer.filtered_df, len(boundaries) - 1
        )

        onsets: dict[ConditionVariants, Optional[np.ndarray]] = {
            condition: None for condition in conditions
        }
        if getattr(analyzer, "stimulus_onsets", None) is not None:
            # Only the first segment's onsets are stored on the analyser; they are
            # already local to it because that segment starts at sample 0.
            onsets[conditions[0]] = np.asarray(analyzer.stimulus_onsets)

        return cls(
            data=resolved,
            participants=participants,
            conditions=conditions,
            segment_lengths=segment_lengths,
            onsets=onsets,
            zscore_mode="none",
        )


def _participants_from_metadata(filtered_df, n_subjects: int) -> tuple[str, ...]:
    """Participant label per subject index, read from an analyser's metadata.

    A ``JOINED_TRACKS`` sidecar keeps one row per *recording*, so a participant's two
    rows share one
    :attr:`~src.definitions.fields.SingleDataMetadata.CONCATENATED_PERSON_INDEX`.
    Both carry the same participant ID, so collapsing duplicates is unambiguous.

    :param filtered_df: The analyser's ``filtered_df``.
    :param n_subjects: Number of subjects on the concatenated array's first axis.
    :return: ``n_subjects`` participant labels, ordered by subject index.
    :raises ValueError: If the mapping cannot be built.
    """
    if filtered_df is None or len(filtered_df) == 0:
        raise ValueError("No participant metadata available (filtered_df is empty).")
    if SingleDataMetadata.CONCATENATED_PERSON_INDEX not in filtered_df.columns:
        raise ValueError(
            "Participant metadata has no CONCATENATED_PERSON_INDEX column; the "
            "subject-index mapping is unavailable."
        )

    by_index = dict(
        zip(
            filtered_df[SingleDataMetadata.CONCATENATED_PERSON_INDEX],
            filtered_df[SingleDataMetadata.PARTICIPANT_ID],
        )
    )
    missing = [s for s in range(n_subjects) if s not in by_index]
    if missing:
        raise ValueError(
            f"CONCATENATED_PERSON_INDEX is missing subject index/indices {missing}."
        )
    return tuple(
        "".join(ch for ch in str(by_index[s]) if ch.isdigit())[-3:].zfill(3)
        for s in range(n_subjects)
    )


def _match_participants(
    participants: Mapping[ConditionVariants, Sequence[str]],
    conditions: Sequence[ConditionVariants],
) -> list[str]:
    """Participants present in every condition, in sorted order.

    :param participants: Participant label per subject index, per condition.
    :param conditions: Conditions being concatenated.
    :return: Sorted participant labels common to all conditions.
    :raises ValueError: If a condition lists a participant twice, or no participant is
        common to every condition.
    """
    per_condition: list[set[str]] = []
    for condition in conditions:
        labels = list(participants[condition])
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(
                f"Condition {condition.value} lists participant(s) "
                f"{sorted(duplicates)} more than once; a participant contributes at "
                "most one recording per condition."
            )
        per_condition.append(set(labels))

    common = sorted(set.intersection(*per_condition))
    if not common:
        raise ValueError(
            "No participant is present in every condition, so no pair can be formed "
            f"(per-condition counts: "
            f"{ {c.value: len(s) for c, s in zip(conditions, per_condition)} })."
        )
    return common


def concatenate_condition_tracks(
    tracks: Mapping[ConditionVariants, AnalysisData],
    participants: Mapping[ConditionVariants, Sequence[str]],
    *,
    conditions: Sequence[ConditionVariants] = REAL_CONDITIONS,
    zscore_mode: str = "per_condition",
    onsets: Optional[Mapping[ConditionVariants, Optional[np.ndarray]]] = None,
    label: Optional[str] = None,
) -> PairedConditionTracks:
    """Match participants across conditions and concatenate their tracks over time.

    Consumes the per-condition datasets as they already exist — typically the wavelet
    power returned by
    :func:`~scripts.notebook_helpers.compute_wavelet_datasets` and loaded from the
    existing per-condition cache — so no re-alignment or re-preprocessing is needed.
    The two conditions keep their own time bases and may differ in length; only their
    feature dimensions must agree — and neither need their **subject** axes, since
    each condition arrives with every recording it has and the participant matching is
    what reconciles them.

    :param tracks: Per condition, the dataset to concatenate. Every entry must agree on
        the axes *between* the subject axis and the last (time) axis, and on ``sfreq``.
        The subject and time axes may both differ: the subject axis is reconciled by the
        participant matching, and differing track lengths are the point of this join.
    :param participants: Per condition, the participant label of each subject index —
        e.g. from :func:`~scripts.analysis_common.participant_labels`. Used to match
        subjects across conditions; only participants present in every condition are
        kept, sorted by label.
    :param conditions: Conditions to concatenate, in the order they should appear along
        the time axis. Defaults to Placebo then Psilocybin.
    :param zscore_mode: One of :data:`ZSCORE_MODES`. Defaults to ``"per_condition"``
        — each track standardised on its own before concatenation — which normalises
        away any overall power difference between the conditions. Pass ``"joint"`` to
        keep that difference.
    :param onsets: Optional per-condition stimulus-onset sample indices, local to that
        condition (e.g. ``analyzer.stimulus_onsets``). Carried through to
        :meth:`PairedConditionTracks.condition_onsets`.
    :param label: Label for the concatenated dataset. Defaults to the input labels
        joined with ``+``, which cannot collide with the crop-based ``Joined_*``
        products.
    :return: The concatenated dataset plus the segment bookkeeping needed to split it
        back apart.
    :raises ValueError: If *zscore_mode* is unknown, a condition is missing from
        *tracks* or *participants*, the feature dimensions or sampling rates disagree,
        a participant list does not match its data's subject count, or no participant
        is common to every condition.
    """
    if zscore_mode not in ZSCORE_MODES:
        raise ValueError(
            f"Unknown zscore_mode {zscore_mode!r}; expected one of {list(ZSCORE_MODES)}."
        )

    conditions = tuple(conditions)
    for condition in conditions:
        if condition not in tracks:
            raise ValueError(f"No track supplied for condition {condition.value}.")
        if condition not in participants:
            raise ValueError(
                f"No participant labels supplied for condition {condition.value}."
            )

    reference = tracks[conditions[0]]
    for condition in conditions:
        ad = tracks[condition]
        # Only the axes BETWEEN subject and time have to agree. The subject axis
        # deliberately need not: each condition arrives with every recording it has —
        # 15 Placebo against 16 Psilocybin on the ASSR set — and reconciling that is
        # exactly what the participant matching below does. Comparing it here instead
        # would reject every real cohort in which the two conditions are not already
        # the same size, which is the normal case. Each condition's own subject count is
        # still checked, against its own participant list.
        if ad.data.shape[1:-1] != reference.data.shape[1:-1]:
            raise ValueError(
                f"Track {condition.value} has feature dimensions "
                f"{ad.data.shape[1:-1]}, expected {reference.data.shape[1:-1]} to "
                f"match {conditions[0].value}."
            )
        if not np.isclose(ad.sfreq, reference.sfreq):
            raise ValueError(
                f"Track {condition.value} is sampled at {ad.sfreq} Hz but "
                f"{conditions[0].value} at {reference.sfreq} Hz."
            )
        if len(participants[condition]) != ad.data.shape[0]:
            raise ValueError(
                f"Condition {condition.value} has {ad.data.shape[0]} subject(s) but "
                f"{len(participants[condition])} participant label(s)."
            )

    matched = _match_participants(participants, conditions)

    # Reorder every condition onto the same participant order before concatenating, so
    # subject k really is the same person in both segments.
    segments: list[np.ndarray] = []
    for condition in conditions:
        label_to_index = {
            participant: index
            for index, participant in enumerate(participants[condition])
        }
        rows = [label_to_index[participant] for participant in matched]
        segment = tracks[condition].data[rows]
        if zscore_mode == "per_condition":
            segment = zscore_by_time(segment)
        segments.append(segment)

    concatenated = np.concatenate(segments, axis=-1)
    if zscore_mode == "joint":
        concatenated = zscore_by_time(concatenated)

    resolved_label = (
        label
        if label is not None
        else "+".join(tracks[condition].label for condition in conditions)
    )

    data = AnalysisData(
        data=concatenated,
        sfreq=reference.sfreq,
        representation=reference.representation,
        label=resolved_label,
        feature_names=reference.feature_names,
        info=reference.info,
        metadata={
            **reference.metadata,
            "paired_conditions": [c.value for c in conditions],
            "paired_segment_lengths": [int(s.shape[-1]) for s in segments],
            "paired_participants": list(matched),
            "paired_zscore_mode": zscore_mode,
        },
    )

    return PairedConditionTracks(
        data=data,
        participants=tuple(matched),
        conditions=conditions,
        segment_lengths=tuple(int(s.shape[-1]) for s in segments),
        onsets={
            condition: (
                None
                if onsets is None or onsets.get(condition) is None
                else np.asarray(onsets[condition])
            )
            for condition in conditions
        },
        zscore_mode=zscore_mode,
    )


# ──────────────────────────────────────────────────────────────────────
# Pooling on the subject axis (ConditionVariants.JOINED)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PooledConditionSubjects:
    """A participant-matched, subject-axis-pooled two-condition dataset.

    The counterpart of :class:`PairedConditionTracks` over the other axis: the same
    participant-matched cohort, joined along the **subject** axis instead of the time
    axis, so every *recording* is one subject and each participant appears once per
    condition. This is the array layout
    :attr:`~src.definitions.fields.ConditionVariants.JOINED` names.

    .. code-block:: text

        subject 0      ->  participant 001, Placebo
        ...
        subject P-1    ->  participant 0NN, Placebo
        subject P      ->  participant 001, Psilocybin
        ...
        subject 2P-1   ->  participant 0NN, Psilocybin

    Unlike the time-axis join, this one requires the conditions to **share a time
    base** — the subject axis can only be stacked when every recording has the same
    number of samples. That holds whenever the stimulus/time alignment was fitted over
    every recording of both conditions, which is what
    :meth:`~src.analysis.summary.EEGSummarizedAnalyzer.load_pre_alignment_data` does,
    so the existing per-condition caches can be pooled as they are.

    What a decomposition run on this layout gives you, and how it differs from the
    time-axis join: the mixing is estimated **per subject**, so a participant's two
    recordings get their own patterns and the components are *not* identical between
    conditions by construction. What IVA does tie across them is the shared source
    (the SCV), i.e. both conditions are described by one set of aligned sources. Use
    :meth:`condition_mask` / :meth:`condition_subjects` to read either condition's half
    back out of any subject-axis array.

    :param data: The pooled dataset. Same axes as the inputs, with the first (subject)
        axis holding every condition's subjects end to end — e.g.
        ``(n_pairs * n_conditions, n_channels, n_freqs, n_times)`` for wavelet power.
    :param participants: Participant label per subject index, in subject order. Every
        label appears once per condition.
    :param subject_conditions: Condition of each subject index, aligned with
        :attr:`participants`.
    :param conditions: Condition order along the subject axis.
    :param onsets: Per condition, stimulus-onset sample indices, or ``None`` when the
        experiment has no stimulus annotations. Kept per condition rather than as one
        array because a shared time base still leaves the per-condition onset indices
        free to differ by a sample or two of rounding.
    """

    data: AnalysisData
    participants: tuple[str, ...]
    subject_conditions: tuple[ConditionVariants, ...]
    conditions: tuple[ConditionVariants, ...]
    onsets: Mapping[ConditionVariants, Optional[np.ndarray]] = field(
        default_factory=dict
    )

    @property
    def n_subjects(self) -> int:
        """Length of the pooled subject axis (participants × conditions)."""
        return len(self.participants)

    @property
    def n_pairs(self) -> int:
        """Number of participants contributing every condition."""
        return len(set(self.participants))

    @property
    def subject_labels(self) -> tuple[str, ...]:
        """Per-subject ``"031 Placebo"`` labels, unique across the subject axis.

        :attr:`participants` alone repeats every label once per condition, so plots
        indexed by subject need the condition in the label too.
        """
        return tuple(
            f"{participant} {condition.value}"
            for participant, condition in zip(
                self.participants, self.subject_conditions
            )
        )

    def condition_mask(self, condition: ConditionVariants) -> np.ndarray:
        """Boolean mask selecting the subjects recorded under *condition*.

        :param condition: Condition to select.
        :return: Boolean array of length :attr:`n_subjects`.
        :raises KeyError: If *condition* is not one of the pooled conditions.
        """
        self._condition_index(condition)
        return np.asarray([c is condition for c in self.subject_conditions], dtype=bool)

    def condition_subjects(
        self,
        array: np.ndarray,
        condition: ConditionVariants,
    ) -> np.ndarray:
        """Recover one condition's subjects from any array on the pooled subject axis.

        The reverse of the pooling, and deliberately shape-agnostic: the only
        requirement is that the **first** axis is the pooled subject axis, so it works
        on the data itself ``(S, C, F, T)``, on IVA sources ``(S, K, F, T)``, on
        per-subject topographies ``(S, K, C)``, and so on. Rows come back in
        :attr:`participants` order, so the two conditions' returns are participant-
        matched row for row.

        :param array: Any array whose first axis spans :attr:`n_subjects`.
        :param condition: Condition to extract.
        :return: The rows of *array* belonging to that condition.
        :raises ValueError: If the first axis does not match :attr:`n_subjects`.
        :raises KeyError: If *condition* is not one of the pooled conditions.
        """
        array = np.asarray(array)
        if array.shape[0] != self.n_subjects:
            raise ValueError(
                f"Array's first axis is {array.shape[0]} subject(s) but the pooled "
                f"subject axis is {self.n_subjects}; it does not come from this "
                "pooling."
            )
        return array[self.condition_mask(condition)]

    @property
    def partner_index(self) -> np.ndarray:
        """Map each subject index to the same participant's other-condition index.

        Lets paired (within-participant) statistics index straight into the subject
        axis. Only meaningful for two pooled conditions; entry *k* is ``-1`` where the
        participant has no counterpart, which a matched pooling never produces.

        :return: Integer array of length :attr:`n_subjects`.
        """
        partners = np.full(self.n_subjects, -1, dtype=int)
        positions: dict[str, list[int]] = {}
        for index, participant in enumerate(self.participants):
            positions.setdefault(participant, []).append(index)
        for indices in positions.values():
            if len(indices) == 2:
                first, second = indices
                partners[first], partners[second] = second, first
        return partners

    def select_participants(
        self, participants: Sequence[str]
    ) -> "PooledConditionSubjects":
        """Restrict the pooled cohort to *participants*, keeping every condition.

        The way to take a subject subset of a pooled dataset: slicing the subject axis
        by position would take whole condition blocks instead of whole participants, so
        a "first 5 subjects" slice would silently drop a condition. This keeps each
        named participant's recording in **every** condition, so the result is still
        matched, and preserves the block order of :attr:`conditions`.

        :param participants: Participant labels to keep, e.g. the first few of
            ``sorted(set(pooled.participants))``. Order is ignored; the pooled order is
            preserved.
        :return: A new :class:`PooledConditionSubjects` holding only those
            participants.
        :raises ValueError: If a requested participant is not in the pooled cohort, or
            none is.
        """
        wanted = set(participants)
        unknown = sorted(wanted - set(self.participants))
        if unknown:
            raise ValueError(
                f"Participant(s) {unknown} are not in this pooled cohort; it holds "
                f"{sorted(set(self.participants))}."
            )
        rows = [
            index
            for index, participant in enumerate(self.participants)
            if participant in wanted
        ]
        if not rows:
            raise ValueError("No participants selected; the subset would be empty.")

        data = replace(
            self.data,
            data=self.data.data[rows],
            metadata={
                **self.data.metadata,
                "pooled_participants": sorted(wanted),
                "pooled_subject_conditions": [
                    self.subject_conditions[index].value for index in rows
                ],
            },
        )
        return replace(
            self,
            data=data,
            participants=tuple(self.participants[index] for index in rows),
            subject_conditions=tuple(self.subject_conditions[index] for index in rows),
        )

    def condition_onsets(self, condition: ConditionVariants) -> Optional[np.ndarray]:
        """Stimulus-onset samples for one condition.

        No shifting is needed — pooling is on the subject axis, so every subject
        shares the one time axis.

        :param condition: Condition whose onsets are wanted.
        :return: Onset sample indices, or ``None`` when the experiment has none.
        :raises KeyError: If *condition* is not one of the pooled conditions.
        """
        self._condition_index(condition)
        onsets = self.onsets.get(condition)
        return None if onsets is None else np.asarray(onsets)

    def _condition_index(self, condition: ConditionVariants) -> int:
        """Position of *condition* among the pooled conditions."""
        if condition not in self.conditions:
            raise KeyError(
                f"Condition {condition} is not part of this pooling; it holds "
                f"{[c.value for c in self.conditions]}."
            )
        return self.conditions.index(condition)


def _joined_label(
    tracks: Mapping[ConditionVariants, AnalysisData],
    conditions: Sequence[ConditionVariants],
) -> str:
    """``Joined_<rest>`` label for a pooled dataset, from the input labels.

    Per-condition labels follow ``"<Condition>_<rest>"``, so when every track agrees on
    the part after the condition the pooled label is the same name under the virtual
    :attr:`~src.definitions.fields.ConditionVariants.JOINED` condition. Falls back to
    the input labels joined with ``+`` when they do not share that suffix.

    **Not a path-safe product name.** This is a *data* label and it reproduces whatever
    the inputs carried after the condition — which, for anything that has been through
    :func:`~src.analysis.data_representations.to_wavelet_power`, includes the
    representation annotation, giving ``"Joined_ASSR (wavelet power 1-50 Hz)"``. That
    describes the array correctly but is unfit to name a directory or a figure file.
    A caller naming an output must build ``f"{ConditionVariants.JOINED.value}_{music_type.value}"``
    itself, as ``scripts/run_iva_condition_comparison.py`` and
    ``scripts/run_iva_condition_tracks.py`` both do, so the plot tree agrees with the
    component store (:mod:`src.io.iva_store`), which names the condition from the enum.

    :param tracks: Per condition, the dataset being pooled.
    :param conditions: Conditions being pooled.
    :return: The pooled dataset's label.
    """
    suffixes = {
        tracks[condition].label.split("_", 1)[1]
        for condition in conditions
        if "_" in tracks[condition].label
    }
    if len(suffixes) == 1:
        return f"{ConditionVariants.JOINED.value}_{suffixes.pop()}"
    return "+".join(tracks[condition].label for condition in conditions)


def pool_condition_subjects(
    tracks: Mapping[ConditionVariants, AnalysisData],
    participants: Mapping[ConditionVariants, Sequence[str]],
    *,
    conditions: Sequence[ConditionVariants] = REAL_CONDITIONS,
    onsets: Optional[Mapping[ConditionVariants, Optional[np.ndarray]]] = None,
    label: Optional[str] = None,
) -> PooledConditionSubjects:
    """Match participants across conditions and stack them on the subject axis.

    The subject-axis counterpart of :func:`concatenate_condition_tracks`, and the
    array :attr:`~src.definitions.fields.ConditionVariants.JOINED` describes. Like the
    time-axis join it consumes the per-condition datasets **as they already exist** —
    typically the wavelet power loaded from the existing per-condition caches by
    :func:`~scripts.notebook_helpers.compute_wavelet_datasets` — so nothing upstream is
    re-run, re-aligned or re-transformed, and no new cache is written.

    Unlike the time-axis join, the conditions must agree on **every** axis but the
    subject one, the time axis included: stacking recordings of different lengths into
    one subject axis is not defined. That is satisfied whenever the alignment was
    fitted over both conditions (the project's ASSR crops are), and a mismatch raises
    rather than silently trimming.

    No standardisation happens here, and none is offered: every z-scoring in this
    project normalises each ``(subject, channel, frequency)`` series over time
    independently, so applying it before or after a subject-axis stack gives the
    identical array — there is no per-condition/joint distinction to make, unlike on
    the time axis. Standardise downstream with
    :func:`~src.analysis.wavelet_ica.zscore_by_time` exactly as a single-condition
    workflow does. Note the consequence, which is the same as ``"per_condition"`` on
    the time axis: an overall power difference between the conditions is normalised
    away, so what a pooled decomposition compares is temporal and spectral structure,
    not amplitude.

    :param tracks: Per condition, the dataset to pool. Every entry must have the same
        shape except along the first (subject) axis, and the same ``sfreq``.
    :param participants: Per condition, the participant label of each subject index —
        e.g. from :func:`~scripts.analysis_common.participant_labels`. Used to match
        subjects across conditions; only participants present in every condition are
        kept, sorted by label.
    :param conditions: Conditions to pool, in the order their blocks should appear
        along the subject axis. Defaults to Placebo then Psilocybin.
    :param onsets: Optional per-condition stimulus-onset sample indices (e.g.
        ``analyzer.stimulus_onsets``). Carried through to
        :meth:`PooledConditionSubjects.condition_onsets`.
    :param label: Label for the pooled dataset. Defaults to the canonical
        ``Joined_<MusicType>`` when the inputs share that suffix.
    :return: The pooled dataset plus the per-subject participant/condition bookkeeping
        needed to split it back apart.
    :raises ValueError: If a condition is missing from *tracks* or *participants*, the
        non-subject dimensions or sampling rates disagree, a participant list does not
        match its data's subject count, or no participant is common to every condition.
    """
    conditions = tuple(conditions)
    for condition in conditions:
        if condition not in tracks:
            raise ValueError(f"No track supplied for condition {condition.value}.")
        if condition not in participants:
            raise ValueError(
                f"No participant labels supplied for condition {condition.value}."
            )

    reference = tracks[conditions[0]]
    for condition in conditions:
        ad = tracks[condition]
        if ad.data.shape[1:] != reference.data.shape[1:]:
            raise ValueError(
                f"Track {condition.value} has non-subject dimensions "
                f"{ad.data.shape[1:]}, expected {reference.data.shape[1:]} to match "
                f"{conditions[0].value}. Pooling on the subject axis needs one shared "
                "time base; use concatenate_condition_tracks() to join conditions "
                "that keep their own."
            )
        if not np.isclose(ad.sfreq, reference.sfreq):
            raise ValueError(
                f"Track {condition.value} is sampled at {ad.sfreq} Hz but "
                f"{conditions[0].value} at {reference.sfreq} Hz."
            )
        if len(participants[condition]) != ad.data.shape[0]:
            raise ValueError(
                f"Condition {condition.value} has {ad.data.shape[0]} subject(s) but "
                f"{len(participants[condition])} participant label(s)."
            )

    matched = _match_participants(participants, conditions)

    # One block per condition, each in the same participant order, so subject k and its
    # partner in the other block really are the same person.
    blocks: list[np.ndarray] = []
    subject_participants: list[str] = []
    subject_condition_list: list[ConditionVariants] = []
    for condition in conditions:
        label_to_index = {
            participant: index
            for index, participant in enumerate(participants[condition])
        }
        rows = [label_to_index[participant] for participant in matched]
        blocks.append(tracks[condition].data[rows])
        subject_participants.extend(matched)
        subject_condition_list.extend([condition] * len(matched))

    pooled = np.concatenate(blocks, axis=0)
    resolved_label = label if label is not None else _joined_label(tracks, conditions)

    data = AnalysisData(
        data=pooled,
        sfreq=reference.sfreq,
        representation=reference.representation,
        label=resolved_label,
        feature_names=reference.feature_names,
        info=reference.info,
        metadata={
            **reference.metadata,
            "pooled_conditions": [c.value for c in conditions],
            "pooled_participants": list(matched),
            "pooled_subject_conditions": [c.value for c in subject_condition_list],
        },
    )

    return PooledConditionSubjects(
        data=data,
        participants=tuple(subject_participants),
        subject_conditions=tuple(subject_condition_list),
        conditions=conditions,
        onsets={
            condition: (
                None
                if onsets is None or onsets.get(condition) is None
                else np.asarray(onsets[condition])
            )
            for condition in conditions
        },
    )
