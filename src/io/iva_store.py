"""
Persistent store for the IVA component products: the per-component
time-frequency maps and channel topographies, together with the bookkeeping
needed to tell which recording — which participant, under which condition — each
row belongs to.

Why this exists. An IVA run is expensive (a per-recording PCA plus ``iva_g`` over
a tensor of tens of gigabytes) and its figures are lossy summaries: a group-mean
topomap grid cannot be re-split per participant, and a plotted TF map cannot be
correlated against anything. Keeping the recovered components on disk makes every
downstream question — a condition contrast, a per-participant follow-up, a
comparison against a behavioural score — a load rather than a re-run.

Layout::

    data/processed/<experiment>/iva_results/<Condition>/
        <variant>__<MusicType>__<spectrum>__pca<n_pca>.npz

The condition is a directory (:meth:`~src.definitions.constants.ProjectPaths.get_iva_results_dir`)
and everything that distinguishes two runs *within* one condition is in the
filename, so a whole condition can be listed, copied or deleted as a unit while
sweeps over ``--n_pca`` or over frequency bands never overwrite each other.
``<spectrum>`` is ``broadband`` for a full-grid run and the band name otherwise.

**The participant mapping is part of the file, not a convention.** Every stored
array is indexed ``(recording, component, ...)``, and the subject axis of a
concatenated EEG array is meaningful only against the metadata sidecar that
produced it. A file therefore carries a participant label *and* a condition per
row, and :class:`IvaComponentResults` resolves rows by those labels
(:meth:`~IvaComponentResults.row`, :meth:`~IvaComponentResults.rows`) rather than
asking the caller to remember the order. This matters most for
:attr:`~src.definitions.fields.ConditionVariants.JOINED`, where one participant
owns **two** rows and a bare label is ambiguous — the accessors say so instead of
silently returning the first match.

For :attr:`~src.definitions.fields.ConditionVariants.JOINED_TRACKS` the row is
shared by both conditions and the split is along *time*, so the segment order and
lengths are stored too; recover a condition with
:meth:`~IvaComponentResults.condition_track`.

Which arrays a file holds depends on the decomposition — see
:class:`~src.definitions.fields.IvaComponentArrays`. A channel-mixing run offers a
TF map and a channel topography; the frequency-channel and time variants have no
per-component TF map at all, so a reader asks for a named array and gets a clear
error when the variant never produced it.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ConditionVariants,
    CoordinateSystems,
    ExperimentNames,
    IvaComponentArrays,
    IvaVariants,
    MusicTypeVariants,
    SpectrumTypeVariants,
)

_logger = logging.getLogger(__name__)

#: Bumped whenever the on-disk key set changes in a way a reader must notice.
FORMAT_VERSION = 1

#: Filename suffix of every store entry.
STORE_SUFFIX = ".npz"

#: ``npz`` key prefixes keeping the three namespaces apart, so a diagnostic named
#: ``freqs`` can never shadow the frequency axis.
_ARRAY_PREFIX = "array__"
_EXTRA_PREFIX = "extra__"

#: Prefix of the per-condition stimulus-onset entries in :attr:`IvaComponentResults.extras`.
#: One entry per condition rather than a single array: the conditions keep their own
#: alignments in a time-axis join, so their onset counts need not even match, and on a
#: shared time base the indices can still differ by a sample or two of rounding.
ONSETS_EXTRA_PREFIX = "stimulus_onsets_"

#: Trailing axes of each canonical array, past the leading ``(recording,
#: component)`` pair. Used to validate a write against the coordinate axes it is
#: stored with: a TF map whose frequency axis disagrees with ``freqs`` is a
#: mislabelled file, and the mislabelling is invisible once the run has exited.
_ARRAY_AXES: dict[IvaComponentArrays, tuple[str, ...]] = {
    IvaComponentArrays.TF_MAP: ("freq", "time"),
    IvaComponentArrays.CHANNEL_PATTERN: ("channel",),
    IvaComponentArrays.TIMECOURSE: ("time",),
    IvaComponentArrays.SPECTRAL_PROFILE: ("freq",),
    IvaComponentArrays.FREQUENCY_CHANNEL_PATTERN: ("freq", "channel"),
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _spectrum_token(band: str | None) -> str:
    """Filename token naming the spectral extent of a run.

    :param band: Band name, or ``None`` for a full-grid run.
    :return: ``"broadband"`` or the band name, sanitised for a filename.
    """
    if band is None:
        return SpectrumTypeVariants.BROADBAND.value
    token = re.sub(r"[^A-Za-z0-9_-]", "_", str(band)).strip("_")
    if not token:
        raise ValueError(f"Band name {band!r} has no filename-safe characters.")
    return token


def iva_results_filename(
    variant: IvaVariants,
    music_type: MusicTypeVariants,
    band: str | None,
    n_pca: int,
) -> str:
    """Filename of one store entry.

    :param variant: Decomposition variant.
    :param music_type: Music type of the run (``ASSR`` for the ASSR experiment).
    :param band: Frequency band the run was restricted to, or ``None``.
    :param n_pca: Per-recording PCA dimension, i.e. the component count.
    :return: The ``.npz`` filename, without any directory part.
    :raises ValueError: If *n_pca* is not positive.
    """
    if n_pca <= 0:
        raise ValueError(f"n_pca must be positive; got {n_pca}.")
    return (
        f"{variant.value}__{music_type.value}__{_spectrum_token(band)}"
        f"__pca{n_pca}{STORE_SUFFIX}"
    )


def iva_results_path(
    experiment: ExperimentNames,
    condition: ConditionVariants,
    variant: IvaVariants,
    music_type: MusicTypeVariants,
    band: str | None,
    n_pca: int,
    processed_data_dir: Path | None = None,
) -> Path:
    """Full path of one store entry.

    :param experiment: Experiment the run belongs to.
    :param condition: Condition of the run; becomes the subdirectory.
    :param variant: Decomposition variant.
    :param music_type: Music type of the run.
    :param band: Frequency band the run was restricted to, or ``None``.
    :param n_pca: Per-recording PCA dimension.
    :param processed_data_dir: Processed-data root, or ``None`` for the project's.
    :return: Path to the ``.npz`` file, which need not exist.
    """
    return ProjectPaths.get_iva_results_dir(
        experiment, condition, processed_data_dir=processed_data_dir
    ) / iva_results_filename(variant, music_type, band, n_pca)


def list_iva_results(
    experiment: ExperimentNames,
    condition: ConditionVariants | None = None,
    processed_data_dir: Path | None = None,
) -> list[Path]:
    """List the store entries of an experiment.

    :param experiment: Experiment whose store is listed.
    :param condition: Restrict to one condition's subdirectory, or ``None`` for
        every condition present.
    :param processed_data_dir: Processed-data root, or ``None`` for the project's.
    :return: Sorted paths; empty when nothing has been stored yet.
    """
    root = ProjectPaths.get_iva_results_dir(
        experiment, condition, processed_data_dir=processed_data_dir
    )
    if not root.is_dir():
        return []
    pattern = f"*{STORE_SUFFIX}" if condition is not None else f"*/*{STORE_SUFFIX}"
    return sorted(root.glob(pattern))


# ---------------------------------------------------------------------------
# Loaded results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IvaComponentResults:
    """One stored IVA decomposition, with its per-recording bookkeeping.

    :param variant: Decomposition variant that produced the arrays.
    :param experiment: Experiment the run belongs to.
    :param condition: Condition of the run, i.e. the store subdirectory.
    :param music_type: Music type of the run.
    :param band: Frequency band the run was restricted to, or ``None`` for a
        full-grid run.
    :param n_pca: Per-recording PCA dimension, equal to the component count.
    :param sfreq: Sampling frequency of the time axis, in Hz.
    :param participants: Participant label per recording, in row order. **The**
        mapping — a row means nothing without it.
    :param subject_conditions: Condition of each recording, in row order. Constant
        for a single-condition run; alternating for
        :attr:`~src.definitions.fields.ConditionVariants.JOINED`, where a
        participant owns one row per condition.
    :param arrays: Canonical array name → ``(S, K, ...)`` array. Which names are
        present depends on :attr:`variant`; read them through :meth:`array`.
    :param freqs: ``(F,)`` frequency axis in Hz, or ``None`` when no stored array
        has a frequency axis.
    :param times: ``(T,)`` time axis in seconds, or ``None`` when no stored array
        has a time axis.
    :param channel_names: Channel names of the topography axis, or ``None`` when
        the run had no channel montage available.
    :param segment_conditions: For a time-axis join, the condition of each segment
        of the time axis, in order; empty otherwise.
    :param segment_lengths: Sample count of each segment, aligned with
        :attr:`segment_conditions`; empty otherwise.
    :param extras: Free-form per-run diagnostics kept alongside the components
        (component ranking scores, PCA explained variance, ...). Not validated.
    :param path: File the results were read from, or ``None`` for an in-memory
        instance.
    """

    variant: IvaVariants
    experiment: ExperimentNames
    condition: ConditionVariants
    music_type: MusicTypeVariants
    band: str | None
    n_pca: int
    sfreq: float
    participants: tuple[str, ...]
    subject_conditions: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    freqs: np.ndarray | None = None
    times: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None
    segment_conditions: tuple[str, ...] = ()
    segment_lengths: tuple[int, ...] = ()
    extras: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
    path: Path | None = None

    # ── shape ──────────────────────────────────────────────────────────

    @property
    def n_subjects(self) -> int:
        """Number of recordings on the leading axis of every array."""
        return len(self.participants)

    @property
    def n_components(self) -> int:
        """Number of IVA components, i.e. :attr:`n_pca`."""
        return int(self.n_pca)

    @property
    def label(self) -> str:
        """The canonical ``<Condition>_<MusicType>`` dataset label of the run."""
        return f"{self.condition.value}_{self.music_type.value}"

    # ── the participant mapping ────────────────────────────────────────

    @property
    def participant_rows(self) -> dict[str, tuple[int, ...]]:
        """Participant label → the row indices that participant occupies.

        A tuple rather than a single index on purpose: a subject-axis join gives a
        participant one row per condition, so collapsing to one index would quietly
        drop half the data.
        """
        rows: dict[str, list[int]] = {}
        for index, participant in enumerate(self.participants):
            rows.setdefault(participant, []).append(index)
        return {name: tuple(values) for name, values in rows.items()}

    def participant_frame(self) -> pd.DataFrame:
        """The row bookkeeping as a DataFrame, one row per recording.

        :return: Columns ``subject_index``, ``participant`` and ``condition``.
        """
        return pd.DataFrame(
            {
                "subject_index": np.arange(self.n_subjects),
                "participant": list(self.participants),
                "condition": list(self.subject_conditions),
            }
        )

    def rows(
        self,
        participant: str | None = None,
        condition: ConditionVariants | str | None = None,
    ) -> list[int]:
        """Row indices matching a participant and/or a condition.

        :param participant: Participant label to match, or ``None`` for any.
        :param condition: Condition to match, or ``None`` for any.
        :return: Matching row indices, ascending. Empty when nothing matches.
        """
        wanted = (
            condition.value if isinstance(condition, ConditionVariants) else condition
        )
        return [
            index
            for index in range(self.n_subjects)
            if (participant is None or self.participants[index] == participant)
            and (wanted is None or self.subject_conditions[index] == wanted)
        ]

    def row(
        self,
        participant: str,
        condition: ConditionVariants | str | None = None,
    ) -> int:
        """The single row of one participant, optionally within one condition.

        :param participant: Participant label.
        :param condition: Condition to disambiguate a participant holding several
            rows (a subject-axis join). ``None`` requires the participant to own
            exactly one row.
        :return: The row index.
        :raises KeyError: If no row matches.
        :raises ValueError: If several rows match, i.e. a condition is needed.
        """
        matches = self.rows(participant, condition)
        if not matches:
            known = sorted(set(self.participants))
            suffix = "" if condition is None else f" under condition {condition!r}"
            raise KeyError(
                f"No recording for participant {participant!r}{suffix} in "
                f"{self.label}; stored participants: {known}."
            )
        if len(matches) > 1:
            found = [self.subject_conditions[index] for index in matches]
            raise ValueError(
                f"Participant {participant!r} occupies {len(matches)} rows in "
                f"{self.label} (conditions {found}); pass `condition` to pick one."
            )
        return matches[0]

    def select_participants(self, participants: Sequence[str]) -> IvaComponentResults:
        """A copy restricted to some participants, keeping every row they own.

        :param participants: Participant labels to keep. Order is ignored; the
            stored row order is preserved so the result stays directly comparable
            to the original.
        :return: A new instance sharing nothing mutable with this one.
        :raises KeyError: If a requested participant has no row.
        """
        wanted = set(participants)
        missing = sorted(wanted - set(self.participants))
        if missing:
            raise KeyError(
                f"Participants {missing} have no recording in {self.label}; "
                f"stored: {sorted(set(self.participants))}."
            )
        keep = [i for i in range(self.n_subjects) if self.participants[i] in wanted]
        return dataclasses.replace(
            self,
            participants=tuple(self.participants[i] for i in keep),
            subject_conditions=tuple(self.subject_conditions[i] for i in keep),
            arrays={name: value[keep] for name, value in self.arrays.items()},
        )

    # ── the arrays ─────────────────────────────────────────────────────

    def has(self, name: IvaComponentArrays | str) -> bool:
        """Whether a named component array is present in this file.

        :param name: Canonical array name.
        :return: ``True`` when :meth:`array` would succeed.
        """
        key = name.value if isinstance(name, IvaComponentArrays) else str(name)
        return key in self.arrays

    def array(self, name: IvaComponentArrays | str) -> np.ndarray:
        """One component array, ``(S, K, ...)``.

        :param name: Canonical array name.
        :return: The stored array. Not copied — treat it as read-only.
        :raises KeyError: If this variant never wrote that array. The message lists
            what the file does hold, because the absence is usually a property of
            the decomposition rather than of the run (the frequency-channel and
            time variants have no per-component TF map at all).
        """
        key = name.value if isinstance(name, IvaComponentArrays) else str(name)
        if key not in self.arrays:
            raise KeyError(
                f"{self.variant.value} run {self.label} has no {key!r} array; "
                f"it stored {sorted(self.arrays)}."
            )
        return self.arrays[key]

    @property
    def tf_maps(self) -> np.ndarray:
        """``(S, K, F, T)`` per-recording time-frequency component maps."""
        return self.array(IvaComponentArrays.TF_MAP)

    @property
    def channel_patterns(self) -> np.ndarray:
        """``(S, K, C)`` per-recording forward channel topographies."""
        return self.array(IvaComponentArrays.CHANNEL_PATTERN)

    def for_participant(
        self,
        participant: str,
        condition: ConditionVariants | str | None = None,
    ) -> dict[str, np.ndarray]:
        """Every stored array, sliced to one participant's row.

        :param participant: Participant label.
        :param condition: Condition, when the participant owns several rows.
        :return: Array name → ``(K, ...)`` slice for that recording.
        :raises KeyError: If no row matches.
        :raises ValueError: If several rows match.
        """
        index = self.row(participant, condition)
        return {name: value[index] for name, value in self.arrays.items()}

    # ── the time-axis join ─────────────────────────────────────────────

    @property
    def segment_boundaries(self) -> tuple[int, ...]:
        """Sample index where each time-axis segment starts, plus the final end.

        Has ``len(segment_conditions) + 1`` entries; empty when the run is not a
        time-axis join.
        """
        if not self.segment_lengths:
            return ()
        edges = [0]
        for length in self.segment_lengths:
            edges.append(edges[-1] + int(length))
        return tuple(edges)

    def segment_slice(self, condition: ConditionVariants | str) -> slice:
        """The slice of the time axis belonging to one segment.

        :param condition: Condition whose segment is wanted.
        :return: A ``slice`` into the last axis.
        :raises ValueError: If the run has no stored segments.
        :raises KeyError: If that condition is not one of them.
        """
        if not self.segment_conditions:
            raise ValueError(
                f"{self.label} ({self.variant.value}) stores no time-axis segments; "
                "only a time-axis join such as "
                f"{IvaVariants.CHANNEL_JOINED_TRACKS.value} does."
            )
        wanted = (
            condition.value if isinstance(condition, ConditionVariants) else condition
        )
        if wanted not in self.segment_conditions:
            raise KeyError(
                f"No segment for condition {wanted!r} in {self.label}; segments are "
                f"{list(self.segment_conditions)}."
            )
        index = self.segment_conditions.index(wanted)
        edges = self.segment_boundaries
        return slice(edges[index], edges[index + 1])

    def condition_track(
        self,
        values: np.ndarray | IvaComponentArrays | str,
        condition: ConditionVariants | str,
    ) -> np.ndarray:
        """Cut one condition's segment out of a time-axis join.

        :param values: An array whose last axis is the concatenated time axis, or
            the name of one of the stored arrays.
        :param condition: Condition whose segment is wanted.
        :return: A view of *values* restricted along the last axis.
        :raises ValueError: If the run has no stored segments, or the last axis of
            *values* is not the concatenated length.
        :raises KeyError: If that condition is not one of the segments.
        """
        data = (
            self.array(values)
            if isinstance(values, (IvaComponentArrays, str))
            else np.asarray(values)
        )
        total = self.segment_boundaries[-1] if self.segment_boundaries else 0
        if not total:
            raise ValueError(
                f"{self.label} ({self.variant.value}) stores no time-axis segments."
            )
        if data.shape[-1] != total:
            raise ValueError(
                f"Last axis of the array is {data.shape[-1]} but the stored segments "
                f"total {total} samples; this array is not on the concatenated axis."
            )
        return data[..., self.segment_slice(condition)]

    # ── drawing and epoching the stored components again ───────────────

    def topo_info(
        self,
        coordinate_system: CoordinateSystems = (
            CoordinateSystems.HYDROGEL_257_NO_FIDUCIALS
        ),
    ) -> mne.Info:
        """An MNE ``Info`` for the channel axis, so topographies can be redrawn.

        The reason :attr:`channel_names` is stored at all: a channel pattern is only
        a topography once each value is placed on the scalp, and the montage is not
        recoverable from the numbers. The names are rebuilt into an ``Info`` and the
        project montage applied, giving exactly the layout the run itself drew with.

        :param coordinate_system: Montage to apply. The default is the one the
            preprocessing pipeline uses, so it matches the stored names.
        :return: An EEG-only ``Info`` whose channel order is the stored channel axis.
        :raises ValueError: If the run stored no channel names (no montage was
            available to it), or stored no array with a channel axis.
        """
        if self.channel_names is None:
            raise ValueError(
                f"{self.label} ({self.variant.value}) stored no channel names, so no "
                "topography layout can be rebuilt. Re-run the decomposition with a "
                "montage available."
            )
        montage_path, _ = ProjectPaths.get_coordinates_file_path(coordinate_system)
        info = mne.create_info(
            list(self.channel_names), sfreq=float(self.sfreq), ch_types="eeg"
        )
        info.set_montage(
            mne.channels.read_custom_montage(montage_path), on_missing="warn"
        )
        return info

    @property
    def n_channels(self) -> int:
        """Length of the channel axis, from the stored channel names.

        :raises ValueError: If the run stored no channel names.
        """
        if self.channel_names is None:
            raise ValueError(
                f"{self.label} ({self.variant.value}) stored no channel names."
            )
        return len(self.channel_names)

    def stimulus_onsets(self, condition: ConditionVariants | str) -> np.ndarray | None:
        """Stimulus-onset samples of one condition, or ``None`` when none were stored.

        The frame matches the arrays the same file holds: for a subject-axis join the
        indices are on the one shared time axis; for a time-axis join they are **local
        to that condition's segment**, i.e. the frame :meth:`condition_track` returns.
        Onsets past the stored time axis were dropped when the file was written, so an
        index is always valid.

        :param condition: Condition whose onsets are wanted.
        :return: Sorted onset sample indices, or ``None`` when the experiment has no
            stimulus annotations or the file predates onsets being stored.
        """
        wanted = (
            condition.value if isinstance(condition, ConditionVariants) else condition
        )
        onsets = self.extras.get(f"{ONSETS_EXTRA_PREFIX}{wanted}")
        return None if onsets is None else np.asarray(onsets)

    @property
    def onset_conditions(self) -> list[str]:
        """Conditions this file carries stimulus onsets for, sorted."""
        return sorted(
            key[len(ONSETS_EXTRA_PREFIX) :]
            for key in self.extras
            if key.startswith(ONSETS_EXTRA_PREFIX)
        )

    def times_for(self, condition: ConditionVariants | str) -> np.ndarray:
        """Segment-local time axis of one condition of a time-axis join.

        The join keeps each condition on its own time base, so a segment's times
        restart at zero rather than continuing the concatenated axis.

        :param condition: Condition whose segment is wanted.
        :return: ``(T_c,)`` times in seconds, starting at zero.
        """
        segment = self.segment_slice(condition)
        return np.arange(segment.stop - segment.start) / float(self.sfreq)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _resolve_array_names(
    arrays: Mapping[IvaComponentArrays | str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Normalise array keys to canonical names.

    :param arrays: Mapping keyed by :class:`IvaComponentArrays` members or values.
    :return: Mapping keyed by the canonical string names.
    :raises ValueError: If a key is not a known component array name, or the
        mapping is empty.
    """
    if not arrays:
        raise ValueError(
            "No component arrays to store; pass at least one of "
            f"{[member.value for member in IvaComponentArrays]}."
        )
    resolved: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if isinstance(key, IvaComponentArrays):
            member = key
        else:
            try:
                member = IvaComponentArrays(str(key))
            except ValueError as exc:
                raise ValueError(
                    f"{key!r} is not a component array name; expected one of "
                    f"{[m.value for m in IvaComponentArrays]}."
                ) from exc
        resolved[member.value] = np.asarray(value)
    return resolved


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    n_subjects: int,
    n_pca: int,
    axis_lengths: Mapping[str, int | None],
) -> None:
    """Check every array against the row count, component count and axes.

    :param arrays: Canonical name → array.
    :param n_subjects: Expected leading-axis length.
    :param n_pca: Expected component-axis length.
    :param axis_lengths: Axis name (``freq``/``time``/``channel``) → its stored
        length, or ``None`` when that axis was not supplied.
    :raises ValueError: On the first mismatch, naming the array and the axis.
    """
    for name, value in arrays.items():
        member = IvaComponentArrays(name)
        expected_axes = _ARRAY_AXES[member]
        expected_ndim = 2 + len(expected_axes)
        if value.ndim != expected_ndim:
            raise ValueError(
                f"{name} must be {expected_ndim}-D "
                f"(recording, component, {', '.join(expected_axes)}); "
                f"got shape {value.shape}."
            )
        if value.shape[0] != n_subjects:
            raise ValueError(
                f"{name} has {value.shape[0]} recording(s) but "
                f"{n_subjects} participant label(s) were given."
            )
        if value.shape[1] != n_pca:
            raise ValueError(
                f"{name} has {value.shape[1]} component(s) but n_pca is {n_pca}."
            )
        for axis_name, length in zip(expected_axes, value.shape[2:]):
            expected = axis_lengths.get(axis_name)
            if expected is not None and expected != length:
                raise ValueError(
                    f"{name} has {length} {axis_name} bin(s) but the stored "
                    f"{axis_name} axis has {expected}."
                )


def save_iva_components(
    *,
    experiment: ExperimentNames,
    condition: ConditionVariants,
    variant: IvaVariants,
    music_type: MusicTypeVariants,
    band: str | None,
    n_pca: int,
    sfreq: float,
    participants: Sequence[str],
    arrays: Mapping[IvaComponentArrays | str, np.ndarray],
    subject_conditions: Sequence[str] | None = None,
    freqs: np.ndarray | None = None,
    times: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    segment_conditions: Sequence[ConditionVariants | str] | None = None,
    segment_lengths: Sequence[int] | None = None,
    extras: Mapping[str, np.ndarray] | None = None,
    dtype: np.dtype | str = np.float32,
    processed_data_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> Path:
    """Write one decomposition's component products to the store.

    Overwrites an existing entry for the same
    ``(experiment, condition, variant, music_type, band, n_pca)``: that tuple names
    a rerun of the same analysis, and keeping a stale copy of one would be worse
    than replacing it.

    :param experiment: Experiment the run belongs to.
    :param condition: Condition of the run; becomes the subdirectory.
    :param variant: Decomposition variant.
    :param music_type: Music type of the run.
    :param band: Frequency band the run was restricted to, or ``None``.
    :param n_pca: Per-recording PCA dimension, i.e. the component count.
    :param sfreq: Sampling frequency of the time axis, in Hz.
    :param participants: Participant label per recording, in row order. Required:
        without it the subject axis cannot be interpreted later, which is the whole
        point of the store.
    :param arrays: Canonical name (or :class:`IvaComponentArrays` member) →
        ``(S, K, ...)`` array. At least one entry.
    :param subject_conditions: Condition of each recording, in row order. ``None``
        fills in *condition* for every row, which is right for a single-condition
        run and for a time-axis join; pass it explicitly for a subject-axis join.
    :param freqs: ``(F,)`` frequency axis in Hz, when any array has one.
    :param times: ``(T,)`` time axis in seconds, when any array has one.
    :param channel_names: Channel names of the topography axis, when known.
    :param segment_conditions: For a time-axis join, the condition of each time
        segment in order. Requires *segment_lengths*.
    :param segment_lengths: Sample count of each segment, aligned with
        *segment_conditions*.
    :param extras: Free-form per-run diagnostics to keep alongside the components.
        Stored verbatim and not validated.
    :param dtype: Floating dtype the component arrays are cast to. The default
        ``float32`` halves a store that is dominated by ``(S, K, F, T)`` maps and
        is well past the precision of anything downstream reads off them; pass
        ``float64`` to keep the computed values bit-exact.
    :param processed_data_dir: Processed-data root, or ``None`` for the project's.
    :param logger: Logger to report the write on; ``None`` uses this module's.
    :return: Path written.
    :raises ValueError: If the participant labels, the arrays and the axes do not
        describe one consistent decomposition.
    """
    log = logger or _logger

    labels = [str(participant) for participant in participants]
    if not labels:
        raise ValueError("participants is empty; the subject axis needs labels.")

    resolved = _resolve_array_names(arrays)

    if subject_conditions is None:
        row_conditions = [condition.value] * len(labels)
    else:
        row_conditions = [
            value.value if isinstance(value, ConditionVariants) else str(value)
            for value in subject_conditions
        ]
    if len(row_conditions) != len(labels):
        raise ValueError(
            f"subject_conditions has {len(row_conditions)} entries but there are "
            f"{len(labels)} participant label(s)."
        )

    freq_axis = None if freqs is None else np.asarray(freqs, dtype=float)
    time_axis = None if times is None else np.asarray(times, dtype=float)
    channels = None if channel_names is None else [str(name) for name in channel_names]
    _validate_arrays(
        resolved,
        n_subjects=len(labels),
        n_pca=n_pca,
        axis_lengths={
            "freq": None if freq_axis is None else freq_axis.size,
            "time": None if time_axis is None else time_axis.size,
            "channel": None if channels is None else len(channels),
        },
    )

    if (segment_conditions is None) != (segment_lengths is None):
        raise ValueError(
            "segment_conditions and segment_lengths must be given together; got "
            f"{segment_conditions!r} and {segment_lengths!r}."
        )
    segments: list[str] = []
    lengths: list[int] = []
    if segment_conditions is not None and segment_lengths is not None:
        segments = [
            value.value if isinstance(value, ConditionVariants) else str(value)
            for value in segment_conditions
        ]
        lengths = [int(length) for length in segment_lengths]
        if len(segments) != len(lengths):
            raise ValueError(
                f"segment_conditions has {len(segments)} entries but "
                f"segment_lengths has {len(lengths)}."
            )
        if time_axis is not None and sum(lengths) != time_axis.size:
            raise ValueError(
                f"Segments total {sum(lengths)} samples but the stored time axis "
                f"has {time_axis.size}; the segments do not describe it."
            )

    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION),
        "variant": np.asarray(variant.value),
        "experiment": np.asarray(experiment.value),
        "condition": np.asarray(condition.value),
        "music_type": np.asarray(music_type.value),
        "band": np.asarray("" if band is None else str(band)),
        "n_pca": np.asarray(int(n_pca)),
        "sfreq": np.asarray(float(sfreq)),
        "participants": np.asarray(labels, dtype=str),
        "subject_conditions": np.asarray(row_conditions, dtype=str),
    }
    if freq_axis is not None:
        payload["freqs"] = freq_axis
    if time_axis is not None:
        payload["times"] = time_axis
    if channels is not None:
        payload["channel_names"] = np.asarray(channels, dtype=str)
    if segments:
        payload["segment_conditions"] = np.asarray(segments, dtype=str)
        payload["segment_lengths"] = np.asarray(lengths, dtype=np.int64)
    for name, value in resolved.items():
        payload[_ARRAY_PREFIX + name] = value.astype(dtype, copy=False)
    for name, value in (extras or {}).items():
        payload[_EXTRA_PREFIX + str(name)] = np.asarray(value)

    path = iva_results_path(
        experiment,
        condition,
        variant,
        music_type,
        band,
        n_pca,
        processed_data_dir=processed_data_dir,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    log.info(
        f"[{condition.value}_{music_type.value}] stored {variant.value} components "
        f"({', '.join(sorted(resolved))}) for {len(labels)} recording(s) as "
        f"{path.name} ({path.stat().st_size / 1e6:.1f} MB) in {path.parent}"
    )
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def load_iva_components(
    path: Path | str | None = None,
    *,
    experiment: ExperimentNames | None = None,
    condition: ConditionVariants | None = None,
    variant: IvaVariants | None = None,
    music_type: MusicTypeVariants | None = None,
    band: str | None = None,
    n_pca: int | None = None,
    processed_data_dir: Path | None = None,
) -> IvaComponentResults:
    """Read one store entry, with its participant mapping.

    Either pass *path* directly, or the descriptor of the run
    (*experiment*, *condition*, *variant*, *music_type*, *n_pca*, and *band* when
    the run was band-restricted) and let the path be derived.

    :param path: Path to a stored ``.npz``. When given, every descriptor argument
        is ignored.
    :param experiment: Experiment of the wanted run.
    :param condition: Condition of the wanted run.
    :param variant: Decomposition variant of the wanted run.
    :param music_type: Music type of the wanted run.
    :param band: Frequency band of the wanted run, or ``None`` for broadband.
    :param n_pca: Per-recording PCA dimension of the wanted run.
    :param processed_data_dir: Processed-data root, or ``None`` for the project's.
    :return: The stored components and their bookkeeping.
    :raises ValueError: If neither a path nor a complete descriptor is given, or
        the file's key set is not a readable store entry.
    :raises FileNotFoundError: If the resolved path does not exist.
    """
    if path is None:
        missing = [
            name
            for name, value in (
                ("experiment", experiment),
                ("condition", condition),
                ("variant", variant),
                ("music_type", music_type),
                ("n_pca", n_pca),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"Pass either `path` or the full run descriptor; missing {missing}."
            )
        path = iva_results_path(
            experiment,
            condition,
            variant,
            music_type,
            band,
            n_pca,
            processed_data_dir=processed_data_dir,
        )
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No stored IVA components at {path}.")

    with np.load(path, allow_pickle=False) as stored:
        keys = set(stored.files)
        required = {
            "format_version",
            "variant",
            "experiment",
            "condition",
            "music_type",
            "n_pca",
            "sfreq",
            "participants",
            "subject_conditions",
        }
        absent = sorted(required - keys)
        if absent:
            raise ValueError(
                f"{path} is not a readable IVA component store: missing {absent}."
            )
        version = int(stored["format_version"])
        if version > FORMAT_VERSION:
            raise ValueError(
                f"{path} was written with store format version {version}, newer "
                f"than the supported {FORMAT_VERSION}; update the codebase."
            )
        band_value = str(stored["band"]) if "band" in keys else ""
        results = IvaComponentResults(
            variant=IvaVariants(str(stored["variant"])),
            experiment=ExperimentNames(str(stored["experiment"])),
            condition=ConditionVariants(str(stored["condition"])),
            music_type=MusicTypeVariants(str(stored["music_type"])),
            band=band_value or None,
            n_pca=int(stored["n_pca"]),
            sfreq=float(stored["sfreq"]),
            participants=tuple(str(name) for name in stored["participants"]),
            subject_conditions=tuple(
                str(name) for name in stored["subject_conditions"]
            ),
            arrays={
                key[len(_ARRAY_PREFIX) :]: stored[key]
                for key in sorted(keys)
                if key.startswith(_ARRAY_PREFIX)
            },
            freqs=stored["freqs"] if "freqs" in keys else None,
            times=stored["times"] if "times" in keys else None,
            channel_names=(
                tuple(str(name) for name in stored["channel_names"])
                if "channel_names" in keys
                else None
            ),
            segment_conditions=(
                tuple(str(name) for name in stored["segment_conditions"])
                if "segment_conditions" in keys
                else ()
            ),
            segment_lengths=(
                tuple(int(length) for length in stored["segment_lengths"])
                if "segment_lengths" in keys
                else ()
            ),
            extras={
                key[len(_EXTRA_PREFIX) :]: stored[key]
                for key in sorted(keys)
                if key.startswith(_EXTRA_PREFIX)
            },
            path=path,
        )
    if not results.arrays:
        raise ValueError(f"{path} holds no component arrays.")
    _logger.info(
        f"Loaded {results.variant.value} components for {results.label} from "
        f"{path.name}: {sorted(results.arrays)}, {results.n_subjects} recording(s), "
        f"{results.n_components} component(s)."
    )
    return results


def load_many_iva_components(
    paths: Iterable[Path | str],
) -> list[IvaComponentResults]:
    """Read several store entries.

    :param paths: Paths to stored ``.npz`` files, e.g. from
        :func:`list_iva_results`.
    :return: The loaded results, in the order given.
    """
    return [load_iva_components(path) for path in paths]
