"""
Annotation-based stimulus alignment across participants.

Recordings of the same experiment do not start their stimulus sequence with the
same inter-stimulus padding, so a given stimulus onset (annotated e.g. as ``fam+``)
falls at a different time in every participant. This module equalizes the
inter-stimulus intervals so that stimulus ``k`` lands at the *same sample index* in
every participant of a group, enabling sample-locked cross-participant analysis
(e.g. ISC).

Strategy (per interval between consecutive onsets):
    * The target interval length is the **minimum** of that interval across the
      group (the shortest participant is never trimmed).
    * The last ``keep_tail`` of continuous data immediately preceding the next
      onset is preserved (the pre-stimulus baseline), as is the post-stimulus
      response of the current onset. Only the *middle* of each over-long interval
      is removed.
    * The continuous EEG is physically cropped and spliced, so the resulting
      recordings have equal length and identical onset sample positions.

The recording edges (before the first onset and after the last) are equalized to a
fixed window as well. When recordings have differing stimulus counts, the surplus
(trailing) stimuli are dropped and the post-window after the last common onset is
capped so it never reaches a recording's first surplus onset — preventing an
unaligned stimulus from being spliced into the output.

Key pieces:
    - ``resolve_stimulus_marker``: the experiment's marker label plus its
      per-recording marker→onset calibration (see
      :mod:`src.preprocessing.marker_shift`). Every consumer of an offset should
      start here.
    - ``get_stimulus_onset_samples``: extract onset sample indices from a Raw.
    - ``StimulusAligner``: pure planner — from per-recording onset samples computes
      the keep-segments and the aligned onset positions (no MNE dependency).
    - ``apply_keep_segments`` / ``align_raws``: apply the plan to MNE ``Raw`` objects.
"""

import logging
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import mne

from src.definitions.constants import AssrEpoch, ProjectPaths
from src.definitions.fields import ExperimentNames
from src.preprocessing.marker_shift import load_marker_shift_table
from src.utils.logging_config import LoggerMixin

_logger = logging.getLogger(__name__)

# Default annotation label marking a stimulus onset.
DEFAULT_STIMULUS_LABEL = "fam+"


@dataclass(frozen=True)
class StimulusMarker:
    """
    How a stimulus annotation relates to the acoustic event it marks.

    The label and the offset belong together: reading the annotations without
    applying the offset silently mis-times every onset-locked analysis, so they
    are resolved as one object rather than from two parallel registries.

    The offset is resolved **per recording**. In the ASSR dataset the marker error
    is a file-format artefact that differs from recording to recording (see
    :mod:`src.preprocessing.marker_shift`), so a single constant cannot describe
    it; :attr:`onset_offset_s` is only the fallback for recordings the calibration
    does not cover. Always go through :meth:`onset_offset_for` — reading
    :attr:`onset_offset_s` directly re-introduces the constant-offset bug for every
    calibrated recording.
    """

    # Annotation description marking a stimulus (case-insensitive).
    label: str
    # Fallback seconds added to the marker time to obtain the true stimulus onset,
    # used for recordings absent from `per_recording_offsets`. Negative means the
    # marker lags the stimulus. 0.0 = the marker *is* the onset.
    onset_offset_s: float = 0.0
    # Calibrated per-recording offsets, keyed by recording file stem.
    per_recording_offsets: Mapping[str, float] = field(default_factory=dict)

    def onset_offset_for(self, filename: str | None) -> float:
        """
        The marker→onset offset to apply to one recording.

        :param filename: Recording filename, with or without extension. ``None``
            (no recording in hand) yields the fallback.
        :return: Seconds to add to this recording's marker times.
        """
        if filename is None:
            return self.onset_offset_s
        return self.per_recording_offsets.get(Path(filename).stem, self.onset_offset_s)

    def offsets_for(self, filenames: Sequence[str | None]) -> list[float]:
        """
        Per-recording offsets for a group, in the order given.

        :param filenames: Recording filenames of the group.
        :return: One offset per filename.
        """
        return [self.onset_offset_for(filename) for filename in filenames]


# Per-experiment stimulus marker. Experiments listed here use a stimulus-aware
# coarse crop during preprocessing; others fall back to the fixed start/end crop.
#
# These entries carry the *fallback* offset only. Use `resolve_stimulus_marker` to
# obtain a marker with the per-recording calibration attached; read this registry
# directly only to test whether an experiment has stimulus annotations at all.
EXPERIMENT_STIMULUS_MARKERS = {
    ExperimentNames.ASSR: StimulusMarker(
        label=DEFAULT_STIMULUS_LABEL,
        onset_offset_s=AssrEpoch.MARKER_ONSET_OFFSET_S,
    ),
}


@lru_cache(maxsize=None)
def resolve_stimulus_marker(
    experiment_name: ExperimentNames,
) -> StimulusMarker | None:
    """
    The stimulus marker of an experiment, with its per-recording calibration loaded.

    This is the single entry point every consumer should use. The calibration CSV
    is read once per experiment and cached, so the many places that need an offset
    do not each pay for a file read — and, more importantly, cannot each decide
    differently whether to apply one.

    :param experiment_name: Experiment whose marker is requested.
    :return: The marker with :attr:`StimulusMarker.per_recording_offsets` populated,
        or ``None`` for experiments without stimulus annotations.
    """
    marker = EXPERIMENT_STIMULUS_MARKERS.get(experiment_name)
    if marker is None:
        return None

    offsets = load_marker_shift_table(
        ProjectPaths.get_marker_shift_mapping_path(experiment_name)
    )
    if offsets:
        _logger.info(
            f"Loaded per-recording stimulus-marker offsets for "
            f"{len(offsets)} {experiment_name.value} recording(s); recordings "
            f"without an entry fall back to {marker.onset_offset_s:+.4f} s."
        )
    return replace(marker, per_recording_offsets=offsets)


def get_stimulus_onset_samples(
    raw: mne.io.Raw,
    stimulus_label: str = DEFAULT_STIMULUS_LABEL,
    onset_offset_s: float = 0.0,
) -> np.ndarray:
    """
    Extract the stimulus-onset sample indices of a recording.

    This is the single point through which every consumer reads stimulus onsets,
    so it is also where the marker→onset offset is applied — exactly once.

    :param raw: Recording whose annotations are searched.
    :param stimulus_label: Annotation description marking a stimulus
        (case-insensitive, surrounding whitespace ignored).
    :param onset_offset_s: Seconds added to each marker time to obtain the true
        stimulus onset (see :class:`StimulusMarker`). Defaults to ``0.0``, i.e.
        the raw marker positions — pass the experiment's registered offset from
        :data:`EXPERIMENT_STIMULUS_MARKERS` to get acoustic onsets.
    :return: Sorted array of onset sample indices (relative to the first data
        sample of ``raw``).
    :raises ValueError: If the offset moves an onset outside the recording.
        Dropping such onsets is not an option: stimuli are paired across
        recordings by their order, so a recording losing its first onset would
        silently misalign the whole group.
    """
    target = stimulus_label.strip().lower()
    onset_seconds = [
        onset
        for onset, description in zip(
            raw.annotations.onset, raw.annotations.description
        )
        if description.strip().lower() == target
    ]
    if not onset_seconds:
        return np.array([], dtype=int)

    # Annotation onsets are stored relative to ``orig_time``; ``first_time`` maps
    # them onto the sample grid (which starts at the first data sample).
    relative_seconds = (
        np.asarray(onset_seconds, dtype=float) - raw.first_time + onset_offset_s
    )
    samples = np.sort(raw.time_as_index(relative_seconds, use_rounding=True))

    # Onsets are also used as half-open segment bounds, so ``n_times`` itself is a
    # legal (exclusive) position; anything beyond it, or negative, is not.
    if len(samples) and (samples[0] < 0 or samples[-1] > raw.n_times):
        raise ValueError(
            f"Applying an onset offset of {onset_offset_s:+.3f} s moves "
            f"'{stimulus_label}' onsets outside the recording (samples "
            f"[{samples[0]}, {samples[-1]}] for {raw.n_times} samples). The "
            "recording does not retain enough data around its stimuli; re-run "
            "the coarse crop with a margin larger than the offset."
        )
    return samples


def coarse_crop_to_stimulus_span(
    raw: mne.io.Raw,
    stimulus_label: str = DEFAULT_STIMULUS_LABEL,
    trim_sec: float = 10.0,
    min_keep_sec: float = 0.5,
    onset_offset_s: float = 0.0,
    logger=None,
) -> mne.io.Raw:
    """
    Coarsely crop a recording around its stimulus span (single contiguous crop).

    Intended as an early, *pre-filtering* trim that removes the typically-noisy
    recording lead-in/lead-out without losing any stimulus or its surrounding
    context. Being a single crop (no splicing), it keeps the data continuous so
    filtering and ICA are not corrupted by seams.

    Each end is trimmed by *up to* ``trim_sec``, but never closer than
    ``min_keep_sec`` to the first/last onset. So the recording keeps
    ``max(min_keep_sec, pre - trim_sec)`` before its first onset and
    ``max(min_keep_sec, post - trim_sec)`` after its last onset, where ``pre`` /
    ``post`` are its own lead-in / lead-out. This is per-recording; the cross-subject
    alignment of the first/last onset is finished later by the fine stimulus
    alignment (which keeps the shortest remaining lead-in/lead-out across the group).

    :param raw: Recording to crop (not modified in place).
    :param stimulus_label: Annotation description marking a stimulus onset.
    :param trim_sec: Maximum amount removed from each end of the recording.
    :param min_keep_sec: Minimum data kept before the first and after the last onset.
        Applies to the true onsets, so it must exceed ``abs(onset_offset_s)`` for a
        marker that lags its stimulus, or the crop would cut into the first response.
    :param onset_offset_s: Marker→onset offset, see
        :func:`get_stimulus_onset_samples`.
    :param logger: Optional logger instance. Falls back to module-level logger.
    :return: Cropped copy of ``raw``. If no stimulus annotations are found, returns
        an unchanged copy (with a warning).
    """
    log = logger or _logger

    onsets = get_stimulus_onset_samples(raw, stimulus_label, onset_offset_s)
    if len(onsets) == 0:
        log.warning(
            f"No '{stimulus_label}' annotations found; skipping stimulus-aware crop."
        )
        return raw.copy()

    sfreq = raw.info["sfreq"]
    trim = int(round(trim_sec * sfreq))
    min_keep = int(round(min_keep_sec * sfreq))
    last_sample = raw.n_times - 1

    first, last = int(onsets[0]), int(onsets[-1])
    # Remove up to `trim` from each end, always keeping >= min_keep around the
    # first/last onset.
    start = min(trim, max(0, first - min_keep))
    removed_end = min(trim, max(0, (last_sample - last) - min_keep))
    end = last_sample - removed_end
    log.info(
        f"Coarse stimulus crop: removed {start / sfreq:.2f}s from start / "
        f"{removed_end / sfreq:.2f}s from end (keeping {(first - start) / sfreq:.2f}s "
        f"before first and {(end - last) / sfreq:.2f}s after last of {len(onsets)} "
        f"'{stimulus_label}' onsets); kept samples [{start}, {end}]."
    )
    return raw.copy().crop(tmin=start / sfreq, tmax=end / sfreq, include_tmax=True)


class StimulusAligner(LoggerMixin):
    """
    Plans the trimming needed to align stimulus onsets across a group of recordings.

    The aligner is pure (no MNE dependency): it consumes per-recording onset sample
    indices and recording lengths, and produces, for each recording, the list of
    ``(start, end)`` sample segments to keep, plus the common aligned onset
    positions. Apply the plan with :func:`apply_keep_segments`.
    """

    def __init__(
        self,
        onset_samples: list[np.ndarray],
        recording_lengths: list[int],
        sfreq: float,
        keep_tail_sec: float = 0.1,
        pre_window_sec: float | None = None,
        post_window_sec: float | None = None,
    ):
        """
        :param onset_samples: Per recording, the sorted stimulus-onset sample indices.
        :param recording_lengths: Per recording, the number of samples (``raw.n_times``).
        :param sfreq: Sampling frequency in Hz (assumed identical across recordings).
        :param keep_tail_sec: Continuous data preserved immediately before each onset.
        :param pre_window_sec: Optional cap on the window kept before the first onset.
            ``None`` (default) keeps the per-subject shortest available lead-in (pure
            group-minimum); a value caps the edge at that many seconds.
        :param post_window_sec: Optional cap on the window kept after the last onset.
            ``None`` (default) keeps the per-subject shortest available lead-out.
        :raises ValueError: If fewer than two onsets are common to all recordings.
        """
        if len(onset_samples) != len(recording_lengths):
            raise ValueError(
                "onset_samples and recording_lengths must have the same length."
            )
        if len(onset_samples) == 0:
            raise ValueError("At least one recording is required.")

        self.sfreq = sfreq
        self.keep_tail = self._to_samples(keep_tail_sec)
        self.pre_window = (
            self._to_samples(pre_window_sec) if pre_window_sec is not None else None
        )
        self.post_window = (
            self._to_samples(post_window_sec) if post_window_sec is not None else None
        )

        self.onset_samples = [np.asarray(o, dtype=int) for o in onset_samples]
        self.recording_lengths = list(recording_lengths)
        self.original_counts = [len(o) for o in self.onset_samples]

        # Pair stimuli by order from the first onset; use the common (minimum) count.
        self.common_count = min(self.original_counts)
        if self.common_count < 2:
            raise ValueError(
                "Need at least two common stimulus onsets to align intervals; "
                f"got common count {self.common_count}."
            )
        dropped = [c - self.common_count for c in self.original_counts]
        if any(dropped):
            self.logger.info(
                f"Stimulus counts differ across recordings {self.original_counts}; "
                f"using common count {self.common_count}, dropping trailing "
                f"extras {dropped}."
            )

        self._onsets = [o[: self.common_count] for o in self.onset_samples]

        # Per-interval target length (minimum across recordings), and edge windows
        # clamped to what every recording can actually provide.
        self.interval_targets = self._compute_interval_targets()
        self.pre_target, self.post_target = self._compute_edge_targets()

        # Aligned onset positions in the spliced output (identical for all recordings).
        self.aligned_onset_samples = self._compute_aligned_onsets()
        self.total_length = int(
            self.pre_target + self.interval_targets.sum() + self.post_target
        )

        # Per-recording keep-segments.
        self.keep_segments = [
            self._compute_segments_for_recording(idx)
            for idx in range(len(self._onsets))
        ]

    def _to_samples(self, seconds: float) -> int:
        """Convert a duration in seconds to a non-negative integer sample count."""
        return int(round(seconds * self.sfreq))

    def _compute_interval_targets(self) -> np.ndarray:
        """
        Per interval, the minimum length (in samples) across all recordings.

        :return: Array of length ``common_count - 1`` with target interval lengths.
        """
        # Shape: (n_recordings, common_count - 1)
        gaps = np.stack([np.diff(o) for o in self._onsets], axis=0)
        return gaps.min(axis=0).astype(int)

    def _compute_edge_targets(self) -> tuple[int, int]:
        """
        Pre/post edge windows, clamped to what every recording can provide.

        Stimuli are paired from the first onset, so any surplus stimuli are
        trailing: they fall *after* the last common onset. The post-window is
        therefore additionally capped so it never reaches a recording's first
        surplus (non-aligned) onset — it stops at the last sample before it.
        Without this cap, a long enough post-window would splice an unaligned
        stimulus response into the output and pollute the cross-subject analysis.

        :return: Tuple of (pre_target, post_target) in samples.
        """
        pre_available = min(int(o[0]) for o in self._onsets)

        post_bounds: list[int] = []
        for full_onsets, length in zip(self.onset_samples, self.recording_lengths):
            last_common = int(full_onsets[self.common_count - 1])
            bound = length - last_common
            if len(full_onsets) > self.common_count:
                # First surplus onset; keep only up to the sample before it so the
                # exclusive keep-end lands exactly on the surplus onset, excluding it.
                next_onset = int(full_onsets[self.common_count])
                bound = min(bound, next_onset - last_common)
            post_bounds.append(bound)
        post_available = min(post_bounds)
        pre_target = (
            pre_available
            if self.pre_window is None
            else min(self.pre_window, pre_available)
        )
        post_target = (
            post_available
            if self.post_window is None
            else min(self.post_window, post_available)
        )
        return int(pre_target), int(post_target)

    def _compute_aligned_onsets(self) -> np.ndarray:
        """
        Onset sample positions in the spliced output (shared by all recordings).

        :return: Array of length ``common_count``.
        """
        positions = np.empty(self.common_count, dtype=int)
        positions[0] = self.pre_target
        positions[1:] = self.pre_target + np.cumsum(self.interval_targets)
        return positions

    def _compute_segments_for_recording(self, idx: int) -> list[tuple[int, int]]:
        """
        Build the ``(start, end)`` sample segments to keep for one recording.

        Segments are the complement of the removed regions within the kept span
        ``[first_onset - pre_target, last_onset + post_target]``.

        :param idx: Recording index.
        :return: List of half-open ``(start, end)`` sample ranges to keep.
        """
        onsets = self._onsets[idx]
        keep_start = int(onsets[0]) - self.pre_target
        keep_end = int(onsets[-1]) + self.post_target

        removed: list[tuple[int, int]] = []
        for i in range(self.common_count - 1):
            target = int(self.interval_targets[i])
            tail = min(self.keep_tail, target)
            remove_start = int(onsets[i]) + (target - tail)
            remove_end = int(onsets[i + 1]) - tail
            if remove_end > remove_start:
                removed.append((remove_start, remove_end))

        # Complement of removed regions within [keep_start, keep_end].
        segments: list[tuple[int, int]] = []
        cursor = keep_start
        for remove_start, remove_end in removed:
            if remove_start > cursor:
                segments.append((cursor, remove_start))
            cursor = max(cursor, remove_end)
        if cursor < keep_end:
            segments.append((cursor, keep_end))

        kept = sum(end - start for start, end in segments)
        assert kept == self.total_length, (
            f"Recording {idx}: kept {kept} samples, expected {self.total_length}."
        )
        return segments


def apply_keep_segments(raw: mne.io.Raw, segments: list[tuple[int, int]]) -> mne.io.Raw:
    """
    Crop a recording to the given sample segments and splice them together.

    :param raw: Recording to trim (not modified in place).
    :param segments: Half-open ``(start, end)`` sample ranges to keep, in order.
    :return: New spliced ``Raw`` containing only the kept segments (annotations,
        e.g. stimulus onsets, are preserved and shifted accordingly).
    """
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times

    pieces = []
    for start, end in segments:
        start = max(0, start)
        end = min(n_times, end)
        if end <= start:
            continue
        # crop selects samples [start, end - 1] inclusive -> (end - start) samples.
        pieces.append(
            raw.copy().crop(
                tmin=start / sfreq, tmax=(end - 1) / sfreq, include_tmax=True
            )
        )

    spliced = mne.concatenate_raws(pieces)
    return spliced


def apply_keep_segments_to_array(
    data: np.ndarray,
    segments: list[tuple[int, int]],
) -> np.ndarray:
    """
    Trim an array along its last axis to the given sample segments and concatenate.

    Numpy counterpart to :func:`apply_keep_segments`: applies the same
    keep-segments plan produced by :class:`StimulusAligner` to an arbitrary
    array rather than an MNE ``Raw`` object.  Use this when wavelet transforms
    should be computed on the full continuous recording (to avoid edge artifacts
    at splice points) and the trimming is applied afterwards.

    :param data: Array whose *last* axis is the time axis (e.g.
        ``(n_channels, n_times)`` or ``(n_channels, n_freqs, n_times)``).
    :param segments: Half-open ``(start, end)`` sample ranges to keep, in order.
        Produced by :attr:`StimulusAligner.keep_segments` for one recording.
    :return: Concatenated array containing only the kept time ranges.
    :raises ValueError: If no valid segments remain after clamping to array bounds.
    """
    n_times = data.shape[-1]
    pieces = []
    for start, end in segments:
        start = max(0, start)
        end = min(n_times, end)
        if end > start:
            pieces.append(data[..., start:end])
    if not pieces:
        raise ValueError("No valid segments to keep after clamping to array bounds.")
    return np.concatenate(pieces, axis=-1)


def resolve_per_recording_offsets(
    onset_offset_s: float | Sequence[float],
    n_recordings: int,
) -> list[float]:
    """
    Expand an offset argument into one offset per recording.

    :param onset_offset_s: A single offset shared by every recording, or one
        offset per recording in the same order.
    :param n_recordings: Number of recordings the offsets must cover.
    :return: List of ``n_recordings`` offsets.
    :raises ValueError: If a sequence is given whose length does not match. Silently
        recycling or truncating it would misalign the group by hundreds of
        milliseconds without any visible failure.
    """
    if isinstance(onset_offset_s, (int, float, np.floating, np.integer)):
        return [float(onset_offset_s)] * n_recordings
    offsets = [float(offset) for offset in onset_offset_s]
    if len(offsets) != n_recordings:
        raise ValueError(
            f"Got {len(offsets)} onset offsets for {n_recordings} recordings; "
            "they must correspond one-to-one and in the same order."
        )
    return offsets


def align_raws(
    raws: list[mne.io.Raw],
    stimulus_label: str = DEFAULT_STIMULUS_LABEL,
    keep_tail_sec: float = 0.1,
    pre_window_sec: float | None = None,
    post_window_sec: float | None = None,
    onset_offset_s: float | Sequence[float] = 0.0,
) -> tuple[list[mne.io.Raw], StimulusAligner]:
    """
    Align stimulus onsets across a group of recordings by trimming inter-stimulus
    intervals and splicing the EEG.

    :param raws: Recordings to align (all from the same group, e.g. one
        condition/music type). Assumed to share the sampling frequency.
    :param stimulus_label: Annotation description marking a stimulus onset.
    :param keep_tail_sec: Continuous data preserved immediately before each onset.
    :param pre_window_sec: Optional cap on the window kept before the first onset;
        ``None`` keeps the per-subject shortest available lead-in.
    :param post_window_sec: Optional cap on the window kept after the last onset;
        ``None`` keeps the per-subject shortest available lead-out.
    :param onset_offset_s: Marker→onset offset, see
        :func:`get_stimulus_onset_samples`. The splice is planned around the true
        onsets, so ``keep_tail_sec`` really is the pre-stimulus baseline. Pass one
        offset per recording (same order as *raws*) when the marker error differs
        between recordings, which is what makes their onsets land on a common
        acoustic time rather than a common *marker* time — see
        :class:`StimulusMarker`.
    :return: Tuple of (aligned recordings, the fitted :class:`StimulusAligner`).
    :raises ValueError: If a sequence of offsets does not match the recording count.
    """
    sfreq = raws[0].info["sfreq"]
    offsets = resolve_per_recording_offsets(onset_offset_s, len(raws))
    onset_samples = [
        get_stimulus_onset_samples(raw, stimulus_label, offset)
        for raw, offset in zip(raws, offsets)
    ]
    recording_lengths = [raw.n_times for raw in raws]

    aligner = StimulusAligner(
        onset_samples,
        recording_lengths,
        sfreq,
        keep_tail_sec=keep_tail_sec,
        pre_window_sec=pre_window_sec,
        post_window_sec=post_window_sec,
    )

    aligned = [
        apply_keep_segments(raw, segments)
        for raw, segments in zip(raws, aligner.keep_segments)
    ]
    return aligned, aligner
