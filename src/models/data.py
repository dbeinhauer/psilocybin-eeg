"""
Datasets for the stage-08 contrastive synchrony model.

Read straight from the per-subject wavelet store (:mod:`src.io.wavelet_store`), which
stage 03 writes and every other stage reads. :func:`build_dataset` takes one
``<Condition>_<MusicType>`` label and, per participant and channel, reads only the
selected frequency bins, averages power into frames of ``frame_rate`` Hz, takes
``log10``, pools the bins into bands, normalises, and folds channel x frequency into one
feature axis — ``(subjects, features, frames)``, the model input. Participants are read
in parallel threads; peak memory beyond the result is one channel block per thread.

Everything here is numpy; nothing needs torch.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.definitions.constants import ProjectPaths
from src.definitions.fields import (
    ConditionVariants,
    ExclusionCategories,
    ExperimentNames,
    MusicTypeVariants,
    SingleDataMetadata,
)
from src.definitions.frequency import FREQUENCY_BANDS
from src.filtering.dataset_filter import DatasetFilter
from src.io.wavelet_store import WaveletSet, find_wavelet_set, wavelet_label_dir
from src.models.config import DataConfig

_logger = logging.getLogger(__name__)


def recording_label(condition: ConditionVariants, music_type: MusicTypeVariants) -> str:
    return f"{condition.value}_{music_type.value}"


def frame_average(power: np.ndarray, factor: int) -> np.ndarray:
    """Mean over consecutive blocks of *factor* samples on the last axis.

    A trailing block shorter than *factor* is dropped, so frame ``k`` always covers
    samples ``[k * factor, (k + 1) * factor)``.
    """
    n_frames = power.shape[-1] // factor
    blocks = power[..., : n_frames * factor].reshape(
        *power.shape[:-1], n_frames, factor
    )
    return blocks.mean(axis=-1)


def samples_per_frame(sfreq: float, frame_rate: float) -> int:
    """How many wavelet samples one frame averages.

    :raises ValueError: If *frame_rate* does not divide *sfreq*.
    """
    factor = sfreq / frame_rate
    if factor < 1 or abs(factor - round(factor)) > 1e-6:
        raise ValueError(
            f"frame_rate={frame_rate} Hz must divide the wavelets' {sfreq:g} Hz "
            "sampling rate into a whole number of samples per frame."
        )
    return int(round(factor))


def read_log_frames(
    wavelet_set: WaveletSet,
    participants: list[str],
    channels: list[int],
    freq_indices: np.ndarray,
    factor: int,
    n_samples: int | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Frame-averaged ``log10`` power, ``(subjects, channels, freqs, frames)`` float32.

    Power is averaged within each frame *before* the log, so a frame is the log of its
    mean power rather than the mean of log power. Each channel is read and reduced on
    its own, so a thread never holds more than one ``(freqs, samples)`` block.

    :param n_samples: Read only the first N wavelet samples; ``None`` reads all.
    :param n_jobs: Threads reading participants in parallel (``-1`` = one per CPU).
    """
    lengths = {wavelet_set.open(p).shape[-1] for p in participants}
    if len(lengths) > 1:
        raise ValueError(
            f"{wavelet_set.directory.name}: participants have different time axes "
            f"{sorted(lengths)}; one track must share one aligned time base."
        )
    available = lengths.pop()
    keep = available if n_samples is None else min(n_samples, available)
    out = np.empty(
        (len(participants), len(channels), freq_indices.size, keep // factor),
        np.float32,
    )
    tiny = np.finfo(np.float64).tiny

    def _read(row: int) -> None:
        subject = wavelet_set.open(participants[row])
        for column, channel in enumerate(channels):
            block = np.asarray(subject[channel][freq_indices, :keep], dtype=np.float64)
            out[row, column] = np.log10(np.maximum(frame_average(block, factor), tiny))

    workers = (os.cpu_count() or 1) if n_jobs < 0 else n_jobs
    if workers > 1 and len(participants) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_read, range(len(participants))))
    else:
        for row in range(len(participants)):
            _read(row)
    return out


@dataclass(frozen=True)
class SynchronyDataset:
    """One track of one recording label, ready for the model.

    All subjects heard the same stimulus on one time base, so frame ``t`` of every
    subject is the same moment of the track — the fact the contrastive task rests on.
    """

    label: str
    condition: ConditionVariants
    music_type: MusicTypeVariants
    data: np.ndarray  # (subjects, features, frames), float32
    participants: list[str]
    feature_names: list[str]  # "<channel>@<band or bin>", channel-major
    frame_rate: float
    frame_offset: int = 0  # index of the first frame within the whole track

    @property
    def n_subjects(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_frames(self) -> int:
        return self.data.shape[2]


def kept_participants(
    participants: list[str],
    experiment: ExperimentNames,
    condition: ConditionVariants,
    music_type: MusicTypeVariants,
    exclusions: list[ExclusionCategories],
) -> list[int]:
    """Indices of *participants* that survive the exclusion list of *experiment*.

    Delegates to :class:`~src.filtering.dataset_filter.DatasetFilter`, so a blank
    condition or music type in ``data/excluded_participants/<experiment>.csv`` is a
    wildcard exactly as in every other stage.
    """
    excluded = pd.read_csv(
        ProjectPaths.EXCLUDED_PARTICIPANTS_DIR / f"{experiment.value}.csv",
        sep=";",
        dtype=str,
    )
    metadata = pd.DataFrame(
        {
            SingleDataMetadata.PARTICIPANT_ID: participants,
            SingleDataMetadata.CONDITION: condition.value,
            SingleDataMetadata.MUSIC_TYPE: music_type.value,
        }
    )
    kept = DatasetFilter.filter_dataset_by_all_categories(
        metadata, excluded, [music_type], [condition], exclusions
    )
    return kept.index.tolist()


def pool_frequencies(
    data: np.ndarray, freqs: np.ndarray, freq_range: list[float], pooling: str
) -> tuple[np.ndarray, list[str]]:
    """Restrict the frequency axis (axis 2) to *freq_range* and optionally pool it.

    ``"bands"`` averages the bins of each :data:`~src.definitions.frequency.FREQUENCY_BANDS`
    entry (low edge inclusive, high edge exclusive); ``"none"`` keeps every bin.
    Averaging is of log power, i.e. a geometric mean of power within the band.

    :return: The pooled array and one name per entry of its frequency axis.
    :raises ValueError: If no bin or band falls inside *freq_range*.
    """
    low, high = freq_range
    in_range = (freqs >= low) & (freqs <= high)
    if pooling == "none":
        if not in_range.any():
            raise ValueError(f"No frequency bin inside {freq_range} Hz.")
        return data[:, :, in_range], [f"{f:g}Hz" for f in freqs[in_range]]

    pooled, names = [], []
    for band, (band_low, band_high) in FREQUENCY_BANDS.items():
        selected = in_range & (freqs >= band_low) & (freqs < band_high)
        if selected.any():
            pooled.append(data[:, :, selected].mean(axis=2))
            names.append(band)
    if not pooled:
        raise ValueError(f"No frequency band overlaps {freq_range} Hz.")
    return np.stack(pooled, axis=2), names


def zscore_over_time(data: np.ndarray) -> np.ndarray:
    """Standardise every (subject, feature) series over time.

    Removes each subject's own baseline and scale, which would otherwise be the
    easiest thing for a subject's windows to agree on.
    """
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    return (data - mean) / np.where(std > 0, std, 1.0)


def build_dataset(
    config: DataConfig,
    condition: ConditionVariants,
    music_type: MusicTypeVariants,
    *,
    root: Path | None = None,
    n_subjects: int | None = None,
    n_channels: int | None = None,
    n_times: int | None = None,
    n_jobs: int = 1,
) -> SynchronyDataset:
    """Load one recording label from the wavelet store as model input.

    The subset limits take the *first* N of each axis after exclusions, like every
    other stage, and only that extent — and only the frequency bins inside
    ``config.freq_range`` — is read from the memory-mapped files.

    :param config: The ``data`` section of the run config.
    :param root: Processed-data root holding ``<experiment>/wavelets``; ``None`` uses
        ``data/processed``.
    :param n_subjects: Keep the first N kept participants (by id).
    :param n_channels: Keep the first N selected channels.
    :param n_times: Keep the first N frames (at ``config.frame_rate``).
    :param n_jobs: Threads reading participants in parallel.
    :raises ValueError: If a configured channel is missing, or nobody survives.
    """
    experiment = config.experiment_name
    label = recording_label(condition, music_type)
    wavelet_set = find_wavelet_set(wavelet_label_dir(root, experiment, label))
    factor = samples_per_frame(wavelet_set.sfreq, config.frame_rate)

    stored = wavelet_set.participants
    kept = kept_participants(
        stored, experiment, condition, music_type, config.exclusion_categories
    )
    participants = [stored[i] for i in kept][:n_subjects]
    if not participants:
        raise ValueError(f"[{label}] no participant survives the exclusions.")

    names = wavelet_set.channel_names
    if config.channels is None:
        channels = list(range(len(names)))
    else:
        missing = sorted(set(config.channels) - set(names))
        if missing:
            raise ValueError(f"[{label}] channels {missing} are not in the data.")
        channels = [names.index(name) for name in config.channels]
    channels = channels[:n_channels]

    low, high = config.freq_range
    freq_indices = np.flatnonzero(
        (wavelet_set.freqs >= low) & (wavelet_set.freqs <= high)
    )
    block = read_log_frames(
        wavelet_set,
        participants,
        channels,
        freq_indices,
        factor,
        n_samples=None if n_times is None else n_times * factor,
        n_jobs=n_jobs,
    )
    block, freq_names = pool_frequencies(
        block, wavelet_set.freqs[freq_indices], config.freq_range, config.freq_pooling
    )
    n_subj, n_chan, n_freq, n_frames = block.shape
    features = block.reshape(n_subj, n_chan * n_freq, n_frames)
    if config.normalize == "subject":
        features = zscore_over_time(features)

    _logger.info(
        f"[{label}] {n_subj} subject(s), {n_chan} channel(s) x {n_freq} "
        f"{'band' if config.freq_pooling == 'bands' else 'bin'}(s), {n_frames} frames "
        f"({n_frames / config.frame_rate:.0f} s at {config.frame_rate:g} Hz)"
    )
    return SynchronyDataset(
        label=label,
        condition=condition,
        music_type=music_type,
        data=features.astype(np.float32, copy=False),
        participants=participants,
        feature_names=[f"{names[c]}@{f}" for c in channels for f in freq_names],
        frame_rate=config.frame_rate,
    )


def crop_frames(dataset: SynchronyDataset, start: int, stop: int) -> SynchronyDataset:
    """Frames ``[start, stop)`` of a track, remembering where they sit in it."""
    return replace(
        dataset,
        data=dataset.data[..., start:stop],
        frame_offset=dataset.frame_offset + start,
    )


def cross_fit_split(
    dataset: SynchronyDataset, fold: int, n_folds: int, min_frames: int
) -> tuple[list[SynchronyDataset], SynchronyDataset]:
    """The training segments and the held-out block of one cross-fitting fold.

    The track is cut into *n_folds* contiguous blocks; block *fold* is held out and the
    rest is training data. A contrastive model makes the moments it trained on
    synchronous by construction, so synchrony is only meaningful where it is scored
    by a model that never saw the moment — which cross-fitting gives every window.

    The training part is returned as separate segments rather than glued together, so
    no window spans the gap the held-out block leaves.

    :param min_frames: Drop training segments shorter than this (one window).
    :raises ValueError: If the held-out block is shorter than *min_frames*.
    """
    edges = np.linspace(0, dataset.n_frames, n_folds + 1).round().astype(int)
    held_out = crop_frames(dataset, edges[fold], edges[fold + 1])
    if held_out.n_frames < min_frames:
        raise ValueError(
            f"[{dataset.label}] a {n_folds}-fold block is {held_out.n_frames} frames, "
            f"shorter than one {min_frames}-frame window; use fewer folds."
        )
    segments = [
        crop_frames(dataset, start, stop)
        for start, stop in ((0, edges[fold]), (edges[fold + 1], dataset.n_frames))
        if stop - start >= min_frames
    ]
    return segments, held_out


def window_frames(window_s: float, frame_rate: float) -> int:
    return max(1, int(round(window_s * frame_rate)))


def window_starts(n_frames: int, window: int, stride: int) -> np.ndarray:
    """Start frame of every whole window of length *window*, *stride* apart."""
    return np.arange(0, n_frames - window + 1, max(1, stride))
