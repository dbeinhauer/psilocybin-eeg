"""
Per-condition projection of stored wavelets into the ASSR signal variants.

The stage-06/06b SNR grids compare fixed electrode references against the learned
spatial filters, read two ways: ``full`` (the decomposition's own z-scored sources,
read from the store) and ``roi_masked`` (each filter with every weight outside the ASSR
electrodes zeroed, on the z-scored cache). Stage 06b adds ``non_roi_masked``, the
complement: the ASSR weights zeroed instead. Building those inputs means reading each
participant's wavelets once and applying every operator to the same block, which is
what :func:`project_condition` does.

Extracted from ``scripts/run_assr_snr_grid.py`` so the notebook and the CLI build their
variants with the same code rather than two implementations of it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from src.analysis import assr_trials as at
from src.definitions.fields import IvaVariants
from src.io.wavelet_store import WaveletSet, iter_wavelet_subjects

_logger = logging.getLogger(__name__)


def zscore_time(block: np.ndarray) -> np.ndarray:
    """Z-score a ``(..., time)`` block along its last axis (per channel/frequency)."""
    mean = block.mean(axis=-1, keepdims=True)
    std = block.std(axis=-1, keepdims=True)
    return (block - mean) / np.where(std == 0, 1.0, std)


def row_resolver(results, variant: IvaVariants):
    """A ``(participant, condition) -> store row`` function for either variant.

    The subject-axis join (``channel_joined``) has one row per recording, so a filter is
    condition-specific and ``results.row(p, c)`` picks it. The time-axis join
    (``channel_joined_tracks``) has one row per participant, shared by both conditions,
    so the condition is ignored.
    """
    per_recording = variant == IvaVariants.CHANNEL_JOINED_SEPARATE

    def resolve(participant: str, condition: str) -> int:
        if per_recording:
            return results.row(participant, condition)
        return results.row(participant)

    return resolve, per_recording


@dataclass(frozen=True)
class ProjectedVariants:
    """One condition's projections, one array per group of source rows.

    Kept as named arrays rather than one wide stack because the groups are read on
    different cadences: the reference rows are shared by every variant, the learned rows
    are what a variant swaps in, and ``roi_raw`` skips the normalisation the others get.
    Every array is ``(participants, rows, kept freqs, times)``.
    """

    #: The full and PCA masks on the RAW cache — the reference rows every variant
    #: carries verbatim.
    reference_raw: np.ndarray
    #: The PCA mask on the Z-SCORED cache: the reduction's view of the mask on the
    #: signal the decomposition was handed.
    pca_z: np.ndarray
    #: The mask's electrodes kept UNCOMBINED, so
    #: :func:`~src.analysis.assr_trials.roi_channelwise_snr` can normalise each one
    #: before they are averaged.
    roi_raw: np.ndarray
    #: Each recording's OWN filter, restricted to the anchor electrodes, on the z-scored
    #: cache — the learned rows of the ``roi_masked`` variant.
    masked_z: np.ndarray
    #: The same filter with the anchor electrodes zeroed instead — the learned rows of
    #: the ``non_roi_masked`` variant; ``None`` unless asked for.
    outside_z: np.ndarray | None = None
    #: ``masked_z`` / ``outside_z`` at the ``tf_freq_indices`` bins, for full TF maps;
    #: ``None`` unless asked for.
    masked_tf_z: np.ndarray | None = None
    outside_tf_z: np.ndarray | None = None


def store_channel_index(
    wavelet_set: WaveletSet, store_channels: list[str]
) -> np.ndarray | None:
    """Cache channel positions of the store's channels, or ``None`` if they coincide.

    A store fitted on a channel subset (the ASSR ROI) indexes its filters by its own
    channels, so the cache is read on those channels, in that order.

    :raises ValueError: If a store channel is absent from the cache.
    """
    if wavelet_set.channel_names == store_channels:
        return None
    position = {name: i for i, name in enumerate(wavelet_set.channel_names)}
    missing = [name for name in store_channels if name not in position]
    if missing:
        raise ValueError(
            f"{wavelet_set.directory}: the stored run's channels {missing} are not in "
            "the cache; the filters are indexed by the stored channels."
        )
    return np.array([position[name] for name in store_channels])


def project_condition(
    wavelet_set: WaveletSet,
    condition: str,
    participants: list[str],
    resolve,
    binary: np.ndarray,
    pca_rows: np.ndarray,
    freq_indices: np.ndarray,
    store_channels: list[str],
    n_times: int | None,
    roi_index: np.ndarray,
    masked_filters: np.ndarray,
    *,
    outside_filters: np.ndarray | None = None,
    tf_freq_indices: np.ndarray | None = None,
) -> ProjectedVariants:
    """Read one condition once; return every per-condition array the grid needs.

    Everything is read off the **same** per-participant block, and the z-scored copy of
    it is built once and shared by every operator that reads it.

    The z-scored learned rows of the ``full`` variant are read straight from the store,
    so they are not produced here.

    With *tf_freq_indices*, the block read is the union of both bin sets, so the masked
    filters are also projected at every bin a full TF map needs. A z-score runs along
    time per (channel, bin), so the wider read leaves the *freq_indices* rows unchanged.

    :param binary: ``(channels,)`` electrode-average weights.
    :param pca_rows: ``(rows, channels)`` that average read through each recording's
        decomposition subspace.
    :param roi_index: Store-channel positions of the anchor electrodes.
    :param masked_filters: ``(rows, components, channels)`` per-recording filters with
        every weight outside the anchor electrodes zeroed.
    :param outside_filters: ``(rows, components, channels)`` the same filters with the
        anchor electrodes zeroed instead, for :attr:`ProjectedVariants.outside_z`.
    :param tf_freq_indices: Cache bins at which to also project the masked filters
        (:attr:`ProjectedVariants.masked_tf_z`, ``outside_tf_z``).
    :return: A :class:`ProjectedVariants`.
    :raises ValueError: If a store channel is absent from the cache.
    """
    channel_index = store_channel_index(wavelet_set, store_channels)
    available = wavelet_set.open(participants[0]).shape[-1]
    keep_times = available if n_times is None else min(n_times, available)
    n_comp = masked_filters.shape[1]
    freq_indices = np.asarray(freq_indices, dtype=int)
    read, grid_at, tf_at = freq_indices, None, None
    if tf_freq_indices is not None:
        tf_freq_indices = np.asarray(tf_freq_indices, dtype=int)
        read = np.union1d(freq_indices, tf_freq_indices)
        # Positions of each bin set inside the (sorted) block read.
        grid_at = np.searchsorted(read, freq_indices)
        tf_at = np.searchsorted(read, tf_freq_indices)

    def _empty(rows: int, n_freqs: int = freq_indices.size) -> np.ndarray:
        return np.empty((len(participants), rows, n_freqs, keep_times), np.float32)

    reference_raw = _empty(2)
    pca_z = _empty(1)
    roi_raw = _empty(roi_index.size)
    masked_z = _empty(n_comp)
    outside_z = None if outside_filters is None else _empty(n_comp)
    masked_tf_z = outside_tf_z = None
    if tf_at is not None:
        masked_tf_z = _empty(n_comp, tf_at.size)
        if outside_filters is not None:
            outside_tf_z = _empty(n_comp, tf_at.size)

    _logger.info(
        f"[{condition}] reading {len(participants)} participant(s) from "
        f"{wavelet_set.directory} ({read.size}/{wavelet_set.n_freqs} bins kept"
        + ("" if channel_index is None else f", {len(store_channels)} store channels")
        + ")"
    )
    started = time.time()
    subjects = iter_wavelet_subjects(
        wavelet_set, participants, freq_indices=read, n_times=keep_times
    )
    for row, (name, wide) in enumerate(subjects):
        if channel_index is not None:
            wide = wide[channel_index]
        wide_z = zscore_time(wide)
        block, block_z = wide, wide_z
        if tf_at is not None:
            block, block_z = wide[:, grid_at], wide_z[:, grid_at]
        store = resolve(name, condition)
        pca_row = pca_rows[store][None, :]
        reference_raw[row] = at.project_channels(
            np.concatenate([binary[None, :], pca_row], axis=0), block
        )
        roi_raw[row] = block[roi_index]

        pca_z[row] = at.project_channels(pca_row, block_z)
        # The restricted operators read the z-scored wavelet and NOT raw power: the
        # weights presume that scaling, and a weighted average of raw power is
        # additionally weighted by each electrode's own power level — the very artefact
        # roi_channelwise_snr exists to remove from the binary reference. And no PCA:
        # P^T P mixes every channel back in, which would defeat "read only these
        # electrodes" entirely.
        masked_z[row] = at.project_channels(masked_filters[store], block_z)
        if outside_z is not None:
            outside_z[row] = at.project_channels(outside_filters[store], block_z)
        if tf_at is not None:
            tf_z = wide_z[:, tf_at]
            masked_tf_z[row] = at.project_channels(masked_filters[store], tf_z)
            if outside_tf_z is not None:
                outside_tf_z[row] = at.project_channels(outside_filters[store], tf_z)
        _logger.info(
            f"[{condition}]   {name} ({row + 1}/{len(participants)}, "
            f"{time.time() - started:.0f}s)"
        )
    _logger.info(f"[{condition}] projected in {time.time() - started:.0f}s")
    return ProjectedVariants(
        reference_raw=reference_raw,
        pca_z=pca_z,
        roi_raw=roi_raw,
        masked_z=masked_z,
        outside_z=outside_z,
        masked_tf_z=masked_tf_z,
        outside_tf_z=outside_tf_z,
    )
