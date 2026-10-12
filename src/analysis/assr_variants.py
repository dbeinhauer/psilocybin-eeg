"""
Per-condition projection of stored wavelets into every ASSR signal variant.

The stage-06 analysis compares one fixed electrode reference against several learned
spatial filters. Building those comparison inputs means reading each participant's
wavelets once and applying every operator to the same block, which is what
:func:`project_condition` does.

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
    are what each variant swaps out, and ``roi_raw`` skips the normalisation the others
    get. Every array is ``(participants, rows, kept freqs, times)``.
    """

    #: ``components + 2`` rows on the RAW cache: each recording's own IC filters, then
    #: the full and PCA masks. The reference rows every variant borrows come from here.
    prestim: np.ndarray
    #: The two masks on the Z-SCORED cache.
    zmask: np.ndarray
    #: The mask's electrodes kept UNCOMBINED, so
    #: :func:`~src.analysis.assr_trials.roi_channelwise_snr` can normalise each one
    #: before they are averaged.
    roi_raw: np.ndarray
    #: ``components`` rows: the cohort-mean filter on the RAW cache — "prestim" with a
    #: single thing changed, the filter.
    mean_raw: np.ndarray
    #: The cohort-mean filter on the IVA's own input (z-scored, then this recording's
    #: PCA subspace), rebuilt here because no stored array is ``U_mean`` on that signal.
    mean_stored_z: np.ndarray
    #: Each recording's OWN filter, restricted to the anchor electrodes, on the z-scored
    #: cache.
    masked_z: np.ndarray
    #: The cohort-mean filter under the same restriction.
    mean_masked_z: np.ndarray


def project_condition(
    wavelet_set: WaveletSet,
    condition: str,
    participants: list[str],
    resolve,
    filters: np.ndarray,
    binary: np.ndarray,
    pca_rows: np.ndarray,
    freq_indices: np.ndarray,
    store_channels: list[str],
    n_times: int | None,
    roi_index: np.ndarray,
    mean_filter: np.ndarray,
    masked_filters: np.ndarray,
    mean_masked_filter: np.ndarray,
    projectors: np.ndarray,
) -> ProjectedVariants:
    """Read one condition once; return every per-condition array the grid needs.

    Everything is read off the **same** per-participant block, so the four operators
    the ``mean_*`` / ``masked_*`` variants add cost one more tensordot each rather than
    another read. The z-scored copy of a block is built once and shared by every
    operator that needs it, for the same reason.

    The z-scored IC rows of ``stored_prestim`` are read straight from the store, so they
    are not produced here.

    :param mean_filter: ``(components, channels)`` cohort-mean filter, shared by every
        recording and both conditions.
    :param masked_filters: ``(rows, components, channels)`` per-recording filters with
        every weight outside the anchor electrodes zeroed.
    :param mean_masked_filter: ``(components, channels)`` the same restriction on the
        cohort-mean filter.
    :param projectors: ``(rows, channels, channels)`` PCA reconstruction projectors, so
        the IVA's own input can be rebuilt as ``P^T P (z)`` in channel space.
    :return: A :class:`ProjectedVariants`.
    """
    if wavelet_set.channel_names != store_channels:
        raise ValueError(
            f"{wavelet_set.directory}: channel axis does not match the stored run's; "
            "the filters are indexed by the stored channels."
        )
    available = wavelet_set.open(participants[0]).shape[-1]
    keep_times = available if n_times is None else min(n_times, available)
    n_comp = filters.shape[1]

    def _empty(rows: int) -> np.ndarray:
        return np.empty(
            (len(participants), rows, freq_indices.size, keep_times), np.float32
        )

    prestim = _empty(n_comp + 2)
    zmask = _empty(2)
    roi_raw = _empty(roi_index.size)
    mean_raw = _empty(n_comp)
    mean_stored_z = _empty(n_comp)
    masked_z = _empty(n_comp)
    mean_masked_z = _empty(n_comp)

    _logger.info(
        f"[{condition}] reading {len(participants)} participant(s) from "
        f"{wavelet_set.directory} ({freq_indices.size}/{wavelet_set.n_freqs} bins kept)"
    )
    started = time.time()
    subjects = iter_wavelet_subjects(
        wavelet_set, participants, freq_indices=freq_indices, n_times=keep_times
    )
    for row, (name, block) in enumerate(subjects):
        store = resolve(name, condition)
        # per-recording spatial operators for this (participant, condition)
        full_row = binary[None, :]
        pca_row = pca_rows[store][None, :]
        stacked = np.concatenate(
            [filters[store], full_row, pca_row], axis=0
        )  # (K+2, C)
        prestim[row] = at.project_channels(stacked, block)
        # The cohort-mean filter on the RAW cache: "prestim" with the filter swapped and
        # nothing else, so the difference between the two rows IS the topography.
        mean_raw[row] = at.project_channels(mean_filter, block)
        roi_raw[row] = block[roi_index]

        # The z-scored copy is the expensive intermediate; build it once and hand it to
        # every operator that reads the standardisation the decomposition was fitted on.
        block_z = zscore_time(block)
        zmask[row] = at.project_channels(
            np.concatenate([full_row, pca_row], axis=0), block_z
        )
        # The IVA's own input, rebuilt: z-scored, then this recording's PCA subspace.
        # Composing the two channel operators first keeps the (C, F, T) intermediate
        # from being materialised twice.
        mean_stored_z[row] = at.project_channels(
            mean_filter @ projectors[store], block_z
        )
        # The restricted operators read the z-scored wavelet and NOT raw power: the
        # weights presume that scaling, and a weighted average of raw power is
        # additionally weighted by each electrode's own power level — the very artefact
        # roi_channelwise_snr exists to remove from the binary reference. And no PCA:
        # P^T P mixes every channel back in, which would defeat "read only these
        # electrodes" entirely.
        masked_z[row] = at.project_channels(masked_filters[store], block_z)
        mean_masked_z[row] = at.project_channels(mean_masked_filter, block_z)
        _logger.info(
            f"[{condition}]   {name} ({row + 1}/{len(participants)}, "
            f"{time.time() - started:.0f}s)"
        )
    _logger.info(f"[{condition}] projected in {time.time() - started:.0f}s")
    return ProjectedVariants(
        prestim=prestim,
        zmask=zmask,
        roi_raw=roi_raw,
        mean_raw=mean_raw,
        mean_stored_z=mean_stored_z,
        masked_z=masked_z,
        mean_masked_z=mean_masked_z,
    )