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
"""

from __future__ import annotations

import dataclasses
import logging
import sys
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
    load_analyzers,
    participant_label,
    participant_labels,
    precompute_pre_alignment_wavelet_cache,
    wavelet_transform,
)
from src.analysis.data_representations import AnalysisData
from src.definitions.fields import ExperimentNames
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
    "load_analyzers",
    "participant_label",
    "participant_labels",
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
    :param n_jobs: Parallel jobs for the pre-alignment resampling.
    :returns: Wavelet datasets keyed by label, trimmed to the extent of the
        corresponding *datasets* entry for stimulus experiments.
    """
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
    return transformed
