"""
Intersubject mean-variance analysis.

Computes intersubject variance (variance *across subjects* at each time
point) as a moment-to-moment synchrony measure, windowed statistics, and
per-band equivalents.  All functions operate on 3D NumPy arrays of shape
``(n_subjects, n_channels, n_times)``.

This module implements the analysis demonstrated in
``notebooks/01-raw-mean-variance-analysis/mean_variance_broadband.ipynb`` and
``notebooks/01-raw-mean-variance-analysis/mean_variance_bands.ipynb``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.isc import FREQUENCY_BANDS

__all__ = [
    "FREQUENCY_BANDS",
    "compute_intersubject_stats",
    "compute_windowed_stats",
    "compute_band_intersubject_stats",
    "compute_pairwise_isc_matrices",
]


def compute_intersubject_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute intersubject mean and variance statistics from (z-scored) EEG data.

    For each ``(channel, time)`` cell the variance across subjects quantifies
    moment-to-moment agreement: low values indicate high synchrony.

    :param data: ``(n_subjects, n_channels, n_times)``
    :return: Dict with keys:

        - ``"inter_var"`` — ``(n_channels, n_times)`` variance across subjects
        - ``"inter_mean"`` — ``(n_channels, n_times)`` mean across subjects
        - ``"mean_t"`` — ``(n_times,)`` channel-averaged group mean
        - ``"var_t"`` — ``(n_times,)`` channel-averaged intersubject variance
        - ``"std_t"`` — ``(n_times,)`` square root of ``var_t``
        - ``"mean_over_ch"`` — ``(n_subjects, n_times)`` per-subject channel
          average

    :raises ValueError: If *data* is not 3-dimensional.
    """
    if data.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (n_subjects, n_channels, n_times), "
            f"got shape {data.shape}."
        )

    inter_var = data.var(axis=0)  # (n_channels, n_times)
    inter_mean = data.mean(axis=0)  # (n_channels, n_times)
    mean_t = inter_mean.mean(axis=0)  # (n_times,)
    var_t = inter_var.mean(axis=0)  # (n_times,)

    return {
        "inter_var": inter_var,
        "inter_mean": inter_mean,
        "mean_t": mean_t,
        "var_t": var_t,
        "std_t": np.sqrt(var_t),
        "mean_over_ch": data.mean(axis=1),  # (n_subjects, n_times)
    }


def compute_windowed_stats(
    stats: dict[str, np.ndarray],
    n_times: int,
    sfreq: float,
    window_sec: float = 2.0,
    sync_percentile: float = 10.0,
    step_sec: float | None = None,
) -> pd.DataFrame:
    """
    Compute per-window statistics and label synchrony candidates.

    The recording is divided into overlapping windows of *window_sec* seconds
    advanced by *step_sec* seconds per step.  A window is labelled a
    *synchrony candidate* when its mean intersubject variance falls below the
    ``sync_percentile``-th percentile of the full ``var_t`` distribution.

    :param stats: Output of :func:`compute_intersubject_stats`.
    :param n_times: Number of time points in the original data.
    :param sfreq: Sampling frequency in Hz.
    :param window_sec: Window length in seconds.
    :param sync_percentile: Percentile of ``var_t`` used as the synchrony
        detection threshold.
    :param step_sec: Step between successive windows in seconds.  Defaults to
        ``window_sec / 2`` (50 % overlap).  Pass ``step_sec=window_sec`` for
        non-overlapping windows.
    :return: :class:`pandas.DataFrame` with columns ``window``, ``center``,
        ``t_start``, ``t_end``, ``mean_signal``, ``var_signal``,
        ``mean_variance``, ``sync_candidate``.
    :raises ValueError: If *window_sec* or *step_sec* is non-positive, if
        either produces zero samples at *sfreq*, or if the window is longer
        than the total recording.
    """
    var_t = stats["var_t"]
    mean_over_ch = stats["mean_over_ch"]
    time = np.arange(n_times) / sfreq

    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}.")

    if step_sec is None:
        step_sec = window_sec / 2

    if step_sec <= 0:
        raise ValueError(f"step_sec must be positive, got {step_sec}.")

    win_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    if win_samples < 1:
        raise ValueError(
            f"window_sec={window_sec} at sfreq={sfreq} Hz produces "
            f"{win_samples} samples — must be at least 1."
        )
    if step_samples < 1:
        raise ValueError(
            f"step_sec={step_sec} at sfreq={sfreq} Hz produces "
            f"{step_samples} samples — must be at least 1."
        )
    starts = np.arange(0, n_times - win_samples + 1, step_samples)
    n_windows = len(starts)
    if n_windows < 1:
        total_duration = n_times / sfreq
        raise ValueError(
            f"window_sec={window_sec} is longer than the total recording "
            f"duration ({total_duration:.3f} seconds). Choose a shorter window."
        )
    sync_threshold = np.percentile(var_t, sync_percentile)

    records = []
    for w, start in enumerate(starts):
        sl = slice(start, start + win_samples)
        subj_mean = mean_over_ch[:, sl].mean(axis=1)  # (n_subjects,)
        records.append(
            {
                "window": w + 1,
                "center": time[start + win_samples // 2],
                "t_start": time[start],
                "t_end": time[min(start + win_samples - 1, n_times - 1)],
                "mean_signal": float(subj_mean.mean()),
                "var_signal": float(subj_mean.var()),
                "mean_variance": float(var_t[sl].mean()),
            }
        )

    df = pd.DataFrame(records)
    df["sync_candidate"] = df["mean_variance"] < sync_threshold
    return df


def compute_band_intersubject_stats(
    ad: "AnalysisData",  # noqa: F821
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Apply :func:`compute_intersubject_stats` to each frequency band.

    Each band is computed from a fresh bandpass-filtered copy of *ad*.

    :param ad: :class:`~src.analysis.data_representations.AnalysisData`
        wrapping z-scored time-domain EEG of shape
        ``(n_subjects, n_channels, n_times)``.
    :param bands: Band definitions ``{name: (l_freq, h_freq)}``.
        Defaults to :data:`~src.analysis.isc.FREQUENCY_BANDS`.
    :return: ``{band_name: stats_dict}`` where each ``stats_dict`` is the
        output of :func:`compute_intersubject_stats` for that band's
        filtered data.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    return {
        band: compute_intersubject_stats(ad.filter_to_band(l_freq, h_freq).data)
        for band, (l_freq, h_freq) in bands.items()
    }


def compute_pairwise_isc_matrices(
    band_data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute the mean Pearson correlation matrix between every subject pair
    for each frequency band.

    For z-scored data (mean=0, std=1) the mean product across channels and
    time equals the mean Pearson correlation, following the formula from
    ``notebooks/mean_variance_bands.ipynb``::

        ISC(i, j) = mean_ch( mean_t( x_i * x_j ) )

    :param band_data: ``{band_name: data}`` where *data* has shape
        ``(n_subjects, n_channels, n_times)``.
    :return: ``{band_name: matrix}`` where each matrix has shape
        ``(n_subjects, n_subjects)``.
    """
    results: dict[str, np.ndarray] = {}
    for band, data in band_data.items():
        n_subjects = data.shape[0]
        mat = np.zeros((n_subjects, n_subjects))
        for i in range(n_subjects):
            for j in range(i, n_subjects):
                val = float(np.mean((data[i] * data[j]).mean(axis=1)))
                mat[i, j] = val
                mat[j, i] = val
        results[band] = mat
    return results
