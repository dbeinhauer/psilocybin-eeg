"""
Pure ISC (Inter-Subject Correlation) computation functions.

All functions operate on 3D numpy arrays of shape
``(n_items, n_features, n_samples)`` and are **agnostic** to the underlying
data representation — they work identically on raw EEG channels, ICA
component activations, wavelet amplitudes, mean responses, etc.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Standard EEG frequency bands used for band-specific ISC analysis.
#: Each entry maps a band name to ``(l_freq, h_freq)`` in Hz.
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 70.0),
}


# ---------------------------------------------------------------------------
# Core ISC functions
# ---------------------------------------------------------------------------


def compute_loo_isc(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out Inter-Subject Correlation.

    For each item *s*, the Pearson *r* between item *s* and the mean of all
    other items is computed independently for every feature.

    :param data: ``(n_items, n_features, n_samples)``
    :return: ``(loo_isc, mean_loo_isc)`` with shapes
        ``(n_items, n_features)`` and ``(n_features,)``.
    """
    n_items, n_features, _ = data.shape
    loo_isc = np.zeros((n_items, n_features))
    for s in range(n_items):
        others_mean = np.delete(data, s, axis=0).mean(axis=0)
        for f in range(n_features):
            r = float(pearsonr(data[s, f], others_mean[f])[0])
            loo_isc[s, f] = r
    return loo_isc, loo_isc.mean(axis=0)


def compute_pairwise_isc(data: np.ndarray) -> np.ndarray:
    """
    Pairwise ISC averaged across all features.

    :param data: ``(n_items, n_features, n_samples)``
    :return: Symmetric matrix ``(n_items, n_items)`` where entry ``[i, j]``
        is the mean-across-features Pearson *r* between items *i* and *j*.
    """
    n_items, n_features, _ = data.shape
    pairwise = np.eye(n_items)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rs = np.array(
                [pearsonr(data[i, f], data[j, f])[0] for f in range(n_features)]
            )
            mean_r = np.nanmean(rs)
            pairwise[i, j] = mean_r
            pairwise[j, i] = mean_r
    return pairwise


def compute_pairwise_isc_per_feature(data: np.ndarray) -> np.ndarray:
    """
    Pairwise ISC for every feature separately.

    :param data: ``(n_items, n_features, n_samples)``
    :return: ``(n_features, n_items, n_items)``
    """
    n_items, n_features, _ = data.shape
    pairwise = np.zeros((n_features, n_items, n_items))
    for f in range(n_features):
        for i in range(n_items):
            pairwise[f, i, i] = 1.0
            for j in range(i + 1, n_items):
                r = float(pearsonr(data[i, f], data[j, f])[0])
                pairwise[f, i, j] = r
                pairwise[f, j, i] = r
    return pairwise


def compute_sliding_window_isc(
    data: np.ndarray,
    window_sec: float,
    step_sec: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-resolved LOO-ISC via a sliding window.

    :param data: ``(n_items, n_features, n_samples)``
    :param window_sec: Window length in seconds.
    :param step_sec: Step size in seconds.
    :param sfreq: Sampling frequency (Hz).
    :return: ``(isc_timecourse, window_times)`` with shapes
        ``(n_windows, n_features)`` and ``(n_windows,)``.
    """
    n_items, n_features, n_samples = data.shape
    win_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    n_windows = len(starts)

    isc_timecourse = np.zeros((n_windows, n_features))
    for w_idx, start in enumerate(starts):
        end = start + win_samples
        window_data = data[:, :, start:end]
        window_isc = np.zeros((n_items, n_features))
        for s in range(n_items):
            others_mean = np.delete(window_data, s, axis=0).mean(axis=0)
            for f in range(n_features):
                r = float(pearsonr(window_data[s, f], others_mean[f])[0])
                window_isc[s, f] = r
        isc_timecourse[w_idx] = window_isc.mean(axis=0)

    window_times = (starts + win_samples / 2) / sfreq
    return isc_timecourse, window_times


# ---------------------------------------------------------------------------
# Mean & Variance across items (subjects)
# ---------------------------------------------------------------------------


def compute_mean_variance(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the global mean and variance of the signal across items (subjects).

    For each feature the signal is first averaged across all items at every
    time point, then the temporal mean and variance of that average signal
    are returned.

    :param data: ``(n_items, n_features, n_samples)``
    :return: ``(mean_per_feature, var_per_feature)`` each of shape
        ``(n_features,)``.
    """
    # mean across items -> (n_features, n_samples)
    mean_signal = data.mean(axis=0)
    # temporal mean and variance per feature
    mean_per_feature = mean_signal.mean(axis=1)
    var_per_feature = mean_signal.var(axis=1)
    return mean_per_feature, var_per_feature


def compute_sliding_window_mean_variance(
    data: np.ndarray,
    window_sec: float,
    step_sec: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-resolved mean and variance of the signal via a sliding window.

    Inside each window the signal is averaged across items (subjects) to
    obtain a single time-course per feature, then the temporal mean and
    variance within the window are computed.

    :param data: ``(n_items, n_features, n_samples)``
    :param window_sec: Window length in seconds.
    :param step_sec: Step size in seconds.
    :param sfreq: Sampling frequency (Hz).
    :return: ``(mean_timecourse, var_timecourse, window_times)`` with shapes
        ``(n_windows, n_features)``, ``(n_windows, n_features)`` and
        ``(n_windows,)``.
    """
    n_items, n_features, n_samples = data.shape
    win_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    n_windows = len(starts)

    mean_timecourse = np.zeros((n_windows, n_features))
    var_timecourse = np.zeros((n_windows, n_features))
    for w_idx, start in enumerate(starts):
        end = start + win_samples
        window_data = data[:, :, start:end]
        # average across items -> (n_features, win_samples)
        mean_signal = window_data.mean(axis=0)
        mean_timecourse[w_idx] = mean_signal.mean(axis=1)
        var_timecourse[w_idx] = mean_signal.var(axis=1)

    window_times = (starts + win_samples / 2) / sfreq
    return mean_timecourse, var_timecourse, window_times
