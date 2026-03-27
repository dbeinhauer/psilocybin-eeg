"""
Pure ISC (Inter-Subject Correlation) computation functions.

All functions operate on 3D numpy arrays of shape
``(n_items, n_features, n_samples)`` and are **agnostic** to the underlying
data representation — they work identically on raw EEG channels, ICA
component activations, wavelet amplitudes, mean responses, etc.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.definitions.fields import FrequencyBandNames

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Standard EEG frequency bands used for band-specific ISC analysis.
#: Each entry maps a :class:`FrequencyBandNames` value to ``(l_freq, h_freq)`` in Hz.
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    FrequencyBandNames.DELTA.value: (1.0, 4.0),
    FrequencyBandNames.THETA.value: (4.0, 8.0),
    FrequencyBandNames.ALPHA.value: (8.0, 13.0),
    FrequencyBandNames.BETA.value: (13.0, 30.0),
    FrequencyBandNames.GAMMA.value: (30.0, 70.0),
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
# Spearman variants
# ---------------------------------------------------------------------------


def compute_loo_isc_spearman(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out ISC using **Spearman** rank correlation.

    Spearman correlation is robust to the amplitude outliers present in raw EEG
    data and provides a useful comparison against the Pearson estimate.

    :param data: ``(n_items, n_features, n_samples)``
    :return: ``(loo_isc, mean_loo_isc)`` with shapes
        ``(n_items, n_features)`` and ``(n_features,)``.
    """
    n_items, n_features, _ = data.shape
    loo_isc = np.zeros((n_items, n_features))
    for s in range(n_items):
        others_mean = np.delete(data, s, axis=0).mean(axis=0)
        for f in range(n_features):
            r, _ = spearmanr(data[s, f], others_mean[f])
            loo_isc[s, f] = float(r)
    return loo_isc, loo_isc.mean(axis=0)


def compute_pairwise_isc_spearman(data: np.ndarray) -> np.ndarray:
    """
    Pairwise ISC averaged across all features using **Spearman** rank
    correlation.

    :param data: ``(n_items, n_features, n_samples)``
    :return: Symmetric matrix ``(n_items, n_items)`` where entry ``[i, j]``
        is the mean-across-features Spearman *rho* between items *i* and *j*.
    """
    n_items, n_features, _ = data.shape
    pairwise = np.eye(n_items)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rs = np.array(
                [spearmanr(data[i, f], data[j, f])[0] for f in range(n_features)]
            )
            pairwise[i, j] = float(np.nanmean(rs))
            pairwise[j, i] = pairwise[i, j]
    return pairwise


def compute_sliding_window_isc_spearman(
    data: np.ndarray,
    window_sec: float,
    step_sec: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-resolved LOO-ISC via a sliding window using **Spearman** rank
    correlation.

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
    isc_tc = np.zeros((len(starts), n_features))
    for w_idx, start in enumerate(starts):
        win = data[:, :, start : start + win_samples]
        win_isc = np.zeros((n_items, n_features))
        for s in range(n_items):
            others_mean = np.delete(win, s, axis=0).mean(axis=0)
            for f in range(n_features):
                r, _ = spearmanr(win[s, f], others_mean[f])
                win_isc[s, f] = float(r)
        isc_tc[w_idx] = win_isc.mean(axis=0)
    window_times = (starts + win_samples / 2) / sfreq
    return isc_tc, window_times


# ---------------------------------------------------------------------------
# Mean-field ISC (spatial average first)
# ---------------------------------------------------------------------------


def compute_mean_field_loo_isc(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-one-out ISC on the **spatial mean-field** (mean across features).

    The mean-field is the mean across features (e.g., mean across EEG channels),
    producing a single global time series per item. LOO-ISC is then computed on
    those 1-D signals using both Pearson and Spearman correlation.

    Differs from :func:`compute_loo_isc` in that it averages *signals* first and
    then correlates (``ISC(mean(channels))``), whereas :func:`compute_loo_isc`
    computes per-channel correlation and returns the mean (``mean(ISC(channel_i))``).
    A large discrepancy indicates spatially heterogeneous synchrony.

    :param data: ``(n_items, n_features, n_samples)``
    :return: ``(loo_pearson, loo_spearman)`` each of shape ``(n_items,)``.
    """
    n_items, _, _ = data.shape
    mean_field = data.mean(axis=1)  # (n_items, n_samples)
    loo_pearson = np.zeros(n_items)
    loo_spearman = np.zeros(n_items)
    for s in range(n_items):
        others = np.delete(mean_field, s, axis=0).mean(axis=0)
        loo_pearson[s] = float(pearsonr(mean_field[s], others)[0])
        loo_spearman[s] = float(spearmanr(mean_field[s], others)[0])
    return loo_pearson, loo_spearman


def compute_mean_field_pairwise_isc(data: np.ndarray) -> np.ndarray:
    """
    Pairwise ISC (Pearson) on the **spatial mean-field**.

    :param data: ``(n_items, n_features, n_samples)``
    :return: Symmetric matrix ``(n_items, n_items)`` where entry ``[i, j]``
        is the Pearson *r* between mean-field signals of items *i* and *j*.
    """
    n_items, _, _ = data.shape
    mean_field = data.mean(axis=1)  # (n_items, n_samples)
    pairwise = np.eye(n_items)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            r = float(pearsonr(mean_field[i], mean_field[j])[0])
            pairwise[i, j] = r
            pairwise[j, i] = r
    return pairwise


def compute_mean_field_sliding_window_isc(
    data: np.ndarray,
    window_sec: float,
    step_sec: float,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Time-resolved LOO-ISC on the **spatial mean-field** via a sliding window
    (Pearson correlation).

    Returns a scalar time course (one value per window), unlike
    :func:`compute_sliding_window_isc` which returns a per-feature matrix.

    :param data: ``(n_items, n_features, n_samples)``
    :param window_sec: Window length in seconds.
    :param step_sec: Step size in seconds.
    :param sfreq: Sampling frequency (Hz).
    :return: ``(isc_timecourse, window_times)`` with shapes
        ``(n_windows,)`` and ``(n_windows,)``.
    """
    n_items, _, n_samples = data.shape
    mean_field = data.mean(axis=1)  # (n_items, n_samples)
    win_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    starts = np.arange(0, n_samples - win_samples + 1, step_samples)
    isc_tc = np.zeros(len(starts))
    for w_idx, start in enumerate(starts):
        block = mean_field[:, start : start + win_samples]
        vals = []
        for s in range(n_items):
            others = np.delete(block, s, axis=0).mean(axis=0)
            vals.append(float(pearsonr(block[s], others)[0]))
        isc_tc[w_idx] = float(np.mean(vals))
    window_times = (starts + win_samples / 2) / sfreq
    return isc_tc, window_times
