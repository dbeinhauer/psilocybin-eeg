"""
Unified data containers and adapter functions for versatile EEG analysis.

The :class:`AnalysisData` dataclass wraps a 3D numpy array with metadata so
that all downstream analysis and visualisation functions can operate on *any*
data representation (raw time-domain EEG, ICA activations, wavelet amplitudes,
mean responses, etc.) through the same interface.

Adapter functions (``to_*``) convert between representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import mne
from mne.time_frequency import tfr_array_morlet
from scipy.signal import hilbert
from scipy.stats import zscore, circmean

# ---------------------------------------------------------------------------
# Enum for representation types
# ---------------------------------------------------------------------------


class DataRepresentation(Enum):
    """Supported data representation types."""

    TIME_DOMAIN = "time_domain"
    ICA_ACTIVATIONS = "ica_activations"
    WAVELET_AMPLITUDE = "wavelet_amplitude"
    WAVELET_POWER = "wavelet_power"
    WAVELET_PHASE = "wavelet_phase"
    MEAN_RESPONSE = "mean_response"


# ---------------------------------------------------------------------------
# Core data container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisData:
    """
    Unified container for analysis-ready data regardless of representation.

    The :attr:`data` array always has shape ``(n_items, n_features, n_samples)``
    where the semantic meaning of each axis depends on :attr:`representation`:

    =====================  ==========  ================  ===========
    Representation         n_items     n_features        n_samples
    =====================  ==========  ================  ===========
    TIME_DOMAIN            subjects    channels          time points
    ICA_ACTIVATIONS        subjects    IC components     time points
    WAVELET_AMPLITUDE      subjects    channels          time points
    WAVELET_POWER          subjects    channels          time points
    MEAN_RESPONSE          groups      channels          time points
    =====================  ==========  ================  ===========
    """

    data: np.ndarray
    sfreq: float
    representation: DataRepresentation
    label: str
    feature_names: Optional[List[str]] = None
    info: Optional[mne.Info] = None
    metadata: dict = field(default_factory=dict)

    # -- shape shortcuts ---------------------------------------------------

    @property
    def n_items(self) -> int:
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[2]

    # -- axis labels -------------------------------------------------------

    @property
    def feature_axis_label(self) -> str:
        """Human-readable label for the feature axis (dim 1)."""
        _map = {
            DataRepresentation.TIME_DOMAIN: "Channel",
            DataRepresentation.ICA_ACTIVATIONS: "IC component",
            DataRepresentation.WAVELET_AMPLITUDE: "Channel",
            DataRepresentation.WAVELET_POWER: "Channel",
            DataRepresentation.WAVELET_PHASE: "Channel",
            DataRepresentation.MEAN_RESPONSE: "Channel",
        }
        return _map.get(self.representation, "Feature")

    @property
    def item_axis_label(self) -> str:
        """Human-readable label for the item axis (dim 0)."""
        _map = {
            DataRepresentation.TIME_DOMAIN: "Subject",
            DataRepresentation.ICA_ACTIVATIONS: "Subject",
            DataRepresentation.WAVELET_AMPLITUDE: "Subject",
            DataRepresentation.WAVELET_POWER: "Subject",
            DataRepresentation.WAVELET_PHASE: "Subject",
            DataRepresentation.MEAN_RESPONSE: "Group",
        }
        return _map.get(self.representation, "Item")

    @property
    def value_label(self) -> str:
        """Human-readable label for what the values represent."""
        _map = {
            DataRepresentation.TIME_DOMAIN: "Amplitude (µV)",
            DataRepresentation.ICA_ACTIVATIONS: "Activation (a.u.)",
            DataRepresentation.WAVELET_AMPLITUDE: "Amplitude envelope",
            DataRepresentation.WAVELET_POWER: "Power",
            DataRepresentation.WAVELET_PHASE: "Phase (rad)",
            DataRepresentation.MEAN_RESPONSE: "Amplitude (µV)",
        }
        return _map.get(self.representation, "Value")

    # -- in-place & copy transforms ----------------------------------------

    def normalize(self, axis: int = 2) -> "AnalysisData":
        """Return a **new** ``AnalysisData`` with z-scored data."""
        return AnalysisData(
            data=zscore(self.data, axis=axis),
            sfreq=self.sfreq,
            representation=self.representation,
            label=self.label,
            feature_names=self.feature_names,
            info=self.info,
            metadata=self.metadata,
        )

    def normalize_inplace(self, axis: int = 2) -> None:
        """Z-score :attr:`data` in-place."""
        self.data = zscore(self.data, axis=axis)

    def filter_to_band(self, l_freq: float, h_freq: float) -> "AnalysisData":
        """
        Return a **new** ``AnalysisData`` with band-pass filtered data.

        Uses MNE's FIR filter (Hamming window) applied independently to
        each item's data matrix.
        """
        filtered = np.empty_like(self.data, dtype=float)
        for s in range(self.n_items):
            filtered[s] = mne.filter.filter_data(
                self.data[s].astype(float),
                sfreq=self.sfreq,
                l_freq=l_freq,
                h_freq=h_freq,
                method="fir",
                fir_window="hamming",
                verbose=False,
            )
        return AnalysisData(
            data=filtered,
            sfreq=self.sfreq,
            representation=self.representation,
            label=f"{self.label} [{l_freq}-{h_freq} Hz]",
            feature_names=self.feature_names,
            info=self.info,
            metadata={**self.metadata, "l_freq": l_freq, "h_freq": h_freq},
        )

    def copy(self) -> "AnalysisData":
        """Return a deep copy."""
        return AnalysisData(
            data=self.data.copy(),
            sfreq=self.sfreq,
            representation=self.representation,
            label=self.label,
            feature_names=list(self.feature_names) if self.feature_names else None,
            info=self.info,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        shape_str = f"{self.n_items} items × {self.n_features} features × {self.n_samples} samples"
        return (
            f"AnalysisData('{self.label}', {self.representation.value}, "
            f"{shape_str}, sfreq={self.sfreq})"
        )


# ---------------------------------------------------------------------------
# Adapter / transformation functions
# ---------------------------------------------------------------------------


def from_array(
    data: np.ndarray,
    sfreq: float,
    representation: DataRepresentation = DataRepresentation.TIME_DOMAIN,
    label: str = "Custom data",
    feature_names: Optional[List[str]] = None,
    info: Optional[mne.Info] = None,
    metadata: Optional[dict] = None,
) -> AnalysisData:
    """
    Wrap an arbitrary 3D array into an :class:`AnalysisData` container.

    :param data: ``(n_items, n_features, n_samples)``
    :param sfreq: Sampling frequency (Hz).
    :param representation: What the data represents.
    :param label: Human-readable label for plots.
    :param feature_names: Names for each feature (channel name, IC name, …).
    :param info: Optional MNE Info object.
    :param metadata: Arbitrary extra metadata dict.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data.shape}")
    return AnalysisData(
        data=data,
        sfreq=sfreq,
        representation=representation,
        label=label,
        feature_names=feature_names,
        info=info,
        metadata=metadata or {},
    )


def to_analytic_amplitude(ad: AnalysisData) -> AnalysisData:
    """
    Compute the Hilbert analytic-signal amplitude envelope (broadband).

    Returns a new :class:`AnalysisData` with
    :attr:`~DataRepresentation.WAVELET_AMPLITUDE` representation,
    same shape as the input.
    """
    analytic = hilbert(ad.data, axis=2)
    amplitude = np.abs(analytic)
    return AnalysisData(
        data=amplitude,
        sfreq=ad.sfreq,
        representation=DataRepresentation.WAVELET_AMPLITUDE,
        label=f"{ad.label} (Hilbert amplitude)",
        feature_names=ad.feature_names,
        info=ad.info,
        metadata={**ad.metadata, "transform": "hilbert_amplitude"},
    )


def to_wavelet_power(
    ad: AnalysisData,
    freqs: np.ndarray,
    n_cycles: Optional[Union[np.ndarray, float]] = None,
    *,
    keep_frequency_dim: bool = False,
) -> AnalysisData:
    """
    Compute Morlet-wavelet power, averaged over the frequency axis.

    :param ad: Input data (should be time-domain or ICA activations).
    :param freqs: Frequencies of interest (Hz).
    :param n_cycles: Number of wavelet cycles per frequency.
        Defaults to ``freqs / 2``.
    :param keep_frequency_dim: When ``True``, preserves the frequency
        dimension by flattening ``(feature, frequency)`` into the
        feature axis. Output shape becomes
        ``(n_items, n_features * n_freqs, n_samples)``.
    :return: ``AnalysisData`` with shape ``(n_items, n_features, n_samples)``
        containing the mean wavelet power across *freqs* (default), or
        frequency-resolved flattened output when *keep_frequency_dim* is true.
    """
    if n_cycles is None:
        n_cycles = freqs / 2.0
    tfr = tfr_array_morlet(
        ad.data.astype(float),
        sfreq=ad.sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        verbose=False,
    )
    # tfr: (n_items, n_features, n_freqs, n_samples)
    if keep_frequency_dim:
        power = tfr.reshape(tfr.shape[0], tfr.shape[1] * tfr.shape[2], tfr.shape[3])
        if ad.feature_names is not None:
            feature_names = [
                f"{name}@{freq:.1f}Hz" for name in ad.feature_names for freq in freqs
            ]
        else:
            feature_names = None
    else:
        power = tfr.mean(axis=2)
        feature_names = ad.feature_names
    return AnalysisData(
        data=power,
        sfreq=ad.sfreq,
        representation=DataRepresentation.WAVELET_POWER,
        label=f"{ad.label} (wavelet power {freqs[0]:.0f}-{freqs[-1]:.0f} Hz)",
        feature_names=feature_names,
        info=ad.info,
        metadata={
            **ad.metadata,
            "freqs": freqs,
            "n_cycles": n_cycles,
            "keep_frequency_dim": keep_frequency_dim,
        },
    )


def to_wavelet_phase(
    ad: AnalysisData,
    freqs: np.ndarray,
    n_cycles: Optional[Union[np.ndarray, float]] = None,
    *,
    keep_frequency_dim: bool = False,
) -> AnalysisData:
    """
    Compute Morlet-wavelet phase.

    :param ad: Input data (should be time-domain or ICA activations).
    :param freqs: Frequencies of interest (Hz).
    :param n_cycles: Number of wavelet cycles per frequency.
        Defaults to ``freqs / 2``.
    :param keep_frequency_dim: When ``True``, preserves the frequency
        dimension by flattening ``(feature, frequency)`` into the
        feature axis. Output shape becomes
        ``(n_items, n_features * n_freqs, n_samples)``.
        By default (``False``), phase is averaged across frequencies
        using circular mean.
    :return: ``AnalysisData`` with wavelet phase values.
    """
    if n_cycles is None:
        n_cycles = freqs / 2.0
    tfr = tfr_array_morlet(
        ad.data.astype(float),
        sfreq=ad.sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output="phase",
        verbose=False,
    )
    if keep_frequency_dim:
        phase = tfr.reshape(tfr.shape[0], tfr.shape[1] * tfr.shape[2], tfr.shape[3])
        if ad.feature_names is not None:
            feature_names = [
                f"{name}@{freq:.1f}Hz" for name in ad.feature_names for freq in freqs
            ]
        else:
            feature_names = None
    else:
        phase = circmean(tfr, high=np.pi, low=-np.pi, axis=2)
        feature_names = ad.feature_names
    return AnalysisData(
        data=phase,
        sfreq=ad.sfreq,
        representation=DataRepresentation.WAVELET_PHASE,
        label=f"{ad.label} (wavelet phase {freqs[0]:.0f}-{freqs[-1]:.0f} Hz)",
        feature_names=feature_names,
        info=ad.info,
        metadata={
            **ad.metadata,
            "freqs": freqs,
            "n_cycles": n_cycles,
            "keep_frequency_dim": keep_frequency_dim,
        },
    )


def to_wavelet_tfr(
    ad: AnalysisData,
    freqs: np.ndarray,
    n_cycles: Optional[Union[np.ndarray, float]] = None,
    output: str = "power",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full Morlet time-frequency decomposition (no averaging).

    :param ad: Input data.
    :param freqs: Frequencies of interest (Hz).
    :param n_cycles: Number of wavelet cycles per frequency.
    :param output: ``'power'``, ``'complex'``, or ``'phase'``.
    :return: ``(tfr, freqs)`` where *tfr* has shape
        ``(n_items, n_features, n_freqs, n_samples)``.
    """
    if n_cycles is None:
        n_cycles = freqs / 2.0
    tfr = tfr_array_morlet(
        ad.data.astype(float),
        sfreq=ad.sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output=output,
        verbose=False,
    )
    return tfr, freqs


def extract_ica_activations(
    raws: list,
    icas: list,
    sfreq: Optional[float] = None,
) -> np.ndarray:
    """
    Extract ICA component activations from paired Raw + ICA objects.

    :param raws: List of ``mne.io.Raw`` objects.
    :param icas: List of ``mne.preprocessing.ICA`` objects (same length).
    :param sfreq: Target sampling frequency.  If provided, sources are
        resampled to this rate.
    :return: ``(n_subjects, n_components, n_samples)`` numpy array.
    """
    sources = []
    for raw, ica in zip(raws, icas):
        src = ica.get_sources(raw)
        if sfreq is not None and src.info["sfreq"] != sfreq:
            src = src.resample(sfreq)
        sources.append(src.get_data())

    min_samples = min(s.shape[1] for s in sources)
    return np.array([s[:, :min_samples] for s in sources])


def to_ica_activations(
    raws: list,
    icas: list,
    sfreq: float = 250.0,
    label: str = "ICA Activations",
) -> AnalysisData:
    """
    Build an ``AnalysisData`` from ICA source activations.

    :param raws: List of ``mne.io.Raw`` objects (one per subject).
    :param icas: List of ``mne.preprocessing.ICA`` objects.
    :param sfreq: Target sampling frequency (Hz).
    :param label: Human-readable label.
    """
    data = extract_ica_activations(raws, icas, sfreq=sfreq)
    n_components = data.shape[1]
    return AnalysisData(
        data=data,
        sfreq=sfreq,
        representation=DataRepresentation.ICA_ACTIVATIONS,
        label=label,
        feature_names=[f"IC{i}" for i in range(n_components)],
    )
