"""
Canonical EEG frequency-band ranges and wavelet frequency-grid defaults.

Single source of truth for:

* :data:`FREQUENCY_BANDS` — the ``(low, high)`` Hz range of each standard EEG
  band, keyed by :class:`~src.definitions.fields.FrequencyBandNames` value.
* The default Morlet wavelet frequency grid
  (:data:`WAVELET_FREQ_MIN`, :data:`WAVELET_FREQ_MAX`, :data:`WAVELET_N_FREQS`)
  shared by every wavelet analysis script.

Import these here rather than re-declaring band ranges or grid defaults in
individual analysis modules or CLI scripts.
"""

from __future__ import annotations

from src.definitions.fields import FrequencyBandNames

#: Standard EEG frequency bands. Each entry maps a
#: :class:`~src.definitions.fields.FrequencyBandNames` *value* (a plain string)
#: to its ``(l_freq, h_freq)`` bounds in Hz. Keyed by the string value so the
#: mapping can be indexed directly with CLI / band-name strings.
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    FrequencyBandNames.DELTA.value: (1.0, 4.0),
    FrequencyBandNames.THETA.value: (4.0, 8.0),
    FrequencyBandNames.ALPHA.value: (8.0, 13.0),
    FrequencyBandNames.BETA.value: (13.0, 30.0),
    FrequencyBandNames.GAMMA.value: (30.0, 70.0),
}

#: Default Morlet wavelet frequency grid shared by all wavelet analyses.
#: ``numpy.linspace(WAVELET_FREQ_MIN, WAVELET_FREQ_MAX, WAVELET_N_FREQS)`` yields
#: a 1 Hz-spaced grid over 1-50 Hz. Per-band wavelet analyses slice this grid to
#: each band's :data:`FREQUENCY_BANDS` range, so bands above
#: :data:`WAVELET_FREQ_MAX` (e.g. the upper part of gamma) are covered only up
#: to the grid's maximum frequency.
WAVELET_FREQ_MIN: float = 1.0
WAVELET_FREQ_MAX: float = 50.0
WAVELET_N_FREQS: int = 50
