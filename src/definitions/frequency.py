"""
Canonical EEG frequency-band ranges and wavelet frequency-grid defaults.

Single source of truth for:

* :data:`FREQUENCY_BANDS` — the ``(low, high)`` Hz range of each standard EEG
  band, keyed by :class:`~src.definitions.fields.FrequencyBandNames` value.
* The default Morlet wavelet frequency grid
  (:data:`WAVELET_FREQ_MIN`, :data:`WAVELET_FREQ_MAX`, :data:`WAVELET_N_FREQS`)
  shared by every wavelet analysis script.
* :func:`resolve_band_range` / :func:`band_token` — how a **band restriction** is
  read, whether it names a standard band or gives an explicit Hz window.

Import these here rather than re-declaring band ranges or grid defaults in
individual analysis modules or CLI scripts.
"""

from __future__ import annotations

import math

from collections.abc import Sequence

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

#: How a band restriction is expressed anywhere in the project: the **name** of a
#: :data:`FREQUENCY_BANDS` entry, an explicit ``(low, high)`` Hz window, or ``None``
#: for the whole grid.
#:
#: The explicit window exists because the standard bands are too coarse for some
#: questions. The 40 Hz ASSR is the case that forced it: the narrowest band that
#: contains it is ``gamma``, which on the 1-50 Hz wavelet grid is 21 of 50 bins, and a
#: run that wants 35-45 Hz has no name to ask for. Adding a member to
#: :class:`~src.definitions.fields.FrequencyBandNames` for every such window would turn
#: an enum of *standard* bands into a list of one analysis's choices.
BandSpec = str | Sequence[float] | None


def resolve_band_range(band: BandSpec) -> tuple[float, float] | None:
    """``(low, high)`` Hz bounds of a band restriction.

    :param band: A :data:`FREQUENCY_BANDS` name, an explicit ``(low, high)`` Hz pair,
        or ``None`` for the whole grid.
    :return: The bounds in Hz, or ``None`` when *band* is ``None``.
    :raises ValueError: If a name is not a known band, a pair is not two finite
        numbers, or the low bound exceeds the high one.
    """
    if band is None:
        return None
    if isinstance(band, str):
        if band not in FREQUENCY_BANDS:
            raise ValueError(
                f"Unknown band {band!r}; known bands are "
                f"{sorted(FREQUENCY_BANDS)}. Pass an explicit (low, high) Hz pair "
                "for a window no standard band describes."
            )
        return FREQUENCY_BANDS[band]

    bounds = tuple(band)
    if len(bounds) != 2:
        raise ValueError(
            f"A band range must be exactly (low, high) in Hz; got {bounds!r}."
        )
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Band range bounds must be numbers; got {bounds!r}."
        ) from error
    if not (math.isfinite(low) and math.isfinite(high)):
        raise ValueError(f"Band range bounds must be finite; got {bounds!r}.")
    if low < 0.0:
        raise ValueError(f"Band range bounds must be >= 0 Hz; got low={low}.")
    if low > high:
        raise ValueError(f"Band range low ({low}) is above high ({high}).")
    return low, high


def band_token(band: BandSpec) -> str | None:
    """Canonical name of a band restriction, for filenames and store keys.

    A **named** band keeps its name, so a store or a figure written before explicit
    ranges existed keeps the path it always had. An explicit window becomes
    ``"<low>-<high>hz"`` (e.g. ``"30-50hz"``), which
    :func:`~src.io.iva_store.iva_results_filename` accepts verbatim — its sanitiser
    keeps ``-`` — so a windowed run coexists with the broadband and named-band ones
    rather than overwriting either.

    :param band: A :data:`FREQUENCY_BANDS` name, an explicit ``(low, high)`` Hz pair,
        or ``None``.
    :return: The token, or ``None`` when *band* is ``None``.
    :raises ValueError: Whatever :func:`resolve_band_range` rejects.
    """
    bounds = resolve_band_range(band)  # validates, rather than trusting the input
    if bounds is None:
        return None
    if isinstance(band, str):
        return band
    low, high = bounds
    return f"{low:g}-{high:g}hz"
