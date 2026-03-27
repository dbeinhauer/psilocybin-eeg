# 03 — Wavelet Analysis

## Overview

This directory contains exploratory notebooks for analysing
**wavelet-transformed EEG data** produced by the wavelet persistence pipeline
(`scripts/run_analysis.py --analysis wavelet_power --reuse_wavelets`).

Wavelet power provides a **time–frequency representation** of each subject's
EEG, enabling analyses that the raw broadband signal cannot support.  Whereas
notebooks `01-*` and `02-*` operate on the raw (or z-scored) time series, this
suite works with 4-D arrays of shape
`(n_subjects, n_channels, n_frequencies, n_times)`.

## Notebooks

| Notebook | Scope | Key analyses |
|---|---|---|
| `wavelet_power_exploration.ipynb` | Broadband + per-band | Spectral profile, time–frequency maps, band power time courses, intersubject variance in wavelet domain, wavelet-domain ISC |

## Analyses and rationale

### 1. Grand-average spectral profile

**What:** Average wavelet power across subjects and time to obtain a
`(n_channels, n_frequencies)` matrix; then average over channels to get a
single `(n_frequencies,)` spectrum.

**Why:** Establishes which frequencies carry the most power in each condition
and music type.  Provides a sanity check that classical EEG peaks (alpha ~10 Hz)
are present and that the wavelet transform is behaving as expected.

### 2. Time–frequency map (spectrogram)

**What:** For a representative channel (or channel-averaged), plot the 2-D
`(n_frequencies × n_times)` wavelet power image averaged across subjects.

**Why:** Reveals how the spectral content evolves over the course of the music
stimulus — e.g. whether alpha suppression coincides with high-arousal passages,
or whether gamma bursts accompany rhythmic accents.

### 3. Per-band power time course

**What:** Average wavelet power within each canonical frequency band
(delta, theta, alpha, beta, gamma) across frequencies to obtain a
`(n_subjects, n_channels, n_times)` time series per band; then compute the
channel-averaged group mean ± SD.

**Why:** Directly comparable to the broadband time series in `01-*` but now
frequency-resolved.  Makes it straightforward to see, for example, that theta
synchrony increases during classical music while beta does not.

### 4. Intersubject variance in the wavelet domain

**What:** For each frequency band, compute the intersubject variance at every
`(channel, time)` cell (same statistic as `01-*`), then plot the
channel-averaged variance time course.

**Why:** Extends the mean-variance synchrony framework from raw EEG to the
wavelet domain.  Regions of low intersubject variance in the wavelet power
indicate moments where subjects' spectral dynamics converge — a
frequency-resolved synchrony signal.

### 5. Wavelet-domain LOO-ISC (per band)

**What:** For each band's averaged wavelet power `(n_subjects, n_channels,
n_times)`, compute leave-one-out ISC exactly as in `02-*`.  Report the
distribution across channels and the per-band mean.

**Why:** Provides a direct wavelet-domain counterpart to the broadband ISC
in `02-*`.  We expect bands that track stimulus features (e.g. delta/theta for
rhythm, alpha for attentional modulation) to show higher ISC.

## Future directions

Below are additional analyses that could extend this exploratory notebook into
a full production pipeline:

1. **Phase-based ISC** — repeat all analyses using wavelet *phase* instead of
   power (`representation="phase"`).  Phase coherence can capture synchrony
   that amplitude-based measures miss.

2. **Condition comparison (Placebo vs. Psilocybin)** — once HPC-computed
   wavelets are available for both conditions, overlay or statistically compare
   spectral profiles, ISC curves, and variance time courses.

3. **Topographic mapping** — project per-channel band power or ISC onto the
   scalp montage using MNE topomaps to reveal spatial patterns of
   frequency-specific synchrony.

4. **Time–frequency ISC** — instead of averaging within a band and then
   computing ISC, compute ISC at every `(frequency, time)` cell to build a
   2-D ISC spectrogram.

5. **Statistical testing** — apply permutation-based tests (e.g. circular
   shift surrogates) to assess whether observed ISC values exceed chance
   levels.

6. **Sliding-window wavelet ISC** — combine the sliding-window approach of
   `02-*` Section 3 with frequency-resolved wavelet data for time-resolved,
   band-specific ISC.

7. **Cross-frequency coupling** — explore whether power in one band predicts
   the phase or power in another band, which would indicate nested oscillatory
   dynamics.
