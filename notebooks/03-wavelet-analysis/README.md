# 03 — Wavelet Analysis

## Overview

This directory contains exploratory notebooks for analysing
**wavelet-transformed EEG data** produced by the wavelet persistence pipeline
(`scripts/run_analysis.py --analysis wavelet_power` or `wavelet_phase` with `--reuse_wavelets`).

Wavelet transforms provide a **time–frequency representation** of each
subject's EEG, enabling analyses that the raw broadband signal cannot support.
Whereas notebooks `01-*` and `02-*` operate on the raw (or z-scored) time
series, this suite works with 4-D arrays of shape
`(n_subjects, n_channels, n_frequencies, n_times)`.

Two representations are explored:
- **Power** — the magnitude of oscillatory activity at each frequency/time.
- **Phase** — the instantaneous angle (−π, π] of the oscillation, capturing
  *timing* independently of amplitude.

## Notebooks

| Notebook | Scope | Key analyses |
|---|---|---|
| `wavelet_power_exploration.ipynb` | Broadband + per-band | Spectral profile, time–frequency maps, band power time courses, intersubject variance, wavelet-domain LOO-ISC |
| `wavelet_phase_exploration.ipynb` | Broadband + per-band | Phase distribution check, ITPC spectrum, time–frequency ITPC map, per-band ITPC time course, phase-based LOO-ISC, ITPC vs ISC comparison |

## HPC job scripts

Job scripts in `jobs/metacentrum/03-wavelet-analysis/`:

| Script | Purpose |
|---|---|
| `run_wavelet_power.pbs` | Run wavelet power analysis (Placebo, both music types) |
| `run_wavelet_phase.pbs` | Run wavelet phase analysis (Placebo, both music types) |
| `store_wavelets.pbs` | Bulk-store power + phase wavelets for all conditions and music types |

## Analyses and rationale

### Power analyses (`wavelet_power_exploration.ipynb`)

#### 1. Grand-average spectral profile

**What:** Average wavelet power across subjects and time to obtain a
`(n_channels, n_frequencies)` matrix; then average over channels to get a
single `(n_frequencies,)` spectrum.

**Why:** Establishes which frequencies carry the most power in each condition
and music type.  Provides a sanity check that classical EEG peaks (alpha ~10 Hz)
are present and that the wavelet transform is behaving as expected.

#### 2. Time–frequency map (spectrogram)

**What:** For a representative channel (or channel-averaged), plot the 2-D
`(n_frequencies × n_times)` wavelet power image averaged across subjects.

**Why:** Reveals how the spectral content evolves over the course of the music
stimulus — e.g. whether alpha suppression coincides with high-arousal passages,
or whether gamma bursts accompany rhythmic accents.

#### 3. Per-band power time course

**What:** Average wavelet power within each canonical frequency band
(delta, theta, alpha, beta, gamma) across frequencies to obtain a
`(n_subjects, n_channels, n_times)` time series per band; then compute the
channel-averaged group mean ± SD.

**Why:** Directly comparable to the broadband time series in `01-*` but now
frequency-resolved.  Makes it straightforward to see, for example, that theta
synchrony increases during classical music while beta does not.

#### 4. Intersubject variance in the wavelet domain

**What:** For each frequency band, compute the intersubject variance at every
`(channel, time)` cell (same statistic as `01-*`), then plot the
channel-averaged variance time course.

**Why:** Extends the mean-variance synchrony framework from raw EEG to the
wavelet domain.  Regions of low intersubject variance in the wavelet power
indicate moments where subjects' spectral dynamics converge — a
frequency-resolved synchrony signal.

#### 5. Wavelet-domain LOO-ISC (per band)

**What:** For each band's averaged wavelet power `(n_subjects, n_channels,
n_times)`, compute leave-one-out ISC exactly as in `02-*`.  Report the
distribution across channels and the per-band mean.

**Why:** Provides a direct wavelet-domain counterpart to the broadband ISC
in `02-*`.  We expect bands that track stimulus features (e.g. delta/theta for
rhythm, alpha for attentional modulation) to show higher ISC.

### Phase analyses (`wavelet_phase_exploration.ipynb`)

Phase is a circular quantity in (−π, π], so standard linear statistics
(mean, variance, Pearson *r*) are replaced by circular analogues.

#### 1. Phase distribution sanity check

**What:** Histogram and polar plot of phase angles pooled across all
dimensions.

**Why:** Wavelet phase should be approximately uniformly distributed.  A
strongly peaked distribution would indicate artefacts, DC offsets, or
data-processing issues.

#### 2. Inter-Trial Phase Coherence (ITPC) spectrum

**What:** ITPC — also called Phase-Locking Value (PLV) — measures the
consistency of phase angles across subjects:
`ITPC(f, t) = |mean(exp(i·phase))|`.  Averaging over time gives a
frequency-domain ITPC profile.

**Why:** Identifies which frequencies exhibit stimulus-locked phase alignment
across participants.  High ITPC at a given frequency indicates that subjects'
oscillatory timing is entrained to the stimulus in a consistent way.

#### 3. Time–frequency ITPC map

**What:** Channel-averaged ITPC as a 2-D `(n_freqs × n_times)` image.

**Why:** Reveals *when* and at *which frequencies* subjects' phases align —
e.g. stimulus-locked phase resets at musical transitions, or sustained theta
entrainment during rhythmic passages.

#### 4. Per-band ITPC time course

**What:** Band-averaged, channel-averaged ITPC as a function of time.

**Why:** Frequency-resolved counterpart to the broadband analyses in `01-*`,
showing *phase consistency* instead of amplitude.  Directly comparable to the
per-band power time course in the power notebook.

#### 5. Phase-based LOO-ISC

**What:** Convert phase to a linear feature via `cos(phase)`, then compute
Pearson LOO-ISC — the same method used in `02-*` and the power notebook.

**Why:** Cosine projection captures the real component of the oscillation.
Pearson ISC on `cos(phase)` tests whether subjects share the same oscillatory
*timing* independently of amplitude.  Complementary to ITPC.

#### 6. ITPC vs. phase-ISC comparison

**What:** Side-by-side grouped bar chart and scatter plot comparing per-band
ITPC and LOO-ISC(cos φ).

**Why:** Bands where both metrics are elevated indicate robust, convergent
evidence for stimulus-driven phase synchrony.  Discrepancies can highlight
cases where one measure is more sensitive than the other.

## Future directions

Below are additional analyses that could extend these exploratory notebooks
into a full production pipeline:

1. **Condition comparison (Placebo vs. Psilocybin)** — once HPC-computed
   wavelets are available for both conditions, overlay or statistically compare
   spectral profiles, ISC curves, and variance time courses.

2. **Topographic mapping** — project per-channel band power or ISC onto the
   scalp montage using MNE topomaps to reveal spatial patterns of
   frequency-specific synchrony.

3. **Time–frequency ISC** — instead of averaging within a band and then
   computing ISC, compute ISC at every `(frequency, time)` cell to build a
   2-D ISC spectrogram.

4. **Statistical testing** — apply permutation-based tests (e.g. circular
   shift surrogates) to assess whether observed ISC values exceed chance
   levels.

5. **Sliding-window wavelet ISC** — combine the sliding-window approach of
   `02-*` Section 3 with frequency-resolved wavelet data for time-resolved,
   band-specific ISC.

6. **Cross-frequency coupling** — explore whether power in one band predicts
   the phase or power in another band, which would indicate nested oscillatory
   dynamics.

7. **Power–phase joint analysis** — overlay per-band power ISC and phase ISC
   results to identify bands where both amplitude and timing are synchronised
   vs. bands where only one modality is synchronised.
