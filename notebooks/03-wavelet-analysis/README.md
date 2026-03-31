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
| `wavelet_power_exploration.ipynb` | Broadband + per-band | Spectral profile, time–frequency maps, band power time courses, intersubject variance, wavelet-domain LOO-ISC, topographic mapping, time–frequency ISC, sliding-window ISC, cross-frequency coupling, power–phase joint analysis |
| `wavelet_phase_exploration.ipynb` | Broadband + per-band | Phase distribution check, ITPC spectrum, time–frequency ITPC map, per-band ITPC time course, phase-based LOO-ISC, ITPC vs ISC comparison, topographic phase-ISC mapping |

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

#### 6. Topographic mapping

**What:** Project per-channel mean LOO-ISC for each frequency band onto the
scalp montage using MNE topomaps.

**Why:** Reveals which brain regions show the strongest frequency-specific
synchrony, adding a spatial dimension to the per-band ISC results.

#### 7. Time–frequency ISC

**What:** Reshape broadband data to `(n_subjects, n_channels × n_freqs,
n_times)` and compute LOO-ISC per frequency to obtain a channel-averaged
ISC-vs-frequency profile.

**Why:** Provides a continuous frequency-resolved view of synchrony without
committing to discrete band boundaries.

#### 8. Sliding-window wavelet ISC

**What:** For each band, collapse the frequency dimension and compute LOO-ISC
in successive time windows, yielding a time-resolved ISC trace per band.

**Why:** Combines the sliding-window temporal resolution of `02-*` with the
frequency specificity of the wavelet representation.

#### 9. Cross-frequency coupling

**What:** Compute the Pearson correlation between every pair of bands'
channel-averaged, subject-mean power time courses, yielding a band × band
correlation matrix.

**Why:** Identifies co-modulation patterns between frequency bands, which can
indicate nested oscillatory dynamics driven by the stimulus.

#### 10. Power–phase joint analysis

**What:** Overlay per-band power LOO-ISC and phase LOO-ISC (computed via
`cos(phase)` projection from the phase wavelet cache) in a grouped bar chart.

**Why:** Distinguishes bands where both amplitude and timing are synchronised
from bands where only one modality is synchronised, providing a more complete
picture of inter-subject coherence.

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

#### 7. Topographic mapping of per-band phase-ISC

**What:** Project per-channel mean LOO-ISC(cos φ) for each frequency band onto
the scalp montage using MNE topomaps.  Channel positions are obtained from the
analyzer's `info` attribute and subsetted to match the number of channels used
in the notebook.

**Why:** Reveals which brain regions exhibit the strongest inter-subject
*phase* synchrony at each frequency, providing a spatial complement to the
band-level bar charts.  Directly comparable to the power-ISC topomap in
`wavelet_power_exploration.ipynb` — differences between the two highlight
regions where amplitude and timing are decoupled.

## Future directions

Below are additional analyses that could extend these exploratory notebooks
into a full production pipeline:

1. **Condition comparison (Placebo vs. Psilocybin)** — once HPC-computed
   wavelets are available for both conditions, overlay or statistically compare
   spectral profiles, ISC curves, and variance time courses.

2. **Statistical testing** — apply permutation-based tests (e.g. circular
   shift surrogates) to assess whether observed ISC values exceed chance
   levels.
