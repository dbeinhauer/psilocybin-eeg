# Stage 04 — ICA / PCA Decomposition of Wavelet Power

This directory contains exploratory notebooks for applying **ICA** (Independent
Component Analysis) and **PCA** (Principal Component Analysis) to 4-D wavelet
power arrays of shape `(n_subjects, n_channels, n_frequencies, n_times)`.

The goal is to discover latent structure in the wavelet-power representation by
reshaping the data into 2-D matrices and decomposing them with standard
blind-source-separation / dimensionality-reduction techniques.

---

## Motivation

Wavelet power captures frequency-resolved amplitude dynamics across subjects,
channels, and time.  Direct ISC or variance analyses (stages 02 and 03)
operate on individual channels or bands.  ICA/PCA can reveal **multi-channel,
multi-frequency, and multi-subject** patterns that would be invisible to
univariate analyses — for example shared spectral-spatial modes of neural
processing that emerge during music listening.

---

## Six Decomposition Approaches

### Approach 1 — Super-Brain (temporal decomposition)

**Notebook:** `wavelet_ica_superbrain.ipynb`

Concatenate person, channel, and frequency into a single *feature* axis:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_channels × n_freqs,  n_times)
```

PCA/ICA treats **time** as the observation axis and finds a small number of
**temporal components** (independent or orthogonal time courses).  Each
component has loadings on every `(subject, channel, frequency)` triplet, which
can be decomposed into:

| Loading dimension | Interpretation |
|-------------------|----------------|
| Subject loadings  | Which participants contribute most to the component |
| Channel loadings  | Spatial topography (scalp map) of the component |
| Frequency loadings | Spectral profile — which bands dominate |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| 1 | PCA scree plot | How many components capture meaningful variance? |
| 2 | Component time courses | Temporal dynamics of each component |
| 3 | Subject loadings per component | Inter-individual variability |
| 4 | Channel loadings (topomap) | Spatial distribution |
| 5 | Frequency loadings per component | Spectral profile of each component |
| 6 | Component power spectrogram | Time–frequency view of each component |
| 7 | ISC of component time courses | Which components are shared across subjects? |
| 8 | Cross-component correlation | Dependence between temporal components |
| 8.2 | ICA topomap — mean loading | Mean ICA channel loading across subjects |
| 8.3 | ICA topomap — variance | Variance of ICA channel loading across subjects |

### Approach 2 — Inter-Subject Frequency-Channel (spatial-spectral decomposition)

**Notebook:** `wavelet_ica_intersubject.ipynb`

Concatenate person and time into the *observation* axis, with channel and
frequency as *features*:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_times,  n_channels × n_freqs)
```

PCA/ICA treats every `(subject, time-point)` pair as an observation and finds
**spatial-spectral components** — recurring channel × frequency patterns that
appear across subjects and time.

| Loading dimension | Interpretation |
|-------------------|----------------|
| Channel × frequency map | The spatial-spectral "fingerprint" of the component |
| Per-subject activation | How strongly each subject activates the pattern over time |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| 1 | PCA scree plot | Intrinsic dimensionality of the spectral-spatial space |
| 2 | Component channel × frequency maps | Spatial-spectral patterns |
| 3 | Per-subject activation time courses | Subject-specific dynamics for each component |
| 4 | Component topographic maps | Scalp projection of each spatial-spectral component |
| 5 | Inter-subject similarity | Correlation of component activations across subjects |
| 6 | Band-resolved component loadings | Aggregate component weights per frequency band |
| 7 | Temporal dynamics of components | Time-averaged component activation across subjects |
| 8 | Component correlation matrix | Dependence between spatial-spectral components |
| 8.2 | ICA topomap — mean loading | Mean ICA channel loading across subjects |
| 8.3 | ICA topomap — variance | Variance of ICA channel loading across subjects |

### Approach 3 — Temporal (temporal-spectral decomposition)

**Notebook:** `wavelet_ica_temporal.ipynb`

Concatenate person and channel into the *observation* axis, with frequency
and time as *features*:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_channels,  n_freqs × n_times)
```

Each observation is a single electrode from a single subject, described by
its full frequency × time power surface.  ICA discovers **temporal-spectral
components** — recurring frequency × time patterns shared across subjects
and channels.  The focus is on **time**: components that are active in the
same temporal intervals across subjects indicate stimulus-driven processing.

| Loading dimension | Interpretation |
|-------------------|----------------|
| Subject loadings  | Which participants contribute most to the component |
| Channel loadings  | Spatial topography (scalp map) of the component |

| Component dimension | Interpretation |
|---------------------|----------------|
| Frequency × time map | When and at what frequency the component is active |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| 1 | PCA scree plot | Intrinsic dimensionality of the freq × time space |
| 2 | ICA frequency × time maps | Temporal-spectral component patterns |
| 3 | ICA topomap — mean loading | Mean channel loading across subjects |
| 4 | ICA topomap — variance | Inter-individual variability in channel loading |
| 5 | Per-subject ICA channel loadings | Subject-specific topographic maps |
| 6 | ICA time courses | Frequency-marginalized temporal profiles |
| 7 | ICA spectral profiles | Time-marginalized frequency profiles |
| 8 | Inter-subject similarity | Correlation of channel loadings across subjects |
| 9 | Band-resolved energy | Which frequency bands dominate each component |
| 10 | Component correlation matrix | Dependence between temporal-spectral components |

### Approach 4 — Inverted Super-Brain (observation-per-triplet temporal decomposition)

**Notebook:** `wavelet_ica_inverted_superbrain.ipynb`

Uses the same 2-D matrix as the super-brain, but **swaps observations and
features** — person × channel × frequency are *observations* and time is the
*feature* axis:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_channels × n_freqs,  n_times)
                ─────── observations ───────────────  features
```

Each observation is a specific subject–electrode–frequency combination,
described by its T-length power time course.  PCA/ICA discover **temporal
component patterns** (each of length T) shared across the S × C × F
observations.  The score vector for each component can be reshaped to
`(n_subjects, n_channels, n_freqs)` and decomposed into subject, channel, and
frequency loadings — exactly as in the super-brain, but from the score side.

| Quantity | Shape | Interpretation |
|----------|-------|----------------|
| Component pattern | `(T,)` | Temporal pattern shared across observations |
| Subject loadings | `(n_subjects,)` | Mean |score| over channels and frequencies |
| Channel loadings | `(n_channels,)` | Spatial topography (scalp map) |
| Frequency loadings | `(n_freqs,)` | Spectral profile |

**Key advantage:** the observation-to-feature ratio is very high (S×C×F >> T
after subsampling), which makes ICA estimation statistically robust.

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| 1 | PCA scree plot | Intrinsic dimensionality of the temporal feature space |
| 2 | ICA temporal component patterns | Time-domain components shared across observations |
| 3 | Subject loadings per IC | Inter-individual variability |
| 4 | ICA channel loadings (topomap) | Spatial distribution from ICA scores |
| 5 | ICA topomap — mean loading | Mean channel loading across subjects |
| 6 | ICA topomap — variance | Inter-individual variability in channel loading |
| 7 | Frequency loadings per IC | Spectral profile of each component |
| 8 | ICA component spectrograms | Time–frequency view of each component pattern |
| 9 | Inter-individual IC correlations | Consistency of channel loadings across subjects |
| 10 | Cross-component correlation | PCA vs ICA score dependence |

### Approach 5 — Subject-Frequency (spatial-temporal decomposition)

**Notebook:** `wavelet_ica_subject_frequency.ipynb`

Concatenate person and frequency into the *observation* axis, with channel
and time as *features*:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_freqs,  n_channels × n_times)
                ──── observations ────  ──── features ────────
```

Each observation is a specific subject–frequency combination, described by
its full channel × time power surface.  PCA/ICA discover **spatial-temporal
components** — recurring channel × time patterns shared across subjects
and frequencies.

| Loading dimension | Interpretation |
|-------------------|----------------|
| Channel × time map | The spatial-temporal "fingerprint" of the component |
| Per-subject frequency profile | How strongly each subject activates the pattern across frequencies |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| 1 | PCA scree plot | Intrinsic dimensionality of the channel × time space |
| 2 | PCA channel × time maps | Spatial-temporal component patterns |
| 3 | Per-subject frequency profiles | Subject-specific frequency activation per component |
| 4 | PCA topographic maps | Time-averaged channel marginal → scalp maps |
| 5 | Cross-component correlation | PCA vs ICA dependence structure |
| 6 | ICA channel × time maps | Independent spatial-temporal patterns |
| 7 | ICA topomap — mean loading | Mean channel loading across time |
| 8 | ICA topomap — variance | Variance of channel loading across time |
| 9 | ICA time courses | Channel-averaged temporal profiles |
| 10 | Inter-individual IC correlations | Consistency of frequency profiles across subjects |

### Approach 6 — Combined Features (focused ICA analysis)

**Notebook:** `wavelet_ica_combined_features.ipynb`

Uses the same reshape as the inverted super-brain (Approach 4) — all three
main dimensions (subjects, channels, frequencies) are combined into one
observation axis with time as features:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_channels × n_freqs,  n_times)
                ─────── observations ───────────────  features
```

This notebook focuses on a curated set of five analyses that together
characterise each ICA component across subjects, space, frequency, and time:

| # | Analysis | Purpose |
|---|----------|---------|
| (a) | Intersubject correlation matrix | Consistency of IC loadings across participants |
| (b) | Temporal loadings (mean ± std) | When each IC is active, with inter-subject variability |
| (c) | Time–frequency spectrograms | Spectral content of each IC temporal pattern |
| (d) | Mean / variance topomaps | Spatial distribution and inter-individual variability |
| (e) | Per-subject loading bars | Individual-level contribution to each IC |

---

## Data Requirements

All six notebooks load wavelet-power data via `scripts.analysis_common`
utilities (the same pipeline used in `notebooks/03-wavelet-analysis/`).
The wavelet cache directory defaults to
`notebooks/04-wavelet-ica-analysis/wavelet_cache/`.

Subject, channel, and time subsets are controlled in the configuration cell so
that the notebooks remain fast for interactive exploration (defaults:
5 subjects, 32 channels, 10 000 time samples).

---

## Dependencies

All decomposition methods use **scikit-learn**:

- `sklearn.decomposition.PCA` — principal component analysis
- `sklearn.decomposition.FastICA` — independent component analysis

These are already available in the project environment (scikit-learn is a
dependency of MNE-Python).

---

## Future Directions

- **Condition comparison**: Run the same decomposition on Psilocybin data and
  compare component structure (angle between subspaces, Procrustes alignment).
- **Statistical testing**: Permutation tests on component loadings.
- **Higher-order decomposition**: Tensor decomposition (e.g. Tucker, CP) that
  avoids the 2-D reshape altogether and decomposes the full 4-D tensor
  directly.
- **Production pipeline**: Migrate mature analyses to `src/analysis/` as pure
  functions and expose via `scripts/run_wavelet_ica.py`.
