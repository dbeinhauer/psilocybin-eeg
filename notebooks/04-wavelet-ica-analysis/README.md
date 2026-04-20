# Stage 04 — ICA / PCA Decomposition of Wavelet Power (3 Approaches)

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

## Three Decomposition Approaches

### Approach 1 — Combined Features (S×C×F observations, T features)

**Notebook:** `wavelet_ica_combined_features.ipynb`

All three main dimensions (subjects, channels, frequencies) are combined into
one observation axis with time as features:

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
frequency loadings.

| Quantity | Shape | Interpretation |
|----------|-------|----------------|
| Component pattern | `(T,)` | Temporal pattern shared across observations |
| Subject loadings | `(n_subjects,)` | Mean score over channels and frequencies |
| Channel loadings | `(n_channels,)` | Spatial topography (scalp map) |
| Frequency loadings | `(n_freqs,)` | Spectral profile |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| (a) | Intersubject correlation matrix | Consistency of IC loadings across participants |
| (b) | Temporal loadings (mean ± std) | When each IC is active, with inter-subject variability |
| (c) | Frequency × time mean-loading heatmap | Mean IC loading per subject at each wavelet frequency and time |
| (d) | Mean / variance topomaps | Spatial distribution and inter-individual variability |
| (e) | Per-subject loading bars | Individual-level contribution to each IC |
| (f) | IC temporal patterns | Raw component waveforms (temporal fingerprint) |
| (g) | Per-subject per-component heatmap | Subjects × ICs loading strength matrix |
| (i) | Subject-consistency bar plot | Mean pairwise inter-subject correlation per IC |
| (j) | Frequency profile per component | Which band dominates each IC |

### Approach 2 — Subject-Frequency Features (S×F observations, C×T features)

**Notebook:** `wavelet_ica_subject_freq_features.ipynb`

Concatenate subjects and frequencies into the observation axis, with
channels and time as features:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_subjects × n_freqs,  n_channels × n_times)
                ──── observations ────  ──── features ────────
```

Each observation is a specific subject–frequency combination, described by
its full channel × time power surface.  PCA/ICA discover **spatial-temporal
component patterns** — recurring channel × time fingerprints shared across
subjects and frequencies.  The ICA scores can be reshaped to `(S, F, K)` to
reveal which subjects and which frequency bands activate each mode.

| Aspect | Approach 1 (Combined Features) | Approach 2 (Subject-Freq Features) |
|--------|--------------------------------|------------------------------------|
| Observations | S × C × F | S × F |
| Features | T | C × T |
| Components represent | Temporal patterns `(T,)` | Spatial-temporal modes `(C, T)` |
| Channel info lives in | Scores (axis 1) | Components (axis 1) |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| (a) | Intersubject correlation matrix | Consistency of IC frequency profiles across participants |
| (b) | Temporal profiles (mean ± std) | When each mode is active, with inter-subject variability |
| (c) | Frequency × time mean-loading heatmap | Mean IC loading per subject at each wavelet frequency and time |
| (d) | Mean / variance topomaps | Spatial distribution and inter-individual variability |
| (e) | Per-subject loading bars | Individual-level participation in each mode |

### Approach 3 — Freq-Channel Features (F×C observations, S×T features)

**Notebook:** `wavelet_ica_freq_channel_features.ipynb`

Concatenate frequencies and channels into the observation axis, with
subjects and time as features:

```
(n_subjects, n_channels, n_freqs, n_times)
  → reshape →  (n_freqs × n_channels,  n_subjects × n_times)
                ──── observations ─────  ──── features ──────
```

Each observation is a specific frequency–channel combination, described by
its full subject × time power surface.  PCA/ICA discover **subject-temporal
component patterns** — recurring subject × time fingerprints shared across
frequencies and channels.  The ICA scores can be reshaped to `(F, C, K)` to
reveal which frequencies and which channels activate each mode; ICA
components reshape to `(K, S, T)` giving the subject × time pattern.

| Aspect | Approach 2 (Subject-Freq Features) | Approach 3 (Freq-Channel Features) |
|--------|--------------------------------------|--------------------------------------|
| Observations | S × F | F × C |
| Features | C × T | S × T |
| Components represent | Spatial-temporal modes `(C, T)` | Subject-temporal modes `(S, T)` |
| Scores represent | Per-subject-freq weights `(S, F, K)` | Per-freq-channel weights `(F, C, K)` |
| Subject info lives in | Scores (axis 0) | Components (axis 1) |

**Analyses in the notebook:**

| # | Analysis | Purpose |
|---|----------|---------|
| (a) | Intersubject correlation matrix | Consistency of IC temporal profiles across participants |
| (b) | Temporal profiles (mean ± std) | When each mode is active, with inter-subject variability |
| (c) | Frequency × time mean-loading heatmap | Mean IC loading per subject at each wavelet frequency and time |
| (d) | Mean / variance topomaps | Spatial distribution and inter-individual variability |
| (e) | Per-subject loading bars | Individual-level participation in each mode |
| (f) | IC temporal patterns | Raw component waveforms (temporal fingerprint) |
| (g) | Intersubject temporal overlay | All subjects' temporal profiles overlaid per IC |
| (h) | Subject-consistency bar plot | Mean pairwise inter-subject correlation per IC |

---

## Data Requirements

All three notebooks load wavelet-power data via `scripts.analysis_common`
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
  functions and expose via dedicated CLI scripts.
