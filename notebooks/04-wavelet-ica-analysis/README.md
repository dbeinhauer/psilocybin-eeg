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

## Two Decomposition Approaches

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

---

## Data Requirements

Both notebooks load wavelet-power data via `scripts.analysis_common` utilities
(the same pipeline used in `notebooks/03-wavelet-analysis/`).  The wavelet
cache directory defaults to `notebooks/04-wavelet-ica-analysis/wavelet_cache/`.

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
