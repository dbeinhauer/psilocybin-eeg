# EEG Analysis Package

This package contains all analysis modules for the psilocybin-EEG dataset. It computes
**Inter-Subject Correlation (ISC)**, **intersubject mean-variance synchrony**, and related
metrics from preprocessed, time-aligned EEG recordings, and supports multiple data
representations (raw channels, ICA activations, wavelet amplitudes, mean responses).

## Module Overview

| Module | Purpose |
|--------|---------|
| `data_representations.py` | `AnalysisData` container + adapters to convert raw EEG into wavelet, ICA, or mean-response representations |
| `isc.py` | Pure ISC computation functions (leave-one-out, pairwise, sliding-window) and the `FREQUENCY_BANDS` constant |
| `mean_variance.py` | Intersubject mean-variance synchrony analysis (notebook-based implementation) |
| `results_store.py` | CSV export for analysis results consumed by the Interactive Explorer (`viz_catalog/`) |
| `summary.py` | `EEGSummarizedAnalyzer` — high-level orchestrator for loading data, running ISC analyses, and persisting results |

---

## Data Shape Convention

All analysis functions operate on 3D NumPy arrays of shape:

```
(n_items, n_features, n_samples)
```

| Axis | Dimension | Typical meaning |
|------|-----------|-----------------|
| 0 | `n_items` | subjects (or groups) |
| 1 | `n_features` | EEG channels (or ICA components, wavelet bands, …) |
| 2 | `n_samples` | time points |

The `AnalysisData` dataclass from `data_representations.py` wraps this array with metadata
(sampling frequency, representation type, feature names, MNE info) so that downstream
functions can interpret axes correctly regardless of the underlying representation.

---

## `data_representations.py`

### `AnalysisData` dataclass

Central container used by all analysis and visualization functions.

```python
from src.analysis.data_representations import AnalysisData, DataRepresentation

ad = AnalysisData(
    data=array,          # shape (n_items, n_features, n_samples)
    sfreq=500.0,
    representation=DataRepresentation.TIME_DOMAIN,
    label="Placebo / Classical",
    feature_names=raw.ch_names,
    info=raw.info,
)
```

### Adapter functions

| Function | Input → Output |
|----------|----------------|
| `from_array` | raw ndarray → `AnalysisData` |
| `to_analytic_amplitude` | TIME_DOMAIN → WAVELET_AMPLITUDE (Hilbert) |
| `to_wavelet_power` | TIME_DOMAIN → WAVELET_POWER (Morlet) |
| `to_wavelet_phase` | TIME_DOMAIN → WAVELET_PHASE (Morlet) |
| `to_wavelet_tfr` | TIME_DOMAIN → full TFR (n_items, n_features, n_freqs, n_samples) |
| `extract_ica_activations` | TIME_DOMAIN + `ICA` → ICA_ACTIVATIONS |
| `to_ica_activations` | TIME_DOMAIN + `ICA` → ICA_ACTIVATIONS (`AnalysisData`) |

---

## `isc.py`

Pure functions — no side effects, no file I/O. All accept `data: np.ndarray` of shape
`(n_items, n_features, n_samples)`.

### Pearson ISC functions

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_loo_isc(data)` | Leave-one-out ISC: each subject vs mean of the rest | `(loo_isc, mean_loo_isc)` shapes `(n_items, n_features)` and `(n_features,)` |
| `compute_pairwise_isc(data)` | Mean-across-features Pearson r for every subject pair | Symmetric `(n_items, n_items)` matrix |
| `compute_pairwise_isc_per_feature(data)` | Pairwise ISC kept per feature | `(n_items, n_items, n_features)` |
| `compute_sliding_window_isc(data, window_sec, step_sec, sfreq)` | LOO-ISC in overlapping time windows (mean across subjects) | `(isc_timecourse, window_times)` shapes `(n_windows, n_features)` and `(n_windows,)` |

### Spearman ISC functions

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_loo_isc_spearman(data)` | Leave-one-out ISC using Spearman rank correlation | `(loo_isc, mean_loo_isc)` shapes `(n_items, n_features)` and `(n_features,)` |
| `compute_pairwise_isc_spearman(data)` | Mean-across-features Spearman r for every subject pair | Symmetric `(n_items, n_items)` matrix |
| `compute_sliding_window_isc_spearman(data, window_sec, step_sec, sfreq)` | Spearman LOO-ISC in overlapping time windows | `(isc_timecourse, window_times)` shapes `(n_windows, n_features)` and `(n_windows,)` |

### Mean-field ISC functions

These functions first average the signal across all channels to obtain a single global
mean-field time series per subject, then compute ISC on those 1-D signals.
`ISC(mean(channels))` can differ substantially from `mean(ISC(channel_i))` when
synchrony is spatially heterogeneous.

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_mean_field_loo_isc(data)` | LOO-ISC on the spatial mean-field (Pearson + Spearman) | `(loo_pearson, loo_spearman)` each shape `(n_items,)` |
| `compute_mean_field_pairwise_isc(data)` | Pearson pairwise ISC on the mean-field | Symmetric `(n_items, n_items)` matrix |
| `compute_mean_field_sliding_window_isc(data, window_sec, step_sec, sfreq)` | Time-resolved mean-field LOO-ISC | `(isc_timecourse, window_times)` shapes `(n_windows,)` and `(n_windows,)` |

### Frequency band constant

```python
from src.analysis.isc import FREQUENCY_BANDS
# {FrequencyBandNames.DELTA.value: (1.0, 4.0), ..., FrequencyBandNames.GAMMA.value: (30.0, 70.0)}
```

---

## `mean_variance.py`

Implements the intersubject mean-variance synchrony analysis demonstrated in
`notebooks/01-raw-mean-variance-analysis/mean_variance_broadband.ipynb` and
`notebooks/01-raw-mean-variance-analysis/mean_variance_bands.ipynb`.

All functions work on z-scored data of shape `(n_subjects, n_channels, n_times)`.

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_intersubject_stats(data)` | Compute intersubject variance, mean, std, and per-subject channel-average | Dict with keys `inter_var`, `inter_mean`, `mean_t`, `var_t`, `std_t`, `mean_over_ch` |
| `compute_windowed_stats(stats, n_times, sfreq, window_sec, sync_percentile, step_sec)` | Windowed statistics with synchrony-candidate labelling (50% overlap by default; pass `step_sec=window_sec` for non-overlapping) | `pandas.DataFrame` with per-window statistics |
| `compute_band_intersubject_stats(ad, bands)` | Apply `compute_intersubject_stats` to each frequency band after bandpass filtering | `{band_name: stats_dict}` |
| `compute_pairwise_isc_matrices(band_data)` | Mean Pearson correlation matrix between every subject pair per band | `{band_name: (n_subjects, n_subjects) matrix}` |

### Typical workflow

```python
from src.analysis.mean_variance import (
    compute_intersubject_stats,
    compute_windowed_stats,
    compute_band_intersubject_stats,
)

# ad is an AnalysisData with z-scored data of shape (n_subjects, n_channels, n_times)
stats = compute_intersubject_stats(ad.data)

# Windowed analysis — 2-second windows, mark lowest-10% variance as sync candidates
df_windows = compute_windowed_stats(
    stats, n_times=ad.data.shape[2], sfreq=ad.sfreq,
    window_sec=2.0, sync_percentile=10.0,
)

# Per-band stats
band_stats = compute_band_intersubject_stats(ad)
```

---

## `summary.py`

`EEGSummarizedAnalyzer` is the high-level orchestrator. It handles:

- **Data loading** — reads concatenated NumPy arrays and participant metadata via `DatasetHandler`
- **Filtering** — restricts the dataset to a specific condition, music type, or participant subset
- **Normalization** — optional z-score normalization across time (axis 2)
- **ISC computation** — delegates to `isc.py` pure functions
- **Persistence** — saves/loads processed arrays and metadata to/from disk

### Typical workflow

```python
from src.analysis.summary import EEGSummarizedAnalyzer
from src.definitions.fields import ConditionVariants, MusicTypeVariants

analyzer = EEGSummarizedAnalyzer()
analyzer.load_and_prepare_data(
    condition=ConditionVariants.PLACEBO,
    music_type=MusicTypeVariants.CLASSICAL,
)

# Run leave-one-out ISC
loo_isc, mean_isc = analyzer.compute_loo_isc()

# Band-specific ISC (filters data to frequency band, then computes LOO ISC)
band_loo, band_mean = analyzer.compute_band_isc(band=FrequencyBandNames.ALPHA)

# Persist results
analyzer.save_data(save_path)
```

### Key methods

| Method | Description |
|--------|-------------|
| `load_and_prepare_data(condition, music_type)` | Load concatenated array + metadata, apply optional exclusions |
| `save_data(path)` / `load_data(path)` | Persist and restore the array + `filtered_df` metadata |
| `normalize(axis)` | In-place z-score normalization |
| `to_analysis_data(label)` | Export as `AnalysisData` for use with adapter functions |
| `compute_loo_isc()` | Leave-one-out ISC across all channels |
| `compute_pairwise_isc()` | Pairwise ISC matrix |
| `compute_sliding_window_isc(window, step)` | Sliding-window LOO ISC |
| `compute_band_isc(band)` | Bandpass-filter then LOO ISC for a named frequency band |
| `compute_band_sliding_window_isc(band, window, step)` | Sliding-window LOO ISC per band |

---

## Adding a New Analysis

1. **Sketch in a Jupyter notebook** — explore the data, prototype the computation, and validate results visually.
2. Add the core computation as a **pure function** in `isc.py` (for ISC) or a new module — no file I/O, no side effects.
3. Expose it through `EEGSummarizedAnalyzer` in `summary.py` if it needs data loading / persistence.
4. Create a **CLI script** in `scripts/` and a corresponding **HPC job template** in `jobs/metacentrum/`.
5. Write tests in `tests/test_<module>.py` targeting the pure function and the analyzer method.
6. Use **Seaborn** for all statistical plots; fall back to Matplotlib only for EEG topomaps.

## Keeping This Documentation Up to Date

This README is the authoritative reference for the analysis package. **When you make changes to
this package, update this file accordingly.** Specifically:

- **New module added** → add a row to the Module Overview table and a dedicated section.
- **New function in `isc.py`** → add a row to the ISC function table.
- **New function in `mean_variance.py`** → add a row to the mean_variance function table.
- **New method in `EEGSummarizedAnalyzer`** → add a row to the Key methods table and update the typical workflow if the usage pattern changes.
- **New adapter in `data_representations.py`** → add a row to the Adapter functions table and update the `DataRepresentation` enum table if a new representation is introduced.
- **Data shape convention changes** → update the shape table and any function signatures that reference it.
- **Renamed or removed symbols** → remove or correct the corresponding rows/examples.

If you discover that any part of this README is inaccurate or out of date, correct it as part of
the same commit that changes the code — do not leave stale documentation behind.

---

## Running the Analysis

```bash
# ISC analysis (broadband + per-band + mean-field, Placebo condition, both music types)
python scripts/run_isc.py --music_type CLASSIC PSYTRANCE

# ISC with custom multi-scale window sizes and channel subsampling for Spearman
python scripts/run_isc.py --music_type CLASSIC \
    --window_fine_sec 1.0 --window_sec 5.0 --window_large_sec 15.0 \
    --n_ch_subsample 64

# Mean-variance analysis (dedicated script)
python scripts/run_mean_variance.py --music_type CLASSIC PSYTRANCE

# Mean-variance with custom window and sync percentile
python scripts/run_mean_variance.py --music_type PSYTRANCE \
    --window_sec 3.0 --sync_percentile 10

# Wavelet power analysis
python scripts/run_analysis.py --analysis wavelet_power \
    --wavelet_data_dir data/processed/psilo_music/wavelets
```
