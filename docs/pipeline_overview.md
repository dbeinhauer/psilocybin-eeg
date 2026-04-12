# Pipeline Overview

This document provides an end-to-end description of the psilocybin-EEG analysis workflow,
from raw EDF recordings to group-level ISC (Inter-Subject Correlation) results.

## 1. Raw Data Acquisition

Participants listen to music (Classical or Psytrance) under two conditions
(Placebo and Psilocybin). EEG is recorded using a 257-channel GSN-HydroCel cap
in EDF format. A dedicated TAG (stimulus) channel records the music signal for
time alignment.

### File naming

```
PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{date}_{time}.edf
```

## 2. Configuration

Static configuration files are stored in `config/`:

| Directory | Contents |
|-----------|----------|
| `config/coordinates/` | Electrode coordinate files (`.sfp`) for montage setup |
| `config/excluded_electrodes/` | CSVs listing boundary electrodes to drop |
| `config/participant_mappings/` | CSVs mapping condition IDs (A/B) to labels (Placebo/Psilocybin) |

## 3. Preprocessing (`src/preprocessing/`)

### Stage 1 — Channel Preparation & Filtering

1. Rename channels to montage conventions (`EEG {n}` → `E{n}`)
2. Set channel types (EEG, ECG, TAG/stimulus)
3. Apply electrode coordinate montage
4. Drop boundary electrodes
5. Crop noisy first/last 10 seconds
6. Notch filter (50 Hz harmonics) + FIR bandpass (1–100 Hz)
7. RANSAC bad channel detection → interpolation → average reference
8. AutoReject bad epoch annotation

### Stage 2 — ICA Artifact Removal

1. Extended Infomax ICA decomposition
2. ICLabel classification (Brain, Muscle, Eye, Heart, Line/Channel noise, Other)
3. Threshold-based artifact component exclusion
4. Signal reconstruction without artifact components

### Stage 3 — Time Alignment

1. Extract TAG signals from all preprocessed recordings
2. Pairwise cross-correlation to find optimal temporal shifts
3. Select reference signal (highest mean correlation)
4. Determine common time window and crop all recordings

## 4. Data I/O (`src/io/`)

| Module | Purpose |
|--------|---------|
| `parsing.py` | Extract metadata from EDF filenames |
| `loading.py` | Load raw or processed EEG files |
| `saving.py` | Save processed outputs |

## 5. Metadata Filtering (`src/filtering/`)

The `DatasetFilter` class provides static methods to filter metadata DataFrames
by condition (Placebo/Psilocybin), music type (Classical/Psytrance), participant
exclusion categories, and participant IDs.

## 6. Analysis (`src/analysis/`)

| Module | Purpose |
|--------|---------|
| `isc.py` | Pure ISC computation functions (LOO, pairwise, sliding window) |
| `mean_variance.py` | Intersubject mean-variance synchrony analysis |
| `data_representations.py` | `AnalysisData` container and adapter functions (time-domain, wavelet, ICA activations, mean response) |
| `results_store.py` | CSV export for analysis results (consumed by the Interactive Explorer) |
| `summary.py` | High-level `EEGSummarizedAnalyzer` orchestrator for loading data and computing ISC metrics |

### ISC Methods

- **Leave-One-Out (LOO) ISC**: Correlate each subject with the mean of all others
- **Pairwise ISC**: Correlate every subject pair
- **Sliding Window ISC**: Time-resolved LOO-ISC across the recording
- **Band-specific ISC**: Apply to standard frequency bands (delta, theta, alpha, beta, gamma)

## 7. Visualization (`src/visualization/`)

| Module | Purpose |
|--------|---------|
| `preprocessing_plots.py` | Topomaps, power spectra, signal overlap, cross-correlation plots |
| `isc_plots.py` | ISC heatmaps, distributions, sliding window plots, band comparisons |
| `mean_variance_plots.py` | Mean-variance synchrony plots |

## 8. CLI Entry Points (`scripts/`)

```bash
# Full preprocessing
python scripts/run_preprocessing.py --raw_processing --process_excluded_ic --plot_results

# Time alignment
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC

# ISC analysis (broadband, per-band, mean-field)
python scripts/run_isc.py --music_type CLASSIC PSYTRANCE

# Mean-variance synchrony analysis
python scripts/run_mean_variance.py --music_type CLASSIC PSYTRANCE
```

`scripts/analysis_common.py` provides shared helpers (`load_analyzers`, `analyzers_to_datasets`)
used by both `run_isc.py` and `run_mean_variance.py`.

## 9. HPC Job Submission (`jobs/`)

PBS job scripts for the Metacentrum cluster are in `jobs/metacentrum/`.
See `docs/hpc_guide.md` for details.
