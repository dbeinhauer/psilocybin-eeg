# EEG Preprocessing Pipeline

This package contains the core preprocessing modules for the psilocybin-EEG dataset. The pipeline transforms raw EDF recordings into cleaned, time-aligned data suitable for group-level Inter-Subject Correlation (ISC) analysis.

## Module Overview

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Top-level orchestration — ties together parsing, preprocessing, filtering, alignment, and plotting (`DatasetHandler` and `DatasetPreprocessor` classes) |
| `channel_prep.py` | Channel renaming, montage application, type setting, electrode exclusion |
| `filtering.py` | Notch & bandpass filtering, RANSAC bad channel detection, bad epoch annotation |
| `ica.py` | ICA decomposition, ICLabel classification, artifact component exclusion |
| `time_alignment.py` | Cross-correlation–based alignment of recordings using the TAG (stimulus) channel |

## Preprocessing Pipeline

### Stage 1 — Initial Preprocessing & Bad Channel Interpolation

Orchestrated by `DatasetPreprocessor.initial_preprocessing_and_bad_channel_interpolation()`.

1. **Channel preparation** (`channel_prep.py`)
   - Rename channels to match montage conventions (`EEG {n}` → `E{n}`, `EEG VREF` → `Cz`).
   - Set channel types (EEG, ECG, TAG/stimulus).
   - Apply electrode coordinate montage from the SFP file.
   - Drop boundary electrodes listed in `config/excluded_electrodes/`.

2. **Temporal cropping** (`channel_prep.py`)
   - Remove the first and last **10 seconds** (typically noisy transition periods).

3. **Frequency filtering** (`filtering.py`)
   - Notch filter at 50 Hz harmonics (50–250 Hz, step 50 Hz) to suppress line noise.
   - FIR bandpass filter: **1–100 Hz**.

4. **Bad channel detection & interpolation** (`filtering.py`)
   - RANSAC algorithm (2-second epochs, 100 resamples, `min_corr=0.7`) identifies bad channels.
   - Spherical-spline interpolation repairs them.
   - Average reference is applied.

5. **Bad epoch annotation** (`filtering.py`)
   - AutoReject marks noisy 2-second epochs as `BAD_epoch` in annotations (data is preserved, not removed).

### Stage 2 — ICA Decomposition & Artifact Removal

Orchestrated by `DatasetPreprocessor.apply_ica_component_filtering()`, delegating to `ica.py`.

1. **ICA decomposition**
   - Method: Extended Infomax (`n_components=0.99`, `random_state=97`).
   - Bad-epoch annotations from Stage 1 are respected (`reject_by_annotation=True`).

2. **IC component labeling** (ICLabel)
   - Each component receives probabilities across seven classes: Brain, Muscle, Eye, Heart, Line noise, Channel noise, Other.

3. **Artifact component exclusion**
   - A component is excluded when its artifact probability exceeds the class threshold **and** its brain probability is below `0.3`:

   | Class | Threshold |
   |-------|-----------|
   | Eye | ≥ 0.40 |
   | Muscle | ≥ 0.60 |
   | Heart | ≥ 0.40 |
   | Channel noise | ≥ 0.50 |

4. **Reconstruct signal**
   - ICA is applied to remove the excluded components.
   - Auxiliary channels (ECG, TAG) are re-attached.

### Stage 3 — Time Alignment (for group analysis)

Implemented in `time_alignment.py`.

1. **Extract TAG signals** from all preprocessed recordings for a given music type and condition.
2. **Cross-correlate** all signal pairs to find optimal temporal shifts.
3. **Select reference signal** (highest mean correlation to all others).
4. **Determine common time window** — the overlap region across all shifted signals.
5. **Crop** every participant's raw data to this window and save as the `cropped` variant.

## Running the Pipeline

```bash
# Full preprocessing (ICA + optional IC metadata + optional plots)
python scripts/run_preprocessing.py --raw_processing --process_excluded_ic --plot_results

# Time alignment for a specific condition and music type
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC
```
