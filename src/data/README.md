# EEG Data Preprocessing Pipeline

This directory contains the core preprocessing modules for the psilocybin-EEG dataset. The pipeline transforms raw EDF recordings into cleaned, time-aligned data suitable for group-level Inter-Subject Correlation (ISC) analysis.

## Module Overview

| Module | Class | Purpose |
|--------|-------|---------|
| `dataset_handler.py` | `DatasetHandler` | Main orchestrator — ties together parsing, preprocessing, filtering, alignment, and plotting |
| `dataset_parsing.py` | `DatasetParser` | Extracts metadata (participant ID, condition, music type) from EDF filenames |
| `dataset_preprocessing.py` | `DatasetPreprocessor` | Signal processing: channel setup, filtering, ICA, artifact removal |
| `dataset_filtering.py` | `DatasetFilter` | Filters metadata DataFrames by condition, music type, and exclusion categories |
| `dataset_plotting.py` | `DatasetPlotter` | Visualization utilities (topomaps, power spectra, correlation heatmaps) |
| `time_aligner.py` | `TimeAligner` | Cross-correlation–based alignment of recordings using the TAG (stimulus) channel |

## Input Data Format

- **File format:** EDF (`.edf`)
- **Filename convention:** `PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{date}.edf`
  - `participant_id`: 3-digit code (e.g. `018`)
  - `condition_id`: single letter `A` or `B` (mapped to Placebo/Psilocybin via participant mapping CSV)
  - `music_type`: `CLASSIC` or `PSYTRANCE`
- **Electrode cap:** GSN-HydroCel-257 (256 + reference)
- **Participant mapping CSV** (`data/participant_mappings/{experiment}.csv`, semicolon-delimited) maps each `(participant, EEG condition)` pair to the actual experimental condition.

## Preprocessing Pipeline

### Stage 1 — Initial Preprocessing & Bad Channel Interpolation

Implemented in `DatasetPreprocessor.initial_preprocessing_and_bad_channel_interpolation()`.

1. **Channel preparation**
   - Rename channels to match montage conventions (`EEG {n}` → `E{n}`, `EEG VREF` → `Cz`).
   - Set channel types (EEG, ECG, TAG/stimulus).
   - Apply electrode coordinate montage from the SFP file.
   - Drop boundary electrodes listed in `data/excluded_electrodes/`.

2. **Temporal cropping**
   - Remove the first and last **10 seconds** (typically noisy transition periods).

3. **Frequency filtering**
   - Notch filter at 50 Hz harmonics (50–250 Hz, step 50 Hz) to suppress line noise.
   - FIR bandpass filter: **1–100 Hz**.

4. **Bad channel detection & interpolation**
   - RANSAC algorithm (2-second epochs, 100 resamples, `min_corr=0.7`) identifies bad channels.
   - Spherical-spline interpolation repairs them.
   - Average reference is applied.

5. **Bad epoch annotation**
   - AutoReject marks noisy 2-second epochs as `BAD_epoch` in annotations (data is preserved, not removed).

### Stage 2 — ICA Decomposition & Artifact Removal

Implemented in `DatasetPreprocessor.apply_ica_component_filtering()`.

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

Implemented in `TimeAligner`.

1. **Extract TAG signals** from all preprocessed recordings for a given music type and condition.
2. **Cross-correlate** all signal pairs to find optimal temporal shifts.
3. **Select reference signal** (highest mean correlation to all others).
4. **Determine common time window** — the overlap region across all shifted signals.
5. **Crop** every participant's raw data to this window and save as the `cropped` variant.

## Output Structure

```
data/processed/{experiment_name}/
├── before_ica/           # Raw data state before ICA (optional)
├── after_ica/            # Main preprocessed output (.fif)
├── ica_components/       # Saved ICA decomposition objects
├── ic_probabilities/     # ICLabel probability arrays (.npy)
├── raw_excluded_ic/      # Metadata CSV of excluded ICs (label, ID, probability, etc.)
├── cropped/              # Time-aligned, cropped data
└── excluded_ics_mapping.csv
```

Output files keep the original base filename with a `.fif` extension.

## Running the Pipeline

```bash
# Full preprocessing (ICA + optional IC metadata + optional plots)
python scripts/run_dataset_preprocessing.py --raw_processing --process_excluded_ic --plot_results

# Time alignment for a specific condition and music type
python scripts/run_time_aligner.py --condition Placebo --music_type CLASSIC
```

## Exclusion Mechanisms

| Mechanism | Source | Applied In |
|-----------|--------|------------|
| Boundary electrodes | `data/excluded_electrodes/*.csv` | Stage 1 (channel preparation) |
| Bad channels (RANSAC) | Automatic detection | Stage 1 (interpolation) |
| Bad epochs (AutoReject) | Automatic detection | Stage 1 (annotated, excluded from ICA fitting) |
| Artifact ICs (ICLabel) | Automatic classification | Stage 2 (removed from signal) |
| Participant-level exclusion | `data/excluded_participants/*.csv` | Filtering before analysis (categories: `bad_music`, `bad_power_spectrum`, `missing_trials`) |
