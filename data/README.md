# Data Directory

This directory contains all data artefacts for the psilocybin-EEG project.
Most subdirectories are **gitignored** because the data files are large and/or
sensitive. The expected layout is documented below so that collaborators can
reproduce it locally.

## Expected Layout

```
data/
├── README.md                # This file
├── raw/                     # Original EDF files (untouched)
│   └── psilo_music/         # One subdirectory per experiment
│       ├── PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218.edf
│       ├── PSI018_EEGB_MUSIC_PSYTRANCE_EC_20171124_014218.edf
│       └── ...
├── interim/                 # Intermediate products (before ICA, etc.)
│   └── psilo_music/
│       ├── before_ica/      # Raw data state before ICA
│       ├── ica_components/  # Saved ICA decomposition objects (.fif)
│       └── ic_probabilities/# ICLabel probability arrays (.npy)
├── processed/               # Final preprocessed data
│   └── psilo_music/
│       ├── after_ica/       # Main preprocessed output (.fif)
│       ├── cropped/         # Time-aligned, cropped data (.fif)
│       └── raw_excluded_ic/ # Metadata CSV of excluded ICs
└── excluded_participants/   # Exclusion CSVs with explanations
    └── psilo_music.csv
```

## File Naming Convention

Raw EDF files follow the pattern:

```
PSI{participant_id}_EEG{condition_id}_MUSIC_{music_type}_EC_{date}_{time}.edf
```

- `participant_id`: 3-digit code (e.g. `018`)
- `condition_id`: single letter `A` or `B` (mapped to Placebo/Psilocybin via
  participant mapping CSV in `config/participant_mappings/`)
- `music_type`: `CLASSIC` or `PSYTRANCE`

Processed files keep the original base filename with a `.fif` or `.npy`
extension, stored in a subdirectory named after the processing variant
(e.g. `after_ica/`, `cropped/`).

## Participant Exclusion

The `excluded_participants/` directory contains CSV files (semicolon-delimited)
listing participants excluded from analysis, along with the reason. Exclusion
categories are defined in `src.definitions.fields.ExclusionCategories`:

| Category | Description |
|----------|-------------|
| `bad_music` | Wrong TAG channel signal |
| `bad_power_spectrum` | Abnormal power spectrum (bad data quality) |
| `missing_trials` | Some of the trials are missing for the participant |
