# Data Dictionary

This document catalogues all field names, enum values, and CSV schemas used
in the psilocybin-EEG project.

## Enum Definitions (`src/definitions/fields.py`)

### ExperimentNames

| Member | Value | Description |
|--------|-------|-------------|
| `PSILO_MUSIC` | `"psilo_music"` | Experiment with placebo and psilocybin, with music listening |

### CoordinateSystems

| Member | Value |
|--------|-------|
| `HYDROGEL_257` | `"GSN-HydroCel-257"` |
| `HYDROGEL_257_NO_FIDUCIALS` | `"GSN-HydroCel-257_no-fiducials"` |

### ConditionVariants

| Member | Value |
|--------|-------|
| `PLACEBO` | `"Placebo"` |
| `PSILOCYBIN` | `"Psilocybin"` |

### MusicTypeVariants

| Member | Value |
|--------|-------|
| `CLASSICAL` | `"CLASSIC"` |
| `PSYTRANCE` | `"PSYTRANCE"` |

### EEGConditions

| Member | Value |
|--------|-------|
| `CONDITION_A` | `"A"` |
| `CONDITION_B` | `"B"` |

### SingleDataMetadata

Defines column names for per-recording metadata:

| Member | Value | Description |
|--------|-------|-------------|
| `PARTICIPANT_ID` | `"participant_id"` | 3-digit participant code |
| `EEG_CONDITION_ID` | `"eeg_condition_id"` | Raw condition id from filename (A or B) |
| `CONDITION` | `"condition"` | Placebo or Psilocybin |
| `MUSIC_TYPE` | `"music_type"` | CLASSIC or PSYTRANCE |
| `FILENAME` | `"filename"` | Exact filename of the data file |
| `EXCLUSION_EXPLANATION` | `"explanation"` | Reason for exclusion (if applicable) |

### ChannelTypes

| Member | Value | MNE Type |
|--------|-------|----------|
| `EEG` | `"eeg"` | `"eeg"` |
| `EEG_REF` | `"eeg_ref"` | `"eeg"` |
| `ECG` | `"ecg"` | `"ecg"` |
| `TAG` | `"tag"` | `"stim"` |

### ICLabelComponentsClasses

| Member | Value |
|--------|-------|
| `BRAIN` | `"brain"` |
| `MUSCLE` | `"muscle"` |
| `EYE` | `"eog"` |
| `HEART` | `"ecg"` |
| `LINE` | `"line_noise"` |
| `CHANNEL` | `"ch_noise"` |
| `OTHER` | `"other"` |

### ExclusionCategories

| Member | Value | Description |
|--------|-------|-------------|
| `BAD_MUSIC` | `"bad_music"` | Wrong TAG channel signal |
| `BAD_POWER_SPECTRUM` | `"bad_power_spectrum"` | Abnormal power spectrum |
| `MISSING_TRIALS` | `"missing_trials"` | Some trials are missing |

### PreprocessedDataVariants

| Member | Value | Description |
|--------|-------|-------------|
| `RAW_BEFORE_ICA` | `"before_ica"` | Raw data before ICA |
| `RAW_AFTER_ICA` | `"after_ica"` | Raw data after ICA |
| `ICA_COMPONENTS` | `"ica_components"` | Saved ICA decomposition |
| `IC_PROBABILITIES` | `"ic_probabilities"` | ICLabel probability arrays |
| `RAW_EXCLUDED_IC` | `"raw_excluded_ic"` | Excluded IC data series |
| `RAW_CROPPED` | `"cropped"` | Time-aligned cropped data |
| `CONCATENATED` | `"concatenated"` | Concatenated data across participants |

### ExcludedICsMetadata

Defines columns in the excluded ICs mapping CSV:

| Member | Value |
|--------|-------|
| `ORIGINAL_FILENAME` | `"original_filename"` |
| `TIMESERIES_FILENAME` | `"timeseries_filename"` |
| `IC_ID` | `"ic_id"` |
| `IC_CATEGORY` | `"ic_category"` |
| `TOTAL_ICS` | `"total_ics"` |
| `MAIN_PROBABILITY` | `"main_probability"` |

## Channel Name Mappings (`src/definitions/mappings.py`)

### RAW_CHANNEL_NAMES

Prefixes in raw EDF files:

| Channel Type | Prefix |
|-------------|--------|
| EEG | `"EEG"` |
| EEG Reference | `"EEG VREF"` |
| ECG | `"ECG"` |
| TAG | `"TAG"` |

### MONTAGE_CHANNEL_NAMES

Channel names used in the montage file:

| Channel Type | Name |
|-------------|------|
| EEG | `"E"` (+ number) |
| EEG Reference | `"Cz"` |

## CSV Schemas

### Participant Mapping CSV (`config/participant_mappings/{experiment}.csv`)

Semicolon-delimited with columns:

| Column | Example | Description |
|--------|---------|-------------|
| `participant` | `PSI018` | Participant ID with prefix |
| `eeg` | `EEGA` | EEG condition with prefix |
| `condition` | `Placebo` | Experimental condition label |

### Excluded Participants CSV (`data/excluded_participants/{experiment}.csv`)

Semicolon-delimited with columns matching `SingleDataMetadata` fields plus
`explanation` for the exclusion reason.

### Excluded Electrodes CSV (`config/excluded_electrodes/{coordinate_system}.csv`)

Single column `electrode_name` listing electrode names to exclude.

## Frequency Bands (`src/analysis/isc.py`)

| Band | Low (Hz) | High (Hz) |
|------|----------|-----------|
| delta | 1.0 | 4.0 |
| theta | 4.0 | 8.0 |
| alpha | 8.0 | 13.0 |
| beta | 13.0 | 30.0 |
| gamma | 30.0 | 70.0 |
