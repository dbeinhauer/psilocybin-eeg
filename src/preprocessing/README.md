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
| `stimulus_alignment.py` | Annotation-based alignment of stimulus onsets across participants; resolves each recording's marker→onset offset (`resolve_stimulus_marker`) |
| `marker_shift.py` | Measures the per-recording stimulus-marker timing error against the `.evt` exports and persists it as a calibration mapping |

## Stimulus-Marker Calibration (ASSR)

The `fam+` annotations read from the raw EDFs are **late** with respect to the sound
they mark, by a different amount in every recording (measured range 372–455 ms). The
EDF header can only record the recording start to the nearest whole second, so the
sub-second remainder is lost and re-appears as a per-recording offset on every
annotation. The diagnosis is in
[`notebooks/00-preprocessing/assr_annotation_discrepancy.ipynb`](../../notebooks/00-preprocessing/assr_annotation_discrepancy.ipynb).

The remainder is recoverable from the recording-native `.evt` exports in
`data/events/`, so it is calibrated once and reused:

```bash
# Measure every recording -> config/participant_mappings/assr_time_shift.csv
python scripts/run_stimulus_shift_calibration.py --experiment assr

# Inspect what would be written, without touching the file
python scripts/run_stimulus_shift_calibration.py --experiment assr --dry_run
```

Everything that reads stimulus onsets goes through
`resolve_stimulus_marker(experiment_name)`, which returns the marker label together
with that calibration; `StimulusMarker.onset_offset_for(filename)` yields one
recording's offset and `.offsets_for(filenames)` a whole group's. Recordings absent
from the mapping fall back to `AssrEpoch.MARKER_ONSET_OFFSET_S` (the median of the
measured shifts) — a stopgap that still leaves them mis-timed by up to ~50 ms.

### Regenerating after a calibration change

Changing the calibration (or the fallback constant) invalidates **every** product
from the coarse crop onwards. Regenerate in this order:

```bash
python scripts/run_preprocessing.py --experiment assr --raw_processing   # before_ica / after_ica
python scripts/run_stimulus_alignment.py --condition Placebo             # -> RAW_CROPPED
python scripts/run_stimulus_alignment.py --condition Psilocybin
python scripts/run_analysis.py --analysis wavelet_power --experiment assr \
    --condition Placebo --process_and_save --n_jobs -1                   # concatenated + wavelets
```

> **Preprocessing really is included.** It is tempting to assume the offset only
> matters when annotations are read, but the stimulus-aware coarse crop
> (`coarse_crop_to_stimulus_span`) trims around the *offset-adjusted* onsets, so its
> bounds — and hence the lead-in every later stage inherits — depend on the offset.
> Reusing recordings cropped under a different offset silently shortens the
> pre-stimulus baseline, and because `StimulusAligner` takes `pre_target` as the
> group minimum, a single under-cropped recording shortens it for the whole group.
> No error is raised when it drops below `AssrEpoch.PRE_ONSET_S`.
>
> `--process_and_save` forces the concatenated rebuild; without it the stale cache is
> reused. Omit `--reuse_wavelets` so the wavelet cache is rebuilt too.

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
