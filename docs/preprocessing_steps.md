# Preprocessing Steps

This document describes the detailed step-by-step processing applied to the raw EEG data.
The pipeline is implemented in `src/preprocessing/` and driven by `scripts/run_preprocessing.py`.

The stages must be executed in order: **channel_prep → filtering → ica → time_alignment**

---

## Stage 1 — Channel Preparation (`channel_prep.py`)

1. **Load coordinates** — Read the GSN-HydroCel-257 montage from an SFP file (`config/coordinates/`).
2. **Rename channels** — Map raw EDF channel names (e.g., `"EEG 1"`) to montage names (e.g., `"E1"`); `"EEG VREF"` → `"Cz"`.
3. **Set channel types** — Assign MNE channel types: all montage channels → `eeg`, ECG channel → `ecg`, TAG channel → `misc`.
4. **Apply montage** — Attach the GSN-HydroCel-257 digitisation to the Raw object.
5. **Exclude boundary electrodes** — Drop electrodes listed in `config/excluded_electrodes/` that sit on the neck/hairline boundary and degrade signal quality.

---

## Stage 2 — Signal Filtering (`filtering.py`)

1. **Crop** — Trim the recording to the relevant time window before heavy computation.
2. **Notch filter** — Remove 50 Hz line noise and harmonics up to 250 Hz (`mne.io.Raw.notch_filter`).
3. **Bandpass FIR filter** — Apply 1–100 Hz zero-phase FIR filter to suppress slow drifts and high-frequency noise.
4. **RANSAC bad-channel detection** — Epoch the data into 2 s segments and run RANSAC (`autoreject.Ransac`) to identify consistently bad channels.
5. **Interpolate bad channels** — Reconstruct bad channels using spherical spline interpolation (`mne.io.Raw.interpolate_bads`).
6. **Average reference** — Re-reference all EEG channels to the common average.
7. **AutoReject** — Detect and repair or reject remaining bad epochs (`autoreject.AutoReject`).

---

## Stage 3 — ICA Artifact Removal (`ica.py`)

1. **Run Extended-Infomax ICA** — Decompose the filtered data into independent components (`mne.preprocessing.ICA`, method `"infomax"` with extended mode).
2. **ICLabel classification** — Apply `mne_icalabel.iclabel_label_components` to obtain per-component probability scores across seven classes: brain, muscle, eye, heart, line noise, channel noise, other.
3. **Component selection** — Mark components for exclusion when their artifact-class probability ≥ threshold (e.g., 0.40 for eye, 0.60 for muscle/heart/line) **and** brain probability < 0.30.
4. **Reconstruction** — Remove excluded components and reconstruct the EEG signal (`ICA.apply`).

---

## Stage 4 — Time Alignment (`time_alignment.py`)

1. **Extract TAG markers** — Read stimulus onset times from the TAG channel for each recording.
2. **Pairwise cross-correlation** — Compute cross-correlation between all pairs of recordings to find relative temporal offsets.
3. **Select reference recording** — Choose the recording that minimises the total shift across all pairs.
4. **Shift and crop** — Apply the computed time shifts and crop every recording to the common overlapping time window, ensuring all participants are synchronised.
