# 00 — Preprocessing

## Overview

This directory contains notebooks for **inspecting and verifying** the EEG data
before and during the preprocessing pipeline
(**channel_prep → filtering → ica → time_alignment**). They are not part of the
production pipeline itself — they are used to catch data-collection problems,
verify alignment, and visually inspect recordings at each stage.

Two experiments are covered: the main **music-listening** dataset (257-channel
EGI, Placebo/Psilocybin × CLASSIC/PSYTRANCE) and the **ASSR** dataset (annotated
`.edf` recordings with `fam+` stimulus markers).

## Notebooks

| Notebook | Purpose |
|---|---|
| `raw_dataset_inspection.ipynb` | Metadata-level integrity check of a raw dataset *before* preprocessing |
| `eeg_data_inspection.ipynb` | Interactive inspection of a single recording across preprocessing stages |
| `time_alignment.ipynb` | Run and verify cross-correlation time alignment (music dataset, TAG channel) |
| `stimulus_alignment.ipynb` | Run and verify two-stage stimulus alignment (ASSR dataset, `fam+` markers) |
| `assr_data_inspection.ipynb` | Inspect ASSR annotation labels, counts, and stimulus-onset timing |

## What each notebook does

### `raw_dataset_inspection.ipynb` — Integrity & completeness

Inspects the metadata of a selected raw dataset before any preprocessing, to
catch collection problems early. Reports dataset size and per-participant
recording counts, condition/music-type balance, a **presence matrix** over every
expected `participant × condition × music_type` cell, and flags **duplicate**,
**missing**, and **unexpected** trials plus recordings on the excluded list.

> Parameter: `EXPERIMENT`.

### `eeg_data_inspection.ipynb` — Stage-by-stage signal inspection

Interactive tool to inspect a single recording at different pipeline stages
(before ICA, after ICA, cropped). Browse metadata, view the raw time series
(MNE viewer), spectral power, and per-band/broadband topomaps; compare
before/after ICA; inspect excluded ICA components, their label probabilities,
and topographies; and visualise the sensor layout.

> Parameters: `PARTICIPANT_ID`, `MUSIC_TYPE`, `CONDITION`, `DATA_STAGE`.

### `time_alignment.ipynb` — TAG-channel time alignment (music dataset)

Runs and verifies cross-correlation-based time alignment across participants
using the TAG (stimulus marker) channel. Plots per-participant cross-correlation
curves, overlays aligned TAG signals, compares aligned vs. unaligned, and shows
the pairwise correlation heatmap. Optionally crops and saves the aligned
(`cropped`) recordings. Includes a per-participant Global Field Power overlay
that flags amplitude outliers for possible exclusion.

> Parameters: `CONDITION`, `MUSIC_TYPE`, `EXCLUSION_CATEGORIES`,
> `PLOT_WINDOW_SEC`, `CROSSCORR_MAX_LAG_SEC`, `REUSE_ALIGNMENT`.

### `stimulus_alignment.ipynb` — Stimulus alignment (ASSR dataset)

Runs the **fine** alignment stage on preprocessed (`after_ica`) ASSR data so
that stimulus *k* lands on the same sample index in every recording (the coarse
crop happens during preprocessing). Reports per-recording summaries and
inter-stimulus intervals, shows onset alignment before vs. after, and runs
alignment-quality checks (onset-overlap raster, lag-0 inter-subject
cross-correlation). Can reuse cached `cropped` recordings instead of
recomputing.

> Requires preprocessing to have run for the experiment. Parameters include
> `REUSE_ALIGNMENT`, `SAVE_ALIGNMENT`.

### `assr_data_inspection.ipynb` — ASSR annotations

Inspects the annotations of every recording in the ASSR raw dataset (read from
EDF headers, no full signal load). Reports which annotation labels appear and
their counts per recording, and analyses stimulus-onset timing — consecutive
`fam+` intervals, the `bgin → fam+` within-pair gap, and special cases such as
orphan `bgin` markers not followed by a `fam+`.

> Parameters in the *Configuration* cell.
