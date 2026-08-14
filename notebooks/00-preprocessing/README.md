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
| `assr_annotation_discrepancy.ipynb` | Diagnose why the MNE-read `fam+` markers disagree with the `.evt` event exports |
| `assr_stimulus_timing_verification.ipynb` | Acceptance test: verify from the data alone that the stimulus onsets of the aligned products are not shifted |

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

### `assr_annotation_discrepancy.ipynb` — Why the markers are late

Checks the `fam+` annotations MNE reads from the raw EDF against the
recording-native `.evt` exports in `data/events/` (microsecond event times, one
file per recording), and asks whether the known marker error is *systematic* or
*random*. Walks one recording event by event, then scans all 38: per-recording
shift, within-recording residual, `bgin` as a cross-check, the mechanism, and
whether any preprocessing stage adds to it.

**Finding.** Fully systematic and exactly recoverable. Within a recording the
markers are one constant shift late (residual ≤ 2 µs); across recordings the shift
is a per-recording constant of 372–455 ms that equals the offset between the two
files' time origins — the EDF header stores the recording start only to the nearest
whole second, and the `.evt` wall-clock anchor recovers the dropped remainder. This
is the exact source of the ≈ −0.4 s lag that `assr_stimulus_onset_offset.ipynb`
measured by ITC and froze into `AssrEpoch.MARKER_ONSET_OFFSET_S`; the global
constant leaves a per-subject residual spanning 83 ms (> 3 cycles at 40 Hz).

> Needs `data/events/*.evt`. Diagnostic only — changes nothing on disk.

### `assr_stimulus_timing_verification.ipynb` — Are the stored onsets right?

Acceptance test for the stimulus-aligned ASSR products (`concatenated/` +
`.stimulus_onsets.npy`, and the wavelet cache) after the per-recording marker
correction. Verifies **from the data alone** — ignoring the annotations — that
the driven response sits where the stored onsets claim, using two complementary
read-outs: the 40 Hz **inter-trial coherence envelope** (absolute, ~±50 ms) and
the 40 Hz **evoked phase clustering across recordings** (relative, ~±3 ms,
because a 25 ms cycle turns a timing error into a phase rotation). Covers a
global shift, every pair of recordings, the two halves of one recording's
stimulus sequence, and the two sessions of a participant; includes positive
controls that re-inject the pre-fix error to show the tests can fail.

**Finding.** The response spans **[+32, +532] ms** relative to the stored onsets
— the paradigm's 500 ms train, delayed by the auditory transmission latency
only. Recordings agree with one another to **≤ 5 ms** (a single global offset
would leave ~22 ms), the deviations no longer track the calibration residual,
and the placebo/psilocybin sessions of a participant differ by **+0.3 ms** on
average. Side finding: the wavelet cache and the concatenated array are built by
two independent alignment plans and are **not exactly sample-registered** — the
offset wanders over ~28 ms peak-to-peak along a recording. Harmless for wavelet
*power* (its 40 Hz kernel smears by ±80 ms), but it matters for any future
onset-locked analysis of wavelet *phase*.

> Runs on the products under test plus `RAW_AFTER_ICA` for the session-pair
> step. ~20 min end to end (streaming the 52 GB wavelet cache dominates); set
> `RUN_WAVELET_CHECK` / `RUN_SESSION_PAIR_CHECK` to `False` to skip the heavy
> steps. Diagnostic only — changes nothing on disk.
