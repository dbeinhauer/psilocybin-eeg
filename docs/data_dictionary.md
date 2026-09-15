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
| `JOINED` | `"Joined"` |
| `JOINED_TRACKS` | `"JoinedTracks"` |

`JOINED` and `JOINED_TRACKS` are **virtual** conditions — no recording ever carries
either. Both select both real conditions at once, restricted to participants that
contribute a recording to both, so both are balanced within-subject designs and both
select exactly the same recordings. Use `REAL_CONDITIONS` (a tuple of `PLACEBO` and
`PSILOCYBIN`) wherever code must map a *recording* to a condition, and
`JOINED_CONDITIONS` for the two virtual ones.

They differ only in the axis the conditions are pooled along, and each is usable
anywhere a single condition is (`--condition Joined`, `--condition JoinedTracks`).

Neither needs any extra preprocessing — both read the same `RAW_CROPPED` as every
single-condition run.

- `JOINED` products are named `Joined_<MusicType>` (e.g. `Joined_ASSR.npy`).
- `JOINED_TRACKS` products are named `JoinedTracks_<MusicType>`, with an extra
  `.segment_boundaries.npy` sidecar recording where each condition's time segment
  starts and ends. Its wavelets are **assembled in memory** from the two per-condition
  caches rather than cached separately — the caches already share a time base, so the
  join is a selection plus a concatenation, not worth another ~85 GB on disk. Those
  per-condition caches must exist first: build them by running the same workflow with
  `--condition Placebo` and `--condition Psilocybin`.

For `JOINED`, the subject axis is ordered as one contiguous block per condition — all
Placebo, then all Psilocybin, participants in the same order within each block — so
subject `k` and subject `k + n_pairs` are the same participant. Use `subject_conditions`,
`condition_index_mask`, `participant_condition_labels` and `paired_subject_index` from
`scripts/analysis_common.py` to recover the split rather than slicing by position.

`JOINED` **wavelets** are assembled in memory too, for the same reason as
`JOINED_TRACKS`: the two per-condition caches already share the aligned time base, so
stacking them on the subject axis is a selection plus a concatenation rather than
another ~100 GB on disk. `scripts/notebook_helpers.load_joined_condition_wavelets`
loads each condition from its existing cache (one at a time, so the memory peak stays
at one condition) and hands both to
`src.analysis.condition_tracks.pool_condition_subjects`, which returns a
`PooledConditionSubjects` carrying the per-subject participant/condition bookkeeping —
`condition_mask`, `condition_subjects`, `partner_index`, `subject_labels` and
`select_participants`, the subject-axis counterparts of the `analysis_common` helpers
above, working off the pooled object instead of an analyser's metadata. Asking
`compute_wavelet_datasets` for a `Joined_*` label directly would instead write that
second copy of the cache.

Either join still has to read the two per-condition caches once, and that read — tens
of GB decompressed per condition — is the dominant cost. Pass `subset_cache_dir` (see
`resolve_notebook_wavelet_cache_dir` and
[`notebooks/03-wavelet-analysis/README.md`](../notebooks/03-wavelet-analysis/README.md))
to have the trimmed tensor stored per extent under the stage-03 notebook, so later runs
skip that read entirely. The entries are keyed by condition and extent rather than by
workflow, so they are shared with every single-condition notebook.

### Two ways to join the conditions

`JOINED` is one of two variants, and they are not interchangeable:

| | `ConditionVariants.JOINED` | `ConditionVariants.JOINED_TRACKS` |
|---|---|---|
| Pools along | subject axis | **time** axis |
| Subject axis | `2 × n_pairs` (each participant twice) | `n_pairs` (each participant once) |
| Time axis | one common aligned base | Placebo track then Psilocybin track |
| Upstream work | none | none |
| IVA components | per-recording patterns; the sources (SCVs) are aligned across both conditions | one set shared by both conditions by construction |
| Recover a condition | `condition_index_mask` (subject axis) | `PairedConditionTracks.condition_track` (time axis) |

The track variant lives in `src/analysis/condition_tracks.py`, and each variant has its
own CLI entry point and Metacentrum job: `scripts/run_iva_condition_comparison.py` /
`jobs/metacentrum/06-iva-condition-comparison/run_condition_comparison.pbs` for the
subject axis, `scripts/run_iva_condition_tracks.py` / `run_condition_tracks.pbs` for the
time axis. Both share every numerical step through
`src/analysis/iva_condition_comparison.py`, so the two variants cannot drift apart. Both variants select the
same participants — those present and not excluded under both conditions — and both are
selected the same way, by naming the condition. `PairedConditionTracks.from_analyzer`
rebuilds the per-condition split from the `.segment_boundaries.npy` sidecar, so
`condition_track` works even in a bare CLI run that never called the paired loader.

Why the standard workflow needs no changes for `JOINED_TRACKS`: its cached wavelets are
stored **already per-condition z-scored**. Both segments therefore have mean 0 and unit
variance per `(subject, channel, frequency)`, so the pooled series does too — making the
usual `zscore_by_time` first step an exact identity rather than a re-normalisation that
would undo the per-condition standardisation.

Its `zscore_mode` decides what a between-condition difference means. The default
`"per_condition"` standardises Placebo and Psilocybin separately and concatenates the
standardised tracks, so each enters the decomposition exactly as it would in a
single-condition workflow; the comparison is then about temporal and spectral
*structure*, since an overall power difference is normalised away. `"joint"` z-scores
over the whole concatenated recording instead, keeping such a difference as a mean
offset between the segments. `"none"` assumes the caller already standardised.

### MusicTypeVariants

| Member | Value |
|--------|-------|
| `CLASSICAL` | `"CLASSIC"` |
| `PSYTRANCE` | `"PSYTRANCE"` |
| `ASSR` | `"ASSR"` |

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
| `CONCATENATED_PERSON_INDEX` | `"concatenated_person_index"` | Index on axis 0 of the concatenated NumPy array |

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
| `ARTIFACTS` | `"artifacts"` | Too many artifacts in the data (also after preprocessing) |

### FrequencyBandNames

| Member | Value | Range (Hz) |
|--------|-------|------------|
| `DELTA` | `"delta"` | 1–4 |
| `THETA` | `"theta"` | 4–8 |
| `ALPHA` | `"alpha"` | 8–13 |
| `BETA` | `"beta"` | 13–30 |
| `GAMMA` | `"gamma"` | 30–70 |

### AnalysisVariants

Analysis keywords accepted by `scripts/run_analysis.py`:

| Member | Value |
|--------|-------|
| `ISC` | `"isc"` |
| `MEAN_VARIANCE` | `"mean_variance"` |
| `WAVELET_POWER` | `"wavelet_power"` |
| `WAVELET_PHASE` | `"wavelet_phase"` |

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

The alignment producing `RAW_CROPPED` is fitted **once**, over every recording of both
conditions, so all conditions share one time base and carry the same stimuli. That is
what makes condition selection and custom participant subsetting pure filtering: the
time axis never changes, results stay comparable across analyses, and no wavelet is
ever recomputed for a different selection.

The cost is small and paid once: because each inter-stimulus interval is trimmed to the
group minimum, widening the group to all 38 ASSR recordings gives a common stimulus
count of 148 (vs 148 Placebo-only, 149 Psilocybin-only) and a time axis roughly 0.3–0.5%
shorter.

Exclusions applied at alignment time are deliberately minimal, because a recording
dropped there can never be selected later — it will not have been aligned. For music
that means `BAD_MUSIC` only (a wrong TAG channel would corrupt the cross-correlation for
the whole group); for ASSR, only `WRONG_CONDITION`.

### IvaVariants

IVA decomposition variants. The value is used verbatim as the analysis-type
subdirectory under `plots/` **and** as the first filename token in the component store
(`src/io/iva_store.py`), so the figures and the arrays of one run are named the same.

| Member | Value | Mixing (independent) axis | Sample axis | Per-component products |
|--------|-------|---------------------------|-------------|------------------------|
| `CHANNEL` | `"iva_channel"` | channels | time × frequency | TF map `(F, T)` + channel topography `(C,)` |
| `FREQUENCY_CHANNEL` | `"iva_frequency_channel"` | channel × frequency | time | timecourse `(T,)` + spectro-spatial pattern `(F, C)` |
| `TIME` | `"iva_time"` | time | channel × frequency | temporal pattern `(T,)` + spectro-spatial score map `(F, C)` |
| `CHANNEL_JOINED` | `"iva_channel_joined"` | channels | time × frequency | as `CHANNEL`, on the subject-axis join |
| `CHANNEL_JOINED_TRACKS` | `"iva_channel_joined_tracks"` | channels | time × frequency | as `CHANNEL`, on the time-axis join (one shared topography per participant) |

Only the channel-mixing variants have a per-component time-frequency map: the other two
put frequency on the mixing axis, so what a component owns there is a `(F, C)` pattern.
A reader must therefore ask for a named array rather than assume one is present.

### IvaComponentArrays

Canonical names of the arrays kept in the IVA component store. Every one is indexed
`(recording, component, ...)`, so the per-row participant/condition bookkeeping applies
unchanged to all of them.

| Member | Value | Shape | Description |
|--------|-------|-------|-------------|
| `TF_MAP` | `"tf_map"` | `(S, K, F, T)` | Per-recording time-frequency source map |
| `CHANNEL_PATTERN` | `"channel_pattern"` | `(S, K, C)` | Forward (mixing) channel topography |
| `TIMECOURSE` | `"timecourse"` | `(S, K, T)` | Temporal profile of the component |
| `SPECTRAL_PROFILE` | `"spectral_profile"` | `(S, K, F)` | Spectral profile of the component |
| `FREQUENCY_CHANNEL_PATTERN` | `"frequency_channel_pattern"` | `(S, K, F, C)` | Spectro-spatial pattern |

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

## IVA Component Store (`data/processed/{experiment}/iva_results/`)

Written by the IVA CLI scripts under `--store_components` and read through
`src/io/iva_store.py`. Layout:

```
data/processed/{experiment}/iva_results/
└── {Condition}/                                        # Placebo | Psilocybin | Joined | JoinedTracks
    └── {variant}__{MusicType}__{spectrum}__pca{N}.npz
```

`{variant}` is an `IvaVariants` value, `{spectrum}` is `broadband` or a band name, and
`{N}` is the per-recording PCA dimension. The condition is a directory so a whole
condition can be listed, copied or dropped as a unit; everything that distinguishes two
runs *within* a condition is in the filename, so a `--n_pca` or band sweep never
overwrites a sibling. A rerun of the same tuple replaces its own entry.

Each `.npz` holds:

| Key | Description |
|-----|-------------|
| `array__{name}` | One `IvaComponentArrays` entry, `(S, K, ...)`, `float32` by default |
| `participants` | Participant label per row — **the mapping**; without it the subject axis is unreadable |
| `subject_conditions` | Condition per row (a participant owns one row per condition in a `Joined` run) |
| `freqs`, `times`, `channel_names` | Coordinate axes of the stored arrays; each present only when some array has that axis |
| `segment_conditions`, `segment_lengths` | Time-axis segments of a `JoinedTracks` run, so a condition is recoverable as a slice |
| `extra__stimulus_onsets_{Condition}` | Onset samples per condition, trimmed to the stored axis, so an onset-locked read-out needs no re-derivation. Frame matches the arrays: shared axis for `Joined`, segment-local for `JoinedTracks` |
| `variant`, `experiment`, `condition`, `music_type`, `band`, `n_pca`, `sfreq` | The run descriptor, i.e. the filename's contents in readable form |
| `extra__{name}` | Per-run diagnostics (component ranking, PCA explained variance, sign alignment) |
| `format_version` | On-disk format version; a newer file is refused rather than misread |

`load_iva_components` returns an `IvaComponentResults`, which resolves rows by
participant and condition (`row`, `rows`, `for_participant`, `select_participants`)
rather than by position, and reports an ambiguous label instead of silently returning
its first match. It also rebuilds what the arrays need to be interpreted again:
`topo_info()` gives an MNE `Info` with the project montage applied (why the channel
names are stored at all — a channel pattern is only a topography once its values sit
on a scalp), `stimulus_onsets(condition)` / `onset_conditions` read the onsets back,
and `condition_track()` / `times_for()` split a `JoinedTracks` time axis.

The notebooks under `notebooks/06-iva-condition-comparison/`
(`iva_component_analysis_joined.ipynb` and
`iva_component_analysis_joined_tracks.ipynb`) are the worked examples: they read a
stored entry and do the condition comparison, the marginals, a paired
within-participant contrast and the onset-locked view without decomposing anything.

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
