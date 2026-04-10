# Visualization Sketch — Psilocybin-EEG Analysis

> **Purpose of this document:** A living reference that maps every analysis in the project to its:
> - data shape and transformation chain,
> - logical order of operations,
> - real-life / neuroscientific interpretation,
> - notebook location and expected output file names,
> - ASCII sketch of what each plot looks like.
>
> Also discusses possible visualization delivery tools (static images, GitHub Pages, interactive dashboards, etc.).

---

## Table of Contents

1. [Pipeline Overview (End-to-End Data Flow)](#pipeline-overview)
2. [Stage 0 — Preprocessing](#stage-0--preprocessing)
3. [Stage 1 — Mean-Variance Synchrony Analysis](#stage-1--mean-variance-synchrony-analysis)
4. [Stage 2 — ISC (Inter-Subject Correlation)](#stage-2--isc-inter-subject-correlation)
5. [Visualization Delivery Tools](#visualization-delivery-tools)
6. [Plot Naming Conventions](#plot-naming-conventions)

---

## Pipeline Overview

```
Raw EDF files (257-ch, ~500 Hz)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — PREPROCESSING  (src/preprocessing/)                          │
│                                                                         │
│  channel_prep → filtering → ICA → time_alignment                       │
│                                                                         │
│  Input shape:  (n_channels=257, n_times_raw)  — per participant .edf   │
│  Output shape: (n_channels≈240, n_times_aligned)  — .fif / .npy       │
└─────────────────────────────────────────────────────────────────────────┘
        │
        ▼  load_analyzers() / EEGSummarizedAnalyzer
        │  stacks participants → (n_subjects, n_channels, n_times)
        │
        ├──────────────────────────────────────────────────────────────────
        │  STAGE 1 — MEAN-VARIANCE SYNCHRONY  (01-raw-mean-variance-analysis)
        │
        │  z-score across time  →  var across subjects  →  windowed stats
        │  Shape: (n_subjects, n_channels, n_times)  → (n_times,) variance trace
        │
        ├──────────────────────────────────────────────────────────────────
        │  STAGE 2 — ISC ANALYSIS  (02-isc-broadband-analysis)
        │
        │  [broadband]  (n_subjects, n_channels, n_times)
        │       → LOO-ISC  (n_subjects, n_channels)
        │       → pairwise  (n_subjects, n_subjects)
        │       → sliding-window  (n_windows, n_channels)
        │
        │  [per-band]  bandpass filter  ×5 bands
        │       → same ISC shapes, once per band
        │
        └──────────────────────────────────────────────────────────────────
```

---

## Stage 0 — Preprocessing

**Notebooks:** `notebooks/00-preprocessing/`

### 0-A · EEG Data Inspection
**File:** `notebooks/00-preprocessing/eeg_data_inspection.ipynb`

#### Purpose
Visually verify the EEG signal at each stage of the pipeline (raw → after channel prep → after filtering → after ICA → after time alignment). Sanity-check electrode positions, power spectra, and epoch quality.

#### Data Shapes

| Step | Shape | Notes |
|------|-------|-------|
| Raw EDF loaded | `(257, n_times_raw)` | All channels including non-EEG |
| After channel prep | `(~240, n_times_raw)` | Boundary electrodes dropped, montage set |
| After filtering | `(~240, n_times_raw)` | Notch (50/60 Hz), bandpass (1–70 Hz) |
| After ICA | `(~240, n_times_raw)` | Artefact components removed |
| After time alignment | `(~240, n_times_aligned)` | All participants share same time window |

#### Order of Operations
```
Load .edf  →  set montage  →  drop bad channels  →  notch filter
    →  bandpass filter  →  RANSAC / bad epoch detection
    →  ICA fit  →  ICLabel artefact classification  →  component removal
    →  TAG cross-correlation alignment  →  crop to common window
```

#### Plots

---

**Plot 0-A-1 · Raw signal overlay (all channels)**
- Notebook section: *Signal quality — raw vs after filtering*
- File: `plots/preprocessing/raw_signal_overlay.png`
- Library: **Matplotlib** (MNE `.plot()`)

```
μV
 │
 │  ┌���─ch1──────────────────────────────────────
+100│  ch2──────────────────────────────────────
 │  ┌──ch3──────────────────────────────────────
   0│  ...
-100│
 │
 └────────────────────────────────────────────► time (s)
```
*Interpretation: High-amplitude bursts = artefacts (eye blinks, muscle). After ICA, these should be absent.*

---

**Plot 0-A-2 · Power spectral density (PSD) — before vs after filtering**
- Notebook section: *Frequency inspection*
- File: `plots/preprocessing/psd_before_after.png`
- Library: **Matplotlib** (MNE `.plot_psd()`)

```
Power
(dB)
 │  ── before filtering
 │  ── after filtering
40│       │ 50 Hz notch
20│  ╲____|____/──────────────────────────────
 0│       bandpass cutoffs at 1 Hz and 70 Hz
 └──────────────────────────────────────────► Frequency (Hz)
      1        50  70
```
*Interpretation: Clean 1/f slope with no 50 Hz power-line artefact after filtering.*

---

**Plot 0-A-3 · EEG Topomap — mean power per channel**
- Notebook section: *Spatial quality check*
- File: `plots/preprocessing/topomap_mean_power.png`
- Library: **Matplotlib** (MNE `plot_topomap()`)

```
        ●●●●●
      ●●●●●●●●●
    ●●●● [hot] ●●●●
   ●●●● [cold] ●●●●
    ●●●● [hot] ●●●●
      ●●●●●●●●●
        ●●●●●
  [cold = low power, hot = high power]
```
*Interpretation: Uniform spatial distribution expected for good data; focal "hot spots" indicate bad channels.*

---

### 0-B · Time Alignment
**File:** `notebooks/00-preprocessing/time_alignment.ipynb`

#### Purpose
Verify that stimulus-onset markers (TAG channel) are correctly extracted and that the pairwise cross-correlation alignment produces a consistent shared time window across participants.

#### Data Shapes

| Step | Shape | Notes |
|------|-------|-------|
| TAG channel | `(1, n_times_raw)` | Stimulus marker channel |
| Cross-correlation | `(n_participants, n_participants)` | Pairwise lag matrix |
| Aligned data | `(~240, n_times_aligned)` | Per participant after crop |

#### Plots

---

**Plot 0-B-1 · TAG channel overlay across participants**
- Notebook section: *Stimulus marker extraction*
- File: `plots/time_alignment/tag_overlay.png`
- Library: **Matplotlib**

```
Amplitude
 │  participant 1:  ____│▌▌▌│____│▌▌│____
 │  participant 2:  ______│▌▌▌│____│▌▌│__
 │  participant 3:  ___│▌▌▌│____│▌▌│_____
 │                       ↑ lag offset
 └────────────────────────────────────────► time (s)
```
*Interpretation: Offsets between marker pulses indicate recording start lags; cross-correlation aligns them.*

---

**Plot 0-B-2 · Pairwise lag heatmap**
- Notebook section: *Cross-correlation lags*
- File: `plots/time_alignment/pairwise_lag_heatmap.png`
- Library: **Seaborn** (`heatmap`)

```
        P01  P02  P03  ...  PN
P01  [  0   -2s  +1s  ...  ]
P02  [ +2s   0   +3s  ...  ]
P03  [ -1s  -3s   0   ...  ]
...
PN   [ ...              0  ]
      (antisymmetric — lag in seconds)
```
*Interpretation: Small lags → participants started near-simultaneously; large lags → manual crop needed.*

---

## Stage 1 — Mean-Variance Synchrony Analysis

**Notebooks:** `notebooks/01-raw-mean-variance-analysis/`

### Overview

Mean-variance synchrony is a simple, model-free proxy for inter-subject agreement. After z-scoring each participant's signal (mean = 0, std = 1 across time), the **variance across subjects** at each time point measures momentary disagreement — low variance = high synchrony.

```
Input:  (n_subjects, n_channels, n_times)   z-scored EEG
        │
        ├── var(axis=0)  →  (n_channels, n_times)   inter_var
        ├── mean(axis=0) →  (n_channels, n_times)   inter_mean
        ├── mean(axis=1) →  (n_times,)               var_t  (channel-averaged)
        └── windowed    →  DataFrame[window, mean_variance, sync_candidate]
```

---

### 1-A · Broadband Mean-Variance
**File:** `notebooks/01-raw-mean-variance-analysis/mean_variance_broadband.ipynb`

#### Plots

---

**Plot 1-A-1 · Intersubject variance time course**
- Notebook section: *Global variance trace*
- File: `plots/01-raw-mean-variance-analysis/broadband/variance_timecourse.png`
- Library: **Matplotlib** / **Seaborn** (`lineplot`)

```
Inter-subject
variance
 │
1.2│ ╲ _  CLASSIC ── ─
 │  ╲/ ╲_/╲___/╲_
0.8│             ╲___/╲___
 │         PSYTRANCE ── ─
0.6│  _____/──────────────
 │
 └──────────────────────────────────────────► time (min)
      0      5     10     15     20
  ████ = synchrony candidate windows (low variance)
```
*Interpretation: Periods of low variance indicate moments when all subjects' brain signals are similar — shared neural response to the music. Psytrance may show more sustained low-variance epochs if it drives stronger entrainment.*

---

**Plot 1-A-2 · Windowed variance distribution (histogram)**
- Notebook section: *Synchrony candidate labelling*
- File: `plots/01-raw-mean-variance-analysis/broadband/windowed_variance_hist.png`
- Library: **Seaborn** (`histplot`)

```
# windows
 │
40│  ▓▓
30│  ▓▓▓▓
20│  ▓▓▓▓▓▓
10│  ▓▓▓▓▓▓▓▓░░░░  ← sync threshold (10th percentile)
 0│  ▓▓▓▓▓▓▓▓░░░░░░
 └──────────────────────────────────────────► mean window variance
      low                         high
    (sync)                     (no sync)
```
*Interpretation: The left tail of the distribution (below the threshold) represents time windows where subjects were most synchronised.*

---

**Plot 1-A-3 · Per-subject channel average time series**
- Notebook section: *Individual traces*
- File: `plots/01-raw-mean-variance-analysis/broadband/subject_traces.png`
- Library: **Matplotlib**

```
z-score
 │  S01 ──────────────────────────────────
 │  S02  ────────────────────────────────
+2│  S03 ──────────────────────────────────
 0│  ……
-2│  S_N ──────────────────────────────────
 │
 └──────────────────────────────────────────► time (min)
  ████ = synchrony candidate windows
```
*Interpretation: Visually check whether the highlighted synchrony windows correspond to moments where individual traces converge.*

---

**Plot 1-A-4 · Pairwise ISC matrix (broadband, z-scored)**
- Notebook section: *Pairwise correlation*
- File: `plots/01-raw-mean-variance-analysis/broadband/pairwise_isc_matrix.png`
- Library: **Seaborn** (`heatmap`)

```
        S01  S02  S03  ...  SN
S01  [ 1.0  0.12 0.08  ...  ]
S02  [ 0.12 1.0  0.15  ...  ]
S03  [ 0.08 0.15 1.0   ...  ]
...
SN   [ ...              1.0 ]
      (symmetric Pearson r; diagonal = 1)
```
*Interpretation: Consistently high off-diagonal values indicate group-level synchrony. Can be compared between CLASSIC and PSYTRANCE.*

---

### 1-B · Per-Band Mean-Variance
**File:** `notebooks/01-raw-mean-variance-analysis/mean_variance_bands.ipynb`

Same data shape as broadband but applied after bandpass filtering into 5 EEG bands:

| Band | Range | Neuroscientific Significance |
|------|-------|------------------------------|
| Delta | 1–4 Hz | Deep states, slow cortical potentials |
| Theta | 4–8 Hz | Memory, meditation, emotional processing |
| Alpha | 8–13 Hz | Relaxed attention, thalamo-cortical loops |
| Beta | 13–30 Hz | Active cognition, motor control |
| Gamma | 30–70 Hz | Feature binding, high-level perception |

#### Plots

---

**Plot 1-B-1 · Per-band variance time courses (multi-panel)**
- Notebook section: *Band-specific variance traces*
- File: `plots/01-raw-mean-variance-analysis/bands/variance_timecourse_bands.png`
- Library: **Matplotlib** / **Seaborn**

```
 [DELTA]   variance ──────────────────────────────────
 [THETA]   variance ──────────────────────────────────
 [ALPHA]   variance ──────────────────────────────────
 [BETA]    variance ──────────────────────────────────
 [GAMMA]   variance ──────────────────────────────────
                    ────────────────────────────────► time (min)
```
*Interpretation: Alpha and theta bands are expected to show stronger synchrony under psilocybin, which amplifies thalamo-cortical and limbic rhythms.*

---

**Plot 1-B-2 · Pairwise ISC matrices per band (facet grid)**
- Notebook section: *Per-band pairwise correlation*
- File: `plots/01-raw-mean-variance-analysis/bands/pairwise_isc_bands.png`
- Library: **Seaborn** (`FacetGrid` + `heatmap`)

```
 DELTA      THETA      ALPHA      BETA       GAMMA
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│ ISC │   │ ISC │   │ ISC │   │ ISC │   │ ISC │
│ mat │   │ mat │   │ mat │   │ mat │   │ mat │
└─────┘   └─────┘   └─────┘   └─────┘   └─────┘
         (symmetric NxN heatmaps, one per band)
```
*Interpretation: Bands with uniformly high off-diagonal values indicate frequency-specific inter-subject alignment.*

---

## Stage 2 — ISC (Inter-Subject Correlation)

**Notebooks:** `notebooks/02-isc-broadband-analysis/`

### Overview

ISC is the primary analysis measure. Three variants are computed:

| Method | Shape In | Shape Out | Description |
|--------|----------|-----------|-------------|
| Leave-one-out (LOO) | `(N, F, T)` | `(N, F)` + `(F,)` | Subject vs mean of rest, per feature |
| Pairwise | `(N, F, T)` | `(N, N)` | Mean-across-features r for every pair |
| Sliding-window | `(N, F, T)` | `(W, F)` + `(W,)` | Time-resolved LOO ISC |
| Mean-field | `(N, F, T)` | `(N,)` | LOO on spatial mean first |

Where N=subjects, F=channels/features, T=time samples, W=windows.

Both **Pearson** and **Spearman** variants exist for all methods.

---

### 2-A · Broadband ISC
**File:** `notebooks/02-isc-broadband-analysis/isc_broadband.ipynb`

#### Data Shape Chain
```
(n_subjects, n_channels, n_times)
    │
    ├──[LOO-ISC]──────────────────────────────────────────────────────────
    │   compute_loo_isc()  →  (n_subjects, n_channels)
    │   .mean(axis=0)      →  (n_channels,)   = mean_loo_isc
    │
    ├──[Pairwise ISC]─────────────────────────────────────────────────────
    │   compute_pairwise_isc()  →  (n_subjects, n_subjects)
    │
    ├──[Sliding-Window ISC]───────────────────────────────────────────────
    │   compute_sliding_window_isc(window_sec, step_sec, sfreq)
    │       →  (n_windows, n_channels)   isc_timecourse
    │       →  (n_windows,)              window_times
    │
    └──[Mean-Field ISC]───────────────────────────────────────────────────
        compute_mean_field_loo_isc()
            →  mean across channels first  →  (n_subjects,) per method
```

#### Plots

---

**Plot 2-A-1 · LOO-ISC distribution (histogram)**
- Notebook section: *LOO-ISC distribution*
- File: `plots/02-isc-broadband-analysis/broadband/loo_isc_distribution.png`
- Library: **Matplotlib** (`plot_loo_isc_distribution`)
- Function: `src.visualization.isc_plots.plot_loo_isc_distribution`

```
# channels
 │
50│  ▓▓
40│  ▓▓▓▓
30│  ▓▓▓▓▓░░  ← CLASSIC
20│  ░░░░░░░░░
10│  ░░░░░░░░░░░░  ← PSYTRANCE
 0│  ░░░░░░░░░░░░░░░
 └──────────────────────────────────────────► Pearson r (LOO-ISC)
      -0.1     0      0.1    0.2    0.3
      │                 │
      ← desync       sync →
  dashed lines = distribution means
```
*Interpretation: Distributions shifted right of zero indicate above-chance synchrony. Psilocybin condition compared to placebo should show rightward shift, especially for PSYTRANCE.*

---

**Plot 2-A-2 · Sliding-window ISC (time-resolved heatmap + mean trace)**
- Notebook section: *Time-resolved ISC*
- File: `plots/02-isc-broadband-analysis/broadband/sliding_window_isc.png`
- Library: **Matplotlib** (`plot_sliding_window_isc`)
- Function: `src.visualization.isc_plots.plot_sliding_window_isc`

```
Mean LOO-ISC (r)                           [CLASSIC]
0.15 │         ╭─╮     ╭──╮
0.10 │       ╭─╯ ╰─────╯  ╰─╮
0.05 │ ──────╯                ╰──────────── threshold r
 0   │─────────────────────────────────────
     └──────────────────────────────────────► time (min)
      0       5       10      15      20
     ████████        ██████       (gold = above threshold)

Channel index                             [heatmap]
 200 │░░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░
 100 │░░░▓▓▓▓▓▓▓▓░░░░░░░░░▓▓▓▓▓▓░░░░░░░░░░░
   0 │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
     └──────────────────────────────────────► time (min)
           RdBu_r colormap: blue=neg, red=pos
```
*Interpretation: Gold-highlighted windows = significant synchrony epochs. Heatmap reveals which electrodes drive the synchrony (e.g., frontal vs occipital).* 

---

**Plot 2-A-3 · Pairwise ISC matrix**
- Notebook section: *Pairwise ISC heatmap*
- File: `plots/02-isc-broadband-analysis/broadband/pairwise_isc_matrix.png`
- Library: **Seaborn** (`heatmap`)

```
        S01  S02  S03  ...  SN
S01  [ 1.0  0.08 0.12  ...  ]
S02  [ 0.08 1.0  0.10  ...  ]
     (N×N symmetric Pearson r, diagonal = 1)
     (mean across all channels)
```
*Interpretation: Outlier subjects (row/column consistently near 0) may be non-responders or have poor data quality.*

---

**Plot 2-A-4 · Mean-field LOO-ISC per subject (bar chart)**
- Notebook section: *Mean-field ISC*
- File: `plots/02-isc-broadband-analysis/broadband/mean_field_loo_isc.png`
- Library: **Seaborn** (`barplot`)
- Function: `src.visualization.isc_plots.plot_mean_field_loo_isc`

```
LOO-ISC (r)
0.15│  ■ Pearson  □ Spearman
0.10│  ■□  ■□  □■  ■□  ■□  ■□  ■□  ■□  ■□  ■□
0.05│
 0  │───────────────────────────────────────────
-0.05│                           □■
    └────────────────────────────────────────► Subject
         S01 S02 S03 S04 ... SN
```
*Interpretation: Discrepancy between Pearson and Spearman indicates outlier time points or non-Gaussian signals. Subjects below zero are desynchronised from the group.*

---

**Plot 2-A-5 · Mean-field pairwise ISC heatmap**
- Notebook section: *Mean-field pairwise ISC*
- File: `plots/02-isc-broadband-analysis/broadband/mean_field_pairwise_isc.png`
- Library: **Seaborn** (`heatmap`)
- Function: `src.visualization.isc_plots.plot_mean_field_pairwise_isc`

```
     (same layout as pairwise matrix above but computed on
      spatial-mean-field signal, not mean across channel-wise r)
      A large discrepancy vs Plot 2-A-3 suggests spatially
      heterogeneous synchrony across the scalp.)
```

---

**Plot 2-A-6 · Significant synchrony intervals (text / annotated timeline)**
- Notebook section: *Significant intervals*
- File: *text output to notebook cell output (no saved figure)*
- Function: `src.visualization.isc_plots.print_significant_intervals`

```
=== CLASSIC ===
Threshold: r > 0.035  |  Significant windows: 42/240 (17.5%)
  #   Start (min)   End (min)  Duration (s)   Mean r   Peak r
  1        2.083      4.250         130.0   0.0521   0.0842
  2        9.417     11.083          99.6   0.0478   0.0713
  ...
```
*Interpretation: Long significant intervals correspond to specific musical passages that elicit shared neural responses across participants.*

---

### 2-B · Per-Band ISC
**File:** `notebooks/02-isc-broadband-analysis/isc_bands.ipynb`

#### Data Shape Chain
```
(n_subjects, n_channels, n_times)
    │
    ╔═ for each band (DELTA, THETA, ALPHA, BETA, GAMMA): ════════════════
    ║   bandpass filter  →  (n_subjects, n_channels, n_times)
    ║   compute_loo_isc  →  (n_subjects, n_channels) + (n_channels,)
    ║   compute_sliding_window_isc  →  (n_windows, n_channels) + (n_windows,)
    ╚═══════════════════════════════════════════════════════════════════════
```

#### Plots

---

**Plot 2-B-1 · Band ISC distributions (facet histogram)**
- Notebook section: *Per-band LOO-ISC distributions*
- File: `plots/02-isc-broadband-analysis/bands/band_isc_distributions.png`
- Library: **Matplotlib** (`plot_band_isc_distributions`)
- Function: `src.visualization.isc_plots.plot_band_isc_distributions`

```
DELTA (1-4Hz)  THETA (4-8Hz)  ALPHA (8-13Hz)  BETA (13-30Hz)  GAMMA (30-70Hz)
┌──────────┐  ┌──────────┐   ┌──────────┐    ┌──────────┐    ┌──────────┐
│  ▓▓      │  │    ▓▓    │   │   ░▓▓    │    │  ▓▓░░    │    │  ▓▓░░    │
│  ▓▓░░    │  │   ▓▓░░   │   │  ░▓▓░░   │    │  ▓▓░░░   │    │  ▓▓░░░   │
└──────────┘  └──────────┘   └──────────┘    └──────────┘    └──────────┘
      ▓ = CLASSIC    ░ = PSYTRANCE
      dashed lines = distribution means
```
*Interpretation: If alpha/theta bands show higher ISC under psilocybin (vs placebo), this supports the hypothesis that psilocybin entrains low-frequency oscillations. Gamma band ISC under psytrance may reflect rhythmic entrainment to the beat.*

---

**Plot 2-B-2 · Band mean ISC bar chart**
- Notebook section: *Mean ISC per band summary*
- File: `plots/02-isc-broadband-analysis/bands/band_mean_isc_bar.png`
- Library: **Matplotlib** (`plot_band_mean_isc_bar`)
- Function: `src.visualization.isc_plots.plot_band_mean_isc_bar`

```
Mean LOO-ISC (r) ± SD
0.20│
0.15│   ■■   ■■   ■■   ■■   ■■
0.10│   ■■   ■■   ■■   ■■   ■■
0.05│   ■■   □□   ■■   □□   □□
 0  │─────────────────────────────
    │ DELTA THETA ALPHA BETA GAMMA
         ■ CLASSIC   □ PSYTRANCE
         (error bars = SD across channels)
```
*Interpretation: Bands with the tallest bars are frequency ranges where group-level neural synchrony is strongest. Compare music types to see if e.g. psytrance drives stronger beta/gamma synchrony (rhythmic entrainment) while classical drives alpha.*

---

**Plot 2-B-3 · Band sliding-window ISC (per-band trace + heatmap grid)**
- Notebook section: *Time-resolved per-band ISC*
- File: `plots/02-isc-broadband-analysis/bands/band_sliding_window_isc.png`
- Library: **Matplotlib** (`plot_band_sliding_window_isc`)
- Function: `src.visualization.isc_plots.plot_band_sliding_window_isc`

```
         [CLASSIC]                   [PSYTRANCE]
DELTA ┌─────────────────────┐  ┌─────────────────────┐
      │ trace + heatmap     │  │ trace + heatmap     │
      └─────────────────────┘  └─────────────────────┘
THETA ┌─────────────────────┐  ┌─────────────────────┐
      │ trace + heatmap     │  │ trace + heatmap     │
      └─────────────────────┘  └─────────────────────┘
ALPHA ┌─────────────────────┐  ...
BETA  ┌─────────────────────┐  ...
GAMMA ┌─────────────────────┐  ...
      │time (min)→          │
      (gold = above-threshold windows)
```
*Interpretation: Shows whether synchrony is temporally stable (broadband entrainment) or episodic and frequency-specific. Gamma synchrony in psytrance expected near strong beat onset moments.*

---

**Plot 2-B-4 · Band overlap — consensus synchrony windows**
- Notebook section: *Cross-band synchrony consensus*
- File: `plots/02-isc-broadband-analysis/bands/band_overlap.png`
- Library: **Matplotlib** (`plot_band_overlap`)
- Function: `src.visualization.isc_plots.plot_band_overlap`

```
           [CLASSIC]
Band raster (binary, one row per band):
DELTA   │████░░░░░░████░░░░░░░░░░████░░░│
THETA   │░░░████░░░░░░░░████░░░░░░░░░░░░│
ALPHA   │░░░░░░████████░░░░░░████████░░░│
BETA    │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
GAMMA   │░░░░████░░░░░░░████░░░░░████░░░│
Broadband│░░░████░░░░░░░░░░░░░░░░████░░░│
         └──────────────────────────────► time (min)
Stacked area (# significant bands per window):
5 │      ╱╲       ╱╲        ╱╲
3 │  ╱╲╱  ╲╱──╱╲╱  ╲╱───╱╲╱  ╲╱
0 │──────────────────────────────► time (min)
★ = all bands simultaneously significant
```
*Interpretation: Windows where all (or most) frequency bands are simultaneously significant represent the strongest, most global neural synchrony moments. These are the best candidates for "peak synchrony" — correlatable with musical features or subjective experience ratings.*

---

**Plot 2-B-5 · Band significant intervals summary (text)**
- Notebook section: *Per-band significant intervals*
- File: *text output (no saved figure)*
- Function: `src.visualization.isc_plots.print_band_significant_intervals`

---

## Visualization Delivery Tools

### Option Comparison

| Tool | Setup effort | Interactivity | Hosting | Best for |
|------|-------------|---------------|---------|----------|
| **Static PNGs in repo** | None | None | GitHub (inline in notebooks/README) | Current approach; always works |
| **Jupyter nbviewer** | None | None | nbviewer.org (free) | Sharing rendered notebooks with plots without running them |
| **GitHub Pages (static HTML)** | Low–Medium | None–Low | GitHub (free, auto-deploy) | Polished documentation site with embedded images |
| **Quarto** | Low | Medium (via Observable JS) | GitHub Pages / Posit Connect | Reproducible reports + interactive sliders; renders `.ipynb` natively |
| **Plotly / Bokeh** | Medium | High | Any server or HTML file | Pan/zoom on time-course plots; hover on heatmaps |
| **Panel / Voilà** | Medium | High | Binder / HPC / local | Dashboard with live parameter widgets directly in notebook |
| **Streamlit** | Low | High | Streamlit Cloud (free tier) | Quick interactive app; no Jupyter needed |
| **Weights & Biases (W&B)** | Low | Medium | W&B cloud (free tier) | Experiment tracking + auto-logged plots across runs |
| **MLflow** | Low | Low | Local / self-hosted | Lightweight run tracking, good for HPC |

---

### Recommendation

For this project the most pragmatic progression is:
```
Phase 1 (now):  Static PNGs saved from notebooks → committed to repo
                → already implemented via SAVE_PLOTS + PLOTS_DIR pattern

Phase 2 (soon): GitHub Pages with MkDocs or Quarto
                → auto-renders notebooks on push via GitHub Actions
                → zero extra infrastructure, stays in this repo

Phase 3 (if needed): Plotly/Bokeh for time-course and heatmap interactivity
                     → add as optional rendering layer inside notebooks
                     → does not replace Seaborn/Matplotlib production plots
```

#### Interactive Explorer (implemented)

The **Interactive Explorer** page in `viz_catalog/` provides Phase-3-level
interactivity today, without replacing the existing static-image workflow:

```
Analysis pipeline ──► src.analysis.results_store ──► CSV results database
                                                           │
                                                           ▼
                                              viz_catalog/ Streamlit app
                                              └── 📊 Interactive Explorer
                                                  ├── Time-series viewer
                                                  ├── Windowed stats viewer
                                                  ├── LOO-ISC distribution
                                                  └── Pairwise ISC matrix
```

**How it works:**

1. Analysis scripts call `save_intersubject_timeseries()`,
   `save_windowed_stats()`, `save_loo_isc()`, `save_pairwise_isc()` from
   `src.analysis.results_store` to export results as CSV files.
2. CSV files are stored in a canonical directory layout:
   `<root>/<Condition>_<MusicType>/<broadband|bands/band>/<analysis>.csv`
3. The **Interactive Explorer** scans this directory, offers sidebar filters
   (condition, music type, band, analysis type), and renders each CSV with a
   dedicated interactive Streamlit visualisation (line charts, area charts,
   bar charts, styled heatmap DataFrames, histograms, summary metrics).
4. Users can zoom, hover, adjust downsampling, and inspect raw data tables —
   all in the browser, no Jupyter or Python needed.

#### GitHub Pages — Is It a Good Fit?

**Yes**, with caveats:

✅ Free, zero-infrastructure, integrates with GitHub Actions CI  
✅ Can render notebook outputs (via `nbconvert` or Quarto) automatically on every push  
✅ Embeds static images natively — no extra hosting needed  
✅ Good for sharing with collaborators and supervisors via a URL  

⚠️ Not interactive by default (no live sliders / hover)  
⚠️ Large notebooks (with plot outputs) can slow down page load  
⚠️ EEG topomaps from MNE don't render well without explicit PNG export  

**Recommended setup:** Use [Quarto](https://quarto.org/) + GitHub Pages:
1. Add a `_quarto.yml` config pointing at the `notebooks/` folder.
2. Add a GitHub Actions workflow (`.github/workflows/docs.yml`) to run `quarto render` on push to `develop`.
3. GitHub automatically publishes the rendered site to `gh-pages` branch → accessible at `https://dbeinhauer.github.io/psilocybin-eeg/`.

---

#### Simpler Alternatives (Lower Barrier)

- **nbviewer links in README.md** — just paste the notebook GitHub URL into `https://nbviewer.org/`. Zero setup.
- **Binder badge** — add a `[![Binder](...)]` badge to README; users get a live, runnable environment in the cloud.
- **Streamlit app** — one `app.py` file that loads saved `.png` plots and `.csv` results; deployable to Streamlit Cloud in minutes.

---

## Plot Naming Conventions

All saved plots follow this pattern:

```
notebooks/
└── NN-<analysis-name>/
    └── plots/
        └── <scope>/
            └── <descriptive_snake_case_name>.png
```

| Variable | Example values |
|----------|---------------|
| `NN` | `00`, `01`, `02` |
| `<analysis-name>` | `preprocessing`, `raw-mean-variance-analysis`, `isc-broadband-analysis` |
| `<scope>` | `broadband`, `bands`, `time_alignment` |
| `<name>` | `loo_isc_distribution`, `sliding_window_isc`, `band_overlap` |

**Full example:**
```
notebooks/02-isc-broadband-analysis/plots/broadband/sliding_window_isc_CLASSIC.png
notebooks/02-isc-broadband-analysis/plots/bands/band_overlap_PSYTRANCE.png
```

The `PLOTS_DIR` constant in each notebook's configuration cell is set to the appropriate subdirectory:
```python
PLOTS_DIR = ProjectPaths.NOTEBOOKS_DIR / "02-isc-broadband-analysis" / "plots" / "broadband"
```