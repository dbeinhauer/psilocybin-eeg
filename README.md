# Psilocybin-EEG

Analysis pipeline for studying the effects of **psilocybin** on neural synchrony during
music listening, measured with 257-channel (GSN-HydroCel) EEG.

The pipeline takes raw EDF recordings through multi-stage preprocessing (channel
preparation, filtering, ICA artifact removal, stimulus-based time alignment) and then
through a family of group-level analyses: inter-subject correlation, mean-variance
synchrony, wavelet power/phase, and several blind-source-separation approaches (ICA, IVA,
joint ICA) applied to wavelet tensors.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Data Layout](#data-layout)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Analysis Stages](#analysis-stages)
7. [Running Tests](#running-tests)
8. [HPC Submission](#hpc-submission)
9. [Documentation](#documentation)

---

## Project Overview

| Aspect | Details |
|---|---|
| **EEG system** | EGI GSN-HydroCel, 257 channels |
| **Experiments** | `psilo_music` (music listening), `assr` (auditory steady-state response) |
| **File format** | EDF (raw), MNE `.fif` (processed), NumPy `.npy` / `.npz` (probabilities, components) |
| **Conditions** | Placebo / Psilocybin (within-subject, counterbalanced A/B) |
| **Music types** | Classical (`CLASSIC`), Psytrance (`PSYTRANCE`) |
| **Analyses** | ISC, mean-variance synchrony, wavelet power/phase, wavelet ICA, wavelet IVA, joint ICA, ASSR trial statistics |
| **Python** | ≥ 3.12 |

The two experiments differ in how stimulus timing is recovered. `psilo_music` uses a TAG
channel carrying the music signal and cross-correlation alignment. `assr` uses `fam+`
annotations, handled by `stimulus_alignment.py` and `marker_shift.py`.

---

## Repository Structure

```
psilocybin-eeg/
├── config/          # Electrode coordinates, excluded electrodes, participant maps
├── data/            # Raw, interim, and processed data (gitignored)
├── src/
│   ├── definitions/ #   Enums, path constants, frequency bands, channel mappings
│   ├── io/          #   Filename parsing, loading, saving, IVA component store
│   ├── preprocessing/ # Channel prep, filtering, ICA, time & stimulus alignment
│   ├── analysis/    #   ISC, mean-variance, wavelet ICA/IVA/jICA, ASSR trials
│   ├── visualization/ # Preprocessing, ISC, mean-variance, wavelet, IVA, jICA plots
│   ├── filtering/   #   Metadata-level DataFrame filtering
│   └── utils/       #   Logging helpers
├── scripts/         # CLI entry points
├── notebooks/       # Parameter-exploration notebooks, stages 00-07
├── tests/           # pytest suite (core logic; no plotting or notebook tests)
├── jobs/metacentrum/# PBS job templates, one directory per stage
├── docs/            # Extended documentation
└── pyproject.toml
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- (Optional) CUDA 12.x for GPU-accelerated ICA via PyTorch

### Installation

```bash
git clone https://github.com/dbeinhauer/psilocybin-eeg.git
cd psilocybin-eeg

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The package is **not** installed into the environment — `import src` works because the
repository root is on `sys.path`, which scripts and notebooks arrange themselves. Run
everything from the repository root.

On the group machines the working interpreter is the conda env `psilo-music`.

### Configuration

Place the following in `config/` before running the pipeline:

| Path | Description |
|---|---|
| `config/coordinates/*.sfp` | Electrode coordinate files for the GSN-HydroCel montage |
| `config/excluded_electrodes/*.csv` | Boundary electrodes to exclude |
| `config/assr_electrodes/*.csv` | Electrode subsets used by the ASSR checks |
| `config/participant_mappings/*.csv` | Semicolon-delimited maps from EEG condition IDs (A/B) to labels (Placebo/Psilocybin) |

Raw EDF data goes in `data/raw/<experiment>/`.

---

## Data Layout

```
data/
├── raw/{psilo_music,assr}/           # Original .edf files (untouched)
├── events/                           # Event/annotation sidecars
├── interim/{psilo_music,assr}/       # Intermediate products
│   ├── before_ica/                   #   Raw data before ICA (.fif)
│   ├── ica_components/               #   Fitted ICA objects (.fif)
│   └── ic_probabilities/             #   ICLabel probabilities (.npy)
├── processed/{psilo_music,assr}/     # Final outputs
│   ├── after_ica/                    #   Artifact-cleaned EEG (.fif)
│   ├── cropped/                      #   Time-aligned, cropped (.fif)
│   ├── concatenated/                 #   Concatenated data
│   ├── wavelets/                     #   Wavelet-power cache
│   ├── iva_results/                  #   Stored IVA/ICA component products (.npz)
│   └── raw_excluded_ic/              #   Excluded IC metadata
├── wrong_condition/                  # Recordings with a mislabelled condition
└── excluded_participants/
    └── {psilo_music,assr}.csv        # Exclusion lists with reasons
```

Exclusion reasons in those CSVs are the literal values of `ExclusionCategories` in
`src/definitions/fields.py` — the enum and the data share one vocabulary.

See [`data/README.md`](data/README.md) for naming conventions.

---

## Pipeline Workflow

### 1. Preprocessing (`scripts/run_preprocessing.py`)

1. **Channel preparation** — rename raw EDF channels to montage names, set channel types
   (EEG, ECG, TAG/stim), apply the GSN-HydroCel montage, exclude boundary electrodes
2. **Signal filtering** — crop first/last 10 s, 50 Hz notch, FIR bandpass 1–100 Hz,
   RANSAC bad-channel detection and interpolation, average reference, AutoReject epoch
   annotation
3. **ICA artifact removal** — extended-Infomax ICA, ICLabel classification, threshold-based
   component exclusion, signal reconstruction

### 2. Time alignment

- `scripts/run_time_alignment.py` — TAG-channel cross-correlation alignment
  (`psilo_music`)
- `scripts/run_stimulus_alignment.py` — annotation-driven alignment (`assr`), with
  `scripts/run_stimulus_shift_calibration.py` as the one-off calibration tool whose
  result is baked into `src/definitions/constants.py`

### 3. Analysis

```bash
python scripts/run_preprocessing.py --raw_processing --plot_results
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC

python scripts/run_isc.py --music_type CLASSIC PSYTRANCE
python scripts/run_mean_variance.py --music_type CLASSIC PSYTRANCE
python scripts/run_analysis.py --analysis wavelet_power   # --analysis is required
```

`scripts/run_analysis.py` is the **wavelet** entry point (`wavelet_power` /
`wavelet_phase`); ISC has its own script. Every script accepts `--help`.

---

## Analysis Stages

Notebooks, plots and PBS jobs are organised by the same stage numbering:

| Stage | Topic |
|---|---|
| `00-preprocessing` | Data inspection, ASSR timing verification, stimulus alignment, raw/wavelet PCA |
| `01-raw-mean-variance-analysis` | Intersubject mean-variance synchrony |
| `02-isc-broadband-analysis` | Broadband and per-band ISC |
| `03-wavelet-analysis` | Wavelet power and phase exploration |
| `04-wavelet-ica-analysis` | ICA on wavelet tensors, five axis variants |
| `05-wavelet-iva-analysis` | IVA on wavelet tensors, plus component quality |
| `06-iva-condition-comparison` | IVA across conditions (joined / joined-tracks) |
| `07-ica-condition-comparison` | Joint ICA across conditions |

A notebook is for **parameter selection**: config, the grid being explored, a small-scale
call into the same `src/` entry point the script and job use, and the resulting figure.
Shared logic belongs in `src/`.

---

## Running Tests

```bash
python -m pytest tests/ -v      # ~1,340 tests, roughly 9 minutes
python -m pytest tests/test_isc.py -v
```

Tests cover **core logic** — the numerical and statistical code, filename parsing, data
alignment and the component stores. Plotting and notebooks are deliberately not tested.

---

## HPC Submission

PBS job templates live under `jobs/metacentrum/`, one directory per analysis stage:

```bash
qsub jobs/metacentrum/00-preprocessing/run_full_preprocessing.pbs
qsub jobs/metacentrum/05-wavelet-iva-analysis/run_channel.pbs
```

See [`docs/hpc_guide.md`](docs/hpc_guide.md) for resource requirements. When adding a
template, copy an existing one's resource shape rather than inventing a new one.

---

## Documentation

| Document | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | Architecture and conventions, for both agents and humans |
| [`docs/preprocessing_steps.md`](docs/preprocessing_steps.md) | Detailed preprocessing stages |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Enums, stored products, file naming |
| [`docs/hpc_guide.md`](docs/hpc_guide.md) | Cluster submission and resources |
| [`docs/claude_workflow.md`](docs/claude_workflow.md) | Claude Code skills used in this repo |
| [`src/preprocessing/README.md`](src/preprocessing/README.md) | Preprocessing module reference |
| [`src/analysis/README.md`](src/analysis/README.md) | Analysis module reference |
| [`data/README.md`](data/README.md) | Data naming conventions |
