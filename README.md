# Psilocybin-EEG

Analysis pipeline for studying the effects of **psilocybin** on neural
synchrony during music listening, measured with 257-channel
(GSN-HydroCel) EEG.

The pipeline takes raw EDF recordings through multi-stage preprocessing
(channel preparation, filtering, ICA artifact removal, stimulus-based time
alignment) and computes **Inter-Subject Correlation (ISC)** to quantify
shared neural responses across participants under placebo and psilocybin
conditions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Data Layout](#data-layout)
5. [Pipeline Workflow](#pipeline-workflow)
6. [Running the Pipeline](#running-the-pipeline)
7. [Visualization Catalog](#visualization-catalog)
8. [Running Tests](#running-tests)
9. [HPC Submission](#hpc-submission)
10. [Documentation](#documentation)

---

## Project Overview

| Aspect            | Details |
|-------------------|---------|
| **EEG system**    | EGI GSN-HydroCel 257 channels |
| **File format**   | EDF (raw), MNE `.fif` (processed), NumPy `.npy` (probabilities) |
| **Conditions**    | Placebo / Psilocybin (within-subject, counterbalanced A/B) |
| **Music types**   | Classical (`CLASSIC`), Psytrance (`PSYTRANCE`) |
| **Core analysis** | Leave-one-out ISC, pairwise ISC, sliding-window ISC per frequency band |
| **Python**        | ≥ 3.12 |

---

## Repository Structure

```
psilocybin-eeg/
├── config/                        # Electrode coordinates, excluded electrodes, participant maps
├── data/                          # Raw, interim, and processed data (gitignored)
├── src/                           # Source package
│   ├── definitions/               #   Enums, constants, channel mappings
│   ├── io/                        #   Filename parsing, data loading & saving
│   ├── preprocessing/             #   Channel prep, filtering, ICA, time alignment
│   ├── analysis/                  #   ISC computation, data representations, summary
│   ├── visualization/             #   Preprocessing & ISC plotting
│   ├── filtering/                 #   Metadata-level DataFrame filtering
│   └── utils/                     #   Logging helpers
├── scripts/                       # CLI entry points
├── notebooks/                     # Exploratory Jupyter notebooks
├── tests/                         # pytest test suite
├── jobs/                          # HPC job scripts (Metacentrum, Umbriel)
├── docs/                          # Extended documentation
├── CODEBASE_STRUCTURE.md          # Full directory layout description
├── pyproject.toml                 # Build & dependency configuration
└── requirements.txt               # Pinned dependencies
```

See [`CODEBASE_STRUCTURE.md`](CODEBASE_STRUCTURE.md) for the complete,
annotated directory tree.

---

## Getting Started

### Prerequisites

- Python 3.12+
- (Optional) CUDA 12.x for GPU-accelerated ICA via PyTorch

### Installation

```bash
# Clone the repository
git clone https://github.com/dbeinhauer/psilocybin-eeg.git
cd psilocybin-eeg

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# — or, for an editable install using pyproject.toml —
pip install -e .
```

### Configuration

Place the following files in `config/` before running the pipeline:

| Path | Description |
|------|-------------|
| `config/coordinates/*.sfp` | Electrode coordinate files for the GSN-HydroCel montage |
| `config/excluded_electrodes/*.csv` | CSVs listing boundary electrodes to exclude |
| `config/participant_mappings/*.csv` | Semicolon-delimited CSVs mapping EEG condition IDs (A/B) to labels (Placebo/Psilocybin) |

Place raw EDF data in `data/raw/psilo_music/`.

---

## Data Layout

```
data/
├── raw/psilo_music/                  # Original .edf files (untouched)
├── interim/psilo_music/              # Intermediate products
│   ├── before_ica/                   #   Raw data before ICA (.fif)
│   ├── ica_components/               #   Fitted ICA objects (.fif)
│   └── ic_probabilities/             #   ICLabel probabilities (.npy)
├── processed/psilo_music/            # Final outputs
│   ├── after_ica/                    #   Artifact-cleaned EEG (.fif)
│   ├── cropped/                      #   Time-aligned, cropped (.fif)
│   ├── concatenated/                 #   Concatenated data
│   └── raw_excluded_ic/              #   Excluded IC metadata
└── excluded_participants/
    └── psilo_music.csv               # Exclusion list with explanations
```

See [`data/README.md`](data/README.md) for full naming conventions and
expected file structure.

---

## Pipeline Workflow

The pipeline has three main stages:

### 1. Preprocessing (`scripts/run_preprocessing.py`)

1. **Channel preparation** — rename raw EDF channels to montage names,
   set channel types (EEG, ECG, TAG/stim), apply GSN-HydroCel montage,
   exclude boundary electrodes
2. **Signal filtering** — crop first/last 10 s, notch filter at 50 Hz,
   FIR bandpass 1–100 Hz, RANSAC bad-channel detection, interpolation,
   average reference, AutoReject epoch annotation
3. **ICA artifact removal** — extended-Infomax ICA decomposition, ICLabel
   classification (brain, muscle, eye, heart, line noise, channel noise,
   other), threshold-based component exclusion, signal reconstruction

### 2. Time alignment (`scripts/run_time_alignment.py`)

4. **Stimulus alignment** — extract TAG channel from each recording,
   compute pairwise cross-correlation, select a reference, shift and
   crop all recordings to a common time window

### 3. Analysis (`scripts/run_analysis.py`)

5. **ISC computation** — leave-one-out ISC, pairwise ISC, sliding-window
   ISC across delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz),
   beta (13–30 Hz), and gamma (30–70 Hz) frequency bands
6. **Visualization** — topomaps, power spectra, ISC heatmaps, band
   comparisons, sliding-window plots

See [`docs/pipeline_overview.md`](docs/pipeline_overview.md) for a
detailed stage-by-stage description.

---

## Running the Pipeline

```bash
# 1. Full preprocessing (channel prep + filtering + ICA)
python scripts/run_preprocessing.py --raw_processing --plot_results

# 2. Time alignment for a specific condition × music type
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC

# 3. ISC analysis
python scripts/run_analysis.py --condition Placebo --music_type CLASSIC
```

Each script accepts `--help` for a full list of options.

---

## Running Tests

```bash
# Run the full test suite (194 tests)
python -m pytest tests/ -v
```

Tests cover all source modules:

| Test file | Module under test |
|-----------|-------------------|
| `test_dataset_parsing.py` | `src/io/parsing.py` — filename → metadata extraction |
| `test_enum_utils.py` | `src/io/parsing.py` — `check_enum_value_in_variants` |
| `test_fields.py` | `src/definitions/fields.py` — all Enum definitions |
| `test_constants.py` | `src/definitions/constants.py` — `ProjectPaths` |
| `test_mappings.py` | `src/definitions/mappings.py` — channel name/type maps |
| `test_loading.py` | `src/io/loading.py` — data-dir routing, path construction |
| `test_saving.py` | `src/io/saving.py` — save dispatch by data type |
| `test_dataset_filter.py` | `src/filtering/dataset_filter.py` — metadata filtering |
| `test_isc.py` | `src/analysis/isc.py` — LOO, pairwise, sliding-window ISC |
| `test_data_representations.py` | `src/analysis/data_representations.py` — AnalysisData |
| `test_channel_prep.py` | `src/preprocessing/channel_prep.py` — channel ops |
| `test_filtering_preprocessing.py` | `src/preprocessing/filtering.py` — signal filtering |
| `test_ica.py` | `src/preprocessing/ica.py` — ICLabel, component exclusion |
| `test_time_alignment.py` | `src/preprocessing/time_alignment.py` — TAG alignment |
| `test_logging_config.py` | `src/utils/logging_config.py` — logging helpers |
| `test_visualization.py` | `src/visualization/preprocessing_plots.py` — plot utils |

---

## HPC Submission

PBS job scripts for Metacentrum and Umbriel clusters are in `jobs/`:

```bash
# Metacentrum
qsub jobs/metacentrum/run_full_preprocessing.pbs

# Umbriel
qsub jobs/umbriel/run_dataset_preprocessing.pbs
```

See [`docs/hpc_guide.md`](docs/hpc_guide.md) for resource requirements and
configuration details.

---

## Visualization Catalog

An interactive **Streamlit** app for browsing the analysis catalog and exploring
locally generated result plots. It lives entirely in `viz_catalog/` and has
no dependencies on the main Python package.

### Quick start

```bash
# Recommended — with uv (no prior install needed):
uv run --with "streamlit>=1.32.0" --with "pyyaml>=6.0" --with "numpy>=1.24.0" --with "matplotlib>=3.7.0" \
    streamlit run viz_catalog/app.py
```

The app opens at **`http://localhost:8501`** and provides three pages:

| Page | Description |
|------|-------------|
| **📋 Analysis Catalog** | Reference guide for every analysis type: data shapes, order of operations, interpretation notes, notebook links, and matplotlib sketches |
| **🔬 Results Browser** | Local file browser for actual computed plot images generated by the pipeline |
| **📊 Interactive Explorer** | Interactive visualisation of precomputed CSV analysis results (time-series, ISC distributions, windowed stats) |

### Results Browser — expected directory structure

The Results Browser indexes plot files according to the canonical pipeline output
layout:

```
plots/
└── {NN}-{analysis-name}/          ← stage (e.g. 02-isc-broadband-analysis)
    └── {Condition}_{MusicType}/   ← e.g. Placebo_CLASSIC
        ├── broadband/             ← broadband (no per-band filtering)
        │   └── {analysis_type}/  ← e.g. loo_isc, pairwise_isc, sliding_window
        │       └── *.png
        └── bands/                 ← per-frequency-band analyses
            └── {analysis_type}/  ← e.g. loo_isc, pairwise_isc, windowed
                └── *.png
```

Sidebar filters: **Analysis stage**, **Condition**, **Music type**,
**Spectrum type** (`broadband` / `bands`), **Analysis type**, **Frequency band**,
and a free-text **Filename** search. All filters default to empty (nothing shown)
until at least one is selected.

See [`viz_catalog/README.md`](viz_catalog/README.md) for full usage
instructions including how to copy results from the HPC cluster.

### Interactive Explorer — results database

The **Interactive Explorer** page reads precomputed CSV result files and
renders interactive Streamlit charts (line/area/bar charts, styled DataFrames).
Use `src.analysis.results_store` save functions to generate the CSV database:

```
results_db/
└── {NN}-{analysis-name}/           ← e.g. 01-raw-mean-variance-analysis
    └── {Condition}_{MusicType}/    ← e.g. Placebo_CLASSIC
        ├── broadband/
        │   ├── intersubject_timeseries.csv
        │   ├── windowed_stats.csv
        │   ├── loo_isc.csv
        │   └── pairwise_isc.csv
        └── bands/{band}/           ← e.g. alpha, delta
            └── *.csv
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [`CODEBASE_STRUCTURE.md`](CODEBASE_STRUCTURE.md) | Annotated directory tree and naming conventions |
| [`docs/pipeline_overview.md`](docs/pipeline_overview.md) | End-to-end workflow description |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | All field names, enum values, CSV schemas |
| [`docs/hpc_guide.md`](docs/hpc_guide.md) | HPC job submission guide |
| [`viz_catalog/README.md`](viz_catalog/README.md) | Visualization Catalog & Results Browser usage |
| [`data/README.md`](data/README.md) | Expected data directory layout |
| [`src/preprocessing/README.md`](src/preprocessing/README.md) | Preprocessing pipeline details |