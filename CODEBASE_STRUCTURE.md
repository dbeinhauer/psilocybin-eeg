# Suggested Codebase Structure

This document proposes a directory layout optimised for neuroscience EEG data-analysis projects.
The recommendations are based on common patterns in MNE-Python workflows and the specific needs
of psilocybin-EEG research.

## Current Structure

```
psilocybin-eeg/
├── data/                          # Dataset assets (coordinates, exclusion lists, raw & processed data)
│   ├── coordinates/
│   ├── excluded_electrodes/
│   ├── excluded_participants/     # (gitignored)
│   ├── participant_mappings/      # (gitignored)
│   ├── raw/                       # (gitignored)
│   └── processed/                 # (gitignored)
├── src/                           # Main Python package
│   ├── data/                      #   Preprocessing & I/O
│   ├── definitions/               #   Constants, enums, mappings
│   ├── features/                  #   Analysis (ISC, data representations)
│   └── utils/                     #   Logging
├── scripts/                       # CLI entry-point scripts
├── notebooks/                     # Exploratory Jupyter notebooks
├── tests/                         # Unit tests
├── metacentrum_scripts/           # HPC job scripts (PBS)
├── umbriel_job_scripts/           # HPC job scripts (PBS)
├── main.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Proposed Structure

The suggested layout separates concerns more clearly, makes the analysis workflow
easier to follow, and keeps infrastructure files out of the way.

```
psilocybin-eeg/
│
├── README.md                      # Project overview, setup, quickstart
├── CODEBASE_STRUCTURE.md          # This file
├── pyproject.toml                 # Build & dependency config
├── requirements.txt               # Pinned dependencies (or use uv.lock)
│
├── config/                        # ── Configuration ──────────────────
│   ├── coordinates/               #   Electrode coordinate files (.sfp)
│   ├── excluded_electrodes/       #   CSVs listing boundary electrodes to drop
│   └── participant_mappings/      #   CSVs mapping condition IDs to labels
│
├── data/                          # ── Data (gitignored, except READMEs) ──
│   ├── README.md                  #   Documents expected layout & naming
│   ├── raw/                       #   Original EDF files (untouched)
│   ├── interim/                   #   Intermediate products (before_ica, etc.)
│   ├── processed/                 #   Final preprocessed data (after_ica, cropped)
│   └── excluded_participants/     #   Exclusion CSVs with explanations
│
├── src/                           # ── Source package ─────────────────
│   ├── __init__.py
│   │
│   ├── definitions/               #   Enums, constants, channel mappings
│   │   ├── __init__.py
│   │   ├── fields.py              #     All Enum definitions
│   │   ├── constants.py           #     Project paths & global settings
│   │   └── mappings.py            #     Channel-name mappings
│   │
│   ├── io/                        #   Data I/O & metadata parsing
│   │   ├── __init__.py
│   │   ├── parsing.py             #     Filename → metadata extraction
│   │   ├── loading.py             #     Loading raw / processed files
│   │   └── saving.py              #     Saving processed outputs
│   │
│   ├── preprocessing/             #   Signal preprocessing pipeline
│   │   ├── __init__.py
│   │   ├── README.md              #     Pipeline documentation
│   │   ├── pipeline.py            #     Top-level orchestration (current DatasetHandler preprocessing)
│   │   ├── channel_prep.py        #     Renaming, montage, type setting, electrode exclusion
│   │   ├── filtering.py           #     Notch, bandpass, RANSAC, bad-epoch detection
│   │   ├── ica.py                 #     ICA decomposition, ICLabel, component exclusion
│   │   └── time_alignment.py      #     TAG-based cross-correlation alignment & cropping
│   │
│   ├── analysis/                  #   Feature extraction & statistical analysis
│   │   ├── __init__.py
│   │   ├── isc.py                 #     Inter-Subject Correlation computation
│   │   ├── data_representations.py #    AnalysisData container & adapters (time-domain, wavelet, etc.)
│   │   └── summary.py             #     High-level analysis orchestrator (current overall_analysis.py)
│   │
│   ├── visualization/             #   All plotting code
│   │   ├── __init__.py
│   │   ├── preprocessing_plots.py #     Topomap, power spectrum, signal-overlap plots
│   │   └── isc_plots.py           #     ISC heatmaps, sliding-window plots, band comparison
│   │
│   ├── filtering/                 #   Metadata / DataFrame filtering
│   │   ├── __init__.py
│   │   └── dataset_filter.py      #     Filter by condition, music type, exclusion category
│   │
│   └── utils/                     #   Cross-cutting utilities
│       ├── __init__.py
│       └── logging_config.py      #     LoggerMixin, setup helpers
│
├── scripts/                       # ── CLI entry points ──────────────
│   ├── run_preprocessing.py       #   Full preprocessing pipeline
│   ├── run_time_alignment.py      #   Stimulus-based time alignment
│   └── run_analysis.py            #   ISC & group-level analysis
│
├── notebooks/                     # ── Exploratory analysis ──────────
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_qc.ipynb
│   ├── 03_isc_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/                         # ── Tests ──────────────────────────
│   ├── conftest.py                #   Shared fixtures
│   ├── test_parsing.py
│   ├── test_preprocessing.py
│   ├── test_isc.py
│   └── test_filtering.py
│
├── jobs/                          # ── HPC job scripts ───────────────
│   ├── metacentrum/
│   │   ├── preprocessing_job_template.pbs
│   │   └── run_full_preprocessing.pbs
│   └── umbriel/
│       └── run_preprocessing.pbs
│
└── docs/                          # ── Extended documentation ─────────
    ├── pipeline_overview.md       #   End-to-end description of the workflow
    ├── data_dictionary.md         #   All field names, enum values, CSV schemas
    └── hpc_guide.md               #   How to submit jobs on Metacentrum / Umbriel
```

## Key Differences from Current Layout

| Concern | Current | Proposed | Rationale |
|---------|---------|----------|-----------|
| Static config files | `data/coordinates/`, `data/excluded_electrodes/` | `config/` | Separates checked-in configuration from gitignored data artefacts |
| Intermediate data | Mixed in `data/processed/` | `data/interim/` vs `data/processed/` | Distinguishes intermediate products (before ICA) from final outputs |
| I/O logic | Inside `DatasetHandler` | Dedicated `src/io/` package | Keeps file handling separate from signal processing |
| Preprocessing | Single large `dataset_preprocessing.py` | `src/preprocessing/` sub-package | Smaller, focused modules that are easier to test and navigate |
| Plotting | Split across `dataset_plotting.py` and `isc_visualization.py` | `src/visualization/` package | Groups all plotting in one place |
| HPC scripts | Two top-level directories | `jobs/{cluster}/` | Single location for all job submission infrastructure |
| Documentation | Minimal `README.md` | `docs/` folder + per-module READMEs | Makes the project self-documenting |
| Notebooks | Ungrouped | Numbered sequence | Guides new contributors through the analysis workflow |

## Naming Conventions

To keep the codebase consistent, the following conventions are recommended:

- **Enum values** are the single source of truth for categorical labels.
  Use `ConditionVariants.PLACEBO` / `ConditionVariants.PSILOCYBIN` and
  `MusicTypeVariants.CLASSICAL` / `MusicTypeVariants.PSYTRANCE` instead of
  hard-coded strings.
- **DataFrame column names** use lowercase `snake_case` (e.g. `"condition"`, `"music_type"`),
  matching the values in `SingleDataMetadata`.
- **File and directory names** use lowercase `snake_case`.
- **Class names** use `PascalCase`; **functions and methods** use `snake_case`.
- **Constants** use `UPPER_SNAKE_CASE`.
