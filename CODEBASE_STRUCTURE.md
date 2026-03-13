# Codebase Structure

This document describes the directory layout of the psilocybin-EEG project.
The structure separates concerns clearly, makes the analysis workflow
easier to follow, and keeps infrastructure files out of the way.

## Directory Layout

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
│   │   ├── pipeline.py            #     Top-level orchestration (DatasetHandler & DatasetPreprocessor)
│   │   ├── channel_prep.py        #     Renaming, montage, type setting, electrode exclusion
│   │   ├── filtering.py           #     Notch, bandpass, RANSAC, bad-epoch detection
│   │   ├── ica.py                 #     ICA decomposition, ICLabel, component exclusion
│   │   └── time_alignment.py      #     TAG-based cross-correlation alignment & cropping
│   │
│   ├── analysis/                  #   Feature extraction & statistical analysis
│   │   ├── __init__.py
│   │   ├── isc.py                 #     Inter-Subject Correlation computation
│   │   ├── data_representations.py #    AnalysisData container & adapters (time-domain, wavelet, etc.)
│   │   └── summary.py             #     High-level analysis orchestrator (EEGSummarizedAnalyzer)
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
│       └── logging_config.py      #     LoggerMixin, setup helpers
│
├── scripts/                       # ── CLI entry points ──────────────
│   ├── run_preprocessing.py       #   Full preprocessing pipeline
│   ├── run_time_alignment.py      #   Stimulus-based time alignment
│   ├── run_analysis.py            #   ISC & group-level analysis
│   ├── organize_plots.sh          #   Organize plot files by participant
│   └── zip_data_subset.sh         #   Create zip archives of processed data
│
├── notebooks/                     # ── Exploratory analysis ──────────
│   ├── data_analysis.ipynb
│   ├── test_notebook.ipynb
│   └── time_alignment.ipynb
│
├── tests/                         # ── Tests ──────────────────────────
│   ├── conftest.py                #   Shared fixtures
│   └── test_dataset_parsing.py
│
├── jobs/                          # ── HPC job scripts ───────────────
│   ├── metacentrum/
│   │   ├── preprocessing_job_template.pbs
│   │   ├── run_excluded_plot.pbs
│   │   └── run_full_preprocessing.pbs
│   └── umbriel/
│       └── run_dataset_preprocessing.pbs
│
└── docs/                          # ── Extended documentation ─────────
    ├── pipeline_overview.md       #   End-to-end description of the workflow
    ├── data_dictionary.md         #   All field names, enum values, CSV schemas
    └── hpc_guide.md               #   How to submit jobs on Metacentrum / Umbriel
```

## Naming Conventions

To keep the codebase consistent, the following conventions are used:

- **Enum values** are the single source of truth for categorical labels.
  Use `ConditionVariants.PLACEBO` / `ConditionVariants.PSILOCYBIN` and
  `MusicTypeVariants.CLASSICAL` / `MusicTypeVariants.PSYTRANCE` instead of
  hard-coded strings.
- **DataFrame column names** use lowercase `snake_case` (e.g. `"condition"`, `"music_type"`),
  matching the values in `SingleDataMetadata`.
- **File and directory names** use lowercase `snake_case`.
- **Class names** use `PascalCase`; **functions and methods** use `snake_case`.
- **Constants** use `UPPER_SNAKE_CASE`.
