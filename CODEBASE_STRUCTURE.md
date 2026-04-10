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
│   │   ├── README.md              #     Package documentation and function reference
│   │   ├── isc.py                 #     Inter-Subject Correlation computation (Pearson, Spearman, mean-field)
│   │   ├── mean_variance.py       #     Intersubject mean-variance synchrony analysis
│   │   ├── data_representations.py #    AnalysisData container & adapters (time-domain, wavelet, etc.)
│   │   ├── results_store.py       #     CSV export for analysis results (Interactive Explorer)
│   │   └── summary.py             #     High-level analysis orchestrator (EEGSummarizedAnalyzer)
│   │
│   ├── visualization/             #   All plotting code
│   │   ├── __init__.py
│   │   ├── preprocessing_plots.py #     Topomap, power spectrum, signal-overlap plots
│   │   ├── isc_plots.py           #     ISC histograms, pairwise heatmaps, multi-scale sliding-window plots
│   │   └── mean_variance_plots.py #     Mean-variance synchrony plots
│   │
│   ├── filtering/                 #   Metadata / DataFrame filtering
│   │   ├── __init__.py
│   │   └── dataset_filter.py      #     Filter by condition, music type, exclusion category
│   │
│   └── utils/                     #   Cross-cutting utilities
│       └── logging_config.py      #     LoggerMixin, setup helpers
│
├── scripts/                       # ── CLI entry points ──────────────
│   ├── analysis_common.py         #   Shared helpers: load_analyzers, analyzers_to_datasets
│   ├── run_preprocessing.py       #   Full preprocessing pipeline
│   ├── run_time_alignment.py      #   Stimulus-based time alignment
│   ├── run_isc.py                 #   ISC analysis (broadband, per-band, mean-field)
│   ├── run_mean_variance.py       #   Mean-variance synchrony analysis
│   ├── run_analysis.py            #   Legacy ISC & group-level analysis entry point
│   ├── organize_plots.sh          #   Organize plot files by participant
│   └── zip_data_subset.sh         #   Create zip archives of processed data
│
├── notebooks/                     # ── Exploratory analysis ──────────
│   ├── 00-preprocessing/          #   Preprocessing inspection and time alignment
│   ├── 01-raw-mean-variance-analysis/  #   Mean-variance synchrony (broadband + per-band)
│   ├── 02-isc-broadband-analysis/ #   ISC analysis (broadband + per-band)
│   ├── 03-wavelet-analysis/       #   Wavelet power and phase exploration
│   └── [legacy notebooks]         #   data_analysis.ipynb, time_alignment.ipynb, etc.
│
├── tests/                         # ── Tests ──────────────────────────
│   ├── conftest.py                #   Shared fixtures
│   ├── test_dataset_parsing.py
│   ├── test_isc.py
│   ├── test_mean_variance.py
│   ├── test_visualization.py
│   └── [further test modules]     #   One per src/ module
│
├── jobs/                          # ── HPC job scripts ───────────────
│   └── metacentrum/
│       ├── 00-preprocessing/      #   Preprocessing job scripts
│       ├── 01-raw-mean-variance-analysis/  #   Mean-variance job script
│       ├── 02-isc-broadband-analysis/      #   ISC job script
│       ├── 03-wavelet-analysis/   #   Wavelet power/phase job scripts
│       ├── preprocessing_job_template.pbs  #   Legacy flat scripts
│       ├── run_excluded_plot.pbs
│       ├── run_full_preprocessing.pbs
│       ├── run_wavelet_analysis.pbs
│       └── store_wavelet_data.pbs
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
