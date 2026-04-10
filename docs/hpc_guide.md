# HPC Guide

This guide explains how to submit preprocessing and analysis jobs on the
Metacentrum and Umbriel HPC clusters.

## Directory Structure

All HPC job scripts are under `jobs/metacentrum/`:

```
jobs/
└── metacentrum/
    ├── 00-preprocessing/
    │   ├── preprocessing_job_template.pbs   # Basic preprocessing
    │   ├── run_excluded_plot.pbs            # IC exclusion + plotting
    │   ├── run_full_preprocessing.pbs       # Full pipeline (ICA + IC + plots)
    │   └── run_time_alignment.pbs           # Time alignment
    ├── 01-raw-mean-variance-analysis/
    │   └── run_mean_variance.pbs            # Mean-variance synchrony analysis
    ├── 02-isc-broadband-analysis/
    │   └── run_isc.pbs                      # ISC analysis
    ├── 03-wavelet-analysis/
    │   ├── run_wavelet_phase.pbs            # Wavelet phase analysis
    │   ├── run_wavelet_power.pbs            # Wavelet power analysis
    │   └── store_wavelets.pbs              # Wavelet data storage
    ├── preprocessing_job_template.pbs   # Legacy flat scripts (use NN-* versions above)
    ├── run_excluded_plot.pbs
    ├── run_full_preprocessing.pbs
    ├── run_wavelet_analysis.pbs
    └── store_wavelet_data.pbs
```

## Metacentrum

### Prerequisites

- Access to Metacentrum (https://metavo.metacentrum.cz)
- Singularity image with the project environment at
  `/storage/praha1/home/$USER/singularity-images/psilo_music.sif`
- Project data at `/storage/praha1/home/$USER/psilocybin-eeg/`

### Submitting Jobs

```bash
# Full preprocessing with IC metadata and plots
qsub jobs/metacentrum/00-preprocessing/run_full_preprocessing.pbs

# Just IC exclusion metadata + plots
qsub jobs/metacentrum/00-preprocessing/run_excluded_plot.pbs

# Time alignment
qsub jobs/metacentrum/00-preprocessing/run_time_alignment.pbs

# Mean-variance synchrony analysis
qsub jobs/metacentrum/01-raw-mean-variance-analysis/run_mean_variance.pbs

# ISC analysis
qsub jobs/metacentrum/02-isc-broadband-analysis/run_isc.pbs

# Wavelet analysis
qsub jobs/metacentrum/03-wavelet-analysis/run_wavelet_power.pbs
qsub jobs/metacentrum/03-wavelet-analysis/run_wavelet_phase.pbs
```

### Job Resources

| Script | Walltime | CPUs | Memory | Scratch |
|--------|----------|------|--------|---------|
| `00-preprocessing/preprocessing_job_template.pbs` | 14h | 4 | 100 GB | 100 GB |
| `00-preprocessing/run_excluded_plot.pbs` | 8h | 4 | 100 GB | 100 GB |
| `00-preprocessing/run_full_preprocessing.pbs` | 24h | 4 | 100 GB | 100 GB |

### How the Scripts Work

1. The PBS script sets up a scratch directory
2. Creates a runner script (`run_inside.sh`) that activates the virtualenv
   and calls the Python entry point
3. Runs the script inside a Singularity container, binding the project
   directory as `/mnt`
4. Cleans up scratch on exit

## Data Transfer

For transferring intermediate results, use the `scripts/zip_data_subset.sh`
script to create zip archives from processed data:

```bash
qsub -v input=PSI018_EEGA_MUSIC_CLASSIC_EC_20171124_014218 scripts/zip_data_subset.sh
```

## Troubleshooting

- **SCRATCHDIR not set**: Ensure PBS is properly configured and scratch
  allocation is available.
- **Python script failed**: Check the job output log for error messages.
  Common issues include missing data files or incorrect paths.
- **Singularity errors**: Verify the `.sif` image exists at the expected
  path and contains the correct Python environment.
