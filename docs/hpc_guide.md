# HPC Guide

This guide explains how to submit preprocessing and analysis jobs on the
Metacentrum and Umbriel HPC clusters.

## Directory Structure

All HPC job scripts are under `jobs/`:

```
jobs/
├── metacentrum/
│   ├── preprocessing_job_template.pbs   # Basic preprocessing
│   ├── run_excluded_plot.pbs            # IC exclusion + plotting
│   └── run_full_preprocessing.pbs       # Full pipeline (ICA + IC + plots)
└── umbriel/
    └── run_dataset_preprocessing.pbs    # Umbriel preprocessing
```

## Metacentrum

### Prerequisites

- Access to Metacentrum (https://metavo.metacentrum.cz)
- Singularity image with the project environment at
  `/storage/praha1/home/$USER/singularity-images/psilo_music.sif`
- Project data at `/storage/praha1/home/$USER/psilocybin-eeg/`

### Submitting Jobs

```bash
# Basic preprocessing
qsub jobs/metacentrum/preprocessing_job_template.pbs

# Full preprocessing with IC metadata and plots
qsub jobs/metacentrum/run_full_preprocessing.pbs

# Just IC exclusion metadata + plots
qsub jobs/metacentrum/run_excluded_plot.pbs
```

### Job Resources

| Script | Walltime | CPUs | Memory | Scratch |
|--------|----------|------|--------|---------|
| `preprocessing_job_template.pbs` | 14h | 4 | 100 GB | 100 GB |
| `run_excluded_plot.pbs` | 8h | 4 | 100 GB | 100 GB |
| `run_full_preprocessing.pbs` | 24h | 4 | 100 GB | 100 GB |

### How the Scripts Work

1. The PBS script sets up a scratch directory
2. Creates a runner script (`run_inside.sh`) that activates the virtualenv
   and calls the Python entry point
3. Runs the script inside a Singularity container, binding the project
   directory as `/mnt`
4. Cleans up scratch on exit

## Umbriel

### Prerequisites

- Access to the Umbriel cluster
- Singularity-compatible project directory at `/home/dbeinhauer/psilocybin-eeg`

### Submitting Jobs

```bash
qsub jobs/umbriel/run_dataset_preprocessing.pbs
```

### Job Resources

| Script | Walltime | Nodes | PPN | Queue |
|--------|----------|-------|-----|-------|
| `run_dataset_preprocessing.pbs` | 24h | 1 | 64 | qprodu |

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
