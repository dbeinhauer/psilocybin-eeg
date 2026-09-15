# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

EEG analysis pipeline studying psilocybin effects on neural synchrony during music
listening. 257-channel EGI GSN-HydroCel EEG, two conditions (Placebo/Psilocybin), two
music types (CLASSIC/PSYTRANCE).

Two experiments, both live:

- **`psilo_music`** — the music-listening recordings. Stimulus timing comes from a TAG
  channel carrying the music signal.
- **`assr`** — auditory steady-state response recordings. Stimulus timing comes from
  `fam+` annotations, not a TAG channel, so it uses a different alignment path
  (`stimulus_alignment.py`, `marker_shift.py`).

Analysis stages 00-07 are complete. Current work is a cleanup pass ahead of a new phase
based on ML methods.

## Environment

The working interpreter is the conda env **`psilo-music`**:

```bash
/storage/praha1/home/dbeinhauer/.conda/envs/psilo-music/bin/python
```

There is no usable `.venv` in the repo, and the package is **not installed** — `import src`
resolves only because the repo root is on `sys.path`. Scripts and notebooks bootstrap this
with `sys.path.insert(0, <repo root>)`. Run everything from the repo root.

## Common Commands

```bash
python -m pytest tests/ -v            # full suite (~1,340 tests, ~9 min)
python -m pytest tests/test_isc.py -v # one module
ruff check .                          # lint
ruff format .                         # format
```

Pipeline stages:

```bash
python scripts/run_preprocessing.py --raw_processing --plot_results
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC
python scripts/run_isc.py --music_type CLASSIC PSYTRANCE
python scripts/run_mean_variance.py --music_type CLASSIC PSYTRANCE
python scripts/run_analysis.py --analysis wavelet_power   # --analysis is required
```

`scripts/analysis_common.py` holds the shared wavelet/ISC workflow helpers;
`scripts/notebook_helpers.py` is the stable import surface used by the notebooks (and, in
practice, by three CLI scripts too).

## Architecture

### `src/`

- **`definitions/`** — single source of truth. `fields.py` (17 enums), `constants.py`
  (`ProjectPaths`), `frequency.py` (`FREQUENCY_BANDS`, wavelet grid defaults),
  `mappings.py` (channel names). Never hardcode a categorical value as a string.
- **`io/`** — `parsing.py` (filename → metadata), `loading.py` (recordings, electrode
  CSVs, the streaming wavelet cache reader), `saving.py`, `iva_store.py` (IVA component
  `.npz` products).
- **`preprocessing/`** — `channel_prep` → `filtering` → `ica` → `time_alignment` for
  `psilo_music`; `stimulus_alignment` and `marker_shift` carry the ASSR path.
  `pipeline.py` orchestrates. Uses MNE `Raw`/`Epochs`.
- **`analysis/`** — 11 modules: `isc.py`, `mean_variance.py`, `wavelet_ica.py` (time-axis
  z-scoring + IVA polarity), `wavelet_jica.py`, `iva_quality.py`,
  `iva_condition_comparison.py`, `condition_tracks.py`, `assr_trials.py`,
  `pca_polarity.py`, `data_representations.py` (`AnalysisData`), `summary.py`
  (`EEGSummarizedAnalyzer`). Input shape convention:
  `(n_subjects, n_channels, n_timepoints)`; wavelet tensors are
  `(n_subjects, n_channels, n_freqs, n_times)`.
- **`visualization/`** — 8 modules: `preprocessing_plots`, `isc_plots`,
  `mean_variance_plots`, `wavelet_plots`, `iva_quality_plots`, `iva_condition_plots`,
  `jica_plots`, `assr_filter_plots`. Seaborn for statistical plots, Matplotlib for
  topomaps. Most plot functions take an optional `save_path`, but this is not universal —
  `preprocessing_plots.py` uses `plot_path` / `save_fig` instead.
- **`filtering/`** — `DatasetFilter`, metadata-level DataFrame filtering.
- **`utils/`** — `LoggerMixin`, `setup_logging`, `LogLevel`.

There is no shared visualization helper module yet; `iva_quality_plots.py` is the de-facto
one (`save_fig`, `safe_vlim`, `participant_sort_key`, `grid_shape` are imported from it by
three siblings). All `__init__.py` are empty — import by full module path.

### Key enums (`src/definitions/fields.py`)

`ExperimentNames`, `ConditionVariants`, `MusicTypeVariants`, `FrequencyBandNames`,
`AnalysisVariants`, `SpectrumTypeVariants`, `IvaVariants`, `JicaVariants`,
`IvaComponentArrays`, `PreprocessedDataVariants`, `CoordinateSystems`,
`ExclusionCategories`, `ICLabelComponentsClasses`.

Frequency bands (`src/definitions/frequency.py`, **not** `fields.py`): delta 1-4 Hz,
theta 4-8, alpha 8-13, beta 13-30, gamma 30-70. Wavelet grid defaults: 1-50 Hz, 50 bins.

Enum values are also a **data vocabulary** — `ExclusionCategories` values appear verbatim
in `data/excluded_participants/*.csv`. Removing a member can break loading real data, so
check the data before pruning one.

### Conventions

- Analysis functions return pandas DataFrames with metadata columns (participant,
  condition, music_type, band, channel).
- All paths via `ProjectPaths`; never hardcode.
- Default to the **Placebo** condition unless told otherwise.
- Never mutate input arrays without copying; validate dimensions early.

### Notebooks

`notebooks/NN-<kebab-case-name>/`, stages `00-preprocessing` through
`07-ica-condition-comparison`. Imports go in the first cell with `# noqa: E402` after the
`sys.path.insert` block.

**A notebook is for parameter selection**, not for holding workflow logic: config, the
parameter grid being explored, a small-scale call into the same `src/` entry point the
script and PBS job use, and the resulting figure. Shared logic belongs in `src/`. Several
notebooks still violate this badly and are being slimmed.

Notebook outputs are stripped before commit (CI runs `nbstripout`).

### Plot output layout

```
plots/{NN}-{analysis-name}/{Condition}_{MusicType}/{broadband|bands}/{analysis_type}/*.png
```

Band files are prefixed with the band name. Never write directly into `broadband/` or
`bands/` — always an `{analysis_type}/` subdirectory. Note the tree on disk has drifted
from this (stage 05 is split across three directories); it is not being reorganized,
since plot outputs are treated as data.

## Code Conventions

- Python ≥3.12, Ruff lint + format (double quotes, trailing commas, ≤88 chars).
- Type hints on all function signatures.
- **Sphinx-style docstrings** (`:param:` / `:return:` / `:raises:`), not Google style.
- Say only what the name and signature do not. Skip the docstring on private one-liners.
  Reserve full docstrings for functions with real complexity, non-obvious contracts or
  shape conventions. Keep a short module docstring at the top of each file.
- Comment only genuinely hard-to-read code: deep nesting, inline tricks, non-obvious
  indexing or broadcasting. Do not restate the identifier.
- Tests cover **core logic only** — no tests for plotting, none for notebooks. Prefer
  runtime validation at the boundary where real data meets the definitions over tests
  that assert an enum holds its own literal.
- Test naming `tests/test_<module>.py`, fixtures from `conftest.py`.
- Ruff: `per-file-ignores = { "tests/*.py" = ["F401"] }`.

## Git Workflow

- Current working branch: **`fix/repository-cleanup`**. The repository default is `main`.
- Data files are gitignored (`data/`, `results/`, `plots/`, `logs/`, wavelet caches).

## HPC

PBS job templates live in `jobs/metacentrum/{NN}-{analysis-name}/` (45 of them), which is
the only cluster target. Each template runs its script inside a container via a heredoc'd
inner script. When adding one, copy an existing template's resource shape rather than inventing
a new one, and keep the job in the stage directory matching its script.

## Adding a New Analysis

1. `notebooks/NN-<name>/` — parameter exploration only
2. implement in `src/analysis/`
3. plotting in `src/visualization/`
4. thin CLI in `scripts/`
5. PBS template in `jobs/metacentrum/NN-<name>/`
6. tests in `tests/test_<module>.py` — core logic only
