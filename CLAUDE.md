# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EEG analysis pipeline studying psilocybin effects on neural synchrony during music listening. 257-channel EGI GSN-HydroCel EEG, two conditions (Placebo/Psilocybin), two music types (CLASSIC/PSYTRANCE). Core metric: Inter-Subject Correlation (ISC).

## Claude Code Skills

Custom skills for this project (see [`docs/claude_workflow.md`](docs/claude_workflow.md) for full details):

- **`/work-on <issue-number>`** — Fetch GitHub issue, check model tier, create branch, implement, open PR
- **`/validate`** — Run 8 consistency checks (enum sync, test coverage, import isolation, etc.)
- **`/new-analysis <name>`** — Scaffold full analysis workflow (notebook → src → script → job → test → catalog)
- **`/sync-docs`** — Update all documentation to match current project structure

Model selection: `fix/*`, `docs/*`, `test/*`, `catalog/*`, `refactor/simple` → Sonnet. `feature/*`, `refactor/complex` → Opus.

## Common Commands

```bash
# Run full test suite
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_isc.py -v

# Lint
ruff check .

# Auto-format
ruff format .

# Install (editable)
pip install -e .

# Run pipeline stages
python scripts/run_preprocessing.py --raw_processing --plot_results
python scripts/run_time_alignment.py --condition Placebo --music_type CLASSIC
python scripts/run_isc.py --music_type CLASSIC PSYTRANCE
python scripts/run_mean_variance.py --music_type CLASSIC PSYTRANCE

# scripts/analysis_common.py provides shared helpers (load_analyzers, analyzers_to_datasets)
# used by both run_isc.py and run_mean_variance.py

# Run visualization catalog (Streamlit app, independent of src/)
uv run --with "streamlit>=1.32.0" --with "pyyaml>=6.0" --with "numpy>=1.24.0" --with "matplotlib>=3.7.0" \
    streamlit run viz_catalog/app.py
```

## Architecture

### Source Package (`src/`)

- **`definitions/`** — Single source of truth for enums (`fields.py`), path constants (`constants.py` → `ProjectPaths`), and channel mappings (`mappings.py`). Never hardcode categorical values as strings; always use enums.
- **`io/`** — Filename parsing (`parsing.py`), data loading (`loading.py`), saving (`saving.py`). All file I/O routes through here.
- **`preprocessing/`** — Pipeline order: **channel_prep → filtering → ica → time_alignment**. Uses MNE-Python `Raw`/`Epochs` objects.
- **`analysis/`** — ISC computation (`isc.py`), intersubject mean-variance synchrony (`mean_variance.py`), data containers (`data_representations.py` → `AnalysisData`), results CSV export (`results_store.py`), high-level orchestrator (`summary.py` → `EEGSummarizedAnalyzer`). Input shape convention: `(n_subjects, n_channels, n_timepoints)`.
- **`visualization/`** — Seaborn for statistical plots, Matplotlib for EEG topomaps. All plot functions accept an optional `save_path` parameter.
- **`filtering/`** — `DatasetFilter` for metadata-level DataFrame filtering.
- **`utils/`** — `LoggerMixin` for class-level logging.

### Key Patterns

- **Enums everywhere**: `ConditionVariants`, `MusicTypeVariants`, `FrequencyBandNames`, `AnalysisVariants`, `PreprocessedDataVariants`, `CoordinateSystems` from `src/definitions/fields.py`. Frequency bands: delta (1-4 Hz), theta (4-8), alpha (8-13), beta (13-30), gamma (30-70).
- **ProjectPaths**: All paths via `src/definitions/constants.py`, never hardcoded.
- **Analysis results as DataFrames**: Analysis functions return pandas DataFrames with metadata columns (participant, condition, music_type, band, channel).
- **Default to Placebo condition** in all analyses, notebooks, and scripts unless explicitly told otherwise.

### Notebooks (`notebooks/`)

Numbered subdirectories: `NN-<kebab-case-name>/` (e.g., `00-preprocessing/`, `01-raw-mean-variance-analysis/`, `02-isc-broadband-analysis/`, `03-wavelet-analysis/`). Each notebook follows a 6-cell template: setup (imports + path resolver) → title → config → data loading → dataset selection → one cell per analysis step. All imports in the first cell with `# noqa: E402` after the `sys.path.insert` block.

### Plot Output Structure

CLI scripts write plots to `plots/` following this canonical layout (used by the Results Browser):

```
plots/
└── {NN}-{analysis-name}/
    └── {Condition}_{MusicType}/       # e.g. Placebo_CLASSIC
        ├── broadband/
        │   └── {analysis_type}/      # e.g. loo_isc, pairwise_isc, sliding_window
        │       └── *.png
        └── bands/
            └── {analysis_type}/
                └── {band}_*.png      # band name is the first token of the filename
```

Notebooks save to `notebooks/{NN}-{name}/plots/{broadband|bands}/{analysis_type}/`. Never place files directly in `broadband/` or `bands/` — always use an `{analysis_type}/` subdirectory.

### Visualization Catalog (`viz_catalog/`)

Self-contained Streamlit app — **no imports from `src/`**. Content defined in `catalog.yaml`. Three pages: Analysis Catalog, Results Browser, Interactive Explorer. Adding analyses requires only YAML changes; new sketch types go in `pages/1_📋_Catalog.py` registered in `SKETCH_FUNCTIONS`.

## Code Conventions

- Python ≥3.12, Ruff linter + Black formatting (double quotes, trailing commas, ≤88 chars)
- Type hints on all function signatures
- Google-style docstrings for public functions
- Test file naming: `tests/test_<module>.py`, uses pytest fixtures from `conftest.py`
- Ruff config: `per-file-ignores = { "tests/*.py" = ["F401"] }`
- Never mutate input data without copying first
- Validate input dimensions early in analysis functions

## Git Workflow

- Main branch: `develop`
- GitHub notebook links use `develop` branch (see `viz_catalog/` `GITHUB_BASE` constant)
- Data files are gitignored (`data/`, `results/`, `.venv/`, `.worktrees/`)

### Parallel work with git worktrees

Each issue gets an isolated worktree under `.worktrees/issue-N/` so multiple issues can be worked on simultaneously without branch-switching.

**Setup** (handled automatically by `/work-on`):
```bash
git worktree add .worktrees/issue-N -b claude/issue-N origin/develop
cd .worktrees/issue-N
python -m venv .venv && .venv/bin/pip install -e .
ln -s ../../data data && ln -s ../../plots plots
ln -s ../../results results && ln -s ../../results_db results_db
```

**Starting a session**: open a new terminal, `cd .worktrees/issue-N`, then run `claude`.

**Cleanup** (after PR is merged):
```bash
git worktree remove .worktrees/issue-N
git branch -d claude/issue-N
```

## Adding a New Analysis

1. Create `notebooks/NN-<name>/` with exploration notebook(s)
2. Implement in `src/analysis/`
3. Create CLI script in `scripts/`
4. Add HPC job template in `jobs/metacentrum/NN-<name>/`
5. Write tests in `tests/test_<module>.py`
6. Update `viz_catalog/catalog.yaml` for the catalog app
