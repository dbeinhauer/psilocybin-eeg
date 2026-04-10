---
name: validate
description: Run consistency checks across the codebase to find convention violations
user-invocable: true
---

Run the following consistency checks and report results as a table:

## 1. Enum sync
Search for hardcoded categorical strings that should use enums from `src/definitions/fields.py`:
- Grep for `"Placebo"`, `"Psilocybin"`, `"CLASSIC"`, `"PSYTRANCE"`, `"delta"`, `"theta"`, `"alpha"`, `"beta"`, `"gamma"` in `src/`, `scripts/`, `notebooks/`, `tests/`
- Exclude: comments, docstrings, filenames, plot labels, `fields.py` itself, and test assertions
- Flag any that should use `ConditionVariants`, `MusicTypeVariants`, or `FrequencyBandNames`

## 2. Path sync
- Grep for hardcoded paths (string literals containing `data/`, `plots/`, `results/`, `config/`) in `src/`, `scripts/`
- Verify they use `ProjectPaths` from `src/definitions/constants.py`

## 3. Test coverage
- For each `src/<pkg>/<mod>.py` (excluding `__init__.py`), check `tests/test_<mod>.py` exists
- List any missing test files

## 4. Import isolation
- Grep `viz_catalog/` for any `from src` or `import src` statements
- Should find zero matches

## 5. Catalog sync
- List all analysis modules in `src/analysis/` (excluding `__init__.py`, `data_representations.py`, `results_store.py`, `summary.py`)
- Check each has a corresponding entry in `viz_catalog/catalog.yaml`

## 6. Notebook template
- For each `.ipynb` in `notebooks/NN-*/`, verify:
  - First cell is a code cell (setup with imports)
  - Contains `sys.path` setup
  - Has `# noqa: E402` markers after path setup

## 7. Script-job parity
- For each `scripts/run_*.py`, check for a matching `.pbs` file in `jobs/metacentrum/`

## 8. Doc freshness
- Check if `CODEBASE_STRUCTURE.md` lists all current `src/` modules
- Check if `src/analysis/README.md` and `src/preprocessing/README.md` are up to date

Report as a markdown table: Check | Status | Details
