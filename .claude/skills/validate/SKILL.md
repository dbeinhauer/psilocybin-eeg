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

## 3. Enum/data vocabulary sync
Several enums double as a vocabulary used inside stored data, so a member that looks
unused in code may still be live:
- Check every value in `data/excluded_participants/*.csv` resolves to an `ExclusionCategories` member
- Check every experiment directory under `data/raw/` resolves to an `ExperimentNames` member
- Flag mismatches in **both** directions, and never propose removing an enum member that appears in data

## 4. Test coverage
- For each `src/<pkg>/<mod>.py` (excluding `__init__.py` and everything in `src/visualization/`), check a test file exists
- Naming is usually `tests/test_<mod>.py`, but a few differ (`test_dataset_parsing.py` → `io/parsing.py`, `test_filtering_preprocessing.py` → `preprocessing/filtering.py`); match by module under test, not filename alone
- Plotting modules are deliberately untested — do not report them as gaps

## 5. Notebook template
- For each `.ipynb` in `notebooks/NN-*/`, verify:
  - First cell is a code cell (setup with imports)
  - Contains `sys.path` setup
  - Has `# noqa: E402` markers after path setup
  - No cell holds workflow logic that belongs in `src/` — flag any code cell over ~80 lines
  - Outputs are stripped (no `outputs` on any code cell)

## 6. Script-job parity
- For each `scripts/run_*.py`, check for a matching `.pbs` file in `jobs/metacentrum/`
- Known exception: `run_stimulus_shift_calibration.py` is a one-off tool with no job
- Also check the reverse — every script named in a `.pbs` file still exists

## 7. Doc freshness
- Check `CLAUDE.md` lists the current `src/` modules, and that its code fences are balanced (an odd count means a section is swallowed into a code block)
- Check `README.md`, `src/analysis/README.md` and `src/preprocessing/README.md` are up to date
- Grep the repo for references to paths that no longer exist

Report as a markdown table: Check | Status | Details
