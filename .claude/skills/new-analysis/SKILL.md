---
name: new-analysis
description: Scaffold a new analysis following the full CLAUDE.md workflow
user-invocable: true
argument-hint: "[name]"
---

Scaffold a new analysis called `{{ name }}`. Follow the "Adding a New Analysis" workflow from CLAUDE.md:

## Steps

1. **Determine the next analysis number**: Look at existing `notebooks/NN-*/` directories and pick the next available two-digit ID. Use the same number for the notebook, plot and job directories.

2. **Create notebook directory**: `notebooks/NN-{{ name }}/`

3. **Create exploration notebook** following the 6-cell template:
   - Cell 1 (code): Setup — project root resolver, all imports with `# noqa: E402`
   - Cell 2 (markdown): Title and description
   - Cell 3 (code): Configuration — condition (default Placebo), music types, parameters
   - Cell 4 (code): Data loading
   - Cell 5 (code): Dataset selection
   - Cell 6+ (code): Analysis steps (placeholder)

   The notebook selects **parameters**; it does not hold workflow logic. Every cell should
   be short and call into `src/`. If a cell grows past ~80 lines, that code belongs in a
   module.

4. **Create src module**: `src/analysis/{{ name_snake }}.py` with:
   - Type-hinted function signatures
   - **Sphinx-style** docstrings (`:param:` / `:return:` / `:raises:`), only where the
     name and signature do not already say it; no docstrings on private one-liners
   - Input dimension validation
   - Returns pandas DataFrame with metadata columns
   - Copies input data before mutation

5. **Create plotting module** (if the analysis needs figures):
   `src/visualization/{{ name_snake }}_plots.py`. Reuse the shared helpers rather than
   copying them, and close every figure you create.

6. **Create CLI script**: `scripts/run_{{ name_snake }}.py` — argparse wiring and
   orchestration only, defaulting to Placebo. Numerical work goes in `src/`, not here.

7. **Create HPC job template**: `jobs/metacentrum/NN-{{ name }}/run_{{ name_snake }}.pbs`.
   Copy an existing template's structure and resource shape rather than inventing one.

8. **Create test file**: `tests/test_{{ name_snake }}.py` covering the **core logic**:
   - Fixtures from conftest.py where applicable
   - Happy path and edge cases, with meaningful numerical assertions
   - No plotting tests, no notebook tests, no tests that restate argparse defaults

9. **Update docs**: `CLAUDE.md` (module lists) and `src/analysis/README.md`.

10. **Verify**: Run `ruff check .` and `python -m pytest tests/test_{{ name_snake }}.py -v`

Ask the user to describe the analysis before scaffolding. Use the description to fill in meaningful content rather than generic placeholders.
