---
name: new-analysis
description: Scaffold a new analysis following the full CLAUDE.md workflow
user_invocable: true
args: name
---

Scaffold a new analysis called `{{ name }}`. Follow the "Adding a New Analysis" workflow from CLAUDE.md:

## Steps

1. **Determine the next analysis number**: Look at existing `notebooks/NN-*/` directories and pick the next available two-digit ID.

2. **Create notebook directory**: `notebooks/NN-{{ name }}/`

3. **Create exploration notebook** following the 6-cell template:
   - Cell 1 (code): Setup — project root resolver, all imports with `# noqa: E402`
   - Cell 2 (markdown): Title and description
   - Cell 3 (code): Configuration — condition (default Placebo), music types, parameters
   - Cell 4 (code): Data loading
   - Cell 5 (code): Dataset selection
   - Cell 6+ (code): Analysis steps (placeholder)

4. **Create src module**: `src/analysis/{{ name_snake }}.py` with:
   - Type-hinted function signatures
   - Google-style docstrings
   - Input dimension validation
   - Returns pandas DataFrame with metadata columns
   - Copies input data before mutation

5. **Create CLI script**: `scripts/run_{{ name_snake }}.py` with argparse, defaulting to Placebo

6. **Create HPC job template**: `jobs/metacentrum/NN-{{ name }}/run_{{ name_snake }}.pbs`

7. **Create test file**: `tests/test_{{ name_snake }}.py` with:
   - Fixtures from conftest.py where applicable
   - Happy path and edge case tests
   - Mocked file I/O

8. **Update catalog**: Add entry to `viz_catalog/catalog.yaml` with all required fields

9. **Update docs**: Update `CODEBASE_STRUCTURE.md` and `src/analysis/README.md`

10. **Verify**: Run `ruff check .` and `python -m pytest tests/test_{{ name_snake }}.py -v`

Ask the user to describe the analysis before scaffolding. Use the description to fill in meaningful content rather than generic placeholders.
