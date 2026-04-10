---
name: sync-docs
description: Update documentation files to reflect current project structure
user_invocable: true
---

Scan the project and update all documentation to match the current state:

## 1. CODEBASE_STRUCTURE.md
- Regenerate the directory tree to reflect all current `src/` modules, `scripts/`, `notebooks/`, `jobs/`, `tests/`
- Do not list files in `data/`, `results/`, `.venv/`, or other gitignored dirs

## 2. src/analysis/README.md
- List all analysis modules with one-line descriptions
- Document the data shape conventions
- Update the ISC workflow if new analysis types exist

## 3. src/preprocessing/README.md
- Verify the pipeline order matches actual code
- Update if new preprocessing steps were added

## 4. viz_catalog/README.md
- Verify run instructions work
- List all registered sketch types from `SKETCH_FUNCTIONS` in `pages/1_📋_Catalog.py`
- List all analysis groups from `catalog.yaml`

## 5. docs/
- `pipeline_overview.md` — verify pipeline stages match current code
- `data_dictionary.md` — check all enums from `fields.py` are documented
- `hpc_guide.md` — verify all PBS templates are listed

## 6. CLAUDE.md
- Check the "Architecture" section matches current modules
- Verify command examples still work
- Update sketch type count if changed

For each file: read current content, compare against actual project state, and edit only the stale sections. Do not rewrite files that are already accurate. Report what was changed.
