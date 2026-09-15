---
name: sync-docs
description: Update documentation files to reflect current project structure
user-invocable: true
---

Scan the project and update all documentation to match the current state.

`CLAUDE.md` and `README.md` are the two top-level documents; everything else is a topic
file. Do not create new overview documents — the previous set drifted apart precisely
because the same module inventory lived in four places.

## 1. CLAUDE.md
- Check the "Architecture" section lists the current `src/` modules, per package
- Verify the command examples still run (correct flags, scripts that exist)
- **Check the code fences balance** — an odd count means a heading is being swallowed into a code block
- Verify the stated working branch and environment are still right

## 2. README.md
- Verify the repository tree, the analysis-stage table and the documentation table
- Check the test count and runtime are roughly current
- Check every linked path exists

## 3. src/analysis/README.md
- List all analysis modules with one-line descriptions
- Document the data shape conventions
- Note which analyses are current and which are finished

## 4. src/preprocessing/README.md
- Verify the pipeline order matches actual code, for both the `psilo_music` and `assr` paths

## 5. docs/
- `data_dictionary.md` — check all enums from `fields.py` are documented, including which
  ones double as a vocabulary inside stored data
- `hpc_guide.md` — verify the PBS templates listed match `jobs/metacentrum/`
- `preprocessing_steps.md` — verify against `src/preprocessing/`
- `claude_workflow.md` — verify it describes the skills that actually exist in `.claude/skills/`

## 6. Dangling references
- Grep for links and paths that no longer exist anywhere in the repo, and remove them
  rather than leaving them to rot

For each file: read current content, compare against actual project state, and edit only the stale sections. Do not rewrite files that are already accurate. Report what was changed.
