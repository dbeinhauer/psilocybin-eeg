# Claude Code Workflow

This document describes how Claude Code is used for development on this project.

## Overview

All development is done locally via the Claude Code CLI on a Claude Pro subscription.
GitHub is used for issue tracking, pull requests, and code review — no API-powered
GitHub Actions are involved.

## Prerequisites

- **Claude Pro subscription** ($20/month) — covers Claude Code CLI usage
- **`gh` CLI** — GitHub's official CLI, used for issue fetching and PR creation
- **Claude Code** installed locally (`npm install -g @anthropic-ai/claude-code`)

## Workflow

```
1. Create issue on GitHub (using issue template)
2. In terminal:  /work-on <issue-number>
3. Claude reads issue, checks model, creates branch, implements, tests, opens PR
4. Review PR on GitHub
5. Optionally: /validate to run consistency checks
6. Merge on GitHub
```

## Skills

Four custom skills are available in `.claude/skills/`:

### `/work-on <issue-number>` — Main entry point

Fetches the GitHub issue, determines the right model tier based on the issue
category, creates a `claude/issue-N` branch, implements the work, and opens a PR.

**Model selection logic:**
- Sonnet (cheaper, faster): `fix/*`, `docs/*`, `test/*`, `refactor/simple`, `catalog/*`
- Opus (smarter, costlier): `feature/*`, `refactor/complex`

If the wrong model is active, the skill stops and tells you to switch:
```
/model sonnet    # or /model opus
/work-on 42      # re-run
```

### `/validate` — Consistency checker

Runs 8 checks across the codebase:

1. **Enum sync** — no hardcoded categorical strings (use enums from `fields.py`)
2. **Path sync** — no hardcoded paths (use `ProjectPaths`)
3. **Test coverage** — every `src/` module has a test file
4. **Import isolation** — `viz_catalog/` has zero imports from `src/`
5. **Catalog sync** — every analysis module has a `catalog.yaml` entry
6. **Notebook template** — notebooks follow the 6-cell template
7. **Script-job parity** — every CLI script has an HPC job template
8. **Doc freshness** — documentation matches current project structure

Reports results as a pass/fail table. Run this after any significant change.

Recommended model: **Sonnet** (checklist execution, no creativity needed).

### `/new-analysis <name>` — Scaffold a new analysis

Creates the full analysis workflow in one command:
- `notebooks/NN-<name>/` with exploration notebook
- `src/analysis/<name>.py` module
- `scripts/run_<name>.py` CLI script
- `jobs/metacentrum/NN-<name>/` HPC job template
- `tests/test_<name>.py` test file
- `viz_catalog/catalog.yaml` entry
- Documentation updates

Recommended model: **Sonnet** (follows rigid template from CLAUDE.md).

### `/sync-docs` — Update stale documentation

Scans the project and updates:
- `CODEBASE_STRUCTURE.md`
- `src/analysis/README.md`, `src/preprocessing/README.md`
- `viz_catalog/README.md`
- `docs/` files (pipeline overview, data dictionary, HPC guide)
- `CLAUDE.md` (architecture section, sketch type count)

Only edits sections that are actually stale. Reports what was changed.

Recommended model: **Sonnet** (read + compare + update text).

## Issue Templates

Two templates in `.github/ISSUE_TEMPLATE/`:

### Simple task (`simple-task.yml`)

For: bug fixes, lint fixes, typos, doc updates, adding tests, simple refactors,
catalog updates.

Categories: `fix/bug`, `fix/lint`, `fix/typo`, `docs/update`, `docs/sync`,
`test/add`, `test/fix`, `refactor/simple`, `catalog/update`

Model: **Sonnet** — est. ~$0 (Pro subscription), 10–20 messages per task.

### Complex task (`complex-task.yml`)

For: new analyses, new visualizations, new pipeline steps, infrastructure changes,
multi-module refactors.

Categories: `feature/analysis`, `feature/viz`, `feature/pipeline`, `feature/infra`,
`refactor/complex`

Model: **Opus** — est. ~$0 (Pro subscription), 30–60 messages per task.

## Label Catalog

| Label | Model | Color | Typical use |
|---|---|---|---|
| `fix/bug` | Sonnet | red | Fix broken ISC calculation, test failure |
| `fix/lint` | Sonnet | red | Fix ruff/formatting violations |
| `fix/typo` | Sonnet | red | Typo in code, docs, comments |
| `docs/update` | Sonnet | blue | Update README, docstrings |
| `docs/sync` | Sonnet | blue | Sync docs with current structure |
| `test/add` | Sonnet | light blue | Add missing tests |
| `test/fix` | Sonnet | light blue | Fix broken test |
| `refactor/simple` | Sonnet | yellow | Rename, extract, move (single module) |
| `catalog/update` | Sonnet | light blue | Update catalog YAML or sketch |
| `feature/analysis` | Opus | teal | New analysis (full workflow) |
| `feature/viz` | Opus | teal | New plot type, explorer feature |
| `feature/pipeline` | Opus | teal | New preprocessing step, CLI script |
| `feature/infra` | Opus | teal | HPC job template, CI/CD |
| `refactor/complex` | Opus | yellow | Multi-module restructuring |

## Weekly Capacity (Pro subscription, 2.5 days/week)

| Model | Per 5-hour window | Per week (~6-7 windows) |
|---|---|---|
| Opus | ~1 complex task | 5–7 complex tasks max |
| Sonnet | ~8–10 simple tasks | 40–60 simple tasks max |

**Typical mixed week:** 2 features (Opus) + 5–8 simple tasks (Sonnet) + validate/sync runs.

If you hit rate limits, switch to the other model or wait for the 5-hour window to reset.

## Branch Convention

All Claude-created branches follow the pattern `claude/issue-<number>`.
The `/work-on` skill checks for other `claude/*` branches before editing shared files
to avoid merge conflicts.

## File Inventory

```
.github/
├── workflows/
│   └── code-quality.yml              # Existing linter (Black, Ruff, nbstripout)
└── ISSUE_TEMPLATE/
    ├── simple-task.yml               # fix/docs/test/refactor/catalog
    └── complex-task.yml              # feature/refactor-complex

.claude/
└── skills/
    ├── work-on.md                    # /work-on <N> — main entry point
    ├── validate.md                   # /validate — consistency checks
    ├── new-analysis.md               # /new-analysis <name> — scaffold
    └── sync-docs.md                  # /sync-docs — update docs
```
