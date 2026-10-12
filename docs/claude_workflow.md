# Claude Code Workflow

How Claude Code is used for development on this project.

## Overview

Development happens **locally and interactively** in the Claude Code CLI — describe the
task, iterate on it in the session, review the diff, commit. There is no issue-driven
automation and no API-powered GitHub Actions. GitHub is used for tracking and review only.

## Prerequisites

- **Claude Code** installed locally (`npm install -g @anthropic-ai/claude-code`)
- **`gh` CLI** for anything that touches issues or PRs

## Environment

Run Claude Code from the repository root, so `import src` resolves. The working
interpreter is the conda env `psilo-music`; there is no usable `.venv` in the repo and the
package is not installed.

## Skills

Three custom skills live in `.claude/skills/`:

### `/validate` — Consistency checker

Runs seven checks and reports a pass/fail table:

1. **Enum sync** — no hardcoded categorical strings (use enums from `fields.py`)
2. **Path sync** — no hardcoded paths (use `ProjectPaths`)
3. **Enum/data vocabulary sync** — values stored in `data/` still resolve to enum members,
   in both directions
4. **Test coverage** — every core-logic module has tests (plotting is deliberately exempt)
5. **Notebook template** — notebooks follow the template, hold no workflow logic, and have
   stripped outputs
6. **Script-job parity** — scripts and PBS templates still line up, both ways
7. **Doc freshness** — `CLAUDE.md` and `README.md` match the code, and fences balance

Worth running after any significant change.

### `/new-analysis <name>` — Scaffold a new analysis

Creates notebook, `src/analysis/` module, plotting module, thin CLI script, PBS template,
tests and doc updates, following the "Adding a New Analysis" workflow in `CLAUDE.md`.

### `/sync-docs` — Update stale documentation

Compares `CLAUDE.md`, `README.md`, the two `src/*/README.md` files and `docs/` against the
actual project state and edits only what has gone stale.

## Conventions worth stating to Claude

These are in `CLAUDE.md`, but they are the ones most often gotten wrong:

- **Sphinx docstrings**, kept short. Say only what the name and signature do not.
- **Tests for core logic only** — no plotting tests, no notebook tests, nothing that
  restates an argparse default or an enum's own literal.
- **Notebooks select parameters**; shared logic belongs in `src/`.
- **Enum members can be a data vocabulary** — check `data/` before removing one.
- Plot outputs, `results/` and `data/` are treated as data and are not reorganized.

## Issue Templates

`.github/ISSUE_TEMPLATE/` holds `simple-task.yml` and `complex-task.yml`, left over from
an earlier issue-driven flow. They are still usable for tracking, but nothing consumes
their category labels automatically any more.

## File Inventory

```
.github/
├── workflows/
│   └── code-quality.yml       # Black, Ruff, nbstripout on PRs
└── ISSUE_TEMPLATE/
    ├── simple-task.yml
    └── complex-task.yml

.claude/
└── skills/
    ├── validate/SKILL.md      # /validate     — consistency checks
    ├── new-analysis/SKILL.md  # /new-analysis — scaffold
    └── sync-docs/SKILL.md     # /sync-docs    — update docs
```
