---
name: work-on
description: Fetch a GitHub issue, recommend model tier, create branch, and implement
user-invocable: true
argument-hint: "[issue-number]"
---

Work on GitHub issue #{{ issue_number }}. Follow these steps:

## 1. Fetch the issue

Run `gh issue view {{ issue_number }}` to get the title, body, and labels.

## 2. Determine model tier

Parse the issue body or labels for the category prefix:

- **Sonnet tasks** (simple, mechanical): `fix/*`, `docs/*`, `test/*`, `refactor/simple`, `catalog/*`
- **Opus tasks** (creative, multi-file): `feature/*`, `refactor/complex`

If the current model does not match the recommended tier, tell the user:

> **This is a [simple/complex] task. Recommended model: [Sonnet/Opus].**
> Run `/model [sonnet/opus]` to switch, then re-run `/work-on {{ issue_number }}`.

Then STOP — do not proceed with the wrong model.

If the model matches (or the user has already been warned), continue.

## 3. Create a git worktree

The worktree and container are set up automatically by `scripts/claude-orchestrator.sh`.
When this skill runs, the worktree is already active at `/workspace` and shared directories
are mounted at `/data`, `/plots`, `/results`, `/results_db`.
Skip directly to step 4.

If running manually (outside the orchestrator), create the worktree from the **root of the
main checkout**:

```bash
git fetch origin develop
git worktree add .worktrees/issue-{{ issue_number }} -b claude/issue-{{ issue_number }} origin/develop
```

No symlinking needed — data/plots/results are bind-mounted by the container.

**STOP here.** Do not implement in the main checkout. All implementation happens inside the
worktree session (container or otherwise).

---

*The steps below apply only when this skill is invoked from inside the worktree.*

## 4. Check parallel work

Run `git branch -r --list 'origin/claude/*'` to see other in-flight branches.
If any exist, check if they touch shared files (fields.py, constants.py, catalog.yaml,
CLAUDE.md, CODEBASE_STRUCTURE.md, conftest.py) before editing those files.

## 5. Implement

1. Read CLAUDE.md thoroughly
2. Implement what the issue asks for
3. Follow the "Adding a New Analysis" workflow if applicable
4. Run: `ruff check . && ruff format . && python -m pytest tests/ -v`
5. Commit with conventional commit messages referencing #{{ issue_number }}

## 6. Create PR

Push the branch and create a PR to develop using `gh pr create`.
Include the issue reference (Closes #{{ issue_number }}) in the PR body.

## 7. Clean up (after PR is merged)

From the **main checkout**, remove the worktree:

```bash
git worktree remove .worktrees/issue-{{ issue_number }}
git branch -d claude/issue-{{ issue_number }}
```
