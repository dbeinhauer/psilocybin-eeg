---
name: work-on
description: Fetch a GitHub issue, recommend model tier, create branch, and implement
user_invocable: true
args: issue_number
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

## 3. Create working branch

```bash
git checkout develop
git pull origin develop
git checkout -b claude/issue-{{ issue_number }}
```

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
