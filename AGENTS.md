# Agent Workflow Contract

This repository enforces a branch-first workflow for all feature work by any agent.

## Mandatory Rules

1. Never commit directly to `main`.
2. Never push directly to `main`.
3. New work must start from a dedicated branch:
   - `feature/<topic>`
   - `fix/<topic>`
   - `chore/<topic>`
4. Changes must be committed in small, readable batches (single script or single logical unit).
5. Each commit must be validated before commit (lint/smoke tests relevant to touched files).
6. Open a Pull Request to `main` after pushing branch.
7. Merge to `main` is allowed only after required checks and review approval.
8. Any behavior/config/CLI/output-path change must update `README.md` (and `USAGE.txt` when command usage changes), or explicitly justify `N/A` in PR.
9. Any feature/fix/chore change must update `CHANGELOG.md` (or explicitly justify `N/A` in PR).

## README-First Analysis Rule

10. Before substantial analysis, architecture design, or cross-file implementation, read `README.md` first (at minimum: structure, model integration, runtime/CI constraints sections).
11. If code behavior and `README.md` are inconsistent, update `README.md` in the same PR before (or together with) the code change.
12. For newly added models or major model-path refactors, append/update that model's configuration flow in `README.md` (input adapter, channel mapping, wrapper step strategy, config keys).

## Commit and Push Pattern

1. `git switch -c feature/<topic>`
2. Edit one script/logical unit.
3. Run minimal validation for that unit.
4. `git add <files>`
5. `git commit -m "<clear message>"`
6. `git push -u origin <branch>`
7. Create PR and request review.
8. Update `CHANGELOG.md` under `Unreleased` before opening PR (unless `N/A` is justified).

## PR Checklist (minimum)

- Why this change is needed.
- What files changed.
- Validation commands and outputs.
- Risks and rollback notes.

