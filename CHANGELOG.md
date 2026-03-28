# Changelog

All notable changes to this project are documented in this file.

This format is inspired by Keep a Changelog and adapted for this repository.

## Versioning Convention

- Version format: `vMAJOR.MINOR.PATCH`
- `MAJOR`: incompatible behavior or interface changes
- `MINOR`: backward-compatible feature additions
- `PATCH`: backward-compatible fixes and non-functional improvements

## Unreleased

### Version
- v0.3.0 (planned)

### DateTime
- TBD (merge time)

### Merged PR/Branch
- TBD

### Summary of Changes (all)
- CI: `pr-gate` installs [`requirements-ci.txt`](requirements-ci.txt) before the sanity import step so GitHub Actions has the same minimal third-party imports as local entrypoints (`netCDF4` for `cepri_loader`, `onnxruntime` for FuXi, `numpy<2` for ORT ABI compatibility, etc.), without requiring the full e2s conda stack.
- CI: `checkout` uses `fetch-depth: 0` and the changelog gate diffs `pull_request.base.sha` vs `head.sha` so git no longer exits 128 on shallow merge checkouts.
- TBD

### Test Report (brief)
- TBD

### Related Commits
- TBD

---

## v0.2.0 - 2026-03-27

### Merged PR/Branch
- main direct history (pre-PR-governance baseline)

### Summary of Changes (all)
- Added FuXi cascade inference mode with configurable split (`short -> medium`).
- Added zforecast-style FuXi temb mode as default with legacy fallback option.
- Added optional `surface_tp_6h` support in adapters and FuXi channel mapping fallback policy.
- Added repository governance files (`AGENTS.md`, Cursor rule, PR template, CODEOWNERS, PR gate workflow).
- Added docs synchronization requirement in agent and PR rules.

### Test Report (brief)
- FuXi strategy/temb/TP smoke checks passed.
- Branch protection real push test reported expected remote rejection (`GH013`).
- Lint diagnostics for touched files reported clean.

### Related Commits
- `abb8c7b`, `403c681`, `3cf9df4`, `733a014`, `42622ec`, `9d4e1db`

