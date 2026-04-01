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

- Feature: new `core/monitoring/inference_timing.py` module providing independent, reusable CPU + GPU/DCU inference timing. Exposes `RollingInferenceTiming` (per-model `begin_model` / `add_frames` / `end_model` accumulator), `ProcessCpuStopwatch` (`time.process_time()` diff), `TorchGpuSegmentStopwatch` (CUDA/HIP `torch.cuda.Event` + sync), and a generic `timed_segment` context manager for use in other scripts. Both metrics are enabled by default; GPU timing auto-skips when `torch.cuda.is_available()` is False.
- Feature: `run_rolling.py` and `rolling_pipeline.run_rolling()` now emit per-model `[timing]` log lines (frames, total CPU s, avg CPU/frame, total GPU s, avg GPU/frame) after each model's rolling segment completes, and a final summary table after all models finish. Range covers only the date/lead inference + write-step loop, excluding weight load and `unload()`. Disable with `--no-cpu-timing` (CPU) or `--no-gpu-timing` (GPU) independently.
- Feature: new `--truth-source` CLI parameter for `run_rolling.py` and `run_eval_npy.py` decouples the init data source from the evaluation truth source. Default behaviour unchanged (truth = data_source). Enables running inference from GRIB init fields while evaluating against ERA5 truth. (`core/data/ecmwf_init_grib_adapter.py`) reads ECMWF/GFS GRIB1 initial condition files (`G_YYYYMMDDHH_fh_{0,1}.grib1`) via `pygrib`, converting to the unified blob contract (same as ERA5 adapters). Supports automatic level reorder/interpolation to PANGU_LEVELS, z/gh detection, RH→q fallback, TP retrieval from fh_0/fh_1, lon -180..180 → 0..360 reorder, and bilinear regridding from non-0.25° grids (e.g. 0.1°) to the standard 721×1440.
- Feature: new `--truth-source` CLI parameter for `run_rolling.py` and `run_eval_npy.py` decouples the init data source from the evaluation truth source. Default behaviour unchanged (truth = data_source). Enables running inference from GRIB init fields while evaluating against ERA5 truth.
- Config: added `ecmwf_init` data source in `data.yaml` pointing to `/public/share/aciwgvx1jd/ecmwf_init0/new_init/res`.
- Data: `detector.py` auto-detects `ecmwf_init_grib` format when directory contains `G_*_fh_*.grib1`; gracefully skips registration if `pygrib` is not installed.
- Docs/UX: `graphcast_official_operational` logs now explain cache-miss vs full JAX rollout wall time; README §12.F notes post-rollout progress gaps (truth I/O + eval every 8 logged steps, not a second JAX run).
- Docs: `AGENTS.md` adds rules 13–14 (this repo only): prefer extending `submit_*.sh` via env vars; new `scripts/*.sh` only for a genuinely new Slurm entrypoint, with README coverage.
- Feature: integrated official DeepMind JAX GraphCast with operational parameters (0.25°, 13 pressure levels, mesh 2to6) as a 6th model (`graphcast_official_operational`). New model wrapper (`core/models/graphcast_official_operational_model.py`) uses a "pre-compute + cache read" strategy: missing NPYs trigger an on-demand subprocess rollout; rolling + eval use existing `submit_rolling.sh` with `MODELS` including `graphcast_official_operational` (no extra submit script).
- Feature: extended `run_graphcast_official_rollout_gundong.py` to extract and save `msl` (mean sea level pressure) alongside existing u10/v10/t2m, enabling full 4-variable evaluation alignment with the unified pipeline.
- Config: added `models.graphcast_official_operational` entry in `models.yaml` with operational param path, JAX embed, and rollout cache configuration.
- Pipeline: updated `_EVAL_MODEL_ORDER` and display-name mappings in `rolling_pipeline.py`, `run_eval_npy.py`, `run_evaluate.py`, and `verify_pipeline.py` to include `GC_Official_Oper`.
- Docs: README §8.0.1 records failed `torchrun` multi-model parallel rolling on DCU (FuXi `SIGABRT` at `local_rank` 2; jobs e.g. 110494072/110494391); recommend `WORLD_SIZE=1` for production until ORT/driver follow-up.
- Data: `gundong_20260324` accepts pressure NC under `{root}/pressure/` (flat) or `{root}/pressure/pressure/` (nested); detector matches both.
- Rolling: intersect `--date-range` with `adapter.list_dates()` before loading models; clarify skip logs when pressure/surface files are missing; default ORT init stagger 8s/LR (override `ROLLING_ORT_STAGGER_SEC`).
- Rolling: `torchrun` multi-process uses rank0-only hardware monitor and per-`LOCAL_RANK` startup staggering to reduce ROCm/ONNX concurrent-init SIGABRT risk.
- Data: `gundong_20260324` adapter reads `tp` from `surface/*_surface_accum.nc` when instant surface has no precipitation variable, populating `surface_tp_6h` for FuXi 70ch input.
- Fix: `gundong_20260324` adapter now keeps `surface_tp_6h` in source-native units (ERA5 commonly `m`) instead of converting `m -> mm`, aligning FuXi TP input scale with `zforecast.py`.
- Analysis: compared historical jobs `110330301` (`tp_mean=0`) and `110494967` (`tp_mean=0.101713`) for `20260303T12`; FuXi metrics CSV rows and forecast NPY outputs (`t2m/u10/v10/msl`) are byte-identical (`max_abs_diff=0`), indicating no observable TP impact on evaluated variables for this case.
- CI: `pr-gate` installs `[requirements-ci.txt](requirements-ci.txt)` before the sanity import step so GitHub Actions has the same minimal third-party imports as local entrypoints (`netCDF4` for `cepri_loader`, `onnxruntime` for FuXi, `numpy<2` for ORT ABI compatibility, etc.), without requiring the full e2s conda stack.
- CI: `checkout` uses `fetch-depth: 0` and the changelog gate diffs `pull_request.base.sha` vs `head.sha` so git no longer exits 128 on shallow merge checkouts.
- Docs: optimize README architecture mermaid diagram for clarity and add README-first rule to AGENTS.md.
- Feature (experimental): added `graphcast_official_operational_stepwise` — in-process stepwise JAX model that loads the official checkpoint and drives it one `step()` at a time, matching the unified `WeatherModel` contract. Dual-track with the existing cache-based `graphcast_official_operational` (kept unchanged as stable baseline).
- Config: added `models.graphcast_official_operational_stepwise` entry in `models.yaml` (`type: jax_official_stepwise`).
- Pipeline: updated `_EVAL_MODEL_ORDER` and display-name mappings in `rolling_pipeline.py`, `verify_pipeline.py`, `run_eval_npy.py`, `run_evaluate.py` to include `GC_Stepwise`.
- Scripts: added `scripts/compare_stepwise_vs_cache.py` for offline numerical comparison between stepwise and cache-based official JAX outputs.
- Docs: README §5 documents stepwise model config keys and experimental status; §12.2 G describes the stepwise model flow with validation commands.
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

