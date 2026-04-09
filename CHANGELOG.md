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

- Feature(rolling/truth-cache): 引入作业内全局真值缓存管理器 `TruthCacheManager`（`pipelines/truth_cache.py`），解决多模型场景下 `truth load` 逐帧 I/O 热点。
  - **短路路径**：当 `skip_plots=True` 且 `enable_eval=False` 时，`need_truth=False`，真值路径完全跳过（不加载、不缓存、不重采样），滚动推理仅执行 `model.step + npy write`。
  - **预加载**：当 `need_truth=True`，在每个模型每个起报时刻开始前一次性预加载并缓存整个预报区间（`[6h, 12h, ..., max_lead]`）的真值，帧循环中从缓存读取，消除逐帧 `truth load + regrid`。
  - **内存缓存（memory）**：进程级 dict，key=valid_dt，模型切换不清理（跨模型复用相同 valid_dt），作业结束随进程释放。
  - **磁盘缓存（disk）**：Pickle 写入 `{output_root}/_truth_cache/{job_id}/rank{R}/`，原子写（`.tmp + os.replace`），作业结束默认删除（`KEEP_TRUTH_CACHE=1` 保留）。
  - **自动模式（auto，默认）**：对第一帧真值估算 nbytes，与 `n_steps * safety_factor` 相乘估算总需求；读取 `/proc/meminfo` 或 `psutil.virtual_memory().available`；若需求 ≤ 可用内存 × `budget_ratio`（默认 0.4）则用 memory，否则用 disk；每次运行打印决策依据。
  - **多 rank 安全**：memory 缓存天然进程私有；disk 模式默认 `rank{R}` 子目录隔离，消除跨 rank 写冲突。
  - **生命周期**：`truth_cache.close()` 在所有模型+评估完成后调用；磁盘缓存目录按 job_id 分隔，不影响下次作业的缓存内容。
  - **降级保护**：预加载失败（truth adapter 抛出异常）时记录警告并跳过该帧；帧循环中缓存未命中时降级为在线加载（兜底语义不变）；磁盘写入失败时自动回退内存缓存。
  - **修复副作用**：同时消除了 `compare_plot_dir.mkdir` 在帧循环内的冗余调用（现仅在初始化段调用一次）。
- CLI(run_rolling): 新增 `--truth-cache-mode {auto,memory,disk}`、`--truth-cache-budget-ratio`、`--truth-cache-safety-factor`、`--keep-truth-cache` 四个参数；均支持对应环境变量（`TRUTH_CACHE_MODE` 等）。
- Slurm(submit_rolling): 新增 `TRUTH_CACHE_MODE`、`TRUTH_CACHE_BUDGET_RATIO`、`TRUTH_CACHE_SAFETY_FACTOR`、`KEEP_TRUTH_CACHE` 变量；透传至 `run_rolling.py` 命令行；`sbatch_repro_env_snapshot` 中包含这些变量。
- 验证(三组 sbatch benchmark，PanGu/ecmwf_init/20260407T06/48h/8frames)：

  | 组别 | 作业ID | truth_cache_mode | skip_plots | step_profile avg truth | CPU均/帧 | GPU均/帧 |
  |---|---|---|---|---|---|---|
  | G1: 短路（纯推理） | 111221873 | auto (短路) | 1 | **0.000s** | 5.920s | 9.041s |
  | G2: memory 缓存  | 111221875 | memory | 0 | **0.000s** | 29.719s | 40.837s |
  | G3: disk 缓存    | 111221876 | disk   | 0 | **0.483s** | 30.327s | 40.873s |

  结论：
  - G1 确认真值路径完全短路（truth=0.000s），纯推理 CPU均/帧=5.920s（9.041s GPU），与预期一致。
  - G2 memory 模式：8 帧全命中（hits=8），帧循环内 truth=0.000s（与短路路径相同），绘图耗时 5.380s/帧是主要瓶颈。
  - G3 disk 模式：8 帧全命中（hits=8），truth=0.483s/帧（磁盘 I/O，每帧 336.7MB Pickle 读取），绘图 5.594s/帧为主瓶颈。
  - memory vs disk：帧循环内 truth 开销从 0.483s（disk）→ 0.000s（memory），memory 模式帧内 I/O 完全消除；总 CPU均/帧差异较小（~0.6s）因绘图主导。
  - 历史对比（改前，online 逐帧加载）：truth=30-35s/帧；memory 缓存后降至 0.000s，**降幅 100%**。
  - 磁盘缓存清理确认：job 111221876 结束日志打印 `关闭缓存: mode=disk, preloaded=8, hits=8, mem_entries=0`，目录自动删除。

- Docs: ECMWF share layout — operational GRIBs live directly under `.../ecmwf_init0/new_init`; historical `res/` / `res2/` subdirectories are no longer used. README §12.2 H and `config/data.yaml` updated; `_load_data_source` error message hints `--data-source ecmwf_init` when a stale path is passed.
- Fix: `run_rolling.py` — `MODELS=all` now calls `get_all_enabled_models()` (was incorrectly `_get_all_enabled_models` after config_loader refactor).
- Perf(rolling): reduced per-frame CPU overhead in `rolling_pipeline.py` hot loop.
  - Added opt-in step segment profiler (`ROLLING_STEP_PROFILE=1`, `ROLLING_STEP_PROFILE_EVERY`), reporting per-window average wall time for `model_step` / `truth_load` / `npy_write` / `nc_write` / `plot` / `eval`.
  - `truth_adapter.load_blob_for_valid_time(valid_dt)` is now lazy-loaded only when plotting or downscale eval is active (previously unconditional every frame).
  - Hoisted repeated per-frame setup (`nc_dir.mkdir`, `plot_dir.mkdir`, pressure-level constant import/array build) to per-model/per-init scope.
- Perf(models): reduced repeated ONNX input-metadata lookups in autoregressive `step()`:
  - `FengWuModel` caches session input name during `load()` and reuses it in each `step()`.
  - `FuXiModel` caches per-session input specs (`name`, `is_temb`) and removes redundant `astype(np.float32)` conversion in `step()` return path.
- Fix: In `PARALLEL_MODE=model` with `WORLD_SIZE>1`, the per-rank `[timing]` summary was printed independently by each rank as it finished, causing the output to be split across different timestamps in the log. `RollingInferenceTiming` now exposes `to_dict_list()` and a `summary_rows_from_dicts()` static method. After the model loop, each rank atomically writes a JSON timing-shard to `{output_root}/_timing_shards/{job_id}/rank{R:04d}_of_{W:04d}.json`; rank0 then polls (≤300 s timeout) for all shards, merges them in original model-list order, and prints one unified `[timing] ===== 推理时间汇总（全部 rank 汇总）=====` block. Non-`model`-parallel paths (single process or `date` mode) retain the existing per-rank summary. The inline display-name dict in the model loop is replaced by a module-level `_MODEL_DISPLAY_NAMES` constant.
- Fix: `run_rolling.py` — when `LOCAL_RANK` is set and `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` already lists multiple IDs (e.g. `0,1`), keep that list and call `torch.cuda.set_device(LOCAL_RANK)` instead of shrinking env to a single index. Avoids rank1 seeing no ROCm device / ORT `SIGABRT` on some Slurm+DCU stacks.
- Fix: `CapabilityMemoryStore` now uses an `fcntl` lock file plus per-PID temp names when saving `source_model_capabilities.json`, and merges on-disk state before write. Prevents `FileNotFoundError` on `tmp.replace(...)` when `torchrun` / `WORLD_SIZE>1` ranks update the same `OUTPUT_ROOT/_capability_memory/` path concurrently (e.g. `PARALLEL_MODE=model` with two DCUs).
- **Breaking (ECMWF share layout)**: Default `OUTPUT_ROOT` for `ecmwf_init` is now the single directory `.../LYQ/ecmwf_init` (no per-`init_hour` `ECMWF_Init_Infer_result_${h}h` subfolder). Existing trees can be lifted with `scripts/flatten_ecmwf_init_roots.py`; duplicate side experiments become `{init_tag}_12h_ab_full_window` style siblings.
- **Breaking**: Rolling / eval output layout is now **init-first**: NPY and `meta_{init_tag}.json` live under `{output_root}/{init_tag}/{output_slug}/` (see `zk_io/rolling_paths.py` and README §9). Plots: `{output_root}/{init_tag}/plots/{output_slug}/`. Eval: `{output_root}/{init_tag}/eval_*_{max_lead}h/` (no trailing `_{init_tag}` on eval folder names). NC per-step dirs use the same `output_slug` under `{init_tag}/nc/`. `run_eval_npy`, `run_evaluate`, and multigrid eval **fall back** to legacy `{DisplayName}/ERA5_6H/` for NPY reads. External notebooks or scripts that hard-coded `Model/ERA5_6H` or root-level `eval_*_{init_tag}` must be updated, or run `scripts/migrate_rolling_output_layout.py` on existing trees.
- Rolling: `--save-nc` now writes 13-level `pres_*` in `lead_*.nc` for **FengWu** (when 69-ch outputs decode to `pangu_*` in blob) and **GraphCast Official Operational** (JAX rollout saves `pangu_{z,t,u,v,q}_{init_tag}.npy` beside surface caches; old caches without these files trigger a one-time full rollout refresh). `run_graphcast_official_rollout.py` adds `_select_3d`.
- Config: `sources.ecmwf_init.root` in `config/data.yaml` is now `/public/share/aciwgvx1jd/ecmwf_init0/new_init` (GRIB tree root; previously `.../new_init/res`).
- Feature: `ProcessCpuStopwatch` supports `pause`/`resume`; `RollingInferenceTiming` adds `pause_cpu`/`resume_cpu`. Rolling inference accepts `--cpu-timing-exclude-plots` / `pipeline.rolling.cpu_timing_exclude_plots` / `CPU_TIMING_EXCLUDE_PLOTS=1` in `submit_rolling.sh` to omit per-step matplotlib work from `[timing]` CPU totals (GPU device interval unchanged).
- Config: `graphcast_official_operational_stepwise` is now `enabled: false` by default so `MODELS=all` runs six models (seven registered, one disabled); enable explicitly in `models.yaml` or pass `MODELS=...graphcast_official_operational_stepwise` when needed.
- Config: `graphcast_official_operational.rollout_blob_input_mode` default is now `init_only` (`models.yaml` + wrapper fallback); use `full_window` or `GC_BLOB_INPUT_MODE=full_window` when future-time blobs are complete and intended for rollout input.
- Docs/agent: document that Slurm/Hardware logs live under gitignored `logs/` (`rolling_<JOBID>.out`, etc.); add `.cursor/rules/project-logs.mdc`, `AGENTS.md` §15, and README §8.1 note so diagnostics read disk paths directly.
- Slurm: `submit_rolling.sh` and `submit_evaluate.sh` log `sbatch_repro_copy_paste` (and WorkDir/SubmitLine-derived lines) so jobs record a copy-paste `sbatch` invocation including env-var prefix; Slurm `SubmitLine` alone omits `VAR=value` prefixes.
- Feature: unified missing-channel capability memory for rolling. Added persistent source/model memory file `${output_root}/_capability_memory/source_model_capabilities.json` and per-init fallback audit `meta_{init_tag}.json` next to NPY under `{output_root}/{init_tag}/{output_slug}/`. Rolling records observed blob keys per source, model optional-channel decisions, and fallback actions (skip/zero-fill/truth-missing).
- Data: adapters now optionally expose vertical velocity as `pangu_w` when available (`era5_adapter.py`, `gundong_adapter.py`, `ecmwf_init_grib_adapter.py`); GRIB keeps `w` optional and non-blocking.
- GraphCast official: `run_graphcast_official_rollout_gundong.py` and cache wrapper now prefer real `w` / `tp6` / `surface_z_at_surface` / `land_sea_mask` inputs when provided by adapter blob, and only fallback to zero with explicit audit/meta records.
- FuXi: TP fallback path is now auditable via a shared fallback recorder (still controlled by `tp_fallback: zero|error`).
- Feature: multi-resolution evaluation for rolling + `run_eval_npy.py`: **downscale** (existing bilinear truth to model grid), **fidelity_nn** / **fidelity_exact** (masked W-MAE/W-RMSE on model grid from native GRIB surface without full-field downscaling), **upscale** (bilinear forecast to native truth grid). Adds `ECMWFInitGribAdapter.load_surface_native_blob` / `load_surface_native_for_valid_time`, `core/evaluation/multigrid_eval.py`, `metrics.compute_step_metrics_masked`, CLI `--eval-modes` / `--fidelity-max-deg` / `--auto-multigrid`, `defaults.yaml` keys `evaluation.fidelity_max_deg` / `evaluation.auto_multigrid`, Slurm env `EVAL_MODES` / `FIDELITY_MAX_DEG` / `AUTO_MULTIGRID` in `submit_rolling.sh`.
- Fix/behavior: multigrid skip is based on **truth vs model grid alignment** (`truth_grid_matches_model`), not on ECMWF GRIB alone; `load_truth_blob_for_multigrid` falls back to `load_blob_for_valid_time` when no native surface API exists.
- Slurm: `scripts/submit_evaluate.sh` supports `EVAL_ENGINE=npy` to run `run_eval_npy.py` (same adapters as rolling; multigrid / downscale re-eval). Legacy `run_evaluate.py` remains the default when `EVAL_ENGINE` unset or `legacy`.
- Eval: `--spatial-plots` / `SPATIAL_PLOTS=1` drives multigrid (`fidelity`_* / `upscale`) field triple plots under `{init_tag}/plots/{output_slug}/`; rolling post-pass multigrid respects `--skip-plots` inverse.
- Feature: new `core/monitoring/inference_timing.py` module providing independent, reusable CPU + GPU/DCU inference timing. Exposes `RollingInferenceTiming` (per-model `begin_model` / `add_frames` / `end_model` accumulator), `ProcessCpuStopwatch` (`time.process_time()` diff), `TorchGpuSegmentStopwatch` (CUDA/HIP `torch.cuda.Event` + sync), and a generic `timed_segment` context manager for use in other scripts. Both metrics are enabled by default; GPU timing auto-skips when `torch.cuda.is_available()` is False.
- Feature: `run_rolling.py` and `rolling_pipeline.run_rolling()` now emit per-model `[timing]` log lines (frames, total CPU s, avg CPU/frame, total GPU s, avg GPU/frame) after each model's rolling segment completes, and a final summary table after all models finish. Range covers only the date/lead inference + write-step loop, excluding weight load and `unload()`. Disable with `--no-cpu-timing` (CPU) or `--no-gpu-timing` (GPU) independently.
- Fix (superseded by flat layout): ECMWF 默认输出曾按 `ECMWF_Init_Infer_result_${init_hour}h` 分目录；现已改为统一根 `.../LYQ/ecmwf_init` + `{init_tag}` 子目录（见上条 Breaking）。
- Fix: `graphcast_official_operational` rollout 缓存默认跟随 `output_root`（`${output_root}/_gc_oper_cache`）；新增 `lock_rollout_cache_dir` 允许固定回 `models.yaml` 中的 `rollout_cache_dir`。
- Fix/UX: GC 官方模型 `blob_input` 缺失日志改为摘要输出（缺失数量/区间/示例时次），并明确回退语义为 forward reuse；rolling 启动时新增 `enable_eval` 状态提示，避免误判“未出 RMSE/MAE”为静默失败。
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
- Refactor: centralized model name maps (`SLUG_TO_DISPLAY`, `EVAL_MODEL_ORDER`) into `zk_io/rolling_paths.py`; removed 4 duplicate dicts from `rolling_pipeline.py`, `run_eval_npy.py`, `multigrid_eval.py`, `evaluate_models.py`, `verify_pipeline.py`.
- Refactor: centralized `PANGU_LEVELS` — canonical definition remains in `cepri_loader.py`; duplicates in `gundong_adapter.py`, `ecmwf_init_grib_adapter.py`, `graphcast_official_operational_model.py`, `infer_cepri_onnx.py` replaced with imports. Re-exported via `core.data.PANGU_LEVELS`.
- Refactor: new `core/config_loader.py` with `load_defaults()`, `resolve_output_root()`, `get_all_enabled_models()`. `defaults.yaml` gains `pipeline.output_roots` map. All entry points (`run_rolling.py`, `run_verify.py`, `run_evaluate.py`, `run_eval_npy.py`) use the shared functions instead of inline duplicates.
- Refactor: `progress()` timestamp+rank log helper moved to `core/monitoring/__init__.py`. All pipeline/entry callers delegate to it.
- Refactor: `_load_pred_stack` extracted to `zk_io/npy_reader.py`; replaced copies in `run_evaluate.py`, `run_eval_npy.py`, `multigrid_eval.py`.
- Refactor: shared shell boilerplate (conda activation, DTK/CUDA module load, env vars, Python verify) extracted to `scripts/_common.sh`; `submit_rolling.sh`, `submit_verify.sh`, `submit_evaluate.sh` now `source _common.sh`.
- Refactor: `runtime_paths.py` adds `bootstrap()` for idempotent `sys.path` + `os.chdir` setup; all entry points and pipelines call it instead of inline `sys.path.insert` + `os.chdir`.
- Deprecation: `evaluate_models.py` marked deprecated with warning header; `run_evaluate.py` docstring notes `run_eval_npy.py` as preferred replacement.
- Moved `scripts/submit_ecmwf_apr7_2026_4inits_2dcu.sh` to `scripts/examples/`.
- Fix: `pick_providers()` in `infer_cepri_onnx.py` now accepts `device_id` and returns provider options with `{"device_id": device_id}` for GPU EPs. Pangu/FengWu/FuXi model wrappers pass `LOCAL_RANK` to avoid all ORT ranks defaulting to GPU 0 in model-parallel mode.
- Fix: `timing.end_model()` moved into the `finally` block in `rolling_pipeline.py` so timing data is recorded even if the date loop raises an exception.
- Tests: new `tests/test_smoke.py` (pytest) with import, config, name-consistency, output-root, rolling-paths, metrics, and timing tests.
- CI: `pr-gate.yml` runs `pytest tests/ -x -v` instead of inline import check; `requirements-ci.txt` adds `pytest>=7.0`.
- Docs: README §3.1 tree updated with new/missing modules; §12.2 heading corrected to "7 models (6 enabled by default)"; §5 TP read order corrected (`fh_0` first). USAGE.txt marks `submit_gundong_20260303_5models.sh` deprecated.

### Test Report (brief)

- Code sanity: `python -m py_compile` passed for modified files (`pipelines/rolling_pipeline.py`, `core/models/{pangu,fengwu,fuxi}_model.py`).
- CPU regression evidence baseline (historical logs): fast jobs around `CPU均/帧 3~5s` (e.g. `rolling_111130762.out`) vs slower jobs `17~28s` (e.g. `rolling_111189970.out`, `rolling_111164996.out`).
- Runtime recheck note: attempted local short-run A/B (`skip_plots=1/0`, `max_lead=12`) was killed by host resource limits before model load completed (exit 137), so on-cluster rerun is still required for final perf delta confirmation.

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