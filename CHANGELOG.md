# Changelog

All notable changes to this project are documented in this file.

This format is inspired by Keep a Changelog and adapted for this repository.

## Versioning Convention

- Version format: `vMAJOR.MINOR.PATCH`
- `MAJOR`: incompatible behavior or interface changes
- `MINOR`: backward-compatible feature additions
- `PATCH`: backward-compatible fixes and non-functional improvements

语义与 **SemVer** 一致：主版本表示不兼容变更，次版本表示向后兼容的新功能，修订号表示修复与文档等非破坏性更新。

## [v0.3.0] - 2026-04-11

### 概要

相对 **v0.2.0**，本版本将统一流水线扩展为可在集群上稳定运行的 **业务化滚动推理**：支持 ECMWF GRIB 初始场、官方 GraphCast（JAX）operational、作业内真值缓存与 init-first 输出目录、多分辨率评估与可复现实验配置，并补充推理计时、工程化入口与 CI。

### 里程碑（重点功能）

| 主题 | 说明 |
|------|------|
| **作业内全局真值缓存** | `TruthCacheManager`（[`pipelines/truth_cache.py`](pipelines/truth_cache.py)）；`auto` / `memory` / `disk`；当 `need_truth=False`（例如 `SKIP_PLOTS=1` 且未开评估）时完全跳过真值加载与缓存。 |
| **输出目录 init-first** | NPY / meta / 图 / 评估路径统一为 `{output_root}/{init_tag}/...`；[`zk_io/rolling_paths.py`](zk_io/rolling_paths.py)；[`scripts/migrate_rolling_output_layout.py`](scripts/migrate_rolling_output_layout.py) 迁移旧树；ECMWF 默认输出根目录扁平化（见 Breaking）。 |
| **GraphCast Official Operational** | JAX 预计算 rollout + 缓存读取；`ecmwf_init` GRIB 适配；`rollout_cache_dir` / `lock_rollout_cache_dir` / `rollout_blob_input_mode` 等配置。 |
| **多分辨率评估** | [`core/evaluation/multigrid_eval.py`](core/evaluation/multigrid_eval.py)、`run_eval_npy`、CLI `--eval-modes` / `--truth-source` / `--auto-multigrid` 等。 |
| **推理计时** | [`core/monitoring/inference_timing.py`](core/monitoring/inference_timing.py)：进程 CPU（`process_time`）与 GPU 区间（CUDA/HIP Event）；`[stage_totals(wall)]` 墙钟分段；`init_state` 与逐步 `step` 分计（JAX 全量预计算模型的耗时主要在 `init`）。 |
| **工程化** | [`core/config_loader.py`](core/config_loader.py)、[`scripts/_common.sh`](scripts/_common.sh)、[`runtime_paths.bootstrap()`](runtime_paths.py)；Slurm `sbatch_repro_copy_paste`；能力记忆 `_capability_memory/`。 |

### 详细变更（按主题）

#### 滚动推理与 Slurm

- 真值：区间预加载、内存/磁盘决策、多 rank 磁盘子目录隔离、作业结束清理与降级路径。
- `run_rolling.py` / `submit_rolling.sh`：真值缓存 CLI 与环境变量；可选从 CPU 计时中排除画图；ORT 多进程 stagger 与 `device_id`；`torchrun` 下多 rank 计时 shard 合并；`CapabilityMemoryStore` 文件锁与合并写。
- **Breaking**：init-first 布局；ECMWF 业务默认 `OUTPUT_ROOT` 不再按 `ECMWF_Init_Infer_result_${h}h` 分目录；[`config/data.yaml`](config/data.yaml) 中 `ecmwf_init.root` 为 **`.../ecmwf_init0/new_init`**（GRIB 树根，不含历史 `res/` 子目录）。旧树可用 `scripts/flatten_ecmwf_init_roots.py` 处理。
- 能力记忆：`source_model_capabilities.json`、各起报 `meta_{init_tag}.json` 记录通道与降级。

#### 数据与适配器

- `ecmwf_init_grib_adapter`：GRIB1 初始场；`detector` 自动识别；可选垂直速度等通道；`--truth-source` 与初始场数据源解耦。
- `gundong`：pressure 扁平/嵌套布局、`surface_tp_6h`、FuXi TP 单位与文档。
- **v0.3.0 末次提交**：ERA5 平面适配器可选 `pangu_w`；`channel_mapper` 为 FuXi TP 零填充增加 `on_fallback`；`extract_surface_vars` 支持 `strict` / `on_missing`。

#### 评估与指标

- multigrid：downscale、fidelity、upscale；native surface 加载与跳过逻辑基于格点对齐。
- `run_eval_npy.py`、`EVAL_ENGINE=npy`、空间图与 rolling 后处理。
- **v0.3.0 末次提交**：[`compute_step_metrics_masked`](core/evaluation/metrics.py)（掩膜加权 W-MAE / W-RMSE）。

#### 模型（ONNX / JAX）

- Pangu / FengWu / FuXi：ORT `device_id`、输入元数据缓存、热循环开销优化。
- `graphcast_official_operational_stepwise`：实验性逐步 JAX，默认 **不** 纳入 `MODELS=all`。
- **v0.3.0 末次提交**：stepwise 包装器在缺失必需地表预测通道时显式失败，并预留回退记录器钩子。

#### 文档与 CI

- README / AGENTS：架构与路径约定；`logs/` 为 gitignore，**规则 15**：诊断集群作业请直接读取 `logs/rolling_<JOBID>.out` 等。
- PR gate：`pytest`、`requirements-ci.txt`、浅克隆修复。
- `compare_stepwise_vs_cache` 示例输出路径更新。

#### 测试

- [`tests/test_smoke.py`](tests/test_smoke.py)（pytest）。

### 参考基准（历史作业 ID，非性能保证）

| 场景 | 作业 ID | 备注 |
|------|---------|------|
| 真值路径短路（纯推理） | 111221873 | `truth≈0` |
| memory 缓存 + 绘图 | 111221875 | |
| disk 缓存 + 绘图 | 111221876 | 含 `step_profile` 窗口均值 |

本地短跑可能因宿主机资源限制被 kill（exit 137）；以集群 `sbatch` 结果为准。

### 版本元数据

- **版本**：`v0.3.0`
- **发布日期**：2026-04-11
- **主开发分支**：`feature/graphcast-official-operational`（已合并）
- **合并至 `main`**：merge commit [`e8db910`](https://github.com/esesmalls/unified_pipeline/commit/e8db9102904f504731de6f7d2f88c899d85b195d)（`将当前的业务化脚本库同步入主分支`）。GitHub 上可在仓库 **Pull requests → Closed** 查看对应合并 PR。
- **Git 标签**：`v0.3.0` 指向 **`main` 上含本条目与合并说明的提交**（合并完成后由维护者打标签并 `git push origin v0.3.0`）。

### Related commits

相对 **`main` 祖先提交 `9d4e1db`** 起，本版本分支包含（从新到旧）；**并入 `main` 的合并提交**：

```
e8db910 将当前的业务化脚本库同步入主分支
b9dda64 feat(data,metrics,models): optional pangu_w; audit hooks; masked metrics; stepwise validation
41fce00 fix(timing): add init_state_s to stage_totals, label as wall time
cba1b61 feat(timing): 始终开启逐段计时统计，推理结束输出 [stage_totals]
9b18f71 fix(rolling_paths, multigrid_eval): add migration utility functions + fix NPY path layout
af17a4e fix(npy_writer): align npy_filename call with rolling_paths API
50a201d fix(_common.sh): restore module purge + unset PYTHONPATH + ROCm env vars
8e673ff fix(monitoring): add progress() function missing from Phase 1D refactor
994fe3e fix: restore core/evaluation/multigrid_eval.py missing source
602198b fix(runtime_paths): add bootstrap() function missing from run_rolling.py refactor
810af4c chore: commit previously untracked source files
98ba9f2 fix(npy_reader): restore ndim>3 squeeze branch from original
fd13d03 fix: restore missing source files and correct ecmwf_init data root
4b0ae61 feat(rolling): 作业内全局真值缓存（TruthCacheManager）+ 真值路径短路
1860dcb feat(graphcast): official operational JAX model, ECMWF GRIB init, timing
810887d fix(fuxi): align TP input scale and document no-impact case
d31d150 docs: record failed torchrun parallel rolling (FuXi SIGABRT); A-only sbatch example
17bd234 fix(gundong): support flat pressure/YYYY_MM_DD_pressure.nc layout
0009c27 fix(rolling): filter dates to available pressure days, clearer skip, longer ORT stagger
97ea640 fix(rolling): rank0-only HW monitor and ORT init stagger for torchrun
2745057 feat(gundong): surface_tp_6h from surface_accum for FuXi
febea2b Merge branch 'main' into feature/changelog-governance
5327ab9 docs: improve architecture diagram and README-first rules
e0a4336 Merge pull request #1 from esesmalls/feature/changelog-governance
65f3bf4 ci: fix pr-gate git exit 128 on shallow PR checkout
c69cd37 ci: install requirements-ci.txt before pr-gate import check
669df5b Add changelog governance, template, and CI enforcement.
```

等价命令：`git log 9d4e1db..HEAD --oneline`（含后续仅文档/元数据提交）。

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
