# unified_pipeline

`unified_pipeline` 是 `ZK_Models` 的统一推理与评估平台，覆盖：

- 功能一：初步推理验证（`run_verify.py`）
- 功能二：滚动推理（`run_rolling.py`）
- 仅评估：基于已有 NPY 重跑指标（`run_eval_npy.py` / `run_evaluate.py`）

该目录聚焦推理与评估，不包含训练流程。

## 1. 项目目标

通过统一的“模型注册 + 数据适配 + 通道映射 + 流水线编排”架构，解耦：

- 数据格式（`era5_flat` / `gundong_20260324` / 可扩展）
- 模型实现（Pangu/FengWu/FuXi/GraphCast/GraphCast_CS）
- 运行场景（verify / rolling / offline eval）

## 2. 实际架构（与当前代码一致）

```mermaid
flowchart TD
    cliVerify[run_verify.py] --> verifyPipe[pipelines/verify_pipeline.py]
    cliRolling[run_rolling.py] --> rollingPipe[pipelines/rolling_pipeline.py]
    cliEvalNpy[run_eval_npy.py] --> metricsCore[core/evaluation/metrics.py]
    cliEvalLegacy[run_evaluate.py] --> metricsCore

    verifyPipe --> dataDetect[core/data/detector.py]
    rollingPipe --> dataDetect
    dataDetect --> adapters[core/data/*_adapter.py]
    adapters --> mapper[core/data/channel_mapper.py]

    verifyPipe --> modelRegistry[core/models/model_registry.py]
    rollingPipe --> modelRegistry
    modelRegistry --> modelImpl[core/models/*_model.py]

    rollingPipe --> ioLayer[zk_io/npy_writer.py + nc_writer.py + plot_utils.py]
    verifyPipe --> ioLayer
    rollingPipe --> metricsCore
```

## 3. 目录结构

```text
unified_pipeline/
├── config/
│   ├── models.yaml
│   ├── data.yaml
│   └── defaults.yaml
├── core/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── monitoring/
├── pipelines/
│   ├── verify_pipeline.py
│   └── rolling_pipeline.py
├── zk_io/
│   ├── npy_writer.py
│   ├── nc_writer.py
│   └── plot_utils.py
├── scripts/
│   ├── submit_verify.sh
│   ├── submit_rolling.sh
│   ├── submit_evaluate.sh
│   └── submit_gundong_20260303_5models.sh
├── run_verify.py
├── run_rolling.py
├── run_eval_npy.py
├── run_evaluate.py
├── runtime_paths.py
└── USAGE.txt
```

## 4. 路径约定

见 `runtime_paths.py`：

- `UNIFIED_PIPELINE_ROOT`：本目录（代码与配置）
- `ZK_MODELS_ROOT`：上级 `ZK_Models`（ONNX 权重等）
- `GRAPH_CAST_ROOT`：`graphcast` 工程根（运行时会 `os.chdir` 到此）

因此脚本可从任意目录调用，但默认相对路径行为以 `GRAPH_CAST_ROOT` 为准。

## 5. 配置文件说明

### `config/models.yaml`

- 管理模型注册、启用状态、权重路径和模型元信息
- 支持占位符：
  - `${ZK_ROOT}` -> `ZK_MODELS_ROOT`
  - `${GRAPH_ROOT}` -> `GRAPH_CAST_ROOT`
- 当前内置模型：
  - `pangu`
  - `fengwu`
  - `fuxi`
  - `graphcast`
  - `graphcast_cs`
- `pangu` 支持调度策略：
  - `scheduler_mode: six_hour_only`：仅使用 6h 模型（当前默认，便于和历史脚本对齐复核）
  - `scheduler_mode: hybrid_24h`：`+24h/+48h/...` 使用 24h 模型，其余使用 6h 模型
- `fuxi` 支持可配置级联与 temb：
  - `infer_mode: cascade|fixed`（默认 `cascade`）
  - `cascade_split_step: 20`（默认前 20 步 `short`，后续 `medium`）
  - `temb_mode: zforecast|legacy`（默认 `zforecast`）
  - `tp_fallback: zero|error`（缺失 `surface_tp_6h` 时回退策略）

### `config/data.yaml`

- 管理数据源名称到真实路径与格式的映射
- `format` 可显式指定，也可由 `core/data/detector.py` 自动探测
- 当前内置数据源：
  - `test_era5`（`era5_flat`）
  - `gundong_20260324`（`gundong_20260324`）

### `config/defaults.yaml`

- verify/rolling/eval 默认参数
- CLI 参数优先级高于该文件

## 6. 环境与依赖

**默认 conda 环境（推荐，与 `scripts/submit_*.sh` 中 `CONDA_ENV` 默认值一致）：**

- `torch2.4_dtk25.04_cp310_e2s`

本地或交互式运行前请先：

```bash
conda activate torch2.4_dtk25.04_cp310_e2s
```

如需改用其他环境，Slurm 提交时设置 `CONDA_ENV=你的环境名` 即可覆盖默认值。

代码依赖（从导入与脚本推断）：

- `python>=3.10`
- `numpy`
- `PyYAML`
- `xarray`
- `pandas`
- `matplotlib`
- `onnxruntime`
- `torch`
- `netCDF4`（xarray 读取 NC 常见后端）

集群脚本（`submit_verify.sh` / `submit_rolling.sh` / `submit_evaluate.sh`）默认使用上述 conda 环境；可通过环境变量 `CONDA_ENV` 覆盖。

**GitHub Actions（PR 门禁）**：`.github/workflows/pr-gate.yml` 在合并前会做编译与轻量 `import` 自检。Runner 通过根目录 [`requirements-ci.txt`](requirements-ci.txt) 安装与这些自检一致的最小 pip 依赖（含 `netCDF4`、`onnxruntime` 等 import 链所需项；**不含** `torch` 等与 GraphCast 实跑相关的重型包）。完整推理与评估仍以本节的 conda / 集群环境为准。

- Slurm + DTK/ROCm（或 `USE_CUDA=1` 切 CUDA）

## 7. 快速开始

### 7.1 查看参数

```bash
cd /public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast
python ZK_Models/unified_pipeline/run_rolling.py --help
python ZK_Models/unified_pipeline/run_verify.py --help
```

### 7.2 功能一：初步推理验证

```bash
python ZK_Models/unified_pipeline/run_verify.py \
  --models pangu fengwu \
  --data-source test_era5 \
  --date 20260308 \
  --hour 12 \
  --num-steps 2 \
  --all-surface
```

输出默认在 `results/verify`（可通过 `--output-root` 指定）。

### 7.3 功能二：滚动推理

```bash
python ZK_Models/unified_pipeline/run_rolling.py \
  --models all \
  --data-source gundong_20260324 \
  --date-range 20260308 \
  --init-hour 12 \
  --lead-step 6 \
  --max-lead 240
```

常用开关：

- `--enable-eval`：推理过程中内嵌评估
- `--save-diff` / `--save-diff-nc`：保存差值场
- `--skip-plots`：跳过对比图
- `--save-nc`：保存逐步 NC
- `--parallel-mode auto|date|model`：多卡分片策略
- `--lead-step` 必须能被模型步长整除，否则会直接报错退出（避免错配推理）

### 7.4 已有 NPY 的离线评估（推荐）

```bash
python ZK_Models/unified_pipeline/run_eval_npy.py \
  --data-source gundong_20260324 \
  --date-range 20260310 \
  --init-hour 12 \
  --max-lead 240 \
  --lead-step 6 \
  --models pangu fengwu fuxi graphcast graphcast_cs \
  --output-root /public/share/aciwgvx1jd/GunDong_Infer_result_12h
```

### 7.5 旧入口评估（兼容）

```bash
python ZK_Models/unified_pipeline/run_evaluate.py \
  --time-tag 20260308T12 \
  --models FengWu GraphCast FuXi PanGu \
  --variables u10 v10 t2m
```

## 8. Slurm 提交

任务提交规范：**仅使用以下三个脚本**（通过环境变量覆盖参数）。

在 `unified_pipeline` 目录下：

```bash
sbatch scripts/submit_verify.sh
sbatch scripts/submit_rolling.sh
TIME_TAG=20260308T12 sbatch scripts/submit_evaluate.sh
```

`scripts/submit_gundong_20260303_5models.sh` 已标记为兼容用途（deprecated），不建议新任务继续使用。

脚本参数支持环境变量覆盖，例如：

```bash
MODELS="fengwu fuxi" DATE_RANGE="20260301:20260318" ENABLE_EVAL=1 sbatch scripts/submit_rolling.sh
```

### 8.1 日志首屏字段解读（排障建议）

三个 `submit_*` 脚本都会在日志开头打印：

- `submit_script`：仓库内对应脚本的绝对路径（例如 `.../unified_pipeline/scripts/submit_rolling.sh`）。**请勿与 Slurm 在计算节点上的临时副本混淆**（见下）。
- `slurm_batch_copy`（仅 Slurm 作业）：`sbatch` 在节点上实际执行的脚本副本路径，常见为 `.../spool_slurmd/job<ID>/slurm_script`，这是正常现象。
- `slurm_submit_dir`（仅 Slurm 作业）：执行 `sbatch` 时的工作目录（`SLURM_SUBMIT_DIR`）。
- `python_entry`：本次执行的 Python 入口绝对路径
- `ENV snapshot`：关键环境变量快照（模型、时间范围、步长、并行参数等）
- `CMD`：最终展开后的命令行
- `FuXi first step`：首步会打印 `mode`、`active`、`split`、`temb_mode`、`layout`，用于核对级联和时间嵌入配置

建议每次作业先核对上述字段，再进入结果分析，避免“脚本/参数/模型路径不一致”导致误判。

## 9. 输出说明

### 滚动推理输出

- 预报 NPY：`{output_root}/{Model}/ERA5_6H/*.npy`
- 对比图：`{output_root}/plots/{model_slug}/{init_tag}/*.png`
- 可选 NC：`{output_root}/{init_tag}/nc/{model_slug}/lead_*.nc`

### 评估输出

- 评估目录：`{output_root}/eval_{max_lead}h_{time_tag}/`
- 典型产物：
  - 指标汇总 CSV
  - 指标时序图
  - 可选 diff npy/nc

## 10. 设计日志对照审查结论

相对设计日志 `zk_models_refactor_a47e936d.plan.md`，当前仓库状态结论如下：

- 已落地：`config/`、`core/data`、`core/models`、`core/evaluation`、`core/monitoring`、`pipelines/verify_pipeline.py`、`pipelines/rolling_pipeline.py`、`run_verify.py`、`run_rolling.py`、`run_evaluate.py`、`scripts/*.sh`
- 实际增强：
  - 目录名采用 `zk_io/`（非设计稿中的 `io/`）
  - 增加 `run_eval_npy.py`（仅评估、无需加载模型）
  - 模型注册包含 `graphcast_cs`
  - 保留 `evaluate_models.py`、`infer_cepri_onnx.py` 作为兼容/历史工具

## 11. 常见问题

- `ValueError: 无法自动识别数据格式`
  - 在 `config/data.yaml` 显式填写 `format`
- 模型加载失败
  - 检查 `config/models.yaml` 权重路径占位符展开后是否存在
- 评估无结果
  - 检查 `date-range`、`init-hour`、`lead-step` 与输出 NPY 的 `init_tag` 是否一致
- GPU 利用率低
  - `WORLD_SIZE=1` 串行多模型时属于预期；可用 `WORLD_SIZE>1` + `--parallel-mode model|date` 提升并发
- FuXi 首步日志里 `tp_mean` 接近 0
  - 说明当前数据源未提供 `surface_tp_6h`，触发了 `tp_fallback`
  - 若需严格对齐官方 70ch 输入语义，建议在数据适配器中补充 6h 累计降水字段

## 12. 扩展指南

### 新增模型

1. 在 `core/models/` 添加新模型类并实现 `WeatherModel` 接口
2. 在 `core/models/model_registry.py` 注册模型名到类
3. 在 `config/models.yaml` 增加模型条目

### 新增数据格式

1. 在 `core/data/` 新增 `DataAdapter` 子类
2. 在 `core/data/detector.py` 的 `_FORMAT_MAP` 注册格式名
3. 在 `config/data.yaml` 增加数据源条目

### 新增评估指标

1. 在 `core/evaluation/metrics.py` 添加指标计算
2. 通过 CLI `--metrics` 或 `config/defaults.yaml` 使用

## 13. Branch and PR Workflow

为避免功能开发污染 `main`，本仓库采用分支开发 + PR 审核合并流程：

1. 新需求在独立分支开发（`feature/*`、`fix/*`、`chore/*`）
2. 每个脚本/逻辑单元独立验证后再 commit
3. 推送分支后创建 PR 到 `main`
4. `pr-gate` 检查通过 + reviewer 批准后才允许合并

推荐命令：

```bash
git switch -c feature/your-topic
# edit + validate
git add <files>
git commit -m "your message"
git push -u origin feature/your-topic
gh pr create --base main --head feature/your-topic
```

配套文件：

- `AGENTS.md`：仓库级 agent 工作契约
- `.cursor/rules/branch-pr-workflow.mdc`：Cursor agent 强约束规则
- `.github/pull_request_template.md`：PR 模板
- `.github/CODEOWNERS`：审查责任定义
- `.github/workflows/pr-gate.yml`：PR 必需检查

### Main Protection (GitHub Settings)

在 GitHub 仓库的 `Settings -> Branches -> Branch protection rules` 中为 `main` 设置：

- Require a pull request before merging
- Require approvals: `1`（可按团队提升到 `2`）
- Require status checks to pass before merging: `pr-gate / basic-checks`
- Require conversation resolution before merging
- Restrict who can push to matching branches（建议仅维护者）
- Include administrators（建议开启）

## 14. Version and Change Log

本仓库使用 `CHANGELOG.md` 记录每次主分支合并后的版本信息与测试简报。

维护要求：

1. 对功能/修复/配置/流程改动，PR 必须同步更新 `CHANGELOG.md`（或在 PR 里说明 `N/A` 原因）。
2. 优先写入 `Unreleased` 区域，合并后整理为具体版本块。
3. 每条版本记录至少包含：
   - Version
   - DateTime
   - Merged PR/Branch
   - Summary of Changes (all)
   - Test Report (brief)
   - Related Commits

版本号约定（最小语义化版本）：

- `vMAJOR.MINOR.PATCH`
- `MAJOR`：不兼容变更
- `MINOR`：向后兼容的新功能
- `PATCH`：向后兼容的修复与非功能性优化
