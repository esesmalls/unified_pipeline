#!/bin/bash
# =============================================================
# 功能二：滚动推理（含可选定量评估）Slurm 提交脚本
#
# 默认行为（如无参数覆盖）：
#   - 全部启用模型
#   - gundong_20260324 数据源
#   - 20260308 单日，12h 起报，6h步长，240h
#   - 全部地表变量
#   - 不开启评估（需要则设 ENABLE_EVAL=1）
#
# 仅基于已有 NPY 重跑合并评估（无 ONNX），在计算节点手动执行示例：
#   cd /public/home/.../graphcast/ZK_Models/unified_pipeline
#   conda activate torch2.4_dtk25.04_cp310_e2s
#   python run_eval_npy.py --data-source gundong_20260324 \\
#     --date-range 20260310 --init-hour 12 --max-lead 240 \\
#     --models pangu fengwu fuxi graphcast --output-root /path/to/LYQ/gundong
#
# 用法示例：
#   sbatch scripts/submit_rolling.sh
#
#   # 自定义参数
#   MODELS="fengwu fuxi" DATE_RANGE="20260301:20260318" \
#   ENABLE_EVAL=1 SAVE_DIFF=1 \
#   sbatch scripts/submit_rolling.sh
#
#   # 指定不同 conda 环境
#   CONDA_ENV=my_env DTK_VERSION=25.04 sbatch scripts/submit_rolling.sh
#
#   # 多卡并行（通过 torchrun 或 srun --ntasks 分片日期）
#   WORLD_SIZE=8 sbatch --ntasks=8 scripts/submit_rolling.sh
# =============================================================
#SBATCH -J zk_rolling
#SBATCH -p kshkexclu01
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=dcu:8
#SBATCH -o logs/rolling_%j.out
#SBATCH -e logs/rolling_%j.err

set -euo pipefail

# ---- 共享环境设置（conda / DTK / 路径）----
# Slurm copies batch scripts to spool; $0 is not the repo path. Use absolute path.
_COMMON_SH="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast/ZK_Models/unified_pipeline/scripts/_common.sh"
if [ ! -f "${_COMMON_SH}" ]; then
    # Fallback: running from repo dir directly (non-Slurm)
    _COMMON_SH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
fi
source "${_COMMON_SH}"

# ---- 可配置参数 ----
MODELS="${MODELS:-all}"
DATA_SOURCE="${DATA_SOURCE:-gundong_20260324}"
DATE_RANGE="${DATE_RANGE:-20260308}"
INIT_HOUR="${INIT_HOUR:-12}"
LEAD_STEP="${LEAD_STEP:-6}"
MAX_LEAD="${MAX_LEAD:-240}"
VARIABLES="${VARIABLES:-}"                # 空=模型默认全地表变量
if [ -z "${OUTPUT_ROOT+x}" ]; then
  if [ "${DATA_SOURCE}" = "ecmwf_init" ] || [[ "${DATA_SOURCE}" == *"ecmwf_init"* ]]; then
    OUTPUT_ROOT="/public/share/aciwgvx1jd/LYQ/ecmwf_init"
  else
    OUTPUT_ROOT="/public/share/aciwgvx1jd/LYQ/gundong/GunDong_Infer_result_12h"
  fi
fi
DEVICE="${DEVICE:-auto}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
# 1=run_rolling.py --cpu-timing-exclude-plots（日志 [timing] CPU 不含逐步出图）
CPU_TIMING_EXCLUDE_PLOTS="${CPU_TIMING_EXCLUDE_PLOTS:-0}"
SAVE_NC="${SAVE_NC:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-0}"           # 1=开启内嵌评估
SAVE_DIFF="${SAVE_DIFF:-0}"              # 1=保存 diff npy（需 ENABLE_EVAL=1）
SAVE_DIFF_NC="${SAVE_DIFF_NC:-0}"        # 1=保存 diff nc（需 ENABLE_EVAL=1）
METRICS="${METRICS:-W-MAE W-RMSE}"
# 多分辨率评估（需 ENABLE_EVAL=1；细网格 ecmwf_init GRIB 真值时可用）
# EVAL_MODES 例: "downscale fidelity_nn upscale" 或仅 "fidelity_nn"（跳过对齐降尺度 CSV）
EVAL_MODES="${EVAL_MODES:-}"
FIDELITY_MAX_DEG="${FIDELITY_MAX_DEG:-}"
AUTO_MULTIGRID="${AUTO_MULTIGRID:-0}"     # 1=未设 EVAL_MODES 时自动加 fidelity_nn+upscale
# GC 官方模型输入构造策略（models.yaml 默认为 init_only；可用 GC_BLOB_INPUT_MODE 覆盖）：
#   init_only: 仅预存[-step,0]初始化 blob（默认）
#   full_window: 预存[-step,0,+6h..+max+step] blob，缺失时 forward reuse
GC_BLOB_INPUT_MODE="${GC_BLOB_INPUT_MODE:-}"

# ---- 真值缓存策略 ----
# TRUTH_CACHE_MODE:          auto（默认）| memory | disk
#   auto   — 运行时估算内存，自动选择 memory 或 disk
#   memory — 进程级缓存，模型切换不清理，适合内存宽裕场景
#   disk   — 磁盘缓存（output_root/_truth_cache/{job_id}/rank{R}/…），内存受限时用
# 当 SKIP_PLOTS=1 且 ENABLE_EVAL=0 时，真值路径完全短路，以下参数不生效
TRUTH_CACHE_MODE="${TRUTH_CACHE_MODE:-auto}"
TRUTH_CACHE_BUDGET_RATIO="${TRUTH_CACHE_BUDGET_RATIO:-0.4}"
TRUTH_CACHE_SAFETY_FACTOR="${TRUTH_CACHE_SAFETY_FACTOR:-1.3}"
KEEP_TRUTH_CACHE="${KEEP_TRUTH_CACHE:-0}"    # 1=作业结束后保留磁盘缓存（调试用）

# ---- 多卡并行策略 ----
# WORLD_SIZE:      并行进程数（默认 1=单进程）
# PARALLEL_MODE:   auto | date | model
#   auto  — 日期数 >= WORLD_SIZE 时按日期分片，否则按模型分片
#   date  — 多日任务：18天 × WORLD_SIZE=8 → 每卡 2~3 天
#   model — 单日任务：1天 × 4模型 × WORLD_SIZE=4 → 每卡 1 个模型
WORLD_SIZE="${WORLD_SIZE:-1}"
PARALLEL_MODE="${PARALLEL_MODE:-auto}"
# 多作业同时跑 torchrun 时避免同节点 master 端口冲突（例：并行作业 B 可设 MASTER_PORT=29501）
MASTER_PORT="${MASTER_PORT:-29500}"
# Slurm 将批处理脚本复制到 spool（$0 常为 .../slurm_script），不代表仓库路径
SCRIPT_PATH="${UNIFIED_ROOT}/scripts/submit_rolling.sh"
PY_ENTRY="${UNIFIED_ROOT}/run_rolling.py"

# ---- 在作业日志中输出可复制的 sbatch 复现命令 ----
# 注意：sacct 的 SubmitLine 不含「VAR=value sbatch …」里的环境变量前缀，此处用当前生效变量拼接。
_sb_q() { printf '%q' "$1"; }

_emit_sbatch_repro_rolling() {
    shopt -s extglob 2>/dev/null || true
    local wd="" sl="" raw=""
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        raw=$(sacct -j "${SLURM_JOB_ID}" -X -n --format=WorkDir,SubmitLine --parsable2 --noheader 2>/dev/null | head -1)
        if [[ -n "${raw}" ]]; then
            wd="${raw%%|*}"
            sl="${raw#*|}"
        fi
    fi
    local subdir="${SLURM_SUBMIT_DIR:-${wd}}"
    [[ -z "${subdir}" ]] && subdir="${UNIFIED_ROOT}"
    echo "[info] sbatch_repro_workdir=${wd:-n/a}"
    echo "[info] sbatch_repro_submit_dir=$(_sb_q "${subdir}")"
    [[ -n "${sl}" ]] && echo "[info] sbatch_repro_slurm_submit_line=${sl}"
    local ep=""
    ep+="CONDA_ENV=$(_sb_q "${CONDA_ENV}") "
    if [[ -n "${CONDA_BASE:-}" ]]; then ep+="CONDA_BASE=$(_sb_q "${CONDA_BASE}") "; fi
    ep+="DTK_VERSION=$(_sb_q "${DTK_VERSION}") "
    ep+="USE_CUDA=$(_sb_q "${USE_CUDA}") "
    ep+="MODELS=$(_sb_q "${MODELS}") "
    ep+="DATA_SOURCE=$(_sb_q "${DATA_SOURCE}") "
    ep+="DATE_RANGE=$(_sb_q "${DATE_RANGE}") "
    ep+="INIT_HOUR=$(_sb_q "${INIT_HOUR}") "
    ep+="LEAD_STEP=$(_sb_q "${LEAD_STEP}") "
    ep+="MAX_LEAD=$(_sb_q "${MAX_LEAD}") "
    ep+="OUTPUT_ROOT=$(_sb_q "${OUTPUT_ROOT}") "
    ep+="DEVICE=$(_sb_q "${DEVICE}") "
    ep+="SKIP_PLOTS=$(_sb_q "${SKIP_PLOTS}") "
    ep+="CPU_TIMING_EXCLUDE_PLOTS=$(_sb_q "${CPU_TIMING_EXCLUDE_PLOTS}") "
    ep+="SAVE_NC=$(_sb_q "${SAVE_NC}") "
    ep+="ENABLE_EVAL=$(_sb_q "${ENABLE_EVAL}") "
    ep+="SAVE_DIFF=$(_sb_q "${SAVE_DIFF}") "
    ep+="SAVE_DIFF_NC=$(_sb_q "${SAVE_DIFF_NC}") "
    ep+="METRICS=$(_sb_q "${METRICS}") "
    if [[ -n "${EVAL_MODES}" ]]; then
        ep+="EVAL_MODES=$(_sb_q "${EVAL_MODES}") "
    fi
    if [[ -n "${FIDELITY_MAX_DEG}" ]]; then
        ep+="FIDELITY_MAX_DEG=$(_sb_q "${FIDELITY_MAX_DEG}") "
    fi
    ep+="AUTO_MULTIGRID=$(_sb_q "${AUTO_MULTIGRID}") "
    if [[ -n "${GC_BLOB_INPUT_MODE}" ]]; then
        ep+="GC_BLOB_INPUT_MODE=$(_sb_q "${GC_BLOB_INPUT_MODE}") "
    fi
    ep+="WORLD_SIZE=$(_sb_q "${WORLD_SIZE}") "
    ep+="PARALLEL_MODE=$(_sb_q "${PARALLEL_MODE}") "
    ep+="MASTER_PORT=$(_sb_q "${MASTER_PORT}") "
    ep+="TRUTH_CACHE_MODE=$(_sb_q "${TRUTH_CACHE_MODE}") "
    ep+="TRUTH_CACHE_BUDGET_RATIO=$(_sb_q "${TRUTH_CACHE_BUDGET_RATIO}") "
    ep+="TRUTH_CACHE_SAFETY_FACTOR=$(_sb_q "${TRUTH_CACHE_SAFETY_FACTOR}") "
    ep+="KEEP_TRUTH_CACHE=$(_sb_q "${KEEP_TRUTH_CACHE}") "
    if [[ -n "${VARIABLES}" ]]; then
        ep+="VARIABLES=$(_sb_q "${VARIABLES}") "
    fi
    ep="${ep%%+([[:space:]])}"
    echo "[info] sbatch_repro_env_snapshot=${ep}"
    local sbrest=""
    if [[ "${sl}" =~ ^sbatch[[:space:]]+(.+)[[:space:]]([^[:space:]]+\.sh)$ ]]; then
        sbrest="${BASH_REMATCH[1]}"
    fi
    if [[ -n "${sbrest}" ]]; then
        echo "[info] sbatch_repro_copy_paste: cd $(_sb_q "${subdir}") && ${ep} sbatch ${sbrest} $(_sb_q "${SCRIPT_PATH}")"
    else
        echo "[info] sbatch_repro_copy_paste: cd $(_sb_q "${subdir}") && ${ep} sbatch $(_sb_q "${SCRIPT_PATH}")"
    fi
}

# ---- DCU / CPU 说明（Slurm 配额）----
# 逐模型串行 + WORLD_SIZE=1 时，同一时间只有 1 张 DCU 在跑推理，其余卡空闲属正常；
# rocm-smi 固定间隔采样也可能在步间采到 0% 利用率。
# 多卡并行：WORLD_SIZE>1 且 PARALLEL_MODE=model（按模型分卡）或 date（按日期分卡）时，
# 须同时降低 #SBATCH --cpus-per-task 或提高节点 CPU 配额，使
#   ntasks * cpus-per-task <= 节点可用核数
# 例如 8 进程时可设：  sbatch --ntasks-per-node=8 --cpus-per-task=8 ...

# ---- 打印运行信息 ----
echo "=========================================="
echo "[info] job=${SLURM_JOB_ID:-local}"
echo "[info] date=$(date)"
echo "[info] conda_env=${CONDA_ENV}"
echo "[info] dtk_version=${DTK_VERSION:-none}"
echo "[info] submit_script=${SCRIPT_PATH}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[info] slurm_batch_copy=$(readlink -f "$0" 2>/dev/null || echo "$0")"
  echo "[info] slurm_submit_dir=${SLURM_SUBMIT_DIR:-n/a}"
fi
echo "[info] python_entry=${PY_ENTRY}"
echo "[info] models=${MODELS}"
echo "[info] data_source=${DATA_SOURCE}"
echo "[info] date_range=${DATE_RANGE}"
echo "[info] init_hour=${INIT_HOUR}  lead_step=${LEAD_STEP}  max_lead=${MAX_LEAD}"
echo "[info] output=${OUTPUT_ROOT}"
echo "[info] enable_eval=${ENABLE_EVAL}"
echo "[info] skip_plots=${SKIP_PLOTS}  cpu_timing_exclude_plots=${CPU_TIMING_EXCLUDE_PLOTS}"
echo "[info] truth_cache_mode=${TRUTH_CACHE_MODE}  budget_ratio=${TRUTH_CACHE_BUDGET_RATIO}  safety_factor=${TRUTH_CACHE_SAFETY_FACTOR}  keep=${KEEP_TRUTH_CACHE}"
echo "=========================================="

# ---- 构建参数 ----
ARGS=()
ARGS+=(--data-source "${DATA_SOURCE}")
ARGS+=(--date-range "${DATE_RANGE}")
ARGS+=(--init-hour "${INIT_HOUR}")
ARGS+=(--lead-step "${LEAD_STEP}")
ARGS+=(--max-lead "${MAX_LEAD}")
ARGS+=(--device "${DEVICE}")
ARGS+=(--output-root "${OUTPUT_ROOT}")
ARGS+=(--metrics ${METRICS})

if [ "${MODELS}" = "all" ]; then
    ARGS+=(--models all)
else
    ARGS+=(--models ${MODELS})
fi

if [ -n "${VARIABLES}" ]; then
    ARGS+=(--variables ${VARIABLES})
fi
if [ "${SKIP_PLOTS}" = "1" ]; then
    ARGS+=(--skip-plots)
fi
if [ "${CPU_TIMING_EXCLUDE_PLOTS}" = "1" ]; then
    ARGS+=(--cpu-timing-exclude-plots)
fi
if [ "${SAVE_NC}" = "1" ]; then
    ARGS+=(--save-nc)
fi
if [ "${ENABLE_EVAL}" = "1" ]; then
    ARGS+=(--enable-eval)
fi
if [ -n "${EVAL_MODES}" ]; then
    ARGS+=(--eval-modes ${EVAL_MODES})
fi
if [ -n "${FIDELITY_MAX_DEG}" ]; then
    ARGS+=(--fidelity-max-deg "${FIDELITY_MAX_DEG}")
fi
if [ "${AUTO_MULTIGRID}" = "1" ]; then
    ARGS+=(--auto-multigrid)
fi
if [ "${SAVE_DIFF}" = "1" ]; then
    ARGS+=(--save-diff)
fi
if [ "${SAVE_DIFF_NC}" = "1" ]; then
    ARGS+=(--save-diff-nc)
fi

ARGS+=(--parallel-mode "${PARALLEL_MODE}")

# ---- 真值缓存参数 ----
ARGS+=(--truth-cache-mode "${TRUTH_CACHE_MODE}")
ARGS+=(--truth-cache-budget-ratio "${TRUTH_CACHE_BUDGET_RATIO}")
ARGS+=(--truth-cache-safety-factor "${TRUTH_CACHE_SAFETY_FACTOR}")
if [ "${KEEP_TRUTH_CACHE}" = "1" ]; then
    ARGS+=(--keep-truth-cache)
fi

echo "[info] parallel_mode=${PARALLEL_MODE}  world_size=${WORLD_SIZE}  master_port=${MASTER_PORT}"
echo "[info] ENV snapshot: MODELS=${MODELS} DATA_SOURCE=${DATA_SOURCE} DATE_RANGE=${DATE_RANGE} INIT_HOUR=${INIT_HOUR} LEAD_STEP=${LEAD_STEP} MAX_LEAD=${MAX_LEAD} WORLD_SIZE=${WORLD_SIZE} PARALLEL_MODE=${PARALLEL_MODE} ENABLE_EVAL=${ENABLE_EVAL} CPU_TIMING_EXCLUDE_PLOTS=${CPU_TIMING_EXCLUDE_PLOTS}"
if [ -n "${GC_BLOB_INPUT_MODE}" ]; then
  export GC_BLOB_INPUT_MODE
  echo "[info] GC_BLOB_INPUT_MODE=${GC_BLOB_INPUT_MODE}"
fi
_emit_sbatch_repro_rolling
echo "[info] CMD(base): python ${PY_ENTRY} ${ARGS[*]}"
echo "[info] 开始时间: $(date)"

if [ "${WORLD_SIZE}" -gt "1" ]; then
    echo "[info] 多进程模式: WORLD_SIZE=${WORLD_SIZE}  PARALLEL_MODE=${PARALLEL_MODE}"
    torchrun \
        --nproc_per_node="${WORLD_SIZE}" \
        --master_port="${MASTER_PORT}" \
        "${PY_ENTRY}" "${ARGS[@]}"
else
    python "${PY_ENTRY}" "${ARGS[@]}"
fi

echo "[info] 完成时间: $(date)"
