#!/bin/bash
# 2026-03-03 12Z：五模型 verify + rolling（含评估）。请在 unified_pipeline 目录下执行。
set -euo pipefail

echo "[warn] DEPRECATED: 推荐直接使用 submit_verify.sh / submit_rolling.sh / submit_evaluate.sh 并通过环境变量传参。"
echo "[warn] 此脚本仅保留兼容历史流程，不建议新任务继续使用。"

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${UNIFIED_ROOT}"

export MODELS="pangu fengwu fuxi graphcast graphcast_cs"
export DATA_SOURCE="gundong_20260324"
export DATE="20260303"
export HOUR="12"
export DATE_RANGE="20260303"
export INIT_HOUR="12"
export ENABLE_EVAL="1"
export SKIP_PLOTS="${SKIP_PLOTS:-0}"
export ALL_SURFACE="${ALL_SURFACE:-1}"
export NUM_STEPS="${NUM_STEPS:-2}"
export MAX_LEAD="${MAX_LEAD:-240}"
export LEAD_STEP="${LEAD_STEP:-6}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export PARALLEL_MODE="${PARALLEL_MODE:-auto}"

mkdir -p logs

echo "[info] Job1: verify"
J1=$(sbatch --export=ALL --parsable scripts/submit_verify.sh)
echo "[info] verify job_id=${J1}"

echo "[info] Job2: rolling + eval"
J2=$(sbatch --export=ALL --parsable scripts/submit_rolling.sh)
echo "[info] rolling job_id=${J2}"

echo "[info] logs: ${UNIFIED_ROOT}/logs/verify_${J1}.out ... (或 Slurm 提交目录下 logs/)"
