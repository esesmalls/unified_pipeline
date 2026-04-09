#!/bin/bash
# =============================================================
# 共享集群环境设置 — 所有 submit_*.sh 通过 source 引入本文件
#
# 提供：
#   - WORKDIR / ZK_ROOT / UNIFIED_ROOT / LOG_DIR 路径变量
#   - CONDA_ENV / CONDA_BASE / DTK_VERSION / USE_CUDA 默认值
#   - _activate_conda()      conda 初始化 + 激活
#   - _setup_dtk_or_cuda()   DTK/CUDA module 加载 + HIP_VISIBLE_DEVICES 报告
#   - _verify_python_env()   Python / torch / ORT 环境验证输出
#
# 在本文件末尾自动执行上述三个步骤并 cd 到 WORKDIR。
# =============================================================

# ---- 工作目录 ----
WORKDIR="/public/home/aciwgvx1jd/new-onescience/onescience/examples/earth/graphcast"
ZK_LEGACY="${WORKDIR}/ZK_Models"
UNIFIED_ROOT="${ZK_LEGACY}/unified_pipeline"
ZK_ROOT="${ZK_LEGACY}"
LOG_DIR="${UNIFIED_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ---- 环境选择（通过环境变量覆盖）----
CONDA_ENV="${CONDA_ENV:-torch2.4_dtk25.04_cp310_e2s}"
CONDA_BASE="${CONDA_BASE:-}"
DTK_VERSION="${DTK_VERSION:-25.04}"
USE_CUDA="${USE_CUDA:-0}"

# ---- conda 初始化与激活 ----
_activate_conda() {
    # 如未显式指定 CONDA_BASE，依次尝试常见安装路径
    if [ -z "${CONDA_BASE}" ]; then
        for _d in \
            "${HOME}/miniconda3" \
            "${HOME}/anaconda3" \
            "/opt/miniconda3" \
            "/opt/anaconda3" \
            "/public/home/aciwgvx1jd/miniconda3"; do
            if [ -f "${_d}/etc/profile.d/conda.sh" ]; then
                CONDA_BASE="${_d}"
                break
            fi
        done
    fi

    if [ -n "${CONDA_BASE}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        return 0
    fi

    # conda 已在 PATH 中（某些节点通过 module 提供）
    if command -v conda &>/dev/null; then
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        return 0
    fi

    echo "[warn] _activate_conda: 未找到 conda 初始化脚本，尝试继续" >&2
    return 1
}

# ---- DTK / CUDA module 加载 ----
_setup_dtk_or_cuda() {
    if [ "${USE_CUDA}" = "1" ]; then
        echo "[info] USE_CUDA=1，跳过 DTK module 加载，使用系统 CUDA"
    else
        if command -v module &>/dev/null; then
            module load "compiler/dtk/${DTK_VERSION}" 2>/dev/null \
                && echo "[info] 已加载 compiler/dtk/${DTK_VERSION}" \
                || echo "[info] 警告: DTK module compiler/dtk/${DTK_VERSION} 加载失败或不可用，将尝试继续"
        else
            echo "[info] 无 module 命令，跳过 DTK 加载（非模块化节点）"
        fi
    fi

    # 报告 HIP_VISIBLE_DEVICES 状态
    local _hip="${HIP_VISIBLE_DEVICES:-}"
    local _cuda="${CUDA_VISIBLE_DEVICES:-}"
    if [ -n "${_hip}" ]; then
        echo "[info] HIP_VISIBLE_DEVICES 已预设，保持: ${_hip}"
    elif [ -n "${_cuda}" ]; then
        echo "[info] CUDA_VISIBLE_DEVICES 已预设，保持: ${_cuda}"
    else
        echo "[info] HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES 未设置，由 ROCm 自动使用所有设备"
    fi
}

# ---- Python / torch / ORT 环境验证 ----
_verify_python_env() {
    echo "[info] Python: $(python --version 2>&1)"
    python - <<'PYEOF' 2>&1 || echo "[info] 警告: torch/ORT 环境验证失败（可忽略，继续推理）"
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[info] torch {torch.__version__} | cuda? {cuda_ok}")
except ImportError:
    print("[info] torch 未安装")

try:
    import onnxruntime as ort
    print(f"[info] ORT providers: {ort.get_available_providers()}")
except ImportError:
    print("[info] onnxruntime 未安装")
PYEOF
}

# ---- 自动执行 ----
_activate_conda || true
conda activate "${CONDA_ENV}"
echo "[info] 已激活 conda 环境: ${CONDA_ENV}"
_setup_dtk_or_cuda
_verify_python_env
cd "${WORKDIR}"
