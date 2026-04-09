#!/bin/bash
# =============================================================
# 共享集群环境设置 — 所有 submit_*.sh 通过 source 引入本文件
#
# 提供：
#   - WORKDIR / ZK_LEGACY / UNIFIED_ROOT / ZK_ROOT / LOG_DIR 路径变量
#   - CONDA_ENV / CONDA_BASE / DTK_VERSION / USE_CUDA 默认值
#   - _activate_conda()      conda 初始化 + 激活
#   - _setup_dtk_or_cuda()   DTK/CUDA module 加载 + ROCm 环境变量 + unset PYTHONPATH
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

# ==============================================================
# ---- conda 激活 ----
# ==============================================================
_activate_conda() {
    if [ -n "${CONDA_BASE}" ]; then
        local init_sh="${CONDA_BASE}/etc/profile.d/conda.sh"
        if [ -f "${init_sh}" ]; then
            source "${init_sh}"
            return 0
        fi
    fi
    for candidate in \
        "/public/home/aciwgvx1jd/miniconda3" \
        "/public/home/aciwgvx1jd/anaconda3" \
        "${HOME}/miniconda3" \
        "${HOME}/anaconda3" \
        "/opt/miniconda3" \
        "/opt/conda"
    do
        if [ -f "${candidate}/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            source "${candidate}/etc/profile.d/conda.sh"
            CONDA_BASE="${candidate}"
            return 0
        fi
    done
    echo "[warn] 未找到 conda 初始化脚本，尝试直接激活..." >&2
    return 1
}

# ==============================================================
# ---- DTK / CUDA 模块加载 + ROCm 环境变量 ----
# ==============================================================
_setup_dtk_or_cuda() {
    module purge
    if [ "${USE_CUDA}" = "1" ]; then
        echo "[info] USE_CUDA=1，跳过 DTK module，使用 CUDA 模式"
        if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
            export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        else
            echo "[info] HIP_VISIBLE_DEVICES 已预设，保持: ${HIP_VISIBLE_DEVICES}"
        fi
        if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
            export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        else
            echo "[info] CUDA_VISIBLE_DEVICES 已预设，保持: ${CUDA_VISIBLE_DEVICES}"
        fi
    else
        if [ -n "${DTK_VERSION}" ]; then
            module load "compiler/dtk/${DTK_VERSION}"
            echo "[info] 已加载 compiler/dtk/${DTK_VERSION}"
        else
            echo "[warn] DTK_VERSION 为空，跳过 module load" >&2
        fi
        export LD_LIBRARY_PATH=${ROCM_PATH:+$ROCM_PATH/lib:$ROCM_PATH/hip/lib:$ROCM_PATH/llvm/lib:$ROCM_PATH/miopen/lib:}${LD_LIBRARY_PATH:-}
        export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
        export HSA_ENABLE_SDMA=0
        export HSA_ENABLE_SDMA_GANG=0
        export HSA_FORCE_FINE_GRAIN_PCIE=1
        if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
            export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        else
            echo "[info] HIP_VISIBLE_DEVICES 已预设，保持: ${HIP_VISIBLE_DEVICES}"
        fi
    fi

    # DGL / OMP 环境
    export DGL_GRAPHBOLT=0
    export DGL_USE_GRAPHBOLT=0
    export DGL_LOAD_GRAPHBOLT=0
    export OMP_NUM_THREADS=16

    # 清除 conda activate.d 注入的 /opt/onnxruntime_rocm_py310_site PYTHONPATH，
    # 让 Python 使用 conda env 内置的 ORT（torch2.4_dtk25.04_cp310_e2s/site-packages）
    unset PYTHONPATH || true
}

# ==============================================================
# ---- Python 环境校验 ----
# ==============================================================
_verify_python_env() {
    echo "[info] Python: $(python --version 2>&1)"
    python -c "import torch; print('[info] torch', torch.__version__, '| cuda?', torch.cuda.is_available())" || true
    python -c "import onnxruntime as ort; print('[info] ORT providers:', ort.get_available_providers())" || true
}

# ==============================================================
# ---- 自动执行基础设置 ----
# ==============================================================
_activate_conda || true
conda activate "${CONDA_ENV}"
echo "[info] 已激活 conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"
_setup_dtk_or_cuda
_verify_python_env
cd "${WORKDIR}"
