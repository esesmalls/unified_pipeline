"""
滚动推理输出路径规范。

init-first 目录布局（见 README §9）::

    {output_root}/
      {init_tag}/                          # e.g. 20260308T12
        {model_slug}/                      # e.g. FengWu, GC_Official_Oper
          {var}_{init_tag}.npy             # shape (n_steps, H, W)
          {var}_surface_{init_tag}.npy     # PanGu 命名（pangu_suffix=True）
          meta_{init_tag}.json             # 推理元数据
        nc/
          {model_slug}/                    # *.nc 输出
        plots/
          {model_slug}/                    # *.png 对比图
        eval_{max_lead}h/                  # 评估结果（RMSE/MAE 等）

所有路径函数只计算路径，不创建目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


# ---------------------------------------------------------------
# 模型 slug → 展示名映射
# 展示名用于 output_root 中的子目录名（model_slug）
# ---------------------------------------------------------------
SLUG_TO_DISPLAY: Dict[str, str] = {
    "pangu":                                  "PanGu",
    "fengwu":                                 "FengWu",
    "fuxi":                                   "FuXi",
    "graphcast":                              "GraphCast",
    "graphcast_cs":                           "GraphCast_CS",
    "graphcast_official_operational":         "GC_Official_Oper",
    "graphcast_official_operational_stepwise":"GC_Stepwise",
}

# 评估/绘图中的标准模型排列顺序（按展示名）
EVAL_MODEL_ORDER: List[str] = [
    "PanGu",
    "FengWu",
    "FuXi",
    "GraphCast",
    "GraphCast_CS",
    "GC_Official_Oper",
    "GC_Stepwise",
]


# ---------------------------------------------------------------
# 路径函数
# ---------------------------------------------------------------

def npy_dir(output_root: Path, init_tag: str, model_slug: str) -> Path:
    """NPY 文件所在目录：{output_root}/{init_tag}/{model_slug}/"""
    return Path(output_root) / init_tag / model_slug


def npy_filename(var: str, init_tag: str, pangu_suffix: bool = False) -> str:
    """
    单变量 NPY 文件名。

    pangu_suffix=True  → ``{var}_surface_{init_tag}.npy``
    pangu_suffix=False → ``{var}_{init_tag}.npy``
    """
    if pangu_suffix:
        return f"{var}_surface_{init_tag}.npy"
    return f"{var}_{init_tag}.npy"


def npy_path(
    output_root: Path,
    model_slug: str,
    var: str,
    init_tag: str,
    pangu_suffix: bool = False,
) -> Path:
    """完整 NPY 文件路径。"""
    return npy_dir(output_root, init_tag, model_slug) / npy_filename(
        var, init_tag, pangu_suffix
    )


def meta_path(output_root: Path, init_tag: str, model_slug: str) -> Path:
    """推理元数据 JSON 路径：{output_root}/{init_tag}/{model_slug}/meta_{init_tag}.json"""
    return npy_dir(output_root, init_tag, model_slug) / f"meta_{init_tag}.json"


def nc_dir(output_root: Path, init_tag: str, model_slug: str) -> Path:
    """NC 输出目录：{output_root}/{init_tag}/nc/{model_slug}/"""
    return Path(output_root) / init_tag / "nc" / model_slug


def plot_dir(output_root: Path, init_tag: str, model_slug: str) -> Path:
    """对比图输出目录：{output_root}/{init_tag}/plots/{model_slug}/"""
    return Path(output_root) / init_tag / "plots" / model_slug


def eval_downscale_dir(output_root: Path, init_tag: str, max_lead: int) -> Path:
    """评估结果目录：{output_root}/{init_tag}/eval_{max_lead}h/"""
    return Path(output_root) / init_tag / f"eval_{max_lead}h"
