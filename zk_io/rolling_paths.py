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

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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


# ---------------------------------------------------------------
# 迁移与工具函数（供 migrate_rolling_output_layout.py 使用）
# ---------------------------------------------------------------

_DISPLAY_TO_SLUG: Dict[str, str] = {v: k for k, v in SLUG_TO_DISPLAY.items()}

# init_tag 正则：8位日期 + T + 2位小时，如 20260308T12
_INIT_TAG_RE = re.compile(r"(\d{8}T\d{2})$")
# 旧式 eval 目录名，如 eval_240h_20260308T12 或 eval_W-MAE_240h_20260308T12
_LEGACY_EVAL_RE = re.compile(r"^(eval_.+?)_(\d{8}T\d{2})$")


def output_slug_for_display(display_name: str) -> str:
    """展示名 → 输出子目录名（通常与展示名相同；提供反向查找接口）。"""
    return display_name


def output_slug_for_registry_slug(slug: str) -> str:
    """模型 registry slug → 输出子目录展示名（SLUG_TO_DISPLAY 映射）。"""
    return SLUG_TO_DISPLAY.get(slug.lower(), slug)


def plot_dir_for_registry_slug(
    output_root: Path, init_tag: str, reg_slug: str
) -> Path:
    """用 registry slug 获取对比图输出目录（转换为展示名后调用 plot_dir）。"""
    disp = output_slug_for_registry_slug(reg_slug)
    return plot_dir(output_root, init_tag, disp)


def parse_init_tag_from_npy_name(name: str) -> Optional[Tuple[str, str]]:
    """
    从 NPY 文件名中解析 (var_stem, init_tag)。

    支持：
      ``{var}_{init_tag}.npy``               → ("u10", "20260308T12")
      ``{var}_surface_{init_tag}.npy``       → ("u10_surface", "20260308T12")
    """
    stem = name
    if stem.endswith(".npy"):
        stem = stem[:-4]
    m = _INIT_TAG_RE.search(stem)
    if not m:
        return None
    init_tag = m.group(1)
    var_stem = stem[: m.start()].rstrip("_")
    return (var_stem, init_tag)


def parse_eval_dir_legacy_name(name: str) -> Optional[Tuple[str, str]]:
    """
    从旧式 eval 目录名解析 (new_base, init_tag)。

    示例：
      ``eval_240h_20260308T12``       → ("eval_240h", "20260308T12")
      ``eval_W-MAE_240h_20260308T12`` → ("eval_W-MAE_240h", "20260308T12")
    """
    m = _LEGACY_EVAL_RE.match(name)
    if not m:
        return None
    return (m.group(1), m.group(2))
