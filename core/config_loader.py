"""
集中配置加载器。

提供对 config/defaults.yaml、config/models.yaml、config/data.yaml 的统一读取接口，
避免各 run_*.py 入口重复实现相同的 YAML 加载逻辑。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# 项目根目录（本文件位于 core/ 下）
_UNIFIED_ROOT = Path(__file__).resolve().parent.parent


def load_defaults(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    读取并返回 config/defaults.yaml 的内容（dict）。

    Args:
        cfg_path: 覆盖默认路径（主要用于单元测试）

    Returns:
        YAML 内容 dict；文件不存在时返回空 dict。
    """
    path = cfg_path if cfg_path is not None else _UNIFIED_ROOT / "config" / "defaults.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_all_enabled_models(
    models_cfg_path: Optional[Path] = None,
) -> List[str]:
    """
    读取 config/models.yaml，返回所有 ``enabled: true`` 的模型 slug 列表。

    Args:
        models_cfg_path: 覆盖默认路径

    Returns:
        List[str] of model slug names (e.g. ["pangu", "fengwu", ...])
    """
    path = models_cfg_path if models_cfg_path is not None else _UNIFIED_ROOT / "config" / "models.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return [
        name
        for name, cfg in raw.get("models", {}).items()
        if cfg.get("enabled", True)
    ]


def resolve_output_root(
    data_source: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    根据数据源名称推导默认输出根目录。

    优先级：
      1. 如果 ``data_source`` 本身是一个已存在的目录路径，则基于其名称派生输出路径
      2. 查找 config/data.yaml 中 ``sources.<data_source>.output_root`` 字段（扩展预留）
      3. 回退到 defaults.yaml 中 ``pipeline.rolling.output_root``
      4. 硬编码兜底：``/public/share/aciwgvx1jd/LYQ/<data_source>``

    Args:
        data_source:  数据源 key（config/data.yaml 中的键）或路径字符串
        defaults:     load_defaults() 返回值（None 时自动加载）

    Returns:
        Path 对象
    """
    if defaults is None:
        defaults = load_defaults()

    # --- 从 defaults.yaml 读取 ---
    r_def = defaults.get("pipeline", {}).get("rolling", {}).get("output_root")
    if r_def:
        return Path(r_def)

    # --- 兜底：按数据源名称构造路径 ---
    ds_slug = Path(data_source).name if Path(data_source).exists() else data_source
    return Path("/public/share/aciwgvx1jd/LYQ") / ds_slug


def load_data_sources(cfg_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    读取 config/data.yaml 的 sources 字典。

    Returns:
        dict mapping source_name → source config dict
    """
    path = cfg_path if cfg_path is not None else _UNIFIED_ROOT / "config" / "data.yaml"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("sources", {})
