"""
共享 NPY 预报栈加载工具。

提供向后兼容的 `load_pred_stack()`，支持：
- 标准 .npy 文件（推荐）
- 遗留的 raw float32 memmap 文件（无 .npy 头）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_pred_stack(path: Path, n_steps: int, h: int, w: int) -> np.ndarray:
    """
    加载预报栈，返回 float32 ndarray，shape=(n_steps, H, W)。

    向后兼容：
      - 优先尝试 ``np.load(..., mmap_mode="r")``
      - 若遇 ValueError（无 .npy 头），回退为 float32 memmap 读取

    Args:
        path:     NPY 文件路径
        n_steps:  预报步数（用于 memmap fallback 的 shape）
        h:        纬向格点数
        w:        经向格点数

    Returns:
        float32 ndarray, shape (n_steps, H, W)
    """
    try:
        arr = np.load(str(path), mmap_mode="r")
    except ValueError:
        arr = np.memmap(str(path), dtype=np.float32, mode="r", shape=(n_steps, h, w))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr
