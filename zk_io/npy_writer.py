"""
NPY 输出工具。

NpyStackWriter 使用 memmap 方式，边推理边写，避免全部积累在内存。
输出布局（见 ``zk_io/rolling_paths``）：
  {output_root}/{init_tag}/{output_slug}/{var}_{TAG}.npy
  shape: (n_steps, H, W), dtype=float32
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.lib.format import open_memmap

from zk_io.rolling_paths import npy_dir, npy_filename


class NpyStackWriter:
    """
    预分配 memmap 文件，逐步写入预报结果（节省内存）。

    Args:
        output_root:  结果根目录
        model_name:   展示名（如 "FengWu"、"GC_Official_Oper"），与 rolling 中 display_name 一致
        init_tag:     时间标签（如 "20260308T12"）
        variables:    地表变量名列表（如 ["u10","v10","t2m","msl"]）
        n_steps:      预报总步数
        shape_hw:     (H, W)，通常 (721, 1440)
        pangu_suffix: PanGu 使用 {var}_surface_{tag}.npy 命名，设为 True
    """

    def __init__(
        self,
        output_root: Path,
        model_name: str,
        init_tag: str,
        variables: List[str],
        n_steps: int,
        shape_hw: Tuple[int, int] = (721, 1440),
        pangu_suffix: bool = False,
        on_missing_var=None,
    ):
        self.output_root = Path(output_root)
        self.model_name = model_name
        self.init_tag = init_tag
        self.variables = variables
        self.n_steps = n_steps
        self.shape_hw = shape_hw
        self._pangu_suffix = pangu_suffix
        self._step_idx = 0
        self._on_missing_var = on_missing_var
        self._missing_events: List[Dict[str, object]] = []

        base = npy_dir(self.output_root, init_tag, model_name)
        base.mkdir(parents=True, exist_ok=True)
        self._memmaps: Dict[str, np.memmap] = {}
        for var in variables:
            fname = npy_filename(var, init_tag, pangu_suffix=pangu_suffix)
            path = base / fname
            # Use NumPy's .npy-aware memmap writer so outputs are standards-compliant
            # and can be loaded by np.load(..., mmap_mode="r") in evaluation scripts.
            mm = open_memmap(
                str(path),
                dtype=np.float32,
                mode="w+",
                shape=(n_steps, *shape_hw),
            )
            self._memmaps[var] = mm

    def write_step(self, step_idx: int, preds: Dict[str, np.ndarray]) -> None:
        """写入第 step_idx 步的预报值（0-based）。"""
        for var, mm in self._memmaps.items():
            if var in preds:
                arr = np.asarray(preds[var], dtype=np.float32)
                if arr.shape != self.shape_hw:
                    arr = arr[:self.shape_hw[0], :self.shape_hw[1]]
                mm[step_idx] = arr
            else:
                evt = {"step_idx": int(step_idx), "var": str(var), "strategy": "skip_write"}
                self._missing_events.append(evt)
                if self._on_missing_var is not None:
                    self._on_missing_var(step_idx, var)

    def flush(self) -> None:
        """强制刷写所有 memmap。"""
        for mm in self._memmaps.values():
            mm.flush()

    def close(self) -> None:
        self.flush()
        for mm in self._memmaps.values():
            del mm
        self._memmaps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def get_paths(self) -> Dict[str, Path]:
        base = npy_dir(self.output_root, self.init_tag, self.model_name)
        result = {}
        for var in self.variables:
            fname = npy_filename(var, self.init_tag, pangu_suffix=self._pangu_suffix)
            result[var] = base / fname
        return result

    def get_missing_events(self) -> List[Dict[str, object]]:
        return list(self._missing_events)
