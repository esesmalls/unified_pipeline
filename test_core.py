#!/usr/bin/env python3
"""基础核心模块测试脚本。"""
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import GRAPH_CAST_ROOT

os.chdir(GRAPH_CAST_ROOT)

print("Step 1: importing data modules...")
from core.data.base_adapter import DataAdapter
from core.data.detector import detect_format, get_adapter
from core.data.channel_mapper import blob_to_pangu_onnx, blob_to_fengwu_69ch
print("  OK")

print("Step 2: importing model base...")
from core.models.base_model import WeatherModel, ModelState
from core.models.model_registry import ModelRegistry
print("  OK")

print("Step 3: importing evaluation metrics...")
from core.evaluation.metrics import MetricsAccumulator, compute_step_metrics
print("  OK")

print("Step 4: importing monitoring...")
from core.monitoring.hardware_logger import HardwareLogger
print("  OK")

print("Step 5: importing zk_io...")
from zk_io.npy_writer import NpyStackWriter
from zk_io.nc_writer import write_step_nc
from zk_io.plot_utils import plot_compare
print("  OK")

print("Step 6: format detection + GunDong I/O...")
_GUNDONG_ROOT = Path("/public/share/aciwgvx1jd/20260324")
_PDIR = _GUNDONG_ROOT / "pressure" / "pressure"
if _PDIR.is_dir() and any(_PDIR.glob("*_pressure.nc")):
    fmt = detect_format(_GUNDONG_ROOT)
    print(f"  gundong 格式: {fmt}")

    print("Step 7: metrics computation...")
    lats = np.linspace(90, -90, 721, dtype=np.float32)
    acc = MetricsAccumulator(lats)
    pred = np.ones((721, 1440), dtype=np.float32)
    truth = np.zeros((721, 1440), dtype=np.float32)
    diff, mvals = acc.add("FengWu", "u10", 6, pred, truth)
    print(f"  指标: {mvals}")

    print("Step 8: GunDong adapter blob + surface_tp_6h (accum fallback)...")
    adapter = get_adapter(_GUNDONG_ROOT, fmt="gundong_20260324")
    dates = adapter.list_dates()
    print(f"  可用日期 {len(dates)} 天: {dates[:3]}...")
    if dates:
        blob = adapter.load_blob(dates[0], 12)
        print(f"  blob keys: {list(blob.keys())}")
        print(f"  surface_t2m shape: {blob['surface_t2m'].shape}")
        print(f"  pangu_z shape: {blob['pangu_z'].shape}")
        if "surface_tp_6h" in blob:
            tp = blob["surface_tp_6h"]
            print(f"  surface_tp_6h shape: {tp.shape}, mean={float(np.nanmean(tp)):.6f}")
        else:
            print("  surface_tp_6h: (missing)")
    if "20260303" in dates or (_GUNDONG_ROOT / "surface" / "2026_03_03_surface_instant.nc").is_file():
        b03 = adapter.load_blob("20260303", 12)
        assert "surface_tp_6h" in b03, "20260303 blob should load tp from surface_accum"
        assert float(np.nanmean(b03["surface_tp_6h"])) > 0.0, "tp mean should be positive for real accum"
        print("  20260303@12h: surface_tp_6h from accum OK")
else:
    print("  skip (no sample tree at /public/share/aciwgvx1jd/20260324/pressure/pressure)")
    print("Step 7: metrics computation...")
    lats = np.linspace(90, -90, 721, dtype=np.float32)
    acc = MetricsAccumulator(lats)
    pred = np.ones((721, 1440), dtype=np.float32)
    truth = np.zeros((721, 1440), dtype=np.float32)
    _, mvals = acc.add("FengWu", "u10", 6, pred, truth)
    print(f"  指标: {mvals}")

print("\n所有测试通过！")
