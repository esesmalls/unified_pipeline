"""
统一流水线路径锚点，让所有脚本都能用同一套坐标系找到代码、权重和配置文件。

本目录：ZK_Models/unified_pipeline（代码 + config）
ZK_MODELS_ROOT：上级 ZK_Models（ONNX 权重 pangu/fengwu/fuxi 等仍放于此）
GRAPH_CAST_ROOT：graphcast 示例工程根（conf/、graphcast_model/、chdir 目标）

bootstrap()：将 UNIFIED_PIPELINE_ROOT 加入 sys.path 并 chdir 到 GRAPH_CAST_ROOT，
             与其他入口文件（run_verify.py 等）的 inline 两行等价。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

UNIFIED_PIPELINE_ROOT = Path(__file__).resolve().parent
ZK_MODELS_ROOT = UNIFIED_PIPELINE_ROOT.parent
GRAPH_CAST_ROOT = ZK_MODELS_ROOT.parent


def bootstrap() -> None:
    """将 UNIFIED_PIPELINE_ROOT 加入 sys.path（幂等）并 chdir 到 GRAPH_CAST_ROOT。"""
    root = str(UNIFIED_PIPELINE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(GRAPH_CAST_ROOT)
