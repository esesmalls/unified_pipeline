"""
Pangu-Weather ONNX 模型封装。

支持 6h/24h 阶梯调度：
  - lead % 24 == 0 → 24h ONNX
  - 其余 → 6h ONNX
每次 step() 前进 6h（使用适当 ONNX）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel
from ..data.channel_mapper import blob_to_pangu_onnx, pangu_onnx_to_blob

_ZK_ROOT = Path(__file__).resolve().parents[2]
if str(_ZK_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZK_ROOT))

from infer_cepri_onnx import (  # noqa: E402
    create_session,
    log_ort_session,
    pangu_one_step,
    pick_providers,
)


class PanguModel(WeatherModel):
    MODEL_NAME = "pangu"

    def __init__(self):
        self._sessions: Dict[str, Any] = {}
        self._loaded = False
        self._step_h = 6
        self._scheduler_mode = "hybrid_24h"

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        paths = cfg.get("paths", {})
        self._scheduler_mode = str(cfg.get("scheduler_mode", "hybrid_24h")).strip().lower()
        if self._scheduler_mode not in ("hybrid_24h", "six_hour_only"):
            raise ValueError(
                f"Pangu: 非法 scheduler_mode={self._scheduler_mode}，"
                "仅支持 hybrid_24h | six_hour_only"
            )
        providers = pick_providers(str(device) if not isinstance(device, str) else device)
        for key, p_str in paths.items():
            p = Path(p_str)
            if p.is_file():
                self._sessions[key] = create_session(p, providers)
        if "6h" not in self._sessions:
            raise FileNotFoundError("Pangu: 找不到 pangu_weather_6.onnx，请检查 config/models.yaml 路径")
        for key, sess in self._sessions.items():
            log_ort_session(f"Pangu/{key}", sess)
        print(f"[Pangu] scheduler_mode={self._scheduler_mode}", flush=True)
        self._loaded = True

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Any = None,
    ) -> ModelState:
        p, s = blob_to_pangu_onnx(init_blob)
        return ModelState(
            data={"p": p, "s": s},
            blob=init_blob,
            lead=0,
        )

    def step(self, state: ModelState) -> ModelState:
        next_lead = state.lead + self._step_h
        # 可配置策略：
        # - six_hour_only: 永远使用 6h
        # - hybrid_24h:    +24h/+48h/... 使用 24h
        use_24 = (
            self._scheduler_mode == "hybrid_24h"
            and (next_lead % 24 == 0)
            and ("24h" in self._sessions)
        )
        sess_key = "24h" if use_24 else "6h"
        sess = self._sessions[sess_key]
        # 仅首步与「第一次」到达 +24h 预报时效时各打一行，不在 48h/72h/... 重复打
        if next_lead == self._step_h or next_lead == 24:
            print(f"[Pangu] lead={next_lead}h uses {sess_key}", flush=True)

        p_new, s_new = pangu_one_step(sess, state.data["p"], state.data["s"])
        blob_new = pangu_onnx_to_blob(p_new, s_new)
        blob_new["lat"] = state.blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        blob_new["lon"] = state.blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))
        return ModelState(
            data={"p": p_new, "s": s_new},
            blob=blob_new,
            lead=next_lead,
        )

    def unload(self) -> None:
        """释放所有 ONNX Session，归还 GPU/CPU 显存。"""
        import gc
        self._sessions.clear()
        self._loaded = False
        gc.collect()

    def get_surface_var_names(self) -> List[str]:
        return ["msl", "u10", "v10", "t2m"]

    def get_pressure_var_names(self) -> List[str]:
        return ["z", "q", "t", "u", "v"]

    def get_step_hours(self) -> int:
        return self._step_h
