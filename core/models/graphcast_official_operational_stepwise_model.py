"""
GraphCast Official (Operational) JAX model — **stepwise** variant.

Unlike the cache-based ``graphcast_official_operational_model.py`` which
pre-computes the full rollout in a subprocess, this experimental wrapper
loads the official JAX model **in-process** and drives it one ``step()`` at
a time, matching the unified ``WeatherModel`` contract directly.

The single-step logic mirrors ``graphcast.rollout.chunked_prediction_generator``
with ``num_steps_per_chunk=1``:

  1. ``load()``  — loads checkpoint, stats, builds & JITs the predictor.
  2. ``init_state()`` — reads the gundong initial condition, calls
     ``data_utils.extract_inputs_targets_forcings`` to produce the initial
     ``inputs``, ``targets_template`` and ``forcings``; stores them in
     ``ModelState.data``.
  3. ``step()`` — slices one target-step from the stored templates, calls
     the jitted predictor, updates ``current_inputs`` using the official
     ``_get_next_inputs`` logic, and converts the output to a unified blob.

Experimental status:
  - Requires JAX + graphcast packages in the process (e2s embed must be
    activated before ``import``).
  - JAX and PyTorch may coexist on the same GPU; test stability on your
    cluster before including in multi-model production runs.
"""
from __future__ import annotations

import dataclasses
import functools
import gc
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from .base_model import ModelState, WeatherModel

_ZK_MODELS = Path(__file__).resolve().parents[3]


def _ensure_graphcast_importable() -> None:
    """Add the e2s embed site-packages to sys.path if graphcast is not yet
    importable.  This is a best-effort helper; the caller should have
    activated the embed venv or set PYTHONPATH before starting Python."""
    try:
        import graphcast  # noqa: F401
        return
    except ImportError:
        pass
    embed_sp = _ZK_MODELS / ".e2s_official_embed" / "lib" / "python3.10" / "site-packages"
    if embed_sp.is_dir() and str(embed_sp) not in sys.path:
        sys.path.insert(0, str(embed_sp))


class GraphCastOfficialStepwiseModel(WeatherModel):
    MODEL_NAME = "graphcast_official_operational_stepwise"

    _SURFACE_KEYS = {
        "10m_u_component_of_wind": "surface_u10",
        "10m_v_component_of_wind": "surface_v10",
        "2m_temperature": "surface_t2m",
        "mean_sea_level_pressure": "surface_msl",
    }

    def __init__(self):
        self._cfg: Dict = {}
        self._step_h: int = 6
        self._loaded: bool = False
        self._predictor_fn = None
        self._model_config = None
        self._task_config = None
        self._params = None
        self._hk_state: dict = {}
        self._stats: Dict[str, xr.Dataset] = {}
        self._fallback_recorder = None
        self._enabled_optional_channels: List[str] = []

    # ------------------------------------------------------------------
    # WeatherModel interface
    # ------------------------------------------------------------------

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        _ensure_graphcast_importable()

        os.environ.setdefault("JAX_PLATFORMS", "gpu")

        import jax  # noqa: E402
        from graphcast import checkpoint, graphcast as gc_mod  # noqa: E402

        self._cfg = cfg
        self._step_h = int(cfg.get("step_hours", 6))

        assets_root = Path(cfg["assets_root"])
        param_file = str(cfg["param_file"])
        param_path = assets_root / "params" / param_file
        if not param_path.is_file():
            raise FileNotFoundError(f"param file not found: {param_path}")

        t0 = time.perf_counter()
        print("[GC_Stepwise] loading checkpoint …", flush=True)
        with param_path.open("rb") as f:
            ckpt = checkpoint.load(f, gc_mod.CheckPoint)
        self._model_config = ckpt.model_config
        self._task_config = ckpt.task_config
        self._params = ckpt.params
        self._hk_state = {}

        self._stats = {
            "diffs_stddev_by_level": xr.load_dataset(
                assets_root / "stats" / "diffs_stddev_by_level.nc"
            ).compute(),
            "mean_by_level": xr.load_dataset(
                assets_root / "stats" / "mean_by_level.nc"
            ).compute(),
            "stddev_by_level": xr.load_dataset(
                assets_root / "stats" / "stddev_by_level.nc"
            ).compute(),
        }

        self._gundong_root = Path(cfg["gundong_root"])

        # ---- build & JIT the predictor ----
        sys.path.insert(0, str(_ZK_MODELS))
        import run_graphcast_official_rollout as off  # noqa: E402

        run_forward = off._build_predictor(
            self._model_config,
            self._task_config,
            self._stats["diffs_stddev_by_level"],
            self._stats["mean_by_level"],
            self._stats["stddev_by_level"],
        )

        def _with_params(fn):
            return functools.partial(fn, params=self._params, state=self._hk_state)

        def _drop_state(fn):
            return lambda **kw: fn(**kw)[0]

        self._predictor_fn = _drop_state(_with_params(jax.jit(run_forward.apply)))

        elapsed = time.perf_counter() - t0
        print(
            f"[GC_Stepwise] loaded in {elapsed:.1f}s  "
            f"resolution={self._model_config.resolution}  "
            f"mesh_size={self._model_config.mesh_size}",
            flush=True,
        )
        self._loaded = True

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Optional[datetime] = None,
    ) -> ModelState:
        if init_dt is None:
            raise ValueError("GraphCast Stepwise requires init_dt")

        import jax
        from graphcast import data_utils

        sys.path.insert(0, str(_ZK_MODELS))
        from run_graphcast_official_rollout_gundong import (
            _build_example_batch_from_gundong,
        )

        max_lead = int(self._cfg.get("max_lead_hours", 240))
        n_steps = max_lead // self._step_h

        init_dt_tz = init_dt.replace(tzinfo=timezone.utc) if init_dt.tzinfo is None else init_dt

        t0 = time.perf_counter()
        print(
            f"[GC_Stepwise] building example_batch from gundong "
            f"(init={init_dt_tz.strftime('%Y%m%dT%H')}, {n_steps} steps) …",
            flush=True,
        )
        example_batch = _build_example_batch_from_gundong(
            self._gundong_root,
            init_dt_tz,
            max_lead,
            self._step_h,
            self._task_config,
            float(self._model_config.resolution),
        )

        eval_inputs, eval_targets, eval_forcings = (
            data_utils.extract_inputs_targets_forcings(
                example_batch,
                target_lead_times=slice("6h", f"{n_steps * self._step_h}h"),
                **dataclasses.asdict(self._task_config),
            )
        )

        # ---- prepare rollout state (mirrors chunked_prediction_generator) ----
        inputs = xr.Dataset(eval_inputs)
        targets_template = xr.Dataset(eval_targets * np.nan)
        forcings = xr.Dataset(eval_forcings)

        if "datetime" in inputs.coords:
            del inputs.coords["datetime"]

        output_datetime = None
        if "datetime" in targets_template.coords:
            output_datetime = targets_template.coords["datetime"]
            del targets_template.coords["datetime"]

        if "datetime" in forcings.coords:
            del forcings.coords["datetime"]

        targets_chunk_time = targets_template.time.isel(time=slice(0, 1))

        lat = init_blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        lon = init_blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))

        elapsed = time.perf_counter() - t0
        print(
            f"[GC_Stepwise] init_state ready  "
            f"inputs.time={inputs.dims['time']}  "
            f"targets.time={targets_template.dims['time']}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        return ModelState(
            data={
                "current_inputs": inputs,
                "targets_template": targets_template,
                "forcings": forcings,
                "targets_chunk_time": targets_chunk_time,
                "output_datetime": output_datetime,
                "rng": jax.random.PRNGKey(0),
                "chunk_index": 0,
                "n_steps": n_steps,
            },
            blob=init_blob,
            lead=0,
            extra={
                "init_dt": init_dt,
                "lat": lat,
                "lon": lon,
            },
        )

    def step(self, state: ModelState) -> ModelState:
        import jax

        d = state.data
        chunk_index: int = d["chunk_index"]
        n_steps: int = d["n_steps"]
        if chunk_index >= n_steps:
            raise IndexError(
                f"Step index {chunk_index} exceeds total steps ({n_steps})"
            )

        current_inputs: xr.Dataset = d["current_inputs"]
        targets_template: xr.Dataset = d["targets_template"]
        forcings: xr.Dataset = d["forcings"]
        targets_chunk_time: xr.DataArray = d["targets_chunk_time"]
        output_datetime = d["output_datetime"]
        rng = d["rng"]

        target_slice = slice(chunk_index, chunk_index + 1)
        current_targets = targets_template.isel(time=target_slice)
        actual_target_time = current_targets.coords["time"]
        current_targets = current_targets.assign_coords(
            time=targets_chunk_time
        ).compute()

        current_forcings = forcings.isel(time=target_slice)
        current_forcings = current_forcings.assign_coords(
            time=targets_chunk_time
        ).compute()

        rng, this_rng = jax.random.split(rng)

        predictions = self._predictor_fn(
            rng=this_rng,
            inputs=current_inputs,
            targets_template=current_targets,
            forcings=current_forcings,
        )

        is_last = chunk_index == n_steps - 1
        if is_last:
            next_inputs = current_inputs
        else:
            next_frame = xr.merge([predictions, current_forcings])
            next_inputs = self._get_next_inputs(current_inputs, next_frame)
            next_inputs = next_inputs.assign_coords(
                time=current_inputs.coords["time"]
            )

        predictions = predictions.assign_coords(time=actual_target_time)
        if output_datetime is not None:
            predictions.coords["datetime"] = output_datetime.isel(
                time=target_slice
            )

        predictions_np = jax.device_get(predictions)

        blob_new = self._predictions_to_blob(predictions_np, state)

        new_data = dict(d)
        new_data["current_inputs"] = next_inputs
        new_data["rng"] = rng
        new_data["chunk_index"] = chunk_index + 1

        return ModelState(
            data=new_data,
            blob=blob_new,
            lead=state.lead + self._step_h,
            extra=state.extra,
        )

    def unload(self) -> None:
        self._predictor_fn = None
        self._params = None
        self._hk_state = {}
        self._stats = {}
        self._model_config = None
        self._task_config = None
        self._loaded = False
        gc.collect()

    def get_surface_var_names(self) -> List[str]:
        return ["u10", "v10", "t2m", "msl"]

    def get_pressure_var_names(self) -> List[str]:
        return []

    def get_step_hours(self) -> int:
        return self._step_h

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_next_inputs(
        prev_inputs: xr.Dataset,
        next_frame: xr.Dataset,
    ) -> xr.Dataset:
        """Equivalent to ``graphcast.rollout._get_next_inputs``."""
        next_inputs_keys = list(
            set(next_frame.keys()).intersection(set(prev_inputs.keys()))
        )
        next_inputs = next_frame[next_inputs_keys]
        num_inputs = prev_inputs.dims["time"]
        return (
            xr.concat(
                [prev_inputs, next_inputs], dim="time", data_vars="different"
            )
            .tail(time=num_inputs)
        )

    def _predictions_to_blob(
        self, predictions: xr.Dataset, state: ModelState
    ) -> Dict[str, np.ndarray]:
        """Convert a single-step xr.Dataset prediction to a unified blob."""
        lat = state.extra.get(
            "lat", np.linspace(90.0, -90.0, 721, dtype=np.float32)
        )
        lon = state.extra.get(
            "lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32)
        )
        blob: Dict[str, np.ndarray] = {"lat": lat, "lon": lon}
        missing_surface: List[str] = []

        for xr_key, blob_key in self._SURFACE_KEYS.items():
            if xr_key in predictions:
                da = predictions[xr_key]
                if "batch" in da.dims:
                    da = da.isel(batch=0)
                if "time" in da.dims:
                    da = da.isel(time=0)
                blob[blob_key] = np.asarray(da, dtype=np.float32)
            else:
                missing_surface.append(xr_key)

        if missing_surface:
            if self._fallback_recorder is not None:
                for ch in missing_surface:
                    self._fallback_recorder(
                        channel=ch,
                        strategy="error",
                        detail="stepwise predictions missing required surface variable",
                    )
            raise RuntimeError(
                f"GC_Stepwise missing required prediction channels: {missing_surface}"
            )

        return blob
