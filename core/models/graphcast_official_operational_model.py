"""
GraphCast Official (Operational) JAX model wrapper.

Uses the official DeepMind GraphCast JAX implementation with operational
parameters (0.25 degree, 13 pressure levels, mesh 2to6).

Architecture:
  The official JAX rollout produces a *complete* prediction stack in one shot
  (all lead times at once).  Because the unified rolling pipeline calls
  ``step()`` one lead at a time, this wrapper adopts a "pre-compute + cache"
  strategy:

  1. ``load()``  – stores configuration; no heavyweight model loading.
  2. ``init_state()`` – locates (or spawns) the pre-computed NPY stacks for
     the given *init_tag*, then memory-maps them.
  3. ``step()`` – returns the next slice from the cached stacks.
  4. ``unload()`` – releases the cache.

Data source support:
  - **gundong (legacy)**: reads NetCDF from ``gundong_root`` directly.
  - **adapter-based (ecmwf_init etc.)**: the rolling pipeline sets
    ``model._data_adapter`` before ``init_state()``.  The wrapper
    pre-saves adapter blobs as NPZ files, then passes ``--blob-dir``
    to the rollout subprocess so the JAX embed can read them without
    needing pygrib or other adapter dependencies.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_model import ModelState, WeatherModel

PANGU_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


class GraphCastOfficialOperationalModel(WeatherModel):
    MODEL_NAME = "graphcast_official_operational"

    _ROLLOUT_SUBDIR = "GraphCast_official"
    _NPY_SUBDIR = "ERA5_6H"
    _SURFACE_VARS = ("u10", "v10", "t2m", "msl")

    def __init__(self):
        self._cfg: Dict = {}
        self._step_h: int = 6
        self._loaded: bool = False
        self._cache: Optional[Dict[str, np.ndarray]] = None
        self._step_idx: int = 0
        self._data_adapter: Any = None

    # ------------------------------------------------------------------
    # WeatherModel interface
    # ------------------------------------------------------------------

    def load(self, cfg: Dict, device: Any = "auto") -> None:
        self._cfg = cfg
        self._step_h = int(cfg.get("step_hours", 6))

        self._rollout_script = Path(cfg["rollout_script"])
        self._assets_root = Path(cfg["assets_root"])
        self._param_file = str(cfg["param_file"])
        self._gundong_root = Path(cfg["gundong_root"])
        self._embed_python = str(cfg.get("embed_python", ""))
        self._rollout_cache = Path(cfg["rollout_cache_dir"])
        self._max_lead = int(cfg.get("max_lead_hours", 240))

        if not self._assets_root.is_dir():
            raise FileNotFoundError(
                f"assets_root not found: {self._assets_root}"
            )

        self._loaded = True
        print(
            f"[GC_Official_Oper] config ok  "
            f"assets={self._assets_root}  param={self._param_file}",
            flush=True,
        )

    def init_state(
        self,
        init_blob: Dict,
        prev_blob: Optional[Dict] = None,
        init_dt: Optional[datetime] = None,
    ) -> ModelState:
        if init_dt is None:
            raise ValueError(
                "GraphCast Official Operational requires init_dt"
            )

        init_tag = init_dt.strftime("%Y%m%dT%H")
        npy_dir = (
            self._rollout_cache / self._ROLLOUT_SUBDIR / self._NPY_SUBDIR
        )

        if not self._all_npys_exist(npy_dir, init_tag):
            n = self._max_lead // self._step_h
            print(
                f"[GC_Official_Oper] cache miss: no NPY under {npy_dir} for "
                f"{init_tag}.  Starting one-shot official JAX rollout "
                f"({n} steps, {self._max_lead}h total) — this often takes "
                f"many minutes on GPU; rolling `step()` loop only begins "
                f"after this finishes.  Reuse the same `rollout_cache_dir` "
                f"next time to skip.",
                flush=True,
            )
            blob_dir = self._prepare_blob_inputs(init_dt) if self._data_adapter else None
            self._run_rollout(init_tag, blob_dir=blob_dir)

        if not self._all_npys_exist(npy_dir, init_tag):
            raise FileNotFoundError(
                f"Official rollout NPY missing after rollout attempt.  "
                f"Expected directory: {npy_dir}  init_tag: {init_tag}  "
                f"vars: {self._SURFACE_VARS}"
            )

        self._cache = {
            var: np.load(str(npy_dir / f"{var}_{init_tag}.npy"))
            for var in self._SURFACE_VARS
        }
        self._step_idx = 0

        lat = init_blob.get(
            "lat", np.linspace(90.0, -90.0, 721, dtype=np.float32)
        )
        lon = init_blob.get(
            "lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32)
        )
        print(
            f"[GC_Official_Oper] loaded NPY cache  "
            f"shape={self._cache['u10'].shape}  init_tag={init_tag}",
            flush=True,
        )
        return ModelState(
            blob=init_blob,
            lead=0,
            extra={"init_dt": init_dt, "lat": lat, "lon": lon},
        )

    def step(self, state: ModelState) -> ModelState:
        if self._cache is None:
            raise RuntimeError("Cache not initialised; call init_state first")

        idx = self._step_idx
        n_steps = self._cache["u10"].shape[0]
        if idx >= n_steps:
            raise IndexError(
                f"Step index {idx} exceeds cached steps ({n_steps})"
            )

        lat = state.extra.get(
            "lat", np.linspace(90.0, -90.0, 721, dtype=np.float32)
        )
        lon = state.extra.get(
            "lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32)
        )

        blob_new: Dict[str, np.ndarray] = {
            "surface_u10": np.asarray(self._cache["u10"][idx], np.float32),
            "surface_v10": np.asarray(self._cache["v10"][idx], np.float32),
            "surface_t2m": np.asarray(self._cache["t2m"][idx], np.float32),
            "surface_msl": np.asarray(self._cache["msl"][idx], np.float32),
            "lat": lat,
            "lon": lon,
        }
        self._step_idx += 1

        return ModelState(
            blob=blob_new,
            lead=state.lead + self._step_h,
            extra=state.extra,
        )

    def unload(self) -> None:
        self._cache = None
        self._step_idx = 0
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

    def _all_npys_exist(self, npy_dir: Path, init_tag: str) -> bool:
        return all(
            (npy_dir / f"{v}_{init_tag}.npy").is_file()
            for v in self._SURFACE_VARS
        )

    def _prepare_blob_inputs(self, init_dt: datetime) -> Path:
        """Pre-save adapter blobs as NPZ for the JAX rollout subprocess.

        Saves blob_YYYYMMDDHH.npz for each available time step.
        The rollout script fills missing steps with the nearest available.
        """
        adapter = self._data_adapter
        blob_dir = self._rollout_cache / "blob_inputs" / init_dt.strftime("%Y%m%dT%H")
        blob_dir.mkdir(parents=True, exist_ok=True)

        leads = list(range(
            -self._step_h,
            self._max_lead + self._step_h + 1,
            self._step_h,
        ))
        times = [init_dt + timedelta(hours=h) for h in leads]

        saved_count = 0
        for t in times:
            tag = t.strftime("%Y%m%d%H")
            out_path = blob_dir / f"blob_{tag}.npz"
            if out_path.is_file():
                saved_count += 1
                continue
            blob = adapter.load_blob_safe(t.strftime("%Y%m%d"), t.hour)
            if blob is None:
                continue
            self._save_blob_as_npz(blob, out_path)
            saved_count += 1

        print(
            f"[GC_Official_Oper] saved {saved_count}/{len(times)} blob NPZ "
            f"files to {blob_dir}",
            flush=True,
        )
        return blob_dir

    @staticmethod
    def _save_blob_as_npz(blob: Dict[str, np.ndarray], path: Path) -> None:
        """Convert canonical blob to the field layout expected by the rollout."""
        levels = np.array(PANGU_LEVELS, dtype=np.float32)
        lat = blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
        lon = blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))

        sfc_z = blob.get(
            "surface_z_at_surface",
            np.zeros_like(blob["surface_t2m"]),
        )
        lsm_arr = blob.get(
            "land_sea_mask",
            np.zeros_like(blob["surface_t2m"]),
        )

        np.savez_compressed(
            str(path),
            pressure_level=levels,
            lat=lat, lon=lon,
            z=blob["pangu_z"],
            t=blob["pangu_t"],
            u=blob["pangu_u"],
            v=blob["pangu_v"],
            q=blob["pangu_q"],
            t2m=blob["surface_t2m"],
            u10=blob["surface_u10"],
            v10=blob["surface_v10"],
            msl=blob["surface_msl"],
            sfc_z=sfc_z,
            lsm=lsm_arr,
        )

    def _run_rollout(
        self,
        init_tag: str,
        blob_dir: Optional[Path] = None,
    ) -> None:
        """Spawn the official JAX rollout as a subprocess."""
        python_bin = self._embed_python or sys.executable
        if not Path(python_bin).is_file():
            raise FileNotFoundError(
                f"embed_python not found: {python_bin}  "
                "Ensure the e2s JAX embed is prepared "
                "(see prepare_e2s_official_embed.sh)."
            )
        if not self._rollout_script.is_file():
            raise FileNotFoundError(
                f"rollout_script not found: {self._rollout_script}"
            )

        cmd = [
            python_bin,
            str(self._rollout_script),
            "--init-datetime", init_tag,
            "--lead-step-hours", str(self._step_h),
            "--max-lead-hours", str(self._max_lead),
            "--assets-root", str(self._assets_root),
            "--param-file", self._param_file,
            "--output-root", str(self._rollout_cache),
        ]
        if blob_dir:
            cmd += ["--blob-dir", str(blob_dir)]
        else:
            cmd += ["--gundong-root", str(self._gundong_root)]

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "gpu")

        t0 = time.perf_counter()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        src_label = f"blob-dir={blob_dir}" if blob_dir else f"gundong={self._gundong_root}"
        print(
            f"[GC_Official_Oper] [{ts}] JAX rollout subprocess START "
            f"(timeout 7200s, {src_label}): "
            f"{python_bin} {self._rollout_script.name} …",
            flush=True,
        )
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-20:]:
                print(f"  [rollout-stdout] {line}", flush=True)
        if result.returncode != 0:
            tail = result.stderr[-3000:] if result.stderr else "(empty)"
            raise RuntimeError(
                f"Official JAX rollout failed (exit {result.returncode}).\n"
                f"stderr tail:\n{tail}"
            )
        elapsed = time.perf_counter() - t0
        ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[GC_Official_Oper] [{ts_end}] JAX rollout subprocess END OK "
            f"(wall {elapsed:.1f}s); unified pipeline `step()` reads cache next.",
            flush=True,
        )
