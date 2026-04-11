"""
ECMWF/GFS 初始场 GRIB1 格式适配器。

目录布局：
  {root}/G_YYYYMMDDHH_fh_0.grib1   分析场 / 0h 预报
  {root}/G_YYYYMMDDHH_fh_1.grib1   可选：1h 预报（含累积降水等）

YYYYMMDDHH 为起报时次 (UTC)，例如 2026032712。

高空变量 (pygrib shortName)：
  z(geopotential, m²/s²) 或 gh(geopotential height, ×9.80665)
  t / u / v / q(specific humidity) 或 r(relative humidity→q)
面场变量：2t(→t2m) / 10u(→u10) / 10v(→v10) / msl|mslet|prmsl(→msl)
TP（总降水）优先从 fh_0 读取，缺失试 fh_1，仍缺则填零。

若 GRIB 分辨率与标准 0.25°(721×1440) 不同，自动双线性降采样。

参考：https://github.com/tpys/FuXi/blob/main/make_gfs_input.py
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base_adapter import DataAdapter

try:
    import pygrib as _pg
except ImportError:
    _pg = None  # deferred check in __init__

PANGU_LEVELS: List[int] = [
    1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50,
]

_INIT_TAG_RE = re.compile(r"G_(\d{10})_fh_(\d+)\.grib1$")

_STD_LAT = np.linspace(90.0, -90.0, 721, dtype=np.float64)
_STD_LON = np.arange(0.0, 360.0, 0.25, dtype=np.float64)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def _rh_to_q(rh: np.ndarray, t_k: np.ndarray, p_hpa: float) -> np.ndarray:
    """Relative humidity → specific humidity (kg/kg).

    RH auto-detected as percent (>1.5 max → divide by 100).
    """
    rh_f = np.asarray(rh, dtype=np.float64)
    if np.nanmax(rh_f) > 1.5:
        rh_f = rh_f / 100.0
    t = np.asarray(t_k, dtype=np.float64)
    es = 6.112 * np.exp(17.67 * (t - 273.15) / (t - 29.65)) * 100.0
    p_pa = float(p_hpa) * 100.0
    qs = 0.622 * es / np.maximum(p_pa - 0.378 * es, 1.0)
    return (np.clip(rh_f, 0.0, 1.0) * qs).astype(np.float32)


def _select_safe(grbs, **kwargs):
    """pygrib select with ValueError → empty list."""
    try:
        return grbs.select(**kwargs)
    except ValueError:
        return []


def _read_pl_fields(
    grbs, short_name: str, target_levels: List[int],
) -> Optional[Dict[int, np.ndarray]]:
    """Read isobaric fields, return {level_hPa: 2D array} or None."""
    msgs = _select_safe(
        grbs, shortName=short_name, typeOfLevel="isobaricInhPa",
        level=target_levels,
    )
    if not msgs:
        msgs = _select_safe(
            grbs, shortName=short_name, typeOfLevel="isobaricInhPa",
        )
    if not msgs:
        return None
    result: Dict[int, np.ndarray] = {}
    for m in msgs:
        lv = int(m.level)
        if lv in target_levels:
            result[lv] = np.array(m.values, dtype=np.float32)
    return result or None


def _read_sfc_field(grbs, *candidates: str) -> Optional[np.ndarray]:
    """Read a surface variable, trying shortName candidates in order."""
    for sn in candidates:
        msgs = _select_safe(grbs, shortName=sn)
        if msgs:
            return np.array(msgs[0].values, dtype=np.float32)
    return None


def _interp_levels(
    x_src: np.ndarray, p_src: np.ndarray, p_tgt: np.ndarray,
) -> np.ndarray:
    """Log-pressure interpolation / reorder (same as gundong_adapter)."""
    src = np.asarray(p_src, dtype=np.float64)
    tgt = np.asarray(p_tgt, dtype=np.float64)
    if src.shape == tgt.shape and np.allclose(
        np.sort(src), np.sort(tgt), atol=1e-6,
    ):
        idx = [
            int(np.where(np.isclose(src, lv, atol=1e-6))[0][0]) for lv in tgt
        ]
        return np.asarray(x_src[idx], dtype=np.float32)
    order = np.argsort(np.log(np.clip(src, 1.0, 2000.0)))
    ps = src[order]
    xs = x_src[order]
    log_ps = np.log(np.clip(ps, 1.0, 2000.0))
    log_pt = np.log(np.clip(tgt, 1.0, 2000.0))
    out = np.empty((len(tgt),) + x_src.shape[1:], dtype=np.float32)
    flat_src = xs.reshape(xs.shape[0], -1)
    flat_out = out.reshape(out.shape[0], -1)
    for i in range(flat_src.shape[1]):
        flat_out[:, i] = np.interp(
            log_pt, log_ps, flat_src[:, i],
        ).astype(np.float32)
    return out


# ------------------------------------------------------------------
# Bilinear regridding for regular lat/lon grids
# ------------------------------------------------------------------

_RegridWeights = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray, np.ndarray]

_cached_weights: Optional[Tuple[int, int, _RegridWeights]] = None


def _build_regrid_weights(
    src_h: int, src_w: int,
    src_lat_ns: np.ndarray, src_lon: np.ndarray,
    tgt_lat_ns: np.ndarray, tgt_lon: np.ndarray,
) -> _RegridWeights:
    """Precompute bilinear interpolation indices and weights.

    src_lat_ns / tgt_lat_ns: 1-D, N→S (decreasing).
    src_lon / tgt_lon: 1-D, 0→360 (increasing).
    Returns (i0, i1, j0, j1, wi, wj) — all 1-D.
    """
    dlat = abs(float(src_lat_ns[0] - src_lat_ns[1]))
    dlon = float(src_lon[1] - src_lon[0])

    fi = (float(src_lat_ns[0]) - tgt_lat_ns) / dlat
    fi = np.clip(fi, 0, src_h - 1.0001)

    fj = (tgt_lon - float(src_lon[0])) / dlon
    fj = np.clip(fj, 0, src_w - 1.0001)

    i0 = np.floor(fi).astype(np.intp)
    j0 = np.floor(fj).astype(np.intp)
    i1 = np.minimum(i0 + 1, src_h - 1)
    j1 = np.minimum(j0 + 1, src_w - 1)

    wi = (fi - i0).astype(np.float32)
    wj = (fj - j0).astype(np.float32)
    return i0, i1, j0, j1, wi, wj


def _regrid_2d(
    data: np.ndarray, weights: _RegridWeights,
) -> np.ndarray:
    """Apply precomputed bilinear weights to a 2-D field."""
    i0, i1, j0, j1, wi, wj = weights
    wi2 = wi[:, None]
    wj2 = wj[None, :]
    return (
        (1 - wi2) * (1 - wj2) * data[i0[:, None], j0[None, :]]
        + (1 - wi2) * wj2 * data[i0[:, None], j1[None, :]]
        + wi2 * (1 - wj2) * data[i1[:, None], j0[None, :]]
        + wi2 * wj2 * data[i1[:, None], j1[None, :]]
    ).astype(np.float32)


def _regrid_3d(
    data: np.ndarray, weights: _RegridWeights,
) -> np.ndarray:
    """Apply bilinear regridding to a 3-D (nlev, H, W) field."""
    return np.stack([_regrid_2d(data[k], weights) for k in range(data.shape[0])])


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------

class ECMWFInitGribAdapter(DataAdapter):
    """ECMWF/GFS 初始场 GRIB1 适配器。"""

    FORMAT_NAME = "ecmwf_init_grib"

    def __init__(self, root: Path, **kwargs):
        if _pg is None:
            raise ImportError(
                "ECMWFInitGribAdapter requires pygrib: "
                "conda install -c conda-forge pygrib  或  pip install pygrib"
            )
        super().__init__(root, **kwargs)
        self._index = self._build_index()
        self._regrid_cache: Optional[Tuple[int, int, _RegridWeights]] = None

    # ---- index ---------------------------------------------------

    def _build_index(self) -> Dict[str, Dict[int, Path]]:
        """Scan for G_YYYYMMDDHH_fh_N.grib1 → {init_tag: {fh: path}}."""
        idx: Dict[str, Dict[int, Path]] = {}
        for f in sorted(self.root.iterdir()):
            m = _INIT_TAG_RE.match(f.name)
            if m:
                tag, fh = m.group(1), int(m.group(2))
                idx.setdefault(tag, {})[fh] = f
        return idx

    def _grib_path(
        self, date_yyyymmdd: str, hour: int, fh: int = 0,
    ) -> Path:
        tag = f"{date_yyyymmdd}{hour:02d}"
        fh_map = self._index.get(tag)
        if fh_map is None:
            raise FileNotFoundError(
                f"No GRIB files for init_tag={tag} in {self.root}"
            )
        if fh not in fh_map:
            raise FileNotFoundError(
                f"Missing fh_{fh} for init_tag={tag} "
                f"(available fh: {sorted(fh_map)})"
            )
        return fh_map[fh]

    # ---- grid helpers --------------------------------------------

    @staticmethod
    def _grid_info(grbs) -> Tuple[np.ndarray, np.ndarray, bool, Optional[np.ndarray]]:
        """Return (lat_NS, lon_0_360, need_flip_data, lon_reorder_idx).

        ``need_flip_data`` is True when GRIB data rows run S→N.
        ``lon_reorder_idx`` is an argsort index array if lon needs -180..180 → 0..360
        reordering, or None if already 0..360.
        """
        msg = grbs[1]
        lat = np.array(msg.distinctLatitudes, dtype=np.float64)
        lon = np.array(msg.distinctLongitudes, dtype=np.float64)

        # Ensure lat is N→S (descending)
        if lat.size >= 2 and lat[0] < lat[-1]:
            lat = lat[::-1].copy()

        # Ensure lon is 0..360, sorted ascending
        lon_reorder: Optional[np.ndarray] = None
        if np.any(lon < -0.01):
            lon[np.abs(lon) < 1e-6] = 0.0  # snap fp noise near 0
            lon_360 = np.where(lon < -1e-6, lon + 360.0, lon)
            lon_reorder = np.argsort(lon_360).astype(np.intp)
            lon = lon_360[lon_reorder].astype(np.float64)

        first_lat = msg["latitudeOfFirstGridPointInDegrees"]
        last_lat = msg["latitudeOfLastGridPointInDegrees"]
        need_flip = first_lat < last_lat  # data rows are S→N
        return lat, lon, need_flip, lon_reorder

    @staticmethod
    def _flip_if_needed(arr: np.ndarray, need_flip: bool) -> np.ndarray:
        if not need_flip:
            return arr
        if arr.ndim == 2:
            return arr[::-1, :]
        return arr[:, ::-1, :]

    @staticmethod
    def _reorder_lon(
        arr: np.ndarray, idx: Optional[np.ndarray],
    ) -> np.ndarray:
        """Reorder columns from -180..180 to 0..360 using precomputed index."""
        if idx is None:
            return arr
        if arr.ndim == 2:
            return arr[:, idx]
        return arr[:, :, idx]

    def _get_regrid_weights(
        self, src_h: int, src_w: int,
        src_lat: np.ndarray, src_lon: np.ndarray,
    ) -> Optional[_RegridWeights]:
        """Return regridding weights if source grid differs from standard."""
        if src_h == 721 and src_w == 1440:
            return None
        if (self._regrid_cache is not None
                and self._regrid_cache[0] == src_h
                and self._regrid_cache[1] == src_w):
            return self._regrid_cache[2]
        w = _build_regrid_weights(
            src_h, src_w, src_lat, src_lon, _STD_LAT, _STD_LON,
        )
        self._regrid_cache = (src_h, src_w, w)
        return w

    # ---- main read -----------------------------------------------

    def _prep_3d(
        self,
        raw: Dict[int, np.ndarray],
        levels: List[int],
        need_flip: bool,
        lon_idx: Optional[np.ndarray],
        rw: Optional[_RegridWeights],
        scale: float = 1.0,
    ) -> np.ndarray:
        """Stack, flip, lon-reorder, regrid a pressure-level dict in-place."""
        stk = np.stack([raw[lv] for lv in levels], axis=0)
        raw.clear()  # free source immediately
        if scale != 1.0:
            stk *= scale
        stk = self._flip_if_needed(stk, need_flip)
        stk = self._reorder_lon(stk, lon_idx)
        if rw is not None:
            stk = _regrid_3d(stk, rw)
        return stk

    def _prep_2d(
        self,
        arr: np.ndarray,
        need_flip: bool,
        lon_idx: Optional[np.ndarray],
        rw: Optional[_RegridWeights],
    ) -> np.ndarray:
        arr = self._flip_if_needed(arr, need_flip)
        arr = self._reorder_lon(arr, lon_idx)
        if rw is not None:
            arr = _regrid_2d(arr, rw)
        return np.asarray(arr, dtype=np.float32)

    def load_blob(  # noqa: C901
        self, date_yyyymmdd: str, hour: int,
    ) -> Dict[str, np.ndarray]:
        fh0_path = self._grib_path(date_yyyymmdd, hour, fh=0)
        grbs = _pg.open(str(fh0_path))
        try:
            lat, lon, need_flip, lon_idx = self._grid_info(grbs)

            pl = list(PANGU_LEVELS)

            z_f = _read_pl_fields(grbs, "z", pl)
            z_is_height = False
            if z_f is None:
                z_f = _read_pl_fields(grbs, "gh", pl)
                z_is_height = True

            t_f = _read_pl_fields(grbs, "t", pl)
            u_f = _read_pl_fields(grbs, "u", pl)
            v_f = _read_pl_fields(grbs, "v", pl)

            q_f = _read_pl_fields(grbs, "q", pl)
            r_f = None if q_f else _read_pl_fields(grbs, "r", pl)

            t2m = _read_sfc_field(grbs, "2t", "t2m")
            u10 = _read_sfc_field(grbs, "10u", "u10")
            v10 = _read_sfc_field(grbs, "10v", "v10")
            msl = _read_sfc_field(grbs, "msl", "mslet", "prmsl", "sp")
            tp = _read_sfc_field(grbs, "tp", "apcpsfc")

            sfc_z_msgs = _select_safe(
                grbs, shortName="z", typeOfLevel="surface",
            )
            sfc_z = (
                np.array(sfc_z_msgs[0].values, dtype=np.float32)
                if sfc_z_msgs else _read_sfc_field(grbs, "orog")
            )
            lsm = _read_sfc_field(grbs, "lsm")
        finally:
            grbs.close()

        # TP fallback: try fh_1
        if tp is None:
            tag = f"{date_yyyymmdd}{hour:02d}"
            fh_map = self._index.get(tag, {})
            if 1 in fh_map:
                g1 = _pg.open(str(fh_map[1]))
                try:
                    tp = _read_sfc_field(g1, "tp", "apcpsfc")
                finally:
                    g1.close()

        # ---- validate required fields ----------------------------
        missing: List[str] = []
        if z_f is None:
            missing.append("z or gh (geopotential)")
        if t_f is None:
            missing.append("t (temperature)")
        if u_f is None:
            missing.append("u (u-wind)")
        if v_f is None:
            missing.append("v (v-wind)")
        if q_f is None and r_f is None:
            missing.append("q or r (humidity)")
        if t2m is None:
            missing.append("2t / t2m")
        if u10 is None:
            missing.append("10u / u10")
        if v10 is None:
            missing.append("10v / v10")
        if msl is None:
            missing.append("msl / mslet / prmsl / sp")
        if missing:
            raise ValueError(
                f"GRIB {fh0_path} missing required fields: {missing}"
            )

        # ---- determine available levels --------------------------
        avail_levels = sorted(
            set(z_f) & set(t_f) & set(u_f) & set(v_f)  # type: ignore[arg-type]
        )
        hum_f = q_f if q_f is not None else r_f
        avail_levels = sorted(set(avail_levels) & set(hum_f))  # type: ignore[arg-type]

        # ---- compute grid transform once -------------------------
        sample = next(iter(z_f.values()))  # type: ignore[union-attr]
        src_h, src_w = sample.shape
        rw = self._get_regrid_weights(src_h, src_w, lat, lon)

        tgt_lv = np.array(PANGU_LEVELS, dtype=np.float64)
        src_lv = np.array(avail_levels, dtype=np.float64)

        # ---- process each PL variable sequentially (memory-efficient) ---
        # q/r must be processed BEFORE t is cleared (RH→q needs raw T)
        if q_f is not None:
            q_proc = self._prep_3d(q_f, avail_levels, need_flip, lon_idx, rw)
        else:
            r_stk = np.stack([r_f[lv] for lv in avail_levels], axis=0)  # type: ignore[index]
            t_stk_raw = np.stack([t_f[lv] for lv in avail_levels], axis=0)  # type: ignore[index]
            q_stk = np.empty_like(r_stk)
            for i, lv in enumerate(avail_levels):
                q_stk[i] = _rh_to_q(r_stk[i], t_stk_raw[i], float(lv))
            del r_stk, t_stk_raw, r_f
            q_stk = self._flip_if_needed(q_stk, need_flip)
            q_stk = self._reorder_lon(q_stk, lon_idx)
            if rw is not None:
                q_stk = _regrid_3d(q_stk, rw)
            q_proc = q_stk

        z_scale = 9.80665 if z_is_height else 1.0
        z13 = _interp_levels(
            self._prep_3d(z_f, avail_levels, need_flip, lon_idx, rw, z_scale),  # type: ignore[arg-type]
            src_lv, tgt_lv,
        )

        t13 = _interp_levels(
            self._prep_3d(t_f, avail_levels, need_flip, lon_idx, rw),  # type: ignore[arg-type]
            src_lv, tgt_lv,
        )
        q13 = _interp_levels(q_proc, src_lv, tgt_lv)
        del q_proc

        u13 = _interp_levels(
            self._prep_3d(u_f, avail_levels, need_flip, lon_idx, rw),  # type: ignore[arg-type]
            src_lv, tgt_lv,
        )
        v13 = _interp_levels(
            self._prep_3d(v_f, avail_levels, need_flip, lon_idx, rw),  # type: ignore[arg-type]
            src_lv, tgt_lv,
        )

        # ---- surface fields (small, process all at once) ---------
        t2m = self._prep_2d(t2m, need_flip, lon_idx, rw)
        u10 = self._prep_2d(u10, need_flip, lon_idx, rw)
        v10 = self._prep_2d(v10, need_flip, lon_idx, rw)
        msl = self._prep_2d(msl, need_flip, lon_idx, rw)
        if tp is not None:
            tp = self._prep_2d(tp, need_flip, lon_idx, rw)
        if sfc_z is not None:
            sfc_z = self._prep_2d(sfc_z, need_flip, lon_idx, rw)
        if lsm is not None:
            lsm = self._prep_2d(lsm, need_flip, lon_idx, rw)

        if rw is not None:
            lat = _STD_LAT
            lon = _STD_LON

        blob: Dict[str, np.ndarray] = {
            "surface_msl": msl,
            "surface_u10": u10,
            "surface_v10": v10,
            "surface_t2m": t2m,
            "pangu_z": z13,
            "pangu_q": q13,
            "pangu_t": t13,
            "pangu_u": u13,
            "pangu_v": v13,
            "pressure_src": src_lv.astype(np.float32),
            "lat": np.asarray(lat, dtype=np.float32),
            "lon": np.asarray(lon, dtype=np.float32),
        }
        if tp is not None:
            blob["surface_tp_6h"] = tp
        if sfc_z is not None:
            blob["surface_z_at_surface"] = sfc_z
        if lsm is not None:
            blob["land_sea_mask"] = lsm

        return blob

    # ---- catalogue -----------------------------------------------

    def list_dates(self) -> List[str]:
        dates: set[str] = set()
        for tag in self._index:
            dates.add(tag[:8])
        return sorted(dates)

    def list_init_tags(self) -> List[str]:
        """All available init tags (YYYYMMDDHH), sorted."""
        return sorted(self._index.keys())

    def available_hours(self, date_yyyymmdd: str) -> List[int]:
        """Hours with at least fh_0 present on a given date."""
        return sorted(
            int(tag[8:10])
            for tag in self._index
            if tag[:8] == date_yyyymmdd and 0 in self._index[tag]
        )
