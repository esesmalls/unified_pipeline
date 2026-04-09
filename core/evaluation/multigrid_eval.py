"""
多分辨率评估：保真（fidelity）与对齐升尺度（upscale）。

是否执行保真/升尺度由 **真值网格与模型（预报）网格是否一致** 决定：若真值
``lat/lon`` 与 ``coarse_lat/coarse_lon`` 维数相同且在容差内对齐，则与对齐降尺度
等价，默认跳过；仅当二者不一致时才计算（与数据格式无关，只要适配器能读出真值）。

细网格真值优先使用适配器的 ``load_surface_native_for_valid_time``（若存在），
否则退回 ``load_blob_for_valid_time``（与降尺度评估同一套网格）。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from core.data.channel_mapper import extract_surface_vars
from core.data.ecmwf_init_grib_adapter import _build_regrid_weights, _regrid_2d
from core.data.surface_units import harmonize_surface_pair
from core.evaluation.metrics import compute_step_metrics_masked
from zk_io.plot_utils import plot_compare

_EVAL_MODEL_ORDER = [
    "PanGu", "FengWu", "FuXi", "GraphCast", "GraphCast_CS",
    "GC_Official_Oper", "GC_Stepwise",
]

_SLUG_TO_DISPLAY = {
    "pangu": "PanGu",
    "fengwu": "FengWu",
    "fuxi": "FuXi",
    "graphcast": "GraphCast",
    "graphcast_cs": "GraphCast_CS",
    "graphcast_official_operational": "GC_Official_Oper",
    "graphcast_official_operational_stepwise": "GC_Stepwise",
}


def load_truth_blob_for_multigrid(
    truth_adapter: Any,
    valid_dt: datetime,
) -> Optional[Dict[str, np.ndarray]]:
    """
    读取用于保真/升尺度的真值 blob：优先 ``load_surface_native_for_valid_time``（细网格），
    否则 ``load_blob_for_valid_time``（与对齐降尺度相同网格）。
    """
    fn = getattr(truth_adapter, "load_surface_native_for_valid_time", None)
    if callable(fn):
        blob = fn(valid_dt)
        if blob is not None:
            return blob
    return truth_adapter.load_blob_for_valid_time(valid_dt)


def truth_grid_matches_model(
    truth_lat: np.ndarray,
    truth_lon: np.ndarray,
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    *,
    atol_deg: float = 1e-3,
) -> bool:
    """
    真值 1D lat/lon 与模型网格一致时返回 True（应跳过保真/升尺度）。

    维数不同或坐标在容差外对齐则返回 False（可跑多分辨率评估）。
    """
    tl = np.asarray(truth_lat, dtype=np.float64).ravel()
    tc = np.asarray(coarse_lat, dtype=np.float64).ravel()
    gl = np.asarray(truth_lon, dtype=np.float64).ravel()
    gc = np.asarray(coarse_lon, dtype=np.float64).ravel()
    if tl.shape != tc.shape or gl.shape != gc.shape:
        return False
    if not np.allclose(tl, tc, rtol=0.0, atol=float(atol_deg)):
        return False
    dlon = np.abs(gl - gc)
    dlon = np.minimum(dlon, 360.0 - dlon)
    return bool(np.all(dlon <= float(atol_deg)))


def _angular_lon_delta_deg(lon_a: np.ndarray, lon_b: np.ndarray) -> np.ndarray:
    d = np.abs(np.asarray(lon_a, dtype=np.float64) - np.asarray(lon_b, dtype=np.float64))
    return np.minimum(d, 360.0 - d)


def _truth_on_coarse_separable_nn(
    truth_f: np.ndarray,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_f: np.ndarray,
    lon_f: np.ndarray,
    mode: str,
    max_deg: float,
    exact_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在模型网格上填真值（可分离最近邻索引），并给出有效 mask。

    mode: ``nn_threshold`` | ``exact``
    """
    Hc, Wc = len(lat_c), len(lon_c)
    lat_c64 = np.asarray(lat_c, dtype=np.float64)
    lon_c64 = np.asarray(lon_c, dtype=np.float64)
    lat_f64 = np.asarray(lat_f, dtype=np.float64)
    lon_f64 = np.asarray(lon_f, dtype=np.float64)

    fi = np.abs(lat_f64[:, None] - lat_c64[None, :]).argmin(axis=0)
    dlon_1d = _angular_lon_delta_deg(lon_f64[:, None], lon_c64[None, :])
    fj = np.argmin(dlon_1d, axis=0)

    dlat = np.abs(lat_f64[fi][:, None] - lat_c64[:, None])
    dlon_2d = _angular_lon_delta_deg(lon_f64[fj][None, :], lon_c64[None, :])

    if mode == "exact":
        mask = (dlat < exact_eps) & (dlon_2d < exact_eps)
    else:
        dist = np.sqrt(dlat ** 2 + dlon_2d ** 2)
        mask = dist <= float(max_deg)

    fi2 = fi[:, None]
    fj2 = fj[None, :]
    sampled = truth_f[fi2, fj2].astype(np.float32)
    out = np.where(mask, sampled, np.nan).astype(np.float32)
    return out, mask.astype(np.float64)


def regrid_pred_to_native(
    pred: np.ndarray,
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    lat_f: np.ndarray,
    lon_f: np.ndarray,
) -> np.ndarray:
    """双线性：将粗网格预报插到原生 lat_f×lon_f 网格上。"""
    sh, sw = pred.shape
    w = _build_regrid_weights(
        sh,
        sw,
        np.asarray(lat_c, dtype=np.float64),
        np.asarray(lon_c, dtype=np.float64),
        np.asarray(lat_f, dtype=np.float64),
        np.asarray(lon_f, dtype=np.float64),
    )
    return _regrid_2d(np.asarray(pred, dtype=np.float32), w)


def _npy_path(
    output_root: Path,
    display_name: str,
    slug: str,
    var: str,
    init_tag: str,
) -> Path:
    """
    init-first 布局：{output_root}/{init_tag}/{display_name}/{fname}
    PanGu 使用 {var}_surface_{init_tag}.npy，其余使用 {var}_{init_tag}.npy。
    """
    if slug == "pangu":
        fname = f"{var}_surface_{init_tag}.npy"
    else:
        fname = f"{var}_{init_tag}.npy"
    return output_root / init_tag / display_name / fname


from zk_io.npy_reader import load_pred_stack as _load_pred_stack


def _save_metrics_dataframe(
    records: List[dict],
    save_dir: Path,
    time_tag: str,
    model_order: Optional[List[str]],
) -> None:
    from zk_io.plot_utils import plot_metrics_timeseries

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if df.empty:
        print(f"[multigrid_eval] 无记录，跳过写入: {save_dir}", flush=True)
        return
    csv_path = save_dir / f"timeseries_metrics_{time_tag}.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"[multigrid_eval] CSV: {csv_path}", flush=True)

    df_models = set(df["Model"].unique())
    if model_order:
        models: List[str] = []
        seen: set = set()
        for m in model_order:
            if m in df_models and m not in seen:
                models.append(m)
                seen.add(m)
        models.extend(sorted(df_models - seen))
    else:
        models = sorted(df_models)
    variables = sorted(df["Variable"].unique())
    max_lead = int(df["Lead_Time"].max()) if not df.empty else 240
    for metric in ("W-MAE", "W-RMSE"):
        if metric not in df.columns:
            continue
        fname = f"{metric.replace('-', '_')}_{time_tag}.png"
        plot_metrics_timeseries(
            df,
            metric_name=metric,
            y_label=metric,
            save_path=save_dir / fname,
            variables=variables,
            models=models,
            max_lead=max_lead,
        )
        print(f"[multigrid_eval] 图: {save_dir / fname}", flush=True)


def run_multigrid_eval_from_npy(
    *,
    output_root: Path,
    truth_adapter: Any,
    init_dt: datetime,
    init_tag: str,
    max_lead: int,
    lead_step: int,
    model_slugs: Sequence[str],
    variables: List[str],
    metrics: List[str],
    coarse_lat: np.ndarray,
    coarse_lon: np.ndarray,
    eval_modes: Set[str],
    fidelity_max_deg: float = 0.125,
    fidelity_exact_eps: float = 1e-4,
    model_order: Optional[List[str]] = None,
    spatial_plots: bool = False,
) -> None:
    """
    从已写 NPY 与 ``truth_adapter`` 原生面场跑保真/升尺度评估。

    eval_modes 可含 ``fidelity_nn``、``fidelity_exact``、``upscale``（可多选）。

    ``spatial_plots=True`` 时写三联图到 ``{output_root}/plots/{model}/{init_tag}/``，
    文件名带 ``fidelity_nn_`` / ``fidelity_exact_`` / ``upscale_`` 前缀以免与对齐降尺度图混淆。
    """
    modes = {m.lower() for m in eval_modes}
    wanted = modes & {"fidelity_nn", "fidelity_exact", "upscale"}
    if not wanted:
        return

    n_steps = max_lead // lead_step
    leads = list(range(lead_step, max_lead + 1, lead_step))
    if len(leads) != n_steps:
        raise ValueError("max_lead 必须能被 lead_step 整除")

    lat_c = np.asarray(coarse_lat, dtype=np.float32)
    lon_c = np.asarray(coarse_lon, dtype=np.float32)
    Hc, Wc = len(lat_c), len(lon_c)

    probe = load_truth_blob_for_multigrid(truth_adapter, init_dt)
    if probe is None:
        print(
            f"[multigrid_eval] 无法读取真值 init={init_dt}，跳过保真/升尺度",
            flush=True,
        )
        return
    lat0 = probe.get("lat")
    lon0 = probe.get("lon")
    if lat0 is None or lon0 is None:
        print("[multigrid_eval] 真值 blob 缺少 lat/lon，跳过保真/升尺度", flush=True)
        return
    if truth_grid_matches_model(lat0, lon0, lat_c, lon_c):
        print(
            "[multigrid_eval] 真值网格与模型网格一致（维数与坐标对齐），"
            "与对齐降尺度等价；跳过保真/升尺度",
            flush=True,
        )
        return

    pred_stacks: Dict[Tuple[str, str], np.ndarray] = {}
    for slug in model_slugs:
        disp = _SLUG_TO_DISPLAY.get(slug.lower(), slug)
        for var in variables:
            pth = _npy_path(output_root, disp, slug.lower(), var, init_tag)
            if not pth.is_file():
                continue
            pred_stacks[(disp, var)] = _load_pred_stack(
                pth, n_steps=n_steps, h=Hc, w=Wc,
            )

    order = model_order or _EVAL_MODEL_ORDER

    if "fidelity_nn" in wanted:
        _run_fidelity_mode(
            output_root=output_root,
            truth_adapter=truth_adapter,
            init_dt=init_dt,
            init_tag=init_tag,
            max_lead=max_lead,
            leads=leads,
            n_steps=n_steps,
            model_slugs=model_slugs,
            variables=variables,
            metrics=metrics,
            lat_c=lat_c,
            lon_c=lon_c,
            pred_stacks=pred_stacks,
            mode="nn_threshold",
            max_deg=fidelity_max_deg,
            exact_eps=fidelity_exact_eps,
            subdir=f"eval_fidelity_nn_{max_lead}h_{init_tag}",
            model_order=order,
            plot_eval_label="fidelity_nn",
            spatial_plots=spatial_plots,
        )

    if "fidelity_exact" in wanted:
        _run_fidelity_mode(
            output_root=output_root,
            truth_adapter=truth_adapter,
            init_dt=init_dt,
            init_tag=init_tag,
            max_lead=max_lead,
            leads=leads,
            n_steps=n_steps,
            model_slugs=model_slugs,
            variables=variables,
            metrics=metrics,
            lat_c=lat_c,
            lon_c=lon_c,
            pred_stacks=pred_stacks,
            mode="exact",
            max_deg=fidelity_max_deg,
            exact_eps=fidelity_exact_eps,
            subdir=f"eval_fidelity_exact_{max_lead}h_{init_tag}",
            model_order=order,
            plot_eval_label="fidelity_exact",
            spatial_plots=spatial_plots,
        )

    if "upscale" in wanted:
        _run_upscale_mode(
            output_root=output_root,
            truth_adapter=truth_adapter,
            init_dt=init_dt,
            init_tag=init_tag,
            max_lead=max_lead,
            leads=leads,
            n_steps=n_steps,
            model_slugs=model_slugs,
            variables=variables,
            metrics=metrics,
            lat_c=lat_c,
            lon_c=lon_c,
            pred_stacks=pred_stacks,
            subdir=f"eval_upscale_{max_lead}h_{init_tag}",
            model_order=order,
            spatial_plots=spatial_plots,
        )


def _run_fidelity_mode(
    *,
    output_root: Path,
    truth_adapter: Any,
    init_dt: datetime,
    init_tag: str,
    max_lead: int,
    leads: List[int],
    n_steps: int,
    model_slugs: Sequence[str],
    variables: List[str],
    metrics: List[str],
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    pred_stacks: Dict[Tuple[str, str], np.ndarray],
    mode: str,
    max_deg: float,
    exact_eps: float,
    subdir: str,
    model_order: List[str],
    plot_eval_label: str,
    spatial_plots: bool = False,
) -> None:
    records: List[dict] = []
    for si, current_lead in enumerate(leads):
        if si >= n_steps:
            break
        valid_dt = init_dt + timedelta(hours=current_lead)
        native = load_truth_blob_for_multigrid(truth_adapter, valid_dt)
        if native is None:
            continue
        lat_f = np.asarray(native["lat"], dtype=np.float32)
        lon_f = np.asarray(native["lon"], dtype=np.float32)
        truth_full = extract_surface_vars(native, variables)

        for slug in model_slugs:
            disp = _SLUG_TO_DISPLAY.get(slug.lower(), slug)
            for var_name in variables:
                key = (disp, var_name)
                if key not in pred_stacks or var_name not in truth_full:
                    continue
                stack = pred_stacks[key]
                if si >= stack.shape[0]:
                    continue
                pred_arr = np.asarray(stack[si], dtype=np.float32)
                t_native = truth_full[var_name]
                t_c, mask = _truth_on_coarse_separable_nn(
                    t_native, lat_c, lon_c, lat_f, lon_f,
                    mode="exact" if mode == "exact" else "nn_threshold",
                    max_deg=max_deg,
                    exact_eps=exact_eps,
                )
                p_h, t_h = harmonize_surface_pair(var_name, pred_arr, t_c)
                if not np.any(mask):
                    continue
                _, mvals = compute_step_metrics_masked(
                    p_h, t_h, lat_c, mask, metrics=metrics,
                )
                rec = {
                    "Model": disp,
                    "Variable": var_name,
                    "Lead_Time": current_lead,
                    "EvalMode": (
                        "fidelity_exact" if mode == "exact" else "fidelity_nn"
                    ),
                }
                rec.update(mvals)
                records.append(rec)

                if spatial_plots:
                    plot_dir = Path(output_root) / "plots" / slug.lower() / init_tag
                    cmap = (
                        "RdBu_r" if var_name in ("u10", "v10") else "viridis"
                    )
                    plot_compare(
                        plot_dir
                        / f"{plot_eval_label}_{var_name}_lead{current_lead:03d}.png",
                        p_h,
                        t_h,
                        title=(
                            f"{disp} +{current_lead}h {var_name} | {init_tag} | "
                            f"{plot_eval_label} (NN truth on model grid)"
                        ),
                        cmap=cmap,
                    )

    out_dir = output_root / subdir
    _save_metrics_dataframe(records, out_dir, init_tag, model_order)


def _run_upscale_mode(
    *,
    output_root: Path,
    truth_adapter: Any,
    init_dt: datetime,
    init_tag: str,
    max_lead: int,
    leads: List[int],
    n_steps: int,
    model_slugs: Sequence[str],
    variables: List[str],
    metrics: List[str],
    lat_c: np.ndarray,
    lon_c: np.ndarray,
    pred_stacks: Dict[Tuple[str, str], np.ndarray],
    subdir: str,
    model_order: List[str],
    spatial_plots: bool = False,
) -> None:
    from core.evaluation.metrics import compute_step_metrics

    lat_c64 = np.asarray(lat_c, dtype=np.float64)
    lon_c64 = np.asarray(lon_c, dtype=np.float64)
    records: List[dict] = []

    for si, current_lead in enumerate(leads):
        if si >= n_steps:
            break
        valid_dt = init_dt + timedelta(hours=current_lead)
        native = load_truth_blob_for_multigrid(truth_adapter, valid_dt)
        if native is None:
            continue
        lat_f = np.asarray(native["lat"], dtype=np.float32)
        lon_f = np.asarray(native["lon"], dtype=np.float32)
        truth_full = extract_surface_vars(native, variables)

        for slug in model_slugs:
            disp = _SLUG_TO_DISPLAY.get(slug.lower(), slug)
            for var_name in variables:
                key = (disp, var_name)
                if key not in pred_stacks or var_name not in truth_full:
                    continue
                stack = pred_stacks[key]
                if si >= stack.shape[0]:
                    continue
                pred_arr = np.asarray(stack[si], dtype=np.float32)
                pred_f = regrid_pred_to_native(
                    pred_arr, lat_c64, lon_c64, lat_f, lon_f,
                )
                t_native = truth_full[var_name]
                p_h, t_h = harmonize_surface_pair(var_name, pred_f, t_native)
                _, mvals = compute_step_metrics(
                    p_h, t_h, lat_f, metrics=metrics,
                )
                rec = {
                    "Model": disp,
                    "Variable": var_name,
                    "Lead_Time": current_lead,
                    "EvalMode": "upscale",
                }
                rec.update(mvals)
                records.append(rec)

                if spatial_plots:
                    plot_dir = Path(output_root) / "plots" / slug.lower() / init_tag
                    cmap = (
                        "RdBu_r" if var_name in ("u10", "v10") else "viridis"
                    )
                    plot_compare(
                        plot_dir
                        / f"upscale_{var_name}_lead{current_lead:03d}.png",
                        p_h,
                        t_h,
                        title=(
                            f"{disp} +{current_lead}h {var_name} | {init_tag} | "
                            "upscale (forecast bilinear → native truth grid)"
                        ),
                        cmap=cmap,
                    )

    out_dir = output_root / subdir
    _save_metrics_dataframe(records, out_dir, init_tag, model_order)
