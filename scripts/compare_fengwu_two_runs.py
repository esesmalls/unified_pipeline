#!/usr/bin/env python3
"""Compare two FengWu numpy outputs (same init): pairwise diffs and W-RMSE vs truth.

Truth sources (mutually exclusive):

  - **Path B** (``run_eval_npy.py`` / rolling eval): pass ``--data-source ecmwf_init``
    (and optional ``--truth-source``). Truth via ``get_adapter`` +
    ``load_blob_for_valid_time``, ``extract_surface_vars``, ``harmonize_surface_pair``,
    then ``compute_step_metrics`` W-RMSE — same as pipeline.

  - **ERA5 surface NetCDF**: ``--era5-surface-dir`` + ``YYYY_MM_DD_surface_*.nc``.

  - **Legacy raw GRIB**: ``--truth-dir`` only — pygrib + scipy zoom (not path B).
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from scipy.ndimage import zoom

# Repo root (unified_pipeline)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.data.channel_mapper import extract_surface_vars
from core.data.detector import get_adapter
from core.data.surface_units import harmonize_surface_pair
from core.evaluation.metrics import compute_step_metrics

# NPY channel stem -> GRIB shortName for surface instantaneous fields
VAR_MAP = {
    "t2m": "2t",
    "msl": "msl",
    "u10": "10u",
    "v10": "10v",
}


def regrid_to_fengwu(truth_hi: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    zy = out_shape[0] / truth_hi.shape[0]
    zx = out_shape[1] / truth_hi.shape[1]
    return zoom(truth_hi.astype(np.float64), (zy, zx), order=1)


def load_truth_surface_all(grib_path: Path, short_names: list[str]) -> dict[str, np.ndarray]:
    import pygrib

    want = set(short_names)
    out: dict[str, np.ndarray] = {}
    grbs = pygrib.open(str(grib_path))
    try:
        for m in grbs:
            if m.typeOfLevel != "surface":
                continue
            if m.shortName in want and m.shortName not in out:
                out[m.shortName] = m.values.astype(np.float64)
            if len(out) == len(want):
                break
    finally:
        grbs.close()
    missing = want - set(out.keys())
    if missing:
        raise KeyError(f"Missing surface fields {missing} in {grib_path}")
    return out


def grib_path_truth(truth_dir: Path, valid: datetime) -> Path:
    return truth_dir / f"G_{valid:%Y%m%d%H}_fh_0.grib1"


def default_model_label(dir_path: Path) -> str:
    p = dir_path.resolve()
    if len(p.parts) >= 2:
        return f"{p.parts[-2]}/{p.parts[-1]}"
    return p.name


def era5_var_name(ds: xr.Dataset, stem: str) -> str:
    if stem in ds.data_vars:
        return stem
    alt = stem.replace("u10", "10u").replace("v10", "10v")
    if alt in ds.data_vars:
        return alt
    raise KeyError(f"No ERA5 variable for {stem} in dataset")


def load_era5_lats_sample(era5_dir: Path) -> np.ndarray:
    pat = str(era5_dir / "*.nc")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No .nc under {era5_dir}")
    with xr.open_dataset(files[0]) as ds0:
        return np.asarray(ds0.latitude.values, dtype=np.float64)


def _load_data_source_name(
    source_name: str, data_cfg_path: Path
) -> Tuple[Path, Any, dict]:
    with open(data_cfg_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f) or {}
    sources = dcfg.get("sources", {})
    if source_name in sources:
        src = sources[source_name]
        return Path(src["root"]), src.get("format"), src
    p = Path(source_name)
    if p.exists():
        return p, None, {}
    raise ValueError(f"未找到数据源 '{source_name}'（检查 config/data.yaml）")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dir-a", type=Path, required=True, help="First FengWu output dir")
    ap.add_argument("--dir-b", type=Path, required=True, help="Second FengWu output dir")
    ap.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Path B: data.yaml key (e.g. ecmwf_init). Uses Adapter + harmonize like run_eval_npy.",
    )
    ap.add_argument(
        "--truth-source",
        type=str,
        default=None,
        help="Optional; truth adapter key when it should differ from --data-source.",
    )
    ap.add_argument(
        "--data-config",
        type=Path,
        default=_ROOT / "config" / "data.yaml",
        help="Path to config/data.yaml for --data-source.",
    )
    ap.add_argument(
        "--truth-dir",
        type=Path,
        default=None,
        help="Legacy: GRIB dir + scipy zoom (not path B).",
    )
    ap.add_argument(
        "--era5-surface-dir",
        type=Path,
        default=None,
        help="ERA5 instant surface NetCDF dir (YYYY_MM_DD_surface_*.nc).",
    )
    ap.add_argument("--init", type=str, default="2026032712", help="Init as YYYYMMDDHH UTC")
    ap.add_argument("--tag", type=str, default="20260327T12", help="NPY filename tag")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--lead-offset",
        type=int,
        default=1,
        help="NPY index t maps to lead hours 6*(t+lead_offset). Default 1 => 6*(t+1).",
    )
    ap.add_argument("--label-a", type=str, default=None)
    ap.add_argument("--label-b", type=str, default=None)
    args = ap.parse_args()

    mode_count = sum(
        x is not None
        for x in (args.data_source, args.era5_surface_dir, args.truth_dir)
    )
    if mode_count != 1:
        ap.error(
            "Specify exactly one of: --data-source (path B), "
            "--era5-surface-dir, or --truth-dir (legacy GRIB zoom).",
        )

    use_adapter = args.data_source is not None
    use_era5 = args.era5_surface_dir is not None

    label_a = args.label_a or default_model_label(args.dir_a)
    label_b = args.label_b or default_model_label(args.dir_b)

    init = datetime.strptime(args.init, "%Y%m%d%H")
    init_tag_short = f"{init:%Y-%m-%d %H}Z"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = args.out_dir / "diff_maps_a_minus_b"
    diff_dir.mkdir(exist_ok=True)

    tag = args.tag
    vars_ = list(VAR_MAP.keys())
    path_t2m_a = args.dir_a / f"t2m_{tag}.npy"
    fcst_probe = np.load(path_t2m_a, mmap_mode="r")
    n_time, ny, nx = fcst_probe.shape

    fcst_a = {s: np.load(args.dir_a / f"{s}_{tag}.npy", mmap_mode="r") for s in vars_}
    fcst_b = {s: np.load(args.dir_b / f"{s}_{tag}.npy", mmap_mode="r") for s in vars_}

    leads_h = [6 * (t + args.lead_offset) for t in range(n_time)]
    valid_times = [init + timedelta(hours=(i + args.lead_offset) * 6) for i in range(n_time)]

    truth_adapter: Optional[Any] = None
    ds_cache: Dict[str, xr.Dataset] = {}
    current_date: Optional[str] = None
    time_dim_cache: Optional[str] = None
    era5_dir: Optional[Path] = None

    def _get_era5_truth(stem: str, valid_dt: datetime) -> np.ndarray:
        nonlocal current_date, time_dim_cache, ds_cache
        assert era5_dir is not None
        target_date_str = valid_dt.strftime("%Y_%m_%d")
        if current_date != target_date_str:
            for ds in ds_cache.values():
                ds.close()
            ds_cache.clear()
            era5_files = glob.glob(str(era5_dir / f"{target_date_str}_surface_*.nc"))
            if not era5_files:
                raise FileNotFoundError(
                    f"Missing ERA5 files for {target_date_str} under {era5_dir} "
                    f"(expected {target_date_str}_surface_*.nc)",
                )
            for f in era5_files:
                ds_cache[f] = xr.open_dataset(f)
            current_date = target_date_str
        if not ds_cache:
            raise RuntimeError("ERA5 cache empty")

        if time_dim_cache is None:
            sample = next(iter(ds_cache.values()))
            time_dim_cache = "valid_time" if "valid_time" in sample.dims else "time"

        td = time_dim_cache
        for ds_day in ds_cache.values():
            if valid_dt not in ds_day[td].values:
                continue
            name = era5_var_name(ds_day, stem)
            arr = ds_day[name].sel({td: valid_dt}).values
            return np.asarray(arr, dtype=np.float64)
        raise FileNotFoundError(f"No ERA5 slice at valid_time={valid_dt} for {stem}")

    if use_adapter:
        assert args.data_source is not None
        d_root, d_fmt, d_cfg = _load_data_source_name(args.data_source, args.data_config)
        adapter = get_adapter(
            d_root,
            fmt=d_fmt,
            use_monthly_subdir=d_cfg.get("use_monthly_subdir", False),
        )
        t_src = args.truth_source or args.data_source
        if t_src != args.data_source:
            t_root, t_fmt, t_cfg = _load_data_source_name(t_src, args.data_config)
            truth_adapter = get_adapter(
                t_root,
                fmt=t_fmt,
                use_monthly_subdir=t_cfg.get("use_monthly_subdir", False),
            )
        else:
            truth_adapter = adapter
        # ECMWF ecmwf_init_grib 标准输出为 721×（与 NPY 一致）；与 load_blob 中 _STD_LAT 对齐
        lats = np.linspace(90.0, -90.0, ny, dtype=np.float64)
        truth_backend_label = f"path B {args.data_source} / {t_src} ({d_root.name})"
    elif use_era5:
        era5_dir = args.era5_surface_dir.resolve()
        lats = load_era5_lats_sample(era5_dir)
        if lats.shape[0] != ny:
            raise ValueError(
                f"ERA5 latitude dim {lats.shape[0]} != forecast H {ny}; "
                "use ERA5 on the same 721 grid as the NPY files.",
            )
        truth_backend_label = f"ERA5 NC ({era5_dir.name})"
    else:
        assert args.truth_dir is not None
        lats = np.linspace(90.0, -90.0, ny, dtype=np.float64)
        truth_backend_label = f"legacy GRIB+zoom ({args.truth_dir.name})"

    metrics = []
    rmse_a = {v: [] for v in vars_}
    rmse_b = {v: [] for v in vars_}

    for t in range(n_time):
        lead = leads_h[t]
        valid = init + timedelta(hours=lead)

        if use_adapter:
            assert truth_adapter is not None
            valid_dt = valid_times[t]
            _lite = getattr(truth_adapter, "load_regridded_surface_only_for_valid_time", None)
            if callable(_lite):
                truth_blob = _lite(valid_dt)
            else:
                truth_blob = truth_adapter.load_blob_for_valid_time(valid_dt)
            if truth_blob is None:
                raise FileNotFoundError(f"No adapter truth for valid={valid_dt} (lead {lead}h)")
            truth_sfc = extract_surface_vars(truth_blob, vars_, strict=True)
            for stem in vars_:
                fa = fcst_a[stem][t].astype(np.float64)
                fb = fcst_b[stem][t].astype(np.float64)
                tr = truth_sfc[stem]
                p_a, t_a = harmonize_surface_pair(stem, fa, tr)
                p_b, t_b = harmonize_surface_pair(stem, fb, tr)
                if p_a.shape != t_a.shape or p_b.shape != t_b.shape:
                    raise ValueError(
                        f"{stem}: pred/truth shape mismatch "
                        f"A {p_a.shape} vs {t_a.shape}, B {p_b.shape} vs {t_b.shape}",
                    )
                _, ma = compute_step_metrics(p_a, t_a, lats, metrics=["W-RMSE"])
                _, mb = compute_step_metrics(p_b, t_b, lats, metrics=["W-RMSE"])
                ra = float(ma["W-RMSE"])
                rb = float(mb["W-RMSE"])
                rmse_a[stem].append(ra)
                rmse_b[stem].append(rb)
                metrics.append((lead, stem, ra, rb))
        elif use_era5:
            valid_dt = valid_times[t]
            truth_by_stem = {s: _get_era5_truth(s, valid_dt) for s in vars_}
            for stem in vars_:
                truth = truth_by_stem[stem]
                fa = fcst_a[stem][t].astype(np.float64)
                fb = fcst_b[stem][t].astype(np.float64)
                if truth.shape != fa.shape:
                    raise ValueError(f"{stem}: truth {truth.shape} pred {fa.shape}")
                _, ma = compute_step_metrics(fa, truth, lats, metrics=["W-RMSE"])
                _, mb = compute_step_metrics(fb, truth, lats, metrics=["W-RMSE"])
                ra = float(ma["W-RMSE"])
                rb = float(mb["W-RMSE"])
                rmse_a[stem].append(ra)
                rmse_b[stem].append(rb)
                metrics.append((lead, stem, ra, rb))
        else:
            assert args.truth_dir is not None
            gp = grib_path_truth(args.truth_dir, valid)
            if not gp.is_file():
                raise FileNotFoundError(f"Missing truth GRIB for valid {valid}: {gp}")
            shorts = [VAR_MAP[s] for s in vars_]
            truth_hi_all = load_truth_surface_all(gp, shorts)
            truth_by_stem = {
                s: regrid_to_fengwu(truth_hi_all[VAR_MAP[s]], (ny, nx)) for s in vars_
            }
            for stem in vars_:
                truth = truth_by_stem[stem]
                fa = fcst_a[stem][t].astype(np.float64)
                fb = fcst_b[stem][t].astype(np.float64)
                if truth.shape != fa.shape:
                    raise ValueError(f"{stem}: truth {truth.shape} pred {fa.shape}")
                _, ma = compute_step_metrics(fa, truth, lats, metrics=["W-RMSE"])
                _, mb = compute_step_metrics(fb, truth, lats, metrics=["W-RMSE"])
                ra = float(ma["W-RMSE"])
                rb = float(mb["W-RMSE"])
                rmse_a[stem].append(ra)
                rmse_b[stem].append(rb)
                metrics.append((lead, stem, ra, rb))

    if use_era5:
        for ds in ds_cache.values():
            ds.close()
        ds_cache.clear()

    csv_path = args.out_dir / "rmse_vs_truth_by_lead.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        f.write(f"# truth_backend={truth_backend_label}\n")
        w.writerow(["lead_hours", "variable", "w_rmse_dir_a", "w_rmse_dir_b"])
        for lead, stem, ra, rb in metrics:
            w.writerow([lead, stem, f"{ra:.6g}", f"{rb:.6g}"])

    ylabel_for = {
        "t2m": "W-RMSE vs truth (K)",
        "msl": "W-RMSE vs truth (Pa)",
        "u10": "W-RMSE vs truth (m/s)",
        "v10": "W-RMSE vs truth (m/s)",
    }

    plot_suptitle = f"W-RMSE vs truth — {truth_backend_label} (init {init_tag_short})"

    fig_all, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    axs_flat = axs.ravel()
    for ax, stem in zip(axs_flat, vars_):
        ax.plot(leads_h, rmse_a[stem], label=label_a, linewidth=1.8, color="#1f77b4")
        ax.plot(leads_h, rmse_b[stem], label=label_b, linewidth=1.8, color="#ff7f0e", linestyle="--")
        ax.set_title(stem.upper())
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(ylabel_for.get(stem, "W-RMSE"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig_all.suptitle(plot_suptitle, fontsize=12)
    fig_all.tight_layout()
    fig_all.savefig(args.out_dir / "rmse_vs_truth_timeseries.png", dpi=150)
    plt.close(fig_all)

    fig_line, axs_line = plt.subplots(len(vars_), 1, figsize=(9, 11), sharex=True, constrained_layout=False)
    color_a, color_b = "#1f77b4", "#ff7f0e"
    line_handles = []
    for i, stem in enumerate(vars_):
        ax = axs_line[i]
        (l_a,) = ax.plot(
            leads_h,
            rmse_a[stem],
            color=color_a,
            linestyle="-",
            linewidth=2.0,
            label=label_a,
        )
        (l_b,) = ax.plot(
            leads_h,
            rmse_b[stem],
            color=color_b,
            linestyle="--",
            linewidth=2.0,
            label=label_b,
        )
        if i == 0:
            line_handles = [l_a, l_b]
        ax.set_ylabel(ylabel_for.get(stem, "W-RMSE"))
        ax.grid(True, alpha=0.3)
        ax.set_title(stem.upper(), loc="left", fontsize=11, fontweight="bold")
    axs_line[-1].set_xlabel("Lead time (h)")
    fig_line.legend(
        line_handles,
        [label_a, label_b],
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=fig_line.transFigure,
    )
    fig_line.suptitle(plot_suptitle, fontsize=12, y=1.02)
    fig_line.subplots_adjust(top=0.94, hspace=0.28)
    fig_line.savefig(args.out_dir / "rmse_vs_lead_two_models_linechart.png", dpi=150, bbox_inches="tight")
    plt.close(fig_line)

    sel_leads = [h for h in (6, 72, 120, 240) if h in leads_h]
    if sel_leads:
        fig_b, axs_b = plt.subplots(1, len(sel_leads), figsize=(3.2 * len(sel_leads), 4), sharey=False)
        if len(sel_leads) == 1:
            axs_b = [axs_b]
        x = np.arange(len(vars_))
        for axb, h in zip(axs_b, sel_leads):
            idx = leads_h.index(h)
            wbar = 0.35
            axb.bar(x - wbar / 2, [rmse_a[s][idx] for s in vars_], wbar, label=label_a)
            axb.bar(x + wbar / 2, [rmse_b[s][idx] for s in vars_], wbar, label=label_b)
            axb.set_xticks(x)
            axb.set_xticklabels([v.upper() for v in vars_], rotation=15, ha="right")
            axb.set_ylabel("W-RMSE")
            axb.set_title(f"{h} h lead")
            axb.grid(True, axis="y", alpha=0.3)
        axs_b[0].legend(fontsize=8)
        fig_b.suptitle(f"W-RMSE by variable — {truth_backend_label}", y=1.02)
        fig_b.tight_layout()
        fig_b.savefig(args.out_dir / "rmse_vs_truth_bars_selected_leads.png", dpi=150)
        plt.close(fig_b)

    lead_pick = 240
    if lead_pick in leads_h:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        idx = leads_h.index(lead_pick)
        labs = [v.upper() for v in vars_]
        xa = np.arange(len(vars_))
        ax1.bar(xa - 0.18, [rmse_a[v][idx] for v in vars_], 0.35, label=label_a)
        ax1.bar(xa + 0.18, [rmse_b[v][idx] for v in vars_], 0.35, label=label_b)
        ax1.set_xticks(xa)
        ax1.set_xticklabels(labs)
        ax1.set_ylabel("W-RMSE")
        ax1.set_title(f"W-RMSE at {lead_pick} h — {truth_backend_label}")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(args.out_dir / f"rmse_vs_truth_bar_lead{lead_pick}h.png", dpi=150)
        plt.close(fig1)

    def plot_diff(stem: str, idx: int, lead_label: str) -> None:
        fa = fcst_a[stem][idx].astype(np.float64)
        fb = fcst_b[stem][idx].astype(np.float64)
        d = fa - fb
        vmax = float(np.percentile(np.abs(d), 99))
        if vmax < 1e-6:
            vmax = float(np.max(np.abs(d)) + 1e-9)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(d, origin="upper", aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, shrink=0.7, label=f"{stem} difference (A-B)")
        ax.set_title(f"{stem} A−B at {lead_label} lead (init {init:%Y%m%d%H}Z)")
        ax.set_xlabel("longitude index")
        ax.set_ylabel("latitude index")
        fig.tight_layout()
        safe = stem + "_" + lead_label.replace(" ", "")
        fig.savefig(diff_dir / f"diff_{safe}.png", dpi=130)
        plt.close(fig)

    for h in (6, 120, 240):
        if h not in leads_h:
            continue
        idx = leads_h.index(h)
        lbl = f"{h}h"
        for stem in vars_:
            plot_diff(stem, idx, lbl)

    print(f"Wrote figures and CSV under {args.out_dir} (truth: {truth_backend_label})")


if __name__ == "__main__":
    main()
