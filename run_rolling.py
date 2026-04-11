#!/usr/bin/env python3
"""
功能二 CLI 入口：滚动推理（含可选定量评估）。

对应原代码：GunDong_Infer/run_gundong_infer.py + evaluate_models.py。

示例：
  # 单日，FengWu + FuXi，240h，全地表变量，开启评估
  python run_rolling.py \\
      --models fengwu fuxi \\
      --data-source gundong_20260324 \\
      --date-range 20260308 \\
      --init-hour 12 --lead-step 6 --max-lead 240 \\
      --enable-eval

  # 多日范围，全部模型，不出图（只保存 NPY）
  python run_rolling.py \\
      --models all \\
      --data-source gundong_20260324 \\
      --date-range 20260301:20260318 \\
      --skip-plots

  # 自定义变量 + 保存 diff + 保存 NC
  python run_rolling.py \\
      --models pangu graphcast \\
      --data-source gundong_20260324 \\
      --date-range 20260308 \\
      --variables u10 v10 t2m \\
      --save-nc --enable-eval --save-diff \\
      --output-root /public/share/aciwgvx1jd/my_results

  # 已有 NPY 预报栈，仅重跑合并评估（CSV + 多模型时序图，可选三联图）
  cd /path/to/graphcast/ZK_Models/unified_pipeline
  python run_eval_npy.py \\
      --data-source gundong_20260324 --date-range 20260310 \\
      --init-hour 12 --max-lead 240 --lead-step 6 \\
      --models pangu fengwu fuxi graphcast \\
      --output-root /public/share/aciwgvx1jd/LYQ/gundong/GunDong_Infer_result_12h
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------
# 必须在任何 ROCm/CUDA 相关 import 之前设置每进程可见设备。
# torchrun / srun 会注入 LOCAL_RANK；单进程时不做修改。
#
# 若已显式给出逗号分隔列表（如 HIP_VISIBLE_DEVICES=0,1），则**不再**把各环境变量
# 压成单个索引：在部分 ROCm/Slurm 组合下，仅暴露「设备 1」会导致 rank1 上
# torch.cuda.is_available() 为 False 与 ORT SIGABRT。此时保留列表并在 chdir 后
# torch.cuda.set_device(LOCAL_RANK)。
# ---------------------------------------------------------------
_local_rank = os.environ.get("LOCAL_RANK")
_multi_gpu_visible = False
if _local_rank is not None:
    _vis = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or ""
    )
    if "," in str(_vis).strip():
        _multi_gpu_visible = True

if _local_rank is not None and not _multi_gpu_visible:
    for _k in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
               "HIP_VISIBLE_DEVICES", "HSA_VISIBLE_DEVICES"):
        os.environ[_k] = str(_local_rank)

import argparse
from pathlib import Path
from typing import List, Optional

from runtime_paths import UNIFIED_PIPELINE_ROOT, GRAPH_CAST_ROOT, bootstrap
bootstrap()

_ZK_ROOT = UNIFIED_PIPELINE_ROOT
_GRAPH_ROOT = GRAPH_CAST_ROOT

if _local_rank is not None and _multi_gpu_visible:
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(int(_local_rank))

import yaml
from pipelines.rolling_pipeline import run_rolling


def _env_true_str(val: str) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}
from core.config_loader import load_defaults as _load_defaults, get_all_enabled_models, resolve_output_root
from core.monitoring import start_hardware_logger


def main():
    defaults = _load_defaults()
    r_defaults = defaults.get("pipeline", {}).get("rolling", {})
    e_defaults = defaults.get("evaluation", {})

    ap = argparse.ArgumentParser(
        description="功能二：滚动推理（+ 可选定量评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- 模型与数据 ----
    ap.add_argument(
        "--models", nargs="+", default=["fengwu", "fuxi"],
        help="模型列表，或 'all' 代表全部启用模型",
    )
    ap.add_argument(
        "--data-source", default="gundong_20260324",
        help="数据源名称（config/data.yaml key）或数据根目录路径",
    )

    # ---- 时间参数 ----
    ap.add_argument(
        "--date-range", required=True,
        help="日期 'yyyymmdd' 或范围 'yyyymmdd:yyyymmdd'，如 20260308 或 20260301:20260318",
    )
    ap.add_argument(
        "--init-hour", type=int, default=r_defaults.get("init_hour", 12),
        help="起报时刻 UTC（默认 12）",
    )
    ap.add_argument(
        "--lead-step", type=int, default=r_defaults.get("lead_step_hours", 6),
        help="每步步长小时（默认 6）",
    )
    ap.add_argument(
        "--max-lead", type=int, default=r_defaults.get("max_lead_hours", 240),
        help="最大预报时长小时（默认 240）",
    )

    # ---- 变量 ----
    ap.add_argument(
        "--variables", nargs="*", metavar="VAR",
        help="输出变量（None=模型默认全部地表变量）",
    )

    # ---- 输出 ----
    ap.add_argument(
        "--output-root", type=Path,
        default=None,
        help="结果根目录（默认根据 --data-source 自动选择 LYQ/ecmwf_init 或 LYQ/gundong）",
    )
    ap.add_argument(
        "--skip-plots", action="store_true",
        default=r_defaults.get("skip_plots", False),
        help="跳过逐步对比图（节省时间）",
    )
    ap.add_argument(
        "--save-nc", action="store_true",
        default=r_defaults.get("save_nc", False),
        help="保存 per-step NetCDF 文件",
    )

    # ---- 评估 ----
    ap.add_argument(
        "--enable-eval", action="store_true",
        default=r_defaults.get("enable_eval", False),
        help="开启内嵌定量评估（自动输出 RMSE/MAE CSV + 时序图）",
    )
    ap.add_argument(
        "--save-diff", action="store_true",
        default=r_defaults.get("save_diff", False),
        help="保存差值场 npy 文件（需 --enable-eval）",
    )
    ap.add_argument(
        "--save-diff-nc", action="store_true",
        help="保存差值场 nc 文件（需 --enable-eval）",
    )
    ap.add_argument(
        "--metrics", nargs="+",
        default=e_defaults.get("metrics", ["W-MAE", "W-RMSE"]),
        help="评估指标（默认 W-MAE W-RMSE）",
    )
    ap.add_argument(
        "--eval-modes",
        nargs="*",
        default=None,
        metavar="MODE",
        help=(
            "enable-eval 时的评估模式。可含 downscale（对齐降尺度，默认）、"
            "fidelity_nn、fidelity_exact、upscale。多选；省略时等同 downscale。"
        ),
    )
    ap.add_argument(
        "--fidelity-max-deg",
        type=float,
        default=float(e_defaults.get("fidelity_max_deg", 0.125)),
        help="fidelity_nn：模型格点与真值格点最大允许距离（度）",
    )
    ap.add_argument(
        "--auto-multigrid",
        action="store_true",
        default=bool(e_defaults.get("auto_multigrid", False)),
        help=(
            "未指定 --eval-modes 时自动附加 fidelity_nn 与 upscale（仅细于 721×1440 的 "
            "ecmwf_init GRIB 真值生效）"
        ),
    )

    # ---- 真值数据源 ----
    ap.add_argument(
        "--truth-source", default=None,
        help=(
            "评估/对比图使用的真值数据源（config/data.yaml key 或路径）。"
            "默认与 --data-source 一致。"
            "示例：--data-source ecmwf_init --truth-source gundong_20260324"
        ),
    )

    # ---- 真值缓存 ----
    ap.add_argument(
        "--truth-cache-mode",
        default=os.environ.get("TRUTH_CACHE_MODE", "auto"),
        choices=["auto", "memory", "disk"],
        help=(
            "真值缓存模式（默认 auto）:\n"
            "  auto   — 运行时估算内存，自动选择 memory 或 disk\n"
            "  memory — 进程级 dict，模型切换不清理，作业结束随进程释放\n"
            "  disk   — {output_root}/_truth_cache/{job_id}/rank{R}/…，作业结束默认清理\n"
            "当 --skip-plots 且不 --enable-eval 时，完全跳过真值路径，此参数无效。"
        ),
    )
    ap.add_argument(
        "--truth-cache-budget-ratio",
        type=float,
        default=float(os.environ.get("TRUTH_CACHE_BUDGET_RATIO", "0.4")),
        metavar="RATIO",
        help="auto 模式：可用内存预留给缓存的比例（默认 0.4 即 40%%）",
    )
    ap.add_argument(
        "--truth-cache-safety-factor",
        type=float,
        default=float(os.environ.get("TRUTH_CACHE_SAFETY_FACTOR", "1.3")),
        metavar="FACTOR",
        help="auto 模式：估算总需求时的安全系数（默认 1.3）",
    )
    ap.add_argument(
        "--keep-truth-cache",
        action="store_true",
        default=_env_true_str(os.environ.get("KEEP_TRUTH_CACHE", "0")),
        help="保留磁盘缓存目录（调试用；默认作业结束后删除）",
    )

    # ---- 硬件/配置 ----
    ap.add_argument(
        "--device", default="auto",
        choices=["auto", "dcu", "cuda", "cpu"],
        help="推理设备",
    )
    ap.add_argument(
        "--parallel-mode", default="auto",
        choices=["auto", "date", "model"],
        help=(
            "多卡并行策略（需配合 torchrun/WORLD_SIZE 使用）:\n"
            "  auto  — 日期数 >= WORLD_SIZE 时按日期分片，否则按模型分片\n"
            "  date  — 每 rank 处理不同日期（18天×8卡 等多日场景）\n"
            "  model — 每 rank 处理不同模型（单日×4模型×4卡 等场景）"
        ),
    )
    ap.add_argument(
        "--models-config", type=Path,
        default=_ZK_ROOT / "config" / "models.yaml",
    )
    ap.add_argument(
        "--data-config", type=Path,
        default=_ZK_ROOT / "config" / "data.yaml",
    )

    # ---- 硬件监控 ----
    ap.add_argument(
        "--no-monitor", action="store_true",
        help="禁用硬件监控（默认开启）",
    )
    ap.add_argument(
        "--monitor-interval", type=int, default=30,
        help="硬件监控轮询间隔（秒，默认 30）",
    )

    # ---- 时间统计 ----
    ap.add_argument(
        "--no-cpu-timing", action="store_true",
        help="禁用进程 CPU 时间统计（默认开启）",
    )
    ap.add_argument(
        "--no-gpu-timing", action="store_true",
        help="禁用 GPU/DCU 设备区间时间统计（默认开启；GPU 不可用时自动无操作）",
    )
    ap.add_argument(
        "--cpu-timing-exclude-plots",
        action="store_true",
        default=bool(r_defaults.get("cpu_timing_exclude_plots", False)),
        help=(
            "CPU 计时时排除逐步对比图（matplotlib）；GPU 区间计时仍含整段。"
            "与 --no-cpu-timing 互斥生效（关闭 CPU 统计时无作用）。"
        ),
    )

    args = ap.parse_args()

    _em = args.eval_modes
    if _em is None:
        eval_mode_list = ["downscale"]
        _auto_mg = args.auto_multigrid or bool(
            e_defaults.get("auto_multigrid", False),
        )
    elif len(_em) == 0:
        eval_mode_list = ["downscale"]
        _auto_mg = args.auto_multigrid or bool(
            e_defaults.get("auto_multigrid", False),
        )
    else:
        eval_mode_list = list(_em)
        _auto_mg = bool(args.auto_multigrid)

    if args.models == ["all"]:
        model_names = get_all_enabled_models(args.models_config)
    else:
        model_names = args.models

    variables = args.variables if args.variables else None

    # torchrun 多进程时各 rank 并发 rocm-smi 可能扰动驱动；仅 rank0 轮询整机 DCU
    _rank = int(os.environ.get("RANK", "0"))
    _monitor_ok = not args.no_monitor and _rank == 0

    with start_hardware_logger(
        log_dir=_ZK_ROOT / "logs",
        poll_interval=args.monitor_interval,
        enabled=_monitor_ok,
    ):
        output_root = args.output_root
        if output_root is None:
            output_root = resolve_output_root(args.data_source, defaults)

        _mg = [m for m in eval_mode_list if m.lower() != "downscale"]
        _run_ds = "downscale" in {m.lower() for m in eval_mode_list}
        run_rolling(
            model_names=model_names,
            data_source=args.data_source,
            date_range=args.date_range,
            init_hour=args.init_hour,
            lead_step=args.lead_step,
            max_lead=args.max_lead,
            variables=variables,
            output_root=output_root,
            device=args.device,
            skip_plots=args.skip_plots,
            save_nc=args.save_nc,
            enable_eval=args.enable_eval,
            save_diff=args.save_diff,
            save_diff_nc=args.save_diff_nc,
            metrics=args.metrics,
            models_cfg_path=args.models_config,
            data_cfg_path=args.data_config,
            parallel_mode=args.parallel_mode,
            truth_source=args.truth_source,
            enable_cpu_timing=not args.no_cpu_timing,
            enable_gpu_timing=not args.no_gpu_timing,
            cpu_timing_exclude_plots=bool(args.cpu_timing_exclude_plots),
            eval_multigrid_modes=_mg if args.enable_eval else None,
            fidelity_max_deg=args.fidelity_max_deg,
            auto_multigrid=_auto_mg and args.enable_eval,
            eval_run_downscale=_run_ds if args.enable_eval else True,
            truth_cache_mode=args.truth_cache_mode,
            truth_cache_budget_ratio=args.truth_cache_budget_ratio,
            truth_cache_safety_factor=args.truth_cache_safety_factor,
            keep_truth_cache=args.keep_truth_cache,
        )


if __name__ == "__main__":
    main()
