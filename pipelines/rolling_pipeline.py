"""
功能二：滚动推理流水线。

可定制：
  - 模型选择（多模型串行）
  - 变量选择（地表变量，或全部）
  - 时间范围（单日 / 日期范围）
  - 起报时刻（默认 12h）
  - 步长（默认 6h）
  - 总时长（默认 240h）
  - 输出路径（init-first 目录布局，见 README §9 / zk_io/rolling_paths）
  - 可选：内嵌定量评估（--enable-eval）
  - 可选：diff npy/nc 保存（--save-diff）

多卡并行：通过 RANK/WORLD_SIZE 环境变量分片日期，与 Slurm torchrun 配合使用。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from runtime_paths import UNIFIED_PIPELINE_ROOT, GRAPH_CAST_ROOT, bootstrap
bootstrap()

_ZK_ROOT = UNIFIED_PIPELINE_ROOT
_GRAPH_ROOT = GRAPH_CAST_ROOT

from core.data.detector import get_adapter
from core.data.channel_mapper import extract_surface_vars
from core.data.surface_units import harmonize_surface_pair
from core.capability import (
    CapabilityMemoryStore,
    OPTIONAL_CHANNEL_ALIASES,
    decide_optional_channels,
)
from core.models import build_registry
from core.evaluation.metrics import MetricsAccumulator
from core.evaluation.multigrid_eval import run_multigrid_eval_from_npy
from core.monitoring import RollingInferenceTiming
from core.monitoring import progress as _core_progress
from zk_io.npy_writer import NpyStackWriter
from zk_io.nc_writer import write_step_nc
from zk_io.plot_utils import plot_compare
from zk_io.rolling_paths import (
    EVAL_MODEL_ORDER as _EVAL_MODEL_ORDER,
    SLUG_TO_DISPLAY as _MODEL_DISPLAY_NAMES,
    eval_downscale_dir,
    meta_path,
    nc_dir,
    plot_dir as rolling_plot_dir,
)
from pipelines.truth_cache import TruthCacheManager


def _progress(msg: str) -> None:
    _core_progress(msg, tag="rolling")


def _env_true(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _shard_dates(dates: List[str]) -> List[str]:
    """按 RANK/WORLD_SIZE 分片日期：rank i 处理 dates[i::world]。"""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    return [d for i, d in enumerate(dates) if i % world == rank]


def _shard_models(models: List[str]) -> List[str]:
    """按 RANK/WORLD_SIZE 分片模型：rank i 处理 models[i::world]。
    单日多模型并行时使用，每张卡独立推理一组模型。"""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assigned = [m for i, m in enumerate(models) if i % world == rank]
    if not assigned:
        _progress(f"无分配模型（共 {len(models)} 个模型，{world} 个 rank），退出")
    return assigned


def _decide_parallel_mode(mode: str, dates: List[str], models: List[str]) -> str:
    """
    决定并行策略：
      date  — 每个 rank 处理不同日期（多日任务）
      model — 每个 rank 处理不同模型（单日或模型多于日期时）
      auto  — 自动判断：dates >= WORLD_SIZE → date；否则 → model
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if mode == "auto":
        return "date" if len(dates) >= world else "model"
    return mode


def _load_data_source(source_name: str, data_cfg_path: Optional[Path] = None):
    """返回 (root, fmt, src_cfg)。"""
    if data_cfg_path is None:
        data_cfg_path = _ZK_ROOT / "config" / "data.yaml"
    with open(data_cfg_path, encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)
    sources = dcfg.get("sources", {})
    if source_name in sources:
        src = sources[source_name]
        return Path(src["root"]), src.get("format"), src
    p = Path(source_name)
    if p.exists():
        return p, None, {}
    raise ValueError(
        f"未找到数据源 '{source_name}'：既非 config/data.yaml 中的 sources 名称，路径也不存在。"
        f" ECMWF GRIB 请使用 --data-source ecmwf_init（见 config/data.yaml 中 root，通常为 .../new_init）。"
    )


def _parse_date_range(date_range: str) -> List[str]:
    """
    解析日期范围字符串：
      '20260308'          → ['20260308']
      '20260301:20260318' → ['20260301', '20260302', ..., '20260318']
    """
    if ":" in date_range:
        start_s, end_s = date_range.split(":", 1)
        start = datetime.strptime(start_s.strip(), "%Y%m%d")
        end = datetime.strptime(end_s.strip(), "%Y%m%d")
        dates = []
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return dates
    return [date_range.strip()]


def _write_fallback_meta(
    output_root: Path,
    display_name: str,
    init_tag: str,
    source_id: str,
    model_name: str,
    enabled_optional_channels: List[str],
    missing_optional_channels: List[str],
    fallback_used: List[Dict[str, object]],
) -> None:
    out_path = meta_path(output_root, init_tag, display_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "init_tag": init_tag,
        "model": model_name,
        "source_id": source_id,
        "enabled_optional_channels": enabled_optional_channels,
        "missing_optional_channels": missing_optional_channels,
        "fallback_used": fallback_used,
    }
    out_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_rolling(
    model_names: List[str],
    data_source: str,
    date_range: str,
    init_hour: int = 12,
    lead_step: int = 6,
    max_lead: int = 240,
    variables: Optional[List[str]] = None,
    output_root: Optional[Path] = None,
    device: Any = "auto",
    skip_plots: bool = False,
    save_nc: bool = False,
    enable_eval: bool = False,
    save_diff: bool = False,
    save_diff_nc: bool = False,
    metrics: Optional[List[str]] = None,
    models_cfg_path: Optional[Path] = None,
    data_cfg_path: Optional[Path] = None,
    parallel_mode: str = "auto",
    truth_source: Optional[str] = None,
    enable_cpu_timing: bool = True,
    enable_gpu_timing: bool = True,
    cpu_timing_exclude_plots: bool = False,
    eval_multigrid_modes: Optional[List[str]] = None,
    fidelity_max_deg: float = 0.125,
    auto_multigrid: bool = False,
    eval_run_downscale: bool = True,
    truth_cache_mode: str = "auto",
    truth_cache_budget_ratio: float = 0.4,
    truth_cache_safety_factor: float = 1.3,
    keep_truth_cache: bool = False,
) -> None:
    """
    滚动推理主函数。

    Args:
        model_names:        模型名称列表（如 ["fengwu","fuxi"]）
        data_source:        数据源名称或路径
        date_range:         日期 'yyyymmdd' 或范围 'yyyymmdd:yyyymmdd'
        init_hour:          起报时刻 UTC（默认 12）
        lead_step:          步长小时（默认 6）
        max_lead:           总预报时长小时（默认 240）
        variables:          地表变量列表（None=使用模型默认全部地表变量）
        output_root:        结果根目录（``{init_tag}/{model_slug}/`` NPY 布局，见 README §9）
        device:             推理设备
        skip_plots:         跳过逐步对比图
        save_nc:            是否保存 per-step NetCDF（默认关；部分模型排障/落盘时可开启）
        enable_eval:        是否开启内嵌定量评估（W-RMSE/W-MAE CSV + 时序图）
        save_diff:          是否保存 diff npy 文件（需 enable_eval=True）
        save_diff_nc:       是否保存 diff nc 文件（需 enable_eval=True）
        metrics:            指标列表（None=["W-MAE","W-RMSE"]）
        parallel_mode:      多卡并行策略（auto | date | model）
                              auto  — 日期数 >= WORLD_SIZE → date 模式；否则 → model 模式
                              date  — 每 rank 处理不同日期（多日任务推荐）
                              model — 每 rank 处理不同模型（单日多模型推荐）
        truth_source:       评估/对比真值数据源（None=同 data_source）
        enable_cpu_timing:  统计每模型滚动段进程 CPU 时间及帧均值（默认开启）
        enable_gpu_timing:  统计每模型滚动段 GPU/DCU 设备区间时间及帧均值（默认开启；
                              GPU 不可用时自动跳过，不影响 CPU 统计）
        cpu_timing_exclude_plots:  为 True 时 CPU 统计排除逐步对比图（matplotlib）段，
                              更贴近推理+写帧+真值加载+NC 等（GPU 区间计时不变）
        eval_multigrid_modes:  多分辨率评估子模式（enable_eval 时），可选
                              fidelity_nn / fidelity_exact / upscale；与 downscale 独立
        fidelity_max_deg:      fidelity_nn 最近邻最大距离（度）
        auto_multigrid:        True 且未指定 eval_multigrid_modes 时，自动附加 fidelity_nn+upscale
                              （仅当原生 GRIB 网格细于 721×1440 时生效）
        eval_run_downscale:    False 时跳过逐步 W-MAE/W-RMSE（对齐降尺度），仍可跑保真/升尺度
        truth_cache_mode:      真值缓存模式（auto|memory|disk，默认 auto）
                                 auto   — 运行时根据内存估算自动选择 memory 或 disk
                                 memory — 进程级 dict，模型切换不清理，作业结束随进程释放
                                 disk   — {output_root}/_truth_cache/{job_id}/rank{R}/…
                                          作业结束默认删除（keep_truth_cache=True 时保留）
                               当 skip_plots=True 且 enable_eval=False 时，完全跳过真值路径
        truth_cache_budget_ratio: auto 模式下可用内存预留给缓存的比例（默认 0.4）
        truth_cache_safety_factor: auto 模式估算总需求的安全系数（默认 1.3）
        keep_truth_cache:      True=作业结束后保留磁盘缓存（调试，默认 False）
    """
    if models_cfg_path is None:
        models_cfg_path = _ZK_ROOT / "config" / "models.yaml"
    if output_root is None:
        from core.config_loader import resolve_output_root
        output_root = resolve_output_root(data_source)
    if metrics is None:
        metrics = ["W-MAE", "W-RMSE"]

    # 真值需求短路：skip_plots=True 且 enable_eval=False 时完全跳过真值路径
    need_truth: bool = (not skip_plots) or enable_eval

    n_steps = max_lead // lead_step
    leads = list(range(lead_step, max_lead + 1, lead_step))

    # --- 解析日期（分片前先拿全量，供 auto 模式判断）---
    all_dates = _parse_date_range(date_range)

    # --- 决定并行策略 ---
    pmode = _decide_parallel_mode(parallel_mode, all_dates, model_names)
    _progress(f"并行模式: {pmode} (WORLD_SIZE={os.environ.get('WORLD_SIZE','1')})")

    # --- 按策略分片 ---
    # 保留分片前的完整模型列表，供最终汇总排序使用
    _orig_model_names = list(model_names)
    if pmode == "model":
        dates = all_dates
        model_names = _shard_models(model_names)
        if not model_names:
            return
    else:
        dates = _shard_dates(all_dates)
        if not dates:
            _progress("当前 RANK 无分配日期，退出")
            return

    # --- 加载数据适配器（轻量，无 VRAM 占用，一次性初始化）---
    data_root, data_fmt, src_cfg = _load_data_source(data_source, data_cfg_path)
    _progress(f"数据源: {data_root}（格式: {data_fmt or '自动探测'}）")
    adapter = get_adapter(
        data_root, fmt=data_fmt,
        use_monthly_subdir=src_cfg.get("use_monthly_subdir", False),
    )
    mem_path = Path(output_root) / "_capability_memory" / "source_model_capabilities.json"
    mem_store = CapabilityMemoryStore(mem_path)
    source_id = mem_store.make_source_id(
        source_name=data_source,
        root=data_root,
        fmt=data_fmt,
        adapter_name=adapter.__class__.__name__,
    )
    mem_store.ensure_source(
        source_id=source_id,
        source_name=data_source,
        root=data_root,
        fmt=data_fmt,
        adapter_name=adapter.__class__.__name__,
    )

    # 真值适配器：默认与推理数据源一致，可通过 truth_source 切换
    if truth_source and truth_source != data_source:
        t_root, t_fmt, t_cfg = _load_data_source(truth_source, data_cfg_path)
        _progress(f"真值数据源: {t_root}（格式: {t_fmt or '自动探测'}）")
        truth_adapter = get_adapter(
            t_root, fmt=t_fmt,
            use_monthly_subdir=t_cfg.get("use_monthly_subdir", False),
        )
    else:
        truth_adapter = adapter

    # 与适配器扫描到的有.pressure 文件的日期取交集，避免 load_blob 反复失败却仍先加载大模型
    try:
        avail_dates = set(adapter.list_dates())
    except Exception:
        avail_dates = set()
    if avail_dates:
        before_dates = list(dates)
        dates = [d for d in dates if d in avail_dates]
        missing_ds = [d for d in before_dates if d not in avail_dates]
        if missing_ds:
            sample = sorted(avail_dates)[:8]
            _progress(
                f"警告: 数据目录无以下日期的 pressure 日文件，已剔除: {missing_ds}。"
                f"可选日期示例: {sample}{'...' if len(avail_dates) > 8 else ''}"
            )
    if not dates:
        _progress("无可用日期（与数据目录交集为空），本 rank 退出")
        return

    _progress(f"处理日期: {dates}  模型: {model_names}")
    if enable_eval:
        _progress("评估开关: enable_eval=True，将产出 RMSE/MAE 指标与评估目录")
    else:
        _progress("评估开关: enable_eval=False，本次不会产出 RMSE/MAE CSV 与评估图")

    # 多进程时错开 ONNX/ROCm Session 创建，降低并发初始化 SIGABRT 概率（秒/LR 可用 ROLLING_ORT_STAGGER_SEC 覆盖）
    _ws = int(os.environ.get("WORLD_SIZE", "1"))
    if _ws > 1:
        _lr = int(os.environ.get("LOCAL_RANK", "0"))
        if _lr > 0:
            per_rank = float(os.environ.get("ROLLING_ORT_STAGGER_SEC", "8"))
            delay_s = min(90.0, per_rank * _lr)
            _progress(f"WORLD_SIZE={_ws}: LOCAL_RANK={_lr} 错峰等待 {delay_s:.1f}s 再加载模型")
            time.sleep(delay_s)

    # 真值缓存管理器（作业级，模型切换不清理）
    _job_rank = int(os.environ.get("RANK", "0"))
    _job_id = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    truth_cache: Optional[TruthCacheManager] = None
    if need_truth:
        truth_cache = TruthCacheManager(
            mode=truth_cache_mode,
            cache_root=Path(output_root) / "_truth_cache",
            budget_ratio=truth_cache_budget_ratio,
            safety_factor=truth_cache_safety_factor,
            keep_disk_cache=keep_truth_cache,
            rank=_job_rank,
            job_id=_job_id,
        )
        _progress(
            f"[truth_cache] 初始化: mode_requested={truth_cache_mode}, "
            f"budget_ratio={truth_cache_budget_ratio}, "
            f"safety_factor={truth_cache_safety_factor}"
        )
    else:
        _progress("[truth_cache] 短路: skip_plots=True 且 enable_eval=False，跳过真值路径")

    # 按 init_tag 共享 MetricsAccumulator：所有模型跑完后一次性写 CSV/时序图，避免相互覆盖
    acc_by_tag: Dict[str, MetricsAccumulator] = {}
    valid_times_by_tag: Dict[str, Dict[Tuple[str, str], List]] = {}
    lon_by_tag: Dict[str, np.ndarray] = {}
    lat_by_tag: Dict[str, np.ndarray] = {}
    vars_union_by_tag: Dict[str, set] = {}

    timing = RollingInferenceTiming(
        enable_cpu=enable_cpu_timing,
        enable_gpu=enable_gpu_timing,
    )
    if enable_gpu_timing and not timing.gpu_active:
        _progress("[timing] GPU 计时已请求但当前环境无可用 GPU/DCU，将仅统计 CPU 时间")

    # ---------------------------------------------------------------
    # 外层循环：模型；内层循环：日期
    # 每个模型独占一次 VRAM：加载 → 跑完所有日期 → 卸载 → 下一个模型
    # 避免多模型同时占用 VRAM 导致 OOM。
    # ---------------------------------------------------------------
    for model_name in model_names:
        display_name = _MODEL_DISPLAY_NAMES.get(model_name.lower(), model_name)

        # 逐模型加载
        _progress(f"[{display_name}] 加载模型权重...")
        registry = build_registry(models_cfg_path, device=device, only=[model_name])
        if model_name not in registry:
            _progress(f"[{display_name}] 加载失败，跳过")
            continue
        model = registry.get(model_name)

        if hasattr(model, '_data_adapter'):
            model._data_adapter = adapter
        if model_name.lower() == "graphcast_official_operational" and hasattr(model, "_cfg"):
            blob_mode_env = os.environ.get("GC_BLOB_INPUT_MODE", "").strip().lower()
            if blob_mode_env:
                model._cfg["rollout_blob_input_mode"] = blob_mode_env
                if hasattr(model, "_blob_input_mode"):
                    model._blob_input_mode = blob_mode_env
                _progress(
                    "[GC_Official_Oper] rollout_blob_input_mode="
                    f"{blob_mode_env} (from GC_BLOB_INPUT_MODE)"
                )
            lock_cache_dir = bool(model._cfg.get("lock_rollout_cache_dir", False))
            if not lock_cache_dir:
                runtime_cache_dir = Path(output_root) / "_gc_oper_cache"
                runtime_cache_dir.mkdir(parents=True, exist_ok=True)
                model._cfg["rollout_cache_dir"] = str(runtime_cache_dir)
                if hasattr(model, "_rollout_cache"):
                    model._rollout_cache = runtime_cache_dir
                _progress(
                    "[GC_Official_Oper] rollout_cache_dir="
                    f"{runtime_cache_dir} (runtime default under output_root)"
                )
            else:
                _progress(
                    "[GC_Official_Oper] rollout_cache_dir locked by config: "
                    f"{model._cfg.get('rollout_cache_dir')}"
                )

        sfc_vars = variables if variables else model.get_surface_var_names()
        step_h = model.get_step_hours()
        if lead_step % step_h != 0:
            raise ValueError(
                f"[{display_name}] lead_step={lead_step}h 不能被模型步长 {step_h}h 整除。"
                "请调整 --lead-step 或模型配置。"
            )
        is_pangu = model_name.lower() == "pangu"
        model_fallback_used: List[Dict[str, object]] = []

        def _record_fallback(
            *,
            channel: str,
            strategy: str,
            detail: str,
            init_tag: Optional[str] = None,
            lead_hours: Optional[int] = None,
        ) -> None:
            evt: Dict[str, object] = {
                "channel": channel,
                "strategy": strategy,
                "detail": detail,
            }
            if init_tag is not None:
                evt["init_tag"] = init_tag
            if lead_hours is not None:
                evt["lead_hours"] = int(lead_hours)
            model_fallback_used.append(evt)
            _progress(
                f"[{display_name}] fallback channel={channel} strategy={strategy} detail={detail}"
            )

        if hasattr(model, "_fallback_recorder"):
            model._fallback_recorder = _record_fallback

        try:
            timing.begin_model(display_name)
            for date in dates:
                model_fallback_used = []
                init_dt = datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), init_hour)
                init_tag = init_dt.strftime("%Y%m%dT%H")

                # 加载起报 blob
                try:
                    init_blob = adapter.load_blob(date, init_hour)
                except FileNotFoundError as e:
                    _progress(
                        f"[{display_name}] 跳过 {init_tag}（缺少起报或面场文件，未执行滚动推理）: {e}"
                    )
                    continue
                mem_store.observe_blob_keys(
                    source_id=source_id,
                    blob_keys=list(init_blob.keys()),
                    optional_aliases=OPTIONAL_CHANNEL_ALIASES,
                )

                prev_dt = init_dt - timedelta(hours=step_h)
                prev_blob = adapter.load_blob_safe(prev_dt.strftime("%Y%m%d"), prev_dt.hour)
                if prev_blob is None:
                    _progress(
                        f"[{display_name}] t-{step_h}h ({prev_dt.strftime('%Y%m%d %H')}Z) "
                        f"数据不存在，使用 t=0 初始场作为 prev_blob 替代"
                    )
                    prev_blob = init_blob
                else:
                    mem_store.observe_blob_keys(
                        source_id=source_id,
                        blob_keys=list(prev_blob.keys()),
                        optional_aliases=OPTIONAL_CHANNEL_ALIASES,
                    )

                enabled_optional, missing_optional = decide_optional_channels(
                    model_name=model_name,
                    blob_keys=list(init_blob.keys()),
                    is_available_cb=lambda alias: mem_store.is_optional_available(source_id, alias),
                )
                if hasattr(model, "_enabled_optional_channels"):
                    model._enabled_optional_channels = list(enabled_optional)

                lat = init_blob.get("lat", np.linspace(90.0, -90.0, 721, dtype=np.float32))
                lon = init_blob.get("lon", np.arange(0.0, 360.0, 0.25, dtype=np.float32))
                lon_by_tag[init_tag] = lon
                lat_by_tag[init_tag] = lat
                vars_union_by_tag.setdefault(init_tag, set()).update(sfc_vars)

                _progress(f"[{display_name}] 开始滚动推理 {init_tag}，步长={step_h}h，共 {n_steps} 步")

                # 初始化模型状态
                try:
                    state = model.init_state(init_blob, prev_blob=prev_blob, init_dt=init_dt)
                except (ValueError, RuntimeError, FileNotFoundError) as e:
                    _progress(f"[{display_name}] init_state 失败: {e}")
                    continue

                if enable_eval and eval_run_downscale:
                    if init_tag not in acc_by_tag:
                        acc_by_tag[init_tag] = MetricsAccumulator(
                            lats=lat,
                            metrics=metrics,
                            save_diff=save_diff,
                            save_diff_nc=save_diff_nc,
                        )
                        valid_times_by_tag[init_tag] = {}
                acc = (
                    acc_by_tag.get(init_tag)
                    if (enable_eval and eval_run_downscale)
                    else None
                )

                # 真值预加载：每模型每日期起始前一次性加载整个预报区间，帧循环中直接读缓存
                if need_truth and truth_cache is not None:
                    truth_cache.preload_range(
                        truth_adapter,
                        init_dt,
                        leads,
                        progress_fn=_progress,
                    )

                # 逐段累计耗时（始终开启）：推理/真值加载/NPY写/画图/评估
                _stage_cum: Dict[str, float] = {
                    "model_step_s": 0.0,
                    "truth_load_s": 0.0,
                    "npy_write_s": 0.0,
                    "nc_write_s": 0.0,
                    "plot_s": 0.0,
                    "eval_s": 0.0,
                }
                # 可选：中间过程窗口输出（ROLLING_STEP_PROFILE=1 时每 N 帧打印一次窗口均值）
                step_profile_enabled = _env_true("ROLLING_STEP_PROFILE", "0")
                try:
                    step_profile_every = int(os.environ.get("ROLLING_STEP_PROFILE_EVERY", "8"))
                except ValueError:
                    step_profile_every = 8
                step_profile_every = max(1, step_profile_every)
                step_totals: Dict[str, float] = dict(_stage_cum)  # 窗口副本
                step_count_window = 0

                pressure_var_names = model.get_pressure_var_names()
                level_vals: Optional[np.ndarray] = None
                if save_nc and pressure_var_names:
                    from cepri_loader import PANGU_LEVELS
                    level_vals = np.asarray(PANGU_LEVELS, dtype=np.float32)
                nc_out: Optional[Path] = None
                if save_nc:
                    nc_out = nc_dir(Path(output_root), init_tag, display_name)
                    nc_out.mkdir(parents=True, exist_ok=True)
                compare_plot_dir: Optional[Path] = None
                if not skip_plots:
                    compare_plot_dir = rolling_plot_dir(Path(output_root), init_tag, display_name)
                    compare_plot_dir.mkdir(parents=True, exist_ok=True)

                # NPY 写出器
                with NpyStackWriter(
                    output_root=Path(output_root),
                    model_name=display_name,
                    init_tag=init_tag,
                    variables=sfc_vars,
                    n_steps=n_steps,
                    shape_hw=(len(lat), len(lon)),
                    pangu_suffix=is_pangu,
                    on_missing_var=lambda step_idx, var: _record_fallback(
                        channel=var,
                        strategy="skip_write",
                        detail="prediction missing var when writing npy",
                        init_tag=init_tag,
                        lead_hours=leads[step_idx] if 0 <= step_idx < len(leads) else None,
                    ),
                ) as npy_writer:

                    for si, lead in enumerate(leads):
                        steps_needed = lead_step // step_h
                        t_start = time.perf_counter()
                        for _ in range(steps_needed):
                            state = model.step(state)
                        _dt = time.perf_counter() - t_start
                        _stage_cum["model_step_s"] += _dt
                        if step_profile_enabled:
                            step_totals["model_step_s"] += _dt

                        current_lead = state.lead
                        valid_dt = init_dt + timedelta(hours=current_lead)
                        truth_blob = None
                        if need_truth:
                            t_start = time.perf_counter()
                            if truth_cache is not None:
                                # 从缓存读取（O(1)内存查找或磁盘单文件读取，无重复 I/O）
                                truth_blob = truth_cache.get(valid_dt)
                                if truth_blob is None:
                                    # 缓存未命中（预加载失败），降级为在线加载
                                    try:
                                        truth_blob = truth_adapter.load_blob_for_valid_time(valid_dt)
                                    except Exception:
                                        pass
                            else:
                                # need_truth=True 但无缓存（不应发生），保持原有行为
                                try:
                                    truth_blob = truth_adapter.load_blob_for_valid_time(valid_dt)
                                except Exception:
                                    pass
                            _dt = time.perf_counter() - t_start
                            _stage_cum["truth_load_s"] += _dt
                            if step_profile_enabled:
                                step_totals["truth_load_s"] += _dt

                        pred_sfc = extract_surface_vars(
                            state.blob,
                            sfc_vars,
                            on_missing=lambda var_name, bkey: _record_fallback(
                                channel=var_name,
                                strategy="optional_skip",
                                detail=f"missing in model blob ({bkey})",
                                init_tag=init_tag,
                                lead_hours=current_lead,
                            ),
                        )
                        t_start = time.perf_counter()
                        npy_writer.write_step(si, pred_sfc)
                        _dt = time.perf_counter() - t_start
                        _stage_cum["npy_write_s"] += _dt
                        if step_profile_enabled:
                            step_totals["npy_write_s"] += _dt
                        timing.add_frames(1)

                        # NC 写出
                        if save_nc:
                            t_start = time.perf_counter()
                            vars_2d = {f"sfc_{k}": v for k, v in pred_sfc.items()}
                            vars_3d = None
                            if pressure_var_names:
                                vars_3d = {}
                                for pv in pressure_var_names:
                                    bkey = f"pangu_{pv}"
                                    if bkey in state.blob:
                                        vars_3d[f"pres_{pv}"] = state.blob[bkey]
                            write_step_nc(
                                (nc_out or Path(output_root)) / f"lead_{current_lead:03d}.nc",
                                model=display_name,
                                init_time=init_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                lead_hours=current_lead,
                                valid_time=valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                vars_2d=vars_2d,
                                vars_3d=vars_3d if vars_3d else None,
                                level_values=(level_vals if vars_3d else None),
                                lat=lat,
                                lon=lon,
                            )
                            _dt = time.perf_counter() - t_start
                            _stage_cum["nc_write_s"] += _dt
                            if step_profile_enabled:
                                step_totals["nc_write_s"] += _dt

                        # 对比图
                        if not skip_plots:
                            t_start = time.perf_counter()
                            if cpu_timing_exclude_plots:
                                timing.pause_cpu()
                            try:
                                truth_sfc = (
                                    extract_surface_vars(
                                        truth_blob,
                                        sfc_vars,
                                        on_missing=lambda var_name, bkey: _record_fallback(
                                            channel=var_name,
                                            strategy="truth_missing",
                                            detail=f"truth blob missing ({bkey})",
                                            init_tag=init_tag,
                                            lead_hours=current_lead,
                                        ),
                                    )
                                    if truth_blob else None
                                )
                                for var_name, pred_arr in pred_sfc.items():
                                    truth_arr = truth_sfc.get(var_name) if truth_sfc else None
                                    if truth_arr is not None:
                                        pred_arr, truth_arr = harmonize_surface_pair(
                                            var_name, pred_arr, truth_arr
                                        )
                                    plot_compare(
                                        (compare_plot_dir or Path(output_root)) / f"{var_name}_lead{current_lead:03d}.png",
                                        pred_arr,
                                        truth_arr,
                                        title=(
                                            f"{display_name} +{current_lead}h {var_name} | {init_tag}"
                                        ),
                                        cmap="RdBu_r" if var_name in ("u10", "v10") else "viridis",
                                    )
                            finally:
                                if cpu_timing_exclude_plots:
                                    timing.resume_cpu()
                            _dt = time.perf_counter() - t_start
                            _stage_cum["plot_s"] += _dt
                            if step_profile_enabled:
                                step_totals["plot_s"] += _dt

                        # 内嵌评估
                        if acc is not None and truth_blob is not None:
                            t_start = time.perf_counter()
                            truth_sfc_eval = extract_surface_vars(
                                truth_blob,
                                sfc_vars,
                                on_missing=lambda var_name, bkey: _record_fallback(
                                    channel=var_name,
                                    strategy="truth_missing",
                                    detail=f"eval truth blob missing ({bkey})",
                                    init_tag=init_tag,
                                    lead_hours=current_lead,
                                ),
                            )
                            for var_name, pred_arr in pred_sfc.items():
                                if var_name in truth_sfc_eval:
                                    p_h, t_h = harmonize_surface_pair(
                                        var_name, pred_arr, truth_sfc_eval[var_name]
                                    )
                                    acc.add(
                                        display_name, var_name, current_lead, p_h, t_h
                                    )
                                    if save_diff_nc:
                                        valid_times_by_tag[init_tag].setdefault(
                                            (display_name, var_name), []
                                        ).append(valid_dt)
                            _dt = time.perf_counter() - t_start
                            _stage_cum["eval_s"] += _dt
                            if step_profile_enabled:
                                step_totals["eval_s"] += _dt

                        if step_profile_enabled:
                            step_count_window += 1
                            is_last_step = (si == len(leads) - 1)
                            if step_count_window >= step_profile_every or is_last_step:
                                n = max(1, step_count_window)
                                _progress(
                                    f"[{display_name}] [step_profile] {init_tag} lead={current_lead}h "
                                    f"avg(step={step_totals['model_step_s']/n:.3f}s, "
                                    f"truth={step_totals['truth_load_s']/n:.3f}s, "
                                    f"npy={step_totals['npy_write_s']/n:.3f}s, "
                                    f"nc={step_totals['nc_write_s']/n:.3f}s, "
                                    f"plot={step_totals['plot_s']/n:.3f}s, "
                                    f"eval={step_totals['eval_s']/n:.3f}s)"
                                )
                                for k in step_totals:
                                    step_totals[k] = 0.0
                                step_count_window = 0

                        if si % 8 == 0 or si == 0:
                            _progress(f"[{display_name}] {init_tag} lead={current_lead}h done")

                _progress(f"[{display_name}] {init_tag} NPY 已保存: {npy_writer.get_paths()}")

                # ── 各阶段累计耗时汇总（始终输出）──
                _n_f = max(1, len(leads))
                _stage_parts = [
                    f"model_step={_stage_cum['model_step_s']:.1f}s"
                    f"({_stage_cum['model_step_s']/_n_f:.2f}s/f)",
                    f"truth={_stage_cum['truth_load_s']:.1f}s",
                    f"npy={_stage_cum['npy_write_s']:.1f}s",
                    f"plot={_stage_cum['plot_s']:.1f}s",
                    f"eval={_stage_cum['eval_s']:.1f}s",
                ]
                if _stage_cum["nc_write_s"] > 0:
                    _stage_parts.append(f"nc={_stage_cum['nc_write_s']:.1f}s")
                _progress(
                    f"[{display_name}] [stage_totals] {init_tag}  " + "  ".join(_stage_parts)
                )

                model_fallback_used.extend(npy_writer.get_missing_events())
                _write_fallback_meta(
                    output_root=Path(output_root),
                    display_name=display_name,
                    init_tag=init_tag,
                    source_id=source_id,
                    model_name=model_name,
                    enabled_optional_channels=enabled_optional,
                    missing_optional_channels=missing_optional,
                    fallback_used=model_fallback_used,
                )
                mem_store.update_model_decision(
                    source_id=source_id,
                    model_name=model_name,
                    enabled_channels=enabled_optional,
                    missing_channels=missing_optional,
                    fallback_used=model_fallback_used,
                )

                _progress(f"[{display_name}] 完成 {init_tag}")

        finally:
            cpu_s, gpu_s, frames, avg_cpu, avg_gpu = timing.end_model()
            _parts = [
                f"frames={frames}",
                f"CPU={cpu_s:.2f}s",
                f"CPU均/帧={avg_cpu:.3f}s" if avg_cpu is not None else "CPU均/帧=N/A",
            ]
            if gpu_s is not None:
                _parts += [
                    f"GPU={gpu_s:.2f}s",
                    f"GPU均/帧={avg_gpu:.3f}s" if avg_gpu is not None else "GPU均/帧=N/A",
                ]
            _progress(f"[{display_name}] [timing] " + "  ".join(_parts))
            _progress(f"[{display_name}] 卸载模型，释放 VRAM...")
            model.unload()

    # --- 推理时间汇总 ---
    # model 并行模式下各 rank 独立结束、stdout 交错，需通过分片文件在 rank0 统一打印。
    # date 并行模式或单进程时直接打印本 rank 汇总。
    _rank = int(os.environ.get("RANK", "0"))
    _world = int(os.environ.get("WORLD_SIZE", "1"))

    if _world > 1 and pmode == "model":
        _job_id = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
        _shard_dir = Path(output_root) / "_timing_shards" / _job_id
        _shard_dir.mkdir(parents=True, exist_ok=True)
        _shard_path = _shard_dir / f"rank{_rank:04d}_of_{_world:04d}.json"
        # 原子写：先写 .tmp 再 rename，防止 rank0 读到半完整文件
        _shard_tmp = _shard_path.with_suffix(".tmp")
        _shard_tmp.write_text(
            json.dumps(timing.to_dict_list(), ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(_shard_tmp), str(_shard_path))
        _progress(f"[timing] 计时分片已写出 → {_shard_path}")

        if _rank == 0:
            # 等待所有 rank 的分片文件就绪，超时则跳过缺失 rank
            _deadline = time.time() + 300.0
            _all_records: List[Dict] = []
            for _r in range(_world):
                _sf = _shard_dir / f"rank{_r:04d}_of_{_world:04d}.json"
                while not _sf.exists():
                    if time.time() > _deadline:
                        _progress(
                            f"[timing] 等待 rank{_r} 计时分片超时（300s），该 rank 数据将缺失"
                        )
                        break
                    time.sleep(2.0)
                if _sf.exists():
                    try:
                        _all_records.extend(
                            json.loads(_sf.read_text(encoding="utf-8"))
                        )
                    except Exception as _e:
                        _progress(f"[timing] 读取 rank{_r} 计时分片失败: {_e}")

            # 按原始模型顺序排列汇总行
            _disp_order = [
                _MODEL_DISPLAY_NAMES.get(m.lower(), m) for m in _orig_model_names
            ]
            for row in RollingInferenceTiming.summary_rows_from_dicts(
                _all_records,
                model_order=_disp_order,
                label="全部 rank 汇总",
            ):
                _progress(row)
    else:
        for row in timing.summary_rows():
            _progress(row)

    if enable_eval and eval_run_downscale and acc_by_tag:
        for init_tag, acc in acc_by_tag.items():
            eval_out_dir = eval_downscale_dir(Path(output_root), init_tag, max_lead)
            acc.save(
                save_dir=eval_out_dir,
                time_tag=init_tag,
                lon=lon_by_tag.get(init_tag),
                valid_times=valid_times_by_tag.get(init_tag) if save_diff_nc else None,
                model_order=_EVAL_MODEL_ORDER,
            )
            _progress(f"多模型评估已写入: {eval_out_dir}")

    if enable_eval:
        mg = {m.lower() for m in (eval_multigrid_modes or [])}
        if auto_multigrid and not mg:
            mg = {"fidelity_nn", "upscale"}
        mg &= {"fidelity_nn", "fidelity_exact", "upscale"}
        if mg:
            tag_list = sorted(lat_by_tag.keys()) if lat_by_tag else [
                datetime(
                    int(d[:4]), int(d[4:6]), int(d[6:8]), init_hour,
                ).strftime("%Y%m%dT%H")
                for d in dates
            ]
            for init_tag in tag_list:
                init_dt = datetime(
                    int(init_tag[:4]),
                    int(init_tag[4:6]),
                    int(init_tag[6:8]),
                    int(init_tag[9:11]),
                )
                lat_c = lat_by_tag.get(init_tag)
                lon_c = lon_by_tag.get(init_tag)
                if lat_c is None or lon_c is None:
                    _progress(
                        f"[multigrid] 跳过 {init_tag}（无 lat/lon，可能该日起报未加载）",
                    )
                    continue
                vlist = sorted(vars_union_by_tag.get(init_tag, []))
                if not vlist:
                    vlist = ["u10", "v10", "t2m", "msl"]
                run_multigrid_eval_from_npy(
                    output_root=Path(output_root),
                    truth_adapter=truth_adapter,
                    init_dt=init_dt,
                    init_tag=init_tag,
                    max_lead=max_lead,
                    lead_step=lead_step,
                    model_slugs=[m.lower() for m in model_names],
                    variables=vlist,
                    metrics=metrics,
                    coarse_lat=lat_c,
                    coarse_lon=lon_c,
                    eval_modes=mg,
                    fidelity_max_deg=fidelity_max_deg,
                    model_order=_EVAL_MODEL_ORDER,
                    spatial_plots=not skip_plots,
                )
            _progress(
                f"多分辨率评估 ({', '.join(sorted(mg))}) 已处理 init_tags={tag_list}",
            )

    # 作业结束：统一关闭真值缓存（内存缓存随 dict 释放，磁盘缓存按配置清理）
    if truth_cache is not None:
        _progress(f"[truth_cache] 作业结束，关闭缓存: {truth_cache.stats()}")
        truth_cache.close()

    _progress("全部模型/日期处理完成")
