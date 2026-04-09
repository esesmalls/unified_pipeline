"""
推理时间统计模块：进程 CPU 时间 + GPU（DCU/ROCm）设备区间时间。

- CPU：基于 time.process_time()，统计进程级用户态 + 内核态 CPU 时间。
  note: process_time() 为整进程累计（含子线程），与当前以主线程为主的流水线兼容。
  与墙钟时间不同：若推理主要在 GPU，CPU 秒数可能远小于实际经过时间。

- GPU：基于 torch.cuda.Event(enable_timing=True)，测量默认 CUDA/HIP 流上的
  设备时间轴跨度（含排队与等待）。在 ROCm/DCU 环境下 PyTorch 仍走 torch.cuda
  命名空间（HIP 后端），行为与 NVIDIA CUDA 一致。
  帧间若存在 CPU 侧 I/O（读真值、写盘），GPU 空闲也会体现在区间时长内，建议与
  CPU 计时并列解读。
  torch 不可用或 torch.cuda.is_available()==False 时自动跳过 GPU 统计，不影响 CPU。

典型用法（滚动推理场景）：
    from core.monitoring import RollingInferenceTiming

    timing = RollingInferenceTiming(enable_cpu=True, enable_gpu=True)
    for model_name in model_names:
        timing.begin_model(model_name)
        for ...:
            ... # inference + write_step
            timing.add_frames(1)
        cpu_s, gpu_s, frames, avg_cpu, avg_gpu = timing.end_model()
        logger.info(f"{model_name}: CPU={cpu_s:.1f}s GPU={gpu_s}s frames={frames}")
    for row in timing.summary_rows():
        logger.info(row)

通用上下文管理器（用于其它脚本）：
    from core.monitoring import timed_segment

    with timed_segment("data-load") as seg:
        data = load(...)
    print(seg.cpu_elapsed_s, seg.gpu_elapsed_s)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# GPU 可用性检测（懒加载，仅 import 一次）
# ---------------------------------------------------------------------------

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# ProcessCpuStopwatch
# ---------------------------------------------------------------------------

class ProcessCpuStopwatch:
    """轻量进程 CPU 计时器，支持 pause/resume（用于排除画图等非推理阶段）。

    使用 time.process_time() 差分，返回用户态 + 内核态 CPU 秒数。
    不计 sleep 或 GPU 内核独立执行的时间。
    """

    def __init__(self) -> None:
        self._start: float = time.process_time()
        self._accumulated: float = 0.0
        self._paused: bool = False

    def reset(self) -> None:
        self._start = time.process_time()
        self._accumulated = 0.0
        self._paused = False

    def pause(self) -> None:
        """暂停计时：将当前已经历的时间存入 _accumulated，停止累积。"""
        if not self._paused:
            self._accumulated += time.process_time() - self._start
            self._paused = True

    def resume(self) -> None:
        """恢复计时：重新设定起始点。"""
        if self._paused:
            self._start = time.process_time()
            self._paused = False

    def elapsed_s(self) -> float:
        if self._paused:
            return self._accumulated
        return self._accumulated + (time.process_time() - self._start)


# ---------------------------------------------------------------------------
# TorchGpuSegmentStopwatch
# ---------------------------------------------------------------------------

class TorchGpuSegmentStopwatch:
    """基于 torch.cuda.Event 的 GPU 区间计时器（支持 CUDA / ROCm/DCU）。

    enabled=False 或 GPU 不可用时所有方法为空操作，不触发 synchronize()。
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled and _torch_cuda_available()
        self._start_ev = None
        self._end_ev = None
        if self._enabled:
            import torch
            self._torch = torch
            self._start_ev = torch.cuda.Event(enable_timing=True)
            self._end_ev = torch.cuda.Event(enable_timing=True)

    @property
    def active(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if self._enabled:
            self._start_ev.record()

    def stop_and_elapsed_s(self) -> Optional[float]:
        """记录结束事件、同步、返回区间秒数；不启用时返回 None。"""
        if not self._enabled:
            return None
        self._end_ev.record()
        self._torch.cuda.synchronize()
        return self._start_ev.elapsed_time(self._end_ev) / 1000.0

    def reset(self) -> None:
        if self._enabled:
            self._start_ev = self._torch.cuda.Event(enable_timing=True)
            self._end_ev = self._torch.cuda.Event(enable_timing=True)


# ---------------------------------------------------------------------------
# _ModelRecord
# ---------------------------------------------------------------------------

@dataclass
class _ModelRecord:
    model_key: str
    cpu_s: float = 0.0
    gpu_s: Optional[float] = None
    frames: int = 0

    @property
    def avg_cpu_s(self) -> Optional[float]:
        return self.cpu_s / self.frames if self.frames > 0 else None

    @property
    def avg_gpu_s(self) -> Optional[float]:
        if self.gpu_s is None or self.frames <= 0:
            return None
        return self.gpu_s / self.frames


# ---------------------------------------------------------------------------
# RollingInferenceTiming
# ---------------------------------------------------------------------------

class RollingInferenceTiming:
    """滚动推理 CPU + GPU 时间聚合器。

    线程安全性：不额外加锁，设计目标是主线程顺序调用（与流水线模式一致）。

    Args:
        enable_cpu: 是否统计进程 CPU 时间（默认 True）。
        enable_gpu: 是否统计 GPU 设备时间（默认 True；GPU 不可用时自动退化为 False）。
    """

    def __init__(self, enable_cpu: bool = True, enable_gpu: bool = True) -> None:
        self._enable_cpu = enable_cpu
        self._gpu_sw = TorchGpuSegmentStopwatch(enabled=enable_gpu)
        self._records: List[_ModelRecord] = []
        self._current: Optional[_ModelRecord] = None
        self._cpu_sw: Optional[ProcessCpuStopwatch] = None

    @property
    def gpu_active(self) -> bool:
        return self._gpu_sw.active

    def begin_model(self, model_key: str) -> None:
        """开始统计新模型的滚动段。model_key 应为日志中显示的模型名。"""
        self._current = _ModelRecord(model_key=model_key)
        if self._enable_cpu:
            self._cpu_sw = ProcessCpuStopwatch()
        self._gpu_sw.reset()
        self._gpu_sw.start()

    def add_frames(self, n: int = 1) -> None:
        """每成功输出一帧后调用，递增帧计数。"""
        if self._current is not None:
            self._current.frames += n

    def end_model(self) -> Tuple[float, Optional[float], int, Optional[float], Optional[float]]:
        """结束当前模型段，返回 (cpu_s, gpu_s, frames, avg_cpu_s, avg_gpu_s)。

        gpu_s / avg_gpu_s 为 None 表示 GPU 统计未启用或不可用。
        avg_cpu_s / avg_gpu_s 为 None 表示帧数为 0。
        """
        if self._current is None:
            return 0.0, None, 0, None, None

        if self._enable_cpu and self._cpu_sw is not None:
            self._current.cpu_s = self._cpu_sw.elapsed_s()

        gpu_s = self._gpu_sw.stop_and_elapsed_s()
        if gpu_s is not None:
            self._current.gpu_s = gpu_s

        rec = self._current
        self._records.append(rec)
        self._current = None
        self._cpu_sw = None

        return rec.cpu_s, rec.gpu_s, rec.frames, rec.avg_cpu_s, rec.avg_gpu_s

    def pause_cpu(self) -> None:
        """暂停 CPU 计时（用于排除画图等非推理段）。"""
        if self._enable_cpu and self._cpu_sw is not None:
            self._cpu_sw.pause()

    def resume_cpu(self) -> None:
        """恢复 CPU 计时。"""
        if self._enable_cpu and self._cpu_sw is not None:
            self._cpu_sw.resume()

    def to_dict_list(self) -> List[Dict]:
        """将所有模型记录导出为 dict 列表（用于多 rank 间传递汇总数据）。"""
        result = []
        for r in self._records:
            d: Dict = {
                "model": r.model_key,
                "cpu_s": r.cpu_s,
                "gpu_s": r.gpu_s,
                "frames": r.frames,
            }
            result.append(d)
        return result

    @classmethod
    def summary_rows_from_dicts(
        cls,
        records: List[Dict],
        model_order: Optional[List[str]] = None,
        label: str = "",
    ) -> List[str]:
        """从 to_dict_list() 产出的 dict 列表生成汇总行（多 rank 合并场景）。

        Args:
            records:     每个 dict 包含 model/cpu_s/gpu_s/frames 字段。
            model_order: 指定模型排列顺序（按列表顺序排序输出）。
            label:       汇总标题附加说明（如 "合并 rank"）。
        """
        if not records:
            return []
        recs = [
            _ModelRecord(
                model_key=d["model"],
                cpu_s=d.get("cpu_s", 0.0),
                gpu_s=d.get("gpu_s"),
                frames=d.get("frames", 0),
            )
            for d in records
        ]
        if model_order:
            order_map = {m: i for i, m in enumerate(model_order)}
            recs.sort(key=lambda r: order_map.get(r.model_key, len(model_order)))

        title = f"推理时间汇总（{label}）" if label else "推理时间汇总"
        lines = [f"[timing] ===== {title} ====="]
        has_gpu = any(r.gpu_s is not None for r in recs)
        header = f"{'模型':<28} {'帧数':>6}  {'CPU总(s)':>10}  {'CPU均/帧(s)':>12}"
        if has_gpu:
            header += f"  {'GPU总(s)':>10}  {'GPU均/帧(s)':>12}"
        lines.append(f"[timing] {header}")
        for r in recs:
            avg_cpu = f"{r.avg_cpu_s:.3f}" if r.avg_cpu_s is not None else "  N/A"
            row = (
                f"[timing] {r.model_key:<28} {r.frames:>6}  "
                f"{r.cpu_s:>10.2f}  {avg_cpu:>12}"
            )
            if has_gpu:
                if r.gpu_s is not None:
                    avg_gpu = f"{r.avg_gpu_s:.3f}" if r.avg_gpu_s is not None else "  N/A"
                    row += f"  {r.gpu_s:>10.2f}  {avg_gpu:>12}"
                else:
                    row += f"  {'N/A':>10}  {'N/A':>12}"
            lines.append(row)
        lines.append("[timing] ==========================================")
        return lines

    def summary_rows(self) -> List[str]:
        """返回各模型汇总行字符串列表，便于直接传给日志函数。"""
        return self.summary_rows_from_dicts(
            records=self.to_dict_list(),
            label="本 rank",
        )


# ---------------------------------------------------------------------------
# 通用上下文管理器（供其它脚本复用）
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """单次 timed_segment 的结果。"""
    name: str
    cpu_elapsed_s: float = 0.0
    gpu_elapsed_s: Optional[float] = None


@contextmanager
def timed_segment(
    name: str = "",
    enable_cpu: bool = True,
    enable_gpu: bool = True,
) -> Generator[SegmentResult, None, None]:
    """通用 CPU + GPU 区间计时上下文管理器。

    使用示例::

        from core.monitoring import timed_segment

        with timed_segment("data-load") as seg:
            data = load(...)
        print(f"CPU: {seg.cpu_elapsed_s:.3f}s  GPU: {seg.gpu_elapsed_s}")

    Args:
        name:       区间名称（仅作标识）。
        enable_cpu: 是否统计 CPU 时间。
        enable_gpu: 是否统计 GPU 时间（GPU 不可用时自动跳过）。
    """
    result = SegmentResult(name=name)
    cpu_sw = ProcessCpuStopwatch() if enable_cpu else None
    gpu_sw = TorchGpuSegmentStopwatch(enabled=enable_gpu)
    gpu_sw.start()
    try:
        yield result
    finally:
        if cpu_sw is not None:
            result.cpu_elapsed_s = cpu_sw.elapsed_s()
        result.gpu_elapsed_s = gpu_sw.stop_and_elapsed_s()
