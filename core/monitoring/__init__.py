from datetime import datetime

from .hardware_logger import HardwareLogger, start_hardware_logger
from .inference_timing import (
    ProcessCpuStopwatch,
    TorchGpuSegmentStopwatch,
    RollingInferenceTiming,
    SegmentResult,
    timed_segment,
)


def progress(msg: str, tag: str = "pipeline") -> None:
    """统一进度打印：[YYYY-MM-DD HH:MM:SS] [{tag}] {msg}"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", flush=True)


__all__ = [
    "HardwareLogger",
    "start_hardware_logger",
    "ProcessCpuStopwatch",
    "TorchGpuSegmentStopwatch",
    "RollingInferenceTiming",
    "SegmentResult",
    "timed_segment",
    "progress",
]
