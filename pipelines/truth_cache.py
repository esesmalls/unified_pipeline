"""
作业内全局真值缓存管理器（TruthCacheManager）。

支持 memory / disk / auto 三种模式：
- memory: 进程级 dict 缓存，模型切换不清理，作业结束随进程释放
- disk: 磁盘缓存（按 rank 子目录隔离），作业结束自动清理（除非 keep_disk_cache=True）
- auto: 运行时评估第一帧真值大小与可用内存，自动选择 memory 或 disk

多 rank / DCU 并行安全：
- memory 模式天然是每 rank 进程私有（无共享写风险）
- disk 模式默认使用 rank 子目录隔离：{cache_root}/{job_id}/rank{R}/…
  避免跨 rank 写冲突；每 rank 仅读写自己的目录

使用方式：
    cache = TruthCacheManager(
        mode="auto",
        cache_root=Path(output_root) / "_truth_cache",
        rank=rank,
        job_id=job_id,
    )
    # 每模型开始前（need_truth=True）预加载整个预报区间真值：
    cache.preload_range(truth_adapter, init_dt, leads, progress_fn=_progress)
    # 帧循环中读取（已在缓存中，O(1)内存查找或磁盘单文件读取）：
    blob = cache.get(valid_dt)
    # 作业结束：
    cache.close()
"""
from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# 内存检测工具
# ---------------------------------------------------------------------------

def _get_available_memory_bytes() -> int:
    """获取当前进程可用系统内存（字节）。
    优先用 psutil，回退 /proc/meminfo，都不可用时返回 64 GB 保守估计。
    """
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass
    # 无法获取时返回保守估计，保证 auto 模式不意外使用大量内存
    return 32 * 1024 ** 3  # 32 GB


def _estimate_blob_bytes(blob: dict) -> int:
    """估算 truth blob 所有 numpy 数组的总 nbytes。"""
    import numpy as np
    total = 0
    for v in blob.values():
        if hasattr(v, "nbytes"):
            total += int(v.nbytes)
        elif isinstance(v, (list, tuple)):
            try:
                total += int(np.asarray(v).nbytes)
            except Exception:
                pass
    return total


# ---------------------------------------------------------------------------
# TruthCacheManager
# ---------------------------------------------------------------------------

class TruthCacheManager:
    """
    作业内全局真值缓存管理器。

    模型切换时不清理缓存（跨模型复用相同 valid_dt 的真值）。
    作业结束时调用 close()，默认清除磁盘缓存。
    """

    def __init__(
        self,
        mode: str = "auto",
        cache_root: Optional[Path] = None,
        budget_ratio: float = 0.4,
        safety_factor: float = 1.3,
        keep_disk_cache: bool = False,
        rank: int = 0,
        job_id: str = "job",
    ) -> None:
        """
        Args:
            mode:             缓存模式：auto | memory | disk（默认 auto）
            cache_root:       磁盘缓存根目录（disk/auto 时使用；None=禁用磁盘缓存）
            budget_ratio:     auto 模式：可用内存中预留给缓存的比例（默认 0.4）
            safety_factor:    auto 模式：估算总需求时的安全系数（默认 1.3）
            keep_disk_cache:  True=作业结束后保留磁盘缓存（调试用）
            rank:             当前 rank（用于磁盘子目录隔离）
            job_id:           Slurm job id 或标识（用于磁盘缓存目录名）
        """
        if mode not in ("auto", "memory", "disk"):
            raise ValueError(f"TruthCacheManager: 未知 mode='{mode}'，须为 auto|memory|disk")
        self._mode_request = mode
        self._mode_effective: Optional[str] = None
        self._budget_ratio = budget_ratio
        self._safety_factor = safety_factor
        self._keep_disk_cache = keep_disk_cache
        self._rank = rank
        self._job_id = job_id

        # 内存缓存：key = valid_dt ISO 字符串
        self._mem: Dict[str, dict] = {}

        # 磁盘缓存根目录（含 rank 子目录隔离）
        self._disk_root: Optional[Path] = None
        if cache_root is not None:
            self._disk_root = Path(cache_root) / job_id / f"rank{rank}"

        # 统计
        self._preloaded_count = 0
        self._cache_hits = 0

    # ------------------------------------------------------------------
    # 公有接口
    # ------------------------------------------------------------------

    @property
    def effective_mode(self) -> str:
        """实际生效的缓存模式（auto 决定前返回请求值）。"""
        return self._mode_effective or self._mode_request

    def preload_range(
        self,
        truth_adapter: Any,
        init_dt: datetime,
        leads: List[int],
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        每模型开始前，一次性预加载并缓存整个预报区间的真值。

        已缓存的 valid_dt 会跳过加载（跨模型复用）。

        Args:
            truth_adapter:  真值适配器，须实现 load_blob_for_valid_time(datetime)
            init_dt:        起报时刻
            leads:          预报步长列表（小时），如 [6, 12, ..., 240]
            progress_fn:    进度打印回调（可选）
        """
        valid_dts = [init_dt + timedelta(hours=lead) for lead in leads]
        missing = [
            vdt for vdt in valid_dts
            if not self._is_cached(vdt)
        ]

        if not missing:
            if progress_fn:
                progress_fn(
                    f"[truth_cache] 全部 {len(leads)} 帧真值已在缓存中，跳过预加载"
                )
            return

        if progress_fn:
            progress_fn(
                f"[truth_cache] 预加载 {len(missing)}/{len(leads)} 帧真值 "
                f"(mode_requested={self._mode_request})..."
            )

        first_blob_done = self._mode_effective is not None
        loaded = 0
        for vdt in missing:
            try:
                blob = truth_adapter.load_blob_for_valid_time(vdt)
            except Exception as exc:
                if progress_fn:
                    progress_fn(f"[truth_cache] 警告: 预加载 {vdt} 失败: {exc}")
                continue

            if not first_blob_done:
                self._mode_effective = self._decide_mode(blob, len(leads))
                first_blob_done = True
                if progress_fn:
                    avail_gb = _get_available_memory_bytes() / 1024 ** 3
                    blob_mb = _estimate_blob_bytes(blob) / 1024 ** 2
                    total_mb = blob_mb * len(leads) * self._safety_factor
                    progress_fn(
                        f"[truth_cache] 内存决策: mode={self._mode_effective} "
                        f"(request={self._mode_request}, "
                        f"blob={blob_mb:.1f}MB, "
                        f"total_est={total_mb:.0f}MB, "
                        f"avail={avail_gb:.1f}GB, "
                        f"budget={avail_gb * self._budget_ratio:.1f}GB)"
                    )

            self._store(vdt, blob)
            loaded += 1

        if self._mode_effective is None:
            # 全部预加载失败（无数据）时回退
            self._mode_effective = self._mode_request if self._mode_request != "auto" else "memory"

        self._preloaded_count += loaded
        if progress_fn and loaded > 0:
            progress_fn(
                f"[truth_cache] 预加载完成: {loaded} 帧 → mode={self._mode_effective}, "
                f"内存缓存帧数={len(self._mem)}"
            )

    def get(self, valid_dt: datetime) -> Optional[dict]:
        """获取缓存的真值 blob。未命中返回 None（调用方自行处理缺失）。"""
        key = self._dt_key(valid_dt)
        if key in self._mem:
            self._cache_hits += 1
            return self._mem[key]
        disk_path = self._disk_path(valid_dt)
        if disk_path is not None and disk_path.exists():
            try:
                with open(disk_path, "rb") as fh:
                    blob = pickle.load(fh)
                self._cache_hits += 1
                return blob
            except Exception:
                pass
        return None

    def close(self) -> None:
        """
        作业结束时清理缓存资源。
        - 内存缓存：清空 dict（随进程释放）
        - 磁盘缓存：默认删除，keep_disk_cache=True 时保留（调试）
        """
        self._mem.clear()
        if (
            not self._keep_disk_cache
            and self._disk_root is not None
            and self._disk_root.exists()
        ):
            try:
                shutil.rmtree(str(self._disk_root), ignore_errors=True)
            except Exception:
                pass

    def stats(self) -> str:
        """返回缓存统计摘要字符串。"""
        return (
            f"mode={self.effective_mode}, "
            f"preloaded={self._preloaded_count}, "
            f"hits={self._cache_hits}, "
            f"mem_entries={len(self._mem)}"
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _decide_mode(self, sample_blob: dict, n_steps: int) -> str:
        """根据 sample blob 估算内存需求，决定实际使用的缓存模式。"""
        if self._mode_request != "auto":
            return self._mode_request
        blob_bytes = _estimate_blob_bytes(sample_blob)
        total_needed = blob_bytes * n_steps * self._safety_factor
        available = _get_available_memory_bytes()
        budget = available * self._budget_ratio
        if total_needed <= budget:
            return "memory"
        # 内存不足时尝试磁盘；若 disk_root 未设置则回退 memory
        if self._disk_root is not None:
            return "disk"
        return "memory"

    def _is_cached(self, valid_dt: datetime) -> bool:
        key = self._dt_key(valid_dt)
        if key in self._mem:
            return True
        disk_path = self._disk_path(valid_dt)
        return disk_path is not None and disk_path.exists()

    def _store(self, valid_dt: datetime, blob: dict) -> None:
        mode = self._mode_effective or self._mode_request
        if mode == "disk" and self._disk_root is not None:
            self._store_disk(valid_dt, blob)
        else:
            self._mem[self._dt_key(valid_dt)] = blob

    def _store_disk(self, valid_dt: datetime, blob: dict) -> None:
        disk_path = self._disk_path(valid_dt)
        if disk_path is None:
            self._mem[self._dt_key(valid_dt)] = blob
            return
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = disk_path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(blob, fh, protocol=4)
            os.replace(str(tmp), str(disk_path))
        except Exception as exc:
            # 写磁盘失败时回退内存，避免数据丢失
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            self._mem[self._dt_key(valid_dt)] = blob

    @staticmethod
    def _dt_key(valid_dt: datetime) -> str:
        return valid_dt.strftime("%Y%m%dT%H%M%S")

    def _disk_path(self, valid_dt: datetime) -> Optional[Path]:
        if self._disk_root is None:
            return None
        return self._disk_root / f"{self._dt_key(valid_dt)}.pkl"
