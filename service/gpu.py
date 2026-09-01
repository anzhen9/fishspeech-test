"""
GPU 显存调度器

RTX 3060 只有 12GB, 且本机桌面程序常驻占用 ~2.7GB。
ASR(large-v3, ~3.2GB) + TTS(FishSpeech 1.5, ~3.6GB) + 说话人编码(~0.7GB)
三者同时常驻会逼近 8GB 上限, 一旦叠加峰值激活就会 OOM。

策略:
  1. 预算制 —— 全局显存预算 GPU_TOTAL_BUDGET_MB, 超预算时按 LRU 把最久未用的模型回迁 CPU
  2. 惰性加载 —— 模型首次被请求时才实例化
  3. 空闲回迁 —— 空闲超过 TTL 自动卸载, 把显存让给其它任务
  4. OOM 兜底 —— 捕获 torch.OutOfMemoryError, 清空缓存后以更低精度/更小模型重试
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from loguru import logger

from .config import GPU_TOTAL_BUDGET_MB, MODEL_IDLE_TTL_S, MODEL_VRAM_MB


def vram_stats() -> dict[str, float]:
    """返回当前显存占用(MB)。"""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "total_mb": round(total / 1024**2, 1),
        "free_mb": round(free / 1024**2, 1),
    }


def empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@dataclass
class ResidentModel:
    name: str
    obj: Any
    vram_mb: int
    last_used: float = field(default_factory=time.time)
    offload_fn: Callable[[Any], None] | None = None
    reload_fn: Callable[[], Any] | None = None


class ModelScheduler:
    """按需加载 + LRU 回迁的模型调度器(线程安全)。"""

    def __init__(self, budget_mb: int = GPU_TOTAL_BUDGET_MB, ttl_s: int = MODEL_IDLE_TTL_S):
        self.budget_mb = budget_mb
        self.ttl_s = ttl_s
        self._models: dict[str, ResidentModel] = {}
        self._builders: dict[str, Callable[[], Any]] = {}
        self._lock = threading.RLock()
        self._events: list[dict] = []

    # -------------------- 注册 --------------------
    def register(self, name: str, builder: Callable[[], Any], vram_mb: int | None = None) -> None:
        """注册一个模型的构建函数; 首次 get 时才真正实例化。"""
        self._builders[name] = builder
        if vram_mb is not None:
            MODEL_VRAM_MB[name] = vram_mb

    # -------------------- 获取 --------------------
    def get(self, name: str) -> Any:
        with self._lock:
            m = self._models.get(name)
            if m is not None:
                m.last_used = time.time()
                return m.obj

            logger.info(f"[GPU] 加载模型 '{name}' ...")
            self._evict_if_needed(name)

            before = vram_stats()
            obj = self._builders[name]()
            after = vram_stats()
            used = max(0.0, after["reserved_mb"] - before["reserved_mb"])

            self._models[name] = ResidentModel(
                name=name,
                obj=obj,
                vram_mb=int(max(used, MODEL_VRAM_MB.get(name, 0))),
            )
            ev = {"event": "load", "model": name, "vram_mb": round(used, 1), "ts": time.time()}
            self._events.append(ev)
            logger.info(f"[GPU] '{name}' 就绪, 占用 {used:.0f}MB | {after}")
            return obj

    # -------------------- 驱逐 --------------------
    def _resident_vram(self) -> int:
        return sum(m.vram_mb for m in self._models.values())

    def _evict_if_needed(self, incoming: str) -> None:
        need = MODEL_VRAM_MB.get(incoming, 1000)
        while self._models and self._resident_vram() + need > self.budget_mb:
            victim = min(self._models.values(), key=lambda m: m.last_used)
            logger.warning(
                f"[GPU] 预算不足({self._resident_vram()}+{need} > {self.budget_mb}MB), "
                f"回迁 '{victim.name}' 至 CPU"
            )
            self._unload(victim.name)

    def _unload(self, name: str) -> None:
        m = self._models.pop(name, None)
        if m is None:
            return
        try:
            if m.offload_fn:
                m.offload_fn(m.obj)
            del m.obj
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[GPU] 卸载 '{name}' 时忽略异常: {e}")
        empty_cache()
        self._events.append({"event": "unload", "model": name, "ts": time.time()})

    def unload(self, name: str) -> None:
        with self._lock:
            self._unload(name)

    def unload_all(self) -> None:
        with self._lock:
            for name in list(self._models):
                self._unload(name)

    # -------------------- TTL 巡检 --------------------
    def sweep_idle(self) -> list[str]:
        """回迁空闲超时的模型, 返回被卸载的模型名。"""
        now = time.time()
        swept = []
        with self._lock:
            for name, m in list(self._models.items()):
                if now - m.last_used > self.ttl_s:
                    self._unload(name)
                    swept.append(name)
        return swept

    # -------------------- 状态 --------------------
    def status(self) -> dict:
        return {
            "budget_mb": self.budget_mb,
            "resident": {
                n: {"vram_mb": m.vram_mb, "idle_s": round(time.time() - m.last_used, 1)}
                for n, m in self._models.items()
            },
            "resident_vram_mb": self._resident_vram(),
            "registered": list(self._builders),
            "vram": vram_stats(),
            "events": self._events[-20:],
        }


SCHEDULER = ModelScheduler()


# ---------------------------------------------------------------- OOM 兜底
def retry_on_oom(fn: Callable, *args, on_oom: Callable | None = None, **kwargs):
    """
    执行 fn, 遇到 CUDA OOM 时:
      1. 清空所有常驻模型
      2. 若有降级回调则调用(如 large-v3 -> medium)
      3. 重试一次
    """
    try:
        return fn(*args, **kwargs)
    except torch.OutOfMemoryError as e:
        logger.error(f"[GPU] CUDA OOM: {e}")
        SCHEDULER.unload_all()
        empty_cache()
        if on_oom is not None:
            logger.warning("[GPU] 触发降级策略后重试")
            on_oom()
        empty_cache()
        return fn(*args, **kwargs)


def pick_precision(device: str, prefer: str = "bfloat16") -> torch.dtype:
    """
    3060 (sm_86) 支持 bf16 与 fp16。
    FishSpeech 官方用 bf16; 数值稳定性优于 fp16, 故默认 bf16。
    """
    if device == "cpu":
        return torch.float32
    if not torch.cuda.is_available():
        return torch.float32
    if prefer == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
