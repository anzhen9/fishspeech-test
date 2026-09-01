"""
并发控制 —— 让服务在「多人多任务」下不至于卡死。

## 问题

端点是 `async def`, 但内部调用的是**同步阻塞**的推理函数
(`engine.transcribe(...)` / `tts.synthesize(...)`)。在 asyncio 里,
`async def` 内任何阻塞调用都会**独占整个事件循环** —— 于是一个 8 秒的
TTS 请求会让同一时刻到达的所有其它请求(包括 `/health` 探针)全部排队等 8 秒。

实测(见 REPORT.md「并发压测」): 并发 4 时 `/health` 平均延迟 29.9 秒,
是空载基线 2.1 ms 的 **14000 倍**。服务对外表现为完全卡死。

## 方案

1. **卸载到线程池** —— 用 `run_in_threadpool` 执行阻塞推理, 事件循环立刻
   空出来处理其它请求。这一步单独就能解决「轻任务饿死」。
2. **按模型分组串行** —— 同一个模型实例不能被两个线程同时推理(共享 CUDA
   context、KV cache 与中间状态, 并发会产生错误的输出甚至崩溃)。
   不同模型(asr / tts / speaker)权重与显存区独立, 允许并行, 以提升 GPU 利用率。
3. **全局额度闸门** —— 3060 只有 12GB, 同时跑满三套模型会 OOM。
   用一个全局信号量限制「同时在跑的 GPU 任务数」, 默认 2。

## 效果

吞吐不会翻 N 倍(GPU 本身就是串行的), 但:
  * 轻请求(health / status / voices / 下载)延迟回到毫秒级, 不再被饿死
  * 异类任务(如 ASR 与 TTS)可重叠, 端到端总墙钟下降
  * 显存受控, 不会因为并发放大而 OOM
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, TypeVar

from starlette.concurrency import run_in_threadpool

T = TypeVar("T")

# 同类模型的并发额度。1 = 严格串行。
# 这些数字是「安全」而保守的: 提高 tts 会同时占用两份 KV cache, 易 OOM。
GROUP_SLOTS: dict[str, int] = {
    "asr": 1,       # CTranslate2 内部有自己的线程池, 外层再并发没有收益
    "tts": 1,       # 自回归解码有状态, 必须串行
    "speaker": 1,   # 模型小, 但并发收益甚微
    "render": 2,    # CPU/IO 密集(imageio 编码), 可适度并行
    "cpu": 4,       # 纯 CPU 的轻量工作(拼音/时间轴)
}

# 同时在跑的 GPU 任务总数。0 或负 = 不限制。
GPU_TOTAL_SLOTS = int(os.getenv("USS_GPU_SLOTS", "2"))

_groups: dict[str, asyncio.Semaphore] = {}
_total: asyncio.Semaphore | None = None


def _group_sem(kind: str) -> asyncio.Semaphore:
    s = _groups.get(kind)
    if s is None:
        s = asyncio.Semaphore(GROUP_SLOTS.get(kind, 1))
        _groups[kind] = s
    return s


def _total_sem() -> asyncio.Semaphore:
    global _total
    if _total is None:
        _total = asyncio.Semaphore(max(1, GPU_TOTAL_SLOTS))
    return _total


async def run_gpu(kind: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在线程池中执行阻塞的 GPU 推理, 并按模型分组串行化。

    kind: 'asr' | 'tts' | 'speaker'
    """
    # 先抢全局额度, 再抢分组额度; 释放顺序相反。
    # 不会死锁: 分组额度一定在全局额度之内被获取, 且持有者终将释放。
    async with _total_sem():
        async with _group_sem(kind):
            return await run_in_threadpool(fn, *args, **kwargs)


async def run_cpu(kind: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """在线程池中执行阻塞的 CPU 工作(不占 GPU 额度)。

    kind: 'render' | 'cpu'
    """
    async with _group_sem(kind):
        return await run_in_threadpool(fn, *args, **kwargs)


def queue_depth() -> dict[str, int]:
    """返回各分组当前等待中的任务数(用于 /v1/status 观测)。"""
    out = {}
    for k, s in _groups.items():
        # Semaphore 内部 _waiters 是 deque, 长度即排队数
        out[k] = len(getattr(s, "_waiters", None) or ())
    return out
