"""
异步任务队列 —— 把不可避免的排队显式化、可观测。

## 为什么需要

实测(见 PERFORMANCE.md): 单卡 3060 的 TTS 吞吐上限只有 6.2 req/min,
混合负载约 18 req/min, 且**不随并发提升**。排队是物理必然。

但同步 HTTP 下, 排队表现为:
  * 客户端要设很长超时(单个 TTS 已 9.6s, 排队后可达数十秒)
  * 用户看不到"前面还有几个", 只能干等, 体验很差
  * 长连接长时间占用, 代理/网关层容易提前切断

异步任务化后:
  * 提交立即返回 job_id, 客户端零超时风险
  * 可轮询「队列位置 + 状态」, 前端能显示进度
  * 结果落盘, 随时可取

## 设计取舍

用 asyncio + 内存表实现, 不引入 celery/arq/redis:
  * 单进程服务, 内存队列足够
  * 无外部依赖, 部署简单
代价: **服务重启后任务与结果丢失**(结果文件仍在磁盘上)。
若需持久化/多实例, 再换 arq + redis。

## 并发控制

执行器内部仍走 service.concurrency 的 run_gpu/run_cpu,
因此「同类模型串行、异类并行、全局额度」的约束依然生效。
这里额外用 JOB_SLOTS 限制**同时执行的任务总数**, 默认与 GPU 额度一致。
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger

JOB_SLOTS = int(os.getenv("USS_JOB_SLOTS", "2"))
# 已完成的任务保留多久后清理(秒), 防止内存无限增长
JOB_RETENTION_S = int(os.getenv("USS_JOB_RETENTION", "3600"))

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    kind: str
    params: dict
    status: str = STATUS_PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.created_at, 2)

    def to_dict(self, queue_pos: int = 0) -> dict:
        """手动构造, 不要用 dataclasses.asdict()。

        asdict() 会对每个字段做 deepcopy, 而 `_task` 是 asyncio.Task
        (不可 pickle), 一调用就会抛 TypeError: cannot pickle '_asyncio.Task'。
        顺带也避免了 deepcopy 体积可能很大的 result。
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "elapsed_s": self.elapsed_s,
            "queue_pos": queue_pos,
        }


class JobQueue:
    """内存任务队列。线程安全的状态表 + asyncio 执行。"""

    def __init__(self, slots: int = JOB_SLOTS):
        self.slots = slots
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []          # 提交顺序, 用于算队列位置
        self._sem = asyncio.Semaphore(slots)
        self._executors: dict[str, Callable[[dict], Awaitable[dict]]] = {}

    # -------------------- 注册执行器 --------------------
    def register(self, kind: str, fn: Callable[[dict], Awaitable[dict]]) -> None:
        self._executors[kind] = fn

    @property
    def kinds(self) -> list[str]:
        return sorted(self._executors)

    # -------------------- 提交 --------------------
    async def submit(self, kind: str, params: dict) -> Job:
        if kind not in self._executors:
            raise KeyError(f"未知任务类型: {kind}, 可选: {self.kinds}")

        job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._sweep()

        job._task = asyncio.create_task(self._run(job))
        logger.info(f"[Job] 提交 {job.kind} job={job.id}")
        return job

    async def _run(self, job: Job) -> None:
        async with self._sem:
            if job.status == STATUS_CANCELLED:
                return
            job.status = STATUS_RUNNING
            job.started_at = time.time()
            logger.info(f"[Job] 开始 {job.kind} job={job.id}")
            try:
                job.result = await self._executors[job.kind](job.params)
                job.status = STATUS_DONE
            except asyncio.CancelledError:
                job.status = STATUS_CANCELLED
                raise
            except Exception as e:  # noqa: BLE001
                job.status = STATUS_FAILED
                job.error = f"{type(e).__name__}: {e}"
                logger.exception(f"[Job] 失败 {job.kind} job={job.id}")
            finally:
                job.finished_at = time.time()
                if job.status == STATUS_DONE:
                    logger.info(f"[Job] 完成 {job.kind} job={job.id} "
                                f"用时 {job.elapsed_s:.1f}s")

    # -------------------- 查询 --------------------
    def _queue_pos(self, job_id: str) -> int:
        """在它之前还有多少个 pending 任务。"""
        pos = 0
        for jid in self._order:
            if jid == job_id:
                break
            if self._jobs.get(jid, Job(job_id, "", {})).status == STATUS_PENDING:
                pos += 1
        return pos

    def get(self, job_id: str) -> dict | None:
        j = self._jobs.get(job_id)
        return j.to_dict(self._queue_pos(job_id)) if j else None

    def list_jobs(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict(self._queue_pos(j.id)) for j in jobs[:limit]]

    # -------------------- 取消 --------------------
    def cancel(self, job_id: str) -> bool:
        j = self._jobs.get(job_id)
        if j is None:
            return False
        if j.status == STATUS_PENDING:
            j.status = STATUS_CANCELLED
            j.finished_at = time.time()
            if j._task:
                j._task.cancel()
            return True
        if j.status == STATUS_RUNNING and j._task:
            j._task.cancel()
            return True
        return False

    # -------------------- 清理 --------------------
    def _sweep(self) -> None:
        """清理过期的已结束任务, 防止内存无限增长。"""
        now = time.time()
        done_ids = [jid for jid, j in self._jobs.items()
                    if j.status in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)
                    and j.finished_at and now - j.finished_at > JOB_RETENTION_S]
        for jid in done_ids:
            self._jobs.pop(jid, None)
            if jid in self._order:
                self._order.remove(jid)

    def stats(self) -> dict:
        by_status: dict[str, int] = {}
        for j in self._jobs.values():
            by_status[j.status] = by_status.get(j.status, 0) + 1
        return {
            "slots": self.slots,
            "total": len(self._jobs),
            "by_status": by_status,
            "kinds": self.kinds,
            "retention_s": JOB_RETENTION_S,
        }


QUEUE = JobQueue()
