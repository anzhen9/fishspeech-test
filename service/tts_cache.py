"""
TTS 结果缓存 —— 提升多人场景吞吐的性价比最高的一招。

## 为什么需要

实测(见 PERFORMANCE.md): TTS 是整条链路的绝对瓶颈,
  串行打满吞吐仅 6.2 req/min, 而 ASR 有 44.4 req/min —— 差 7 倍。
根因是 dual-AR 自回归解码 RTF≈1.95, 单卡算力是硬上限, 加并发无效
(并发 8 时效率只有 13%)。

但很多真实负载存在**重复合成**: 固定话术、常用提示音、批量配音、
多人使用同一套预设音色念同一段文案。这类请求命中缓存后完全跳过 GPU,
延迟从 ~9.6s 降到毫秒级。

## 设计

- key = sha256(文本 | 参考音频内容 | 参考文本 | 解码参数)
  参考音频用**内容**而非路径做摘要: 同一个文件被改名/移动不影响命中,
  内容变了则一定不误命中。
- 值 = 已合成的 wav 文件, 落在 CACHE_DIR/<key>.wav
- 淘汰 = LRU(按访问时间), 超过 max_entries 时删最久未用的
- 并发安全 = 用 threading.Lock 保护元数据; 写入采用「临时文件 + 原子替换」,
  避免其它线程读到一个写了一半的 wav

## 一致性

默认 seed=42 且 CUBLAS 确定性, 相同输入应产生相同输出。
但只要解码参数(温度/top_p 等)不同就视为不同 key, 不会串味。
"""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from pathlib import Path

from loguru import logger

from .config import DATA_DIR

CACHE_DIR = DATA_DIR / "tts_cache"
MAX_ENTRIES = int(__import__("os").environ.get("USS_TTS_CACHE_MAX", "500"))
ENABLED = __import__("os").environ.get("USS_TTS_CACHE", "1") not in ("0", "false", "no")

_lock = threading.Lock()
_hits = 0
_misses = 0


def _file_digest(path: Path) -> str:
    """对参考音频内容取 sha256。文件通常 < 2MB, 直接全量读。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_key(text: str,
             reference_audio: Path | None,
             reference_text: str | None,
             **params) -> str:
    """生成缓存 key。任何影响输出的输入都应参与摘要。"""
    parts = [
        text.strip(),
        _file_digest(reference_audio) if reference_audio else "-",
        (reference_text or "").strip(),
    ]
    # 解码参数按名排序后拼接, 保证顺序无关
    for k in sorted(params):
        v = params[k]
        if v is not None:
            parts.append(f"{k}={v}")
    raw = "\x1f".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _touch(path: Path) -> None:
    """更新 mtime, 作为 LRU 的依据。"""
    try:
        now = time.time()
        path.touch()
        # touch() 在某些平台不保证更新, 显式设置一次
        import os
        os.utime(path, (now, now))
    except OSError:
        pass


def get(key: str) -> Path | None:
    """命中返回缓存文件路径, 否则 None。"""
    global _hits
    if not ENABLED:
        return None
    p = CACHE_DIR / f"{key}.wav"
    with _lock:
        if p.is_file() and p.stat().st_size > 0:
            _hits += 1
            _touch(p)
            return p
    return None


def put(key: str, wav_path: Path) -> None:
    """把一次合成结果写入缓存(原子替换)。"""
    global _misses
    if not ENABLED:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{key}.wav"
    tmp = CACHE_DIR / f"{key}.wav.tmp"
    try:
        shutil.copyfile(wav_path, tmp)
        tmp.replace(dest)          # 原子替换, 读者不会看到半截文件
        _misses += 1
        _evict_if_needed()
    except OSError as e:
        logger.warning(f"[Cache] 写入缓存失败: {e}")
        try:
            tmp.unlink()
        except OSError:
            pass


def _evict_if_needed() -> None:
    """按 mtime 做 LRU 淘汰。调用方需持有 _lock。"""
    files = [p for p in CACHE_DIR.glob("*.wav") if p.is_file()]
    if len(files) <= MAX_ENTRIES:
        return
    files.sort(key=lambda p: p.stat().st_mtime)
    for p in files[:len(files) - MAX_ENTRIES]:
        try:
            p.unlink()
        except OSError:
            pass


def stats() -> dict:
    files = list(CACHE_DIR.glob("*.wav")) if CACHE_DIR.is_dir() else []
    total_mb = sum(p.stat().st_size for p in files) / 1024 / 1024
    return {
        "enabled": ENABLED,
        "entries": len(files),
        "max_entries": MAX_ENTRIES,
        "size_mb": round(total_mb, 1),
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / (_hits + _misses), 4) if (_hits + _misses) else 0.0,
    }


def clear() -> int:
    """清空缓存, 返回删除的文件数。"""
    if not CACHE_DIR.is_dir():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.wav"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n
