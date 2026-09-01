"""
ASR 引擎 —— 语音识别 / 字幕生成 (faster-whisper + CTranslate2)

选型理由:
  * CTranslate2 后端比原版 whisper 快 2~4 倍、显存低约 40%, 是 12GB 卡上的首选
  * 原生支持 int8/float16 量化与 word_timestamps, 后者是口型同步模块的输入
  * 3060 上 large-v3(float16) 实测 RTF ≈ 0.05~0.12

输出: 分段文本 + 词级时间戳 + SRT / VTT / JSON
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import ASRConfig, CONFIG
from ..gpu import SCHEDULER, retry_on_oom

_ASR: "ASREngine | None" = None


@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class ASREngine:
    def __init__(self, cfg: ASRConfig | None = None):
        self.cfg = cfg or CONFIG.asr
        self.model = None
        self.size = self.cfg.model_size
        self._load()

    # ------------------------------------------------------------ 加载
    def _resolve_model_ref(self) -> str:
        """优先使用本地已下载的权重目录。

        HuggingFace 的 xet CDN 在国内不可达, 权重由 curl 经 hf-mirror 预置到
        models/whisper/<repo-id>/ 下; 命中时直接按本地路径加载, 不再走网络。
        """
        local = Path(self.cfg.download_root) / f"faster-whisper-{self.size}"
        if (local / "model.bin").exists():
            logger.info(f"[ASR] 命中本地权重: {local}")
            return str(local)
        return self.size

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        logger.info(f"[ASR] 加载 faster-whisper '{self.size}' ({self.cfg.compute_type}) ...")
        self.model = WhisperModel(
            self._resolve_model_ref(),
            device=self.cfg.device if _cuda_ok() else "cpu",
            compute_type=self.cfg.compute_type if _cuda_ok() else "int8",
            download_root=self.cfg.download_root,
            cpu_threads=self.cfg.cpu_threads,
        )
        logger.info("[ASR] 就绪")

    def _downgrade(self) -> None:
        """OOM 后降级到更小的模型。"""
        if self.size != self.cfg.fallback_size:
            logger.warning(f"[ASR] 降级 {self.size} -> {self.cfg.fallback_size}")
            self.size = self.cfg.fallback_size
            self._load()

    # ------------------------------------------------------------ 转写
    def transcribe(
        self,
        audio: str | Path,
        language: str | None = None,
        word_timestamps: bool | None = None,
        beam_size: int | None = None,
    ) -> dict[str, Any]:
        audio = str(audio)
        t0 = time.perf_counter()

        def _run():
            segments, info = self.model.transcribe(
                audio,
                language=language or self.cfg.language,
                beam_size=beam_size or self.cfg.beam_size,
                vad_filter=self.cfg.vad_filter,
                vad_parameters=self.cfg.vad_parameters,
                word_timestamps=(
                    self.cfg.word_timestamps if word_timestamps is None else word_timestamps
                ),
                condition_on_previous_text=False,  # 避免长音频幻觉累积
            )
            segs: list[Segment] = []
            for i, s in enumerate(segments):
                words = [
                    Word(w.word, round(w.start, 3), round(w.end, 3), round(getattr(w, "probability", 1.0), 3))
                    for w in (s.words or [])
                ]
                segs.append(Segment(i, round(s.start, 3), round(s.end, 3), s.text.strip(), words))
            return segs, info

        segs, info = retry_on_oom(_run, on_oom=self._downgrade)
        elapsed = time.perf_counter() - t0

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        text = "".join(s.text for s in segs)
        return {
            "text": text,
            "language": getattr(info, "language", self.cfg.language or "unknown"),
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
            "duration": round(duration, 3),
            "rtf": round(elapsed / duration, 4) if duration > 0 else None,
            "latency_s": round(elapsed, 3),
            "model_size": self.size,
            "segments": [s.to_dict() for s in segs],
        }

    # ------------------------------------------------------------ 字幕序列化
    @staticmethod
    def to_srt(segments: list[dict]) -> str:
        def ts(t: float) -> str:
            h, rem = divmod(max(0.0, t), 3600)
            m, s = divmod(rem, 60)
            ms = int(round((s - int(s)) * 1000))
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

        out = []
        for i, s in enumerate(segments, 1):
            out.append(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n")
        return "\n".join(out)

    @staticmethod
    def to_vtt(segments: list[dict]) -> str:
        def ts(t: float) -> str:
            h, rem = divmod(max(0.0, t), 3600)
            m, s = divmod(rem, 60)
            ms = int(round((s - int(s)) * 1000))
            return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

        out = ["WEBVTT", ""]
        for s in segments:
            out.append(f"{ts(s['start'])} --> {ts(s['end'])}")
            out.append(s["text"])
            out.append("")
        return "\n".join(out)


def _cuda_ok() -> bool:
    import torch

    return torch.cuda.is_available()


def get_asr() -> ASREngine:
    """经调度器获取(惰性加载 + 自动回迁)的 ASR 引擎单例。"""
    global _ASR
    if _ASR is None:
        SCHEDULER.register("asr", lambda: ASREngine())
        _ASR = SCHEDULER.get("asr")
    return _ASR
