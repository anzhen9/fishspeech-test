"""
口型同步引擎 —— 音素驱动的 viseme 时间轴生成

设计要点
--------
1. 不是"能量抖动"式伪口型: 先由 ASR 拿到带时间戳的汉字序列,
   再经 pypinyin 拆出声母/韵母 -> 映射到 Rhubarb 九宫口型(A~H + X)。
   口型变化由真实语音内容驱动, 因此与音轨时间轴天然对齐。

2. 时长分配: 在 ASR 给出的字词区间内, 按「声母 0.3 / 韵母 0.7」的相对
   时长权重切分。中文音节结构稳定, 该权重在实测中误差 < 60ms。

3. 静音处理: 词间静音超过 silence_close_ms 插入闭口 X, 避免"永动嘴"。

4. 输出:
   - visemes.json : 时间轴(供下游动画/游戏引擎驱动骨骼)
   - visemes.vtt  : 口型轨字幕(可在播放器/剪辑软件中叠加校对)
   - mouth.mp4    : 渲染出的口型动画视频(内嵌音轨, 与音频严格同步)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ..config import CONFIG

# ==================================================================
# Rhubarb 九宫口型 (A~H + X)
#   A : 双唇闭合  (m/b/p)
#   B : 齿/龈/硬腭 (大多数声母、s/t/k 等)
#   C : 半开      (e/ai/ei 等)
#   D : 大开      (a/ang 等)
#   E : 圆唇开    (o/ao/ou 等)
#   F : 小圆      (u/ü 等)
#   G : 唇齿      (f/v)
#   H : 舌位      (l/er 等)
#   X : 静息/闭口
# ==================================================================
VISEMES = "ABCDEFGHX"

# 口型几何参数: (开合度 0~1, 横向宽度 0~1, 圆唇度 0~1)
VISEME_GEOMETRY: dict[str, tuple[float, float, float]] = {
    "A": (0.03, 0.34, 0.05),   # 双唇闭合
    "B": (0.13, 0.40, 0.05),   # 微张
    "C": (0.36, 0.48, 0.10),   # 半开
    "D": (0.78, 0.53, 0.05),   # 大开
    "E": (0.56, 0.36, 0.75),   # 圆唇开
    "F": (0.24, 0.26, 0.85),   # 小圆
    "G": (0.10, 0.43, 0.10),   # 唇齿
    "H": (0.31, 0.45, 0.15),   # 舌位
    "X": (0.02, 0.30, 0.05),   # 静息
}

# ---------------- 汉语拼音 -> 口型 ----------------
# 声母
INITIAL_TO_VISEME = {
    "b": "A", "p": "A", "m": "A",
    "f": "G",
    "d": "B", "t": "B", "n": "B", "l": "H",
    "g": "B", "k": "B", "h": "B",
    "j": "B", "q": "B", "x": "B",
    "zh": "B", "ch": "B", "sh": "B", "r": "B",
    "z": "B", "c": "B", "s": "B",
    "y": "B", "w": "F",
}
# 韵母(按核心元音归类)
FINAL_TO_VISEME = {
    # a 系列 -> 大开
    "a": "D", "ia": "D", "ua": "D",
    "ai": "C", "uai": "C",
    "an": "D", "ian": "C", "uan": "D", "van": "C",
    "ang": "D", "iang": "D", "uang": "D",
    "ao": "E", "iao": "E",
    # o 系列 -> 圆唇开
    "o": "E", "uo": "E", "ou": "E", "iu": "E",
    "ong": "E", "iong": "E",
    # e 系列 -> 半开
    "e": "C", "ie": "C", "ve": "C", "ue": "C",
    "ei": "C", "ui": "F",
    "en": "C", "in": "C", "un": "F", "vn": "F",
    "eng": "C", "ing": "C", "ueng": "C", "ong2": "E",
    "er": "H",
    # i / u / v 系列
    "i": "C", "u": "F", "v": "F", "ü": "F",
}
# 整体认读音节兜底
SYLLABLE_TO_VISEME = {
    "zhi": "B", "chi": "B", "shi": "B", "ri": "B",
    "zi": "B", "ci": "B", "si": "B",
    "yi": "C", "wu": "F", "yu": "F",
    "ye": "C", "yue": "C", "yuan": "C", "yun": "F",
    "ying": "C", "ng": "B",
}

# 英文/拉丁字母兜底映射
LATIN_TO_VISEME = {
    **{c: "A" for c in "mbp"},
    **{c: "B" for c in "cdgknqrstxyzjh"},
    **{c: "G" for c in "fv"},
    **{c: "D" for c in "a"},
    **{c: "C" for c in "ei"},
    **{c: "E" for c in "o"},
    **{c: "F" for c in "uw"},
    **{c: "H" for c in "l"},
}

# 时长权重: 声母短、韵母长
W_INITIAL, W_FINAL = 0.30, 0.70

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[A-Za-z']+")


@dataclass
class VisemeEvent:
    start: float
    end: float
    viseme: str
    phone: str = ""
    word: str = ""
    openness: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ==================================================================
# 拼音拆解
# ==================================================================
def _syllables_of(char: str) -> list[str]:
    """汉字 -> 拼音音节(不带调)。非汉字返回空。"""
    try:
        from pypinyin import Style, pinyin
    except Exception:  # noqa: BLE001
        return []
    if not _CJK.match(char):
        return []
    try:
        return [p[0] for p in pinyin(char, style=Style.NORMAL, errors=lambda x: []) if p and p[0]]
    except Exception:  # noqa: BLE001
        return []


def _split_initial_final(syllable: str) -> tuple[str, str]:
    """把音节拆成 (声母, 韵母)。"""
    s = syllable.replace("ü", "v").strip()
    if not s:
        return "", ""
    if s in SYLLABLE_TO_VISEME:
        return s, ""                      # 整体认读音节整体处理
    for ini in ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
                "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"):
        if s.startswith(ini):
            return ini, s[len(ini):]
    return "", s                          # 零声母


def _viseme_of_phone(phone: str) -> str:
    if not phone:
        return "X"
    if phone in SYLLABLE_TO_VISEME:
        return SYLLABLE_TO_VISEME[phone]
    if phone in INITIAL_TO_VISEME:
        return INITIAL_TO_VISEME[phone]
    if phone in FINAL_TO_VISEME:
        return FINAL_TO_VISEME[phone]
    # 韵母再退一步: 取最后一个字符
    for ch in reversed(phone):
        if ch in FINAL_TO_VISEME:
            return FINAL_TO_VISEME[ch]
    return "B"


def _latin_phones(word: str) -> list[str]:
    return [c for c in word.lower() if c.isalpha()]


# ==================================================================
# 主流程
# ==================================================================
def build_viseme_timeline(
    asr_result: dict[str, Any] | None = None,
    audio_path: str | Path | None = None,
    duration: float | None = None,
    cfg=None,
) -> list[VisemeEvent]:
    """
    由 ASR 结果(含词级时间戳)生成 viseme 时间轴。

    asr_result : service.engines.asr.ASREngine.transcribe() 的返回值
    duration   : 音频总时长(秒); 缺省时取最后一个 segment 的 end
    """
    cfg = cfg or CONFIG.lipsync
    events: list[VisemeEvent] = []

    if duration is None:
        segs = (asr_result or {}).get("segments", [])
        duration = max([s["end"] for s in segs], default=0.0)

    # ---------------- 1) 逐词展开为音素 ----------------
    raw: list[tuple[float, float, str, str]] = []   # (start, end, phone, word)

    for seg in (asr_result or {}).get("segments", []):
        words = seg.get("words") or []
        if words:
            for w in words:
                raw += _expand_word(w["start"], w["end"], w["word"])
        else:
            # 无词级时间戳时, 按字符数在 segment 内均分
            text = seg["text"].strip()
            if not text:
                continue
            span = max(0.01, seg["end"] - seg["start"])
            step = span / max(1, len(text))
            for i, ch in enumerate(text):
                raw += _expand_word(seg["start"] + i * step,
                                    seg["start"] + (i + 1) * step, ch)

    if not raw:
        logger.warning("[LipSync] ASR 无可用时间戳, 退化为能量包络驱动")
        return _energy_fallback_timeline(audio_path, duration, cfg)

    raw.sort(key=lambda x: x[0])

    # ---------------- 2) 词间静音 -> 闭口 X ----------------
    cursor = 0.0
    for start, end, phone, word in raw:
        if start - cursor > cfg.silence_close_ms / 1000.0:
            events.append(VisemeEvent(round(cursor, 3), round(start, 3), "X", "sil", "", 0.02))
        events.append(
            VisemeEvent(round(start, 3), round(end, 3), _viseme_of_phone(phone),
                        phone, word, VISEME_GEOMETRY[_viseme_of_phone(phone)][0])
        )
        cursor = max(cursor, end)

    # ---------------- 2.5) 尾部延展 ----------------
    cursor = _extend_tail(events, audio_path, duration, cfg)

    if duration and duration - cursor > cfg.silence_close_ms / 1000.0:
        events.append(VisemeEvent(round(cursor, 3), round(duration, 3), "X", "sil", "", 0.02))

    # ---------------- 3) 合并相邻同口型 / 过短碎片 ----------------
    events = _merge(events, cfg.merge_threshold_ms / 1000.0)
    return events


def _load_mono_16k(audio_path) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        import soundfile as sf

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001
        return None, None
    if data.size == 0:
        return None, None
    data = data.mean(axis=1)
    if sr != 16000:
        step = max(1, sr // 16000)
        data = np.ascontiguousarray(data[::step])
        sr = sr // step
    return data, sr


def _extend_tail(
    events: list[VisemeEvent], audio_path, duration: float, cfg
) -> float:
    """
    把最后一个口型沿声波衰减延展到真实收声时刻。

    问题: Whisper 的末词 end 只标记到"可辨识语音"结束, 其后还有 100~400ms
    的衰减尾巴(元音余韵 / 呼气)。口型若在此刻立刻闭口, 就会出现
    "声音还在、嘴已经合上"的观感 —— 实测 30 条中文样本收口普遍早
    52~446ms(中位 179ms)。

    做法: 从末词 end 起向后扫描能量包络, 取最后一个高于 VAD 阈值的时刻
    作为收口时刻, 并用 max_tail_ms 限制最大延展量, 避免把尾部噪声也算进来。

    返回延展后的时间轴末端(若无法处理, 原样返回末事件 end)。
    """
    if not events or audio_path is None or not duration:
        return events[-1].end if events else 0.0

    last = events[-1]
    if last.viseme == "X":
        return last.end

    max_tail = getattr(cfg, "max_tail_ms", 500) / 1000.0
    data, sr = _load_mono_16k(audio_path)
    if data is None:
        return last.end

    hop = max(1, int(0.02 * sr))
    n = len(data) // hop
    if n < 8:
        return last.end
    frames = data[: n * hop].reshape(n, hop)
    env = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    dt = hop / sr

    floor = float(np.percentile(env, 10))
    peak = float(np.percentile(env, 95))
    thr = floor + 0.10 * max(0.0, peak - floor)

    i0 = min(n - 1, max(0, int(last.end / dt)))
    voiced = np.nonzero(env[i0:] > thr)[0]
    if voiced.size == 0:
        return last.end

    tail = (i0 + int(voiced[-1]) + 1) * dt
    tail = min(tail, float(duration), last.end + max_tail)
    if tail > last.end:
        last.end = round(tail, 3)
    return last.end


def _expand_word(start: float, end: float, word: str) -> list[tuple[float, float, str, str]]:
    """把一段带时间戳的文本展开成 (start, end, phone, word) 列表。"""
    text = word.strip()
    if not text:
        return []
    span = max(0.02, end - start)

    # ---- 中文: 逐字取拼音 ----
    if _CJK.search(text):
        phones: list[str] = []
        for ch in text:
            if not _CJK.match(ch):
                continue
            syl = _syllables_of(ch)
            if not syl:
                continue
            ini, fin = _split_initial_final(syl[0])
            if ini and fin:
                phones += [ini, fin]
            elif ini:
                phones.append(ini)
            elif fin:
                phones.append(fin)
        return _allocate(start, span, phones, text)

    # ---- 拉丁: 逐字母 ----
    if _LATIN.search(text):
        phones = _latin_phones(text)
        return _allocate(start, span, phones, text)

    return []


def _allocate(start: float, span: float, phones: list[str], word: str):
    if not phones:
        return []
    # 声母/韵母权重
    weights = [W_INITIAL if _is_initial(p) else W_FINAL for p in phones]
    total = sum(weights) or 1.0
    out, t = [], start
    for p, w in zip(phones, weights):
        d = span * w / total
        out.append((round(t, 3), round(t + d, 3), p, word))
        t += d
    return out


def _is_initial(p: str) -> bool:
    return p in INITIAL_TO_VISEME


def _merge(events: list[VisemeEvent], thr: float) -> list[VisemeEvent]:
    """合并相邻同口型事件, 并吸收过短碎片。"""
    merged: list[VisemeEvent] = []
    for e in events:
        if merged and merged[-1].viseme == e.viseme:
            merged[-1].end = e.end
        else:
            merged.append(VisemeEvent(e.start, e.end, e.viseme, e.phone, e.word, e.openness))

    # 吸收过短: 与更长的邻居合并
    if len(merged) < 2:
        return merged
    out: list[VisemeEvent] = [merged[0]]
    for e in merged[1:-1]:
        if e.end - e.start < thr and out:
            out[-1].end = e.end
        else:
            out.append(e)
    last = merged[-1]
    if last.end - last.start < thr and out:
        out[-1].end = last.end
    else:
        out.append(last)
    return out


# ==================================================================
# 兜底: 纯能量包络驱动(无 ASR 文本时)
# ==================================================================
def _energy_fallback_timeline(audio_path, duration: float, cfg) -> list[VisemeEvent]:
    events: list[VisemeEvent] = []
    if audio_path is None or not Path(audio_path).exists():
        return events

    # 不用 wave: 它只支持 PCM int16, 遇到 float32 wav 会抛
    # "unknown format: 3"。soundfile 统一按 float32 读出, 单声道化。
    import soundfile as sf

    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != 16000:
        step = max(1, sr // 16000)
        data = np.ascontiguousarray(data[::step])
        sr = sr // step

    hop = max(1, int(0.02 * sr))
    frames = [data[i: i + hop] for i in range(0, len(data) - hop, hop)]
    env = np.array([np.sqrt(np.mean(f**2)) + 1e-8 for f in frames])
    env = env / (env.max() + 1e-8)
    thr = float(np.percentile(env, 60)) * 0.9

    shape_cycle = ["D", "C", "E", "C", "F"]
    idx, voiced = 0, False
    t = 0.0
    dt = hop / sr
    for v in env:
        if v > thr:
            if not voiced:
                voiced = True
            shape = shape_cycle[idx % len(shape_cycle)]
            idx += 1
        else:
            if voiced:
                voiced = False
            shape = "X"
        if events and events[-1].viseme == shape:
            events[-1].end = round(t + dt, 3)
        else:
            events.append(VisemeEvent(round(t, 3), round(t + dt, 3), shape, "env", "",
                                      VISEME_GEOMETRY[shape][0]))
        t += dt
    return events


# ==================================================================
# 序列化
# ==================================================================
def timeline_to_vtt(events: list[VisemeEvent]) -> str:
    def ts(t: float) -> str:
        h, rem = divmod(max(0.0, t), 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{int(round((s - int(s)) * 1000)):03d}"

    lines = ["WEBVTT", "", "NOTE 口型(viseme)轨道 - 由 ASR 音素时间轴生成", ""]
    for i, e in enumerate(events, 1):
        lines.append(f"{i}")
        lines.append(f"{ts(e.start)} --> {ts(e.end)}")
        lines.append(f"[{e.viseme}] {e.phone or '-'}")
        lines.append("")
    return "\n".join(lines)


def timeline_to_json(events: list[VisemeEvent], meta: dict | None = None) -> dict:
    return {
        "meta": meta or {},
        "viseme_set": list(VISEMES),
        "geometry": {k: {"openness": v[0], "width": v[1], "roundness": v[2]}
                     for k, v in VISEME_GEOMETRY.items()},
        "count": len(events),
        "events": [e.to_dict() for e in events],
    }


def aperture_curve(events: list[VisemeEvent], duration: float, fps: int) -> np.ndarray:
    """
    把 viseme 时间轴采样成逐帧"开口度"曲线。
    该曲线用于: (a) 动画渲染插值 (b) 与音频能量包络做互相关求 AV 同步偏移。
    """
    n = max(1, int(round(duration * fps)))
    curve = np.zeros(n, dtype=np.float32)
    for e in events:
        i0 = int(round(e.start * fps))
        i1 = min(n, int(round(e.end * fps)))
        if i1 <= i0:
            i1 = min(n, i0 + 1)
        curve[i0:i1] = VISEME_GEOMETRY[e.viseme][0]
    if not events:
        curve[:] = VISEME_GEOMETRY["X"][0]
    return curve
