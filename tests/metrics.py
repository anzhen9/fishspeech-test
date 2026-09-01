"""
评估指标实现 (纯 numpy / 标准库, 不依赖评估框架)

  ASR     : CER(字错误率) / 字准确率 / 时间戳边界误差 / 静音污染率
  口型     : AV 同步偏移(互相关) / 边界 F1 / 口型覆盖率 / 切换率
  克隆     : x-vector 余弦相似度 (见 service.engines.speaker)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

# ==================================================================
# 文本归一化 (中文 ASR 评测标准做法)
# ==================================================================
_PUNCT = re.compile(
    r"[\s　,.;:!?\"'()\[\]{}<>《》、。，；：！？“”‘’（）【】…—\-~～·|/\\+*=&#$%@^_`]+"
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_DIGITS = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
           "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def normalize_text(s: str, keep_digits_cn: bool = True, drop_latin: bool = False) -> str:
    """全角转半角 -> 去标点空白 -> 中文数字归一化。

    drop_latin=True 时剔除拉丁字母, 只保留汉字与数字。
    这是中文 CER 评测的关键: FLEURS 真值形如
        "这是马特利 (Martelly) 四年来第五次入选海地临时选举委员会 (CEP)。"
    括号内是原文的拉丁转写, 而 ASR 输出纯中文。若不剔除, 这些与发音
    无关的标注会全部计入错误, 严重高估 CER。
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    out = []
    for ch in s:
        if keep_digits_cn and ch in _DIGITS:
            out.append(_DIGITS[ch])
        elif _CJK.match(ch):
            out.append(ch)
        elif ch.isdigit():
            out.append(ch)
        elif ch.isalpha() and not drop_latin:
            out.append(ch.lower())
        # 标点/空白丢弃
    return "".join(out)


def _edit_distance(a: list, b: list) -> int:
    """Levenshtein 距离(DP, O(n*m))。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> dict:
    """
    字错误率 CER = (S + D + I) / N (字符级编辑距离 / 参考长度)

    同时给出两个口径:
      cer     : 只比汉字与数字(剔除拉丁转写标注) —— 中文场景主指标
      cer_raw : 保留拉丁字母 —— 参考口径, 用于观察中英混读表现
    """
    ref = list(normalize_text(reference, drop_latin=True))
    hyp = list(normalize_text(hypothesis, drop_latin=True))
    n = len(ref)
    if n == 0:
        return {"cer": None, "accuracy": None, "ref_len": 0, "hyp_len": len(hyp),
                "dist": len(hyp), "note": "空参考文本"}
    dist = _edit_distance(ref, hyp)
    cer_v = dist / n

    # 参考口径(含拉丁)
    ref_r = list(normalize_text(reference, drop_latin=False))
    hyp_r = list(normalize_text(hypothesis, drop_latin=False))
    cer_raw = _edit_distance(ref_r, hyp_r) / len(ref_r) if ref_r else None

    return {
        "cer": round(cer_v, 4),
        "accuracy": round(max(0.0, 1.0 - cer_v), 4),   # 字准确率(1-CER, 可为负时截断)
        "ref_len": n,
        "hyp_len": len(hyp),
        "dist": dist,
        "cer_raw": round(cer_raw, 4) if cer_raw is not None else None,
    }


# ==================================================================
# 能量 VAD —— 作为时间轴对齐的参考基准
# ==================================================================
def load_wav(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if target_sr and sr != target_sr:
        # 轻量重采样(线性插值), 避免引入 resampy 依赖
        n = int(round(len(data) * target_sr / sr))
        x_old = np.linspace(0, 1, len(data), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0, 1, n, endpoint=False, dtype=np.float64)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = target_sr
    return data, sr


def energy_envelope(wav: np.ndarray, sr: int, hop_ms: int = 20) -> tuple[np.ndarray, float]:
    hop = max(1, int(sr * hop_ms / 1000))
    n = len(wav) // hop
    frames = wav[: n * hop].reshape(n, hop)
    env = np.sqrt((frames**2).mean(axis=1) + 1e-12)
    return env, hop / sr


def energy_vad(
    wav: np.ndarray, sr: int, hop_ms: int = 20,
    pct: float = 10.0, coef: float = 0.10,
    min_speech_ms: int = 150, min_gap_ms: int = 120,
) -> list[tuple[float, float]]:
    """
    自适应阈值能量 VAD。返回 [(start, end), ...] 秒。

    阈值取靠近噪声底的分位(pct)再加 peak-floor 的 coef 比例。
    pct 必须低(默认 10 分位 ≈ 噪声底), 否则低能量的辅音起始会被判成静音,
    使 VAD 起点系统性晚于真实语音起点, 进而高估 ASR 的边界误差。
    """
    env, dt = energy_envelope(wav, sr, hop_ms)
    if env.size == 0:
        return []
    floor = float(np.percentile(env, pct))
    peak = float(np.percentile(env, 95))
    thr = floor + coef * max(0.0, peak - floor)

    voiced = env > thr
    segs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(voiced)))

    # 合并过近的段 + 丢弃过短段
    merged = []
    min_gap_f = max(1, int(min_gap_ms / 1000 / dt))
    min_len_f = max(1, int(min_speech_ms / 1000 / dt))
    for s, e in segs:
        if merged and s - merged[-1][1] <= min_gap_f:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s * dt, e * dt) for s, e in merged if (e - s) >= min_len_f]


def silero_vad_segments(
    wav: np.ndarray, sr: int = 16000,
    min_silence_ms: int = 250, speech_pad_ms: int = 0,
) -> list[tuple[float, float]]:
    """用 Silero VAD(faster-whisper 自带)求语音段, 作为时间轴对齐的参考真值。

    为什么不用能量 VAD 做口型评测的参考:
      能量阈值只能给出"哪一整段有声音", 对轻声/辅音起始不敏感, 且阈值要手工调;
      Silero 是神经网络 VAD, 边界精度约 ±20ms, 且与 Whisper 的时间戳预测头
      相互独立, 不会因同源而给出虚高的评分。
    """
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:  # noqa: BLE001
        return energy_vad(wav, sr)

    if sr != 16000:
        wav, sr = load_wav_resample(wav, sr, 16000)
    opts = VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
    chunks = get_speech_timestamps(wav.astype(np.float32), opts, sampling_rate=sr)
    return [(round(c["start"] / sr, 3), round(c["end"] / sr, 3)) for c in chunks]


def load_wav_resample(wav: np.ndarray, sr: int, target: int) -> tuple[np.ndarray, int]:
    """把内存中的波形重采样到目标采样率(线性插值, 不引入额外依赖)。"""
    if sr == target:
        return wav, sr
    n = int(round(len(wav) * target / sr))
    x_old = np.linspace(0, 1, len(wav), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0, 1, n, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, wav).astype(np.float32), target


# ==================================================================
# ASR 时间戳质量
# ==================================================================
def timestamp_metrics(asr_segments: list[dict], vad_segments: list[tuple[float, float]],
                      wav: np.ndarray, sr: int) -> dict:
    """
    以能量 VAD 为参考, 评估 ASR 段落的时间轴质量。

    局限说明(重要):
      能量 VAD 判定的是"有声音"而非"有词义", 无法提供逐音素真值 ——
      真正的逐字边界需要 Montreal Forced Aligner 之类的强制对齐工具。
      因此这里只报告三个对下游(口型同步)真正有意义的量:
        1. onset/offset bias : 首尾边界的系统性偏移(带符号, 可据此补偿)
        2. voiced_coverage   : ASR 覆盖了多少真实语音活动帧
        3. silence_contamination : ASR 段内混入了多少静音帧
    """
    if not asr_segments or not vad_segments:
        return {"n_matched": 0, "note": "无有效段落"}

    env, dt = energy_envelope(wav, sr, 20)
    if env.size == 0:
        return {"n_matched": 0, "note": "空音频"}
    floor = float(np.percentile(env, 10))
    peak = float(np.percentile(env, 95))
    thr = floor + 0.10 * max(0.0, peak - floor)

    # ---- 1) 首尾边界系统性偏移 ----
    segs = sorted(asr_segments, key=lambda s: s["start"])
    vad = sorted(vad_segments)
    onset_bias = segs[0]["start"] - vad[0][0]
    offset_bias = segs[-1]["end"] - vad[-1][1]

    # ---- 2) 覆盖率 & 静音污染(逐帧) ----
    voiced = env > thr
    marked = np.zeros(len(env), dtype=bool)
    for seg in segs:
        i0 = max(0, int(seg["start"] / dt))
        i1 = min(len(env), max(i0 + 1, int(seg["end"] / dt)))
        marked[i0:i1] = True

    n_voiced = int(voiced.sum())
    n_marked = int(marked.sum())
    hit = int((marked & voiced).sum())
    coverage = hit / n_voiced if n_voiced else None
    contamination = 1.0 - (hit / n_marked) if n_marked else None

    return {
        "n_asr_segments": len(asr_segments),
        "n_vad_segments": len(vad_segments),
        "n_matched": len(segs),
        "onset_error": {
            "mean_abs_ms": round(abs(onset_bias) * 1000, 1),
            "signed_ms": round(onset_bias * 1000, 1),
            "within_100ms": 1.0 if abs(onset_bias) <= 0.100 else 0.0,
        },
        "offset_error": {
            "mean_abs_ms": round(abs(offset_bias) * 1000, 1),
            "signed_ms": round(offset_bias * 1000, 1),
            "within_100ms": 1.0 if abs(offset_bias) <= 0.100 else 0.0,
        },
        "voiced_coverage": round(coverage, 4) if coverage is not None else None,
        "silence_contamination": round(contamination, 4) if contamination is not None else None,
    }


# ==================================================================
# 口型同步质量
# ==================================================================
def av_sync_offset(aperture: np.ndarray, wav: np.ndarray, sr: int, fps: int) -> dict:
    """
    音画同步偏移(SyncNet 思路的简化版):
      把口型开合曲线与音频 RMS 包络重采样到同一帧率, 做互相关,
      取相关峰对应的滞后量作为 AV 偏移。理想值 = 0ms。
      正值时延表示"口型滞后于音频"。
    """
    env, dt = energy_envelope(wav, sr, 20)
    n = len(aperture)
    t_ap = (np.arange(n) + 0.5) / fps
    t_env = (np.arange(len(env)) + 0.5) * dt
    dur = max(t_ap[-1] if n else 0.0, t_env[-1] if len(env) else 0.0)
    grid = np.arange(0, dur, 1.0 / fps)
    if grid.size < 8:
        return {"offset_ms": None, "peak_corr": None}

    a = np.interp(grid, t_ap, aperture).astype(np.float64)
    b = np.interp(grid, t_env, env).astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)

    max_lag_f = int(round(0.25 * fps))       # 搜索 ±250ms
    lags: list[int] = []
    cors: list[float] = []
    best_lag, best_c = 0, -2.0
    for lag in range(-max_lag_f, max_lag_f + 1):
        if lag >= 0:
            aa, bb = a[lag:], b[: len(b) - lag]
        else:
            aa, bb = a[: len(a) + lag], b[-lag:]
        m = min(len(aa), len(bb))
        if m < 8:
            continue
        c = float(np.dot(aa[:m], bb[:m]) / m)
        lags.append(lag)
        cors.append(c)
        if c > best_c:
            best_c, best_lag = c, lag

    # 抛物线亚帧插值: 25fps 下整数滞后只有 40ms 分辨率, 直接用会把
    # ±20ms 的量化误差当成同步误差。对相关峰做二次插值可细化到 ~5ms。
    sub_lag = float(best_lag)
    if lags and best_lag != lags[0] and best_lag != lags[-1]:
        idx = lags.index(best_lag)
        y0, y1, y2 = cors[idx - 1], cors[idx], cors[idx + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            if abs(delta) <= 1.0:
                sub_lag = best_lag + delta

    return {
        "offset_ms": round(sub_lag / fps * 1000, 1),
        "offset_ms_int": round(best_lag / fps * 1000, 1),
        "peak_corr": round(best_c, 4),
        "max_lag_ms": round(max_lag_f / fps * 1000, 1),
        "note": "offset_ms>0 表示口型滞后于音频; |offset|<=40ms 视为同步良好",
    }


def viseme_boundary_f1(
    events: list, vad_segments: list[tuple[float, float]], tolerance_ms: int = 50
) -> dict:
    """
    以能量 VAD 的"起音/收音"边界为真值, 评估 viseme 序列中
    「开口事件」(X -> 非X) 与「闭口事件」(非X -> X) 的边界判定 F1。
    """
    tol = tolerance_ms / 1000.0
    opens_pred, closes_pred = [], []
    prev = "X"
    for e in events:
        v = e["viseme"] if isinstance(e, dict) else e.viseme
        st = e["start"] if isinstance(e, dict) else e.start
        en = e["end"] if isinstance(e, dict) else e.end
        if prev == "X" and v != "X":
            opens_pred.append(st)
        elif prev != "X" and v == "X":
            closes_pred.append(st)
        prev = v

    truth = []
    for s, e in vad_segments:
        truth.append(("open", s))
        truth.append(("close", e))

    def match(pred, kind):
        gt = [t for k, t in truth if k == kind]
        if not pred or not gt:
            return 0.0, 0.0, 0.0, len(gt), len(pred)
        used = set()
        tp = 0
        for p in pred:
            for i, g in enumerate(gt):
                if i in used:
                    continue
                if abs(p - g) <= tol:
                    used.add(i)
                    tp += 1
                    break
        prec = tp / len(pred)
        rec = tp / len(gt)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return round(prec, 3), round(rec, 3), round(f1, 3), len(gt), len(pred)

    po, ro, fo, ngo, npo = match(opens_pred, "open")
    pc, rc, fc, ngc, npc = match(closes_pred, "close")

    return {
        "tolerance_ms": tolerance_ms,
        "open": {"precision": po, "recall": ro, "f1": fo, "gt": ngo, "pred": npo},
        "close": {"precision": pc, "recall": rc, "f1": fc, "gt": ngc, "pred": npc},
        "macro_f1": round((fo + fc) / 2, 3) if (ngo and ngc) else None,
    }


def viseme_frame_sync(
    aperture: np.ndarray, fps: int,
    vad_segments: list[tuple[float, float]],
    open_thr: float = 0.12,
) -> dict:
    """
    帧级口型-语音一致性 (替代原先的"边界 F1")。

    为什么不按"边界"评测:
      口型序列是音节级的(每秒 5~7 次开合), 而 VAD 给出的是语句级的
      (每 5 秒 1~3 段)。两者粒度差一个数量级, 用 ±50ms 容差去配对
      "开口事件"与"VAD 起音点", 得到的 F1 恒接近 0 —— 那是指标本身不成立,
      不是口型不同步。

    改成帧级判定后, 物理含义清晰且可解释:
      voiced_open_rate    : 有声音的帧里, 嘴是张开的占比(越高越好)
      silence_closed_rate : 无声音的帧里, 嘴是闭合的占比(越高越好)
      二者合起来就是口型与语音在"该动的时候动、该停的时候停"的一致性。
    注意 A(双唇闭合 m/b/p) 与 G(唇齿 f) 的开口度本就接近 0,
    所以 voiced_open_rate 不可能到 1.0, 并非缺陷。
    """
    n = len(aperture)
    pred_open = aperture > open_thr
    gt_voiced = np.zeros(n, dtype=bool)
    for s, e in vad_segments:
        i0 = max(0, int(round(s * fps)))
        i1 = min(n, int(round(e * fps)))
        if i1 > i0:
            gt_voiced[i0:i1] = True

    tp = int((pred_open & gt_voiced).sum())
    fp = int((pred_open & ~gt_voiced).sum())
    fn = int((~pred_open & gt_voiced).sum())
    tn = int((~pred_open & ~gt_voiced).sum())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    n_voiced = int(gt_voiced.sum())

    return {
        "open_thr": open_thr,
        "frames": n,
        "voiced_frames": n_voiced,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "voiced_open_rate": round(rec, 3),
        "silence_closed_rate": round(tn / (tn + fp), 3) if (tn + fp) else None,
    }


def viseme_edge_bias(events: list, vad_segments: list[tuple[float, float]]) -> dict:
    """
    首/尾边界的系统性偏移(带符号)。
    只比首尾: 中间的逐音节边界需要 Montreal Forced Aligner 之类的强制对齐
    才能得到真值, 能量/Silero VAD 都给不出, 因此不做。
    """
    voiced = [e for e in events
              if (e["viseme"] if isinstance(e, dict) else e.viseme) != "X"]
    if not voiced or not vad_segments:
        return {"onset_bias_ms": None, "offset_bias_ms": None}

    s0 = voiced[0]["start"] if isinstance(voiced[0], dict) else voiced[0].start
    e0 = voiced[-1]["end"] if isinstance(voiced[-1], dict) else voiced[-1].end
    return {
        "onset_bias_ms": round((s0 - vad_segments[0][0]) * 1000, 1),
        "offset_bias_ms": round((e0 - vad_segments[-1][1]) * 1000, 1),
        "first_open_s": round(float(s0), 3),
        "last_close_s": round(float(e0), 3),
        "vad_onset_s": round(vad_segments[0][0], 3),
        "vad_offset_s": round(vad_segments[-1][1], 3),
        "note": "onset_bias>0 表示开口晚于发声; <0 表示开口早于发声",
    }


def viseme_stats(events: list, duration: float) -> dict:
    if not events:
        return {"count": 0}
    vs = [e["viseme"] if isinstance(e, dict) else e.viseme for e in events]
    cnt = Counter(vs)
    total = sum(cnt.values())
    voiced = [t for t in vs if t != "X"]
    voiced_time = sum(
        (e["end"] - e["start"]) if isinstance(e, dict) else (e.end - e.start)
        for e, t in zip(events, vs) if t != "X"
    )
    probs = np.array([c / total for c in cnt.values()])
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(9))  # 9 种口型
    return {
        "count": len(events),
        "switches_per_sec": round(len(events) / duration, 2) if duration > 0 else None,
        "voiced_ratio": round(len(voiced) / total, 3),
        "voiced_time_ratio": round(voiced_time / duration, 3) if duration > 0 else None,
        "distribution": dict(cnt.most_common()),
        "normalized_entropy": round(entropy, 3),
    }
