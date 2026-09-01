"""
测试 3 —— 口型同步 端到端评测

流程:  音频 -> ASR(词级时间戳) -> 拼音音素 -> viseme 时间轴 -> MP4 口型动画
指标:
  1. AV 同步偏移    : 口型开合曲线 vs 音频能量包络的互相关滞后(SyncNet 简化版), 理想 0ms
  2. 边界 F1        : 开口/闭口事件 与 能量 VAD 边界的匹配(容差 ±50ms)
  3. 口型覆盖率     : 语音段内非静息口型占比(越接近 1 越好)
  4. 口型切换率     : 与真实音节速率对照(中文约 4~7 音节/秒)
  5. 时间轴完整性   : 时间轴总长是否覆盖音频全长
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from service.config import CONFIG, OUTPUT_DIR  # noqa: E402
from service.engines import asr as asr_mod  # noqa: E402
from service.engines import renderer  # noqa: E402
from service.engines import viseme as vis  # noqa: E402
from service.gpu import vram_stats  # noqa: E402
from tests.metrics import (  # noqa: E402
    av_sync_offset, load_wav, silero_vad_segments, viseme_boundary_f1,
    viseme_edge_bias, viseme_frame_sync, viseme_stats,
)

REPORT_DIR = OUTPUT_DIR / "lipsync"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RENDER_N = 5     # 渲染 MP4 的样本数(渲染耗时较长, 其余只算时间轴指标)


def run(limit: int | None = None, render_n: int = RENDER_N) -> dict:
    mf = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    items = mf["asr_set"][: limit or len(mf["asr_set"])]

    logger.info(f"[LIP-TEST] 样本数={len(items)}, fps={CONFIG.lipsync.fps}")

    asr_engine = asr_mod.get_asr()
    results = []
    t_start = time.perf_counter()

    for i, it in enumerate(items, 1):
        wav = it["wav"]
        name = Path(wav).stem
        try:
            info = sf.info(wav)
            duration = float(info.duration)

            # ---- 1) ASR 获取词级时间戳 ----
            t0 = time.perf_counter()
            asr_res = asr_engine.transcribe(wav)
            asr_t = time.perf_counter() - t0

            # ---- 2) 生成 viseme 时间轴 ----
            t0 = time.perf_counter()
            events = vis.build_viseme_timeline(asr_res, wav, duration=duration)
            build_t = time.perf_counter() - t0

            ev_dicts = [e.to_dict() for e in events]

            # ---- 3) 指标 ----
            apex = vis.aperture_curve(events, duration, CONFIG.lipsync.fps)
            y, sr = load_wav(wav, 16000)
            # 参考真值: Silero 神经 VAD(与 Whisper 时间戳预测头相互独立)
            vad = silero_vad_segments(y, sr)

            sync = av_sync_offset(apex, y, sr, CONFIG.lipsync.fps)
            frm = viseme_frame_sync(apex, CONFIG.lipsync.fps, vad)
            edge = viseme_edge_bias(ev_dicts, vad)
            st = viseme_stats(ev_dicts, duration)

            # 音节率对照: 中文一字一音节, 用 ASR 字数/时长作为期望切换率
            n_syl = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", asr_res.get("text", "")))
            syl_rate = round(n_syl / duration, 2) if duration > 0 else None

            # 时间轴覆盖
            span = (events[-1].end if events else 0.0)
            coverage = round(min(1.0, span / duration), 4) if duration > 0 else None

            # ---- 4) 落盘 ----
            (REPORT_DIR / f"{name}.visemes.json").write_text(
                json.dumps(vis.timeline_to_json(events, {"audio": wav}), ensure_ascii=False, indent=2),
                encoding="utf-8")
            (REPORT_DIR / f"{name}.visemes.vtt").write_text(
                vis.timeline_to_vtt(events), encoding="utf-8")

            mp4 = None
            if i <= render_n:
                try:
                    mp4 = renderer.render_lipsync_video(
                        events, wav, REPORT_DIR / f"{name}.mp4",
                        duration, CONFIG.lipsync.fps, CONFIG.lipsync.render_size)
                except Exception as e:  # noqa: BLE001
                    logger.error(f"  渲染失败: {type(e).__name__}: {e}")
                    mp4 = None

            rec = {
                "file": Path(wav).name,
                "duration": duration,
                "n_visemes": len(events),
                "av_offset_ms": sync.get("offset_ms"),
                "peak_corr": sync.get("peak_corr"),
                "frame_f1": frm.get("f1"),
                "voiced_open_rate": frm.get("voiced_open_rate"),
                "silence_closed_rate": frm.get("silence_closed_rate"),
                "onset_bias_ms": edge.get("onset_bias_ms"),
                "offset_bias_ms": edge.get("offset_bias_ms"),
                "voiced_time_ratio": st.get("voiced_time_ratio"),
                "switches_per_sec": st.get("switches_per_sec"),
                "syllables_per_sec": syl_rate,
                "normalized_entropy": st.get("normalized_entropy"),
                "timeline_coverage": coverage,
                "asr_latency_s": round(asr_t, 3),
                "build_latency_s": round(build_t, 3),
                "asr_text": asr_res.get("text", "")[:60],
                "mp4": str(mp4) if mp4 else None,
                "error": None,
            }
            logger.info(
                f"  [{i}/{len(items)}] {name}: offset={rec['av_offset_ms']}ms "
                f"onset={rec['onset_bias_ms']}ms frameF1={rec['frame_f1']} "
                f"open={rec['voiced_open_rate']} closed={rec['silence_closed_rate']} "
                f"sw/s={rec['switches_per_sec']}"
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"  [{i}/{len(items)}] {name} 失败")
            rec = {"file": Path(wav).name, "error": f"{type(e).__name__}: {e}"}
        results.append(rec)

    wall = time.perf_counter() - t_start
    ok = [r for r in results if r.get("n_visemes")]

    summary = {}
    if ok:
        def _col(k):
            return [r[k] for r in ok if r.get(k) is not None]

        off = np.array(_col("av_offset_ms"))
        aoff = np.abs(off)
        onb = np.array(_col("onset_bias_ms"))
        summary = {
            "n_total": len(results), "n_success": len(ok), "n_failed": len(results) - len(ok),
            "av_offset_mean_ms": round(float(off.mean()), 1) if off.size else None,
            "av_offset_abs_mean_ms": round(float(aoff.mean()), 1) if aoff.size else None,
            "av_offset_abs_p95_ms": round(float(np.percentile(aoff, 95)), 1) if aoff.size else None,
            "sync_within_40ms_rate": round(float((aoff <= 40).mean()), 3) if aoff.size else None,
            "peak_corr_mean": round(float(np.mean(_col("peak_corr"))), 3) if _col("peak_corr") else None,
            "frame_f1_mean": round(float(np.mean(_col("frame_f1"))), 3) if _col("frame_f1") else None,
            "voiced_open_rate_mean": round(float(np.mean(_col("voiced_open_rate"))), 3) if _col("voiced_open_rate") else None,
            "silence_closed_rate_mean": round(float(np.mean(_col("silence_closed_rate"))), 3) if _col("silence_closed_rate") else None,
            "onset_bias_mean_ms": round(float(onb.mean()), 1) if onb.size else None,
            "onset_bias_abs_mean_ms": round(float(np.abs(onb).mean()), 1) if onb.size else None,
            "onset_within_100ms_rate": round(float((np.abs(onb) <= 100).mean()), 3) if onb.size else None,
            "offset_bias_mean_ms": round(float(np.mean(_col("offset_bias_ms"))), 1) if _col("offset_bias_ms") else None,
            "offset_bias_abs_mean_ms": round(float(np.mean([abs(v) for v in _col("offset_bias_ms")])), 1)
                if _col("offset_bias_ms") else None,
            "voiced_time_ratio_mean": round(float(np.mean(
                [r["voiced_time_ratio"] for r in ok if r.get("voiced_time_ratio") is not None])), 3),
            "switches_per_sec_mean": round(float(np.mean(
                [r["switches_per_sec"] for r in ok if r.get("switches_per_sec") is not None])), 2),
            "syllables_per_sec_mean": round(float(np.mean(_col("syllables_per_sec"))), 2)
                if _col("syllables_per_sec") else None,
            "timeline_coverage_mean": round(float(np.mean(
                [r["timeline_coverage"] for r in ok if r.get("timeline_coverage") is not None])), 3),
            "visemes_per_sample_mean": round(float(np.mean([r["n_visemes"] for r in ok])), 1),
            "wall_clock_s": round(wall, 2),
        }

    report = {
        "test": "口型同步 (音素驱动 viseme 时间轴)",
        "fps": CONFIG.lipsync.fps,
        "viseme_set": CONFIG.lipsync.viseme_set,
        "silence_close_ms": CONFIG.lipsync.silence_close_ms,
        "merge_threshold_ms": CONFIG.lipsync.merge_threshold_ms,
        "asr_model": CONFIG.asr.model_size,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "summary": summary,
        "vram": vram_stats(),
        "details": results,
    }
    (REPORT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "report.md").write_text(_to_md(report), encoding="utf-8")
    logger.info(f"[LIP-TEST] 完成 -> {REPORT_DIR}")
    return report


def _to_md(r: dict) -> str:
    s = r["summary"]
    lines = [
        "# 口型同步 测试报告",
        "",
        f"- 口型集: **{r['viseme_set']}** (Rhubarb 九宫 A~H+X), 帧率 {r['fps']}fps",
        f"- 时间轴来源: ASR 词级时间戳 -> 拼音声母/韵母 -> 口型 ({r['asr_model']})",
        f"- 静音闭口阈值: {r['silence_close_ms']}ms, 合并阈值: {r['merge_threshold_ms']}ms",
        f"- 样本: {s.get('n_success')}/{s.get('n_total')} 成功",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| 开口时刻偏差 (均值/绝对值) | {s.get('onset_bias_mean_ms')} / {s.get('onset_bias_abs_mean_ms')} ms | "
        "负=开口早于发声, 正=晚于发声 |",
        f"| 开口偏差 ≤100ms 占比 | {s.get('onset_within_100ms_rate')} | |",
        f"| AV 互相关偏移 (均值) | {s.get('av_offset_mean_ms')} ms | 开合曲线 vs 音频包络, 理想 0 |",
        f"| AV 偏移 绝对值均值 / P95 | {s.get('av_offset_abs_mean_ms')} / {s.get('av_offset_abs_p95_ms')} ms | |",
        f"| 互相关峰值 | {s.get('peak_corr_mean')} | 口型与音频包络的形状吻合度 |",
        f"| 帧级 F1 | {s.get('frame_f1_mean')} | 「张嘴」帧 vs Silero VAD 语音帧 |",
        f"| 有声帧张嘴率 | {s.get('voiced_open_rate_mean')} | 越高越好(上限非 1, 见下) |",
        f"| 静音帧闭口率 | {s.get('silence_closed_rate_mean')} | 越接近 1 越好 |",
        f"| 口型切换率 / 音节率 | {s.get('switches_per_sec_mean')} / {s.get('syllables_per_sec_mean')} 次/秒 | 两者应同量级 |",
        f"| 时间轴覆盖率 | {s.get('timeline_coverage_mean')} | 应接近 1.0 |",
        f"| 平均口型事件数 | {s.get('visemes_per_sample_mean')} | |",
        "",
        "> 注: 「有声帧张嘴率」的上限不是 1.0 —— Rhubarb 口型集中的 "
        "A(双唇闭合 m/b/p) 与 G(唇齿 f) 本就是闭口姿态, 有声音时嘴也该是合的。",
        "",
        "## 逐样本明细",
        "",
        "| 文件 | 时长 | 口型数 | 开口偏差ms | AV偏移ms | 帧F1 | 张嘴率 | 闭口率 | 切换/秒 | MP4 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in r["details"]:
        lines.append(
            f"| {d['file']} | {d.get('duration')} | {d.get('n_visemes')} | {d.get('onset_bias_ms')} | "
            f"{d.get('av_offset_ms')} | {d.get('frame_f1')} | {d.get('voiced_open_rate')} | "
            f"{d.get('silence_closed_rate')} | {d.get('switches_per_sec')} | "
            f"{'✓' if d.get('mp4') else '-'} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = run(n)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
