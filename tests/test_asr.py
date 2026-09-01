"""
测试 1 —— ASR / 字幕生成 端到端评测

输入 : FLEURS 中文开发集(真实人声, 带真值文本)
指标 : CER(字错误率) / 字准确率 / RTF / 时间戳边界误差 / 静音污染率
输出 : outputs/asr/report.json + report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

from service.config import CONFIG, OUTPUT_DIR  # noqa: E402
from service.engines import asr as asr_mod  # noqa: E402
from service.gpu import vram_stats  # noqa: E402
from tests.metrics import cer, energy_vad, load_wav, timestamp_metrics  # noqa: E402

REPORT_DIR = OUTPUT_DIR / "asr"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def run(limit: int | None = None) -> dict:
    mf = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    items = mf["asr_set"][: limit or len(mf["asr_set"])]

    logger.info(f"[ASR-TEST] 开始, 样本数={len(items)}, 模型={CONFIG.asr.model_size}")

    engine = asr_mod.get_asr()
    results = []
    t_start = time.perf_counter()

    for i, it in enumerate(items, 1):
        wav = it["wav"]
        gt = it["text"]
        try:
            res = engine.transcribe(wav)
            hyp = res["text"]

            # ---- 字准确率 ----
            c = cer(gt, hyp)

            # ---- 时间戳质量 ----
            y, sr = load_wav(wav, 16000)
            vad = energy_vad(y, sr)
            ts = timestamp_metrics(res["segments"], vad, y, sr)

            rec = {
                "file": Path(wav).name,
                "duration": it["duration"],
                "ref": gt,
                "hyp": hyp,
                **c,
                "rtf": res["rtf"],
                "latency_s": res["latency_s"],
                "lang": res["language"],
                "lang_prob": res["language_probability"],
                "n_segments": len(res["segments"]),
                "cer_raw": c.get("cer_raw"),
                "ts_onset_mean_abs_ms": (ts.get("onset_error") or {}).get("mean_abs_ms"),
                "ts_offset_mean_abs_ms": (ts.get("offset_error") or {}).get("mean_abs_ms"),
                "ts_onset_bias_ms": (ts.get("onset_error") or {}).get("signed_ms"),
                "ts_within_100ms": (ts.get("onset_error") or {}).get("within_100ms"),
                "voiced_coverage": ts.get("voiced_coverage"),
                "silence_contamination": ts.get("silence_contamination"),
                "error": None,
            }
            logger.info(
                f"  [{i}/{len(items)}] CER={c['cer']}  acc={c['accuracy']}  "
                f"RTF={res['rtf']}  | {hyp[:28]}"
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"  [{i}/{len(items)}] 失败: {type(e).__name__}: {e}")
            rec = {"file": Path(wav).name, "duration": it["duration"], "ref": gt,
                   "hyp": None, "cer": None, "accuracy": None, "error": f"{type(e).__name__}: {e}"}
        results.append(rec)

    wall = time.perf_counter() - t_start

    # ---------------- 汇总 ----------------
    ok = [r for r in results if r.get("cer") is not None]
    summary = {}
    if ok:
        cers = np.array([r["cer"] for r in ok])
        accs = np.array([r["accuracy"] for r in ok])
        rtfs = np.array([r["rtf"] for r in ok if r.get("rtf")])
        onsets = [r["ts_onset_mean_abs_ms"] for r in ok if r.get("ts_onset_mean_abs_ms") is not None]
        w100 = [r["ts_within_100ms"] for r in ok if r.get("ts_within_100ms") is not None]
        # 带符号偏移: 反映 ASR 起点相对语音活动起点的系统性提前/滞后
        biases = [r["ts_onset_bias_ms"] for r in ok if r.get("ts_onset_bias_ms") is not None]
        covs = [r["voiced_coverage"] for r in ok if r.get("voiced_coverage") is not None]
        raws = [r["cer_raw"] for r in ok if r.get("cer_raw") is not None]
        summary = {
            "n_total": len(results),
            "n_success": len(ok),
            "n_failed": len(results) - len(ok),
            "cer_mean": round(float(cers.mean()), 4),
            "cer_median": round(float(np.median(cers)), 4),
            "cer_std": round(float(cers.std()), 4),
            "char_accuracy_mean": round(float(accs.mean()), 4),
            "cer_raw_mean": round(float(np.mean(raws)), 4) if raws else None,
            "rtf_mean": round(float(rtfs.mean()), 4) if rtfs.size else None,
            "rtf_median": round(float(np.median(rtfs)), 4) if rtfs.size else None,
            "onset_err_mean_ms": round(float(np.mean(onsets)), 1) if onsets else None,
            "onset_bias_mean_ms": round(float(np.mean(biases)), 1) if biases else None,
            "onset_within_100ms": round(float(np.mean(w100)), 3) if w100 else None,
            "voiced_coverage_mean": round(float(np.mean(covs)), 4) if covs else None,
            "total_audio_s": round(sum(r["duration"] for r in ok), 2),
            "wall_clock_s": round(wall, 2),
        }

    report = {
        "test": "ASR / 字幕生成",
        "model": CONFIG.asr.model_size,
        "compute_type": CONFIG.asr.compute_type,
        "language": CONFIG.asr.language,
        "beam_size": CONFIG.asr.beam_size,
        "vad_filter": CONFIG.asr.vad_filter,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "summary": summary,
        "vram": vram_stats(),
        "details": results,
    }
    (REPORT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "report.md").write_text(_to_md(report), encoding="utf-8")
    logger.info(f"[ASR-TEST] 完成 -> {REPORT_DIR}")
    return report


def _to_md(r: dict) -> str:
    s = r["summary"]
    lines = [
        "# ASR / 字幕生成 测试报告",
        "",
        f"- 模型: **{r['model']}** ({r['compute_type']}), beam={r['beam_size']}, VAD={r['vad_filter']}",
        f"- 设备: {r['device']}",
        f"- 样本: {s.get('n_success')}/{s.get('n_total')} 成功",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 字错误率 CER (均值) | {s.get('cer_mean')} |",
        f"| 字错误率 CER (中位数) | {s.get('cer_median')} |",
        f"| **字准确率** | **{s.get('char_accuracy_mean')}** |",
        f"| CER (含拉丁转写, 参考) | {s.get('cer_raw_mean')} |",
        f"| RTF (均值) | {s.get('rtf_mean')} |",
        f"| 起点绝对误差 (均值) | {s.get('onset_err_mean_ms')} ms |",
        f"| 起点系统性偏移 (带符号) | {s.get('onset_bias_mean_ms')} ms |",
        f"| 起点误差 ≤100ms 占比 | {s.get('onset_within_100ms')} |",
        f"| 语音活动覆盖率 | {s.get('voiced_coverage_mean')} |",
        f"| 音频总时长 | {s.get('total_audio_s')} s |",
        f"| 端到端墙钟 | {s.get('wall_clock_s')} s |",
        "",
        "## 逐样本明细",
        "",
        "| 文件 | 时长 | CER | 字准确率 | RTF | 起点偏移ms | 覆盖率 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in r["details"]:
        lines.append(
            f"| {d['file']} | {d.get('duration')} | {d.get('cer')} | {d.get('accuracy')} | "
            f"{d.get('rtf')} | {d.get('ts_onset_bias_ms')} | {d.get('voiced_coverage')} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = run(n)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
