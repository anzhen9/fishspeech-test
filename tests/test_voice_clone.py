"""
测试 2 —— 语音克隆 (FishSpeech 1.5) 端到端评测

流程:  参考音频(真实人声 ~10s)
        -> Firefly VQGAN 编码得到 prompt_tokens (音色指纹)
        -> 与 prompt_text 一起作为 Dual-AR LLaMA 前缀 (in-context learning)
        -> LLaMA 自回归生成语义 token -> VQGAN 解码为 44.1kHz 波形

数据集构造说明
--------------
FLEURS cmn_hans_cn dev 的 TSV 不含 speaker_id, 因此先用 ECAPA-TDNN 对全部
222 条候选做贪心聚类(余弦阈值 0.55), 得到 5 个说话人簇
(簇内均值 0.790 / 簇间均值 0.200, 交叉率 < 1%), 再从中挑 4 个说话人,
每人 2 段: ref(克隆输入) + held(同人留出真值, 内容与 ref 不同)。

指标
----
  1. 音色保真度  : cos(合成, ref)
  2. 同人验证    : cos(合成, held) —— 比 ref 更严格, 排除"复刻同一段录音"
  3. 区分度      : 4x4 注册矩阵 (行=合成样本, 列=各说话人 held 真值),
                   统计 Top-1 命中率与同人/他人差值
  4. 内容可懂度  : 对合成音频再跑 ASR, 与目标文本算 CER
  5. 效率        : RTF / 延迟 / 显存峰值
"""

from __future__ import annotations

import json
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
from service.engines import clone as clone_mod  # noqa: E402
from service.engines import speaker as spk_mod  # noqa: E402
from service.gpu import vram_stats  # noqa: E402
from tests.metrics import cer  # noqa: E402

REPORT_DIR = OUTPUT_DIR / "clone"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 所有说话人合成同一句, 便于横向比较音色差异
TARGET_TEXT = "今天的天气非常不错，我们一起去公园散步吧。"


def _load_items() -> tuple[list[dict], list[dict]]:
    mf = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    all_items = mf["clone_set"]
    refs = [x for x in all_items if x.get("role") == "ref"]
    helds = [x for x in all_items if x.get("role") == "held"]
    # 若 manifest 是旧格式(无 role 字段), 退化为一半 ref 一半 held
    if not refs:
        refs, helds = all_items[: len(all_items) // 2], all_items[len(all_items) // 2:]
    return refs, helds


def run(limit: int | None = None) -> dict:
    refs, helds = _load_items()
    refs = refs[: limit or len(refs)]
    held_by_spk = {h["speaker_id"]: h for h in helds}

    logger.info(f"[CLONE-TEST] 说话人数={len(refs)}, 目标文本='{TARGET_TEXT}'")

    tts = clone_mod.get_tts()
    enc = spk_mod.get_encoder()
    logger.info(f"[CLONE-TEST] 说话人编码器后端: {enc.backend}")

    results = []
    enr_emb = {}  # speaker_id -> 注册(held)embedding
    for h in helds:
        enr_emb[h["speaker_id"]] = enc.embed(h["wav"])

    t_start = time.perf_counter()

    for i, it in enumerate(refs, 1):
        ref = it["wav"]
        name = Path(ref).stem
        spk = it.get("speaker_id", f"spk{i}")
        try:
            t0 = time.perf_counter()
            gen = tts.synthesize(
                text=TARGET_TEXT,
                reference_audio=ref,
                reference_text=it.get("text") or None,
            )
            out_wav = REPORT_DIR / f"{name}_cloned.wav"
            sf.write(str(out_wav), gen["audio"], gen["sample_rate"])
            gen_emb = enc.embed(str(out_wav))
            wall = time.perf_counter() - t0

            sim_ref = float(enc.cosine(enc.embed(ref), gen_emb))
            sim_held = None
            if spk in enr_emb:
                sim_held = float(enc.cosine(enr_emb[spk], gen_emb))

            # 4x4 注册矩阵行: 该合成样本对所有说话人 held 真值的余弦
            row = {s: round(float(enc.cosine(e, gen_emb)), 4) for s, e in enr_emb.items()}
            top1 = max(row, key=row.get)

            # 内容可懂度: 对合成音频再 ASR
            try:
                asr_res = asr_mod.get_asr().transcribe(str(out_wav))
                c = cer(TARGET_TEXT, asr_res["text"])
            except Exception as e:  # noqa: BLE001
                logger.warning(f"  内容校验 ASR 失败: {e}")
                c, asr_res = {"cer": None, "accuracy": None}, {"text": None}

            others = [v for s, v in row.items() if s != spk]
            rec = {
                "speaker": spk,
                "file": name,
                "gender": it.get("gender"),
                "ref_duration": it.get("duration"),
                "gen_duration": gen["duration"],
                "rtf": gen["rtf"],
                "latency_s": gen["latency_s"],
                "wall_s": round(wall, 3),
                "sample_rate": gen["sample_rate"],
                "cos_ref": round(sim_ref, 4),
                "cos_held": round(sim_held, 4) if sim_held is not None else None,
                "top1_enroll": top1,
                "top1_hit": bool(top1 == spk),
                "cos_other_mean": round(float(np.mean(others)), 4) if others else None,
                "margin_vs_other": (
                    round(float(sim_held - np.mean(others)), 4)
                    if (sim_held is not None and others) else None),
                "gen_cer": c.get("cer"),
                "gen_accuracy": c.get("accuracy"),
                "asr_on_gen": asr_res.get("text"),
                "enroll_row": row,
                "out_wav": str(out_wav),
                "error": None,
            }
            logger.info(
                f"  [{i}/{len(refs)}] {name}: cos_ref={sim_ref:.3f} cos_held={sim_held} "
                f"top1={top1}({'OK' if top1 == spk else 'MISS'}) CER={c.get('cer')} RTF={gen['rtf']}"
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"  [{i}/{len(refs)}] {name} 克隆失败")
            rec = {"speaker": spk, "file": name, "gender": it.get("gender"), "error": f"{type(e).__name__}: {e}"}
        results.append(rec)

    wall_total = time.perf_counter() - t_start

    ok = [r for r in results if r.get("cos_ref") is not None]
    summary: dict = {}
    if ok:
        cr = np.array([r["cos_ref"] for r in ok])
        ch = np.array([r["cos_held"] for r in ok if r.get("cos_held") is not None])
        co = np.array([r["cos_other_mean"] for r in ok if r.get("cos_other_mean") is not None])
        rtfs = np.array([r["rtf"] for r in ok if r.get("rtf")])
        gcer = [r["gen_cer"] for r in ok if r.get("gen_cer") is not None]
        summary = {
            "n_total": len(results), "n_success": len(ok), "n_failed": len(results) - len(ok),
            "cos_ref_mean": round(float(cr.mean()), 4),
            "cos_ref_min": round(float(cr.min()), 4),
            "cos_held_mean": round(float(ch.mean()), 4) if ch.size else None,
            "cos_other_mean": round(float(co.mean()), 4) if co.size else None,
            "margin_vs_other_mean": (
                round(float(ch.mean() - co.mean()), 4) if (ch.size and co.size) else None),
            "same_speaker_rate_ref": round(float(np.mean([1.0 if r["cos_ref"] >= 0.55 else 0.0 for r in ok])), 3),
            "same_speaker_rate_held": round(float(np.mean([1.0 if (r.get("cos_held") or 0) >= 0.55 else 0.0 for r in ok])), 3),
            "top1_enroll_acc": round(float(np.mean([1.0 if r.get("top1_hit") else 0.0 for r in ok])), 3),
            "gen_cer_mean": round(float(np.mean(gcer)), 4) if gcer else None,
            "gen_accuracy_mean": round(float(1 - np.mean(gcer)), 4) if gcer else None,
            "rtf_mean": round(float(rtfs.mean()), 3) if rtfs.size else None,
            "gen_duration_mean_s": round(float(np.mean([r["gen_duration"] for r in ok])), 2),
            "wall_clock_s": round(wall_total, 2),
            "encoder_backend": enc.backend,
        }

    report = {
        "test": "语音克隆 (FishSpeech 1.5)",
        "target_text": TARGET_TEXT,
        "precision": str(tts.precision),
        "top_p": CONFIG.tts.top_p,
        "temperature": CONFIG.tts.temperature,
        "repetition_penalty": CONFIG.tts.repetition_penalty,
        "max_new_tokens": CONFIG.tts.max_new_tokens,
        "chunk_length": CONFIG.tts.chunk_length,
        "seed": CONFIG.tts.seed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "summary": summary,
        "vram": vram_stats(),
        "details": results,
    }
    (REPORT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT_DIR / "report.md").write_text(_to_md(report), encoding="utf-8")
    logger.info(f"[CLONE-TEST] 完成 -> {REPORT_DIR}")
    return report


def _to_md(r: dict) -> str:
    s = r["summary"]
    lines = [
        "# 语音克隆 (FishSpeech 1.5) 测试报告",
        "",
        f"- 精度: **{r['precision']}**, top_p={r['top_p']}, temperature={r['temperature']}, "
        f"repetition_penalty={r['repetition_penalty']}, max_new_tokens={r['max_new_tokens']}, "
        f"chunk_length={r['chunk_length']}, seed={r['seed']}",
        f"- 设备: {r['device']}",
        f"- 目标文本: {r['target_text']}",
        f"- 样本: {s.get('n_success')}/{s.get('n_total')} 成功",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| cos(合成, 参考音频) | {s.get('cos_ref_mean')} | 音色保真度, ECAPA 同人阈值 0.55 |",
        f"| cos(合成, 同人留出真值) | {s.get('cos_held_mean')} | 更严格: 排除复刻同一段录音 |",
        f"| cos(合成, 他人真值) | {s.get('cos_other_mean')} | 应显著低于同人值 |",
        f"| 区分度差值 | {s.get('margin_vs_other_mean')} | 同人留出 - 他人, 越大越好 |",
        f"| 同人判定率(ref / held) | {s.get('same_speaker_rate_ref')} / {s.get('same_speaker_rate_held')} | cos ≥ 0.55 |",
        f"| 说话人验证 Top-1 命中率 | {s.get('top1_enroll_acc')} | 4 选 1 注册判定 |",
        f"| 合成内容 CER / 字准确率 | {s.get('gen_cer_mean')} / {s.get('gen_accuracy_mean')} | 对合成音频再 ASR |",
        f"| RTF (均值) | {s.get('rtf_mean')} | 实时率, <1 即快于实时 |",
        f"| 说话人编码器 | {s.get('encoder_backend')} | |",
        "",
        "## 逐说话人明细",
        "",
        "| 说话人 | 参考音频 | 性别 | 参考时长 | 生成时长 | cos(ref) | cos(held) | cos(他人) | Top-1 | 内容CER | RTF |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in r["details"]:
        lines.append(
            f"| {d.get('speaker')} | {d.get('file')} | {d.get('gender')} | {d.get('ref_duration')} | "
            f"{d.get('gen_duration')} | {d.get('cos_ref')} | {d.get('cos_held')} | "
            f"{d.get('cos_other_mean')} | {'OK' if d.get('top1_hit') else 'MISS'} | "
            f"{d.get('gen_cer')} | {d.get('rtf')} |"
        )

    # 注册矩阵
    rows = [d for d in r["details"] if d.get("enroll_row")]
    if rows:
        cols = list(rows[0]["enroll_row"].keys())
        lines += ["", "## 说话人注册矩阵", "",
                  "行 = 由该说话人克隆合成的音频; 列 = 各说话人的留出真值音频(注册库)。对角应最大。",
                  "", "| 合成来源 \\ 注册库 | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for d in rows:
            cells = []
            for c in cols:
                v = d["enroll_row"].get(c)
                mark = "**" if d.get("speaker") == c else ""
                cells.append(f"{mark}{v}{mark}")
            lines.append(f"| {d.get('speaker')} ({d.get('file')}) | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rep = run(n)
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=2))
