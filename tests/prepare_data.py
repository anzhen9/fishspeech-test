"""
测试数据准备 —— 从 FLEURS 中文(普通话)开发集挑选测试样本

FLEURS dev.tsv 列: id, filename, raw_transcript, tokenized, romanized, num_samples, gender
产出: data/manifest.json
  * asr_set     : 用于字准确率测试(带真值文本)
  * clone_set   : 用于音色克隆测试(按性别挑选不同说话人)
"""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

import soundfile as sf

from service.config import DATA_DIR

TSV = DATA_DIR / "fleurs_zh_dev.tsv"
AUDIO_DIR = DATA_DIR / "fleurs_zh" / "dev"
OUT_DIR = DATA_DIR / "testset"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ASR = 30           # ASR / 口型测试样本数
N_CLONE_SPK = 4      # 克隆测试说话人数

MIN_SEC, MAX_SEC = 3.0, 14.0


def main():
    rows = []
    with TSV.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            rows.append({
                "id": parts[0], "file": parts[1], "text": parts[2],
                "romanized": parts[4], "gender": parts[6],
            })

    # 按文件名去重后的候选, 读取真实时长
    cands = []
    for r in rows:
        src = AUDIO_DIR / r["file"]
        if not src.exists():
            continue
        info = sf.info(str(src))
        dur = info.duration
        if not (MIN_SEC <= dur <= MAX_SEC):
            continue
        cands.append({**r, "duration": round(dur, 3), "sr": info.samplerate,
                      "channels": info.channels, "path": str(src)})

    print(f"候选样本: {len(cands)} (时长 {MIN_SEC}~{MAX_SEC}s)")

    rng = random.Random(42)
    rng.shuffle(cands)

    # ---- ASR / 口型集合 ----
    asr_set = cands[:N_ASR]

    # ---- 克隆集合: 每个性别取 N/2 个, 尽量来自不同句子(说话人不同) ----
    pool = cands[N_ASR:]
    by_gender = {"MALE": [], "FEMALE": []}
    for c in pool:
        by_gender.setdefault(c["gender"], []).append(c)

    clone_set = []
    per = N_CLONE_SPK // 2
    for g in ("MALE", "FEMALE"):
        clone_set += by_gender.get(g, [])[:per]

    # ---- 拷贝到统一目录, 命名规范化 ----
    asr_items, clone_items = [], []
    for i, c in enumerate(asr_set, 1):
        dst = OUT_DIR / f"asr_{i:02d}.wav"
        shutil.copy2(c["path"], dst)
        asr_items.append({
            "wav": str(dst), "text": c["text"], "duration": c["duration"],
            "sr": c["sr"], "gender": c["gender"], "src_id": c["file"],
        })

    for i, c in enumerate(clone_set, 1):
        dst = OUT_DIR / f"clone_{i:02d}_{c['gender'][0]}.wav"
        shutil.copy2(c["path"], dst)
        clone_items.append({
            "wav": str(dst), "text": c["text"], "duration": c["duration"],
            "sr": c["sr"], "gender": c["gender"], "src_id": c["file"],
        })

    manifest = {
        "source": "google/fleurs (cmn_hans_cn, dev split)",
        "n_candidates": len(cands),
        "n_asr": len(asr_items),
        "n_clone": len(clone_items),
        "asr_set": asr_items,
        "clone_set": clone_items,
    }
    mf = DATA_DIR / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n已生成: {mf}")
    print(f"  ASR 集合   : {len(asr_items)} 条 (总时长 {sum(a['duration'] for a in asr_items):.1f}s)")
    print(f"  克隆集合   : {len(clone_items)} 条")
    for c in clone_items:
        print(f"    - {Path(c['wav']).name}  {c['gender']:6s} {c['duration']:5.2f}s  {c['text'][:30]}")
    return manifest


if __name__ == "__main__":
    main()
