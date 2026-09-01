"""
TTS 结果缓存收益测试 —— 量化缓存对多人场景吞吐的提升。

背景: TTS 是整条链路瓶颈(串行打满仅 6.2 req/min, 而 ASR 有 44.4)。
单卡算力是硬上限, 加并发无效(并发 8 效率仅 13%)。
缓存是绕过算力限制的唯一低成本手段。

方法: 用「不同的重复率」构造负载, 对比吞吐与延迟。
   unique   全部唯一文本   -> 命中率 0%(最坏情况)
   mixed    少量文本反复用 -> 命中率高(固定话术/批量配音场景)
每次跑之前清空缓存, 保证公平。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

import requests

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "concurrency"
BASE = "http://127.0.0.1:8080"

AUDIO = "data/testset/clone_02_M_ref.wav"
REF_TXT = "这名患者曾去过尼日利亚，当地曾出现数宗埃博拉病毒的病例。"

# 模拟「固定话术库」——真实客服/播报/批量配音场景中这类文本高度重复
PHRASES = [
    "今天的天气非常不错，我们一起去公园散步吧。",
    "欢迎使用统一语音处理服务。",
    "您的订单已经提交成功，请耐心等待。",
]


def build_plan(mode: str, n: int) -> list[str]:
    """生成待合成文本序列。"""
    if mode == "unique":
        # 每次都不同, 缓存必然未命中
        return [f"这是第 {i} 条独一无二的测试文本，内容不会重复。" for i in range(n)]
    if mode == "mixed":
        # 固定话术循环使用: 首次冷启动, 其余全部命中
        return [PHRASES[i % len(PHRASES)] for i in range(n)]
    raise ValueError(mode)


def run(mode: str, n: int) -> dict:
    s = requests.Session()
    s.trust_env = False

    # 清空缓存, 保证每次测量都从冷状态开始
    requests.post(f"{BASE}/v1/cache/clear", timeout=30)

    plan = build_plan(mode, n)
    lats, hits, fails = [], 0, 0

    t0 = time.time()
    for text in plan:
        ts = time.time()
        try:
            r = s.post(f"{BASE}/v1/tts",
                       data={"text": text, "reference_audio_path": AUDIO,
                             "reference_text": REF_TXT}, timeout=300)
            lat = time.time() - ts
            if r.status_code == 200:
                j = r.json()
                if j.get("cached"):
                    hits += 1
                lats.append(lat)
            else:
                fails += 1
                lats.append(lat)
        except Exception as e:  # noqa: BLE001
            fails += 1
            lats.append(time.time() - ts)
            print(f"    [FAIL] {type(e).__name__}: {e}"[:120])
    wall = time.time() - t0

    return {
        "mode": mode,
        "n": n,
        "n_failed": fails,
        "cache_hits": hits,
        "hit_rate": round(hits / n, 4) if n else 0.0,
        "wall_clock_s": round(wall, 2),
        "latency_mean_s": round(st.mean(lats), 3) if lats else 0,
        "latency_p95_s": round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 3) if lats else 0,
        "throughput_rpm": round(n / wall * 60, 2) if wall > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=12)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("TTS 缓存收益测试")
    print(f"  每模式 {args.requests} 次请求(测前清空缓存)")
    print("=" * 64)

    results = []
    for mode in ("unique", "mixed"):
        print(f"\n[{mode}] ...")
        r = run(mode, args.requests)
        results.append(r)
        print("  命中率 {:.0%} | 平均 {:.2f}s | P95 {:.2f}s | 吞吐 **{:.1f} req/min**".format(
            r["hit_rate"], r["latency_mean_s"], r["latency_p95_s"], r["throughput_rpm"]))

    u = next(r for r in results if r["mode"] == "unique")
    m = next(r for r in results if r["mode"] == "mixed")

    print("\n" + "=" * 64)
    print("缓存收益")
    print("=" * 64)
    print("| 场景 | 命中率 | 平均延迟 | 吞吐 req/min | 相对提升 |")
    print("|---|---|---|---|---|")
    print("| 全部唯一(最坏) | {:.0%} | {:.2f}s | {:.1f} | 1.00x |".format(
        u["hit_rate"], u["latency_mean_s"], u["throughput_rpm"]))
    print("| 固定话术(典型) | **{:.0%}** | **{:.2f}s** | **{:.1f}** | **{:.2f}x** |".format(
        m["hit_rate"], m["latency_mean_s"], m["throughput_rpm"],
        m["throughput_rpm"] / u["throughput_rpm"] if u["throughput_rpm"] else 0))

    payload = {"test": "cache_benefit", "requests": args.requests, "modes": results,
               "speedup": round(m["throughput_rpm"] / u["throughput_rpm"], 2)
               if u["throughput_rpm"] else None}
    (OUT_DIR / "cache_benefit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 -> {OUT_DIR / 'cache_benefit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
