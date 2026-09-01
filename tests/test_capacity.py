"""
容量测试 —— 测出 3060 上每类任务的「单机吞吐上限」, 用于容量规划。

与 test_concurrency.py 的区别:
  并发压测看的是「排队下的响应性」(延迟、饿死);
  容量测试看的是「GPU 的能力上限」(串行打满时的 req/min)。
两者结合才能回答「这台 3060 能撑多少人」。

方法: 单线程连续发同一类请求(无排队干扰), 测平均延迟 -> 反推吞吐上限。
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
TTS_TEXT = "今天的天气非常不错，我们一起去公园散步吧。"


def one(kind: str, sess: requests.Session) -> tuple[bool, float, float | None, str]:
    """返回 (成功, 端到端延迟, 服务端自报处理时间, 错误信息)"""
    t0 = time.time()
    try:
        if kind == "asr":
            r = sess.post(f"{BASE}/v1/asr",
                          data={"audio_path": AUDIO, "fmt": "json"}, timeout=300)
        elif kind == "tts":
            r = sess.post(f"{BASE}/v1/tts",
                          data={"text": TTS_TEXT, "reference_audio_path": AUDIO,
                                "reference_text": REF_TXT}, timeout=300)
        elif kind == "lipsync":
            r = sess.post(f"{BASE}/v1/lipsync",
                          data={"audio_path": AUDIO, "render": "false"}, timeout=300)
        else:
            raise ValueError(kind)
        lat = time.time() - t0
        if r.status_code != 200:
            return False, lat, None, r.text[:150]
        return True, lat, r.json().get("latency_s"), ""
    except Exception as e:  # noqa: BLE001
        return False, time.time() - t0, None, f"{type(e).__name__}: {e}"[:150]


def run(kind: str, n: int) -> dict:
    s = requests.Session()
    s.trust_env = False
    # 预热(触发模型加载, 不计入统计)
    one(kind, s)

    lats, srvs, fails = [], [], 0
    t0 = time.time()
    for _ in range(n):
        ok, lat, srv, err = one(kind, s)
        lats.append(lat)
        if srv is not None:
            srvs.append(srv)
        if not ok:
            fails += 1
            print(f"    [FAIL] {err}")
    wall = time.time() - t0

    return {
        "task": kind,
        "n": n,
        "n_failed": fails,
        "wall_clock_s": round(wall, 2),
        "latency_mean_s": round(st.mean(lats), 3),
        "latency_p95_s": round(sorted(lats)[int(len(lats) * 0.95) - 1], 3) if lats else 0,
        "server_proc_mean_s": round(st.mean(srvs), 3) if srvs else None,
        # 串行打满时的吞吐上限
        "throughput_rpm": round(n / wall * 60, 2) if wall > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["asr", "tts", "lipsync"])
    ap.add_argument("--repeat", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 62)
    print("容量测试 —— 串行打满测单机吞吐上限")
    print(f"  每类任务重复 {args.repeat} 次(已剔除首次模型加载)")
    print("=" * 62)

    results = []
    for k in args.tasks:
        print(f"\n[{k}] 串行 {args.repeat} 次 ...")
        r = run(k, args.repeat)
        results.append(r)
        print("  平均 {:.2f}s | P95 {:.2f}s | 服务端处理 {:.2f}s | 吞吐 **{:.1f} req/min**".format(
            r["latency_mean_s"], r["latency_p95_s"],
            r["server_proc_mean_s"] or 0, r["throughput_rpm"]))

    payload = {"test": "capacity", "repeat": args.repeat, "tasks": results}
    (OUT_DIR / "capacity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("容量规划(按单人交互场景估算)")
    print("=" * 62)
    print("| 任务 | 单次耗时 | 吞吐上限 | 10 人排队时最后一人等待 |")
    print("|---|---|---|---|")
    for r in results:
        t = r["latency_mean_s"]
        print("| {} | {:.2f}s | {:.1f} req/min | {:.0f}s |".format(
            r["task"], t, r["throughput_rpm"], t * 10))

    print(f"\n结果 -> {OUT_DIR / 'capacity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
