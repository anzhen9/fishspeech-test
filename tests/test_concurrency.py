"""
并发 / 多任务压力测试 —— 评估服务在「多人多任务」场景下的承载能力。

设计要点:
  1. 虚拟用户: N 个线程各自独立发请求, 模拟 N 个并发用户
  2. 混合负载: 按权重随机选择任务类型(贴近真实使用比例)
       asr     语音转字幕   —— 轻-中
       tts     克隆合成     —— 重(GPU 自回归解码)
       lipsync 口型同步     —— 重(ASR + 时间轴 + 渲染)
       light   健康检查     —— 极轻(用于观察是否被重任务饿死)
  3. 采集指标:
       - 吞吐 req/min
       - 延迟 P50 / P95 / P99(端到端, 含排队)
       - 排队时间 = 端到端延迟 - 纯处理时间
       - 成功率与错误分布
       - 显存峰值与采样序列
       - 「轻任务饿死比」: 并发下 light 的延迟 / 空载时 light 的延迟
  4. 关键判据:
       - 并发加速比 = N 并发吞吐 / 单并发吞吐
       - 若 ≈ 1.0, 说明请求被串行化(事件循环阻塞或全局锁)
       - 若 light 饿死比很大, 说明轻请求被重请求阻塞, 服务对外表现为「卡死」

用法:
  python -m tests.test_concurrency                      # 默认全套
  python -m tests.test_concurrency --quick              # 只跑 N=1,4
  python -m tests.test_concurrency --concurrency 2 4 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

# 压测走 loopback, 必须绕开系统代理。
# 否则 requests 会把 127.0.0.1:8080 也塞进代理隧道, 表现为随机的
# ProxyError / ReadTimeout, 污染成功率统计(实测曾让成功率虚低到 84%)。
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "concurrency"
BASE = "http://127.0.0.1:8080"

# 测试音频(每人一份, 模拟不同用户上传不同内容)
# 只用 data/testset 下确实存在的文件, 否则服务端会 404 污染成功率统计。
AUDIO_FILES = [
    "data/testset/clone_01_F_ref.wav",
    "data/testset/clone_02_M_ref.wav",
    "data/testset/clone_03_M_ref.wav",
    "data/testset/clone_04_M_ref.wav",
    "data/testset/clone_01_M.wav",
    "data/testset/clone_02_M.wav",
    "data/testset/clone_03_F.wav",
    "data/testset/clone_04_F.wav",
]
AUDIO = {f"u{i + 1}": p for i, p in enumerate(AUDIO_FILES)}
REF_TXT = "虚拟实地考察是高科技的解决方案，学生可以在课堂上观看博物馆的人工制品。"
TTS_TEXT = "今天的天气非常不错，我们一起去公园散步吧。"

# 任务类型 -> 权重(贴近真实: 字幕最多, 合成次之, 口型较少, 探针穿插)
TASK_WEIGHTS = {"asr": 40, "tts": 30, "lipsync": 15, "light": 15}


# ==================================================================
# 单次请求
# ==================================================================
def do_task(kind: str, audio: str, sess: requests.Session) -> dict:
    """执行单个任务, 返回耗时/状态记录。"""
    t0 = time.time()
    rec = {"task": kind, "t_start": t0, "ok": False, "status": None, "error": None,
           "server_latency_s": None}
    try:
        if kind == "asr":
            r = sess.post(f"{BASE}/v1/asr",
                          data={"audio_path": audio, "fmt": "json"}, timeout=300)
        elif kind == "tts":
            r = sess.post(f"{BASE}/v1/tts",
                          data={"text": TTS_TEXT, "reference_audio_path": audio,
                                "reference_text": REF_TXT}, timeout=300)
        elif kind == "lipsync":
            r = sess.post(f"{BASE}/v1/lipsync",
                          data={"audio_path": audio, "render": "false"}, timeout=300)
        else:  # light
            r = sess.get(f"{BASE}/health", timeout=30)

        rec["status"] = r.status_code
        rec["ok"] = r.status_code == 200
        if rec["ok"] and kind != "light":
            try:
                rec["server_latency_s"] = r.json().get("latency_s")
            except Exception:  # noqa: BLE001
                pass
        if not rec["ok"]:
            rec["error"] = (r.text or "")[:200]
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    rec["latency_s"] = time.time() - t0
    return rec


def vram_sample() -> dict | None:
    try:
        return requests.get(f"{BASE}/v1/status", timeout=10).json().get("vram")
    except Exception:  # noqa: BLE001
        return None


# ==================================================================
# 采样显存的后台线程
# ==================================================================
class VRAMMonitor:
    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            v = vram_sample()
            if v:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def stats(self) -> dict:
        if not self.samples:
            return {}
        alloc = [s["allocated_mb"] for s in self.samples]
        reserved = [s["reserved_mb"] for s in self.samples]
        free = [s["free_mb"] for s in self.samples]
        return {
            "n_samples": len(self.samples),
            "allocated_peak_mb": round(max(alloc), 1),
            "allocated_mean_mb": round(st.mean(alloc), 1),
            "reserved_peak_mb": round(max(reserved), 1),
            "free_min_mb": round(min(free), 1),
            # 抖动 = 已分配显存的标准差, 反映模型是否反复加载/卸载(颠簸)
            "allocated_std_mb": round(st.pstdev(alloc), 1),
        }


# ==================================================================
# 一轮压测
# ==================================================================
def run_level(concurrency: int, n_requests: int, seed: int = 42) -> dict:
    rnd = random.Random(seed)
    kinds = list(TASK_WEIGHTS)
    weights = [TASK_WEIGHTS[k] for k in kinds]

    # 预先生成任务序列, 保证各并发度下的负载构成一致(可比)
    plan = []
    for i in range(n_requests):
        k = rnd.choices(kinds, weights=weights, k=1)[0]
        aud = AUDIO[f"u{(i % 8) + 1}"]
        plan.append((k, aud))

    results: list[dict] = []
    lock = threading.Lock()
    local = threading.local()

    def sess() -> requests.Session:
        s = getattr(local, "s", None)
        if s is None:
            s = requests.Session()
            s.trust_env = False      # 显式绕开代理, 见文件顶部说明
            local.s = s
        return s

    def worker(idx: int):
        for i in range(idx, len(plan), concurrency):
            k, aud = plan[i]
            rec = do_task(k, aud, sess())
            rec["worker"] = idx
            with lock:
                results.append(rec)

    with VRAMMonitor() as mon:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(worker, range(concurrency)))
        wall = time.time() - t0

    return {"concurrency": concurrency, "n_requests": n_requests, "wall_clock_s": wall,
            "records": results, "vram": mon.stats()}


# ==================================================================
# 统计
# ==================================================================
def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(level: dict, light_baseline_s: float) -> dict:
    recs = level["records"]
    wall = level["wall_clock_s"]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_task[r["task"]].append(r)

    ok = [r for r in recs if r["ok"]]
    errs = Counter((r["task"], (r["error"] or f"HTTP{r['status']}")[:60])
                   for r in recs if not r["ok"])

    lat = [r["latency_s"] for r in recs]
    # 排队时间必须「逐条配对」做差: 端到端延迟 - 该请求服务端自报的处理时间。
    # 若用两组均值相减, 会因为 light 任务没有 server_latency 而混入无关样本,
    # 甚至算出负值。
    pairs = [(r["latency_s"], r["server_latency_s"]) for r in recs
             if r.get("server_latency_s") is not None]
    queues = [e - s for e, s in pairs]
    srv = [s for _, s in pairs]

    out = {
        "concurrency": level["concurrency"],
        "n_requests": level["n_requests"],
        "n_success": len(ok),
        "n_failed": len(recs) - len(ok),
        "success_rate": round(len(ok) / len(recs), 4) if recs else 0.0,
        "wall_clock_s": round(wall, 2),
        "throughput_rpm": round(len(recs) / wall * 60, 2) if wall > 0 else 0.0,
        "latency_p50_s": round(pct(lat, 0.50), 3),
        "latency_p95_s": round(pct(lat, 0.95), 3),
        "latency_p99_s": round(pct(lat, 0.99), 3),
        "latency_mean_s": round(st.mean(lat), 3) if lat else 0.0,
        "latency_max_s": round(max(lat), 3) if lat else 0.0,
        "server_proc_mean_s": round(st.mean(srv), 3) if srv else None,
        # 排队时间 = 端到端 - 服务端处理(逐条配对后取均值)
        "queue_mean_s": round(st.mean(queues), 3) if queues else None,
        "queue_p95_s": round(pct(queues, 0.95), 3) if queues else None,
        "queue_max_s": round(max(queues), 3) if queues else None,
        "vram": level["vram"],
        "errors": [{"task": k[0], "error": k[1], "count": v} for k, v in errs.most_common(10)],
    }

    # 分任务类型统计
    out["by_task"] = {}
    for k, rs in sorted(by_task.items()):
        l = [r["latency_s"] for r in rs]
        out["by_task"][k] = {
            "n": len(rs),
            "success_rate": round(sum(1 for r in rs if r["ok"]) / len(rs), 4),
            "p50_s": round(pct(l, 0.50), 3),
            "p95_s": round(pct(l, 0.95), 3),
            "mean_s": round(st.mean(l), 3),
            "max_s": round(max(l), 3),
        }

    # 轻任务饿死比: 并发下 light 的平均延迟 / 空载基线
    if "light" in by_task:
        lm = st.mean([r["latency_s"] for r in by_task["light"]])
        out["light_mean_s"] = round(lm, 3)
        out["light_starvation_ratio"] = (round(lm / light_baseline_s, 2)
                                         if light_baseline_s > 0 else None)
    return out


def measure_light_baseline(n: int = 20) -> float:
    """空载时 health 的平均延迟, 作为轻任务延迟基线。"""
    s = requests.Session()
    s.trust_env = False
    lat = []
    for _ in range(n):
        t0 = time.time()
        try:
            s.get(f"{BASE}/health", timeout=10)
        except Exception:  # noqa: BLE001
            pass
        lat.append(time.time() - t0)
    return st.mean(lat)


# ==================================================================
# 报告
# ==================================================================
def to_md(levels: list[dict], light_base: float) -> str:
    L = []
    L.append("# 并发 / 多任务压力测试报告\n")
    L.append(f"- 服务: `{BASE}`")
    L.append(f"- 负载构成: " + ", ".join(f"{k}={v}%" for k, v in TASK_WEIGHTS.items()))
    L.append(f"- 空载 health 基线延迟: **{light_base * 1000:.1f} ms**\n")

    L.append("## 总览\n")
    L.append("| 并发 | 请求数 | 墙钟s | 吞吐 req/min | 成功 | P50 s | P95 s | P99 s | 排队均值s | 排队P95 s | 显存峰值MB |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    base_tp = None
    for s in levels:
        c = s["concurrency"]
        if c == 1:
            base_tp = s["throughput_rpm"]
        L.append("| {} | {} | {:.1f} | {:.2f} | {:.0%} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} |".format(
            c, s["n_requests"], s["wall_clock_s"], s["throughput_rpm"],
            s["success_rate"], s["latency_p50_s"], s["latency_p95_s"],
            s["latency_p99_s"],
            s["queue_mean_s"] if s["queue_mean_s"] is not None else "-",
            s["queue_p95_s"] if s["queue_p95_s"] is not None else "-",
            s["vram"].get("allocated_peak_mb", "-")))

    L.append("\n## 并发加速比\n")
    L.append("| 并发 | 吞吐 req/min | 加速比 | 理想加速比 | 效率 |")
    L.append("|---|---|---|---|---|")
    for s in levels:
        c = s["concurrency"]
        if base_tp:
            sp = s["throughput_rpm"] / base_tp
            L.append(f"| {c} | {s['throughput_rpm']:.2f} | {sp:.2f}x | {c}x | {sp / c:.0%} |")

    L.append("\n## 分任务类型延迟\n")
    L.append("| 并发 | 任务 | 数量 | 成功率 | P50 s | P95 s | 最大 s |")
    L.append("|---|---|---|---|---|---|---|")
    for s in levels:
        for k, v in s["by_task"].items():
            L.append("| {} | {} | {} | {:.0%} | {:.2f} | {:.2f} | {:.2f} |".format(
                s["concurrency"], k, v["n"], v["success_rate"],
                v["p50_s"], v["p95_s"], v["max_s"]))

    L.append("\n## 轻任务饿死情况\n")
    L.append("> `light` = `/health` 探针。若在并发下其延迟暴涨, 说明轻请求被重请求阻塞。\n")
    L.append("| 并发 | light P50 ms | light 平均 ms | 饿死比 |")
    L.append("|---|---|---|---|")
    for s in levels:
        if "light" in s["by_task"]:
            v = s["by_task"]["light"]
            L.append("| {} | {:.0f} | {:.0f} | {} |".format(
                s["concurrency"], v["p50_s"] * 1000, s.get("light_mean_s", 0) * 1000,
                s.get("light_starvation_ratio", "-")))

    L.append("\n## 显存\n")
    L.append("| 并发 | 峰值MB | 均值MB | 抖动(标准差)MB | 最低空闲MB |")
    L.append("|---|---|---|---|---|")
    for s in levels:
        v = s["vram"]
        L.append("| {} | {} | {} | {} | {} |".format(
            s["concurrency"], v.get("allocated_peak_mb", "-"),
            v.get("allocated_mean_mb", "-"), v.get("allocated_std_mb", "-"),
            v.get("free_min_mb", "-")))

    allerr = [e for s in levels for e in s["errors"]]
    if allerr:
        L.append("\n## 错误\n")
        L.append("| 任务 | 错误 | 次数 |")
        L.append("|---|---|---|")
        for e in allerr[:15]:
            L.append(f"| {e['task']} | {e['error']} | {e['count']} |")

    return "\n".join(L) + "\n"


# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--requests", type=int, default=32,
                    help="每个并发级别的请求总数")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-warmup", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.concurrency = [1, 4]
        args.requests = 16

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("并发 / 多任务压力测试")
    print(f"  服务      : {BASE}")
    print(f"  并发级别  : {args.concurrency}")
    print(f"  每级请求数: {args.requests}")
    print(f"  负载构成  : {TASK_WEIGHTS}")
    print("=" * 66)

    # 健康检查
    try:
        h = requests.get(f"{BASE}/health", timeout=10).json()
        print(f"\n服务在线: {h}")
    except Exception as e:  # noqa: BLE001
        print(f"\n[FATAL] 服务不可用: {e}")
        return 1

    if not args.skip_warmup:
        print("\n预热: 逐个加载模型(首次加载较慢)...")
        for k in ("asr", "tts", "lipsync"):
            t0 = time.time()
            _s = requests.Session(); _s.trust_env = False
            r = do_task(k, AUDIO["u1"], _s)
            print(f"  {k:8s} {'OK' if r['ok'] else 'FAIL'}  {time.time() - t0:.1f}s")

    print("\n测量空载 health 基线...")
    light_base = measure_light_baseline()
    print(f"  基线延迟: {light_base * 1000:.1f} ms")

    levels = []
    for c in args.concurrency:
        print(f"\n--- 并发 N={c} ({args.requests} 请求) ---")
        lv = run_level(c, args.requests)
        s = summarize(lv, light_base)
        levels.append(s)
        print("  墙钟 {:.1f}s | 吞吐 {:.2f} req/min | 成功率 {:.0%}".format(
            s["wall_clock_s"], s["throughput_rpm"], s["success_rate"]))
        print("  延迟 P50 {:.2f}s | P95 {:.2f}s | P99 {:.2f}s | 排队 {}".format(
            s["latency_p50_s"], s["latency_p95_s"], s["latency_p99_s"],
            f"{s['queue_mean_s']:.2f}s" if s["queue_mean_s"] is not None else "-"))
        if s["vram"]:
            print("  显存 峰值 {:.0f}MB | 抖动 {:.0f}MB | 最低空闲 {:.0f}MB".format(
                s["vram"].get("allocated_peak_mb", 0),
                s["vram"].get("allocated_std_mb", 0),
                s["vram"].get("free_min_mb", 0)))
        if s["errors"]:
            print(f"  错误: {s['errors'][:2]}")

    # 加速比
    base_tp = next((s["throughput_rpm"] for s in levels if s["concurrency"] == 1), None)
    if base_tp:
        print("\n并发加速比:")
        for s in levels:
            c = s["concurrency"]
            sp = s["throughput_rpm"] / base_tp
            print(f"  N={c}: {sp:.2f}x (理想 {c}x, 效率 {sp / c:.0%})")

    light = [s for s in levels if s.get("light_starvation_ratio")]
    if light:
        print("\n轻任务(/health)饿死比:")
        for s in light:
            print(f"  N={s['concurrency']}: {s['light_starvation_ratio']}x "
                  f"({s['light_mean_s'] * 1000:.0f}ms vs 基线 {light_base * 1000:.1f}ms)")

    # 落盘
    payload = {
        "test": "concurrency",
        "base_url": BASE,
        "task_weights": TASK_WEIGHTS,
        "light_baseline_s": round(light_base, 4),
        "levels": levels,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "report.md").write_text(to_md(levels, light_base), encoding="utf-8")

    print(f"\n结果 -> {OUT_DIR / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
