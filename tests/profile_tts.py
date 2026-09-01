"""
TTS 解码性能剖析 —— 定位「9.88 tokens/s」到底慢在哪。

背景: 3060 带宽 360 GB/s, 模型 1.85 GB, 理论上限约 194 tokens/s,
实测只有 9.88(官方日志 "Bandwidth achieved: 6.30 GB/s", 仅峰值的 1.75%)。
要么是 GPU 没喂饱(CPU/Python 瓶颈), 要么是每 token 做了过多无用功。

方法:
  1) 用 CUDA Event 测 GPU 实际活跃时间, 与墙钟对比 -> 判断 CPU 还是 GPU 瓶颈
  2) 分解各阶段耗时: 参考音频编码 / LLM 自回归 / 声码器解码
  3) 统计每 token 的 kernel launch 次数与 KV cache 清零开销

用法:
  python -m tests.profile_tts
  python -m tests.profile_tts --text "..." --max-tokens 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "fish-speech-1.5.0"))

import torch  # noqa: E402

from service.config import CONFIG  # noqa: E402
from service.engines.clone import FishSpeechEngine  # noqa: E402

REF = "data/testset/clone_02_M_ref.wav"
REF_TXT = "这名患者曾去过尼日利亚，当地曾出现数宗埃博拉病毒的病例。"
TEXT = "今天的天气非常不错，我们一起去公园散步吧。"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_bandwidth(size_mb: int = 512, iters: int = 20) -> float:
    """实测显存带宽(GB/s): 反复拷贝一块大 buffer 计时。"""
    if not torch.cuda.is_available():
        return 0.0
    n = size_mb * 1024 * 1024 // 4          # float32 元素数
    src = torch.empty(n, dtype=torch.float32, device="cuda")
    dst = torch.empty(n, dtype=torch.float32, device="cuda")
    dst.copy_(src)                           # 预热
    sync()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        dst.copy_(src)
    e.record()
    sync()
    ms = s.elapsed_time(e)
    # 每次拷贝读写各一次 -> 2 * size
    gb = size_mb / 1024 * 2 * iters
    del src, dst
    torch.cuda.empty_cache()
    return gb / (ms / 1000) if ms > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="限制生成 token 数; 0=用配置默认值")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--profile", action="store_true",
                    help="用 torch.profiler 统计 kernel 级耗时(定位真正的热点)")
    ap.add_argument("--top", type=int, default=20, help="profiler 输出前 N 个算子")
    args = ap.parse_args()

    print("=" * 66)
    print("TTS 解码性能剖析")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    p = torch.cuda.get_device_properties(0)
    # 注意: PyTorch 2.6 移除了 memory_clock_rate, 不能用它算标称带宽。
    # 这里改为实测: 拷贝一段大 buffer, 用 CUDA Event 计时反推带宽。
    bw = _measure_bandwidth()
    print(f"  实测显存带宽: {bw:.0f} GB/s")
    print("=" * 66)

    print("\n加载模型 ...")
    t0 = time.perf_counter()
    eng = FishSpeechEngine(CONFIG.tts)
    print(f"  加载耗时 {time.perf_counter() - t0:.1f}s")

    # 模型权重体积 -> 理论 token/s 上限
    n_params = sum(p_.numel() for p_ in eng.model.parameters())
    bytes_per_param = 2 if eng.precision == torch.bfloat16 else 4
    model_mb = n_params * bytes_per_param / 1024 / 1024
    print(f"  LLM 参数 {n_params / 1e6:.1f}M  权重 {model_mb:.0f} MB ({eng.precision})")

    for i in range(args.repeat):
        print(f"\n{'=' * 66}\n第 {i + 1} 次\n{'=' * 66}")

        # ---- 1) 参考音频编码 ----
        sync()
        t0 = time.perf_counter()
        pt = eng.encode_reference(REF)
        sync()
        t_enc = time.perf_counter() - t0
        print(f"[1] 参考音频编码(一次性): {t_enc * 1000:.0f} ms")

        # ---- 2) LLM 自回归生成 ----
        override = {}
        if args.max_tokens:
            override["max_new_tokens"] = args.max_tokens

        sync()
        wall0 = time.perf_counter()
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()

        res = eng.synthesize(text=args.text, reference_audio=REF,
                            reference_text=REF_TXT, **override)

        gpu_end.record()
        sync()
        wall = time.perf_counter() - wall0
        gpu_ms = gpu_start.elapsed_time(gpu_end)

        dur = res["duration"]
        print(f"[2] 合成总耗时(墙钟): {wall * 1000:.0f} ms")
        print(f"    GPU 事件计时:      {gpu_ms:.0f} ms")
        print(f"    生成音频:          {dur:.2f}s  -> RTF = {res['rtf']:.2f}")
        print(f"    tokens:            {res.get('tokens', '?')}")

        # ---- 判定 ----
        idle = wall * 1000 - gpu_ms
        print(f"\n    GPU 空闲(等待 CPU): {idle:.0f} ms  占比 {idle / (wall * 1000):.1%}")
        if idle / (wall * 1000) > 0.3:
            print("    >>> 判定: CPU/Python 侧瓶颈 (GPU 在等活干)")
            print("       方向: 减少每 token 的 Python 开销 / kernel launch")
            print("       (CUDA graph, 或减少 fill_/tensor 创建)")
        else:
            print("    >>> 判定: GPU 计算/带宽瓶颈")
            print("       方向: 降低精度 / 减少每 token 的计算量")

        # ---- 3) 每 token 估算 ----
        if res.get("tokens"):
            n_tok = res["tokens"]
            print(f"\n    每 token 墙钟: {wall * 1000 / n_tok:.1f} ms")
            print(f"    token/s:       {n_tok / wall:.2f}")

    # ---- 4) kernel 级剖析 ----
    if args.profile:
        print(f"\n{'=' * 66}\nkernel 级剖析 (top {args.top})")
        print("=" * 66)
        from torch.profiler import ProfilerActivity, profile

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            eng.synthesize(text=args.text, reference_audio=REF,
                           reference_text=REF_TXT, **override)

        ka = prof.key_averages()
        table = ka.table(sort_by="self_cuda_time_total", row_limit=args.top)
        print(table)

        # 汇总: 按 CUDA 自用耗时分类
        # 注意: PyTorch 2.x 把 self_cuda_time_total 改名为 self_device_time_total,
        # 两个名字都要试, 否则会静默取到 0。
        def _self_dev(ev):
            for a in ("self_device_time_total", "self_cuda_time_total"):
                v = getattr(ev, a, None)
                if v:
                    return v
            return 0.0

        def _count(ev):
            return getattr(ev, "count", 0) or 0

        cuda_total = sum(_self_dev(e) for e in ka)
        cpu_total = sum(getattr(e, "self_cpu_time_total", 0) or 0 for e in ka)
        n_calls = sum(_count(e) for e in ka if _self_dev(e) > 0)
        print(f"\n  CUDA kernel 自用总时间: {cuda_total / 1000:.1f} ms")
        print(f"  CPU 自用总时间:         {cpu_total / 1000:.1f} ms")
        print(f"  kernel 调用次数合计:    {n_calls}")
        if n_calls:
            print(f"  平均每次 kernel:        {cuda_total / n_calls:.1f} us")

        # 按算子性质分类 —— 判断时间花在「算」还是「搬运/整形」
        COMPUTE = {"mm", "bmm", "matmul", "addmm", "linear", "conv1d", "conv2d", "sdpa"}
        MOVE = {"copy_", "_to_copy", "to", "clone", "contiguous", "cat", "stack", "fill_"}
        VIEW = {"as_strided", "view", "reshape", "select", "slice", "transpose",
                "unsqueeze", "squeeze", "permute", "expand", "empty_strided", "empty"}

        def _base(name: str) -> str:
            return name.split("::")[-1].split("(")[0]

        buckets = {"计算": [0.0, 0], "搬运/拷贝": [0.0, 0], "视图/整形": [0.0, 0], "其它": [0.0, 0]}
        for e in ka:
            t = _self_dev(e)
            if t <= 0:
                continue
            b = _base(getattr(e, "key", "") or "")
            key = ("计算" if b in COMPUTE else
                   "搬运/拷贝" if b in MOVE else
                   "视图/整形" if b in VIEW else "其它")
            buckets[key][0] += t
            buckets[key][1] += _count(e)

        print("\n  CUDA 时间按算子性质分类:")
        print(f"  {'类别':<12}{'耗时 ms':>12}{'占比':>9}{'调用次数':>12}")
        for k, (t, c) in sorted(buckets.items(), key=lambda x: -x[1][0]):
            if t > 0:
                print(f"  {k:<12}{t / 1000:>12.1f}{t / cuda_total:>9.1%}{c:>12,}")
        non_compute = cuda_total - buckets["计算"][0]
        print(f"\n  >>> 非计算类开销占比: {non_compute / cuda_total:.1%}")
        if non_compute / cuda_total > 0.4:
            print("     大量时间花在搬运/整形而非矩阵运算 -> 属于 launch/访存受限")
            print("     对应手段: torch.compile(算子融合) 或 CUDA Graph(消除 launch 开销)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
