"""对比修复前后的并发压测结果, 输出 Markdown 表格。"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "outputs" / "concurrency"


def load(name: str) -> dict | None:
    p = OUT / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def idx(rep: dict) -> dict[int, dict]:
    return {s["concurrency"]: s for s in rep["levels"]}


def main():
    before = load("report_BEFORE.json")
    after = load("report.json")
    if not before or not after:
        print("需要 report_BEFORE.json 与 report.json")
        return 1

    b, a = idx(before), idx(after)

    print("## 修复前后对比\n")
    print("| 并发 | 吞吐 前→后 | 加速比 前→后 | P50延迟 前→后 | P95延迟 前→后 | light延迟 前→后 | 成功率 前→后 |")
    print("|---|---|---|---|---|---|---|")
    for c in sorted(set(b) | set(a)):
        sb, sa = b.get(c), a.get(c)
        if not sb or not sa:
            continue
        btp, atp = sb["throughput_rpm"], sa["throughput_rpm"]
        b1, a1 = b.get(1, {}).get("throughput_rpm", 0), a.get(1, {}).get("throughput_rpm", 0)
        bl = sb["by_task"].get("light", {}).get("p50_s", 0) * 1000
        al = sa["by_task"].get("light", {}).get("p50_s", 0) * 1000
        print("| {} | {:.1f} → **{:.1f}** | {:.2f}x → **{:.2f}x** | {:.2f}s → **{:.2f}s** | "
              "{:.2f}s → **{:.2f}s** | {:.0f}ms → **{:.0f}ms** | {:.0%} → **{:.0%}** |".format(
                  c, btp, atp,
                  btp / b1 if b1 else 0, atp / a1 if a1 else 0,
                  sb["latency_p50_s"], sa["latency_p50_s"],
                  sb["latency_p95_s"], sa["latency_p95_s"],
                  bl, al,
                  sb["success_rate"], sa["success_rate"]))

    print("\n## 轻任务(/health)饿死比\n")
    print("| 并发 | 修复前 | 修复后 | 改善 |")
    print("|---|---|---|---|")
    for c in sorted(set(b) | set(a)):
        sb, sa = b.get(c), a.get(c)
        if not sb or not sa:
            continue
        rb = sb.get("light_starvation_ratio") or 0
        ra = sa.get("light_starvation_ratio") or 0
        imp = f"{rb / ra:.0f}x" if ra > 0 else "-"
        print(f"| {c} | {rb:,.0f}x | **{ra:,.1f}x** | {imp} |")

    print("\n## 排队时间\n")
    print("| 并发 | 修复前均值s | 修复后均值s | 修复前P95 s | 修复后P95 s |")
    print("|---|---|---|---|---|")
    for c in sorted(set(b) | set(a)):
        sb, sa = b.get(c), a.get(c)
        if not sb or not sa:
            continue
        print("| {} | {} | {} | {} | {} |".format(
            c, sb.get("queue_mean_s", "-"), sa.get("queue_mean_s", "-"),
            sb.get("queue_p95_s", "-"), sa.get("queue_p95_s", "-")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
