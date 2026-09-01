"""
HF 权重并发分片下载器

背景
----
国内访问 HuggingFace 时, hf-mirror.com 的大文件会被 302 重定向到
us.aws.cdn.hf.co (xet 桥接)。该 CDN 对**单连接**限速到几乎 0 B/s,
但支持 HTTP Range 且不限并发数。因此用多连接分片下载可绕过单连接限速。

用法
----
    python tools/hf_parallel_download.py <repo_id> <filename> <output> [连接数]

示例
----
    python tools/hf_parallel_download.py \
        Systran/faster-whisper-large-v3 model.bin \
        models/whisper/faster-whisper-large-v3/model.bin 16
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import requests

ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
DEFAULT_CONN = 16


class Progress:
    """线程安全的进度统计。"""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.t0 = time.time()

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n

    def render(self) -> str:
        with self.lock:
            pct = self.done / self.total * 100 if self.total else 0
            el = time.time() - self.t0
            spd = self.done / el / 1024**2 if el > 0 else 0
            eta = (self.total - self.done) / (self.done / el) if self.done and el else 0
        return f"\r  {self.done/1048576:8.1f}/{self.total/1048576:.1f} MB "
        f"({pct:5.1f}%) {spd:6.2f} MB/s  ETA {eta:5.0f}s   "


def resolve_final_url(repo: str, filename: str) -> tuple[str, int]:
    """跟随 302 拿到 CDN 直链与文件大小。

    部分 CDN 对 HEAD 不返回 content-length, 此时退化用
    `Range: bytes=0-0` 的 GET 并从 content-range 解析总长度。
    """
    url = f"{ENDPOINT}/{repo}/resolve/main/{filename}"

    size = 0
    final = url
    try:
        r = requests.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        final = r.url
        size = int(r.headers.get("content-length") or 0)
    except Exception:  # noqa: BLE001
        pass

    if not size:
        r = requests.get(url, headers={"Range": "bytes=0-0"},
                         allow_redirects=True, stream=True, timeout=60)
        r.raise_for_status()
        final = r.url
        cr = r.headers.get("content-range", "")
        # 形如 "bytes 0-0/1527906378"
        if "/" in cr:
            size = int(cr.rsplit("/", 1)[1])
        else:
            size = int(r.headers.get("content-length") or 0)
        r.close()

    if not size:
        raise RuntimeError(f"无法获取文件大小 (repo={repo}, file={filename})")
    return final, size


def download_range(url: str, start: int, end: int, path: Path, prog: Progress,
                   retries: int = 6) -> None:
    """下载 [start, end] 字节区间并写入文件对应偏移。"""
    for attempt in range(retries):
        try:
            headers = {"Range": f"bytes={start}-{end}"}
            with requests.get(url, headers=headers, stream=True, timeout=90) as resp:
                if resp.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {resp.status_code}")
                with open(path, "r+b") as f:
                    f.seek(start)
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        prog.add(len(chunk))
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            # 签名可能过期, 重新解析直链
            try:
                url, _ = resolve_final_url(*CURRENT[0])
            except Exception:  # noqa: BLE001
                pass


CURRENT: list = []


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    repo, filename, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    n_conn = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_CONN

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"已存在: {out} ({out.stat().st_size/1048576:.1f} MB), 跳过")
        return 0

    print(f"解析直链: {repo}/{filename}")
    url, size = resolve_final_url(repo, filename)
    CURRENT[:] = [(repo, filename)]
    print(f"大小: {size/1048576:.1f} MB | 连接数: {n_conn}")
    print(f"直链 host: {url.split('/')[2]}")

    # 预分配文件
    with open(out, "wb") as f:
        f.truncate(size)

    prog = Progress(size)
    chunk = size // n_conn
    threads = []
    for i in range(n_conn):
        start = i * chunk
        end = size - 1 if i == n_conn - 1 else (start + chunk - 1)
        t = threading.Thread(target=download_range,
                             args=(url, start, end, out, prog), daemon=True)
        t.start()
        threads.append(t)

    while any(t.is_alive() for t in threads):
        sys.stderr.write(prog.render())
        sys.stderr.flush()
        time.sleep(1.0)
    for t in threads:
        t.join()
    sys.stderr.write("\n")

    print(f"完成: {out} ({out.stat().st_size/1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
