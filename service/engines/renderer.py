"""
口型动画渲染器

把 viseme 时间轴渲染成与音轨严格同步的 MP4 视频。

实现说明:
  * 本机无系统级 ffmpeg, 使用 imageio-ffmpeg 自带的静态 ffmpeg 二进制
  * 渲染分两步: 先写无声视频, 再用 ffmpeg 混入原始音轨(-shortest 保证等长)
  * 相邻口型之间做 60ms 的几何插值, 避免口型跳变
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter

from .viseme import VISEME_GEOMETRY, VisemeEvent


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not Path(exe).exists():
        raise RuntimeError("未找到 ffmpeg 可执行文件, 请安装 imageio-ffmpeg")
    return exe


def sample_geometry(events: list[VisemeEvent], duration: float, fps: int,
                    transition_s: float = 0.06) -> np.ndarray:
    """
    逐帧采样 (openness, width, roundness), 口型切换处做线性插值。
    返回 shape = (n_frames, 3)
    """
    n = max(1, int(round(duration * fps)))
    arr = np.zeros((n, 3), dtype=np.float32)

    key = np.array([VISEME_GEOMETRY["X"]], dtype=np.float32)  # 默认静息
    # 构造关键点: 每个事件取"中心时刻"作为目标姿态
    keys: list[tuple[float, np.ndarray]] = []
    for e in events:
        mid = (e.start + e.end) / 2.0
        keys.append((mid, np.array(VISEME_GEOMETRY[e.viseme], dtype=np.float32)))
    if not keys:
        arr[:] = key
        return arr

    # 逐帧: 找到最近的关键点, 在 transition 窗口内与其前一关键点插值
    key_t = np.array([k[0] for k in keys])
    key_v = np.stack([k[1] for k in keys])

    for i in range(n):
        t = i / fps
        j = int(np.searchsorted(key_t, t, side="right"))
        j = min(j, len(keys) - 1)
        cur_t, cur_v = key_t[j], key_v[j]
        if j == 0:
            arr[i] = cur_v
            continue
        prev_t, prev_v = key_t[j - 1], key_v[j - 1]
        span = max(1e-6, cur_t - prev_t)
        alpha = float(np.clip((t - prev_t) / min(span, transition_s / max(span, 1e-6) * span), 0.0, 1.0))
        # 更直观: 在切换点前后 transition_s 内平滑
        w = float(np.clip((t - (cur_t - transition_s)) / (2 * transition_s), 0.0, 1.0))
        arr[i] = prev_v * (1 - w) + cur_v * w
    return arr


def _draw_frame(size: tuple[int, int], g: np.ndarray, frame_idx: int) -> Image.Image:
    """绘制单帧: 极简卡通脸 + 参数化嘴部。"""
    W, H = size
    openness, width, roundness = float(g[0]), float(g[1]), float(g[2])

    img = Image.new("RGB", (W, H), (245, 243, 240))
    d = ImageDraw.Draw(img)

    cx, cy = W // 2, int(H * 0.46)

    # ---- 头部 ----
    head_r = int(H * 0.30)
    d.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r],
              fill=(255, 224, 196), outline=(70, 60, 55), width=3)

    # ---- 眼睛(轻微眨眼, 与帧号弱相关, 增加自然感) ----
    blink = 1.0 if (frame_idx % 97) not in (0, 1) else 0.15
    eye_dx = int(head_r * 0.42)
    eye_y = cy - int(head_r * 0.22)
    for sx in (-1, 1):
        ex = cx + sx * eye_dx
        er = int(head_r * 0.11)
        d.ellipse([ex - er, eye_y - int(er * blink), ex + er, eye_y + int(er * blink)],
                  fill=(255, 255, 255), outline=(60, 50, 45), width=2)
        d.ellipse([ex - int(er * 0.38), eye_y - int(er * 0.38 * blink),
                   ex + int(er * 0.38), eye_y + int(er * 0.38 * blink)], fill=(45, 40, 38))

    # ---- 鼻子 ----
    d.line([cx, eye_y + int(head_r * 0.18), cx - int(head_r * 0.06), cy + int(head_r * 0.10)],
           fill=(120, 100, 90), width=2)

    # ---- 嘴部 ----
    mouth_cy = cy + int(head_r * 0.42)
    max_w = head_r * 0.85
    max_h = head_r * 0.62

    mw = max_w * (0.35 + 0.65 * width)
    mh = max(2.0, max_h * openness)

    # 圆唇时更接近正圆, 否则为扁椭圆
    rw = mw * (1.0 - 0.35 * roundness)
    rh = mh * (1.0 + 0.25 * roundness)

    x0, x1 = cx - rw / 2, cx + rw / 2
    y0, y1 = mouth_cy - rh / 2, mouth_cy + rh / 2

    # 口腔(深色)
    d.ellipse([x0, y0, x1, y1], fill=(96, 40, 48))
    # 唇线
    d.ellipse([x0, y0, x1, y1], outline=(168, 66, 74), width=4)
    # 牙齿(开口足够大时可见上齿)
    if openness > 0.25:
        th = rh * 0.30
        d.rectangle([x0 + 2, y0 + 1, x1 - 2, y0 + th], fill=(250, 250, 245))
        d.line([x0 + 2, y0 + th, x1 - 2, y0 + th], fill=(200, 190, 185), width=1)

    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img


def render_lipsync_video(
    events: list[VisemeEvent],
    audio_path: str | Path,
    out_path: str | Path,
    duration: float,
    fps: int = 25,
    size: tuple[int, int] = (512, 512),
) -> Path:
    """渲染口型动画并混音, 输出 MP4。"""
    import imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    silent = out_path.with_suffix(".silent.mp4")

    geo = sample_geometry(events, duration, fps)
    n_frames = geo.shape[0]
    logger.info(f"[Render] {n_frames} 帧 @ {fps}fps, 时长 {duration:.2f}s -> {out_path.name}")

    writer = imageio.get_writer(
        str(silent), fps=fps, codec="libx264", quality=7,
        pixelformat="yuv420p", macro_block_size=1,
    )
    try:
        for i in range(n_frames):
            writer.append_data(np.asarray(_draw_frame(size, geo[i], i)))
    finally:
        writer.close()

    # ---- 混入原始音轨 ----
    ff = _ffmpeg_exe()
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-i", str(silent),
        "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        logger.error(f"[Render] ffmpeg 混音失败: {r.stderr[:600]}")
        shutil.move(str(silent), str(out_path))
    else:
        silent.unlink(missing_ok=True)

    return out_path
