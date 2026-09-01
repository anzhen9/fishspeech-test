"""
统一语音处理服务 (Unified Speech Service, USS) —— FastAPI 入口

三大能力:
  /v1/asr     语音识别 + 字幕(SRT/VTT)
  /v1/tts     语音合成, 支持参考音频做音色克隆(FishSpeech 1.5)
  /v1/lipsync 音频 -> 口型时间轴 + 口型动画视频(与音轨同步)

启动:
  D:\\fishspeech\\.tools\\python311\\python.exe -m service.server
  (或) uvicorn service.server:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger

from .config import (CONFIG, DATA_DIR, LOG_DIR, MODEL_IDLE_TTL_S, OUTPUT_DIR,
                      VOICES_DIR)
from . import tts_cache
from .concurrency import GPU_TOTAL_SLOTS, queue_depth, run_cpu, run_gpu
from .engines import asr as asr_mod
from .engines import clone as clone_mod
from .engines import renderer, viseme
from .engines import speaker as spk_mod
from .gpu import SCHEDULER, vram_stats
from .jobs import QUEUE

logger.add(LOG_DIR / "service_{time}.log", rotation="50 MB", retention="7 days",
           encoding="utf-8", level="INFO")

START_TS = time.time()


# ==================================================================
# 生命周期
# ==================================================================
async def _ttl_sweeper():
    """周期性回迁空闲模型, 释放显存。"""
    while True:
        await asyncio.sleep(30)
        try:
            swept = SCHEDULER.sweep_idle()
            if swept:
                logger.info(f"[TTL] 已回迁空闲模型: {swept}")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[TTL] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_ttl_sweeper())
    # 在接待任何请求之前修补 speechbrain 的惰性导入。
    # 该补丁只在其模块尚未导入时才有意义, 所以必须在启动时就做一次,
    # 否则谁先碰到 speechbrain 谁就可能踩到 RecursionError。
    spk_mod._patch_speechbrain_lazy()
    logger.info("=" * 60)
    logger.info("统一语音处理服务启动")
    logger.info(f"  设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  显存预算: {CONFIG.__class__.__name__} / {SCHEDULER.budget_mb}MB")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info("=" * 60)
    yield
    task.cancel()
    SCHEDULER.unload_all()


app = FastAPI(
    title="Unified Speech Service",
    description="基于 FishSpeech 1.5 的统一语音处理服务: 口型同步 / 字幕识别 / 语音克隆",
    version="1.0.0",
    lifespan=lifespan,
)


# ==================================================================
# 工具
# ==================================================================
def _job_dir(prefix: str) -> Path:
    d = OUTPUT_DIR / f"{prefix}_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


async def _save_upload(file: UploadFile, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest


def _resolve_audio(audio_path: str | None, upload: UploadFile | None, workdir: Path) -> Path:
    if upload is not None:
        return workdir / (upload.filename or "input.wav")
    if audio_path:
        p = Path(audio_path)
        if not p.exists():
            p = DATA_DIR / audio_path
        if not p.exists():
            raise HTTPException(404, f"音频文件不存在: {audio_path}")
        return p
    raise HTTPException(400, "需要提供 audio 文件或 audio_path")


def _resolve_voice(reference_id: str) -> Path | None:
    """按 id 在注册音色库 data/voices/ 中查找 <id>.wav。"""
    p = VOICES_DIR / f"{reference_id}.wav"
    return p if p.exists() else None


# ==================================================================
# 元信息
# ==================================================================
@app.get("/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - START_TS, 1),
            "cuda": torch.cuda.is_available()}


@app.get("/v1/status")
def status():
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram": vram_stats(),
        "scheduler": SCHEDULER.status(),
        # 各分组当前排队等待的任务数, 用于判断服务是否过载
        "queue": queue_depth(),
        # TTS 结果缓存命中情况 —— 命中率直接决定多人场景的有效吞吐
        "tts_cache": tts_cache.stats(),
        "config": {
            "asr_model": CONFIG.asr.model_size,
            "asr_compute": CONFIG.asr.compute_type,
            "tts_precision": CONFIG.tts.precision,
            "lipsync_fps": CONFIG.lipsync.fps,
            "model_idle_ttl_s": MODEL_IDLE_TTL_S,
            "gpu_total_slots": GPU_TOTAL_SLOTS,
        },
    }


@app.post("/v1/models/unload")
def unload_models(name: str | None = None):
    if name:
        SCHEDULER.unload(name)
    else:
        SCHEDULER.unload_all()
    return {"ok": True, "status": SCHEDULER.status()}


@app.post("/v1/cache/clear")
def clear_cache():
    """清空 TTS 结果缓存。压测前调用, 保证测量的是真实合成耗时。"""
    return {"ok": True, "removed": tts_cache.clear(), "stats": tts_cache.stats()}


@app.get("/v1/cache/stats")
def cache_stats():
    """TTS 缓存命中率。命中率是衡量多人场景有效吞吐的关键指标。"""
    return tts_cache.stats()


# ==================================================================
# 1) ASR / 字幕
# ==================================================================
async def _do_asr(audio_path: str,
                  language: str | None = None,
                  word_timestamps: bool = True,
                  fmt: str = "all") -> dict:
    """ASR 的纯业务逻辑。同步端点与异步任务共用这一份实现。

    只接受服务端路径(不接受 UploadFile): 异步任务场景下文件已在服务端,
    需要上传的场景走同步端点 /v1/asr。
    """
    wd = _job_dir("asr")
    src = _resolve_audio(audio_path, None, wd)

    # 阻塞的推理必须走线程池, 否则会独占事件循环(详见 service/concurrency.py)
    def _job():
        return asr_mod.get_asr().transcribe(
            src, language=language, word_timestamps=word_timestamps)

    res = await run_gpu("asr", _job)

    srt = asr_mod.ASREngine.to_srt(res["segments"])
    vtt = asr_mod.ASREngine.to_vtt(res["segments"])
    (wd / "asr.json").write_text(__import__("json").dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    (wd / "asr.srt").write_text(srt, encoding="utf-8")
    (wd / "asr.vtt").write_text(vtt, encoding="utf-8")

    out: dict[str, Any] = {k: res[k] for k in
                           ("text", "language", "language_probability",
                            "duration", "rtf", "latency_s", "model_size")}
    out["output_dir"] = str(wd)
    out["files"] = {"json": str(wd / "asr.json"), "srt": str(wd / "asr.srt"),
                    "vtt": str(wd / "asr.vtt")}
    if fmt in ("all", "srt"):
        out["srt"] = srt
    if fmt in ("all", "vtt"):
        out["vtt"] = vtt
    if fmt in ("all", "json"):
        out["segments"] = res["segments"]
    out["vram"] = vram_stats()
    return out


@app.post("/v1/asr")
async def api_asr(
    audio: UploadFile | None = File(None),
    audio_path: str | None = Form(None),
    language: str | None = Form(None),
    word_timestamps: bool = Form(True),
    fmt: str = Form("all"),           # all | json | srt | vtt
):
    """
    语音识别 -> 带时间戳字幕。
    fmt=all 时把 SRT/VTT 文本一并返回, 并在服务端落盘。
    """
    wd = _job_dir("asr")
    try:
        # 上传的文件先落盘, 之后交给与异步任务共用的实现
        if audio is not None:
            audio_path = str(await _save_upload(audio, wd / (audio.filename or "input.wav")))
        return await _do_asr(audio_path, language, word_timestamps, fmt)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[ASR] 失败")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ==================================================================
# 2) 语音合成 / 音色克隆
# ==================================================================
async def _do_tts(text: str,
                  reference_audio_path: str | None = None,
                  reference_id: str | None = None,
                  reference_text: str | None = None,
                  **kwargs) -> dict:
    """TTS 的纯业务逻辑。同步端点与异步任务共用这一份实现。"""
    import soundfile as sf

    wd = _job_dir("tts")
    ref_path = None
    if reference_audio_path:
        ref_path = _resolve_audio(reference_audio_path, None, wd)
    elif reference_id:
        ref_path = _resolve_voice(reference_id)
        if ref_path is None:
            raise HTTPException(404, f"未注册的音色 id: {reference_id}")
        if not reference_text:
            ref_txt = ref_path.with_suffix(".txt")
            if ref_txt.exists():
                reference_text = ref_txt.read_text(encoding="utf-8").strip()

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    wav_path = wd / "output.wav"

    # ---- 结果缓存: 命中则完全跳过 GPU 推理 ----
    ck = tts_cache.make_key(text, ref_path, reference_text, **kwargs)
    cached = tts_cache.get(ck)
    if cached is not None:
        shutil.copyfile(cached, wav_path)
        info = sf.info(str(wav_path))
        res = {
            "sample_rate": info.samplerate,
            "duration": round(info.duration, 3),
            "latency_s": 0.0,
            "rtf": 0.0,
            "cached": True,
            "used_reference": ref_path is not None,
        }
        res.update({
            "text": text,
            "output_dir": str(wd),
            "files": {"wav": str(wav_path)},
            "vram": vram_stats(),
        })
        return res

    def _job():
        return clone_mod.get_tts().synthesize(
            text=text,
            reference_audio=ref_path,
            reference_text=reference_text,
            **kwargs,
        )

    res = await run_gpu("tts", _job)

    sf.write(str(wav_path), res["audio"], res["sample_rate"])
    res.pop("audio")
    res["cached"] = False
    tts_cache.put(ck, wav_path)
    res.update({
        "text": text,
        "output_dir": str(wd),
        "files": {"wav": str(wav_path)},
        "vram": vram_stats(),
    })
    return res


@app.post("/v1/tts")
async def api_tts(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_audio_path: str | None = Form(None),
    reference_text: str | None = Form(None),
    reference_id: str | None = Form(None),
    temperature: float | None = Form(None),
    top_p: float | None = Form(None),
    repetition_penalty: float | None = Form(None),
    max_new_tokens: int | None = Form(None),
    seed: int | None = Form(None),
):
    """
    文本转语音。三种指定参考音色的方式(优先级从高到低):
      1. reference_audio      上传音频文件
      2. reference_audio_path 服务端已有音频路径(绝对/相对 DATA_DIR)
      3. reference_id         已注册音色库 data/voices/<id>.wav 的 id

    任一命中即触发音色克隆(in-context, 无需微调)。三者都缺省时为默认音色。
    reference_text 是参考音频对应的转写文本; 不传时服务端会用 ASR 自动补全
    (FishSpeech 要求 prompt_text 与 prompt_tokens 成对, 否则克隆静默失效)。
    """
    wd = _job_dir("tts")
    try:
        # 上传的文件先落盘, 之后交给与异步任务共用的实现
        if reference_audio is not None:
            reference_audio_path = str(
                await _save_upload(reference_audio, wd / (reference_audio.filename or "ref.wav")))
        return await _do_tts(
            text=text,
            reference_audio_path=reference_audio_path,
            reference_id=reference_id,
            reference_text=reference_text,
            temperature=temperature, top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens, seed=seed,
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[TTS] 失败")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/v1/voices")
def list_voices():
    """列出已注册的音色 id(位于 data/voices/)。"""
    voices = []
    for wav in sorted(VOICES_DIR.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        voices.append({
            "id": wav.stem,
            "wav": str(wav),
            "has_text": txt.exists(),
            "text": txt.read_text(encoding="utf-8").strip() if txt.exists() else None,
            "size_kb": round(wav.stat().st_size / 1024, 1),
        })
    return {"count": len(voices), "voices": voices, "dir": str(VOICES_DIR)}


@app.post("/v1/speaker/similarity")
async def api_similarity(
    reference: UploadFile | None = File(None),
    generated: UploadFile | None = File(None),
    reference_path: str | None = Form(None),
    generated_path: str | None = Form(None),
):
    """评估两段音频的音色相似度(x-vector 余弦)。

    两段音频均可任选「上传文件」或「服务端本地路径」之一提供;
    两者都为空则返回 400。
    """
    wd = _job_dir("spk")
    try:
        r, g = None, None
        if reference is not None:
            r = await _save_upload(reference, wd / "ref.wav")
        elif reference_path:
            r = _resolve_audio(reference_path, None, wd)
        if generated is not None:
            g = await _save_upload(generated, wd / "gen.wav")
        elif generated_path:
            g = _resolve_audio(generated_path, None, wd)
        if r is None or g is None:
            raise HTTPException(400, "reference/generated 需各提供文件或路径")
        def _job():
            return spk_mod.get_encoder().similarity(r, g)

        sim = await run_gpu("speaker", _job)
        return {**sim, "vram": vram_stats()}
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[Spk] 失败")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ==================================================================
# 3) 口型同步
# ==================================================================
async def _do_lipsync(audio_path: str,
                      asr_json: str | None = None,
                      fps: int | None = None,
                      render: bool = True,
                      fmt: str = "all") -> dict:
    """口型同步的纯业务逻辑。同步端点与异步任务共用这一份实现。"""
    import json as _json

    import soundfile as sf

    wd = _job_dir("lipsync")
    _t0 = time.time()
    src = _resolve_audio(audio_path, None, wd)

    info = sf.info(str(src))
    duration = float(info.duration)

    # ---- ASR (复用或新跑) ----
    if asr_json and Path(asr_json).exists():
        asr_res = _json.loads(Path(asr_json).read_text(encoding="utf-8"))
    else:
        def _asr_job():
            return asr_mod.get_asr().transcribe(src)

        asr_res = await run_gpu("asr", _asr_job)
        (wd / "asr.json").write_text(
            _json.dumps(asr_res, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- viseme 时间轴(纯 CPU: 拼音 + 时间轴插值) ----
    events = await run_cpu("cpu", viseme.build_viseme_timeline,
                           asr_res, src, duration=duration)
    timeline = await run_cpu("cpu", viseme.timeline_to_json, events, meta={
        "audio": str(src), "duration": duration,
        "fps": fps or CONFIG.lipsync.fps,
        "asr_model": asr_res.get("model_size"),
    })
    (wd / "visemes.json").write_text(
        _json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
    vtt = await run_cpu("cpu", viseme.timeline_to_vtt, events)
    (wd / "visemes.vtt").write_text(vtt, encoding="utf-8")

    out: dict[str, Any] = {
        "duration": round(duration, 3),
        "count": len(events),
        # 与 /v1/asr、/v1/tts 对齐, 便于压测脚本统一采集服务端处理时间
        "latency_s": round(time.time() - _t0, 3),
        "rtf": round((time.time() - _t0) / duration, 4) if duration > 0 else None,
        "output_dir": str(wd),
        "files": {"json": str(wd / "visemes.json"), "vtt": str(wd / "visemes.vtt")},
        "asr_text": asr_res.get("text", ""),
    }
    if fmt in ("all", "json"):
        out["events"] = timeline["events"]
    if fmt in ("all", "vtt"):
        out["vtt"] = vtt

    # ---- 渲染视频 ----
    if render:
        mp4 = await run_cpu(
            "render", renderer.render_lipsync_video,
            events, src, wd / "lipsync.mp4", duration, fps or CONFIG.lipsync.fps,
            CONFIG.lipsync.render_size,
        )
        out["files"]["mp4"] = str(mp4)

    out["vram"] = vram_stats()
    return out


@app.post("/v1/lipsync")
async def api_lipsync(
    audio: UploadFile | None = File(None),
    audio_path: str | None = Form(None),
    asr_json: str | None = Form(None),   # 已有 ASR 结果(json 路径)可复用, 省一次推理
    fps: int | None = Form(None),
    render: bool = Form(True),
    fmt: str = Form("all"),              # all | json | vtt
):
    """
    音频 -> viseme 时间轴 (+ 可选 MP4 口型动画)。
    内部会先跑 ASR 拿到词级时间戳, 因此口型由真实语音内容驱动。
    """
    wd = _job_dir("lipsync")
    try:
        # 上传的文件先落盘, 之后交给与异步任务共用的实现
        if audio is not None:
            audio_path = str(await _save_upload(audio, wd / (audio.filename or "input.wav")))
        return await _do_lipsync(audio_path, asr_json, fps, render, fmt)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[LipSync] 失败")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ==================================================================
# 异步任务: 提交 -> 轮询 -> 取结果
# 排队在单卡上是物理必然(TTS 吞吐仅 6.2 req/min), 这里把排队显式化:
# 提交立即返回 job_id, 可轮询「队列位置 + 状态」, 客户端无需长超时。
# ==================================================================
class JobSubmit(BaseModel):
    kind: str = Field(..., description="任务类型: asr | tts | lipsync")
    params: dict = Field(default_factory=dict, description="任务参数, 见各同步端点")


@app.post("/v1/jobs")
async def submit_job(req: JobSubmit):
    """提交异步任务, 立即返回 job_id(不阻塞)。

    params 与同步端点的参数一致, 但**只支持服务端路径**, 不支持文件上传。
    例: {"kind":"tts","params":{"text":"你好","reference_id":"demo_male"}}
    """
    try:
        job = await QUEUE.submit(req.kind, req.params)
    except KeyError as e:
        raise HTTPException(400, str(e))
    return {"job_id": job.id, "status": job.status, "kind": job.kind,
            "queue_pos": QUEUE._queue_pos(job.id)}


@app.get("/v1/jobs")
def list_jobs(kind: str | None = None, limit: int = 50):
    """列出任务(默认最近 50 条)。"""
    return {"jobs": QUEUE.list_jobs(kind, limit), "stats": QUEUE.stats()}


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """查询任务状态。

    status: pending(排队中) / running(执行中) / done / failed / cancelled
    queue_pos: 前面还有几个任务在排队, 前端可直接展示进度
    """
    j = QUEUE.get(job_id)
    if j is None:
        raise HTTPException(404, f"任务不存在或已过期: {job_id}")
    return j


@app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str):
    """取消任务。仅对 pending / running 有效。"""
    if not QUEUE.cancel(job_id):
        raise HTTPException(404, f"任务不存在或已结束: {job_id}")
    return {"job_id": job_id, "cancelled": True}


@app.get("/v1/output/{job}/{name}")
def get_output(job: str, name: str):
    p = OUTPUT_DIR / job / name
    if not p.exists():
        raise HTTPException(404, "文件不存在")
    return FileResponse(str(p))


# ==================================================================
# 一键流水线: 音频 -> 字幕 + 克隆音色复刻 + 口型
# ==================================================================
@app.post("/v1/pipeline")
async def api_pipeline(
    audio: UploadFile | None = File(None),
    audio_path: str | None = Form(None),
    clone_text: str | None = Form(None),
    render: bool = Form(True),
):
    """
    端到端: 参考音频 -> [ASR 字幕] + [音色克隆合成新句] + [新句口型动画]
    """
    import json as _json

    import soundfile as sf

    wd = _job_dir("pipe")
    try:
        if audio is not None:
            await _save_upload(audio, wd / (audio.filename or "ref.wav"))
        src = _resolve_audio(audio_path, audio, wd)

        result: dict[str, Any] = {"output_dir": str(wd), "files": {}}

        # 1) ASR
        def _asr_job():
            return asr_mod.get_asr().transcribe(src)

        asr_res = await run_gpu("asr", _asr_job)
        (wd / "asr.json").write_text(_json.dumps(asr_res, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        (wd / "asr.srt").write_text(asr_mod.ASREngine.to_srt(asr_res["segments"]), encoding="utf-8")
        result["asr"] = {"text": asr_res["text"], "duration": asr_res["duration"],
                         "rtf": asr_res["rtf"]}
        result["files"]["srt"] = str(wd / "asr.srt")

        # 2) 克隆合成
        if clone_text:
            def _tts_job():
                SCHEDULER.sweep_idle()
                return clone_mod.get_tts().synthesize(
                    clone_text, reference_audio=src,
                    reference_text=asr_res["text"] or None)

            gen = await run_gpu("tts", _tts_job)
            gen_wav = wd / "cloned.wav"
            sf.write(str(gen_wav), gen["audio"], gen["sample_rate"])
            result["clone"] = {k: gen[k] for k in ("duration", "latency_s", "rtf", "sample_rate")}
            result["files"]["cloned_wav"] = str(gen_wav)

            # 3) 新音频的口型
            def _gen_asr_job():
                return asr_mod.get_asr().transcribe(gen_wav)

            gen_asr = await run_gpu("asr", _gen_asr_job)
            events = await run_cpu("cpu", viseme.build_viseme_timeline,
                                   gen_asr, gen_wav, duration=gen["duration"])
            tl = await run_cpu("cpu", viseme.timeline_to_json, events)
            (wd / "visemes.json").write_text(
                _json.dumps(tl, ensure_ascii=False, indent=2), encoding="utf-8")
            result["lipsync"] = {"count": len(events), "duration": gen["duration"]}
            result["files"]["visemes"] = str(wd / "visemes.json")
            if render:
                mp4 = await run_cpu(
                    "render", renderer.render_lipsync_video,
                    events, gen_wav, wd / "cloned_lipsync.mp4", gen["duration"],
                    CONFIG.lipsync.fps, CONFIG.lipsync.render_size)
                result["files"]["mp4"] = str(mp4)

        result["vram"] = vram_stats()
        return result
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("[Pipeline] 失败")
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ==================================================================
# 异步任务执行器注册
# 与同步端点共用 _do_* 实现, 行为完全一致。
# 注意: 异步任务只接受服务端路径, 不支持文件上传
# (需要上传请用同步端点 /v1/asr、/v1/tts、/v1/lipsync)。
# ==================================================================
QUEUE.register("asr", lambda p: _do_asr(**p))
QUEUE.register("tts", lambda p: _do_tts(**p))
QUEUE.register("lipsync", lambda p: _do_lipsync(**p))


if __name__ == "__main__":
    import uvicorn

    from .config import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
