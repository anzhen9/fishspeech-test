"""
语音克隆引擎 —— FishSpeech 1.5 (Dual-AR LLaMA + Firefly VQGAN)

模型结构
--------
  * 主干: DualARTransformer, 24 层 LLaMA(主 AR) + 4 层 fast transformer(副 AR)
          dim=1024, 16 头, vocab=102048, 8 个 codebook × 1024
  * 声码器: Firefly VQGAN (FSQ 8×1024, 21Hz 帧率) -> 44.1kHz 波形
  * 克隆方式: 参考音频经 VQ 编码器得到 prompt_tokens, 作为 LLaMA 的前缀,
              属于 in-context learning, 无需微调, 5~15s 参考音频即可

3060 12GB 适配
--------------
  * bf16 权重约 1.2GB + VQGAN 0.18GB; 自回归解码 batch 恒为 1
  * 显存大头是 KV cache, 由 max_length 决定; 长文本按 chunk_length 分块生成
  * 关闭 gradient checkpointing(推理无需)、关闭 torch.compile(Windows 上
    triton/inductor 不稳定), 用 SDPA 作为注意力后端
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from ..config import CONFIG, FISH_SRC, TTSConfig
from ..gpu import SCHEDULER, empty_cache, pick_precision

_FISH_ON_PATH = False


def _ensure_fish_path() -> None:
    """把 FishSpeech 源码根目录加入 sys.path(仅一次)。"""
    global _FISH_ON_PATH
    if _FISH_ON_PATH:
        return
    p = str(FISH_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
    _FISH_ON_PATH = True


class FishSpeechEngine:
    def __init__(self, cfg: TTSConfig | None = None):
        _ensure_fish_path()
        self.cfg = cfg or CONFIG.tts
        self.device = torch.device(
            f"cuda:{CONFIG.__dict__.get('CUDA_INDEX', 0)}" if torch.cuda.is_available() else "cpu"
        )
        self.precision = pick_precision(self.device.type, self.cfg.precision)
        self.model = None
        self.decode_one_token = None
        self.decoder = None
        self.sample_rate = 44100
        self._load()

    # ------------------------------------------------------------ 加载
    def _load(self) -> None:
        from tools.llama.generate import load_model

        # ---- 1) Dual-AR LLaMA 主干 ----
        logger.info(f"[TTS] 加载 FishSpeech 1.5 主干 (precision={self.precision}) ...")
        t0 = time.perf_counter()
        self.model, self.decode_one_token = load_model(
            self.cfg.checkpoint_dir,
            device=self.device,
            precision=self.precision,
            compile=self.cfg.compile,
        )

        # generate_long() 自身不会建立 KV cache —— 必须由调用方先 setup_caches,
        # 否则 mask 会按残留的默认长度构建, 报 shape mismatch。
        # 该模型用 GQA(n_local_heads=2, head_dim=64), 8192 长度的 KV cache
        # 仅约 120MB, 在 3060 上可直接使用官方的 max_seq_len。
        with torch.device(self.device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=next(self.model.parameters()).dtype,
            )
        logger.info(f"[TTS] 主干就绪, 用时 {time.perf_counter() - t0:.1f}s "
                    f"(KV cache seq_len={self.model.config.max_seq_len})")

        # ---- 2) Firefly VQGAN 声码器 ----
        self.decoder = self._load_decoder()
        self.sample_rate = self.decoder.spec_transform.sample_rate
        logger.info(f"[TTS] 声码器就绪, 采样率 {self.sample_rate}Hz")

    def _load_decoder(self):
        # 不走 hydra.initialize(): 它要求 config_path 为相对路径,
        # 而我们的源码目录在项目外(绝对路径)。直接用 OmegaConf 读取 yaml 更稳。
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        cfg_path = FISH_SRC / "fish_speech" / "configs" / "firefly_gan_vq.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"未找到 VQGAN 配置: {cfg_path}")
        cfg = OmegaConf.load(cfg_path)
        model = instantiate(cfg)

        ckpt = Path(self.cfg.checkpoint_dir) / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        state = torch.load(ckpt, map_location=self.device, mmap=True, weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        if any("generator" in k for k in state):
            state = {k.replace("generator.", ""): v for k, v in state.items() if "generator." in k}
        model.load_state_dict(state, strict=False, assign=True)
        model.eval().to(self.device)
        return model

    # ------------------------------------------------------------ 参考音频编码
    @torch.inference_mode()
    def encode_reference(self, wav_path: str | Path) -> torch.Tensor:
        """参考音频 -> VQ prompt tokens(音色指纹)。"""
        import torchaudio

        audio, sr = torchaudio.load(str(wav_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        audios = audio[None].to(self.device)
        lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        tokens = self.decoder.encode(audios, lengths)[0][0]
        logger.info(f"[TTS] 参考音频编码完成: {tokens.shape}")
        return tokens

    # ------------------------------------------------------------ 参考文本
    def _resolve_prompt_text(self, reference_audio: str | Path | None) -> str | None:
        """FishSpeech 的 prompt 机制要求 prompt_text 与 prompt_tokens 成对出现。

        用户通常只提供音频而不提供对应文本, 此时必须用 ASR 自动转写补全,
        否则 use_prompt=False, 音色克隆会静默退化成随机音色。
        """
        try:
            from . import asr as asr_mod

            engine = asr_mod.get_asr()
            res = engine.transcribe(str(reference_audio))
            txt = (res.get("text") or "").strip()
            if txt:
                logger.info(f"[TTS] 参考音频自动转写: {txt[:40]}...")
                return txt
            logger.warning("[TTS] 参考音频转写为空, 将以无 prompt 模式合成")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[TTS] 参考音频转写失败({type(e).__name__}: {e}), 退化为无 prompt 合成")
        return None

    # ------------------------------------------------------------ 推理
    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        reference_audio: str | Path | None = None,
        reference_text: str | None = None,
        prompt_tokens: torch.Tensor | None = None,
        **override,
    ) -> dict[str, Any]:
        from tools.llama.generate import generate_long

        if prompt_tokens is None and reference_audio is not None:
            prompt_tokens = self.encode_reference(reference_audio)

        # prompt_text 缺失时自动用 ASR 补全 —— 否则克隆不生效
        if prompt_tokens is not None and not (reference_text or "").strip():
            reference_text = self._resolve_prompt_text(reference_audio)

        max_new_tokens = override.get("max_new_tokens", self.cfg.max_new_tokens)
        temperature = override.get("temperature", self.cfg.temperature)
        top_p = override.get("top_p", self.cfg.top_p)
        repetition_penalty = override.get("repetition_penalty", self.cfg.repetition_penalty)
        chunk_length = override.get("chunk_length", self.cfg.chunk_length)
        seed = override.get("seed", self.cfg.seed)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        t0 = time.perf_counter()
        segments: list[np.ndarray] = []
        n_chunks = 0

        gen = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            compile=self.cfg.compile,
            iterative_prompt=chunk_length > 0,
            chunk_length=chunk_length,
            max_length=4096,
            # 注意: generate_long 内部会把 str / Tensor 各自再包装成 list,
            # 这里必须传"未包装"的原始值, 否则会双重包装导致 zip 出 list。
            prompt_text=reference_text if prompt_tokens is not None else None,
            prompt_tokens=prompt_tokens,
        )

        for resp in gen:
            if resp.action == "next":
                break
            with torch.autocast(
                device_type=self.device.type, dtype=self.precision,
                enabled=(self.device.type == "cuda"),
            ):
                seg = self.decoder.decode(
                    indices=resp.codes[None],
                    feature_lengths=torch.tensor([resp.codes.shape[1]], device=self.device),
                )[0].squeeze()
            segments.append(seg.float().cpu().numpy())
            n_chunks += 1

        if not segments:
            raise RuntimeError("未生成任何音频, 请检查输入文本")

        audio = np.concatenate(segments, axis=0)
        elapsed = time.perf_counter() - t0
        dur = len(audio) / self.sample_rate
        empty_cache()

        return {
            "audio": audio.astype(np.float32),
            "sample_rate": self.sample_rate,
            "duration": round(dur, 3),
            "latency_s": round(elapsed, 3),
            "rtf": round(elapsed / dur, 3) if dur > 0 else None,
            "chunks": n_chunks,
            "used_reference": prompt_tokens is not None,
            "tokens": int(sum(len(s) for s in segments)),
        }


def get_tts() -> FishSpeechEngine:
    global _TTS
    try:
        _TTS
    except NameError:
        _TTS = None
    if _TTS is None:
        SCHEDULER.register("fish_tts", lambda: FishSpeechEngine())
        _TTS = SCHEDULER.get("fish_tts")
    return _TTS
