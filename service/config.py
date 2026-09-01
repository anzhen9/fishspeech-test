"""
统一语音处理服务 —— 全局配置

针对 NVIDIA GeForce RTX 3060 (12GB, Ampere sm_86) 的显存约束做了专门设计:
  * TTS / ASR / 说话人编码三套模型峰值需求合计约 8~9GB
  * 用户桌面程序常驻占用约 2.7GB, 因此服务可用预算按 8.5GB 保守设定
  * 通过 ModelScheduler 做「按需驻留 + 空闲回迁 CPU」的模型调度, 避免三模型同时常驻爆显存
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ============================== 路径 ==============================

# 国内网络: HuggingFace 直连不通, 统一走 hf-mirror 镜像。
# 必须在 import huggingface_hub / faster_whisper 之前生效, 故置于本模块顶层。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

ROOT = Path(__file__).resolve().parent.parent
FISH_SRC = ROOT / "src" / "fish-speech-1.5.0"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
# 注册音色库: <id>.wav + <id>.txt(该音频的转写文本) 成对存放。
# 注册后可用 reference_id=<id> 直接复用音色, 不必每次上传参考音频。
VOICES_DIR = DATA_DIR / "voices"
LOG_DIR = ROOT / "logs"

for _d in (MODELS_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR, VOICES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# FishSpeech 1.5 权重目录
FISH_MODEL_DIR = MODELS_DIR / "fish-speech-1.5"

# ============================== 运行参数 ==============================

DEVICE = os.getenv("USS_DEVICE", "cuda")           # cuda / cpu
CUDA_INDEX = int(os.getenv("USS_CUDA_INDEX", "0"))
HOST = os.getenv("USS_HOST", "127.0.0.1")
PORT = int(os.getenv("USS_PORT", "8080"))

# 采样率
SAMPLE_RATE_TTS = 44100      # FishSpeech 1.5 原生输出 (Firefly VQGAN 21Hz -> 44.1k)
SAMPLE_RATE_ASR = 16000      # Whisper / 说话人编码统一 16k

# ============================== 显存预算 ==============================
# 单位: MB。总预算 8704MiB ≈ 8.5GiB, 为 12GB 卡预留约 3.5GB 给桌面程序、碎片与峰值激活。
GPU_TOTAL_BUDGET_MB = int(os.getenv("USS_GPU_BUDGET_MB", "8704"))

# 各模型常驻显存估算(实测标定, 见 README「显存预算标定」)
MODEL_VRAM_MB = {
    "fish_tts": 3600,     # dual_ar 1.2GB 权重 + FireflyVQGAN 0.2GB + KV cache/激活
    "asr": 3200,          # whisper large-v3 fp16 权重 3.0GB + 编码激活
    "asr_small": 1200,    # whisper small fp16
    "speaker": 700,       # ECAPA-TDNN + 特征前端
}

# 模型空闲多久(秒)后回迁 CPU, 给其它模型腾位置
MODEL_IDLE_TTL_S = int(os.getenv("USS_MODEL_TTL", "180"))


@dataclass
class TTSConfig:
    """FishSpeech 1.5 文本转语音(含音色克隆)配置"""

    checkpoint_dir: str = str(FISH_MODEL_DIR)

    # ---- 精度 ----
    # Ampere(sm_86) 原生支持 bf16; FishSpeech 官方即以 bf16 训练/推理。
    # fp16 在 LLAMA 主干上易出现 attention softmax 溢出, 故默认 bf16。
    precision: str = os.getenv("USS_TTS_PRECISION", "bfloat16")

    # ---- 解码 ----
    # 3060 是自回归解码, batch 恒为 1; 真正影响显存的是 KV cache 长度。
    max_new_tokens: int = int(os.getenv("USS_TTS_MAX_TOKENS", "1024"))
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.2
    chunk_length: int = 150          # 长文本分块阈值(字符), 控制单次 KV cache 峰值
    seed: int | None = 42

    # ---- 加速开关 ----
    use_sdpa: bool = True            # 使用 PyTorch 原生 SDPA(flash/mem-efficient)
    compile: bool = False            # Windows 上 triton/inductor 不稳定, 默认关闭
    use_gradient_checkpointing: bool = False  # 推理时关闭以换速度


@dataclass
class ASRConfig:
    """faster-whisper (CTranslate2) 语音识别配置"""

    # 3060 12GB 下 large-v3(fp16) 权重约 3GB, 可常驻;
    # 若与 TTS 并发紧张, 自动降级到 medium / small。
    model_size: str = os.getenv("USS_ASR_MODEL", "large-v3")
    fallback_size: str = os.getenv("USS_ASR_FALLBACK", "medium")
    device: str = DEVICE
    compute_type: str = os.getenv("USS_ASR_COMPUTE", "float16")
    cpu_threads: int = 8
    beam_size: int = 5
    vad_filter: bool = True
    word_timestamps: bool = True     # 口型同步依赖词级时间戳
    language: str | None = os.getenv("USS_ASR_LANG", "zh")
    download_root: str = str(MODELS_DIR / "whisper")

    # VAD 参数 —— 关键: faster-whisper 默认 speech_pad_ms=400,
    # 会把检测到的语音段向前/向后各填充 400ms。对纯转写无害(避免切边),
    # 但对时间戳是灾难: 填充出的静音被交还给 Whisper, 而 Whisper 面对
    # 前置静音时倾向于把首段时间戳预测到 0, 导致段起点系统性偏早。
    # 实测 30 条中文样本: 默认参数 onset 偏早 -370ms(30/30 全负),
    # 改 speech_pad_ms=0 后起点落到真实发声时刻(误差 <30ms)。
    # 保留 30ms 余量以防 Silero VAD 轻微切边。
    vad_parameters: dict = field(default_factory=lambda: {
        "speech_pad_ms": 30,
        # faster_whisper 默认 2000ms, 会把 2s 内的自然停顿全并进一段,
        # 口型就失去了"停顿闭口"的时机; 300ms 更贴近口语停顿。
        "min_silence_duration_ms": 300,
    })


@dataclass
class LipSyncConfig:
    """口型同步配置"""

    # 目标帧率; 25fps 是口型动画与视频的常用折中
    fps: int = int(os.getenv("USS_LIPSYNC_FPS", "25"))
    # 音素 -> 口型映射表版本
    viseme_set: str = "rhubarb9"     # A,B,C,D,E,F,G,H,X
    # 闭口判定: 静音超过此毫秒数则插入闭口(X)口型
    silence_close_ms: int = 60
    # 相邻同口型合并阈值(ms), 避免口型抖动
    merge_threshold_ms: int = 40
    # 尾部延展上限(ms): 末词 end 之后声波还在衰减, 口型应保持到最后。
    # 超过此上限就不再追(防止把尾部噪声也算成语音)。
    max_tail_ms: int = 500
    # 渲染分辨率
    render_size: tuple[int, int] = (512, 512)
    # Wav2Lip 真人视频口型(可选能力)权重路径, 未提供则只输出口型动画
    wav2lip_checkpoint: str | None = os.getenv("USS_WAV2LIP_CKPT")


@dataclass
class SpeakerConfig:
    """说话人音色相似度评估配置"""

    # SpeechBrain ECAPA-TDNN (VoxCeleb 预训练)
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = str(MODELS_DIR / "spkrec-ecapa-voxceleb")
    sample_rate: int = SAMPLE_RATE_ASR


@dataclass
class ServiceConfig:
    tts: TTSConfig = field(default_factory=TTSConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    lipsync: LipSyncConfig = field(default_factory=LipSyncConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)


CONFIG = ServiceConfig()
