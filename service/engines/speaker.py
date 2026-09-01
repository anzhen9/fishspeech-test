"""
说话人编码器 —— 用于量化"音色克隆相似度"

主路径: SpeechBrain ECAPA-TDNN (VoxCeleb 预训练, 192 维 x-vector)
        这是学界通用的说话人验证骨干, 余弦相似度 > 0.55 通常判为同一说话人
降级路径: 若 SpeechBrain 不可用, 退化为自实现的 MFCC 统计特征余弦相似度
        (区分度低于 x-vector, 仅作兜底, 报告中会标注降级)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from ..config import CONFIG


def _patch_speechbrain_lazy() -> None:
    """修复 speechbrain LazyModule 的重入死循环(Windows / zip 标准库必现)。

    背景:
      speechbrain 把已废弃的 `speechbrain.pretrained` 之类的名字在
      sys.modules 里替换成一个 LazyModule 代理。任何对该模块的属性访问都会走
      LazyModule.__getattr__ -> ensure_module()。

      ensure_module() 为了打印弃用警告, 会调用 inspect.getframeinfo() 去找
      调用者; 而 getframeinfo -> findsource -> getmodule 会遍历 sys.modules
      并对每个模块取 __file__ —— 于是又碰到这个代理, 再次进入 __getattr__,
      无限递归直到 RecursionError。

      speechbrain 自己有道守卫:
          if importer_frame.filename.endswith("/inspect.py"): raise AttributeError
      但它在两种情况下失效:
        1) Windows 路径分隔符是 '\\' (D:\\...\\inspect.py)
        2) embeddable/zip 标准库里 co_filename 就是裸的 'inspect.py'
      本项目的 embeddable Python 两条全中, 所以必炸。

    修复:
      给 ensure_module 加一层重入锁。递归进来时若模块还没加载完就直接抛
      AttributeError(与上游守卫行为一致); 已加载则直接返回缓存模块, 不再
      走 inspect。这样无论路径长什么样都不会递归。
    """
    try:
        from speechbrain.utils.importutils import LazyModule
    except Exception:  # noqa: BLE001 - speechbrain 不可用则无需修补
        return
    if getattr(LazyModule, "_uss_reentry_safe", False):
        return

    _orig_ensure = LazyModule.ensure_module
    _state = {"busy": False}

    def ensure_module(self, stacklevel: int = 1):
        if _state["busy"]:
            if self.lazy_module is None:
                raise AttributeError(
                    f"speechbrain 惰性导入重入: {getattr(self, 'target', '?')}")
            return self.lazy_module
        _state["busy"] = True
        try:
            return _orig_ensure(self, stacklevel)
        finally:
            _state["busy"] = False

    LazyModule.ensure_module = ensure_module
    LazyModule._uss_reentry_safe = True


class SpeakerEncoder:
    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG.speaker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = None
        self.model = None
        self._load()

    # ------------------------------------------------------------ 加载
    _NEED_FILES = (
        "hyperparams.yaml",
        "embedding_model.ckpt",
        "classifier.ckpt",
        "mean_var_norm_emb.ckpt",
        "label_encoder.txt",
    )

    @classmethod
    def _local_files_ok(cls, savedir: Path) -> bool:
        """检查本地权重是否完整(存在且非 0 字节)。

        同时清理 0 字节残留: 失败的下载/Windows symlink 会在 savedir 里留下
        <name>.ckpt 空文件, 而 speechbrain 的 fetch() 只要目标文件存在就直接
        短路返回, 于是加载到一个空文件上, 报很难定位的 KeyError。
        """
        if not savedir.is_dir():
            return False
        for f in savedir.iterdir():
            if f.is_file() and f.stat().st_size == 0:
                logger.warning(f"[Spk] 删除 0 字节残留: {f.name}")
                try:
                    f.unlink()
                except OSError:
                    pass
        return all((savedir / f).exists() and (savedir / f).stat().st_size > 0
                   for f in cls._NEED_FILES)

    @staticmethod
    def _ensure_local_hparams(savedir: Path) -> str:
        """生成本地化的 hyperparams 文件, 使权重从本地目录加载。

        speechbrain 的 fetch() 默认 allow_updates=True: 只要 source 是 HuggingFace
        仓库 ID, 每次都会尝试联网比对更新, 网络不稳时会把已下载好的 ckpt 截断成
        0 字节。把 pretrained_path 改成本地绝对目录后, guess_source() 会判定为
        FetchFrom.LOCAL, 从而完全绕开联网。
        """
        src = savedir / "hyperparams.yaml"
        dst = savedir / "hyperparams_local.yaml"
        text = src.read_text(encoding="utf-8")
        local_path = str(savedir.resolve()).replace("\\", "/")
        patched = text.replace(
            "speechbrain/spkrec-ecapa-voxceleb", local_path
        )
        dst.write_text(patched, encoding="utf-8")

        # Pretrainer 会把每个 loadable 按 "<key>.ckpt" 的名字收集到 savedir。
        # label_encoder 的真身是 label_encoder.txt(文本格式), 需要预先复制成
        # label_encoder.ckpt; 否则 speechbrain 会在 Windows 上尝试建符号链接
        # (无权限 → 产生 0 字节文件), 后续解析直接失败。
        target = savedir / "label_encoder.ckpt"
        if (not target.exists()) or target.stat().st_size == 0:
            shutil.copyfile(savedir / "label_encoder.txt", target)
        return dst.name

    def _load(self) -> None:
        # ---- 主路径: SpeechBrain ECAPA-TDNN ----
        try:
            _patch_speechbrain_lazy()   # 必须在首次 import speechbrain 之前
            from speechbrain.inference.speaker import EncoderClassifier

            savedir = Path(self.cfg.savedir)
            have_local = self._local_files_ok(savedir)

            logger.info("[Spk] 加载 ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) ...")
            if have_local:
                # 全部走本地: 禁止联网更新, 使用本地化的 hyperparams
                os.environ["HF_HUB_OFFLINE"] = "1"
                hparams_file = self._ensure_local_hparams(savedir)
                source = str(savedir)
                logger.info(f"[Spk] 使用本地权重(离线模式): {savedir}")
            else:
                hparams_file = "hyperparams.yaml"
                source = self.cfg.source
                logger.info(f"[Spk] 本地权重不完整, 从 Hub 拉取: {source}")

            self.model = EncoderClassifier.from_hparams(
                source=source,
                hparams_file=hparams_file,
                savedir=str(savedir),
                run_opts={"device": str(self.device)},
            )
            self.backend = "ecapa"
            logger.info("[Spk] ECAPA-TDNN 就绪")
            return
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Spk] ECAPA 不可用({type(e).__name__}: {str(e)[:160]}), 降级到 MFCC")

        # ---- 降级路径: MFCC 统计特征 ----
        self.backend = "mfcc"
        logger.info("[Spk] 使用 MFCC 统计特征兜底(区分度较低)")

    # ------------------------------------------------------------ 前处理
    def _load_wav(self, audio: str | Path | np.ndarray, sr: int = 16000) -> torch.Tensor:
        import torchaudio

        if isinstance(audio, torch.Tensor):
            wav = audio.detach().float()
            cur_sr = self.cfg.sample_rate
        elif isinstance(audio, np.ndarray):
            wav = torch.from_numpy(audio.astype(np.float32))
            cur_sr = self.cfg.sample_rate
        else:
            wav, cur_sr = torchaudio.load(str(audio))
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if cur_sr != sr:
            wav = torchaudio.functional.resample(wav, cur_sr, sr)
        return wav

    # ------------------------------------------------------------ 嵌入
    @torch.inference_mode()
    def embed(self, audio: str | Path | np.ndarray) -> np.ndarray:
        wav = self._load_wav(audio)

        if self.backend == "ecapa":
            emb = self.model.encode_batch(wav.to(self.device))
            return emb.squeeze().float().cpu().numpy()

        # MFCC 兜底: 取 MFCC + Δ + ΔΔ 的均值/标准差拼接
        # 注意: torchaudio >= 2.1 移除了 torchaudio.functional.mfcc,
        # 必须用 transforms.MFCC。
        import torchaudio
        from torchaudio.transforms import MFCC

        mfcc = MFCC(
            sample_rate=16000, n_mfcc=40,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 80},
        )(wav).squeeze(0)
        d1 = torchaudio.functional.compute_deltas(mfcc)
        d2 = torchaudio.functional.compute_deltas(d1)
        feat = torch.cat([mfcc, d1, d2], dim=0)
        vec = torch.cat([feat.mean(1), feat.std(1)]).numpy()
        return vec

    # ------------------------------------------------------------ 相似度
    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).ravel()
        b = b.astype(np.float64).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def similarity(self, ref: str | Path | np.ndarray, gen: str | Path | np.ndarray) -> dict:
        e_ref = self.embed(ref)
        e_gen = self.embed(gen)
        cos = self.cosine(e_ref, e_gen)
        return {
            "cosine": round(cos, 4),
            "backend": self.backend,
            "dim": int(e_ref.size),
            # ECAPA 在 VoxCeleb 上的经验阈值: >0.55 同人, <0.35 不同人
            "same_speaker_est": bool(cos >= 0.55) if self.backend == "ecapa" else None,
        }


_ENCODER: SpeakerEncoder | None = None


def get_encoder() -> SpeakerEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = SpeakerEncoder()
    return _ENCODER
