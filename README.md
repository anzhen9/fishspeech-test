# 统一语音处理服务（Unified Speech Service, USS）

基于 **FishSpeech 1.5** 的语音处理服务，在 **NVIDIA GeForce RTX 3060（12 GB）** 上集成三项能力：

| 能力 | 说明 |
| --- | --- |
| **口型同步** | 音频 → 音节级口型时间轴 → 25 fps 口型动画视频（与音轨严格同步） |
| **字幕识别** | 音频 → 带时间戳字幕（SRT / VTT / JSON） |
| **语音克隆** | 10 s 参考音频 → 克隆音色 → 合成任意文本 |

- 功能精度测试报告见 **[REPORT.md](REPORT.md)**
- 性能与容量分析见 **[PERFORMANCE.md](PERFORMANCE.md)**
- 服务监听 `http://127.0.0.1:8080`，自带 Swagger 文档 `http://127.0.0.1:8080/docs`

---

## 1. 技术架构

```
                          FastAPI (service/server.py)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   /v1/asr                     /v1/tts                    /v1/lipsync
        │                           │                           │
  ┌─────▼──────┐          ┌─────────▼─────────┐        ┌────────▼────────┐
  │ ASREngine  │          │  FishTTS (clone)  │        │  viseme 时间轴   │
  │faster-     │          │ FishSpeech 1.5    │        │ ASR 词级时间戳   │
  │whisper     │          │  dual_ar LLaMA    │        │ → pypinyin      │
  │large-v3    │          │  + Firefly VQGAN  │        │   声母/韵母拆分  │
  │fp16        │          │  bf16             │        │ → Rhubarb 9宫   │
  └─────┬──────┘          └─────────┬─────────┘        │ → renderer→MP4  │
        │                           │                  └──────────────────┘
        │              参考音频→VQ 编码→prompt_tokens
        │              （in-context 前缀，克隆无需微调）
        │                           │
        └───────────────┬───────────┘
                        │
              ┌─────────▼──────────┐
              │  SpeakerEncoder    │  ECAPA-TDNN，音色相似度评估
              └─────────┬──────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
  ┌─────▼──────┐              ┌─────────▼────────┐
  │ModelSchedu-│              │  concurrency.py  │
  │ler(gpu.py) │              │ 线程池卸载+分组串行│
  │LRU回迁/OOM │              └──────────────────┘
  └────────────┘
        │
  ┌─────▼──────┐
  │ tts_cache  │ 结果缓存，命中即跳过 GPU
  │ jobs.py    │ 异步任务队列（排队可观测）
  └────────────┘
```

### 模块职责

| 文件 | 职责 |
| --- | --- |
| `service/server.py` | FastAPI 路由、请求编排、音色注册表 |
| `service/config.py` | 全局配置、显存预算、路径 |
| `service/gpu.py` | 显存调度器（LRU 回迁 / OOM 降级） |
| `service/concurrency.py` | 并发控制（线程池卸载 + 按模型分组串行） |
| `service/tts_cache.py` | TTS 结果缓存（实测 4.84x 吞吐） |
| `service/jobs.py` | 异步任务队列（排队显式化、可观测） |
| `service/engines/asr.py` | faster-whisper 封装 + SRT/VTT 导出 |
| `service/engines/clone.py` | FishSpeech 1.5 TTS + in-context 克隆 |
| `service/engines/viseme.py` | 拼音 → 口型时间轴 |
| `service/engines/renderer.py` | 口型动画渲染 MP4 |
| `service/engines/speaker.py` | ECAPA-TDNN 音色相似度 |

---

## 2. 依赖安装

### 2.1 版本要求

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | **3.11.9**（embeddable，64bit） | 3.12+ 与部分依赖不兼容 |
| PyTorch | **2.6.0+cu124** | 必须是 cu124 构建，3060 无 CUDA 13 支持 |
| CUDA Runtime | 12.4 | PyTorch 自带，无需系统 CUDA |
| cuDNN | 9.1.0 | PyTorch 自带 |
| faster-whisper | 1.1.0 | CTranslate2 4.5.0 后端 |
| speechbrain | 1.0.2 | ECAPA-TDNN 说话人编码器 |

### 2.2 安装（国内源）

```bash
# 1) 固化国内源（已写入 .tools/python311/pip.ini）
#    [global]
#    index-url = https://pypi.tuna.tsinghua.edu.cn/simple
#    trusted-host = pypi.tuna.tsinghua.edu.cn

# 2) PyTorch（走清华源镜像）
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    -f https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124/

# 3) 其余依赖
pip install -r requirements.txt

# 4) 校验
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.6.0+cu124 True NVIDIA GeForce RTX 3060
```

### 2.3 模型下载（国内网络）

HuggingFace 直连不通，项目已在 `service/config.py` **顶层**（早于任何 HF 库导入）设置：

```python
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
```

> **注意**：这两行必须在 `import huggingface_hub` / `import faster_whisper` **之前**执行，
> 所以放在 config 模块顶层而不是 main 里。

HF 的 xet CDN 对单连接限速严重，多 GB 权重建议用并发分片下载。

### 2.4 依赖完整性自检

踩过一个隐蔽的坑：**site-packages 中若干包目录被清空但目录仍在**。
由于 PEP 420 命名空间包机制，`import fastapi` 会"成功"（`__file__=None`）却是个空壳，
直到调用时才报 `AttributeError`。部署后建议自检：

```bash
python -c "
import fastapi, jinja2, networkx, cffi, anyio, imageio, pypinyin, soundfile, loguru
for m in (fastapi, jinja2, networkx, cffi, anyio, imageio, pypinyin, soundfile, loguru):
    print(m.__name__, 'OK' if getattr(m, '__file__', None) else '!! 空壳包')
"
```

若出现 `!! 空壳包`：`rm -rf <包目录>` 后 `pip install --force-reinstall --no-deps <包名>`。

---

## 3. 针对 RTX 3060（12 GB）的显存配置

### 3.1 约束分析

| 项 | 值 |
| --- | --- |
| 显卡总显存 | 12 288 MB |
| 桌面程序常驻占用 | ~2 700 MB |
| 碎片 + 峰值激活预留 | ~800 MB |
| **服务可用预算** | **8 704 MB（8.5 GiB）** |

三套模型若同时常驻：`ASR 3.2 GB + TTS 3.6 GB + Speaker 0.7 GB ≈ 7.5 GB`，
看似够用，但一旦叠加峰值激活（尤其 TTS 长文本自回归的 KV cache）就会 OOM。

### 3.2 调度策略（`service/gpu.py`）

```
1. 预算制   —— 全局 8704 MB 上限，超预算时按 LRU 把最久未用的模型回迁 CPU
2. 惰性加载 —— 模型首次被请求时才实例化（首次 TTS 请求约 8 s 加载）
3. 空闲回迁 —— 空闲超 180 s（MODEL_IDLE_TTL_S）自动卸载，让位给其它任务
4. OOM 兜底 —— 捕获 torch.OutOfMemoryError，清空缓存后以更低精度/更小模型重试
```

各模型常驻显存实测标定：

```python
MODEL_VRAM_MB = {
    "fish_tts":  3600,   # dual_ar 1.2GB 权重 + FireflyVQGAN 0.2GB + KV cache/激活
    "asr":       3200,   # whisper large-v3 fp16 权重 3.0GB + 编码激活
    "asr_small": 1200,   # 降级路径
    "speaker":    700,   # ECAPA-TDNN + 特征前端
}
```

### 3.3 精度选择

| 模型 | 精度 | 理由 |
| --- | --- | --- |
| FishSpeech TTS | **bfloat16** | Ampere（sm_86）原生支持；FishSpeech 官方即以 bf16 训练。**fp16 在 LLaMA 主干上易出现 attention softmax 溢出** |
| Whisper ASR | **float16** | CTranslate2 对 fp16 优化最成熟；Whisper 数值范围稳定，无溢出风险 |
| ECAPA-TDNN | fp32 | 模型仅 0.7 GB，无压缩必要，避免精度损失影响相似度判定 |

### 3.4 推理优化

| 手段 | 配置 | 说明 |
| --- | --- | --- |
| **Batch size** | 恒为 1 | TTS 是自回归解码，batch 无意义；真正影响显存的是 **KV cache 长度** |
| **分块解码** | `chunk_length=150` 字符 | 长文本切块，控制单次 KV cache 峰值 |
| **SDPA 注意力** | `use_sdpa=True` | PyTorch 原生 flash / mem-efficient attention |
| **`torch.compile`** | 默认 **关闭** | Windows 上 triton / inductor 不稳定；Linux 可开启 |
| **梯度检查点** | 关闭 | 推理场景关闭以换速度 |
| **KV cache 上限** | `max_new_tokens=1024` | 限制单次生成的 cache 上界 |
| **TF32** | speechbrain 默认启用 | 3060 上卷积/矩阵乘提速 |

---

## 4. 并发模型与容量（重要）

### 4.1 核心数据

实测串行打满时每类任务的吞吐上限：

| 任务 | 单次耗时 | 吞吐上限 | RTF |
| --- | --- | --- | --- |
| ASR 字幕 | 1.35 s | 44.4 req/min | 0.13 |
| 口型同步 | 1.37 s | 43.9 req/min | — |
| **TTS 克隆合成** | **9.60 s** | **6.2 req/min** | **1.95** |

> **TTS 是绝对瓶颈**，吞吐只有 ASR 的 1/7，根因是 dual-AR 自回归解码。

混合负载（ASR 40% / TTS 30% / 口型 15% / 探针 15%）下：
**吞吐上限约 18 req/min，且不随并发提升** —— 并发 8 时效率仅 13%。
这是单卡算力天花板，不是 bug：加并发只会让队列变长，不会让 GPU 变快。

### 4.2 事件循环阻塞问题（已修复）

推理服务有个经典陷阱：**`async def` 端点里直接调用同步阻塞函数，会独占整个事件循环**。
一个 9 秒的 TTS 请求期间，连 `/health` 探针都要排队 9 秒。
实测并发 8 时 `/health` 延迟达 **31 秒**（空载基线 3.7 ms 的 6 113 倍）。

`service/concurrency.py` 用三条原则解决：

```python
async def run_gpu(kind, fn, *args, **kwargs):
    async with _total_sem():          # 全局额度，防显存超限
        async with _group_sem(kind):  # 同类模型串行，异类可并行
            return await run_in_threadpool(fn, *args, **kwargs)
```

| 原则 | 做法 | 理由 |
| --- | --- | --- |
| 不阻塞事件循环 | `run_in_threadpool` | 轻请求不被重请求饿死 |
| 同类模型串行 | `asr`/`tts`/`speaker` 信号量 = 1 | 同一实例并发推理会共享 CUDA context 与 KV cache，会出错 |
| 异类模型并行 | 全局额度 `USS_GPU_SLOTS=2` | 不同模型显存区独立可重叠；额度限制防 OOM |

**修复效果**：`/health` 延迟 31 027 ms → **104 ms**（298 倍），成功率 84% → 100%。
**但吞吐上限不变** —— 买来的是"服务始终可响应"，不是"能干更多活"。

### 4.3 TTS 结果缓存（已实施）

单卡算力是硬上限，加并发无效。**缓存是绕过算力限制的唯一低成本手段**：
相同（文本 + 参考音频 + 参考文本 + 解码参数）直接复用已合成 wav，完全跳过 GPU。

实测（`tests/test_cache_benefit.py`）：

| 场景 | 命中率 | 平均延迟 | 吞吐 | 提升 |
| --- | --- | --- | --- | --- |
| 全部唯一文本 | 0% | 10.44 s | 5.8 req/min | 1.00x |
| **固定话术循环** | **75%** | **2.15 s** | **27.8 req/min** | **4.84x** |

> 只对**重复文本**有效（固定话术、批量配音、共用预设音色）。
> 实时对话这类文本唯一的场景收益为 0，需靠解码加速。

关键设计：参考音频按**内容**摘要（改名/移动不影响命中）、
解码参数参与 key（不会串味）、临时文件+原子替换（并发安全）。

### 4.4 容量参考

| 场景 | 负载构成 | 无缓存 | 缓存命中 75% |
| --- | --- | --- | --- |
| 纯字幕转写 | 100% ASR | ~40 人 | ~40 人（缓存不适用） |
| 纯口型驱动 | 100% 口型 | ~40 人 | ~40 人（缓存不适用） |
| 纯语音克隆（文本不重复） | 100% TTS | ~6 人 | ~6 人 |
| 纯语音克隆（固定话术） | 100% TTS | ~6 人 | **~27 人** |
| 混合（本次压测） | 40/30/15/15 | ~18 人 | 更高 |

> 假设每人每分钟 1 次请求。若每人每 30 秒请求一次，上表人数减半。

---

## 5. 启动服务

```bash
# 方式一
./.tools/python311/python.exe -m service.server

# 方式二
./.tools/python311/python.exe -m uvicorn service.server:app --host 127.0.0.1 --port 8080
```

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `USS_PORT` | 8080 | 监听端口 |
| `USS_HOST` | 127.0.0.1 | 监听地址 |
| `USS_GPU_BUDGET_MB` | 8704 | 显存预算 |
| `USS_MODEL_TTL` | 180 | 模型空闲回迁秒数 |
| `USS_TTS_PRECISION` | bfloat16 | TTS 精度 |
| `USS_ASR_MODEL` | large-v3 | Whisper 模型 |
| `USS_ASR_COMPUTE` | float16 | ASR 精度 |
| `USS_GPU_SLOTS` | 2 | 同时在跑的 GPU 任务数 |
| `USS_TTS_CACHE` | 1 | TTS 结果缓存开关（`0` 关闭） |
| `USS_TTS_CACHE_MAX` | 500 | 缓存 LRU 上限条数 |

启动后访问 `http://127.0.0.1:8080/docs` 查看交互式 API 文档。

---

## 6. API 接口定义

### 6.1 服务管理

| 端点 | 方法 | 说明 |
| --- | --- | --- |
| `/health` | GET | 存活探针 |
| `/v1/status` | GET | 显存、常驻模型、排队深度、缓存命中率、配置快照 |
| `/v1/models/unload` | POST | 手动卸载模型释放显存 |
| `/v1/voices` | GET | 列出已注册音色 |
| `/v1/cache/stats` | GET | TTS 缓存命中率 |
| `/v1/cache/clear` | POST | 清空 TTS 缓存 |
| `/v1/output/{job}/{name}` | GET | 下载产物文件 |

### 6.2 `POST /v1/asr` —— 语音识别 + 字幕

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `audio` | file | 二选一 | 上传音频 |
| `audio_path` | str | 二选一 | 服务端本地路径（相对 `data/` 亦可） |
| `language` | str | 否 | 默认 `zh` |
| `fmt` | str | 否 | `json` / `srt` / `vtt` / `all`，默认 `all` |
| `word_timestamps` | bool | 否 | 默认 `true`。**口型同步依赖词级时间戳** |

返回：`text` / `language` / `duration` / `rtf` / `files`，外加按 `fmt` 内联的 `segments[]` / `srt` / `vtt`。

> `fmt` 只决定响应体内联哪些字段；三种格式**始终落盘**到 `output_dir` 下的
> `asr.json` / `asr.srt` / `asr.vtt`。响应始终是 JSON 信封，不会返回裸 SRT 文本。

### 6.3 `POST /v1/tts` —— 语音合成 + 音色克隆

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `text` | str | ✅ | 待合成文本 |
| `reference_audio` | file | 三选一 | 上传参考音频 |
| `reference_audio_path` | str | 三选一 | 服务端本地路径 |
| `reference_id` | str | 三选一 | 已注册音色 id（见 `/v1/voices`） |
| `reference_text` | str | 否 | 参考音频的转写文本。**强烈建议提供** |
| `temperature` | float | 否 | 默认 0.7 |
| `top_p` | float | 否 | 默认 0.7 |
| `repetition_penalty` | float | 否 | 默认 1.2 |
| `max_new_tokens` | int | 否 | 默认 1024 |
| `seed` | int | 否 | 默认 42 |

> ⚠️ **`reference_text` 是关键**：FishSpeech 的克隆是 in-context learning，
> 参考音频经 VQ 编码得到 `prompt_tokens` 作为 LLaMA 前缀。
> **`prompt_text` 与 `prompt_tokens` 必须成对提供**，否则引擎静默置 `use_prompt=False`，
> 退化成随机音色 —— **没有任何报错**。
> 若用 `reference_id` 且未传 `reference_text`，服务会自动读取同名的 `.txt` 文件。

返回：`wav` / `duration` / `sample_rate` / `rtf` / `used_reference` / **`cached`**

### 6.4 `POST /v1/lipsync` —— 口型同步

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `audio` / `audio_path` | file / str | 二选一 | 输入音频 |
| `asr_json` | str | 否 | 复用已有 ASR 结果的 JSON 路径 |
| `fps` | int | 否 | 默认 25 |
| `render` | bool | 否 | 是否渲染 MP4，默认 `true` |
| `fmt` | str | 否 | `json` / `vtt` / `all`，默认 `all` |

返回：`events[]`（每个 viseme 的 start/end/viseme/phone/word/openness）/ `vtt` / `mp4` / `latency_s`

**viseme 集合**：Rhubarb 九宫 `A`–`H` + `X`（`X` = 闭口）

### 6.5 `POST /v1/speaker/similarity` —— 音色相似度

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `reference` / `reference_path` | file / str | ✅ | 参考音频 |
| `generated` / `generated_path` | file / str | ✅ | 待评估音频 |

返回：`cosine` / `same_speaker_est`（阈值 0.55）/ `backend` / `dim`

### 6.6 `POST /v1/pipeline` —— 全链路

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `audio` / `audio_path` | file / str | 二选一 | 参考音频 |
| `clone_text` | str | 否 | 需克隆合成的文本（不填则只做 ASR） |
| `render` | bool | 否 | 是否渲染口型 MP4 |

一次性完成：**ASR 字幕 → 克隆音色合成新句 → 新句口型动画**

---

## 7. 调用示例

### 7.1 curl

```bash
B=http://127.0.0.1:8080

# 健康检查
curl -s $B/health

# ASR 字幕（服务端本地路径）
curl -s -X POST $B/v1/asr \
  -F "audio_path=data/testset/clone_01_F_ref.wav" \
  -F "fmt=all"

# ASR 字幕（上传文件）
curl -s -X POST $B/v1/asr -F "audio=@./my_audio.wav" -F "language=zh" -F "fmt=srt"

# TTS 克隆（服务端路径 + 转写文本）
curl -s -X POST $B/v1/tts \
  -F "text=欢迎使用统一语音处理服务。" \
  -F "reference_audio_path=data/testset/clone_02_M_ref.wav" \
  -F "reference_text=这名患者曾去过尼日利亚，当地曾出现数宗埃博拉病毒的病例。"

# TTS 克隆（已注册音色 id）
curl -s -X POST $B/v1/tts -F "text=这是通过注册音色合成的语音。" -F "reference_id=demo_male"

# 口型同步
curl -s -X POST $B/v1/lipsync -F "audio_path=data/testset/clone_01_F_ref.wav" -F "fps=25" -F "render=true"

# 音色相似度（同人应 > 0.55）
curl -s -X POST $B/v1/speaker/similarity \
  -F "reference_path=data/testset/clone_02_M_ref.wav" \
  -F "generated_path=data/testset/clone_02_M_held.wav"

# 缓存命中率
curl -s $B/v1/cache/stats

# 全链路
curl -s -X POST $B/v1/pipeline \
  -F "audio_path=data/testset/clone_01_F_ref.wav" \
  -F "clone_text=这是全链路管道生成的语音。" -F "render=true"
```

### 7.2 Python

```python
import requests

B = "http://127.0.0.1:8080"

# ---- ASR ----
r = requests.post(f"{B}/v1/asr",
                  data={"audio_path": "data/testset/clone_01_F_ref.wav", "fmt": "all"})
res = r.json()
print(res["text"])
print(res["srt"][:300])

# ---- TTS 克隆（上传文件）----
with open("ref.wav", "rb") as f:
    r = requests.post(f"{B}/v1/tts",
                      data={"text": "你好，这是我的克隆音色。",
                            "reference_text": "参考音频对应的转写文本"},
                      files={"reference_audio": f})
res = r.json()
assert res["used_reference"], "克隆未生效！检查 reference_text 是否提供"
print(f"生成 {res['duration']:.2f}s 音频, RTF={res['rtf']:.2f}, 缓存命中={res['cached']}")

# ---- 口型同步 ----
r = requests.post(f"{B}/v1/lipsync",
                  data={"audio_path": "data/testset/clone_01_F_ref.wav",
                        "fps": 25, "render": "true"})
res = r.json()
print(f"{res['count']} 个口型事件, MP4: {res['files']['mp4']}")
for e in res["events"][:5]:
    print(f"  {e['start']:.2f}-{e['end']:.2f}s  {e['viseme']}  {e['word']}")

# ---- 音色相似度 ----
r = requests.post(f"{B}/v1/speaker/similarity",
                  data={"reference_path": "data/testset/clone_02_M_ref.wav",
                        "generated_path": "outputs/clone/clone_02_M_ref_cloned.wav"})
res = r.json()
print(f"余弦相似度 {res['cosine']:.4f} -> 同一说话人: {res['same_speaker_est']}")

# ---- 缓存命中率（容量监控关键指标）----
print(requests.get(f"{B}/v1/cache/stats").json())
```

```python
# ---- 异步任务(多人排队场景推荐)----
# 提交立即返回, 不用等几十秒的连接
r = requests.post(f"{B}/v1/jobs", json={
    "kind": "tts",
    "params": {"text": "这是一段需要合成的长文本。", "reference_id": "demo_male"},
})
job_id = r.json()["job_id"]

# 轮询状态, 可把 queue_pos 直接展示给用户
import time
while True:
    j = requests.get(f"{B}/v1/jobs/{job_id}").json()
    if j["status"] in ("done", "failed", "cancelled"):
        break
    print(f"  状态={j['status']} 前面还有 {j['queue_pos']} 个任务")
    time.sleep(2)

print("结果:", j.get("result") or j.get("error"))
```

### 7.3 注册音色

把音频和其转写文本成对放入 `data/voices/`：

```
data/voices/
├── demo_male.wav
└── demo_male.txt     # 内容: 这名患者曾去过尼日利亚，…
```

即可用 `reference_id=demo_male` 复用，无需每次上传：

```bash
curl -s $B/v1/voices     # 列出所有已注册音色
```

---

## 8. 运行测试

```bash
# 三项功能测试（精度指标）
./.tools/python311/python.exe -m tests.test_asr            # 30 条 → outputs/asr/report.json
./.tools/python311/python.exe -m tests.test_voice_clone    # 4 人   → outputs/clone/report.json
./.tools/python311/python.exe -m tests.test_lipsync        # 30 条 → outputs/lipsync/report.json

# 并发压测（混合三项，1/2/4/8）
./.tools/python311/python.exe -m tests.test_concurrency
./.tools/python311/python.exe -m tests.test_concurrency --quick    # 只跑 1/4

# 单项容量上限
./.tools/python311/python.exe -m tests.test_capacity

# TTS 缓存收益
./.tools/python311/python.exe -m tests.test_cache_benefit

# 修复前后对比
./.tools/python311/python.exe -m tests.compare_concurrency
```

每份 `report.json` 含 `summary`（汇总指标）、`details`（逐样本明细）、`vram`（显存快照）。
并发测试额外记录排队时间、分任务延迟与"轻任务饿死比"。

### 核心指标速览

| 能力 | 指标 | 实测 |
| --- | --- | --- |
| ASR | 字准确率 | **97.59 %** |
| ASR | 段起点偏差 | **−0.7 ms** |
| 克隆 | 目标相似度（留出段） | **0.589** |
| 克隆 | 区分度 margin | **+0.371** |
| 口型 | 起口偏差 | **−5.5 ms** |
| 口型 | 帧级一致性 F1 | **0.939** |
| 性能 | 混合负载吞吐 | ~18 req/min（不随并发提升） |
| 性能 | TTS 缓存收益 | **4.84x**（命中率 75%） |

详见 [REPORT.md](REPORT.md) 与 [PERFORMANCE.md](PERFORMANCE.md)。

---

## 9. 目录结构

```
fishspeech/
├── service/                 # 服务主体
│   ├── server.py            # FastAPI 路由
│   ├── config.py            # 全局配置 + 显存预算
│   ├── gpu.py               # 显存调度器
│   ├── concurrency.py       # 并发控制（线程池 + 分组串行）
│   ├── tts_cache.py         # TTS 结果缓存
│   ├── jobs.py              # 异步任务队列
│   └── engines/             # 各能力引擎
│       ├── asr.py           # faster-whisper
│       ├── clone.py         # FishSpeech 1.5 TTS + 克隆
│       ├── viseme.py        # 拼音 → 口型时间轴
│       ├── renderer.py      # 口型动画渲染
│       └── speaker.py       # ECAPA-TDNN 相似度
├── tests/                   # 端到端测试
│   ├── metrics.py           # 评测指标（CER / 帧 F1 / AV 偏移 …）
│   ├── prepare_data.py      # 数据集准备
│   ├── test_asr.py
│   ├── test_voice_clone.py
│   ├── test_lipsync.py
│   ├── test_concurrency.py      # 并发压测（混合三项）
│   ├── test_capacity.py         # 单项吞吐上限
│   ├── test_cache_benefit.py    # TTS 缓存收益
│   └── compare_concurrency.py   # 修复前后对比
├── data/
│   ├── manifest.json        # 测试集清单
│   ├── testset/             # 测试音频
│   ├── tts_cache/           # TTS 缓存（自动创建）
│   └── voices/              # 注册音色库
├── models/                  # 模型权重
├── outputs/                 # 产物 + report.json
├── src/fish-speech-1.5.0/   # FishSpeech 官方源码
├── REPORT.md                # 功能精度测试报告
├── PERFORMANCE.md           # 性能与容量分析
└── requirements.txt
```

---

## 10. 已知限制

- **TTS 慢于实时**：RTF ≈ 1.95，单卡 3060 纯 TTS 场景并发上限约 6 人
- **并发吞吐上限 ~18 req/min**（混合负载），不随并发提升 —— 单卡算力硬上限
- **缓存只对重复文本有效**：实时对话等文本唯一场景收益为 0
- **口型收口偏早** 73.8 ms：Whisper 末词时间戳不含能量衰减尾巴
- **口型为规则驱动的 9 宫 viseme**，非真人视频口型；真人视频需接 Wav2Lip
- **`/v1/status` 的 vram 字段不含 CTranslate2 显存**（PyTorch 分配器不可见），监控需结合 `nvidia-smi`
- **未做长时间稳定性测试**：显存泄漏与模型反复加载/颠簸未验证（建议 30 min+ 长跑）

完整限制说明与改进建议见 [REPORT.md §7–8](REPORT.md) 与 [PERFORMANCE.md §8](PERFORMANCE.md)。
