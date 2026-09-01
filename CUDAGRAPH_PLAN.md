# CUDA Graph 改造落地方案 —— FishSpeech TTS 解码

> 目标：把 TTS 从 RTF 1.95 压到 ~1.0（3060 实测基线见 PERFORMANCE.md §6.5）。
> 依据：每 token 约 11 000 次碎小 kernel launch，非计算开销（搬运+视图整形）占 ~45%，
> 有效带宽仅 4.5% —— 瓶颈是 launch 开销，CUDA Graph 是消除它的正解。

## 1. 为什么现在不能直接 capture

`src/fish-speech-1.5.0/tools/llama/generate.py` 的解码循环（`decode_one_token_ar_agent`，
约 L151-209）每步都有两类**动态性**，会让 naive capture 失效或回放错误结果：

| 动态点 | 现状 | 后果 |
|---|---|---|
| `input_pos` | 每步 `torch.tensor([pos])` 新建并传入 `model.forward_generate(x, input_pos)` | 图捕获时位置被固化，回放永远用同一位置 → 输出错乱 |
| KV cache 写入 | attention 内按 `input_pos` 写入不同 slot | 同上，位置错乱 |
| 采样分支 | top-p / repetition penalty 依赖上一步 logits 形状 | 形状稳定（batch=1, vocab 固定），**可以静态化** |
| 结束条件 | `max_new_tokens` / EOS 提前退出 | 回放图无法提前退出，需生成满长度后裁剪 |

## 2. 方案：静态 buffer + 逐 token copy-in + 分桶 capture

```
capture 阶段（每桶一次）:
  固定 shape: x[1,1,H], input_pos[1], kv_cache[静态满长度], logits[1,1,V]
  warmup 数步 -> torch.cuda.CUDAGraph().capture_begin/end

回放阶段（每 token）:
  x.copy_(next_token_embedding)          # H2D/D2D 拷进静态输入
  input_pos_static.fill_(pos)            # 原位改写位置
  graph.replay()                         # 一次性提交整步所有 kernel
  next_token = sample(logits_static)     # 采样在图外（CPU 侧小算子）
```

要点：

1. **`input_pos` 不再新建张量**：预分配 `input_pos_static[1]`，每步 `.fill_(pos)`（一次 4 字节写，替代原 H2D + tensor 创建）。
2. **KV cache 满长度预分配**：`max_new_tokens=1024` 对应 KV 约几百 MB（637M 模型、24 层），3060 预算内可行；结束时按实际长度裁剪注意力（或直接容忍尾部 padding，采样已由 EOS 决定内容）。
3. **分桶捕获**：无需按长度分多桶 —— 只有一个解码图，`input_pos` 已静态化；仅 `chunk_length=150` 分块时首 token（prefill）不走图，prefill 本身是大算子不受 launch 开销困扰。
4. **采样保留在图外**：`_to_copy`/multinomial 等采样算子（剖析中 6.1 万次 `_to_copy`）数量少且依赖 logits 值分支，图外执行不构成瓶颈；若实测仍热，再二阶段把 top-p 采样也吸进第二张图。

## 3. 改造点清单

| 文件 | 改动 | 风险 |
|---|---|---|
| `tools/llama/generate.py` | 新增 `decode_one_token_graphed()`；`load_model()` 返回后构建 static buffers + capture | 中：需复刻 `decode_one_token_ar_agent` 内部逻辑 |
| `fish_speech/models/text2semantic/llama.py` | `forward_generate` 需支持外部传入预分配 KV cache 与静态 `input_pos`（当前按 pos 动态写） | 中：attention 写槽逻辑改为 `cache.copy_` 风格 |
| `service/engines/clone.py` | 增加 `USS_TTS_CUDA_GRAPH=1` 开关，默认关闭，可回退旧路径 | 低 |
| `service/config.py` | `TTSConfig` 增加 `use_cuda_graph: bool` | 低 |

## 4. 风险与回退

- **正确性风险**：位置/槽位写错会产生乱音。回退开关 `USS_TTS_CUDA_GRAPH=0` 走原路径。
- **Windows 注意**：torch.compile/inductor 不可用（无 triton），但**手动 CUDA Graph API 不依赖 triton**，Windows 可用 —— 这正是本方案与 `compile=True` 的本质区别。
- **显存**：满长度 KV 静态预分配 + 多图常驻约 +0.5~1 GB；3060 预算 8.5GB 内需与 ASR 权重错峰（现有 ModelScheduler 已支持空闲回迁 CPU）。
- **提前 EOS**：图必须回放满 `max_new_tokens`；对短句浪费算力。缓解：按 `chunk_length` 分块后单块 `max_new_tokens` 实际 ~200-300，浪费可控；或实现"回放满长度但 EOS 后置零后续 token"。

## 5. 验证计划（口径沿用 REPORT.md）

| 项 | 通过标准 |
|---|---|
| 克隆音色相似度 | ≥ 0.55（当前 0.589，不得回退） |
| ASR 字准确率（对 TTS 产物） | ≥ 当前水平 97.59% |
| 听感抽查 | 3 条固定话术，人耳比对无误 |
| 性能 | tokens/s ≥ 18（当前 9.8），RTF ≤ 1.0 |
| 开关回退 | `USS_TTS_CUDA_GRAPH=0` 结果与改造前逐位一致 |

## 6. 预期收益

| 场景 | RTF | TTS 吞吐 |
|---|---|---|
| 3060 现状 | 1.95–2.30 | 6.2 req/min |
| 3060 + CUDA Graph（估） | ~1.0 | ~12 req/min |
| 4090 + CUDA Graph（估） | ~0.5 | ~24 req/min |

改造完成后，40 系显卡的带宽倍数才能线性兑现（见 PERFORMANCE.md §9 跨硬件对比）。
