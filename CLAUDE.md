下面是一份可以直接丢给 **gemini-cli / codex** 做实现的 **SPEC（Markdown）**：目标是做一个 **Python 后端 + HTML 前端** 的小 Demo，在**实时麦克风流**上先跑 **Silero VAD（声学 VAD）**，再在**检测到短暂停顿**时把“当前整段 turn 的音频（最多 8 秒）”送入 **Smart Turn（语义 VAD / turn-end detection）** 做 **end-of-turn** 判定。

> 关键信息来自：
>
> * Smart Turn repo 对输入格式/8 秒窗口/与 Silero 级联的建议 ([GitHub][1])
> * HuggingFace smart-turn-v3 文件列表（含 v3.2 CPU/GPU onnx 文件名） ([Hugging Face][2])
> * Pipecat 文档里 Smart Turn 与 Silero 的推荐 stop_secs=0.2（一致思路） ([Pipecat][3])
> * Silero VAD 的 Python 用法与 16k 支持 ([GitHub][4])

---

# SPEC: silero-vad + smart-turn 级联的 Python + HTML 实时 Demo

## 1. 目标（Goals）
- 浏览器端采集麦克风音频，实时传到 Python 服务端。
- 服务端做两级判定：
  1) **Silero VAD**：实时判断“在说话/静音”，并在检测到短暂静音（推荐 0.2s）时触发一次“turn 结束候选点”。
  2) **Smart Turn (smart-turn-v3.2)**：在候选点上，对“当前 turn 的整段音频（尽量给足上下文，最多 8 秒）”做语义 turn-end 推断，输出 end-of-turn 概率与最终决策（END / CONTINUE）。
- 前端页面展示：
  - 当前 VAD 状态（speaking/silence）
  - turn 缓冲时长、最近一次 Smart Turn 推断结果（分数、耗时）
  - END 时把这一段 turn 的音频标记为“完成”，并清空 turn buffer（进入下一轮）

## 2. 非目标（Non-goals）
- 不做 ASR / LLM / TTS；只做 turn detection 可视化。
- 不做生产级音频回声消除/降噪；只做 demo。

## 3. 关键约束（来自上游）
### 3.1 Smart Turn 输入约束
- 输入必须是 **16kHz、mono、PCM 浮点（float32）**音频。:contentReference[oaicite:4]{index=4}
- 最多支持 **8 秒**，建议把“用户当前这一轮说话的完整上下文”尽量塞进去；如果更长，**从开头截断**保留最后 ~8 秒。:contentReference[oaicite:5]{index=5}
- 如果不足 8 秒：**在开头补 0**，确保“真实音频在数组末尾、padding 在开头”。:contentReference[oaicite:6]{index=6}
- Smart Turn 设计上要**与轻量 VAD（如 Silero）配合**：Silero 检测到静音时，再对整段 turn 音频跑 Smart Turn。:contentReference[oaicite:7]{index=7}

### 3.2 Silero VAD 约束
- Silero 支持 **8000 / 16000 Hz**，本 demo 统一走 16k。:contentReference[oaicite:8]{index=8}
- Silero 官方给了 pip 用法示例：`pip install silero-vad` + `load_silero_vad()` 等。:contentReference[oaicite:9]{index=9}

### 3.3 Smart Turn 权重文件（HuggingFace）
- HuggingFace: `pipecat-ai/smart-turn-v3`
- v3.2 文件名（demo 默认用 CPU int8 版）：
  - `smart-turn-v3.2-cpu.onnx`（~8.68MB）
  - `smart-turn-v3.2-gpu.onnx`（~32.4MB）:contentReference[oaicite:10]{index=10}

## 4. 目录结构（建议）
```

smart-turn-demo/
backend/
app.py                 # FastAPI + WebSocket
vad_silero.py          # Silero VAD 包装（流式/阈值/0.2s stop）
smart_turn_onnx.py     # ONNXRuntime 推断 + 8s pad/trim
audio_buffer.py        # turn buffer / ring buffer
requirements.txt
web/
index.html
app.js
style.css              # 可选
models/
smart-turn-v3.2-cpu.onnx

```

## 5. 安装与运行（CLI）
### 5.1 Python 环境
- Python 3.10+（建议 3.11/3.12）
- 依赖：
  - fastapi, uvicorn[standard]
  - numpy
  - onnxruntime（CPU）或 onnxruntime-gpu（可选）
  - silero-vad（pip 包）
  - huggingface_hub（可选：自动下载）
  - soundfile（可选：调试落盘）

`backend/requirements.txt`（可直接照抄）：
```

fastapi==0.115.0
uvicorn[standard]==0.30.6
numpy==2.0.2
onnxruntime==1.19.2
silero-vad==5.1.2
huggingface_hub==0.24.7
soundfile==0.12.1

````

安装：
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### 5.2 下载 Smart Turn 模型（你已 CLI 登录 HF）

（方案 A：huggingface-cli）

```bash
mkdir -p models
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir ./models
```

（方案 B：代码里用 `hf_hub_download`，见下文 `smart_turn_onnx.py`）

> 文件名与版本来源：HF 文件列表里有 `smart-turn-v3.2-cpu.onnx` / `smart-turn-v3.2-gpu.onnx`。([Hugging Face][2])

### 5.3 启动

```bash
# 在 smart-turn-demo/ 下
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
# 浏览器打开 http://localhost:8000
```

## 6. WebSocket 协议（前后端约定）

* WS 地址：`ws://localhost:8000/ws`
* Client -> Server：

  1. JSON 文本消息（init）：

     ```json
     {"type":"init","sample_rate":16000,"format":"f32le","channels":1}
     ```
  2. 后续持续发送 **binary**：`Float32Array` 的 little-endian PCM（单声道 16k）
* Server -> Client：JSON 文本消息

  * VAD 状态更新：

    ```json
    {"type":"vad","state":"speaking","p":0.82,"silence_ms":0,"turn_ms":1340}
    ```
  * Smart Turn 推断结果（在静音候选点触发）：

    ```json
    {
      "type":"smart_turn",
      "turn_ms":3120,
      "model":"smart-turn-v3.2-cpu.onnx",
      "scores":{"end":0.73,"continue":0.27},
      "decision":"END",
      "latency_ms":14
    }
    ```
  * 错误：

    ```json
    {"type":"error","message":"..."}
    ```

## 7. 级联逻辑（核心）

### 7.1 turn buffer 规则

* turn buffer 收集“从 VAD 进入 speaking 开始”到“END 被确认”为止的所有音频。
* 当 Silero 判定静音累计达到 `STOP_SECS=0.2`（Pipecat 也推荐 0.2，符合 Smart Turn 使用方式）时，触发 Smart Turn 推断一次。([Pipecat][3])
* 如果 Smart Turn 判定 CONTINUE（用户只是停顿/没说完）：

  * 不清空 turn buffer，继续收音；
  * 后续如果又出现静音候选点，再对“整个 turn（最近 8 秒窗口）”重新推断。([GitHub][1])
* 如果 Smart Turn 判定 END：

  * 发送 `decision="END"` 到前端；
  * 清空 turn buffer，进入下一轮。

### 7.2 Smart Turn 输入拼装（必须实现）

给定 turn buffer（float32，16k）：

1. 若长度 > 8s：截取最后 8s（从开头截断）([GitHub][1])
2. 若长度 < 8s：在**开头补零**到 8s，使音频在末尾 ([GitHub][1])
3. 得到固定长度：`8 * 16000 = 128000` samples 的 float32

## 8. 代码骨架（必须产出）

> 说明：你不需要完全复刻上游 repo 的 `inference.py`，但要满足其输入要求（16k mono、8s pad/trim、在静音候选点对整段 turn 重跑）。([GitHub][1])

### 8.1 backend/smart_turn_onnx.py

要求：

* 支持两种加载方式：

  * `model_path` 指向本地 `models/smart-turn-v3.2-cpu.onnx`
  * 或者自动 `hf_hub_download(repo_id="pipecat-ai/smart-turn-v3", filename="smart-turn-v3.2-cpu.onnx")`
* 用 onnxruntime 初始化 session（CPU provider 默认即可）
* 提供：

  * `prepare_8s_input(wav_f32: np.ndarray) -> np.ndarray`  # pad/trim
  * `predict(wav_f32: np.ndarray) -> dict` # 返回 end/continue 分数 + 推断耗时

输出分数格式（本 demo 约定）：

* `scores = {"end": float, "continue": float}`
* `decision = "END" if end >= END_THRESHOLD else "CONTINUE"`
* 默认阈值 `END_THRESHOLD=0.5`（写成可配置）

### 8.2 backend/vad_silero.py

要求：

* 使用 silero-vad pip 包的加载方式示例（参考 silero README）([GitHub][4])
* 实现一个“流式 VAD”：

  * 每次喂入 chunk（比如 20ms=320 samples 或 32ms=512 samples 都行；建议 512 samples，与你的前端分片一致）
  * 输出：

    * `p_speech`（0~1）
    * `state`（speaking/silence）
    * `silence_ms`（连续静音累计）
* 参数（可配置）：

  * `SPEECH_THRESHOLD=0.5`
  * `STOP_SECS=0.2`（达到后触发 Smart Turn）

> 注：如果 silero-vad 包没有直接暴露“流式 iterator”，就用“对 chunk 做模型前向得到 speech prob”的方式实现；demo 允许用较朴素的阈值法。

### 8.3 backend/app.py（FastAPI）

要求：

* `GET /` 返回 `web/index.html`
* `GET /app.js` 返回 `web/app.js`
* `WS /ws`：

  * 接收 init JSON
  * 接收 binary float32 chunk
  * 调用 SileroVAD 做状态更新，持续向前端发 `vad` 消息
  * 在 `silence_ms >= STOP_SECS*1000` 的首次到达时：

    * 从 turn buffer 取出音频（float32）
    * 调用 SmartTurn `predict()`
    * 发 `smart_turn` 消息
    * 如果 decision END：清空 turn buffer & 重置 silence 计数

并发/性能要求（demo 级）：

* 只需支持单连接（一个浏览器 tab）稳定运行。
* Smart Turn 推断应只在静音候选点触发（不要每个 chunk 都跑），符合上游建议。([GitHub][1])

## 9. 前端（web/index.html + web/app.js）

### 9.1 index.html

* 一个按钮：Start / Stop
* 三块显示区：

  * VAD 状态（speaking/silence、p_speech、silence_ms、turn_ms）
  * Smart Turn 最新结果（scores、decision、latency）
  * 日志滚动区（append 文本）

### 9.2 app.js（必须实现）

要求：

* `navigator.mediaDevices.getUserMedia({audio:true})`
* 使用 WebAudio 把输入变成 **mono 16k float32**

  * 推荐：AudioWorklet（更稳），或 ScriptProcessor（简化但旧）
* 每 512 samples（32ms@16k）打包成 Float32Array，通过 WS binary 发送
* 接收服务端 JSON：

  * `vad`：更新 UI
  * `smart_turn`：更新 UI + END 时给一个明显提示（例如闪一下/追加日志）

## 10. 调参建议（Demo 默认）

* Silero：

  * `chunk_size = 512 samples` (32ms)
  * `speech_threshold = 0.5`
  * `stop_secs = 0.2`（推荐值与 Pipecat 文档一致）([Pipecat][3])
* Smart Turn：

  * `window_secs = 8.0`（固定）
  * `end_threshold = 0.5`（可调）
* 建议在 UI 上暴露 3 个输入框：speech_threshold / stop_secs / end_threshold（可选加分项）

## 11. 验收标准（Acceptance）

* 打开页面点 Start，对着麦克风说话：

  * UI 显示 speaking；停顿时显示 silence 并累积 silence_ms
* 当停顿超过 0.2s：

  * 触发一次 smart_turn 推断，UI 显示 scores 与 decision
* 如果说“我想，嗯……”这种未完成句：

  * Silero 会在“嗯…”后的短暂停顿触发 smart_turn
  * smart_turn 更倾向 CONTINUE（理想效果，取决于阈值/环境）
* 如果说完整句并自然结束：

  * smart_turn 更倾向 END
  * END 后 turn buffer 清空，下一句重新计时

## 12. 参考链接（实现者可查）

* Smart Turn repo：输入格式/8 秒窗口/与 Silero 级联建议 ([GitHub][1])
* HF 模型文件列表（smart-turn-v3.2-cpu/gpu onnx 文件名） ([Hugging Face][2])
* Pipecat Smart Turn + Silero 集成示例与 stop_secs 推荐 ([Pipecat][3])
* Silero VAD pip 用法与 16k 支持 ([GitHub][4])

```

---

如果你希望我把这个 SPEC 再“压缩成 codex 风格的任务清单”（更短、更指令化、逐文件 TODO 列表），我也可以直接给一版更适配自动写码的格式。
::contentReference[oaicite:24]{index=24}

[1]: https://github.com/pipecat-ai/smart-turn "GitHub - pipecat-ai/smart-turn"
[2]: https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main "pipecat-ai/smart-turn-v3 at main"
[3]: https://docs.pipecat.ai/server/utilities/smart-turn/smart-turn-overview "Smart Turn Overview - Pipecat"
[4]: https://github.com/snakers4/silero-vad "GitHub - snakers4/silero-vad: Silero VAD: pre-trained enterprise-grade Voice Activity Detector"

