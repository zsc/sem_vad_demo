# Silero VAD + Smart Turn 实时 Demo

这是一个浏览器麦克风实时流 demo：先用 **Silero VAD** 判断说话/静音，再在短暂停顿时调用 **Smart Turn** 做语义 end-of-turn 判定。

## 功能概览
- 浏览器采集麦克风 → WebSocket 发送 16kHz mono float32 PCM
- 后端实时 Silero VAD，静音累计到阈值触发 Smart Turn
- UI 展示 VAD 状态、Smart Turn 分数、实时曲线与频谱

## 目录结构
```
backend/      # FastAPI + WebSocket + VAD/Smart Turn
web/          # 前端 UI + WebAudio
models/       # smart-turn 模型（可选）
```

## 环境要求
- Python 3.10+
- macOS / Windows / Linux

## 安装
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 下载模型（推荐）
如果你已登录 HuggingFace：
```bash
mkdir -p models
huggingface-cli download pipecat-ai/smart-turn-v3 smart-turn-v3.2-cpu.onnx --local-dir ./models
```
未下载也可运行，后端会自动下载到 HF cache。

## 运行
```bash
cd /Users/georgezhou/Downloads/sem_vad_demo
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```
浏览器打开：`http://localhost:8000`

## 使用说明
1. 点击 Start
2. 对着麦克风说话
3. 停顿 ≥ 0.2s 时触发 Smart Turn 推断
4. END 时自动清空 turn buffer，进入下一轮

## 常见问题
### 1) ONNX 输入维度错误
如果看到类似 `Expected 80x800` 的报错，说明模型需要 Whisper 特征。当前实现已内置 `WhisperFeatureExtractor`，请确保已安装 `transformers`。

### 2) torch 安装失败
Silero VAD 依赖 torch。请根据你的平台安装合适版本的 torch 后重试。

## 参数默认值
- Silero VAD：`speech_threshold=0.5`, `stop_secs=0.2`
- Smart Turn：`window_secs=8.0`, `end_threshold=0.5`

需要暴露参数到 UI 或想切换 CoreML / GPU，可继续扩展。
