from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from backend.audio_buffer import TurnBuffer
from backend.smart_turn_onnx import SmartTurnONNX
from backend.vad_silero import SileroVAD

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"
MODEL_DIR = BASE_DIR / "models"

SAMPLE_RATE = 16000
CHUNK_SIZE = 512

app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/app.js")
async def app_js() -> FileResponse:
    return FileResponse(WEB_DIR / "app.js", media_type="application/javascript")


@app.get("/style.css")
async def style_css() -> FileResponse:
    return FileResponse(WEB_DIR / "style.css", media_type="text/css")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    vad = SileroVAD(sample_rate=SAMPLE_RATE)
    model_path = MODEL_DIR / "smart-turn-v3.2-cpu.onnx"
    try:
        smart_turn = SmartTurnONNX(model_path=model_path)
    except FileNotFoundError:
        smart_turn = SmartTurnONNX(model_path=None)

    turn_buffer = TurnBuffer(sample_rate=SAMPLE_RATE)
    turn_active = False

    try:
        init_msg = await ws.receive_text()
        init = json.loads(init_msg)
        if init.get("type") != "init":
            await ws.send_text(json.dumps({"type": "error", "message": "expected init"}))
            await ws.close()
            return
        if init.get("sample_rate") != SAMPLE_RATE or init.get("channels") != 1:
            await ws.send_text(
                json.dumps({
                    "type": "error",
                    "message": "expected 16kHz mono float32 stream",
                })
            )
            await ws.close()
            return
    except Exception as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        await ws.close()
        return

    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes") is None:
                if msg.get("text"):
                    # ignore unexpected text messages
                    continue
                continue

            chunk = np.frombuffer(msg["bytes"], dtype=np.float32)
            if chunk.size == 0:
                continue

            vad_result = vad.process(chunk)
            if vad_result.state == "speaking" and not turn_active:
                turn_active = True
                turn_buffer.clear()

            if turn_active:
                turn_buffer.add(chunk)

            turn_ms = turn_buffer.duration_ms()
            await ws.send_text(
                json.dumps(
                    {
                        "type": "vad",
                        "state": vad_result.state,
                        "p": round(vad_result.p_speech, 4),
                        "silence_ms": vad_result.silence_ms,
                        "turn_ms": turn_ms,
                    }
                )
            )

            if turn_active and vad_result.trigger:
                audio = turn_buffer.get_audio()
                result = smart_turn.predict(audio)
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "smart_turn",
                            "turn_ms": turn_ms,
                            "model": smart_turn.model_path.name,
                            "scores": result["scores"],
                            "decision": result["decision"],
                            "latency_ms": result["latency_ms"],
                        }
                    )
                )
                if result["decision"] == "END":
                    turn_buffer.clear()
                    turn_active = False
                    vad.reset()
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        await ws.close()
