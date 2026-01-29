from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from silero_vad import load_silero_vad
except Exception as exc:  # pragma: no cover - dependency errors
    raise RuntimeError("silero-vad and torch are required to run the VAD") from exc


@dataclass
class VADResult:
    p_speech: float
    state: str
    silence_ms: int
    trigger: bool


class SileroVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        speech_threshold: float = 0.5,
        stop_secs: float = 0.2,
    ) -> None:
        self.sample_rate = sample_rate
        self.speech_threshold = speech_threshold
        self.stop_ms = int(stop_secs * 1000)
        self.model = load_silero_vad()
        self.model.eval()
        self.silence_ms = 0
        self._silence_reached = False

    def reset(self) -> None:
        self.silence_ms = 0
        self._silence_reached = False

    def process(self, chunk_f32: np.ndarray) -> VADResult:
        if chunk_f32.size == 0:
            return VADResult(0.0, "silence", self.silence_ms, False)

        audio = torch.from_numpy(chunk_f32.astype(np.float32, copy=False))
        with torch.no_grad():
            try:
                p = self.model(audio, self.sample_rate)
            except TypeError:
                p = self.model(audio)
        if isinstance(p, torch.Tensor):
            p_speech = float(p.detach().cpu().item())
        else:
            p_speech = float(p)

        state = "speaking" if p_speech >= self.speech_threshold else "silence"
        trigger = False
        chunk_ms = int(chunk_f32.size / self.sample_rate * 1000)
        if state == "speaking":
            self.silence_ms = 0
            self._silence_reached = False
        else:
            self.silence_ms += chunk_ms
            if self.silence_ms >= self.stop_ms and not self._silence_reached:
                trigger = True
                self._silence_reached = True
        return VADResult(p_speech, state, self.silence_ms, trigger)
