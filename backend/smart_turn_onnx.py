from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - optional dependency
    hf_hub_download = None


class SmartTurnONNX:
    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        repo_id: str = "pipecat-ai/smart-turn-v3",
        filename: str = "smart-turn-v3.2-cpu.onnx",
        sample_rate: int = 16000,
        window_secs: float = 8.0,
        end_threshold: float = 0.5,
        end_index: int = -1,
        providers: Optional[list[str]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_secs = window_secs
        self.end_threshold = end_threshold
        self.end_index = end_index

        if model_path is None:
            if hf_hub_download is None:
                raise RuntimeError("huggingface_hub is not available to download the model")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Smart Turn model not found: {model_path}")

        self.model_path = model_path
        sess_opts = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers or ort.get_available_providers(),
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=int(self.window_secs))

    def prepare_8s_input(self, wav_f32: np.ndarray) -> np.ndarray:
        target_len = int(self.window_secs * self.sample_rate)
        wav = wav_f32.astype(np.float32, copy=False).reshape(-1)
        if wav.size > target_len:
            wav = wav[-target_len:]
        elif wav.size < target_len:
            pad = np.zeros(target_len - wav.size, dtype=np.float32)
            wav = np.concatenate([pad, wav])
        return wav

    def _make_features(self, wav: np.ndarray) -> np.ndarray:
        inputs = self.feature_extractor(
            wav,
            sampling_rate=self.sample_rate,
            return_tensors="np",
            padding="max_length",
            max_length=int(self.window_secs * self.sample_rate),
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        return np.expand_dims(input_features, axis=0)

    def _parse_outputs(self, outputs: list[np.ndarray]) -> Dict[str, float]:
        out = np.array(outputs[0]).astype(np.float32).reshape(-1)
        end = float(out[0]) if out.size else 0.0
        return {"end": end, "continue": 1.0 - end}

    def predict(self, wav_f32: np.ndarray) -> Dict[str, object]:
        start = time.perf_counter()
        wav = self.prepare_8s_input(wav_f32)
        features = self._make_features(wav)
        outputs = self.session.run(None, {self.input_name: features})
        latency_ms = int((time.perf_counter() - start) * 1000)
        scores = self._parse_outputs(outputs)
        decision = "END" if scores["end"] >= self.end_threshold else "CONTINUE"
        return {"scores": scores, "decision": decision, "latency_ms": latency_ms}
