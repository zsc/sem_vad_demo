from __future__ import annotations

from typing import List, Optional

import numpy as np


class TurnBuffer:
    def __init__(self, sample_rate: int = 16000, max_seconds: Optional[float] = None) -> None:
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self._chunks: List[np.ndarray] = []
        self._num_samples = 0

    def add(self, chunk_f32: np.ndarray) -> None:
        if chunk_f32.size == 0:
            return
        chunk = chunk_f32.astype(np.float32, copy=False)
        self._chunks.append(chunk)
        self._num_samples += int(chunk.size)
        if self.max_seconds is not None:
            max_samples = int(self.max_seconds * self.sample_rate)
            if self._num_samples > max_samples:
                self._trim_left(self._num_samples - max_samples)

    def _trim_left(self, num_samples: int) -> None:
        while num_samples > 0 and self._chunks:
            head = self._chunks[0]
            if head.size <= num_samples:
                self._chunks.pop(0)
                self._num_samples -= int(head.size)
                num_samples -= int(head.size)
            else:
                self._chunks[0] = head[num_samples:]
                self._num_samples -= num_samples
                num_samples = 0

    def clear(self) -> None:
        self._chunks = []
        self._num_samples = 0

    def duration_ms(self) -> int:
        if self.sample_rate == 0:
            return 0
        return int(self._num_samples / self.sample_rate * 1000)

    def get_audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32, copy=False)
