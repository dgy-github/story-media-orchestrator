"""Provider-neutral text-to-speech contracts."""
from __future__ import annotations
import math, wave
from pathlib import Path
from typing import Protocol

class TTSProvider(Protocol):
    def synthesize(self, text: str, output: str | Path) -> Path: ...

class FakeTTSProvider:
    """Deterministic silent WAV provider for tests and local previews."""
    def synthesize(self, text: str, output: str | Path) -> Path:
        target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
        seconds = max(1, math.ceil(len(text) / 12))
        with wave.open(str(target), "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000 * seconds)
        return target
