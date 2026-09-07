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
        seconds = max(3, math.ceil(len(text) / 12))
        with wave.open(str(target), "wb") as audio:
            audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000 * seconds)
        return target


class WindowsTTSProvider:
    """Offline spoken WAV output using an installed Windows desktop voice."""

    def synthesize(self, text: str, output: str | Path) -> Path:
        import json
        import subprocess
        from uuid import uuid4
        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{uuid4().hex}.wav')
        # Text and filenames are data on stdin, never PowerShell source.
        script = '''
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
Add-Type -AssemblyName System.Speech
$speech = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $culture = if ($payload.text -match '[\\u4e00-\\u9fff]') { 'zh-CN' } else { 'en-US' }
  $voice = $speech.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq $culture } | Select-Object -First 1
  if ($null -eq $voice) { throw "No installed speech voice for $culture" }
  $speech.SelectVoice($voice.VoiceInfo.Name)
  $speech.SetOutputToWaveFile($payload.output)
  $speech.Speak($payload.text)
} finally { $speech.Dispose() }
'''
        try:
            subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                           input=json.dumps({'text': text, 'output': str(temporary)}, ensure_ascii=True).encode(),
                           capture_output=True, check=True, timeout=120,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            with wave.open(str(temporary), 'rb') as audio:
                if audio.getnframes() == 0:
                    raise ValueError('Speech provider returned empty audio')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target
