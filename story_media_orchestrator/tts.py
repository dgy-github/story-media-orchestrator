"""Provider-neutral text-to-speech contracts."""
from __future__ import annotations
import math
import os
import re
import struct
import subprocess
import tempfile
import wave
import zlib
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

COSYVOICE_PRESETS: dict[str, Any] = {
    "narrator": "longshu_v2",
    "default": "longxiaochun_v2",
    "elder_male": "longlaobo",
    "male": ["longcheng_v2", "longshuo_v2"],
    "female": ["longwanjun", "longwan_v2", "longxiaochun_v2", "longxiaoxia_v2"],
    "pool": ["longcheng_v2", "longwanjun", "longshuo_v2", "longwan_v2", "longxiaochun_v2"],
}

_LEGACY_V1_VOICE_STEMS = frozenset(
    {"longshu", "longxiaochun", "longcheng", "longshuo", "longjielidou", "longxiaoxia", "longmiao", "longwan"}
)
_SPEAKER_SUFFIX_STRIP_RE = re.compile(r"[\(（][^\)）]*[\)）]\s*$")
_ELDER_VOICE_HINTS = ("老", "伯", "爷", "长老", "族老", "村正", "黄羊", "spirit_beast")

EDGE_NEURAL_PRESETS: dict[str, Any] = {
    "narrator": "zh-CN-YunjianNeural|rate=-4%|pitch=-10Hz|volume=+6%",
    "default": "zh-CN-XiaoxiaoNeural",
    "male": [
        "zh-CN-YunxiNeural|rate=+0%|pitch=+2Hz|volume=+10%",
        "zh-CN-YunyangNeural|rate=-4%|pitch=-6Hz|volume=+10%",
        "zh-CN-YunjianNeural|rate=-8%|pitch=-16Hz|volume=+14%",
    ],
    "female": [
        "zh-CN-XiaoxiaoNeural|rate=-10%|pitch=-12Hz|volume=+8%",
        "zh-CN-XiaoxiaoNeural|rate=-6%|pitch=-8Hz|volume=+8%",
    ],
    "pool": [
        "zh-CN-YunxiNeural",
        "zh-CN-XiaoxiaoNeural",
        "zh-CN-YunyangNeural",
        "zh-CN-XiaoyiNeural",
        "zh-CN-YunjianNeural",
    ],
}

_COSY_TO_EDGE_VOICE_MAP: dict[str, str] = {
    "longshu": "zh-CN-YunjianNeural|rate=-4%|pitch=-10Hz|volume=+6%",
    "longshu_v2": "zh-CN-YunjianNeural|rate=-4%|pitch=-10Hz|volume=+6%",
    "longcheng": "zh-CN-YunxiNeural|rate=+0%|pitch=+2Hz|volume=+10%",
    "longcheng_v2": "zh-CN-YunxiNeural|rate=+0%|pitch=+2Hz|volume=+10%",
    "longshuo": "zh-CN-YunjianNeural|rate=-6%|pitch=-14Hz|volume=+15%",
    "longshuo_v2": "zh-CN-YunjianNeural|rate=-6%|pitch=-14Hz|volume=+15%",
    "longjielidou": "zh-CN-YunxiNeural|rate=+2%|pitch=+0Hz|volume=+8%",
    "longjielidou_v2": "zh-CN-YunxiNeural|rate=+2%|pitch=+0Hz|volume=+8%",
    "longxiaochun": "zh-CN-XiaoxiaoNeural|rate=-3%|pitch=+0Hz|volume=+8%",
    "longxiaochun_v2": "zh-CN-XiaoxiaoNeural|rate=-3%|pitch=+0Hz|volume=+8%",
    "longxiaoxia": "zh-CN-XiaoyiNeural|rate=-2%|pitch=+2Hz|volume=+8%",
    "longxiaoxia_v2": "zh-CN-XiaoyiNeural|rate=-2%|pitch=+2Hz|volume=+8%",
    "longmiao": "zh-CN-XiaoxiaoNeural|rate=-4%|pitch=-2Hz|volume=+8%",
    "longmiao_v2": "zh-CN-XiaoxiaoNeural|rate=-4%|pitch=-2Hz|volume=+8%",
    "longwan": "zh-CN-XiaoxiaoNeural",
    "longwan_v2": "zh-CN-XiaoxiaoNeural",
    "longwanjun": "zh-CN-XiaoyiNeural|rate=-3%|pitch=+0Hz|volume=+8%",
    "longlaobo": "zh-CN-YunjianNeural|rate=-10%|pitch=-18Hz|volume=+12%",
}

_SPEECH_MODES = {"auto", "full_text", "dialogue_only", "narrator_and_dialogue"}


_FEMALE_VOICE_HINTS = ("female", "woman", "girl", "女", "少女", "小姐", "妇人", "女孩", "姐姐", "妹妹", "母亲", "阿姨", "侍女", "巫女")
_MALE_VOICE_HINTS = ("male", "man", "boy", "男", "少年", "先生", "老人", "老者", "老伯", "男孩", "哥哥", "弟弟", "父亲", "叔叔", "将军", "书生", "修钟匠")


def _match_character_entry(
    speaker: str,
    character_bible: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not character_bible or not speaker:
        return None
    target = speaker.strip().lower()
    if not target:
        return None
    for key, entry in character_bible.items():
        if not isinstance(entry, dict):
            continue
        aliases = {str(a).strip().lower() for a in (entry.get("aliases") or ()) if str(a).strip()}
        names = {
            str(key).strip().lower(),
            str(entry.get("name") or "").strip().lower(),
            str(entry.get("node_id") or "").strip().lower(),
            *aliases,
        } - {""}
        if target in names:
            return entry
    return None


def _infer_character_gender(
    speaker: str,
    character_bible: dict[str, dict[str, Any]] | None,
) -> str | None:
    matched_profile = _match_character_entry(speaker, character_bible)
    if matched_profile is None:
        return None
    blob = " ".join(
        str(matched_profile.get(k) or "")
        for k in ("gender", "role", "appearance", "visual_dna")
    ).lower()
    if not blob.strip():
        return None
    from .character_bible import _hint_matches

    is_female = _hint_matches(blob, _FEMALE_VOICE_HINTS)
    is_male = _hint_matches(blob, _MALE_VOICE_HINTS)
    if is_female and not is_male:
        return "female"
    if is_male and not is_female:
        return "male"
    return None


def resolve_role_voice(
    speaker: str | None,
    voice_cast: dict[str, str] | None = None,
    shot_voice: str | None = None,
    default_voice: str = "longxiaochun_v2",
    *,
    character_bible: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Deterministically map a narrator or character speaker name to a CosyVoice voice id."""
    cast = voice_cast or {}
    raw_name = (speaker or "").strip()
    base_name = _SPEAKER_SUFFIX_STRIP_RE.sub("", raw_name).strip()
    if base_name in ("", "旁白", "画外音", "narrator", "Narrator"):
        return (
            cast.get(raw_name)
            or cast.get(base_name)
            or cast.get("旁白")
            or cast.get("narrator")
            or shot_voice
            or COSYVOICE_PRESETS["narrator"]
        )
    for cand in (raw_name, base_name, base_name.lower()):
        if cand in cast and cast[cand]:
            return cast[cand]
    matched_entry = _match_character_entry(base_name, character_bible)
    if matched_entry is not None:
        for field_key in ("node_id", "name", "key"):
            fk = str(matched_entry.get(field_key) or "").strip()
            if fk and fk in cast and cast[fk]:
                return cast[fk]
        for alias in matched_entry.get("aliases") or ():
            ak = str(alias).strip()
            if ak and ak in cast and cast[ak]:
                return cast[ak]
    if shot_voice:
        return shot_voice
    elder_blob = f"{base_name} {(matched_entry or {}).get('role', '')} {(matched_entry or {}).get('appearance', '')}".lower()
    gender = _infer_character_gender(base_name, character_bible)
    if gender != "female" and any(h in elder_blob for h in _ELDER_VOICE_HINTS):
        return str(COSYVOICE_PRESETS.get("elder_male") or "longlaobo")
    if gender == "female" and COSYVOICE_PRESETS.get("female"):
        pool = COSYVOICE_PRESETS["female"]
    elif gender == "male" and COSYVOICE_PRESETS.get("male"):
        pool = COSYVOICE_PRESETS["male"]
    else:
        pool = COSYVOICE_PRESETS["pool"]
    if not pool:
        return default_voice
    canonical = str((matched_entry or {}).get("name") or base_name)
    return pool[zlib.crc32(canonical.encode("utf-8")) % len(pool)]

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


class CosyVoiceTTSProvider:
    """Alibaba DashScope CosyVoice TTS provider outputting WAV audio."""

    supports_roles = True

    def __init__(
        self,
        *,
        model: str = "cosyvoice-v2",
        voice: str = "longxiaochun",
        api_key: str | None = None,
        speech_rate: float = 1.0,
        timeout_millis: int = 30000,
    ) -> None:
        self.model = model
        self.voice = voice
        self.api_key = api_key
        self.speech_rate = speech_rate
        self.timeout_millis = timeout_millis

    def _resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if env_key:
            return env_key
        try:
            from .runtime import resolve_sibling_root, _import_from_root

            image_root = Path(
                os.environ.get("STORY_IMAGE_AGENT_ROOT")
                or resolve_sibling_root("story-image-agent", "microcodex-story-image-agent", module_name="story_image_agent")
            )
            pkg = _import_from_root(image_root, "story_image_agent")
            provider = pkg.DashScopeImageProvider.from_nanocodex_config()
            if getattr(provider, "api_key", None):
                return provider.api_key
        except Exception:
            pass
        raise RuntimeError("Missing DASHSCOPE_API_KEY for CosyVoice TTS")

    @staticmethod
    def _fix_riff_sizes(data: bytes) -> bytes:
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data
        import struct
        buf = bytearray(data)
        struct.pack_into("<I", buf, 4, len(buf) - 8)
        pos = 12
        while pos + 8 <= len(buf):
            chunk_id = bytes(buf[pos:pos + 4])
            chunk_size = struct.unpack_from("<I", buf, pos + 4)[0]
            if chunk_id == b"data":
                actual_size = len(buf) - (pos + 8)
                if chunk_size != actual_size:
                    struct.pack_into("<I", buf, pos + 4, actual_size)
                break
            pos += 8 + chunk_size + (chunk_size & 1)
        return bytes(buf)

    @staticmethod
    def _normalize_voice(model: str, voice: str) -> str:
        clean = (voice or "").strip()
        if "|" in clean or clean.startswith("zh-"):
            head = clean.split("|", 1)[0].strip()
            if "Yunxi" in head:
                clean = "longcheng_v2"
            elif "Yunjian" in head:
                clean = "longshuo_v2"
            elif "Xiaoxiao" in head or "Xiaoyi" in head:
                clean = "longwanjun"
            else:
                clean = "longshu_v2"
        if model.startswith("cosyvoice-v2") and clean in _LEGACY_V1_VOICE_STEMS:
            return f"{clean}_v2"
        return clean

    def synthesize(self, text: str, output: str | Path, *, voice: str | None = None) -> Path:
        import time
        import dashscope
        from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        dashscope.api_key = self._resolve_api_key()
        resolved_voice = self._normalize_voice(self.model, voice or self.voice)
        audio_bytes = b""
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                if attempt > 0:
                    time.sleep(1.5 * attempt)
                else:
                    time.sleep(0.3)
                synthesizer = SpeechSynthesizer(
                    model=self.model,
                    voice=resolved_voice,
                    format=AudioFormat.WAV_22050HZ_MONO_16BIT,
                    speech_rate=self.speech_rate,
                )
                audio_bytes = synthesizer.call(text, timeout_millis=self.timeout_millis)
                if audio_bytes:
                    break
            except Exception as exc:
                last_exc = exc
        if not audio_bytes:
            if last_exc is not None:
                raise last_exc
            raise ValueError("CosyVoice returned empty audio")
        temporary = target.with_name(f".{target.stem}.{uuid4().hex}.wav")
        try:
            if audio_bytes[:4] == b"RIFF":
                temporary.write_bytes(self._fix_riff_sizes(audio_bytes))
            else:
                with wave.open(str(temporary), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(22050)
                    wav_file.writeframes(audio_bytes)
            with wave.open(str(temporary), "rb") as wav_file:
                if wav_file.getnframes() == 0:
                    raise ValueError("CosyVoice returned empty WAV frames")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target


class EdgeNeuralTTSProvider:
    """Multi-role Microsoft Azure Neural TTS provider (`edge-tts`) with studio WAV mastering."""

    supports_roles = True

    def __init__(
        self,
        *,
        voice: str = "zh-CN-YunyangNeural",
        sample_rate: int = 24000,
        studio_master: bool = True,
    ) -> None:
        self.voice = voice
        self.sample_rate = sample_rate
        self.studio_master = studio_master

    @staticmethod
    def _parse_voice_spec(spec: str, text: str = "") -> tuple[str, str, str, str]:
        raw = (spec or "").strip()
        if raw in _COSY_TO_EDGE_VOICE_MAP:
            raw = _COSY_TO_EDGE_VOICE_MAP[raw]
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        voice_name = parts[0] if parts else "zh-CN-YunyangNeural"
        if not voice_name.startswith(("zh-", "en-", "ja-", "ko-")):
            voice_name = _COSY_TO_EDGE_VOICE_MAP.get(voice_name, "zh-CN-YunyangNeural").split("|")[0]
        rate = "+0%"
        pitch = "+0Hz"
        volume = "+0%"
        for part in parts[1:]:
            if "=" not in part:
                continue
            k, v = [x.strip() for x in part.split("=", 1)]
            if k == "rate" and v:
                rate = v
            elif k == "pitch" and v:
                pitch = v
            elif k == "volume" and v:
                volume = v
        # Subtle intensity lift on exclamatory battle lines when not already maxed
        stripped = (text or "").strip()
        if stripped.endswith(("！", "!")) and volume == "+0%":
            volume = "+10%"
        return voice_name, rate, pitch, volume

    @staticmethod
    def _resolve_ffmpeg() -> str:
        import shutil

        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        from .local_media import find_ffmpeg

        return find_ffmpeg()

    def synthesize(self, text: str, output: str | Path, *, voice: str | None = None) -> Path:
        import asyncio
        import subprocess
        import time
        import edge_tts

        target = Path(output).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        voice_name, rate, pitch, volume = self._parse_voice_spec(voice or self.voice, text=text)
        temp_mp3 = target.with_name(f".{target.stem}.{uuid4().hex}.mp3")
        temp_wav = target.with_name(f".{target.stem}.{uuid4().hex}.wav")

        async def _save_mp3() -> None:
            comm = edge_tts.Communicate(
                text=text,
                voice=voice_name,
                rate=rate,
                pitch=pitch,
                volume=volume,
            )
            await comm.save(str(temp_mp3))

        last_exc: Exception | None = None
        try:
            for attempt in range(3):
                try:
                    asyncio.run(_save_mp3())
                    if temp_mp3.is_file() and temp_mp3.stat().st_size > 128:
                        break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time.sleep(0.8 * (attempt + 1))
            if not temp_mp3.is_file() or temp_mp3.stat().st_size <= 128:
                if last_exc is not None:
                    raise last_exc
                raise ValueError("EdgeNeuralTTSProvider returned empty audio")

            ffmpeg = self._resolve_ffmpeg()
            af_chain = (
                "highpass=f=65,acompressor=threshold=-19dB:ratio=2.4:attack=8:release=140,"
                "loudnorm=I=-16:TP=-1.5:LRA=11"
            )
            cmd = [ffmpeg, "-y", "-i", str(temp_mp3)]
            if self.studio_master:
                cmd += ["-af", af_chain]
            cmd += ["-ar", str(self.sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(temp_wav)]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode != 0:
                subprocess.run(
                    [ffmpeg, "-y", "-i", str(temp_mp3), "-ar", str(self.sample_rate), "-ac", "1", "-c:a", "pcm_s16le", str(temp_wav)],
                    check=True,
                    capture_output=True,
                )
            with wave.open(str(temp_wav), "rb") as wav_file:
                if wav_file.getnframes() == 0:
                    raise ValueError("EdgeNeuralTTSProvider produced empty WAV frames")
            temp_wav.replace(target)
        finally:
            temp_mp3.unlink(missing_ok=True)
            temp_wav.unlink(missing_ok=True)
        return target


def _read_wav_pcm(path: Path) -> tuple[tuple[int, int, int], bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        params = (wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate())
        frames = wav_file.readframes(wav_file.getnframes())
        duration = wav_file.getnframes() / wav_file.getframerate() if wav_file.getframerate() else 0.0
    return params, frames, duration


def _detect_video_speech_bursts(video_path: Path) -> tuple[float, list[tuple[float, float]]]:
    """Detect native speech mouth-movement intervals (seconds) from a rendered video's audio track."""
    from .local_media import find_ffmpeg

    ffmpeg = find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="h3_burst_") as td:
        probe_wav = Path(td) / "probe.wav"
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(probe_wav)],
            capture_output=True,
        )
        if proc.returncode != 0 or not probe_wav.is_file():
            return 0.0, []
        with wave.open(str(probe_wav), "rb") as w:
            sr = w.getframerate() or 16000
            nframes = w.getnframes()
            if nframes < sr // 2:
                return 0.0, []
            vid_dur = nframes / sr
            raw = w.readframes(nframes)
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    win = 800  # 50ms at 16kHz
    rms_vals: list[float] = []
    for idx in range(0, len(samples), win):
        chunk = samples[idx : idx + win]
        if not chunk:
            break
        rms_vals.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)))
    if not rms_vals:
        return vid_dur, []
    sorted_rms = sorted(rms_vals)
    noise_floor = sorted_rms[min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.25))]
    peak_rms = sorted_rms[-1]
    if peak_rms < 1400.0 or peak_rms < noise_floor * 1.8:
        return vid_dur, []
    thresh = max(950.0, noise_floor * 2.0, peak_rms * 0.22)
    raw_spans: list[list[float]] = []
    step_sec = win / 16000.0
    for i, r in enumerate(rms_vals):
        if r >= thresh:
            s_t = i * step_sec
            e_t = min(vid_dur, (i + 1) * step_sec)
            if raw_spans and s_t - raw_spans[-1][1] <= 0.36:
                raw_spans[-1][1] = e_t
            else:
                raw_spans.append([s_t, e_t])
    bursts = [(max(0.0, s - 0.04), min(vid_dur, e + 0.08)) for s, e in raw_spans if (e - s) >= 0.26]
    return vid_dur, bursts


def _fit_wav_tempo(src_wav: Path, dst_wav: Path, target_dur: float, framerate: int) -> None:
    """Gently time-fit a WAV segment to target_dur using FFmpeg atempo within natural bounds [0.80, 1.34]."""
    _, _, cur_dur = _read_wav_pcm(src_wav)
    if cur_dur <= 0.05 or target_dur <= 0.05:
        dst_wav.write_bytes(src_wav.read_bytes())
        return
    ratio = max(0.80, min(1.34, cur_dur / target_dur))
    if abs(ratio - 1.0) < 0.04:
        dst_wav.write_bytes(src_wav.read_bytes())
        return
    from .local_media import find_ffmpeg

    ffmpeg = find_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src_wav),
            "-af",
            f"atempo={ratio:.4f}",
            "-ar",
            str(framerate),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(dst_wav),
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not dst_wav.is_file():
        dst_wav.write_bytes(src_wav.read_bytes())


def _try_align_dialogue_to_video_bursts(
    tts: Any,
    shot: Any,
    speaker: str,
    text: str,
    role_voice: str,
    target: Path,
    *,
    supports_roles: bool,
    character_bible: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    """Align on-screen dialogue clauses to detected mouth-movement bursts in an already-rendered video."""
    try:
        if isinstance(tts, FakeTTSProvider):
            return None
        assets = getattr(shot, "assets", None) or {}
        vid_str = assets.get("video")
        if not vid_str or not Path(vid_str).is_file():
            return None
        from .lipsync import classify_shot_speech_delivery

        if classify_shot_speech_delivery(shot, character_bible) != "on_screen_dialogue":
            return None
        vid_dur, bursts = _detect_video_speech_bursts(Path(vid_str))
        if vid_dur < 1.5 or not bursts:
            return None

        clauses = [c.strip() for c in re.split(r"(?<=[，,。！？；：!?;])", text) if c and c.strip()]
        if not clauses:
            clauses = [text]

        # Group clauses into k buckets matching detected speech bursts
        if len(bursts) >= 2 and len(clauses) >= 2:
            k = min(len(bursts), len(clauses))
            # Pick top-k longest/merged bursts in chronological order if more bursts than clauses
            active_bursts = bursts[:k] if len(bursts) == k else None
            if active_bursts is None:
                merged_b = [list(b) for b in bursts]
                while len(merged_b) > k:
                    # Merge closest adjacent pair
                    best_i = min(range(len(merged_b) - 1), key=lambda idx: merged_b[idx + 1][0] - merged_b[idx][1])
                    merged_b[best_i][1] = merged_b[best_i + 1][1]
                    merged_b.pop(best_i + 1)
                active_bursts = [(b[0], b[1]) for b in merged_b]
            b_weights = [max(0.2, e - s) for s, e in active_bursts]
            b_total = sum(b_weights)
            c_lens = [max(1, len(re.sub(r"[\s，,。！？；：!?;]", "", c))) for c in clauses]
            c_total = sum(c_lens)
            grouped_texts: list[str] = []
            c_cursor = 0
            acc_c = 0.0
            acc_b = 0.0
            for b_idx in range(k):
                if b_idx == k - 1:
                    grouped_texts.append("".join(clauses[c_cursor:]))
                    break
                acc_b += b_weights[b_idx] / b_total
                start_c = c_cursor
                while c_cursor < len(clauses) - (k - 1 - b_idx):
                    next_acc = acc_c + (c_lens[c_cursor] / c_total)
                    c_cursor += 1
                    acc_c = next_acc
                    if acc_c >= acc_b * 0.92:
                        break
                if c_cursor == start_c:
                    c_cursor += 1
                grouped_texts.append("".join(clauses[start_c:c_cursor]))
            plan = list(zip(active_bursts, grouped_texts))
        else:
            span = (bursts[0][0], bursts[-1][1])
            plan = [(span, text)]

        temp_files: list[Path] = []
        try:
            synth_parts: list[tuple[float, float, str, Path]] = []
            base_params: tuple[int, int, int] | None = None
            for idx, ((b_s, b_e), seg_txt) in enumerate(plan):
                raw_p = target.with_name(f".{target.stem}.burst_raw{idx}.{uuid4().hex}.wav")
                fit_p = target.with_name(f".{target.stem}.burst_fit{idx}.{uuid4().hex}.wav")
                temp_files.extend([raw_p, fit_p])
                out_p = Path(tts.synthesize(seg_txt, raw_p, voice=role_voice)) if supports_roles else Path(tts.synthesize(seg_txt, raw_p))
                if out_p != raw_p:
                    temp_files.append(out_p)
                params, _, _ = _read_wav_pcm(out_p)
                if base_params is None:
                    base_params = params
                _fit_wav_tempo(out_p, fit_p, max(0.35, b_e - b_s), base_params[2])
                synth_parts.append((b_s, b_e, seg_txt, fit_p))

            assert base_params is not None
            nch, sw, sr = base_params
            frame_bytes = nch * sw
            total_frames = max(1, int(round(vid_dur * sr)))
            canvas = bytearray(b"\0" * (total_frames * frame_bytes))
            sub_caps: list[dict[str, Any]] = []
            min_next_frame = 0
            for b_s, _b_e, seg_txt, fit_p in synth_parts:
                _, pcm_bytes, seg_dur = _read_wav_pcm(fit_p)
                seg_frames = len(pcm_bytes) // frame_bytes
                start_frame = max(min_next_frame, int(round(b_s * sr)))
                end_frame = start_frame + seg_frames
                if end_frame > total_frames:
                    total_frames = end_frame
                    canvas.extend(b"\0" * ((total_frames * frame_bytes) - len(canvas)))
                byte_off = start_frame * frame_bytes
                canvas[byte_off : byte_off + len(pcm_bytes)] = pcm_bytes
                actual_s = start_frame / sr
                actual_e = end_frame / sr
                sub_caps.append(
                    {
                        "speaker": speaker,
                        "text": seg_txt,
                        "start": round(actual_s, 3),
                        "end": round(actual_e, 3),
                        "voice": role_voice,
                    }
                )
                min_next_frame = end_frame + int(0.06 * sr)

            out_tmp = target.with_name(f".{target.stem}.aligned.{uuid4().hex}.wav")
            temp_files.append(out_tmp)
            with wave.open(str(out_tmp), "wb") as w_out:
                w_out.setnchannels(nch)
                w_out.setsampwidth(sw)
                w_out.setframerate(sr)
                w_out.writeframes(bytes(canvas))
            out_tmp.replace(target)
            if hasattr(shot, "sub_captions"):
                shot.sub_captions = sub_caps
            return target
        finally:
            for tf in temp_files:
                tf.unlink(missing_ok=True)
    except Exception:
        return None


def _resolve_ffmpeg_bin() -> str:
    import shutil

    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return str(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return "ffmpeg"


def _apply_inner_monologue_fx(wav_path: Path, speaker: str, tts: Any) -> Path:
    """Apply subtle spatial reverb/bandpass to inner monologue or telepathy lines."""
    if isinstance(tts, FakeTTSProvider):
        return wav_path
    from .cinematic_rules import is_inner_monologue_speaker

    if not is_inner_monologue_speaker(speaker):
        return wav_path
    tmp_fx = wav_path.with_name(f".{wav_path.stem}.monologue_fx.{uuid4().hex}.wav")
    try:
        proc = subprocess.run(
            [
                _resolve_ffmpeg_bin(),
                "-y",
                "-v",
                "error",
                "-i",
                str(wav_path),
                "-af",
                "highpass=f=220,lowpass=f=3800,aecho=0.85:0.75:45|90:0.22|0.10",
                str(tmp_fx),
            ],
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0 and tmp_fx.is_file() and tmp_fx.stat().st_size > 44:
            tmp_fx.replace(wav_path)
    except OSError:
        pass
    finally:
        tmp_fx.unlink(missing_ok=True)
    return wav_path


def synthesize_shot_audio(
    tts: Any,
    shot: Any,
    output: str | Path,
    *,
    speech_mode: str = "auto",
    voice_cast: dict[str, str] | None = None,
    pause_seconds: float = 0.18,
    character_bible: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Synthesize shot speech with optional role separation and sub-caption timings."""
    if speech_mode not in _SPEECH_MODES:
        raise ValueError(f"unsupported speech_mode: {speech_mode}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    supports_roles = bool(isinstance(tts, CosyVoiceTTSProvider) or getattr(tts, "supports_roles", False))
    dialogue_lines = getattr(shot, "dialogue_lines", None) or []

    if speech_mode == "full_text" or (speech_mode == "auto" and not (dialogue_lines and supports_roles)):
        if hasattr(shot, "sub_captions"):
            shot.sub_captions = []
        if supports_roles:
            narrator_voice = resolve_role_voice(
                "旁白", voice_cast, getattr(shot, "voice", None), character_bible=character_bible
            )
            return Path(tts.synthesize(shot.text, target, voice=narrator_voice))
        return Path(tts.synthesize(shot.text, target))

    utterances: list[tuple[str, str]] = []
    if speech_mode == "dialogue_only":
        utterances = [
            (str(line.get("speaker") or "角色"), str(line.get("text") or "").strip())
            for line in dialogue_lines
            if isinstance(line, dict) and str(line.get("text") or "").strip()
        ]
    else:
        narration = (getattr(shot, "narration", None) or "").strip()
        if narration:
            utterances.append(("旁白", narration))
        for line in dialogue_lines:
            if isinstance(line, dict) and str(line.get("text") or "").strip():
                utterances.append((str(line.get("speaker") or "角色"), str(line["text"]).strip()))

    if not utterances:
        if hasattr(shot, "sub_captions"):
            shot.sub_captions = []
        if supports_roles:
            narrator_voice = resolve_role_voice(
                "旁白", voice_cast, getattr(shot, "voice", None), character_bible=character_bible
            )
            return Path(tts.synthesize(shot.text, target, voice=narrator_voice))
        return Path(tts.synthesize(shot.text, target))

    if len(utterances) == 1:
        speaker, text = utterances[0]
        role_voice = resolve_role_voice(
            speaker, voice_cast, getattr(shot, "voice", None), character_bible=character_bible
        )
        aligned = _try_align_dialogue_to_video_bursts(
            tts,
            shot,
            speaker,
            text,
            role_voice,
            target,
            supports_roles=supports_roles,
            character_bible=character_bible,
        )
        if aligned is not None:
            return aligned
        result = Path(tts.synthesize(text, target, voice=role_voice)) if supports_roles else Path(tts.synthesize(text, target))
        result = _apply_inner_monologue_fx(result, speaker, tts)
        _, _, dur = _read_wav_pcm(result)
        if hasattr(shot, "sub_captions"):
            shot.sub_captions = [{"speaker": speaker, "text": text, "start": 0.0, "end": round(dur, 3), "voice": role_voice}]
        return result

    temp_parts: list[Path] = []
    sub_captions: list[dict[str, Any]] = []
    combined_frames = bytearray()
    base_params: tuple[int, int, int] | None = None
    cursor = 0.0
    try:
        for index, (speaker, seg_text) in enumerate(utterances):
            part = target.with_name(f".{target.stem}.part{index:02d}.{uuid4().hex}.wav")
            temp_parts.append(part)
            role_voice = resolve_role_voice(
                speaker, voice_cast, getattr(shot, "voice", None), character_bible=character_bible
            )
            actual_part = Path(tts.synthesize(seg_text, part, voice=role_voice)) if supports_roles else Path(tts.synthesize(seg_text, part))
            actual_part = _apply_inner_monologue_fx(actual_part, speaker, tts)
            if actual_part != part:
                temp_parts.append(actual_part)
            params, frames, dur = _read_wav_pcm(actual_part)
            if base_params is None:
                base_params = params
            elif params != base_params:
                raise ValueError(f"Mismatched WAV format across dialogue segments: {params} != {base_params}")
            if index > 0 and pause_seconds > 0:
                nchannels, sampwidth, framerate = base_params
                pause_frames = max(0, int(round(pause_seconds * framerate)))
                combined_frames.extend(b"\0" * (pause_frames * nchannels * sampwidth))
                cursor += pause_frames / framerate
            combined_frames.extend(frames)
            sub_captions.append({
                "speaker": speaker,
                "text": seg_text,
                "start": round(cursor, 3),
                "end": round(cursor + dur, 3),
                "voice": role_voice,
            })
            cursor += dur
        assert base_params is not None
        temporary = target.with_name(f".{target.stem}.{uuid4().hex}.wav")
        temp_parts.append(temporary)
        with wave.open(str(temporary), "wb") as out_wav:
            out_wav.setnchannels(base_params[0])
            out_wav.setsampwidth(base_params[1])
            out_wav.setframerate(base_params[2])
            out_wav.writeframes(bytes(combined_frames))
        temporary.replace(target)
        if hasattr(shot, "sub_captions"):
            shot.sub_captions = sub_captions
        return target
    finally:
        for temp in temp_parts:
            temp.unlink(missing_ok=True)
