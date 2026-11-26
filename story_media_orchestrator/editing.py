"""Automated short-drama video editing utilities: canvas, subtitles, BGM ducking, and transitions."""
from __future__ import annotations

import math
import re
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

CANVAS_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1080, 1080),
}
AUTO_ASPECT_IMAGE_SIZES: dict[str, str] = {
    "16:9": "1280*720",
    "9:16": "720*1280",
    "1:1": "1024*1024",
}
COMFY_ASPECT_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (864, 480),
    "9:16": (480, 864),
    "1:1": (640, 640),
}
SCALE_MODES = {"contain", "cover", "blur_pad"}
ALIGNMENT_MODES = {"pad", "elastic"}
SUBTITLE_STYLES = {"soft", "burn_in"}
SPEECH_MODES = {"auto", "full_text", "dialogue_only", "narrator_and_dialogue"}
MOTIONS = {"none", "push_in", "pull_out", "pan", "pan_left", "pan_up", "ken_burns"}
TRANSITIONS = {"cut", "fade", "xfade_dissolve", "xfade_wipeleft", "xfade_slideup"}
XFADE_TRANSITIONS = {
    "xfade_dissolve": "dissolve",
    "xfade_wipeleft": "wipeleft",
    "xfade_slideup": "slideup",
}
BUILTIN_BGM = ("builtin:warm", "builtin:tension", "builtin:ambient")
DEFAULT_XFADE_OVERLAP = 0.36
_CHARACTER_PRESENCE_HINTS = (
    "她", "他", "主角", "少女", "少年", "女子", "男子", "女孩", "男孩",
    "女人", "男人", "身影", "老者", "老人", "将军", "侦探", "总裁", "剑客", "侍女",
)


def shot_has_character_presence(shot: Any, character_bible: dict[str, Any] | None = None) -> bool:
    """Return True when a shot depicts or locks an identifiable character."""
    if getattr(shot, "character_locks", None):
        return True
    for line in getattr(shot, "dialogue_lines", None) or ():
        if not isinstance(line, dict):
            continue
        speaker = str(line.get("speaker") or "").strip().lower()
        if speaker and speaker not in {"旁白", "narrator"}:
            return True
    ctx = getattr(shot, "scene_context", None)
    if isinstance(ctx, dict):
        if ctx.get("characters") or ctx.get("_has_character") is True:
            return True
        visual_action = str(ctx.get("visual_action") or "")
    else:
        visual_action = ""
    blob = f"{getattr(shot, 'text', '') or ''} {visual_action}"
    if any(token in blob for token in _CHARACTER_PRESENCE_HINTS):
        return True
    if isinstance(character_bible, dict):
        for profile in character_bible.values():
            if not isinstance(profile, dict):
                continue
            name = str(profile.get("name") or "").strip()
            if name and name in blob:
                return True
            for alias in profile.get("aliases") or ():
                alias_str = str(alias).strip()
                if alias_str and alias_str in blob:
                    return True
    return False


def resolve_safe_shot_transition(
    shot_a: Any,
    shot_b: Any | None,
    *,
    character_bible: dict[str, Any] | None = None,
) -> str:
    """Avoid xfade_dissolve double-exposure ghosting between adjacent character shots."""
    raw = getattr(shot_a, "transition", "cut") or "cut"
    if raw != "xfade_dissolve" or shot_b is None:
        return raw
    ctx_a = getattr(shot_a, "scene_context", None)
    if isinstance(ctx_a, dict) and ctx_a.get("allow_character_dissolve") is True:
        return raw
    if shot_has_character_presence(shot_a, character_bible) and shot_has_character_presence(
        shot_b, character_bible
    ):
        scene_a = getattr(shot_a, "scene_id", "scene-1")
        scene_b = getattr(shot_b, "scene_id", "scene-1")
        return "fade" if scene_a != scene_b else "cut"
    return raw


def resolve_canvas_size(aspect_ratio: str = "16:9") -> tuple[int, int]:
    """Return pixel (width, height) for a supported aspect ratio."""
    if aspect_ratio not in CANVAS_SIZES:
        raise ValueError(f"unsupported aspect_ratio: {aspect_ratio}")
    return CANVAS_SIZES[aspect_ratio]


def default_image_size_for_aspect(aspect_ratio: str = "16:9") -> str:
    """Return DashScope-compatible WIDTH*HEIGHT size matching aspect_ratio."""
    return AUTO_ASPECT_IMAGE_SIZES.get(aspect_ratio, "1280*720")


def validate_edit_settings(
    *,
    aspect_ratio: str = "16:9",
    scale_mode: str = "contain",
    alignment_mode: str = "pad",
    subtitle_style: str = "soft",
    speech_mode: str = "auto",
    bgm_volume: float = 0.22,
) -> None:
    """Validate project-level editing parameters before synthesis or rendering."""
    if aspect_ratio not in CANVAS_SIZES:
        raise ValueError(f"unsupported aspect_ratio: {aspect_ratio}")
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"unsupported scale_mode: {scale_mode}")
    if alignment_mode not in ALIGNMENT_MODES:
        raise ValueError(f"unsupported alignment_mode: {alignment_mode}")
    if subtitle_style not in SUBTITLE_STYLES:
        raise ValueError(f"unsupported subtitle_style: {subtitle_style}")
    if speech_mode not in SPEECH_MODES:
        raise ValueError(f"unsupported speech_mode: {speech_mode}")
    if not math.isfinite(bgm_volume) or not 0.0 <= bgm_volume <= 1.0:
        raise ValueError("bgm_volume must be between 0.0 and 1.0")


def probe_duration(ffmpeg: str, media_path: str | Path) -> float | None:
    """Probe media container duration in seconds via FFmpeg stderr."""
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(media_path)], capture_output=True)
    match = re.search(rb"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", probe.stderr)
    if not match:
        return None
    hours, minutes, seconds = map(float, match.groups())
    return hours * 3600 + minutes * 60 + seconds


def resolve_elastic_shot_duration(
    audio_duration: float,
    video_duration: float | None,
    *,
    tts_source: str = "silent",
    alignment_mode: str = "pad",
    max_stretch: float = 1.6,
) -> float:
    """Compute shot duration under legacy pad or spoken-audio elastic alignment."""
    speech = max(1.0, float(audio_duration))
    if video_duration is None or video_duration <= 0:
        return speech
    if tts_source == "silent" or alignment_mode == "pad":
        return max(speech, float(video_duration))
    min_visual = max(1.0, float(video_duration) / max_stretch)
    return max(speech, min_visual)


def snapped_xfade_overlap(duration_a: float, duration_b: float, target: float = DEFAULT_XFADE_OVERLAP) -> float:
    """Snap cross-shot transition overlap to exact 25fps frames without exceeding 1/3 of either shot."""
    raw = min(target, duration_a / 3.0, duration_b / 3.0)
    frames = max(2, round(raw * 25))
    if frames / 25.0 >= min(duration_a, duration_b) * 0.45:
        return 0.0
    return frames / 25.0


def _split_text_clauses(text: str, max_chars: int) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return [cleaned] if cleaned else []
    raw_parts = [p.strip() for p in re.split(r"(?<=[，,、；;：:。！？.!?…])|\n+", cleaned) if p and p.strip()]
    merged: list[str] = []
    for part in raw_parts:
        clean_part = part.strip("，,、；; ")
        if not clean_part:
            clean_part = part.strip()
        if not merged:
            merged.append(clean_part)
        elif len(merged[-1]) + len(clean_part) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]} {clean_part}" if merged[-1][-1].isascii() else f"{merged[-1]}{clean_part}"
        else:
            merged.append(clean_part)
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            pieces = max(2, math.ceil(len(chunk) / max_chars))
            step = math.ceil(len(chunk) / pieces)
            for idx in range(0, len(chunk), step):
                part = chunk[idx : idx + step].strip()
                if part:
                    final.append(part)
    return final or [cleaned]


def _allocate_clause_timings(clauses: list[str], start: float, duration: float) -> list[dict[str, Any]]:
    if not clauses:
        return []
    if len(clauses) == 1:
        return [{"start": start, "end": start + duration, "text": clauses[0]}]
    weights = [max(1, len(re.sub(r"[\s，,。！？.!?\-]", "", c))) for c in clauses]
    total_weight = sum(weights)
    min_cue = min(0.4, duration / (len(clauses) * 1.5))
    flexible = max(0.0, duration - min_cue * len(clauses))
    cues: list[dict[str, Any]] = []
    cursor = start
    for idx, (clause, weight) in enumerate(zip(clauses, weights)):
        span = min_cue + flexible * (weight / total_weight)
        end = start + duration if idx == len(clauses) - 1 else cursor + span
        cues.append({"start": cursor, "end": end, "text": clause})
        cursor = end
    return cues


def build_subtitle_cues(shots: list[Any], aspect_ratio: str = "16:9") -> list[dict[str, Any]]:
    """Build proportional clause-split subtitle cues aligned to normalized shot durations."""
    max_chars = 16 if aspect_ratio == "9:16" else 26
    clock = 0.0
    cues: list[dict[str, Any]] = []
    for shot in shots:
        shot_end = clock + shot.duration
        custom_subtitle = getattr(shot, "subtitle", None)
        sub_caps = getattr(shot, "sub_captions", None) or []
        if not custom_subtitle and sub_caps:
            cap_total = max((float(c.get("end", 0.0)) for c in sub_caps), default=0.0)
            scale = (shot.duration / cap_total) if cap_total > 0 else 1.0
            for cap in sub_caps:
                cap_text = str(cap.get("text") or "").strip()
                if not cap_text:
                    continue
                c_start = clock + float(cap.get("start", 0.0)) * scale
                c_end = min(shot_end, clock + float(cap.get("end", shot.duration)) * scale)
                c_dur = max(0.25, c_end - c_start)
                cues.extend(_allocate_clause_timings(_split_text_clauses(cap_text, max_chars), c_start, c_dur))
            if cues and cues[-1]["end"] < shot_end:
                cues[-1]["end"] = shot_end
        else:
            raw_text = (custom_subtitle or shot.text or "").strip()
            clauses = _split_text_clauses(raw_text, max_chars) if len(raw_text) > max_chars else [raw_text]
            cues.extend(_allocate_clause_timings(clauses, clock, shot.duration))
        clock = shot_end
    return cues


def _fmt_srt_time(seconds: float) -> str:
    ms_total = max(0, round(seconds * 1000))
    hours, rem = divmod(ms_total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_srt(cues: list[dict[str, Any]]) -> str:
    """Format subtitle cues as standard UTF-8 SRT."""
    lines: list[str] = []
    for idx, cue in enumerate(cues, 1):
        lines += [str(idx), f"{_fmt_srt_time(cue['start'])} --> {_fmt_srt_time(cue['end'])}", cue["text"], ""]
    return "\n".join(lines)


def _fmt_ass_time(seconds: float) -> str:
    cs_total = max(0, round(seconds * 100))
    hours, rem = divmod(cs_total, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\r\n", r"\N").replace("\n", r"\N")


def format_ass(cues: list[dict[str, Any]], aspect_ratio: str = "16:9") -> str:
    """Format subtitle cues into short-drama styled ASS with safe bottom margins."""
    width, height = resolve_canvas_size(aspect_ratio)
    if aspect_ratio == "9:16":
        font_size, margin_lr, margin_v = 48, 42, 240
    elif aspect_ratio == "1:1":
        font_size, margin_lr, margin_v = 48, 56, 96
    else:
        font_size, margin_lr, margin_v = 42, 60, 48
    header = "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H0010141C,&H80000000,-1,0,0,0,100,100,1,0,1,3,1,2,{margin_lr},{margin_lr},{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    events = [
        f"Dialogue: 0,{_fmt_ass_time(c['start'])},{_fmt_ass_time(c['end'])},Default,,0,0,0,,{_escape_ass_text(c['text'])}"
        for c in cues
    ]
    return header + "\n" + "\n".join(events) + "\n"


def _synthesize_builtin_bgm_wav(preset: str, target: Path, duration: float = 12.0) -> Path:
    import numpy as np

    sample_rate = 44100
    total_samples = max(sample_rate, int(round(duration * sample_rate)))
    t = np.arange(total_samples, dtype=np.float64) / sample_rate
    two_pi = 2.0 * math.pi

    if preset == "warm":
        # Warm C-major / A-minor pentatonic progression (bar = 3.2s)
        bar_sec = 3.2
        bars = [
            ((220.00, 261.63, 329.63), (440.00, 523.25, 659.25, 587.33), 659.25),
            ((174.61, 220.00, 261.63), (349.23, 440.00, 523.25, 659.25), 587.33),
            ((130.81, 196.00, 261.63), (392.00, 523.25, 587.33, 659.25), 523.25),
            ((196.00, 246.94, 293.66), (392.00, 440.00, 523.25, 440.00), 493.88),
        ]
    elif preset == "tension":
        # Dramatic D-minor xianxia battle/confrontation score (bar = 2.4s)
        bar_sec = 2.4
        bars = [
            ((146.83, 220.00, 293.66), (293.66, 349.23, 440.00, 523.25), 587.33),
            ((130.81, 196.00, 261.63), (261.63, 329.63, 392.00, 523.25), 659.25),
            ((116.54, 174.61, 233.08), (233.08, 349.23, 440.00, 587.33), 523.25),
            ((110.00, 164.81, 220.00), (220.00, 329.63, 440.00, 349.23), 440.00),
        ]
    else:
        # Epic melancholic Guofeng / Xianxia score (D Yu-pentatonic: D F G A C, bar = 3.6s)
        bar_sec = 3.6
        bars = [
            ((146.83, 220.00, 293.66), (293.66, 440.00, 523.25, 587.33), 587.33),
            ((116.54, 174.61, 261.63), (349.23, 440.00, 523.25, 659.25), 659.25),
            ((174.61, 261.63, 349.23), (349.23, 523.25, 587.33, 783.99), 783.99),
            ((130.81, 196.00, 293.66), (392.00, 440.00, 523.25, 440.00), 523.25),
            ((196.00, 293.66, 349.23), (293.66, 392.00, 523.25, 587.33), 659.25),
            ((146.83, 220.00, 261.63), (293.66, 349.23, 440.00, 587.33), 587.33),
            ((130.81, 220.00, 261.63), (261.63, 392.00, 440.00, 523.25), 440.00),
            ((146.83, 220.00, 293.66), (293.66, 440.00, 349.23, 293.66), 587.33),
        ]

    left = np.zeros(total_samples, dtype=np.float64)
    right = np.zeros(total_samples, dtype=np.float64)
    bar_samples = max(1, int(round(bar_sec * sample_rate)))
    num_bars = math.ceil(total_samples / bar_samples)

    for b_idx in range(num_bars):
        b_start = b_idx * bar_samples
        b_end = min(total_samples, b_start + bar_samples)
        n = b_end - b_start
        if n <= 0:
            continue
        bt = np.arange(n, dtype=np.float64) / sample_rate
        chord, arp_notes, flute_freq = bars[b_idx % len(bars)]

        # 1. Warm bowed strings pad with gentle attack/release & stereo detune
        attack_s = min(0.35, bar_sec * 0.2)
        release_s = min(0.45, bar_sec * 0.25)
        pad_env = np.ones(n, dtype=np.float64)
        att_n = max(1, int(attack_s * sample_rate))
        rel_n = max(1, int(release_s * sample_rate))
        if att_n < n:
            pad_env[:att_n] = np.linspace(0.15, 1.0, att_n)
        if rel_n < n:
            pad_env[-rel_n:] *= np.linspace(1.0, 0.2, rel_n)
        swell = 0.82 + 0.18 * np.sin(two_pi * (0.35 if preset != "tension" else 1.2) * bt)

        pad_l = np.zeros(n, dtype=np.float64)
        pad_r = np.zeros(n, dtype=np.float64)
        for c_i, cf in enumerate(chord):
            w = 0.42 if c_i == 0 else 0.30
            pad_l += w * (0.70 * np.sin(two_pi * cf * bt) + 0.22 * np.sin(two_pi * (2.0 * cf) * bt) + 0.08 * np.sin(two_pi * (3.0 * cf) * bt))
            pad_r += w * (0.70 * np.sin(two_pi * (cf * 1.0018) * bt + 0.4) + 0.22 * np.sin(two_pi * (2.002 * cf) * bt + 0.8) + 0.08 * np.sin(two_pi * (3.0 * cf) * bt))
        left[b_start:b_end] += 0.32 * pad_l * pad_env * swell
        right[b_start:b_end] += 0.32 * pad_r * pad_env * swell

        # 2. Guzheng / Guqin plucked pentatonic arpeggios (4 plucks per bar, ringing tail)
        step_sec = bar_sec / len(arp_notes)
        for p_i, pf in enumerate(arp_notes):
            p_offset = b_start + int(round(p_i * step_sec * sample_rate))
            if p_offset >= total_samples:
                break
            ring_n = min(total_samples - p_offset, int(round(1.85 * sample_rate)))
            pt = np.arange(ring_Gate := ring_n, dtype=np.float64) / sample_rate
            pluck_att = np.minimum(1.0, pt / 0.008)
            pluck_env = pluck_att * np.exp(-pt * 2.6)
            pluck_sig = (
                0.52 * np.sin(two_pi * pf * pt)
                + 0.28 * np.sin(two_pi * (2.0 * pf) * pt) * np.exp(-pt * 1.5)
                + 0.13 * np.sin(two_pi * (3.0 * pf) * pt) * np.exp(-pt * 3.0)
                + 0.07 * np.sin(two_pi * (4.0 * pf) * pt) * np.exp(-pt * 4.5)
            ) * pluck_env
            pan = 0.30 + 0.40 * (p_i / max(1, len(arp_notes) - 1))
            left[p_offset : p_offset + ring_Gate] += 0.38 * (1.0 - pan * 0.55) * pluck_sig
            right[p_offset : p_offset + ring_Gate] += 0.38 * (0.45 + pan * 0.55) * pluck_sig

        # 3. Expressive Bamboo Xiao / Dizi sustained melody phrase on second beat
        f_offset = b_start + int(round(0.18 * bar_sec * sample_rate))
        if f_offset < total_samples:
            f_dur = min(bar_sec * 0.72, (total_samples - f_offset) / sample_rate)
            fn = int(round(f_dur * sample_rate))
            if fn > 100:
                ft = np.arange(fn, dtype=np.float64) / sample_rate
                vib_depth = np.clip((ft - 0.22) / 0.35, 0.0, 1.0) * 0.0045
                phase = two_pi * flute_freq * (ft + vib_depth * np.sin(two_pi * 5.2 * ft) / (two_pi * 5.2))
                f_att = np.minimum(1.0, ft / 0.16)
                f_rel = np.minimum(1.0, np.maximum(0.0, (f_dur - ft) / 0.28))
                f_env = f_att * f_rel
                flute_sig = (
                    0.68 * np.sin(phase)
                    + 0.22 * np.sin(2.0 * phase + 0.2)
                    + 0.10 * np.sin(3.0 * phase + 0.5)
                ) * f_env
                left[f_offset : f_offset + fn] += 0.26 * flute_sig
                right[f_offset : f_offset + fn] += 0.28 * flute_sig

        # 4. Deep ceremonial ancient drum on bar downbeat
        d_n = min(n, int(round(0.65 * sample_rate)))
        if d_n > 10:
            dt = np.arange(d_n, dtype=np.float64) / sample_rate
            d_phase = two_pi * (52.0 * dt + 28.0 * (1.0 - np.exp(-dt * 18.0)) / 18.0)
            drum = np.sin(d_phase) * np.exp(-dt * 6.5) * np.minimum(1.0, dt / 0.005)
            drum_gain = 0.34 if preset == "tension" else 0.22
            left[b_start : b_start + d_n] += drum_gain * drum
            right[b_start : b_start + d_n] += drum_gain * drum

    peak = float(max(np.max(np.abs(left)), np.max(np.abs(right)), 1e-6))
    scale = 24500.0 / peak
    interleaved = np.empty(total_samples * 2, dtype=np.int16)
    interleaved[0::2] = np.clip(left * scale, -32767, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(right * scale, -32767, 32767).astype(np.int16)
    pcm = interleaved.tobytes()

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.wav")
    try:
        with wave.open(str(temporary), "wb") as wav_out:
            wav_out.setnchannels(2)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(pcm)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def ensure_bgm_track(bgm_source: str, work_dir: str | Path, *, duration: float = 12.0) -> Path:
    """Resolve a local BGM file or generate a deterministic built-in procedural BGM WAV."""
    source = bgm_source.strip()
    if source.startswith("builtin:"):
        preset = source.split(":", 1)[1].strip() or "warm"
        if f"builtin:{preset}" not in BUILTIN_BGM:
            raise ValueError(f"unsupported built-in BGM preset: {source}")
        target_dur = max(12.0, math.ceil(duration / 4.0) * 4.0)
        cached = Path(work_dir) / ".bgm" / f"{preset}_{int(target_dur)}s.wav"
        if cached.is_file():
            return cached
        return _synthesize_builtin_bgm_wav(preset, cached, duration=target_dur)
    local_path = Path(source)
    if not local_path.is_file():
        raise FileNotFoundError(f"BGM audio file not found: {source}")
    return local_path


def mix_audio_with_bgm(
    ffmpeg: str,
    voice_path: str | Path,
    bgm_source: str,
    output_path: str | Path,
    *,
    duration: float,
    volume: float = 0.22,
    ducking: bool = True,
    work_dir: str | Path | None = None,
) -> Path:
    """Mix spoken voice WAV with looped BGM using optional sidechain vocal ducking."""
    target = Path(output_path)
    bgm_path = ensure_bgm_track(bgm_source, work_dir or target.parent, duration=duration)
    fade_out = min(0.8, duration / 3.0)
    fade_start = max(0.0, duration - fade_out)
    if ducking:
        filter_graph = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo,asplit=2[v][sc];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,atrim=0:{duration:.3f},volume={volume:.3f}[bgm];"
            "[bgm][sc]sidechaincompress=threshold=0.08:ratio=2.2:attack=60:release=520[ducked];"
            f"[v][ducked]amix=inputs=2:duration=first:normalize=0,afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[out]"
        )
    else:
        filter_graph = (
            "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[v];"
            f"[1:a]aformat=sample_rates=44100:channel_layouts=stereo,atrim=0:{duration:.3f},"
            f"volume={volume:.3f},afade=t=out:st={fade_start:.3f}:d={fade_out:.3f}[bgm];"
            "[v][bgm]amix=inputs=2:duration=first:normalize=0[out]"
        )
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.wav")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-xerror",
                "-i",
                str(voice_path),
                "-stream_loop",
                "-1",
                "-i",
                str(bgm_path),
                "-filter_complex",
                filter_graph,
                "-map",
                "[out]",
                "-t",
                f"{duration:.3f}",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
            capture_output=True,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def compose_xfade_segments(
    ffmpeg: str,
    segment_paths: list[Path],
    shots: list[Any],
    output_path: str | Path,
    *,
    character_bible: dict[str, Any] | None = None,
) -> Path:
    """Join rendered shot segments using zero-drift xfade transitions or hard cuts."""
    target = Path(output_path)
    if len(segment_paths) <= 1:
        if segment_paths:
            target.write_bytes(Path(segment_paths[0]).read_bytes())
        return target

    inputs: list[str] = []
    for seg in segment_paths:
        inputs += ["-i", str(seg)]

    filter_steps: list[str] = []
    prev_label = "[0:v]"
    first_trans = resolve_safe_shot_transition(shots[0], shots[1], character_bible=character_bible)
    first_overlap = (
        snapped_xfade_overlap(shots[0].duration, shots[1].duration)
        if first_trans in XFADE_TRANSITIONS
        else 0.0
    )
    cumulative = shots[0].duration + first_overlap
    for i in range(1, len(shots)):
        prev_trans = resolve_safe_shot_transition(
            shots[i - 1], shots[i], character_bible=character_bible
        )
        next_trans = (
            resolve_safe_shot_transition(shots[i], shots[i + 1], character_bible=character_bible)
            if i + 1 < len(shots)
            else "cut"
        )
        next_overlap = (
            snapped_xfade_overlap(shots[i].duration, shots[i + 1].duration)
            if i + 1 < len(shots) and next_trans in XFADE_TRANSITIONS
            else 0.0
        )
        out_label = f"[v{i}]"
        if prev_trans in XFADE_TRANSITIONS:
            overlap = snapped_xfade_overlap(shots[i - 1].duration, shots[i].duration)
            if overlap > 0:
                offset = max(0.0, cumulative - overlap)
                xf_mode = XFADE_TRANSITIONS[prev_trans]
                filter_steps.append(f"{prev_label}[{i}:v]xfade=transition={xf_mode}:duration={overlap:.2f}:offset={offset:.2f}{out_label}")
                cumulative = cumulative - overlap + shots[i].duration + next_overlap
            else:
                filter_steps.append(f"{prev_label}[{i}:v]concat=n=2:v=1:a=0{out_label}")
                cumulative += shots[i].duration + next_overlap
        else:
            filter_steps.append(f"{prev_label}[{i}:v]concat=n=2:v=1:a=0{out_label}")
            cumulative += shots[i].duration + next_overlap
        prev_label = out_label

    total_duration = sum(s.duration for s in shots)
    filter_steps.append(f"{prev_label}fps=25,setsar=1,format=yuv420p[vout]")
    script_file = target.with_name(f".xfade-{uuid4().hex}.txt")
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.mp4")
    try:
        script_file.write_text(";".join(filter_steps), encoding="utf-8")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-xerror",
                *inputs,
                "-filter_complex_script",
                str(script_file),
                "-map",
                "[vout]",
                "-t",
                f"{total_duration:.3f}",
                "-c:v",
                "libx264",
                str(temporary),
            ],
            check=True,
            capture_output=True,
        )
        temporary.replace(target)
    finally:
        script_file.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
    return target
