"""Local lightweight VAD + ASR speech audit, CER verification, and subtitle alignment."""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .h3_bundle import extract_dialogue_tags, normalize_dialogue_text
from .local_media import find_ffmpeg

SPEECH_AUDIT_SCHEMA = "speech-audit-report/v1"
VAD_SAMPLE_RATE = 16000

WHISPER_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"字幕[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"谢谢观看[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"感谢观看[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"请不吝点赞[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"订阅[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"优优独播[^\n。！？?!]*", re.IGNORECASE),
    re.compile(r"\bAmara(?:\.org)?\b[^\n.!?]*", re.IGNORECASE),
    re.compile(r"\bThank you for watching\b[^\n.!?]*", re.IGNORECASE),
    re.compile(r"\bSubtitles by\b[^\n.!?]*", re.IGNORECASE),
)


@dataclass(frozen=True)
class SpeechSegment:
    """Contiguous active speech interval detected by VAD in seconds."""

    start: float
    end: float

    def to_dict(self) -> dict[str, float]:
        return {"start": round(float(self.start), 3), "end": round(float(self.end), 3)}


@dataclass(frozen=True)
class TimedWord:
    """Single ASR token/word with segment-relative timestamps."""

    text: str
    start: float
    end: float
    probability: float = 1.0


@dataclass(frozen=True)
class TranscriptSegment:
    """Single ASR segment with confidence metadata."""

    text: str
    start: float
    end: float
    no_speech_prob: float
    avg_logprob: float
    words: tuple[TimedWord, ...] = ()


@dataclass(frozen=True)
class TranscriptResult:
    """Full ASR output returned by a speech audit backend."""

    text: str
    language: str
    segments: tuple[TranscriptSegment, ...]
    words: tuple[TimedWord, ...]


@dataclass(frozen=True)
class SpeechFinding:
    """Error or warning raised by VAD/ASR speech verification."""

    severity: str
    code: str
    shot_id: str | None
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "shot_id": self.shot_id,
            "message": self.message,
        }
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass
class ShotSpeechAudit:
    """Per-shot VAD + ASR audit outcome and aligned subtitle cues."""

    shot_id: str
    media_path: str
    expected_dialogue: bool
    expected_line: str | None
    ok: bool
    vad_segments: list[SpeechSegment] = field(default_factory=list)
    vad_total_sec: float = 0.0
    transcript_text: str = ""
    cer: float | None = None
    char_recall: float | None = None
    subtitle_cues: list[dict[str, Any]] = field(default_factory=list)
    errors: list[SpeechFinding] = field(default_factory=list)
    warnings: list[SpeechFinding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "media_path": self.media_path,
            "expected_dialogue": self.expected_dialogue,
            "expected_line": self.expected_line,
            "ok": self.ok,
            "vad_segments": [s.to_dict() for s in self.vad_segments],
            "vad_total_sec": self.vad_total_sec,
            "transcript_text": self.transcript_text,
            "cer": self.cer,
            "char_recall": self.char_recall,
            "subtitle_cues": self.subtitle_cues,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


@dataclass
class ChapterSpeechAuditReport:
    """Chapter-level VAD + ASR audit report including clean override suggestions."""

    schema: str
    chapter: str
    ok: bool
    shots: list[ShotSpeechAudit] = field(default_factory=list)
    suggested_audio_source_override: dict[str, str] = field(default_factory=dict)
    subtitle_cues: list[dict[str, Any]] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "chapter": self.chapter,
            "ok": self.ok,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "suggested_audio_source_override": self.suggested_audio_source_override,
            "subtitle_cues": self.subtitle_cues,
            "shots": [s.to_dict() for s in self.shots],
        }


class SpeechAuditBackend(Protocol):
    """Protocol for pluggable local/mocked VAD + ASR engines."""

    def detect_speech_segments(
        self,
        audio_16k: np.ndarray,
        *,
        threshold: float = 0.45,
        min_speech_ms: int = 250,
        min_silence_ms: int = 200,
    ) -> list[SpeechSegment]:
        ...

    def transcribe(
        self,
        audio_16k: np.ndarray,
        *,
        language: str = "zh",
        initial_prompt: str | None = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
    ) -> TranscriptResult:
        ...


class FasterWhisperBackend:
    """Lightweight ONNX Silero VAD + CTranslate2 Whisper INT8 backend (lazy-loaded)."""

    def __init__(
        self,
        *,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_size = model_size or os.environ.get("STORY_MEDIA_ASR_MODEL") or "small"
        self.device = device or os.environ.get("STORY_MEDIA_ASR_DEVICE") or "cpu"
        self.compute_type = compute_type or os.environ.get("STORY_MEDIA_ASR_COMPUTE_TYPE") or "int8"
        self.cache_dir = cache_dir or os.environ.get("STORY_MEDIA_ASR_CACHE_DIR") or None
        self._model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.cache_dir,
            )
        return self._model

    def detect_speech_segments(
        self,
        audio_16k: np.ndarray,
        *,
        threshold: float = 0.45,
        min_speech_ms: int = 250,
        min_silence_ms: int = 200,
    ) -> list[SpeechSegment]:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        opts = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=80,
        )
        raw = get_speech_timestamps(audio_16k.astype(np.float32, copy=False), vad_options=opts, sampling_rate=VAD_SAMPLE_RATE)
        return [
            SpeechSegment(round(seg["start"] / float(VAD_SAMPLE_RATE), 3), round(seg["end"] / float(VAD_SAMPLE_RATE), 3))
            for seg in raw
        ]

    def transcribe(
        self,
        audio_16k: np.ndarray,
        *,
        language: str = "zh",
        initial_prompt: str | None = None,
        word_timestamps: bool = True,
        vad_filter: bool = True,
    ) -> TranscriptResult:
        model = self._ensure_model()
        segs_gen, info = model.transcribe(
            audio_16k.astype(np.float32, copy=False),
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            condition_on_previous_text=False,
            beam_size=3,
        )
        segments: list[TranscriptSegment] = []
        all_words: list[TimedWord] = []
        texts: list[str] = []
        for seg in segs_gen:
            seg_words = tuple(
                TimedWord(
                    text=str(getattr(w, "word", "")),
                    start=float(getattr(w, "start", seg.start)),
                    end=float(getattr(w, "end", seg.end)),
                    probability=float(getattr(w, "probability", 1.0)),
                )
                for w in (getattr(seg, "words", None) or ())
            )
            all_words.extend(seg_words)
            stxt = str(getattr(seg, "text", "") or "").strip()
            if stxt:
                texts.append(stxt)
            segments.append(
                TranscriptSegment(
                    text=stxt,
                    start=float(seg.start),
                    end=float(seg.end),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
                    words=seg_words,
                )
            )
        return TranscriptResult(
            text="".join(texts),
            language=str(getattr(info, "language", language) or language),
            segments=tuple(segments),
            words=tuple(all_words),
        )


def decode_audio_16k_mono(media_path: Path, *, ffmpeg_bin: str | None = None) -> tuple[np.ndarray, float]:
    """Decode any audio/video file to 16 kHz mono float32 samples in `[-1.0, 1.0]`."""
    exe = ffmpeg_bin or find_ffmpeg()
    p = Path(media_path)
    if not p.is_file():
        raise FileNotFoundError(f"Media file not found: {p}")
    proc = subprocess.run(
        [
            exe, "-v", "error", "-i", str(p),
            "-vn", "-acodec", "pcm_f32le", "-ar", str(VAD_SAMPLE_RATE), "-ac", "1", "-f", "f32le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg failed decoding audio for {p}: {err[-1000:]}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return audio, round(audio.size / float(VAD_SAMPLE_RATE), 3)


def normalize_for_vad(
    audio: np.ndarray,
    *,
    target_rms_dbfs: float = -24.0,
    max_gain_db: float = 20.0,
    silence_floor_dbfs: float = -65.0,
) -> np.ndarray:
    """Boost quiet non-silent audio gently toward target RMS for consistent VAD sensitivity."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    rms_db = 20.0 * math.log10(max(1e-12, rms))
    if rms_db <= silence_floor_dbfs:
        return audio.astype(np.float32, copy=False)
    gain_db = max(0.0, min(max_gain_db, target_rms_dbfs - rms_db))
    gain = 10.0 ** (gain_db / 20.0)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0 and peak * gain > 0.92:
        gain = 0.92 / peak
    return (audio * np.float32(gain)).astype(np.float32, copy=False)


def strip_whisper_boilerplate(text: str) -> str:
    """Remove known non-speech Whisper subtitle hallucinations before stray-speech judgment."""
    cleaned = str(text or "").strip()
    for pat in WHISPER_BOILERPLATE_PATTERNS:
        cleaned = pat.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not normalize_dialogue_text(cleaned):
        return ""
    return cleaned


def levenshtein_distance(a: str, b: str) -> int:
    """Standard Levenshtein edit distance on strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def compute_cer(expected: str, hypothesis: str) -> tuple[float, float]:
    """Compute normalized Character Error Rate and matched character recall ratio."""
    norm_exp = normalize_dialogue_text(expected)
    norm_hyp = normalize_dialogue_text(hypothesis)
    if not norm_exp:
        return (0.0 if not norm_hyp else 1.0, 1.0 if not norm_hyp else 0.0)
    dist = levenshtein_distance(norm_exp, norm_hyp)
    cer = round(dist / float(len(norm_exp)), 4)
    matches = sum(block.size for block in difflib.SequenceMatcher(None, norm_exp, norm_hyp, autojunk=False).get_matching_blocks())
    recall = round(matches / float(len(norm_exp)), 4)
    return cer, recall


def split_dialogue_clauses(line: str, *, min_clause_chars: int = 2) -> list[str]:
    """Split Mandarin/English dialogue into readable subtitle clauses while preserving every character."""
    text = str(line or "").strip()
    if not text:
        return []
    tokens = re.findall(r"[^。！？；，、?!,;]+[。！？；，、?!,;]*|[。！？；，、?!,;]+", text)
    if not tokens:
        return [text]

    # Attach punctuation-only tokens to neighbors
    merged_punct: list[str] = []
    for tok in tokens:
        if len(normalize_dialogue_text(tok)) == 0:
            if merged_punct:
                merged_punct[-1] += tok
            else:
                merged_punct.append(tok)
        else:
            if merged_punct and len(normalize_dialogue_text(merged_punct[0])) == 0:
                merged_punct[0] += tok
            else:
                merged_punct.append(tok)

    # Merge tiny non-final fragments (< min_clause_chars)
    out: list[str] = []
    i = 0
    while i < len(merged_punct):
        tok = merged_punct[i]
        nlen = len(normalize_dialogue_text(tok))
        is_final = bool(re.search(r"[。！？?!]\s*$", tok))
        if nlen < min_clause_chars and not is_final:
            if i + 1 < len(merged_punct):
                merged_punct[i + 1] = tok + merged_punct[i + 1]
            elif out:
                out[-1] += tok
            else:
                out.append(tok)
        else:
            out.append(tok)
        i += 1
    return out


def align_clauses_to_words(
    clauses: list[str],
    words: list[TimedWord],
    *,
    shot_duration: float,
    vad_segments: list[SpeechSegment] | None = None,
    lead_pad_sec: float = 0.06,
    tail_pad_sec: float = 0.10,
    min_cue_dur_sec: float = 0.25,
) -> list[tuple[str, float, float]]:
    """Map script clauses cleanly to monotonic `[start, end]` timestamps anchored to ASR words."""
    if not clauses:
        return []
    norm_clauses = [normalize_dialogue_text(c) for c in clauses]
    norm_script = "".join(norm_clauses)
    n_chars = len(norm_script)
    max_end = max(min_cue_dur_sec, shot_duration - 0.04)

    asr_chars: list[tuple[str, float, float]] = []
    for w in words:
        nw = normalize_dialogue_text(w.text)
        if not nw:
            continue
        w0 = float(w.start)
        w1 = max(w0 + 0.04, float(w.end))
        step = (w1 - w0) / float(len(nw))
        for k, ch in enumerate(nw):
            c0 = w0 + k * step
            c1 = w0 + (k + 1) * step
            if vad_segments and (c1 - c0) > 0.28:
                overlaps = [
                    (max(c0, v.start), min(c1, v.end))
                    for v in vad_segments
                    if min(c1, v.end) > max(c0, v.start)
                ]
                if overlaps:
                    c0 = min(ov[0] for ov in overlaps)
                    c1 = max(ov[1] for ov in overlaps)
                    if c1 <= c0 + 0.03:
                        c1 = c0 + 0.04
            asr_chars.append((ch, c0, c1))

    if vad_segments:
        g_start = max(0.0, float(vad_segments[0].start))
        g_end = min(max_end, float(vad_segments[-1].end))
    else:
        g_start = min(0.2, max_end * 0.2)
        g_end = max(g_start + min_cue_dur_sec, max_end - 0.10)
    if g_end <= g_start:
        g_end = min(max_end, g_start + min_cue_dur_sec)

    char_spans: list[tuple[float, float] | None] = [None] * n_chars
    if n_chars > 0 and asr_chars:
        asr_str = "".join(c for c, _, _ in asr_chars)
        sm = difflib.SequenceMatcher(None, norm_script, asr_str, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    char_spans[i1 + k] = (asr_chars[j1 + k][1], asr_chars[j1 + k][2])
            elif tag == "replace" and i2 > i1 and j2 > j1:
                t0 = asr_chars[j1][1]
                t1 = max(t0 + 0.05, asr_chars[j2 - 1][2])
                step = (t1 - t0) / float(i2 - i1)
                for k in range(i2 - i1):
                    char_spans[i1 + k] = (t0 + k * step, t0 + (k + 1) * step)

    # Pass 2: interpolate unmatched script characters cleanly (anchor against right edge across silent pauses)
    clause_starts: set[int] = set()
    p_acc = 0
    for nc in norm_clauses:
        clause_starts.add(p_acc)
        p_acc += len(nc)

    idx = 0
    while idx < n_chars:
        if char_spans[idx] is not None:
            idx += 1
            continue
        run_end = idx
        while run_end < n_chars and char_spans[run_end] is None:
            run_end += 1
        left_t = char_spans[idx - 1][1] if idx > 0 and char_spans[idx - 1] is not None else g_start
        right_t = char_spans[run_end][0] if run_end < n_chars and char_spans[run_end] is not None else g_end
        if right_t <= left_t:
            right_t = min(max_end, left_t + 0.22 * (run_end - idx))
        run_len = run_end - idx
        span_left = left_t
        if idx in clause_starts and (right_t - left_t) > max(0.50, 0.24 * run_len) and run_end < n_chars:
            span_left = max(left_t, right_t - min(0.45, 0.18 * run_len))
        step = max(0.02, (right_t - span_left) / float(run_len))
        for k in range(run_len):
            t0 = min(max_end - 0.05, span_left + k * step)
            t1 = min(max_end, t0 + step)
            char_spans[idx + k] = (t0, max(t0 + 0.02, t1))
        idx = run_end

    # Pass 3: extract per-clause intervals
    raw: list[list[Any]] = []
    pos = 0
    total_norm = max(1, n_chars)
    for clause, nc in zip(clauses, norm_clauses):
        clen = len(nc)
        if clen > 0 and pos + clen <= len(char_spans):
            c0 = char_spans[pos][0] - lead_pad_sec
            c1 = char_spans[pos + clen - 1][1] + tail_pad_sec
        else:
            frac0 = pos / float(total_norm)
            frac1 = (pos + clen) / float(total_norm)
            c0 = g_start + (g_end - g_start) * frac0 - lead_pad_sec
            c1 = g_start + (g_end - g_start) * frac1 + tail_pad_sec
        pos += clen
        st = max(0.0, min(max_end - min_cue_dur_sec, c0))
        en = min(max_end, max(st + min_cue_dur_sec, c1))
        raw.append([clause, st, en])

    # Pass 4: enforce monotonic non-overlapping cue spans
    for k in range(len(raw) - 1):
        if raw[k][2] > raw[k + 1][1]:
            mid = (raw[k][2] + raw[k + 1][1]) * 0.5
            raw[k][2] = max(raw[k][1] + 0.15, mid)
            raw[k + 1][1] = min(raw[k + 1][2] - 0.15, raw[k][2])

    return [(item[0], round(float(item[1]), 2), round(float(item[2]), 2)) for item in raw]


def audit_shot_speech(
    media_path: Path,
    *,
    shot_id: str = "01",
    expected_line: str | None = None,
    speaker: str = "Speaker",
    style: str = "Dialogue",
    backend: SpeechAuditBackend | None = None,
    cer_warn_threshold: float = 0.20,
    cer_fail_threshold: float = 0.45,
    stray_vad_min_sec: float = 0.35,
    stray_vad_force_sec: float = 1.0,
    no_speech_prob_max: float = 0.85,
    avg_logprob_min: float = -1.25,
    ffmpeg_bin: str | None = None,
) -> ShotSpeechAudit:
    """Audit a single shot's audio track with VAD + ASR and align subtitles when spoken."""
    p = Path(media_path)
    be = backend or FasterWhisperBackend()
    audio_16k, dur_sec = decode_audio_16k_mono(p, ffmpeg_bin=ffmpeg_bin)
    norm_audio = normalize_for_vad(audio_16k)

    vad_segs = be.detect_speech_segments(norm_audio)
    vad_total = round(sum(max(0.0, s.end - s.start) for s in vad_segs), 3)
    errors: list[SpeechFinding] = []
    warnings: list[SpeechFinding] = []

    has_expected = bool(expected_line and str(expected_line).strip())
    if not has_expected:
        if vad_total < 0.25:
            return ShotSpeechAudit(
                shot_id=shot_id,
                media_path=str(p),
                expected_dialogue=False,
                expected_line=None,
                ok=True,
                vad_segments=vad_segs,
                vad_total_sec=vad_total,
            )

        tr = be.transcribe(norm_audio, language="zh", initial_prompt=None, word_timestamps=False, vad_filter=True)
        conf_segs = [
            seg for seg in tr.segments
            if seg.no_speech_prob < no_speech_prob_max and seg.avg_logprob > avg_logprob_min
        ]
        cleaned_txt = strip_whisper_boilerplate("".join(seg.text for seg in conf_segs))
        confirmed_chars = normalize_dialogue_text(cleaned_txt)
        if (vad_total >= stray_vad_min_sec and len(confirmed_chars) >= 2) or (
            vad_total >= stray_vad_force_sec and bool(conf_segs)
        ):
            errors.append(
                SpeechFinding(
                    "error",
                    "STRAY_SPEECH",
                    shot_id,
                    f"Detected unscripted speech in silent shot ({vad_total:.2f}s VAD, text={cleaned_txt!r})",
                    {
                        "vad_total_sec": vad_total,
                        "transcript": cleaned_txt,
                        "raw_transcript": tr.text,
                        "segments": [
                            {"text": s.text, "no_speech_prob": round(s.no_speech_prob, 3), "avg_logprob": round(s.avg_logprob, 3)}
                            for s in conf_segs
                        ],
                    },
                )
            )
        elif vad_total >= 0.25:
            warnings.append(
                SpeechFinding(
                    "warning",
                    "POSSIBLE_STRAY_SPEECH",
                    shot_id,
                    f"Short/unconfirmed VAD activity ({vad_total:.2f}s) without confident speech transcript",
                    {"vad_total_sec": vad_total, "raw_transcript": tr.text},
                )
            )
        return ShotSpeechAudit(
            shot_id=shot_id,
            media_path=str(p),
            expected_dialogue=False,
            expected_line=None,
            ok=len(errors) == 0,
            vad_segments=vad_segs,
            vad_total_sec=vad_total,
            transcript_text=cleaned_txt or tr.text,
            errors=errors,
            warnings=warnings,
        )

    # Spoken / voiceover branch — never pass expected_line into initial_prompt so CER remains unbiased
    tr = be.transcribe(
        norm_audio,
        language="zh",
        initial_prompt="以下是普通话简体中文对白。",
        word_timestamps=True,
        vad_filter=True,
    )
    cer, char_recall = compute_cer(expected_line or "", tr.text)
    norm_hyp = normalize_dialogue_text(tr.text)

    if not norm_hyp and vad_total < 0.20:
        errors.append(
            SpeechFinding(
                "error",
                "MISSING_DIALOGUE_SPEECH",
                shot_id,
                f"Expected dialogue {expected_line!r} but detected no speech ({vad_total:.2f}s VAD)",
                {"vad_total_sec": vad_total},
            )
        )
    elif cer > cer_fail_threshold:
        errors.append(
            SpeechFinding(
                "error",
                "DIALOGUE_CER_HIGH",
                shot_id,
                f"Dialogue CER={cer:.2f} exceeds fail threshold {cer_fail_threshold:.2f} (asr={tr.text!r})",
                {"expected": expected_line, "transcript": tr.text, "cer": cer, "char_recall": char_recall},
            )
        )
    elif cer > cer_warn_threshold:
        warnings.append(
            SpeechFinding(
                "warning",
                "DIALOGUE_CER_WARN",
                shot_id,
                f"Dialogue CER={cer:.2f} exceeds warn threshold {cer_warn_threshold:.2f} (asr={tr.text!r})",
                {"expected": expected_line, "transcript": tr.text, "cer": cer, "char_recall": char_recall},
            )
        )

    clauses = split_dialogue_clauses(expected_line or "")
    aligned = align_clauses_to_words(clauses, list(tr.words), shot_duration=dur_sec, vad_segments=vad_segs)
    cues = [
        {
            "shot_id": shot_id,
            "local_start": l_st,
            "local_end": l_en,
            "style": style,
            "speaker": speaker,
            "text": clause_txt,
        }
        for clause_txt, l_st, l_en in aligned
    ]
    return ShotSpeechAudit(
        shot_id=shot_id,
        media_path=str(p),
        expected_dialogue=True,
        expected_line=expected_line,
        ok=len(errors) == 0,
        vad_segments=vad_segs,
        vad_total_sec=vad_total,
        transcript_text=tr.text,
        cer=cer,
        char_recall=char_recall,
        subtitle_cues=cues,
        errors=errors,
        warnings=warnings,
    )


def audit_chapter_speech(
    chapter_dir: Path,
    *,
    bundle_dir: Path | None = None,
    audit_json: Path | None = None,
    backend: SpeechAuditBackend | None = None,
    cer_warn_threshold: float = 0.20,
    cer_fail_threshold: float = 0.45,
    ffmpeg_bin: str | None = None,
) -> ChapterSpeechAuditReport:
    """Run local VAD + ASR audit across all shots in a chapter directory."""
    chap_p = Path(chapter_dir)
    audit_p = Path(audit_json) if audit_json is not None else chap_p / "full_9shot_audit.json"
    if not audit_p.is_file():
        raise FileNotFoundError(f"Missing chapter audit JSON: {audit_p}")
    audit_data = json.loads(audit_p.read_text(encoding="utf-8"))
    chapter_id = str(audit_data.get("chapter") or "01").zfill(2)

    bundle_shots: dict[str, dict[str, Any]] = {}
    if bundle_dir is not None:
        b_path = Path(bundle_dir)
        prompts_p = b_path if b_path.is_file() else b_path / "h3_prompts.json"
        if prompts_p.is_file():
            b_data = json.loads(prompts_p.read_text(encoding="utf-8"))
            for bs in b_data.get("shots") or []:
                bs_id = str(bs.get("id") or "").zfill(2) if str(bs.get("id") or "").isdigit() else str(bs.get("id") or "")
                bundle_shots[bs_id] = bs

    shared_be = backend or FasterWhisperBackend()
    shot_audits: list[ShotSpeechAudit] = []
    all_cues: list[dict[str, Any]] = []

    for s in audit_data.get("shots") or []:
        sid_raw = str(s["id"])
        sid = sid_raw.zfill(2) if sid_raw.isdigit() else sid_raw
        bs = bundle_shots.get(sid, {})
        dlg = s.get("dialogue") if isinstance(s.get("dialogue"), dict) and s["dialogue"].get("line") else bs.get("dialogue")

        expected_line: str | None = None
        speaker = "Speaker"
        style = "Dialogue"
        if isinstance(dlg, dict) and str(dlg.get("line") or "").strip():
            expected_line = str(dlg["line"]).strip()
            speaker = str(dlg.get("speaker") or "Speaker")
            style = "Voiceover" if dlg.get("mode") == "off_screen_voiceover" else "Dialogue"
        else:
            prompt_src = str(bs.get("h3_prompt") or s.get("h3_prompt") or "")
            tags = extract_dialogue_tags(prompt_src)
            if tags:
                expected_line = "".join(tags)

        native_wav = chap_p / "audio" / f"shot_{sid}_native.wav"
        if native_wav.is_file():
            media_p = native_wav
        elif s.get("raw_video") and Path(str(s["raw_video"])).is_file():
            media_p = Path(str(s["raw_video"]))
        else:
            media_p = chap_p / "video" / f"shot_{sid}_raw.mp4"

        res = audit_shot_speech(
            media_p,
            shot_id=sid,
            expected_line=expected_line,
            speaker=speaker,
            style=style,
            backend=shared_be,
            cer_warn_threshold=cer_warn_threshold,
            cer_fail_threshold=cer_fail_threshold,
            ffmpeg_bin=ffmpeg_bin,
        )
        shot_audits.append(res)
        all_cues.extend(res.subtitle_cues)

    # Propose nearest clean non-dialogue shot overrides for any STRAY_SPEECH silent shots
    clean_silent = [sa for sa in shot_audits if not sa.expected_dialogue and sa.ok]
    overrides: dict[str, str] = {}
    for sa in shot_audits:
        if sa.expected_dialogue or not any(e.code == "STRAY_SPEECH" for e in sa.errors):
            continue
        if clean_silent:
            sid_num = int(sa.shot_id) if sa.shot_id.isdigit() else 0
            best = min(
                clean_silent,
                key=lambda cand: (
                    1 if cand.warnings else 0,
                    cand.vad_total_sec,
                    abs((int(cand.shot_id) if cand.shot_id.isdigit() else 0) - sid_num),
                ),
            )
            overrides[sa.shot_id] = best.shot_id

    err_count = sum(len(s.errors) for s in shot_audits)
    warn_count = sum(len(s.warnings) for s in shot_audits)
    return ChapterSpeechAuditReport(
        schema=SPEECH_AUDIT_SCHEMA,
        chapter=chapter_id,
        ok=err_count == 0,
        shots=shot_audits,
        suggested_audio_source_override=overrides,
        subtitle_cues=all_cues,
        error_count=err_count,
        warning_count=warn_count,
    )


def build_auto_chapter_mix(
    chapter_report: ChapterSpeechAuditReport,
    *,
    shots: list[dict[str, Any]],
    ffmpeg_bin: str | None = None,
) -> dict[str, Any]:
    """Combine speech audit results and LUFS gain calibration into a `chapter_mix_and_subs.json` draft."""
    speech_by_id = {s.shot_id: s for s in chapter_report.shots}
    gains: dict[str, float] = {}
    for s in shots:
        sid_raw = str(s["id"])
        sid = sid_raw.zfill(2) if sid_raw.isdigit() else sid_raw
        sa = speech_by_id.get(sid)
        src_sid = chapter_report.suggested_audio_source_override.get(sid, sid)
        media_for_gain = Path(speech_by_id[src_sid].media_path) if src_sid in speech_by_id else Path(s["raw_video"])

        active_win: tuple[float, float] | None = None
        has_dlg = bool(sa and sa.expected_dialogue)
        if sa and sa.subtitle_cues:
            active_win = (
                min(float(c["local_start"]) for c in sa.subtitle_cues),
                max(float(c["local_end"]) for c in sa.subtitle_cues),
            )
        try:
            from .native_assembly import recommend_shot_gain_db_lufs

            rec = recommend_shot_gain_db_lufs(
                media_for_gain,
                has_dialogue=has_dlg,
                active_window=active_win,
                ffmpeg_bin=ffmpeg_bin,
            )
        except ImportError:
            rec = {"gain_db": 0.0}
        gains[sid] = rec["gain_db"]

    return {
        "chapter": chapter_report.chapter,
        "audited": chapter_report.ok,
        "auto_generated": True,
        "audio_source_override": dict(chapter_report.suggested_audio_source_override),
        "shot_gains_db": gains,
        "subtitle_cues": list(chapter_report.subtitle_cues),
        "speech_audit_summary": {
            "schema": chapter_report.schema,
            "ok": chapter_report.ok,
            "error_count": chapter_report.error_count,
            "warning_count": chapter_report.warning_count,
            "per_shot": {
                sa.shot_id: {
                    "ok": sa.ok,
                    "vad_total_sec": sa.vad_total_sec,
                    "cer": sa.cer,
                    "transcript": sa.transcript_text,
                    "errors": [e.code for e in sa.errors],
                    "warnings": [w.code for w in sa.warnings],
                }
                for sa in chapter_report.shots
            },
        },
    }
