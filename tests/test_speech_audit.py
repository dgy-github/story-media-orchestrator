"""Tests for story_media_orchestrator.speech_audit (VAD + ASR gates, CER, and clause alignment)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from story_media_orchestrator.cli import main as cli_main
from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.speech_audit import (
    SpeechSegment,
    TimedWord,
    TranscriptResult,
    TranscriptSegment,
    align_clauses_to_words,
    audit_chapter_speech,
    audit_shot_speech,
    build_auto_chapter_mix,
    compute_cer,
    split_dialogue_clauses,
    strip_whisper_boilerplate,
)


def _make_wav(path: Path, *, dur: float = 3.0, freq: int = 440, vol: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    src = f"sine=frequency={freq}:sample_rate=44100,volume={vol}" if vol > 0 else "anullsrc=channel_layout=stereo:sample_rate=44100"
    subprocess.run(
        [find_ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-t", f"{dur:.3f}", "-i", src, "-ar", "44100", "-ac", "2", str(path)],
        check=True,
    )
    return path


class FakeSpeechBackend:
    """Deterministic test backend returning scripted VAD segments and TranscriptResults."""

    def __init__(self, responses: list[tuple[list[SpeechSegment], TranscriptResult | None]]) -> None:
        self._responses = list(responses)
        self._vad_idx = 0
        self._current_tr: TranscriptResult | None = None

    def detect_speech_segments(self, audio_16k: np.ndarray, **kwargs) -> list[SpeechSegment]:
        segs, tr = self._responses[self._vad_idx]
        self._vad_idx += 1
        self._current_tr = tr
        return segs

    def transcribe(self, audio_16k: np.ndarray, **kwargs) -> TranscriptResult:
        tr = self._current_tr
        assert tr is not None, "Unexpected transcribe call when FakeSpeechBackend expected no transcribe"
        self._current_tr = None
        return tr


def test_split_dialogue_clauses_preserves_exact_text() -> None:
    for line in (
        "这冠太重。戴久了，就看不见地上的人。",
        "凡人之火，也想照进九天？",
        "陆南，你看。从今天起，是人的纪元了。",
    ):
        clauses = split_dialogue_clauses(line)
        assert "".join(clauses) == line

    assert split_dialogue_clauses("这冠太重。戴久了，就看不见地上的人。") == ["这冠太重。", "戴久了，", "就看不见地上的人。"]
    assert split_dialogue_clauses("凡人之火，也想照进九天？") == ["凡人之火，", "也想照进九天？"]
    assert split_dialogue_clauses("陆南，你看。从今天起，是人的纪元了。") == ["陆南，", "你看。", "从今天起，", "是人的纪元了。"]


def test_compute_cer_and_boilerplate_filter() -> None:
    cer_ok, recall_ok = compute_cer("戴久了，就看不见地上的人。", "戴久了就看不见地上的人")
    assert cer_ok == 0.0
    assert recall_ok == 1.0

    # 1-char homophone difference on 10 chars -> 0.1 CER
    cer_hom, _ = compute_cer("戴久了，就看不见地上的人。", "带久了，就看不见地上的人。")
    assert 0.08 <= cer_hom <= 0.12

    cer_bad, _ = compute_cer("守住这罐火", "风雪夜归人")
    assert cer_bad >= 0.8

    assert strip_whisper_boilerplate("字幕由Amara社区提供 谢谢观看 守住火种") == ""
    assert strip_whisper_boilerplate("谢谢观看。请不吝点赞订阅") == ""


def test_align_clauses_to_words_monotonic_and_accurate() -> None:
    clauses = ["这冠太重。", "戴久了，", "就看不见地上的人。"]
    words = [
        TimedWord("这", 0.60, 0.78),
        TimedWord("冠", 0.78, 1.00),
        TimedWord("太", 1.00, 1.22),
        TimedWord("重", 1.22, 1.45),
        TimedWord("带", 3.45, 3.68),  # homophone sub for 戴
        TimedWord("久", 3.68, 3.90),
        TimedWord("了", 3.90, 4.10),
        TimedWord("就看不见", 5.35, 5.95),
        TimedWord("地上的人", 5.95, 6.45),
    ]
    aligned = align_clauses_to_words(clauses, words, shot_duration=7.292)
    assert [a[0] for a in aligned] == clauses
    for idx, (_, st, en) in enumerate(aligned):
        assert 0.0 <= st < en <= 7.252
        if idx > 0:
            assert aligned[idx - 1][2] <= st
    assert 0.45 <= aligned[0][1] <= 0.65
    assert 3.30 <= aligned[1][1] <= 3.50
    assert 5.20 <= aligned[2][1] <= 5.40


def test_audit_shot_speech_silent_clean_vs_stray_speech_vs_flute_warning(tmp_path: Path) -> None:
    wav_clean = _make_wav(tmp_path / "clean.wav", dur=3.0, vol=0.0)
    be_clean = FakeSpeechBackend([([SpeechSegment(0.0, 0.1)], None)])  # < 0.25s VAD -> skip Whisper
    res_clean = audit_shot_speech(wav_clean, shot_id="03", expected_line=None, backend=be_clean)
    assert res_clean.ok is True
    assert res_clean.errors == []

    wav_stray = _make_wav(tmp_path / "stray.wav", dur=3.0, vol=0.5)
    tr_stray = TranscriptResult(
        text="大劫将至",
        language="zh",
        segments=(TranscriptSegment("大劫将至", 0.5, 1.8, no_speech_prob=0.1, avg_logprob=-0.4),),
        words=(),
    )
    be_stray = FakeSpeechBackend([([SpeechSegment(0.5, 1.8)], tr_stray)])
    res_stray = audit_shot_speech(wav_stray, shot_id="02", expected_line=None, backend=be_stray)
    assert res_stray.ok is False
    assert [e.code for e in res_stray.errors] == ["STRAY_SPEECH"]

    # Reed flute blip: short VAD but boilerplate/low-confidence ASR -> warning only, not hard fail
    tr_flute = TranscriptResult(
        text="谢谢观看",
        language="zh",
        segments=(TranscriptSegment("谢谢观看", 0.8, 1.25, no_speech_prob=0.8, avg_logprob=-1.5),),
        words=(),
    )
    be_flute = FakeSpeechBackend([([SpeechSegment(0.8, 1.22)], tr_flute)])
    res_flute = audit_shot_speech(wav_stray, shot_id="02", expected_line=None, backend=be_flute)
    assert res_flute.ok is True
    assert [w.code for w in res_flute.warnings] == ["POSSIBLE_STRAY_SPEECH"]


def test_audit_shot_speech_dialogue_cer_gates_and_cues(tmp_path: Path) -> None:
    wav_dlg = _make_wav(tmp_path / "dlg.wav", dur=4.0, vol=0.8)

    # Accurate voiceover
    tr_ok = TranscriptResult(
        text="凡人之火，也想照进九天？",
        language="zh",
        segments=(TranscriptSegment("凡人之火，也想照进九天？", 0.8, 3.2, 0.05, -0.2),),
        words=(TimedWord("凡人之火", 0.85, 1.65), TimedWord("也想照进九天", 2.05, 3.15)),
    )
    res_ok = audit_shot_speech(
        wav_dlg, shot_id="02", expected_line="凡人之火，也想照进九天？", speaker="苍白圣影", style="Voiceover",
        backend=FakeSpeechBackend([([SpeechSegment(0.8, 3.2)], tr_ok)]),
    )
    assert res_ok.ok is True
    assert res_ok.cer == 0.0
    assert len(res_ok.subtitle_cues) == 2
    assert all(c["style"] == "Voiceover" for c in res_ok.subtitle_cues)

    # Mild homophone warning (cer = 3/10 = 0.30 in (0.20, 0.45])
    tr_warn = TranscriptResult(
        text="繁人只活也想照进九天",
        language="zh",
        segments=(TranscriptSegment("繁人只活也想照进九天", 0.8, 3.2, 0.1, -0.5),),
        words=(TimedWord("繁人只活", 0.85, 1.65), TimedWord("也想照进九天", 2.05, 3.15)),
    )
    res_warn = audit_shot_speech(
        wav_dlg, shot_id="02", expected_line="凡人之火，也想照进九天？",
        backend=FakeSpeechBackend([([SpeechSegment(0.8, 3.2)], tr_warn)]),
    )
    assert res_warn.ok is True
    assert [w.code for w in res_warn.warnings] == ["DIALOGUE_CER_WARN"]

    # Severely wrong transcript -> DIALOGUE_CER_HIGH
    tr_bad = TranscriptResult(
        text="完全无关的胡言乱语台词测试文本",
        language="zh",
        segments=(TranscriptSegment("完全无关的胡言乱语台词测试文本", 0.8, 3.2, 0.1, -0.6),),
        words=(),
    )
    res_bad = audit_shot_speech(
        wav_dlg, shot_id="02", expected_line="凡人之火，也想照进九天？",
        backend=FakeSpeechBackend([([SpeechSegment(0.8, 3.2)], tr_bad)]),
    )
    assert res_bad.ok is False
    assert [e.code for e in res_bad.errors] == ["DIALOGUE_CER_HIGH"]

    # Missing speech -> MISSING_DIALOGUE_SPEECH
    res_miss = audit_shot_speech(
        wav_dlg, shot_id="02", expected_line="凡人之火，也想照进九天？",
        backend=FakeSpeechBackend([([], TranscriptResult("", "zh", (), ()))]),
    )
    assert res_miss.ok is False
    assert [e.code for e in res_miss.errors] == ["MISSING_DIALOGUE_SPEECH"]


def test_audit_chapter_speech_suggests_override_and_builds_auto_mix(tmp_path: Path) -> None:
    chap = tmp_path / "ch03"
    (chap / "audio").mkdir(parents=True)
    w1 = _make_wav(chap / "audio" / "shot_01_native.wav", dur=2.5, vol=0.2)
    w2 = _make_wav(chap / "audio" / "shot_02_native.wav", dur=2.5, vol=0.5)
    w3 = _make_wav(chap / "audio" / "shot_03_native.wav", dur=3.5, vol=1.0)

    audit_data = {
        "chapter": "03",
        "shots": [
            {"id": "01", "frames": 60, "raw_video": str(w1)},
            {"id": "02", "frames": 60, "raw_video": str(w2)},
            {
                "id": "03",
                "frames": 84,
                "raw_video": str(w3),
                "dialogue": {"speaker": "陆南", "mode": "visible", "line": "山门，我就不进了。"},
            },
        ],
    }
    (chap / "full_9shot_audit.json").write_text(json.dumps(audit_data, ensure_ascii=False), encoding="utf-8")

    be = FakeSpeechBackend([
        ([], None),  # S01 clean silent
        (
            [SpeechSegment(0.4, 1.8)],
            TranscriptResult("无关旁白", "zh", (TranscriptSegment("无关旁白", 0.4, 1.8, 0.1, -0.4),), ()),
        ),  # S02 stray speech
        (
            [SpeechSegment(0.6, 2.8)],
            TranscriptResult(
                "山门我就不进了",
                "zh",
                (TranscriptSegment("山门我就不进了", 0.6, 2.8, 0.05, -0.2),),
                (TimedWord("山门", 0.65, 1.25), TimedWord("我就不进了", 1.70, 2.75)),
            ),
        ),  # S03 dialogue
    ])
    rep = audit_chapter_speech(chap, backend=be)
    assert rep.ok is False
    assert rep.error_count == 1
    assert rep.suggested_audio_source_override == {"02": "01"}
    assert [c["text"] for c in rep.subtitle_cues] == ["山门，", "我就不进了。"]

    mix = build_auto_chapter_mix(rep, shots=audit_data["shots"])
    assert mix["audited"] is False
    assert mix["auto_generated"] is True
    assert mix["audio_source_override"] == {"02": "01"}
    assert set(mix["shot_gains_db"].keys()) == {"01", "02", "03"}
    assert mix["speech_audit_summary"]["per_shot"]["02"]["errors"] == ["STRAY_SPEECH"]


def test_cli_audit_speech_with_mocked_backend(tmp_path: Path, monkeypatch, capsys) -> None:
    import story_media_orchestrator.speech_audit as sa_mod

    wav_p = _make_wav(tmp_path / "line.wav", dur=2.5, vol=0.8)

    class _PassBackend:
        def __init__(self, **kwargs) -> None:
            pass

        def detect_speech_segments(self, audio_16k, **kwargs):
            return [SpeechSegment(0.4, 1.9)]

        def transcribe(self, audio_16k, **kwargs):
            return TranscriptResult(
                "只成人",
                "zh",
                (TranscriptSegment("只成人", 0.4, 1.9, 0.05, -0.2),),
                (TimedWord("只成人", 0.45, 1.85),),
            )

    monkeypatch.setattr(sa_mod, "FasterWhisperBackend", _PassBackend)
    json_out = tmp_path / "speech_out.json"
    rc = cli_main(["audit-speech", str(wav_p), "--expected-line", "只成人。", "--speaker", "陆南", "--json-out", str(json_out)])
    assert rc == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["ok"] is True
    assert captured["cer"] == 0.0
    assert json_out.is_file()
