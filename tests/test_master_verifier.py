"""Tests for story_media_orchestrator.master_verifier using real synthetic FFmpeg streams."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.master_verifier import (
    reconcile_subtitles_with_bundles,
    verify_master_release,
)


def _ff(args: list[str]) -> None:
    subprocess.run([find_ffmpeg(), "-y", "-v", "error", *args], check=True)


def _make_clip(path: Path, *, dur: float, color: str, vol: float | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if vol is None or vol <= 0.0:
        a_src = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    else:
        a_src = ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", f"sine=frequency=440:sample_rate=44100,volume={vol}"]
    _ff([
        "-f", "lavfi", "-i", f"color=c={color}:s=640x360:r=24:d={dur:.3f}",
        *a_src,
        "-r", "24", "-vsync", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        str(path),
    ])
    return path


def _concat_clips(clips: list[Path], out: Path) -> Path:
    inputs: list[str] = []
    filt = []
    pairs = []
    for i, c in enumerate(clips):
        inputs.extend(["-i", str(c)])
        filt.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}]")
        pairs.append(f"[v{i}][a{i}]")
    fc = ";".join(filt) + ";" + "".join(pairs) + f"concat=n={len(clips)}:v=1:a=1[vout][aout]"
    _ff([
        *inputs, "-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
        "-r", "24", "-vsync", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
        str(out),
    ])
    return out


def test_verify_master_release_passes_clean_card_plus_body_and_tracks_human_ear_flag(tmp_path: Path) -> None:
    card = _make_clip(tmp_path / "card.mp4", dur=2.0, color="black", vol=None)
    body = _make_clip(tmp_path / "body.mp4", dur=3.0, color="steelblue", vol=0.08)
    combined = _concat_clips([card, body], tmp_path / "clean_master.mp4")

    rep = verify_master_release(
        combined,
        profile="archive",
        expected_frames=120,
        expected_width=640,
        expected_height=360,
        expected_fps=24.0,
        title_card_windows=[(0.0, 2.0)],
        global_cues=[{"chapter": "01", "shot_id": "01", "style": "Dialogue", "text": "测试句", "start": 2.3, "end": 4.5}],
    )
    assert rep.ok is True, rep.to_dict()
    assert rep.metrics["black_window_count"] >= 1
    assert rep.metrics["dialogue_consistency"]["count"] == 1
    assert rep.human_ear_check_points
    assert all(pt["audited_by_human"] is False for pt in rep.human_ear_check_points)


def test_verify_master_release_catches_midclip_black_silence_and_frame_mismatch(tmp_path: Path) -> None:
    c1 = _make_clip(tmp_path / "c1.mp4", dur=2.0, color="steelblue", vol=0.1)
    bad_mid = _make_clip(tmp_path / "bad_mid.mp4", dur=2.5, color="black", vol=None)
    c2 = _make_clip(tmp_path / "c2.mp4", dur=2.0, color="firebrick", vol=0.1)
    bad = _concat_clips([c1, bad_mid, c2], tmp_path / "bad_master.mp4")

    rep = verify_master_release(
        bad,
        profile="archive",
        expected_frames=999,
        expected_width=640,
        expected_height=360,
        expected_fps=24.0,
        title_card_windows=[],
    )
    codes = {e.code for e in rep.errors}
    assert rep.ok is False
    assert "FRAME_COUNT_MISMATCH" in codes
    assert "UNEXPECTED_BLACK_SCREEN" in codes
    assert "UNEXPECTED_AUDIO_SILENCE" in codes


def test_verify_master_release_video_md5_and_web_loudness_bounds(tmp_path: Path) -> None:
    src = _make_clip(tmp_path / "src.mp4", dur=2.5, color="forestgreen", vol=1.0)
    copied = tmp_path / "copied_audio_changed.mp4"
    _ff(["-i", str(src), "-c:v", "copy", "-af", "volume=-15dB", "-c:a", "aac", "-b:a", "128k", str(copied)])

    # Exact video copy passes MD5 check even when audio changed; web profile catches too-low loudness
    rep_web = verify_master_release(
        copied,
        profile="web",
        expected_frames=60,
        expected_width=640,
        expected_height=360,
        expected_fps=24.0,
        title_card_windows=[],
        reference_video_mp4=src,
    )
    assert rep_web.metrics["video_md5"]["match"] is True
    assert "LOUDNESS_TOO_LOW" in {e.code for e in rep_web.errors}

    # Re-encoded video must fail VIDEO_MD5_MISMATCH
    reenc = tmp_path / "reencoded.mp4"
    _ff(["-i", str(src), "-c:v", "libx264", "-crf", "28", "-c:a", "copy", str(reenc)])
    rep_md5 = verify_master_release(
        reenc,
        profile="archive",
        expected_frames=60,
        expected_width=640,
        expected_height=360,
        expected_fps=24.0,
        title_card_windows=[],
        reference_video_mp4=src,
    )
    assert "VIDEO_MD5_MISMATCH" in {e.code for e in rep_md5.errors}

    # Clipping peak fails TRUE_PEAK_EXCEEDED under web profile
    clipped = tmp_path / "clipped.mp4"
    _ff(["-i", str(src), "-c:v", "copy", "-af", "volume=+22dB", "-c:a", "aac", "-b:a", "128k", str(clipped)])
    rep_tp = verify_master_release(
        clipped,
        profile="web",
        expected_frames=60,
        expected_width=640,
        expected_height=360,
        expected_fps=24.0,
        title_card_windows=[],
    )
    assert "TRUE_PEAK_EXCEEDED" in {e.code for e in rep_tp.errors}


def test_reconcile_subtitles_with_bundles_flags_mismatch(tmp_path: Path) -> None:
    b1 = tmp_path / "ch01"
    b1.mkdir()
    (b1 / "h3_prompts.json").write_text(
        json.dumps({"shots": [{"id": "01", "h3_prompt": "Says <d>守住火种。</d>"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    ok_findings = reconcile_subtitles_with_bundles(
        [{"chapter": "01", "subtitle_cues": [{"shot_id": "01", "text": "守住火种"}]}],
        {"01": b1},
    )
    assert ok_findings == []

    bad_findings = reconcile_subtitles_with_bundles(
        [{"chapter": "01", "subtitle_cues": [{"shot_id": "01", "text": "改了台词"}]}],
        {"01": b1},
    )
    assert len(bad_findings) == 1
    assert bad_findings[0].code == "SUBTITLE_SCRIPT_MISMATCH"
