"""Tests for story_media_orchestrator.h3_bundle contract, linter, sanitizer, and ComfyUI verification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from story_media_orchestrator.cli import main as cli_main
from story_media_orchestrator.h3_bundle import (
    compute_shot_input_fingerprint,
    sanitize_silent_soundscape,
    slow_breathing_frames,
    validate_h3_bundle,
    verify_comfyui_history_entry,
)


def _make_img(path: Path, size: tuple[int, int] = (1280, 720), color: tuple[int, int, int] = (90, 110, 130)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def _write_valid_bundle(bundle_dir: Path) -> Path:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    _make_img(bundle_dir / "keyframes" / "01.png")
    _make_img(bundle_dir / "keyframes" / "02.png")
    _make_img(bundle_dir / "keyframes" / "03.png")
    payload = {
        "title": "第01章 测试卷",
        "chapter": "01",
        "fps": 24,
        "shots": [
            {
                "id": "01",
                "title": "静室火罐",
                "frames": 124,
                "first_frame": "keyframes/01.png",
                "action": "缓慢推近火罐",
                "audio": "炉灰轻响。",
                "h3_prompt": "Slow push onto an earthen brazier. overall_soundscape: soft charcoal crackle and wind.",
            },
            {
                "id": "02",
                "title": "低声嘱咐",
                "frames": 141,
                "first_frame": "keyframes/02.png",
                "action": "人物开口说话",
                "audio": "人声贴近。",
                "dialogue": {"id": "S1", "speaker": "陆南", "mode": "visible", "line": "守住这罐火。"},
                "h3_prompt": 'Lu Nan speaks softly: <d>[Chinese] 守住这罐火。</d> overall_soundscape: faint wind.',
            },
            {
                "id": "03",
                "title": "传音入耳",
                "frames": 158,
                "first_frame": "keyframes/03.png",
                "action": "老羊闭口静卧",
                "audio": "画外传音。",
                "dialogue": {"id": "S2", "speaker": "老黄羊", "mode": "off_screen_voiceover", "line": "早点回来。"},
                "h3_prompt": (
                    "Old goat rests still; its lips remain closed. An off-screen voice says: "
                    "<d>早点回来。</d> overall_soundscape: mountain breeze."
                ),
            },
        ],
    }
    (bundle_dir / "h3_prompts.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle_dir


def test_validate_h3_bundle_passes_and_computes_deterministic_fingerprint(tmp_path: Path) -> None:
    bdir = _write_valid_bundle(tmp_path / "bundle_ok")
    rep1 = validate_h3_bundle(bdir, require_chapter_field=True)
    rep2 = validate_h3_bundle(bdir, require_chapter_field=True)
    assert rep1.ok is True
    assert rep1.errors == []
    assert rep1.warnings == []
    assert rep1.shot_count == 3
    assert rep1.bundle_fingerprint is not None
    assert rep1.bundle_fingerprint == rep2.bundle_fingerprint

    fp_a = compute_shot_input_fingerprint("prompt A", bdir / "keyframes" / "01.png", target_frames=141)
    fp_b = compute_shot_input_fingerprint("prompt A", bdir / "keyframes" / "01.png", target_frames=158)
    assert len(fp_a) == 20
    assert fp_a != fp_b


def test_slow_breathing_frames_grid_and_clamp() -> None:
    assert slow_breathing_frames(124) == 141
    assert slow_breathing_frames(158) == 175
    assert slow_breathing_frames(175) == 175
    with pytest.raises(ValueError, match="17k\\+5"):
        slow_breathing_frames(120)


@pytest.mark.parametrize(
    ("mutator", "expected_code", "in_warnings"),
    [
        (
            lambda p: p["shots"][1].update({"h3_prompt": "Says <d>完全不同的台词</d>."}),
            "DIALOGUE_PROMPT_MISMATCH",
            False,
        ),
        (
            lambda p: p["shots"][1].update({"h3_prompt": "Mouth moves silently with no d tag."}),
            "MISSING_D_TAG",
            False,
        ),
        (
            lambda p: p["shots"][0].update({"h3_prompt": "Hallucinated <d>多余台词</d> in silent shot."}),
            "UNEXPECTED_D_TAG",
            False,
        ),
        (
            lambda p: p["shots"][0].update(
                {"h3_prompt": "Quiet room. overall_soundscape: soft breeze and low indistinct conversation without intelligible words."}
            ),
            "SILENT_SOUNDSCAPE_SPEECH_TRIGGER",
            False,
        ),
        (
            lambda p: p["shots"][2].update({"h3_prompt": "Goat looks ahead: <d>早点回来。</d>"}),
            "VOICEOVER_MISSING_CLOSED_LIPS_GUARD",
            True,
        ),
        (
            lambda p: p["shots"][0].update({"h3_prompt": "Warm soil replaces the high reflected gold across his robe."}),
            "CONTRASTIVE_COLOR_TRAP",
            True,
        ),
        (
            lambda p: p["shots"][0].update({"first_frame": "keyframes/missing.png"}),
            "MISSING_KEYFRAME",
            False,
        ),
    ],
)
def test_validate_h3_bundle_catches_contract_and_prompt_defects(
    tmp_path: Path, mutator, expected_code: str, in_warnings: bool
) -> None:
    bdir = _write_valid_bundle(tmp_path / f"bundle_{expected_code}")
    pj = bdir / "h3_prompts.json"
    data = json.loads(pj.read_text(encoding="utf-8"))
    mutator(data)
    pj.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    rep = validate_h3_bundle(bdir)
    codes = [i.code for i in (rep.warnings if in_warnings else rep.errors)]
    assert expected_code in codes
    assert rep.ok is in_warnings


def test_validate_h3_bundle_catches_blank_and_bad_aspect_keyframe(tmp_path: Path) -> None:
    bdir = _write_valid_bundle(tmp_path / "bundle_bad_img")
    _make_img(bdir / "keyframes" / "01.png", size=(1280, 720), color=(0, 0, 0))
    _make_img(bdir / "keyframes" / "02.png", size=(720, 1280), color=(80, 80, 80))

    rep = validate_h3_bundle(bdir)
    err_codes = {e.code for e in rep.errors}
    assert "BLANK_KEYFRAME" in err_codes
    assert "ASPECT_RATIO_MISMATCH" in err_codes


def test_sanitize_silent_soundscape_only_cleans_silent_shots() -> None:
    raw = "Wind blows. overall_soundscape: soft paper whisper, and low indistinct conversation without intelligible words.\n\nnon_diegetic_music: N/A"
    cleaned, rules = sanitize_silent_soundscape(raw, has_dialogue=False)
    assert "paper rustle" in cleaned
    assert "indistinct conversation" not in cleaned
    assert set(rules) >= {"paper_whisper", "indistinct_conversation"}

    unchanged, rules_dlg = sanitize_silent_soundscape(raw, has_dialogue=True)
    assert unchanged == raw
    assert rules_dlg == []


def test_verify_comfyui_history_entry_checks_exact_quality_contract() -> None:
    prompt_txt = "Exact prompt <d>台词</d>"
    hist = {
        "pid-1": {
            "prompt": [
                42,
                "pid-1",
                {
                    "5": {"inputs": {"prompt": prompt_txt, "width": 1024, "height": 576, "length": 141}},
                    "8": {"inputs": {"steps": 20}},
                    "22": {"inputs": {"width": 1920, "height": 1080}},
                },
            ]
        }
    }
    res = verify_comfyui_history_entry(hist, expected_prompt=prompt_txt, expected_length=141)
    assert res["prompt_id"] == "pid-1"
    assert res["steps"] == 20
    assert res["turbo"] is False

    bad_hist = json.loads(json.dumps(hist))
    bad_hist["pid-1"]["prompt"][2]["15"] = {"inputs": {}}
    with pytest.raises(ValueError, match="quality mismatch"):
        verify_comfyui_history_entry(bad_hist, expected_prompt=prompt_txt, expected_length=141)


def test_cli_lint_h3_bundle_exit_codes(tmp_path: Path, capsys) -> None:
    ok_dir = _write_valid_bundle(tmp_path / "cli_ok")
    assert cli_main(["lint-h3-bundle", str(ok_dir)]) == 0
    out_ok = json.loads(capsys.readouterr().out)
    assert out_ok["ok"] is True

    bad_dir = _write_valid_bundle(tmp_path / "cli_bad")
    pj = bad_dir / "h3_prompts.json"
    d = json.loads(pj.read_text(encoding="utf-8"))
    d["shots"][0]["frames"] = 120
    pj.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert cli_main(["lint-h3-bundle", str(bad_dir)]) == 1
    out_bad = json.loads(capsys.readouterr().out)
    assert out_bad["ok"] is False
    assert any(e["code"] == "INVALID_H3_FRAME_GRID" for e in out_bad["errors"])


def test_validate_h3_bundle_accepts_direct_review_json_file(tmp_path: Path) -> None:
    bdir = _write_valid_bundle(tmp_path / "review_pack")
    review_json = bdir / "ch01-h3-review.json"
    review_json.write_text((bdir / "h3_prompts.json").read_text(encoding="utf-8"), encoding="utf-8")
    (bdir / "h3_prompts.json").unlink()

    rep = validate_h3_bundle(review_json, require_chapter_field=True)
    assert rep.ok is True
    assert rep.shot_count == 3
    assert rep.bundle_dir == str(review_json)
