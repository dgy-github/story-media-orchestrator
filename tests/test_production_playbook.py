from __future__ import annotations

import json
from pathlib import Path

import pytest

from story_media_orchestrator.cli import main as cli_main
from story_media_orchestrator.production_playbook import (
    ShotPromotionRecord,
    compute_asset_lineage_fingerprint,
    lint_handoff_manifest,
    resolve_anchored_timeline,
    triage_rough_cut_gap,
    validate_shot_handoff,
)


def _valid_handoff_dict() -> dict:
    return {
        "shot_id": "V5A1_06",
        "version": "V5",
        "insert_after": "E01_S04",
        "next_shot_id": "E01_S05",
        "story_role": "用阿石塞干粮建立兄弟人情与离村动机",
        "gap_evidence": "V4粗剪中离村偏突然，缺送行铺垫",
        "starting_pose": "土墙边站定，阿石双手捧布包粗饼面对陆南",
        "primary_action": "阿石把布包粗饼塞进陆南衣襟",
        "visible_result": "粗饼留在陆南怀中，陆南低头按住衣襟",
        "handoff_state": "陆南怀揣干粮面向村口，衔接离村镜头",
        "target_frames": 90,
        "character_refs": ["refs/ashi.png", "refs/lunan_s1.png"],
        "first_frame": "frames/V5A1_06.png",
        "peak_ref": "peaks/V5A1_06_peak.png",
        "video_mode": "i2va",
        "dialogue_or_voiceover": "阿石（正片对白）：路上带着。",
        "soundscape": "寒风、布料摩擦、远处犬吠",
        "acceptance_checks": ["核对两人身份与手部连续性", "确认粗饼确实进入衣襟"],
    }


def test_validate_shot_handoff_accepts_complete_spec_and_rejects_peak_as_first():
    ok_findings = validate_shot_handoff(_valid_handoff_dict())
    assert ok_findings == []

    bad = _valid_handoff_dict()
    bad["peak_ref"] = bad["first_frame"]
    bad["video_mode"] = "fl2va"
    bad["last_frame"] = None
    codes = {f.code for f in validate_shot_handoff(bad)}
    assert "PEAK_REF_USED_AS_FIRST_FRAME" in codes
    assert "FL2V_MISSING_LAST_FRAME" in codes


def test_promotion_record_enforces_four_gates_and_fingerprint_invalidation(tmp_path: Path):
    kf = tmp_path / "s01.png"
    kf.write_bytes(b"png-v1")
    fp1 = compute_asset_lineage_fingerprint(first_frame=kf, prompt_text="slash right", target_frames=90)

    rec = ShotPromotionRecord("V5A1_06")
    with pytest.raises(ValueError, match="before 'design_approved'"):
        rec.set_gate("keyframe_approved", passed=True)

    rec.set_gate("design_approved", passed=True, evidence="brief-v5")
    rec.set_gate("keyframe_approved", passed=True, evidence="sheet-act1")
    rec.set_gate("video_approved", passed=True, evidence="strip-01", input_fingerprint=fp1)
    rec.set_gate("master_included", passed=True, evidence="v5_master.mp4")
    assert rec.highest_stage() == "master_included"
    assert rec.is_video_cache_fresh(fp1)

    kf.write_bytes(b"png-v2")
    fp2 = compute_asset_lineage_fingerprint(first_frame=kf, prompt_text="slash right", target_frames=90)
    assert fp2 != fp1
    assert not rec.is_video_cache_fresh(fp2)
    cleared = rec.invalidate_for_input_change(fp2)
    assert cleared == ["video_approved", "master_included"]
    assert rec.highest_stage() == "keyframe_approved"


def test_resolve_anchored_timeline_preserves_baseline_and_chains_insertions():
    baseline = ["E01_S01", "E01_S04", "E01_S05"]
    insertions = [
        {"shot_id": "V5A1_06", "insert_after": "E01_S04"},
        {"shot_id": "V5A1_07", "insert_after": "E01_S04"},
        {"shot_id": "V5A1_07B", "insert_after": "V5A1_07"},
        {"shot_id": "CARD_ACT1", "insert_before": "E01_S01"},
    ]
    assert resolve_anchored_timeline(baseline, insertions) == [
        "CARD_ACT1",
        "E01_S01",
        "E01_S04",
        "V5A1_06",
        "V5A1_07",
        "V5A1_07B",
        "E01_S05",
    ]


def test_triage_rough_cut_gap_routes_to_earliest_stage_without_full_act_regen():
    result = triage_rough_cut_gap(
        "keyframe_asset_defect",
        version="V5",
        shot_id="V5A3_19",
        missing_for_viewer="冷灶被画成人物与明火，偏离空灶叙事",
        timecode="09:12",
        neighbor_shots=["E12_S02", "V5A3_20"],
    )
    assert result["earliest_stage"] == "keyframe_png"
    assert result["allow_full_act_regen"] is False
    assert result["neighbor_shots"] == ["E12_S02", "V5A3_20"]


def test_cli_lint_handoff_validates_manifest_and_timeline(tmp_path: Path, capsys):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "V5A1_06.png").write_bytes(b"fake-png")
    manifest = {
        "baseline_ids": ["E01_S04", "E01_S05"],
        "shots": [_valid_handoff_dict()],
    }
    mp = tmp_path / "handoff_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    report = lint_handoff_manifest(mp, require_first_frame_exists=True)
    assert report["ok"] is True
    assert report["resolved_timeline"] == ["E01_S04", "V5A1_06", "E01_S05"]

    rc = cli_main(["lint-handoff", str(mp), "--check-files"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
