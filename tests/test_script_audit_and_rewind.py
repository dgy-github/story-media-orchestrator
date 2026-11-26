from __future__ import annotations

import asyncio
import json
from pathlib import Path

from story_media_orchestrator.cli import main as cli_main
from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.mcp_server import create_mcp_server
from story_media_orchestrator.script_audit import (
    audit_project_script,
    evaluate_rework_escalation_to_script,
    insert_script_bridge_shot,
    patch_shot_script_beat,
    rehearse_project_script,
)


def test_script_audit_catches_six_xinhuo_failure_modes(tmp_path):
    proj = tmp_path / "script_audit_proj"
    proj.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest.create(
        "script_audit_proj",
        "他内心感到绝望与复杂的情绪。漫天光效与全屏亮光瞬间吞噬天地。剑已刺入僵持不下，然后转身接着跳跃随后挥剑最后再落地。",
    )
    manifest.shots[0].scene_context = {"visual_action": '角色说“我绝不放弃”'}
    manifest.save(proj / "project.json")

    report = audit_project_script(proj)
    assert report["ok"] is False
    all_codes = {f["code"] for s in report["shots"] for f in s["findings"]}
    assert "ABSTRACT_INTERIOR_WITHOUT_ACTION" in all_codes
    assert "VAGUE_COMBAT_OR_VFX" in all_codes
    assert "RAW_DIALOGUE_IN_VISUAL_ACTION" in all_codes
    assert "PEAK_POSE_AS_START_FRAME" in all_codes
    assert "MISSING_SCENE_ANCHORS" in all_codes


def test_video_failure_rewind_to_script_patch_and_bridge_insert(tmp_path, capsys):
    proj = tmp_path / "rewind_proj"
    proj.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest.create("rewind_proj", "陆南感到悲痛。陆南冲向石门。")
    s1, s2 = manifest.shots[0], manifest.shots[1]
    s1.assets = {"image": "frames/0001.png", "video": "video/0001.mp4", "audio": "audio/0001.wav"}
    s1.quality_report = {"decision": "failed", "failure_code": "VIDEO_MOTION_FAIL", "triage_category": "motion_lock"}
    s1.rework_history = [{"round": 1}, {"round": 2}]
    s2.assets = {"image": "frames/0002.png", "video": "video/0002.mp4"}
    s2.quality_report = {"decision": "passed"}
    manifest.save(proj / "project.json")

    esc = evaluate_rework_escalation_to_script(s1)
    assert esc["should_rewind_to_script"] is True
    assert esc["recommended_tool"] == "story_patch_script_beat"

    # Patch s1 upstream script -> invalidates s1 visuals only, preserves s1 audio & s2 untouched.
    patched = patch_shot_script_beat(
        proj,
        s1.id,
        new_text="陆南低头攥紧母亲留下的布囊，把最后半块干粮塞进阿石手里。",
        visual_action="陆南攥紧布囊，将半块粗饼塞入阿石衣襟",
        framing="近景",
        location="雨夜石寨土墙边",
        continuity="第一阶段黑发粗衣",
        reason="将抽象悲痛改为可见道具交接动作",
    )
    assert patched["patched"] is True
    assert "image" in patched["invalidated_assets"] and "video" in patched["invalidated_assets"]
    assert "audio" not in patched["invalidated_assets"]
    assert patched["post_audit"]["ok"] is True

    reloaded = ProjectManifest.load(proj / "project.json")
    assert "audio" in reloaded.shot(s1.id).assets
    assert reloaded.shot(s2.id).assets["video"] == "video/0002.mp4"
    assert reloaded.shot(s2.id).quality_report["decision"] == "passed"

    # Insert causal bridge shot after s1 without disturbing s2's ID or assets.
    inserted = insert_script_bridge_shot(
        proj,
        s1.id,
        text="阿石眼眶发红，一把按住陆南手腕，又慢慢松开后退半步。",
        framing="中近景",
        location="雨夜石寨土墙边",
        story_role="reaction_handoff",
    )
    assert inserted["inserted_shot_id"] == f"{s1.id}-b1"
    assert inserted["total_shots"] == 3
    reloaded2 = ProjectManifest.load(proj / "project.json")
    assert [s.id for s in reloaded2.shots] == [s1.id, f"{s1.id}-b1", s2.id]
    assert reloaded2.shot(s2.id).assets["video"] == "video/0002.mp4"

    # Verify CLI + MCP profiles expose the new tools
    core_tools = {t.name for t in asyncio.run(create_mcp_server(profile="core").list_tools())}
    critic_tools = {t.name for t in asyncio.run(create_mcp_server(profile="critic").list_tools())}
    for tool in ("story_audit_script", "story_patch_script_beat", "story_insert_bridge_shot"):
        assert tool in core_tools and tool in critic_tools

    rc = cli_main(["audit-script", str(proj)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 or "average_score" in out


def test_script_rehearsal_combines_genre_gaps_and_demo_flywheel_lessons(tmp_path, capsys):
    fw_dir = tmp_path / "demo_flywheel"
    fw_dir.mkdir(parents=True, exist_ok=True)
    (fw_dir / "playbook.json").write_text(
        json.dumps(
            {
                "schema": "flywheel-playbook/v1",
                "rules": [
                    {
                        "failure_code": "VFX_QUALITY_DEFECT",
                        "title": "禁止全屏亮光敷衍法术交锋",
                        "action": "写清法术来源、接触点与材质受光",
                    }
                ],
                "exemplars": [
                    {
                        "failure_code": "VIDEO_MOTION_FAIL",
                        "defect": "首帧直接摆峰值僵持导致对砍锁死",
                        "fix": "首帧前移至起势态并拆三拍",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    proj = tmp_path / "xianxia_rehearsal"
    proj.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest.create("xianxia_rehearsal", "少年内心感到复杂的情绪。全屏亮光瞬间吞噬天空。")
    manifest.flywheel_path = str(fw_dir)
    manifest.save(proj / "project.json")

    rep = rehearse_project_script(proj, genre="仙侠")
    assert rep["genre"] == "xianxia"
    assert rep["ready_for_production"] is False
    assert rep["p0_blocking_count"] >= 2
    assert rep["demo_lessons_count"] == 2
    assert len(rep["genre_gaps"]) >= 2
    assert any(s["recommended_tool"] == "story_insert_bridge_shot" for s in rep["repair_suggestions"])
    assert "历史成片 Demo 教训" in rep["rehearsal_brief_prompt"]

    core_tools = {t.name for t in asyncio.run(create_mcp_server(profile="core").list_tools())}
    assert "story_rehearse_script" in core_tools

    rc = cli_main(["rehearse-script", str(proj), "--genre", "xianxia"])
    cli_out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert cli_out["genre"] == "xianxia"
