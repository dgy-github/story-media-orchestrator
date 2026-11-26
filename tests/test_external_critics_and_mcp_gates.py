from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
import pytest

from story_media_orchestrator.cli import main as cli_main
from story_media_orchestrator.external_critics import (
    CriticSpec,
    reconcile_panel_verdicts,
    resolve_critic_panel_from_env,
    run_external_critic_panel,
    run_project_critic_panel,
)
from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.mcp_server import create_mcp_server
from story_media_orchestrator.supervisor import (
    TakeoverConflictError,
    claim_takeover,
    inject_external_asset,
    release_takeover,
    submit_quality_verdict,
)


def _png_bytes(rgb: tuple[int, int, int] = (70, 120, 185), *, width: int = 320, height: int = 180) -> bytes:
    img = Image.new("RGB", (width, height), rgb)
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, width - 24, height - 24), fill=(220, 175, 95), outline=(25, 35, 55), width=4)
    draw.line((24, 24, width - 24, height - 24), fill=(255, 255, 255), width=3)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_project(tmp_path: Path, name: str = "panel_proj") -> tuple[Path, ProjectManifest]:
    proj = tmp_path / name
    proj.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest.create(name, "林雾立于雨夜长街，抬头望向霓虹。")
    frame = proj / "frames" / "0001.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(_png_bytes())
    manifest.shots[0].assets["image"] = str(frame)
    manifest.save(proj / "project.json")
    return proj, manifest


def test_role_scoped_takeover_leases_allow_concurrent_producer_and_critic(tmp_path):
    proj, _ = _make_project(tmp_path, "concurrent_roles")
    src_img = tmp_path / "new_frame.png"
    src_img.write_bytes(_png_bytes((40, 160, 110)))

    # Codex holds asset_producer, Gemini 4 holds critic simultaneously without conflict.
    prod_lease = claim_takeover(proj, supervisor_id="codex-image", role="asset_producer")
    critic_lease = claim_takeover(proj, supervisor_id="gemini-4", role="critic")
    assert prod_lease["role"] == "asset_producer"
    assert critic_lease["role"] == "critic"

    # Producer can inject with its token; wrong token is blocked.
    with pytest.raises(TakeoverConflictError):
        inject_external_asset(proj, "shot-01", asset_role="image", source_path=str(src_img), lease_token=critic_lease["lease_token"])
    inj = inject_external_asset(proj, "shot-01", asset_role="image", source_path=str(src_img), lease_token=prod_lease["lease_token"])
    assert inj["shot_id"] == "shot-01"
    assert inj["asset_role"] == "image"
    assert Path(inj["path"]).is_file()

    # Critic can submit verdict with its token; wrong token is blocked.
    with pytest.raises(TakeoverConflictError):
        submit_quality_verdict(proj, "shot-01", stage="image", decision="passed", lease_token=prod_lease["lease_token"])
    verdict = submit_quality_verdict(
        proj,
        "shot-01",
        stage="image",
        decision="passed",
        scores={"character": 0.92},
        supervisor_id="gemini-4",
        lease_token=critic_lease["lease_token"],
    )
    assert verdict["verdict"]["decision"] == "passed"

    # Release both role leases cleanly.
    assert release_takeover(proj, prod_lease["lease_token"])["released"] is True
    assert release_takeover(proj, critic_lease["lease_token"])["released"] is True

    # Exclusive supervisor lock blocks role-scoped claims.
    sup_lease = claim_takeover(proj, supervisor_id="gemini-4-supervisor", role="supervisor")
    with pytest.raises(TakeoverConflictError):
        claim_takeover(proj, supervisor_id="codex-image", role="asset_producer")
    assert release_takeover(proj, sup_lease["lease_token"])["released"] is True


def test_critic_panel_escalation_and_reconciliation(tmp_path):
    proj, manifest = _make_project(tmp_path, "escalation_proj")
    calls: list[str] = []

    def primary_pass(**_kwargs):
        calls.append("primary")
        return {"decision": "passed", "scores": {"character": 0.9}, "reason": "清晰稳定", "supervisor_id": "gemini-4"}

    def tiebreaker_should_not_run(**_kwargs):
        calls.append("tiebreaker")
        return {"decision": "passed", "reason": "ok", "supervisor_id": "grok-3"}

    # 1) Clean primary pass under escalate policy -> tie-breaker not called.
    res1 = run_external_critic_panel(
        proj,
        "shot-01",
        stage="image",
        policy="escalate",
        primary_critic=primary_pass,
        tiebreaker_critic=tiebreaker_should_not_run,
    )
    assert res1["decision"] == "passed"
    assert res1["panel"]["tiebreaker_triggered"] is False
    assert calls == ["primary"]

    # 2) Primary warning -> escalates to tie-breaker (Grok passes -> final passed).
    calls.clear()

    def primary_warn(**_kwargs):
        calls.append("primary")
        return {"decision": "warning", "reason": "轻微光影波动", "supervisor_id": "gemini-4"}

    def tiebreaker_pass(**_kwargs):
        calls.append("tiebreaker")
        return {"decision": "passed", "scores": {"character": 0.88}, "reason": "整体达标", "supervisor_id": "grok-3"}

    res2 = run_external_critic_panel(
        proj,
        "shot-01",
        stage="image",
        policy="escalate",
        primary_critic=primary_warn,
        tiebreaker_critic=tiebreaker_pass,
        persist_verdict=True,
    )
    assert calls == ["primary", "tiebreaker"]
    assert res2["decision"] == "passed"
    assert res2["panel"]["tiebreaker_triggered"] is True
    assert "primary_warning" in res2["panel"]["escalation_reasons"]
    assert res2["persisted"]["verdict"]["decision"] == "passed"

    # 3) Hard-defect disagreement (primary passed, tie-breaker failed with VFX_QUALITY_DEFECT) -> conservative failed.
    merged = reconcile_panel_verdicts(
        {"decision": "passed", "failure_code": "PASS", "supervisor_id": "gemini-4", "reason": "looks ok"},
        {"decision": "failed", "failure_code": "VFX_QUALITY_DEFECT", "supervisor_id": "grok-3", "reason": "贴片光效"},
        policy="always",
        escalation_reasons=["policy_always"],
    )
    assert merged["decision"] == "failed"
    assert merged["failure_code"] == "VFX_QUALITY_DEFECT"

    # 4) Rework history >= 2 triggers escalation even when primary passes.
    manifest = ProjectManifest.load(proj / "project.json")
    manifest.shots[0].rework_history = [{"round": 1}, {"round": 2}]
    manifest.save(proj / "project.json")
    calls.clear()
    res4 = run_external_critic_panel(
        proj,
        "shot-01",
        stage="image",
        policy="escalate",
        primary_critic=primary_pass,
        tiebreaker_critic=tiebreaker_pass,
    )
    assert calls == ["primary", "tiebreaker"]
    assert "rework_round_ge_2" in res4["panel"]["escalation_reasons"]


def test_env_resolution_and_gemini_native_and_grok_protocols(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gem-key")
    monkeypatch.setenv("STORY_GEMINI_CRITIC_MODEL", "gemini-4-pro")
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    monkeypatch.setenv("STORY_GROK_CRITIC_MODEL", "grok-3-vision")
    for k in (
        "STORY_PRIMARY_CRITIC_PROTOCOL",
        "STORY_PRIMARY_CRITIC_API_KEY",
        "STORY_PRIMARY_CRITIC_BASE_URL",
        "STORY_TIEBREAKER_CRITIC_PROTOCOL",
        "STORY_TIEBREAKER_CRITIC_API_KEY",
        "STORY_TIEBREAKER_CRITIC_BASE_URL",
    ):
        monkeypatch.delenv(k, raising=False)

    primary, tiebreaker = resolve_critic_panel_from_env()
    assert primary is not None and primary.protocol == "gemini_native" and primary.model == "gemini-4-pro"
    assert tiebreaker is not None and tiebreaker.protocol == "openai_compat" and tiebreaker.model == "grok-3-vision"

    proj, _ = _make_project(tmp_path, "proto_proj")
    seen_urls: list[str] = []

    class _FakeResp:
        def __init__(self, payload: dict):
            self._raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def read(self) -> bytes:
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=75):
        seen_urls.append(req.full_url)
        if "generativelanguage.googleapis.com" in req.full_url:
            assert req.get_header("X-goog-api-key") == "test-gem-key"
            return _FakeResp(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": json.dumps(
                                            {
                                                "decision": "warning",
                                                "failure_code": "NONE",
                                                "scores": {"character": 0.78},
                                                "reason": "请二审复核发色",
                                            },
                                            ensure_ascii=False,
                                        )
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
        assert "api.x.ai" in req.full_url
        return _FakeResp(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "passed",
                                    "failure_code": "PASS",
                                    "scores": {"character": 0.91},
                                    "reason": "发色符合当前阶段母版",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }
        )

    with patch("story_media_orchestrator.external_critics.urllib_request.urlopen", side_effect=fake_urlopen):
        result = run_external_critic_panel(proj, "shot-01", stage="image", policy="escalate")

    assert len(seen_urls) == 2
    assert result["decision"] == "passed"
    assert result["panel"]["tiebreaker_triggered"] is True


def test_mcp_exposes_hard_gates_and_critic_panel_and_cli(tmp_path, capsys):
    proj, _ = _make_project(tmp_path, "mcp_gates_proj")
    server = create_mcp_server()
    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}
    for expected in (
        "story_lint_handoff",
        "story_lint_h3_bundle",
        "story_audit_speech",
        "story_verify_master",
        "story_run_critic_panel",
        "story_run_project_critic_panel",
    ):
        assert expected in tool_names

    # Invoke story_lint_handoff via MCP tool
    handoff_json = tmp_path / "handoff_manifest.json"
    handoff_json.write_text(
        json.dumps(
            {
                "version": "v1",
                "chapter": "ch01",
                "shots": [
                    {
                        "shot_id": "shot-01",
                        "first_frame": str(proj / "frames" / "0001.png"),
                        "action_beat": "抬头望向霓虹",
                        "speech_delivery": "silent",
                        "states": {
                            "design": "passed",
                            "still": "passed",
                            "video": "pending",
                            "master": "pending",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    res = asyncio.run(server.call_tool("story_lint_handoff", {"manifest_path": str(handoff_json), "check_files": True}))
    content_blocks = res.content if hasattr(res, "content") else (res[0] if isinstance(res, tuple) else res)
    payload = json.loads(content_blocks[0].text)
    assert "ok" in payload

    # Test CLI critic-panel when no external critic env is configured (graceful unavailable -> exit 1)
    rc = cli_main(["critic-panel", str(proj), "shot-01", "--stage", "image"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert rc == 1
    assert parsed["decision"] == "unavailable"


def test_project_critic_panel_batch_sweep_and_concurrent_producer_lease(tmp_path, capsys):
    proj = tmp_path / "batch_proj"
    proj.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest.create("batch_proj", "林雾推门走进钟楼走廊。她停下脚步看向墙上的旧钟表。")
    assert len(manifest.shots) >= 2
    for idx, shot in enumerate(manifest.shots, 1):
        frame = proj / "frames" / f"{idx:04d}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(_png_bytes((50 + idx * 25, 110, 170)))
        shot.assets["image"] = str(frame)
    manifest.save(proj / "project.json")

    # Codex holds asset_producer concurrently while batch critic auto-claims & releases role="critic" lease.
    prod_lease = claim_takeover(proj, supervisor_id="codex-image", role="asset_producer")

    def primary_fn(shot, **_kwargs):
        if shot.id == "shot-01":
            return {"decision": "passed", "scores": {"character": 0.93}, "reason": "达标", "supervisor_id": "gemini-4"}
        return {"decision": "warning", "scores": {"character": 0.72}, "reason": "需要二审", "supervisor_id": "gemini-4"}

    def tiebreaker_fn(**_kwargs):
        return {"decision": "failed", "failure_code": "CHARACTER_INCONSISTENT", "reason": "发色不一致", "supervisor_id": "grok-3"}

    report = run_project_critic_panel(
        proj,
        stage="image",
        policy="escalate",
        persist_verdict=True,
        primary_critic=primary_fn,
        tiebreaker_critic=tiebreaker_fn,
    )
    assert report["ok"] is False
    assert report["counts"]["passed"] == 1
    assert report["counts"]["failed"] == 1
    assert report["escalated_shots"] == ["shot-02"]
    assert report["attention_shots"] == ["shot-02"]
    # Critic lock was auto-released after sweep; producer lock remains intact.
    assert not (proj / ".takeover.critic.lock").exists()
    assert (proj / ".takeover.asset_producer.lock").exists()

    # Second pass with only_pending=True skips already-passed shot-01!
    seen_ids: list[str] = []

    def primary_pass_all(shot, **_kwargs):
        seen_ids.append(shot.id)
        return {"decision": "passed", "scores": {"character": 0.91}, "reason": "修复通过", "supervisor_id": "gemini-4"}

    report2 = run_project_critic_panel(
        proj,
        stage="image",
        only_pending=True,
        persist_verdict=True,
        primary_critic=primary_pass_all,
    )
    assert report2["ok"] is True
    assert report2["skipped_passed_shots"] == ["shot-01"]
    assert seen_ids == ["shot-02"]

    # CLI --all --only-pending exits 0 with 0 evaluated when all shots already passed!
    rc = cli_main(["critic-panel", str(proj), "--all", "--only-pending", "--stage", "image"])
    cli_out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert cli_out["ok"] is True
    assert cli_out["evaluated_count"] == 0
    assert release_takeover(proj, prod_lease["lease_token"])["released"] is True


def test_mcp_tool_profiles_filter_exposed_tools():
    prod_tools = {t.name for t in asyncio.run(create_mcp_server(profile="producer").list_tools())}
    critic_tools = {t.name for t in asyncio.run(create_mcp_server(profile="critic").list_tools())}
    core_tools = {t.name for t in asyncio.run(create_mcp_server(profile="core").list_tools())}
    full_tools = {t.name for t in asyncio.run(create_mcp_server(profile="full").list_tools())}

    assert "story_inject_external_asset" in prod_tools
    assert "story_submit_quality_verdict" not in prod_tools
    assert "story_inspect_shot_multimodal" in critic_tools
    assert "story_inject_external_asset" not in critic_tools
    assert len(prod_tools) < len(core_tools) < len(full_tools)
