"""MCP Supervisor Server exposing multimodal diagnostics, intervention, budget control, and Quality Flywheel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import mcp.types as mcp_types

try:
    from mcp.server.mcpserver import Image as MCPImage, MCPServer, ResourceSecurity
except ImportError:  # Fallback compatibility for MCP SDK v1.x
    from mcp.server.fastmcp import FastMCP as MCPServer, Image as MCPImage  # type: ignore
    ResourceSecurity = None

from .cost_ledger import summarize_cost_ledger
from .flywheel import (
    FAILURE_CODES,
    delete_rule,
    distill_playbook,
    export_flywheel_dataset,
    get_flywheel_stats,
    load_active_playbook,
    resolve_flywheel_dir,
    upsert_manual_rule,
)
from .manifest import ProjectManifest
from .qa_profiles import (
    build_vlm_qa_rubric_prompt,
    get_qa_profile,
    list_qa_profiles,
    resolve_shot_qa_profile,
)
from .supervisor import (
    cast_character_3d_from_photo,
    claim_takeover,
    doctor_check,
    finalize_pipeline,
    get_keyframe_generation_tasks,
    get_pipeline_snapshot,
    get_shot_contact_sheet_bytes,
    inject_external_asset,
    inspect_shot_bundle,
    assemble_parent_segment_group,
    dryrun_parent_segment_group,
    execute_parent_segment_group,
    intervene_and_rework_shot,
    plan_parent_segment_group,
    list_takeover_queue,
    release_takeover,
    run_pipeline_stage,
    submit_quality_verdict,
)


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _to_multimodal_blocks(payload: dict[str, Any], visuals: list[tuple[str, bytes, str]]) -> list[Any]:
    blocks: list[Any] = [mcp_types.TextContent(type="text", text=_json_text(payload))]
    for _, img_bytes, fmt in visuals:
        blocks.append(MCPImage(data=img_bytes, format=fmt).to_image_content())
    return blocks


def _resolve_flywheel_target(
    *,
    default_flywheel_dir: str | Path | None = None,
    project_dir: str | None = None,
    flywheel_dir: str | None = None,
) -> Path:
    manifest = None
    if project_dir:
        proj_file = Path(project_dir) / "project.json"
        if proj_file.is_file():
            manifest = ProjectManifest.load(proj_file)
    resolved = resolve_flywheel_dir(
        manifest=manifest,
        project_root=project_dir,
        explicit_dir=flywheel_dir,
        server_default_dir=default_flywheel_dir,
    )
    if resolved is None:
        return Path.cwd() / "flywheel"
    return Path(resolved)


MCP_TOOL_PROFILES: dict[str, frozenset[str] | None] = {
    "full": None,
    "producer": frozenset(
        {
            "story_get_pipeline_snapshot",
            "story_list_takeover_queue",
            "story_claim_takeover",
            "story_release_takeover",
            "story_get_keyframe_tasks",
            "story_cast_character_3d_from_photo",
            "story_inject_external_asset",
            "story_run_pipeline_stage",
            "story_lint_h3_bundle",
        }
    ),
    "critic": frozenset(
        {
            "story_get_pipeline_snapshot",
            "story_list_takeover_queue",
            "story_claim_takeover",
            "story_release_takeover",
            "story_get_qa_rubric",
            "story_inspect_shot_multimodal",
            "story_submit_quality_verdict",
            "story_intervene_and_rework_shot",
            "story_lint_handoff",
            "story_audit_speech",
            "story_verify_master",
            "story_finalize_pipeline",
            "story_package_delivery",
        }
    ),
    "core": frozenset(
        {
            "story_get_pipeline_snapshot",
            "story_list_takeover_queue",
            "story_claim_takeover",
            "story_release_takeover",
            "story_get_keyframe_tasks",
            "story_cast_character_3d_from_photo",
            "story_inject_external_asset",
            "story_inspect_shot_multimodal",
            "story_submit_quality_verdict",
            "story_intervene_and_rework_shot",
            "story_run_pipeline_stage",
            "story_lint_h3_bundle",
            "story_audit_speech",
            "story_verify_master",
            "story_finalize_pipeline",
            "story_package_delivery",
        }
    ),
}


def create_mcp_server(
    *,
    default_flywheel_dir: str | Path | None = None,
    profile: str = "full",
) -> MCPServer:
    """Create and configure the story-media-supervisor MCPServer instance."""
    clean_profile = (profile or "full").strip().lower()
    if clean_profile not in MCP_TOOL_PROFILES:
        raise ValueError(f"unsupported MCP profile: {profile!r}; expected one of {tuple(MCP_TOOL_PROFILES)}")
    allowed_tools = MCP_TOOL_PROFILES[clean_profile]

    server_kwargs: dict[str, Any] = {
        "name": "story-media-supervisor",
        "title": "Story Media Orchestrator Supervisor & Quality Flywheel",
        "description": (
            "Multimodal supervisor gateway for inspecting generated storyboards, keyframes, "
            "2x2 video contact sheets, intervening on failed shots within CNY/min budgets, "
            "and driving the persistent Quality Flywheel."
        ),
        "instructions": (
            "You are acting as the External Quality Supervisor (e.g. Gemini 4 Carbon) over story-media-orchestrator. "
            "1) Start with story_get_pipeline_snapshot and story_claim_takeover. "
            "2) Inspect each shot via story_inspect_shot_multimodal to view the 2x2 contact sheet and keyframe directly. "
            "3) Always check cost_ledger (<= 100 CNY/min default) before choosing a rework strategy: prefer low-cost "
            "fixes (rewrite_image_prompt ¥0.14 or fallback_25d_motion ¥0.0) over expensive video rerenders unless needed. "
            "4) After intervention and re-inspection, call story_submit_quality_verdict to promote verified fixes into the Quality Flywheel."
        ),
    }
    if ResourceSecurity is not None:
        server_kwargs["resource_security"] = ResourceSecurity(
            reject_path_traversal=True,
            reject_absolute_paths=False,
        )

    server = MCPServer(**server_kwargs)
    _orig_tool = server.tool
    if allowed_tools is not None:

        def _filtered_tool(*t_args: Any, **t_kwargs: Any):
            raw_dec = _orig_tool(*t_args, **t_kwargs)

            def _wrap(fn: Any) -> Any:
                tool_name = t_kwargs.get("name") or getattr(fn, "__name__", "")
                if tool_name in allowed_tools:
                    return raw_dec(fn)
                return fn

            return _wrap

        server.tool = _filtered_tool  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------
    @server.prompt(
        name="supervisor_audit_project",
        title="Full Project Quality Audit & Flywheel Loop",
        description="Structured workflow guide for Gemini 4 Carbon to audit all shots, fix defects within budget, and log flywheel rules.",
    )
    def supervisor_audit_project(project_dir: str) -> str:
        return (
            f"请作为外部质量监督者监督动漫短剧工程目录：{project_dir}\n"
            "执行闭环流程（严格遵循粗剪五类分诊、修法 A/B 隔离与四态验收）：\n"
            "1. 调用 story_get_pipeline_snapshot 获取总览、题材 genre 与成本水位（目标 <= 100 CNY/分钟），再调用 story_claim_takeover 获取接管锁。\n"
            "2. 逐镜头调用 story_inspect_shot_multimodal（查看首帧、2x2 视频诊断图、8帧密集时序条与 ROI 人物局部放大图，以及自动匹配的 qa_profile 与 qa_rubric_prompt），按动漫 7 大维度核对：\n"
            "   - character（人物阶段母版一致性、发色/法器归属、手部与解剖结构、无穿模）\n"
            "   - motion（动作起势-接触-受力/余波连贯、无站桩假动、无后半段僵死；对 SHARP_TOPOLOGY_TEAR 必须回看触发帧及前后动作定性）\n"
            "   - expression（微表情贴合情绪、眼神焦点自然、无惊悚错位）\n"
            "   - camera_displacement（运镜平滑、背景透视刚性稳定、无滑步瞬移与越轴）\n"
            "   - lighting_style（主光源与阴影一致、色温/饱和度不跳变、无闪烁与画风漂移）\n"
            "   - vfx_quality（体积光影与空间透视真实、火场域与冰/雷/剑气严格隔离、无扁平贴片光效）\n"
            "   - audio_sync（对白口型自然对齐、旁白闭口、静音镜无语音幻觉、无爆音与声画时长脱节）\n"
            "3. 按粗剪五类分诊从最早出错环节定性，且严格隔离修法 A（伪字幕/画外音张嘴/尾停单镜清理）与修法 B（无动作/对砍锁死重拆动作）；禁止因单镜问题整幕/整章重出，禁止用人工 waiver 绕过 failed/unavailable 装配硬门。\n"
            f"4. 发现不合格镜头先调用 story_submit_quality_verdict(decision='failed', failure_code=..., reason=...)，其中 failure_code 取自 {FAILURE_CODES}。\n"
            "5. 调用 story_intervene_and_rework_shot 进行单镜头最小代价外科重绘（rewrite_image_prompt / patch_character_dna / reroute_video / fallback_25d_motion），保留未改动镜头和音轨；首帧/提示词/音轨变动后不得复用旧视频缓存。\n"
            "6. 重新检视返工产物并提交 decision='passed'，系统将自动把本次有效修复沉淀进质量飞轮 playbook.json。\n"
            "7. 所有镜头完成设计→静帧→视频验收后，调用 story_finalize_pipeline 合成终版 MP4 并导出剪映草稿，最后释放接管锁。"
        )

    @server.prompt(
        name="supervisor_diagnose_shot",
        title="Single Shot Root-Cause Diagnosis",
        description="Focused single-shot diagnosis & lowest-cost intervention prompt with flywheel exemplars.",
    )
    def supervisor_diagnose_shot(project_dir: str, shot_id: str, defect_hint: str = "") -> str:
        fly_dir = _resolve_flywheel_target(default_flywheel_dir=default_flywheel_dir, project_dir=project_dir)
        playbook = load_active_playbook(project_root=project_dir, explicit_dir=fly_dir)
        exemplars = (playbook.get("exemplars") or [])[:3] if playbook else []
        hint_line = f"已知问题提示：{defect_hint}\n" if defect_hint.strip() else ""
        return (
            f"请对项目 {project_dir} 的镜头 {shot_id} 执行高精度根因诊断与最小成本修复。\n"
            f"{hint_line}"
            "请先调用 story_inspect_shot_multimodal 获取该镜头的画面、2x2 视频采样图、8帧密集条、ROI 局部放大图、有效 prompt、Character Bible 与 cost_ledger。\n"
            f"历史质量飞轮参考案例（Verified Exemplars）：{json.dumps(exemplars, ensure_ascii=False)}\n"
            "先按粗剪五类分诊定位最早出错环节：\n"
            "- PNG 首帧已错（发色/法器/站位/画面文字/峰值图误作首帧）：使用 strategy='rewrite_image_prompt' 仅重绘干净独立 16:9 首帧并锁定阶段母版，保留配音；\n"
            "- 首帧干净但伪字幕/画外音张嘴/尾停：走修法 A（去画面台词原文、加闭唇与静音词清洗，仅尾停优先裁尾）；\n"
            "- 首帧摆拍导致人物站桩或对砍锁死：走修法 B（重拆单拍起势/接触/受力，必要时 reroute_video 切换单首帧 i2v 或过跨度闸的 comfy_fl2v）；若预算紧张且动作幅度不大，使用 fallback_25d_motion。"
        )

    @server.prompt(
        name="codex_free_image_h3_pipeline",
        title="Codex Free Keyframe Generation + Pure H3 Pipeline",
        description="Guide Codex Chat MCP to draw free start/end keyframes with image_gen, inject them, and render via ComfyUI MiniMax H3 FL2VA.",
    )
    def codex_free_image_h3_pipeline(project_dir: str) -> str:
        return (
            f"请作为零生图费用的制片与监督 Agent 接管短剧工程：{project_dir}\n"
            "执行步骤（0 元生图 + 3D 骨相定妆 + ComfyUI MiniMax H3 闭环，严格执行四件套分工与四态验收）：\n"
            "0. （若剧本使用真人照片防撞脸定妆）先调用 story_cast_character_3d_from_photo 获取脱敏 3D 动漫铸模 Prompt，用内置 `image_gen` 附图转为半3D立体骨相动漫母版后再回传 `stylized_portrait_path` 绑定，切勿将原始真人照直接用于正片；按剧情阶段锁定发色与法器母版。\n"
            "1. 调用 story_claim_takeover 获取接管锁，并调用 story_get_keyframe_tasks 获取所有待绘制的关键帧清单。\n"
            "2. 对每个 pending task：先绘制干净的独立 16:9 `asset_role='image'`（起手首帧，停在主要动作发生前；九宫格/峰值图仅作审看对照，禁止将带排版字、小格或峰值图直接作为首帧），随请求传入阶段锁定妆与同轴尾帧接力参考图；"
            "通过文件解码与内容审看后立即调用 story_inject_external_asset(project_dir, shot_id, 'image', source_path=..., lease_token=...)。\n"
            "3. 常规镜头默认单首帧走 H3 I2VA（显式关尾帧）；仅明确终态且首尾构图连续的高动态镜头才绘制结束动作尾帧并注入 `asset_role='last_frame'`（须过跨度闸）。\n"
            "4. 所有关键帧注入完成后，先核对 H3 17k+5 帧长网格与静音镜声音清洗（非对白镜头禁止 whisper/conversation/murmuring 等触发词），可先调用 story_run_pipeline_stage(project_dir, stage='tts', ...) 生成配音，"
            "再调用 story_run_pipeline_stage(project_dir, stage='video', mode='render', execution_strategy='stage_batched', lease_token=...) 批量生成视频；用 story_inspect_shot_multimodal 逐镜审查 2x2 宫格/8帧条带/ROI 放大图（文件存在不等于视听验收通过）。\n"
            "5. 确认无 failed/unavailable 阻断且全部通过视听验收后，调用 story_finalize_pipeline 合成成片并导出剪映草稿，最后释放接管锁。"
        )

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------
    @server.resource(
        "story://projects/{+project_dir}/overview",
        name="project_overview",
        title="Project Pipeline Snapshot",
        mime_type="application/json",
    )
    def resource_project_overview(project_dir: str) -> str:
        clean_dir = unquote(project_dir)
        return _json_text(get_pipeline_snapshot(clean_dir, server_flywheel_dir=default_flywheel_dir))

    @server.resource(
        "story://projects/{+project_dir}/results",
        name="project_results",
        title="Full Shot-by-Shot Results & Lineage",
        mime_type="application/json",
    )
    def resource_project_results(project_dir: str) -> str:
        clean_dir = unquote(project_dir)
        root = Path(clean_dir).resolve()
        manifest = ProjectManifest.load(root / "project.json")
        payload = {
            "schema": "project-results/v1",
            "project_id": manifest.project_id,
            "status": manifest.status,
            "output": manifest.output,
            "jianying_draft": manifest.jianying_draft,
            "timeline": manifest.timeline,
            "character_bible": manifest.character_bible,
            "shots": [
                {
                    "id": s.id,
                    "scene_id": s.scene_id,
                    "text": s.text,
                    "narration": s.narration,
                    "dialogue_lines": s.dialogue_lines,
                    "duration": s.duration,
                    "status": s.status,
                    "mode": s.mode,
                    "video_route": s.video_route,
                    "prompt_override": s.prompt_override,
                    "negative_override": s.negative_override,
                    "assets": s.assets,
                    "quality_report": s.quality_report,
                    "diagnostics": s.diagnostics,
                    "rework_history": s.rework_history,
                }
                for s in manifest.shots
            ],
        }
        return _json_text(payload)

    @server.resource(
        "story://projects/{+project_dir}/ledger",
        name="project_cost_ledger",
        title="Project Cost Ledger",
        mime_type="application/json",
    )
    def resource_project_ledger(project_dir: str) -> str:
        clean_dir = unquote(project_dir)
        manifest = ProjectManifest.load(Path(clean_dir).resolve() / "project.json")
        return _json_text(summarize_cost_ledger(manifest))

    @server.resource(
        "story://projects/{+project_dir}/shots/{shot_id}/contact-sheet",
        name="shot_contact_sheet",
        title="Shot 2x2 Diagnostic Contact Sheet",
        mime_type="image/jpeg",
    )
    def resource_shot_contact_sheet(project_dir: str, shot_id: str) -> bytes:
        return get_shot_contact_sheet_bytes(unquote(project_dir), unquote(shot_id))

    @server.resource(
        "story://flywheel/playbook",
        name="flywheel_playbook",
        title="Distilled Quality Flywheel Playbook",
        mime_type="application/json",
    )
    def resource_flywheel_playbook() -> str:
        fly_dir = _resolve_flywheel_target(default_flywheel_dir=default_flywheel_dir)
        return _json_text(get_flywheel_stats(fly_dir))

    @server.resource(
        "story://qa/profiles",
        name="anime_qa_profiles",
        title="Genre-Specific Anime QA Profiles",
        mime_type="application/json",
    )
    def resource_anime_qa_profiles() -> str:
        return _json_text({"schema": "anime-qa-profiles/v1", "profiles": list_qa_profiles()})

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    @server.tool(
        description=(
            "Return the genre-calibrated anime QA profile (weights, CV/lighting/audio thresholds, "
            "tolerances, veto rules) and ready-to-use VLM rubric prompt."
        )
    )
    def story_get_qa_rubric(
        genre: str | None = None,
        project_dir: str | None = None,
        shot_id: str | None = None,
        stage: str = "video",
    ) -> dict[str, Any]:
        manifest = None
        shot = None
        if project_dir:
            proj_file = Path(project_dir) / "project.json"
            if proj_file.is_file():
                manifest = ProjectManifest.load(proj_file)
                if shot_id:
                    shot = manifest.shot(shot_id)
        if manifest is not None or shot is not None:
            profile = resolve_shot_qa_profile(manifest, shot, genre=genre)
        else:
            profile = get_qa_profile(genre)
        rubric_prompt = build_vlm_qa_rubric_prompt(
            profile,
            stage=stage,
            shot=shot,
            diagnostics=shot.diagnostics if (shot is not None and isinstance(shot.diagnostics, dict)) else None,
        )
        return {
            "schema": "anime-qa-rubric/v1",
            "stage": stage,
            "profile": profile,
            "rubric_prompt": rubric_prompt,
        }
    @server.tool(
        description="List shots that failed auto-rework, flagged warnings, or await external supervisor takeover."
    )
    def story_list_takeover_queue(
        project_dir: str,
        filter_status: str = "needs_attention",
    ) -> dict[str, Any]:
        return list_takeover_queue(
            project_dir,
            filter_status=filter_status,
            server_flywheel_dir=default_flywheel_dir,
        )
    @server.tool(
        description=(
            "Claim a supervisor (exclusive) or role-scoped ('asset_producer' / 'critic') takeover lease "
            "on a project directory. 'asset_producer' and 'critic' leases can coexist concurrently."
        )
    )
    def story_claim_takeover(
        project_dir: str,
        supervisor_id: str = "gemini-4-carbon",
        lease_seconds: int = 300,
        force: bool = False,
        role: str = "supervisor",
    ) -> dict[str, Any]:
        return claim_takeover(
            project_dir,
            supervisor_id=supervisor_id,
            lease_seconds=lease_seconds,
            force=force,
            role=role,
        )

    @server.tool(description="Release an active supervisor or role-scoped takeover lease.")
    def story_release_takeover(
        project_dir: str,
        lease_token: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        return release_takeover(project_dir, lease_token=lease_token, role=role)

    @server.tool(
        description="Inspect project pipeline status across all 5 stages, shot quality verdicts, cost ledger, and flywheel telemetry."
    )
    def story_get_pipeline_snapshot(project_dir: str) -> dict[str, Any]:
        return get_pipeline_snapshot(project_dir, server_flywheel_dir=default_flywheel_dir)

    @server.tool(
        description=(
            "List keyframe prompts (first_frame / last_frame) and reference image paths so Codex Chat can generate "
            "free keyframes using built-in `image_gen` and inject them back via `story_inject_external_asset`."
        )
    )
    def story_get_keyframe_tasks(
        project_dir: str,
        include_generated: bool = False,
    ) -> dict[str, Any]:
        return get_keyframe_generation_tasks(
            project_dir,
            include_generated=include_generated,
        )

    @server.tool(
        description=(
            "Cast or bind a de-identified 3D bone-structure anime character master portrait from a real photo "
            "to avoid anime look-alike plagiarism while preserving 3D facial contours."
        )
    )
    def story_cast_character_3d_from_photo(
        project_dir: str,
        character_name: str,
        source_photo_path: str | None = None,
        stylized_portrait_path: str | None = None,
        appearance: str | None = None,
        clothing: str | None = None,
        visual_dna: str | None = None,
        deidentify_traits: str | None = None,
        execute_now: bool = False,
        max_identity_similarity: float | None = None,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        return cast_character_3d_from_photo(
            project_dir,
            character_name,
            source_photo_path=source_photo_path,
            stylized_portrait_path=stylized_portrait_path,
            appearance=appearance,
            clothing=clothing,
            visual_dna=visual_dna,
            deidentify_traits=deidentify_traits,
            execute_now=execute_now,
            max_identity_similarity=max_identity_similarity,
            lease_token=lease_token,
        )

    @server.tool(
        description=(
            "Inspect a single shot with both structured metadata (story, prompt, character DNA, diagnostics, ledger, "
            "flywheel exemplars) AND actual multimodal images (2x2 video contact sheet, first frame, last frame)."
        )
    )
    def story_inspect_shot_multimodal(
        project_dir: str,
        shot_id: str,
        refresh_diagnostics: bool = True,
        include_candidates: bool = False,
    ) -> list[Any]:
        bundle, visuals = inspect_shot_bundle(
            project_dir,
            shot_id,
            refresh_diagnostics=refresh_diagnostics,
            include_candidates=include_candidates,
            server_flywheel_dir=default_flywheel_dir,
        )
        return _to_multimodal_blocks(bundle, visuals)

    @server.tool(
        description=(
            "Submit a supervisor quality verdict (passed/warning/failed) with dimension scores and failure_code. "
            "When marking a reworked shot as passed, automatically verifies and promotes the fix into the Quality Flywheel."
        )
    )
    def story_submit_quality_verdict(
        project_dir: str,
        shot_id: str,
        decision: str,
        stage: str = "image",
        scores: dict[str, float] | None = None,
        failure_code: str = "NONE",
        reason: str = "",
        failures: list[str] | None = None,
        supervisor_id: str = "gemini-4-carbon",
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        return submit_quality_verdict(
            project_dir,
            shot_id,
            stage=stage,
            decision=decision,
            scores=scores,
            failure_code=failure_code,
            reason=reason,
            failures=failures,
            supervisor_id=supervisor_id,
            lease_token=lease_token,
            server_flywheel_dir=default_flywheel_dir,
        )

    @server.tool(
        description=(
            "Perform surgical single-shot intervention and optional immediate rework within cost budget. "
            "Strategies: rewrite_image_prompt, patch_character_dna, reroute_video, fallback_25d_motion, select_candidate, reset_shot. "
            "prompt_override is explicit shot action/visual text, not the fully assembled prompt; "
            "when provided it is used as the H3 I2V/FL2V action text ahead of persisted scene_context.kinetic_action. "
            "seed_override (including 0) applies to reroute_video and is written only onto the target shot; "
            "omit it to keep the existing seed. "
            "use_last_frame: omit or true keeps current behavior (attach last_frame when the asset exists); "
            "explicit false skips last_frame on this request only, without deleting or clearing the on-disk asset, "
            "and without generating a new end frame or changing the H3 model. "
            "final_prompt_override is a request-scoped exact final H3 prompt for replay/control experiments; "
            "when set it is used as the MiniMaxH3ImageToVideo prompt verbatim and does not persist as scene data "
            "or disable the default VFX standard for other requests. "
            "Returns updated metadata + new diagnostic images."
        )
    )
    def story_intervene_and_rework_shot(
        project_dir: str,
        shot_id: str,
        strategy: str,
        prompt_override: str | None = None,
        negative_override: str | None = None,
        action_guidance: str | None = None,
        trigger_keywords: list[str] | None = None,
        character_name: str | None = None,
        character_dna_patch: dict[str, Any] | None = None,
        video_route: str | None = None,
        motion: str | None = None,
        candidate_id: str | None = None,
        frame_role: str = "image",
        seed_override: int | None = None,
        use_last_frame: bool | None = None,
        final_prompt_override: str | None = None,
        failure_code: str | None = None,
        failure_reason: str | None = None,
        execute_now: bool = True,
        max_cost_cny: float = 5.0,
        supervisor_id: str = "gemini-4-carbon",
        lease_token: str | None = None,
    ) -> list[Any]:
        result, visuals = intervene_and_rework_shot(
            project_dir,
            shot_id,
            strategy=strategy,
            prompt_override=prompt_override,
            negative_override=negative_override,
            action_guidance=action_guidance,
            trigger_keywords=trigger_keywords,
            character_name=character_name,
            character_dna_patch=character_dna_patch,
            video_route=video_route,
            motion=motion,
            candidate_id=candidate_id,
            frame_role=frame_role,
            seed_override=seed_override,
            use_last_frame=use_last_frame,
            final_prompt_override=final_prompt_override,
            failure_code=failure_code,
            failure_reason=failure_reason,
            execute_now=execute_now,
            max_cost_cny=max_cost_cny,
            supervisor_id=supervisor_id,
            lease_token=lease_token,
            server_flywheel_dir=default_flywheel_dir,
        )
        return _to_multimodal_blocks(result, visuals)

    @server.tool(
        description="Inject an externally repaired image/last_frame/video/audio artifact directly into a shot."
    )
    def story_inject_external_asset(
        project_dir: str,
        shot_id: str,
        asset_role: str,
        source_path: str | None = None,
        content_base64: str | None = None,
        provenance: str = "external-agent",
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        return inject_external_asset(
            project_dir,
            shot_id,
            asset_role=asset_role,
            source_path=source_path,
            content_base64=content_base64,
            provenance=provenance,
            lease_token=lease_token,
        )

    @server.tool(
        description=(
            "Manage the persistent Quality Flywheel: stats, playbook, distill, upsert_rule, delete_rule, or export_dataset (DPO/SFT JSONL)."
        )
    )
    def story_manage_flywheel(
        action: str = "stats",
        project_dir: str | None = None,
        flywheel_dir: str | None = None,
        rule_payload: dict[str, Any] | None = None,
        rule_id: str | None = None,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        target_dir = _resolve_flywheel_target(
            default_flywheel_dir=default_flywheel_dir,
            project_dir=project_dir,
            flywheel_dir=flywheel_dir,
        )
        act = (action or "stats").strip().lower()
        if act in {"stats", "playbook"}:
            return get_flywheel_stats(target_dir)
        if act == "distill":
            return {"flywheel_dir": str(target_dir), "playbook": distill_playbook(target_dir)}
        if act == "upsert_rule":
            if not isinstance(rule_payload, dict):
                raise ValueError("upsert_rule requires rule_payload")
            return upsert_manual_rule(target_dir, rule_payload)
        if act == "delete_rule":
            if not rule_id:
                raise ValueError("delete_rule requires rule_id")
            return delete_rule(target_dir, rule_id)
        if act == "export_dataset":
            return export_flywheel_dataset(target_dir, output_path=output_path)
        raise ValueError(f"unsupported flywheel action: {action!r}")

    @server.tool(
        description=(
            "Execute a specific pipeline stage (all, keyframes, tts, video, assemble) with stage_batched or interleaved "
            "strategy, optional lipsync_provider, auto_split_shots, and budget governance."
        )
    )
    def story_run_pipeline_stage(
        project_dir: str,
        stage: str = "all",
        mode: str | None = None,
        execution_strategy: str | None = None,
        image_source: str | None = None,
        tts_source: str | None = None,
        video_provider: str | None = None,
        video_routing_policy: str | None = None,
        lipsync_provider: str | None = None,
        auto_split_shots: bool | None = None,
        max_shot_seconds: float | None = None,
        relay_tail_frames: bool | None = None,
        auto_rework: bool | None = None,
        budget_limit_cny: float | None = None,
        approve_gates: bool = True,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        return run_pipeline_stage(
            project_dir,
            stage=stage,
            mode=mode,
            execution_strategy=execution_strategy,
            image_source=image_source,
            tts_source=tts_source,
            video_provider=video_provider,
            video_routing_policy=video_routing_policy,
            lipsync_provider=lipsync_provider,
            auto_split_shots=auto_split_shots,
            max_shot_seconds=max_shot_seconds,
            relay_tail_frames=relay_tail_frames,
            auto_rework=auto_rework,
            budget_limit_cny=budget_limit_cny,
            approve_gates=approve_gates,
            lease_token=lease_token,
        )

    @server.tool(
        description="Approve pending review gates, assemble final MP4 without re-billing, and optionally export Jianying draft."
    )
    def story_finalize_pipeline(
        project_dir: str,
        approve_gates: bool = True,
        export_jianying: bool = True,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        return finalize_pipeline(
            project_dir,
            approve_gates=approve_gates,
            export_jianying=export_jianying,
            lease_token=lease_token,
        )

    @server.tool(
        description=(
            "Run preflight diagnostics for local ffmpeg/cv2/PIL, sibling repositories, closed-source git push locks, "
            "Codex image readiness, and optional remote 24GB+ ComfyUI MiniMax H3 weight/node probing."
        )
    )
    def story_doctor_check(
        project_dir: str | None = None,
        probe_network: bool = False,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        return doctor_check(
            project_dir,
            probe_network=probe_network,
            timeout=timeout,
        )


    @server.tool(
        description=(
            "Plan or dry-run a parent-segment group: multiple visual segments that share one parent shot's "
            "voiceover without attaching the full parent wav to each segment (which would retrigger 141/107-frame "
            "lengths). execute_now defaults to false. Planning writes no history, ledger, or official project files. "
            "Segment lengths are exact H3 frame counts (73/24 is not rounded to 3.042s). Child segments inherit the "
            "parent rework cap and budget; splitting cannot reset a 5/5 parent. Group assembly is a separate explicit "
            "path and is not enabled inside render_preview or finalize. Live H3 group execute is not enabled."
        )
    )
    def story_plan_parent_segment_group(
        project_dir: str,
        plan: dict[str, Any],
        dry_run: bool = True,
    ) -> dict[str, Any]:
        if dry_run is False:
            # Still planning only; execute requires the execute tool and explicit flags.
            return plan_parent_segment_group(project_dir, plan)
        return dryrun_parent_segment_group(project_dir, plan)

    @server.tool(
        description=(
            "Execute a parent-segment group. execute_now defaults to FALSE. Generation is blocked unless every gate "
            "passes, generation_authorized is explicitly true, and a non-live test provider is supplied on an "
            "unprotected temp project. Official xinhuo-ji-* projects, chapter masters, and live H3 are refused. "
            "Do not use this to bypass parent caps with new shot IDs or injected assets."
        )
    )
    def story_execute_parent_segment_group(
        project_dir: str,
        plan: dict[str, Any],
        execute_now: bool = False,
        generation_authorized: bool = False,
        lease_token: str | None = None,
        supervisor_id: str = "grok-bot-t50u",
    ) -> dict[str, Any]:
        return execute_parent_segment_group(
            project_dir,
            plan,
            execute_now=execute_now,
            generation_authorized=generation_authorized,
            lease_token=lease_token,
            supervisor_id=supervisor_id,
        )

    @server.tool(
        description=(
            "Explicit official assembly for a parent-segment group (concat visuals, play the original parent wav once). "
            "Blocked when any segment is failed or unavailable, or when total duration differs from the parent wav "
            "without duration_change_authorized. This is not hooked into render_preview or finalize, and will not "
            "overwrite chapter masters."
        )
    )
    def story_assemble_parent_segment_group(
        project_dir: str,
        plan: dict[str, Any],
        segment_videos: dict[str, str],
        segment_reports: list[dict[str, Any]],
        duration_change_authorized: bool = False,
        output_path: str | None = None,
    ) -> dict[str, Any]:
        return assemble_parent_segment_group(
            project_dir,
            plan,
            segment_videos=segment_videos,
            segment_reports=segment_reports,
            duration_change_authorized=duration_change_authorized,
            output_path=output_path,
        )

    @server.tool(
        description=(
            "Lint a minimal handoff manifest (four-state acceptance, anchor timeline, rough-cut triage rules) "
            "using production_playbook.lint_handoff_manifest."
        )
    )
    def story_lint_handoff(
        manifest_path: str,
        check_files: bool = False,
    ) -> dict[str, Any]:
        from .production_playbook import lint_handoff_manifest

        return lint_handoff_manifest(
            Path(manifest_path),
            require_first_frame_exists=bool(check_files),
        )

    @server.tool(
        description=(
            "Pre-flight lint an H3 bundle directory (17k+5 frame grid, silent-shot speech trigger word scrubbing, "
            "image dimensions, and chapter continuity)."
        )
    )
    def story_lint_h3_bundle(
        bundle_dir: str,
        max_frames: int = 175,
        check_images: bool = True,
        strict_chapter: bool = False,
    ) -> dict[str, Any]:
        from .h3_bundle import validate_h3_bundle

        report = validate_h3_bundle(
            Path(bundle_dir),
            max_frames=int(max_frames),
            check_images=bool(check_images),
            require_chapter_field=bool(strict_chapter),
        )
        return report.to_dict()

    @server.tool(
        description=(
            "Run VAD + ASR dialogue audit and subtitle alignment check on a chapter directory or single shot MP4."
        )
    )
    def story_audit_speech(
        target: str,
        bundle_dir: str | None = None,
        audit_json: str | None = None,
        expected_line: str | None = None,
        speaker: str = "Speaker",
        style: str = "Dialogue",
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cer_warn: float = 0.20,
        cer_fail: float = 0.45,
    ) -> dict[str, Any]:
        from .speech_audit import FasterWhisperBackend, audit_chapter_speech, audit_shot_speech

        backend = FasterWhisperBackend(model_size=model, device=device, compute_type=compute_type)
        target_p = Path(target)
        if target_p.is_dir():
            report = audit_chapter_speech(
                target_p,
                bundle_dir=Path(bundle_dir) if bundle_dir else None,
                audit_json=Path(audit_json) if audit_json else None,
                backend=backend,
                cer_warn_threshold=float(cer_warn),
                cer_fail_threshold=float(cer_fail),
            )
        else:
            report = audit_shot_speech(
                target_p,
                expected_line=expected_line,
                speaker=speaker,
                style=style,
                backend=backend,
                cer_warn_threshold=float(cer_warn),
                cer_fail_threshold=float(cer_fail),
            )
        return report.to_dict()

    @server.tool(
        description=(
            "Verify a rendered master MP4 or series audit JSON against release quality gates "
            "(loudness, freeze frames, frame counts, and media integrity)."
        )
    )
    def story_verify_master(
        target: str,
        profile: str = "archive",
        media: str | None = None,
        reference_video: str | None = None,
        expected_frames: int | None = None,
        bundle_root: str | None = None,
        strict_freeze: bool = False,
    ) -> dict[str, Any]:
        from .master_verifier import verify_master_release, verify_series_audit

        target_p = Path(target)
        if target_p.suffix.lower() == ".json":
            report = verify_series_audit(
                target_p,
                media_override=Path(media) if media else None,
                profile=profile,
                reference_video_mp4=Path(reference_video) if reference_video else None,
                bundle_root=Path(bundle_root) if bundle_root else None,
                strict_freeze=bool(strict_freeze),
            )
        else:
            report = verify_master_release(
                target_p,
                profile=profile,
                expected_frames=expected_frames,
                reference_video_mp4=Path(reference_video) if reference_video else None,
                strict_freeze=bool(strict_freeze),
            )
        return report.to_dict()

    @server.tool(
        description=(
            "Run the configured multi-model external critic panel (objective hard gates + primary VLM e.g. Gemini 4 "
            "+ conditional tie-breaker e.g. Grok) on a shot, optionally persisting the reconciled verdict."
        )
    )
    def story_run_critic_panel(
        project_dir: str,
        shot_id: str,
        stage: str = "video",
        policy: str = "escalate",
        escalate_after_reworks: int = 2,
        force_tiebreaker: bool = False,
        refresh_diagnostics: bool = True,
        persist_verdict: bool = False,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        from .external_critics import run_external_critic_panel

        return run_external_critic_panel(
            project_dir,
            shot_id,
            stage=stage,
            policy=policy,
            escalate_after_reworks=escalate_after_reworks,
            force_tiebreaker=force_tiebreaker,
            refresh_diagnostics=refresh_diagnostics,
            persist_verdict=persist_verdict,
            lease_token=lease_token,
            server_flywheel_dir=default_flywheel_dir,
        )

    @server.tool(
        description=(
            "Run unattended multi-model external critic sweep across all (or pending/selected) shots in a project, "
            "automatically using a role='critic' lease when persist_verdict=True and no lease_token is supplied."
        )
    )
    def story_run_project_critic_panel(
        project_dir: str,
        stage: str = "video",
        policy: str = "escalate",
        shot_ids: list[str] | None = None,
        only_pending: bool = False,
        escalate_after_reworks: int = 2,
        force_tiebreaker: bool = False,
        refresh_diagnostics: bool = True,
        persist_verdict: bool = False,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        from .external_critics import run_project_critic_panel

        return run_project_critic_panel(
            project_dir,
            stage=stage,
            policy=policy,
            shot_ids=shot_ids,
            only_pending=only_pending,
            escalate_after_reworks=escalate_after_reworks,
            force_tiebreaker=force_tiebreaker,
            refresh_diagnostics=refresh_diagnostics,
            persist_verdict=persist_verdict,
            lease_token=lease_token,
            server_flywheel_dir=default_flywheel_dir,
        )

    @server.tool(
        description=(
            "Run the unattended nightly episode pipeline: optional H3 bundle lint, optional render stages, "
            "multi-model critic panel sweep with latency/JSON/escalation telemetry, optional budget-guarded auto-rework, "
            "and optional finalize + commercial delivery packaging."
        )
    )
    def story_run_nightly_episode(
        project_dir: str,
        stage: str = "video",
        policy: str = "escalate",
        only_pending: bool = True,
        persist_verdict: bool = True,
        run_render_stages: bool = False,
        h3_bundle_dir: str | None = None,
        max_auto_rework_rounds: int = 0,
        max_rework_cost_cny: float = 5.0,
        auto_finalize: bool = False,
        verify_master: bool = True,
        package_delivery_dir: str | None = None,
        client_id: str | None = None,
        lease_token: str | None = None,
    ) -> dict[str, Any]:
        from .nightly_runner import run_unattended_episode

        return run_unattended_episode(
            project_dir,
            stage=stage,
            policy=policy,
            only_pending=only_pending,
            persist_verdict=persist_verdict,
            run_render_stages=run_render_stages,
            h3_bundle_dir=h3_bundle_dir,
            max_auto_rework_rounds=max_auto_rework_rounds,
            max_rework_cost_cny=max_rework_cost_cny,
            auto_finalize=auto_finalize,
            verify_master=verify_master,
            package_delivery_dir=package_delivery_dir,
            client_id=client_id,
            lease_token=lease_token,
            server_flywheel_dir=default_flywheel_dir,
        )

    @server.tool(
        description=(
            "Package a commercial delivery bundle (master MP4, optional nosub MP4, Jianying draft, cost ledger, "
            "QA summary, master verification, isolated character masters, and SHA-256 manifest). "
            "Blocks by default if any shot is failed/unavailable."
        )
    )
    def story_package_delivery(
        project_dir: str,
        output_dir: str | None = None,
        client_id: str | None = None,
        nosub_mp4: str | None = None,
        speech_audit_json: str | None = None,
        require_all_passed: bool = True,
        verify_master: bool = True,
        export_jianying_if_missing: bool = False,
    ) -> dict[str, Any]:
        from .delivery_bundle import package_commercial_delivery

        return package_commercial_delivery(
            project_dir,
            output_dir=output_dir,
            client_id=client_id,
            nosub_mp4=nosub_mp4,
            speech_audit_json=speech_audit_json,
            require_all_passed=require_all_passed,
            verify_master=verify_master,
            export_jianying_if_missing=export_jianying_if_missing,
        )

    @server.tool(
        description=(
            "Initialize a client-isolated workspace and private .flywheel directory so commercial clients "
            "never share character DNA or private rules."
        )
    )
    def story_init_client_workspace(
        clients_root: str,
        client_id: str,
        project_id: str,
        story: str = "",
    ) -> dict[str, Any]:
        from .delivery_bundle import init_isolated_client_workspace

        return init_isolated_client_workspace(
            clients_root,
            client_id,
            project_id,
            story=story,
        )

    server.tool = _orig_tool  # type: ignore[method-assign]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="story-media-mcp")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--flywheel-dir", default=None)
    parser.add_argument(
        "--profile",
        choices=tuple(MCP_TOOL_PROFILES.keys()),
        default="full",
    )
    args = parser.parse_args(argv)

    flywheel_dir = None
    if args.flywheel_dir:
        flywheel_dir = str(Path(args.flywheel_dir).resolve())
        os.environ["STORY_MEDIA_FLYWHEEL_DIR"] = flywheel_dir

    server = create_mcp_server(default_flywheel_dir=flywheel_dir, profile=args.profile)
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
