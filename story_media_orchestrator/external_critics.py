"""Multi-model external critic panel (e.g., Gemini 4 primary + Grok tie-breaker) for automated & MCP QA."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable
from urllib import error as urllib_error, request as urllib_request

from .frame_decomposer import extract_image_diagnostics, extract_video_diagnostics, refresh_shot_diagnostics
from .manifest import ProjectManifest, Shot
from .plot_critic import (
    _collect_character_ref_uris,
    _detect_split_panel_bars,
    _encode_image_for_vlm,
    _normalize_verdict,
    _parse_json_response,
)
from .qa_profiles import (
    build_vlm_qa_rubric_prompt,
    compute_objective_audio_sync_score,
    evaluate_objective_qa_flags,
    resolve_shot_qa_profile,
)

VALID_CRITIC_PROTOCOLS = ("openai_compat", "gemini_native", "http_evaluator")
VALID_PANEL_POLICIES = ("escalate", "always", "primary_only")
HARD_DEFECT_CODES = frozenset(
    {
        "SPLIT_PANEL_TEXT",
        "VFX_QUALITY_DEFECT",
        "CHARACTER_INCONSISTENT",
        "STATIC_OR_BLACK_VIDEO",
    }
)

_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1/chat/completions"


@dataclass(frozen=True)
class CriticSpec:
    """Configuration for a single external VLM/HTTP critic endpoint."""

    critic_id: str
    protocol: str  # openai_compat | gemini_native | http_evaluator
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: float = 75.0
    role: str = "primary"  # primary | tiebreaker

    def to_public_dict(self) -> dict[str, Any]:
        """Return safe metadata without exposing secrets."""
        return {
            "critic_id": self.critic_id,
            "protocol": self.protocol,
            "model": self.model,
            "base_url": self.base_url,
            "role": self.role,
            "configured": bool(self.api_key or self.protocol == "http_evaluator"),
        }


def resolve_critic_panel_from_env() -> tuple[CriticSpec | None, CriticSpec | None]:
    """Resolve primary (e.g. Gemini 4) and tie-breaker (e.g. Grok) critics from environment variables."""
    primary: CriticSpec | None = None
    tiebreaker: CriticSpec | None = None

    # 1) Primary critic: explicit STORY_PRIMARY_CRITIC_* -> GEMINI_API_KEY -> STORY_PLOT_CRITIC_* / DASHSCOPE_API_KEY
    p_proto = os.environ.get("STORY_PRIMARY_CRITIC_PROTOCOL", "").strip().lower()
    p_key = os.environ.get("STORY_PRIMARY_CRITIC_API_KEY", "").strip()
    p_url = os.environ.get("STORY_PRIMARY_CRITIC_BASE_URL", "").strip()
    p_model = os.environ.get("STORY_PRIMARY_CRITIC_MODEL", "").strip()
    p_id = os.environ.get("STORY_PRIMARY_CRITIC_ID", "").strip()

    if p_proto or p_key or p_url:
        proto = p_proto or ("http_evaluator" if (p_url and not p_key) else "openai_compat")
        if proto not in VALID_CRITIC_PROTOCOLS:
            raise ValueError(f"unsupported STORY_PRIMARY_CRITIC_PROTOCOL: {proto!r}")
        primary = CriticSpec(
            critic_id=p_id or p_model or f"primary-{proto}",
            protocol=proto,
            model=p_model or ("gemini-2.5-pro" if proto == "gemini_native" else "qwen-vl-max"),
            base_url=p_url or (_DEFAULT_GEMINI_BASE_URL if proto == "gemini_native" else "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
            api_key=p_key,
            role="primary",
        )
    elif os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip():
        g_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
        g_model = os.environ.get("STORY_GEMINI_CRITIC_MODEL", "gemini-2.5-pro").strip() or "gemini-2.5-pro"
        g_url = os.environ.get("STORY_GEMINI_CRITIC_BASE_URL", _DEFAULT_GEMINI_BASE_URL).strip()
        primary = CriticSpec(
            critic_id=os.environ.get("STORY_GEMINI_CRITIC_ID", g_model).strip() or g_model,
            protocol="gemini_native",
            model=g_model,
            base_url=g_url,
            api_key=g_key,
            role="primary",
        )
    elif os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip():
        primary = CriticSpec(
            critic_id="http-evaluator",
            protocol="http_evaluator",
            base_url=os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip(),
            role="primary",
        )
    elif os.environ.get("STORY_PLOT_CRITIC_API_KEY", "").strip() or os.environ.get("DASHSCOPE_API_KEY", "").strip():
        k = os.environ.get("STORY_PLOT_CRITIC_API_KEY", "").strip() or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        m = os.environ.get("STORY_PLOT_CRITIC_MODEL", "qwen-vl-max").strip() or "qwen-vl-max"
        u = os.environ.get(
            "STORY_PLOT_CRITIC_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        ).strip()
        primary = CriticSpec(
            critic_id=m,
            protocol="openai_compat",
            model=m,
            base_url=u,
            api_key=k,
            role="primary",
        )

    # 2) Tie-breaker critic: explicit STORY_TIEBREAKER_CRITIC_* -> XAI_API_KEY / GROK_API_KEY
    t_proto = os.environ.get("STORY_TIEBREAKER_CRITIC_PROTOCOL", "").strip().lower()
    t_key = os.environ.get("STORY_TIEBREAKER_CRITIC_API_KEY", "").strip()
    t_url = os.environ.get("STORY_TIEBREAKER_CRITIC_BASE_URL", "").strip()
    t_model = os.environ.get("STORY_TIEBREAKER_CRITIC_MODEL", "").strip()
    t_id = os.environ.get("STORY_TIEBREAKER_CRITIC_ID", "").strip()

    if t_proto or t_key or t_url:
        proto = t_proto or ("http_evaluator" if (t_url and not t_key) else "openai_compat")
        if proto not in VALID_CRITIC_PROTOCOLS:
            raise ValueError(f"unsupported STORY_TIEBREAKER_CRITIC_PROTOCOL: {proto!r}")
        tiebreaker = CriticSpec(
            critic_id=t_id or t_model or f"tiebreaker-{proto}",
            protocol=proto,
            model=t_model or ("grok-2-vision-1212" if proto == "openai_compat" else "gemini-2.5-pro"),
            base_url=t_url or (_DEFAULT_GEMINI_BASE_URL if proto == "gemini_native" else _DEFAULT_GROK_BASE_URL),
            api_key=t_key,
            role="tiebreaker",
        )
    elif os.environ.get("XAI_API_KEY", "").strip() or os.environ.get("GROK_API_KEY", "").strip():
        x_key = os.environ.get("XAI_API_KEY", "").strip() or os.environ.get("GROK_API_KEY", "").strip()
        x_model = os.environ.get("STORY_GROK_CRITIC_MODEL", "grok-2-vision-1212").strip() or "grok-2-vision-1212"
        x_url = os.environ.get("STORY_GROK_CRITIC_BASE_URL", _DEFAULT_GROK_BASE_URL).strip()
        tiebreaker = CriticSpec(
            critic_id=os.environ.get("STORY_GROK_CRITIC_ID", x_model).strip() or x_model,
            protocol="openai_compat",
            model=x_model,
            base_url=x_url,
            api_key=x_key,
            role="tiebreaker",
        )

    return primary, tiebreaker


def _build_critic_prompt_and_images(
    manifest: ProjectManifest,
    shot: Shot,
    *,
    stage: str,
    artifact_path: Path,
    diagnostics: dict[str, Any],
) -> tuple[str, list[tuple[str, str]], str, dict[str, Any], dict[str, Any]]:
    """Build canonical prompt text and ordered (label, data_uri) visual evidence for any VLM critic."""
    from .character_bible import resolve_shot_characters
    from .lipsync import classify_shot_speech_delivery

    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    active_profiles = resolve_shot_characters(manifest, shot)
    char_summary = {
        str(v.get("key") or v.get("id") or v.get("name") or idx): {
            "name": v.get("name"),
            "visual_dna": v.get("visual_dna"),
            "appearance": v.get("appearance"),
            "clothing": v.get("clothing"),
        }
        for idx, v in enumerate(active_profiles)
        if isinstance(v, dict)
    }
    canonical_action = str(ctx.get("visual_action") or shot.text or "").strip()
    framing_hint = str(ctx.get("framing") or "").strip()
    location_hint = str(ctx.get("location") or "").strip()
    continuity_hint = str(ctx.get("continuity") or "").strip()
    scene_parts = [
        p
        for p in (
            canonical_action,
            f"场景：{location_hint}" if location_hint else "",
            f"景别：{framing_hint}" if framing_hint else "",
            f"阶段连续性：{continuity_hint}" if continuity_hint else "",
        )
        if p
    ]
    qa_profile = resolve_shot_qa_profile(manifest, shot)
    full_diag = shot.diagnostics if isinstance(shot.diagnostics, dict) and shot.diagnostics else diagnostics
    delivery = classify_shot_speech_delivery(shot, manifest.character_bible)
    rubric_block = build_vlm_qa_rubric_prompt(
        qa_profile,
        stage=stage,
        shot=shot,
        diagnostics=full_diag,
        character_bible=manifest.character_bible,
    )
    speech_rule = (
        "本镜已锁定为画外音/旁白镜头，如果人物或动物明显张嘴说话，记 IMAGE_PROMPT_DRIFT。"
        if delivery == "voiceover"
        else (
            "本镜已锁定为露脸对白镜头（非旁白），人物自然张嘴说话完全正确，严禁以“旁白张嘴”判为 IMAGE_PROMPT_DRIFT。"
            if delivery == "on_screen_dialogue"
            else "本镜无对白，人物自然闭口即可。"
        )
    )
    diag_for_prompt = dict(diagnostics) if isinstance(diagnostics, dict) else {}
    if isinstance(full_diag, dict):
        for extra_k in ("audio", "av_sync", "objective_qa"):
            if extra_k in full_diag and extra_k not in diag_for_prompt:
                diag_for_prompt[extra_k] = full_diag[extra_k]

    instruction = (
        f"你是外部多模态质检监督员。请评估 stage={stage!r} 镜头 {shot.id} 是否达到 H3 电影级标准。\n"
        f"{rubric_block}\n"
        f"剧本动作：{'；'.join(scene_parts)}\n"
        f"本镜出场角色设定：{json.dumps(char_summary, ensure_ascii=False)}\n"
        f"客观诊断指标：{json.dumps(diag_for_prompt, ensure_ascii=False)}\n"
        "硬性失败：画面像素里出现汉字、字幕、对白条或水印，记 LOCAL_DEFECT。"
        f"{speech_rule}"
        "特效硬伤记 VFX_QUALITY_DEFECT；请严格区分修法 A（伪字幕/旁白张嘴/尾停单镜清理）与修法 B（无动作/对砍锁死重拆动作）。\n"
        "严格输出单个 JSON 对象（不要额外解释）：\n"
        '{"decision":"passed|warning|failed","failure_code":"PASS|IMAGE_PROMPT_DRIFT|CHARACTER_INCONSISTENT|LOCAL_DEFECT|VIDEO_MOTION_FAIL|VFX_QUALITY_DEFECT",'
        '"scores":{"character":0.0-1.0,"motion":0.0-1.0,"expression":0.0-1.0,"camera_displacement":0.0-1.0,"lighting_style":0.0-1.0,"vfx_quality":0.0-1.0,"audio_sync":0.0-1.0},'
        '"reason":"简短中文原因","triage_category":"script_beat|motion_lock|still_defect|video_cleanup|edit_transition|none",'
        '"rework_action":{"strategy":"regenerate_first_frame|patch_character_dna|reroute_video|fallback_25d_motion|none","revised_prompt":null,"negative_addition":null,"target_video_route":null}}'
    )

    images: list[tuple[str, str]] = []
    for ref in _collect_character_ref_uris(manifest, shot):
        label = ref.get("name") or ref.get("id") or "角色"
        images.append((f"角色母版参考图（{label}）：", ref["data_uri"]))

    main_uri = _encode_image_for_vlm(artifact_path)
    if main_uri:
        images.append(("待审产物画面：", main_uri))

    if stage == "video" and isinstance(diagnostics, dict):
        dense_path = diagnostics.get("dense_strip_path")
        if isinstance(dense_path, str) and dense_path.strip() and Path(dense_path).is_file():
            dense_uri = _encode_image_for_vlm(Path(dense_path), max_edge=1280)
            if dense_uri:
                images.append(("8帧密集时序拆帧图：", dense_uri))
        roi_path = diagnostics.get("roi_zoom_path")
        if isinstance(roi_path, str) and roi_path.strip() and Path(roi_path).is_file():
            roi_uri = _encode_image_for_vlm(Path(roi_path), max_edge=1280)
            if roi_uri:
                images.append(("人物局部放大对比条：", roi_uri))

    return instruction, images, delivery, qa_profile, full_diag


def _data_uri_to_inline_part(data_uri: str) -> dict[str, Any]:
    """Convert `data:image/jpeg;base64,...` into Gemini REST `inline_data` part."""
    header, _, b64_data = data_uri.partition(",")
    mime_type = "image/jpeg"
    if header.startswith("data:") and ";" in header:
        mime_type = header[5 : header.index(";")] or "image/jpeg"
    return {"inline_data": {"mime_type": mime_type, "data": b64_data}}


def _invoke_critic_spec(
    spec: CriticSpec,
    *,
    manifest: ProjectManifest,
    shot: Shot,
    stage: str,
    artifact_path: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Invoke a single configured external critic and return a normalized quality-evaluation/v2 verdict."""
    instruction, images, delivery, qa_profile, full_diag = _build_critic_prompt_and_images(
        manifest,
        shot,
        stage=stage,
        artifact_path=artifact_path,
        diagnostics=diagnostics,
    )
    if not images:
        return _normalize_verdict(
            {
                "decision": "unavailable",
                "reason": "待审产物不是有效图像或拆帧宫格图，无法调用外部视觉监督员",
                "supervisor_id": spec.critic_id,
            },
            stage=stage,
            shot_id=shot.id,
            diagnostics=diagnostics,
        )

    if spec.protocol == "http_evaluator":
        body = json.dumps(
            {
                "stage": stage,
                "shot_id": shot.id,
                "text": shot.text,
                "qa_profile": qa_profile,
                "character_bible": manifest.character_bible,
                "diagnostics": diagnostics,
                "instruction": instruction,
                "images": [{"label": lbl, "data_uri": uri} for lbl, uri in images],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if spec.api_key:
            headers["Authorization"] = f"Bearer {spec.api_key}"
        req = urllib_request.Request(spec.base_url, body, headers)
        with urllib_request.urlopen(req, timeout=spec.timeout) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
    elif spec.protocol == "gemini_native":
        base = (spec.base_url or _DEFAULT_GEMINI_BASE_URL).rstrip("/")
        url = f"{base}/models/{spec.model}:generateContent"
        parts: list[dict[str, Any]] = [{"text": instruction}]
        for lbl, uri in images:
            parts.append({"text": lbl})
            parts.append(_data_uri_to_inline_part(uri))
        req_payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        req_bytes = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": spec.api_key,
        }
        req = urllib_request.Request(url, req_bytes, headers)
        with urllib_request.urlopen(req, timeout=spec.timeout) as resp:
            body_json = json.loads(resp.read().decode("utf-8"))
        cand = (body_json.get("candidates") or [{}])[0]
        cand_parts = ((cand.get("content") or {}).get("parts") or [])
        text_out = "".join(str(p.get("text") or "") for p in cand_parts if isinstance(p, dict))
        parsed = _parse_json_response(text_out)
    elif spec.protocol == "openai_compat":
        content_blocks: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for lbl, uri in images:
            content_blocks.append({"type": "text", "text": lbl})
            content_blocks.append({"type": "image_url", "image_url": {"url": uri}})
        req_payload = {
            "model": spec.model,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": content_blocks}],
        }
        req_bytes = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {spec.api_key}",
            "User-Agent": "story-media-orchestrator/1.0",
        }
        req = urllib_request.Request(spec.base_url, req_bytes, headers)
        with urllib_request.urlopen(req, timeout=spec.timeout) as resp:
            body_json = json.loads(resp.read().decode("utf-8"))
        content = body_json["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        parsed = _parse_json_response(str(content))
    else:
        raise ValueError(f"unsupported critic protocol: {spec.protocol!r}")

    parsed.setdefault("supervisor_id", spec.critic_id)
    obj_aud = compute_objective_audio_sync_score(
        full_diag,
        qa_profile,
        has_dialogue=(delivery == "on_screen_dialogue"),
    )
    if obj_aud is not None and isinstance(parsed.get("scores"), dict):
        vlm_aud = parsed["scores"].get("audio_sync")
        reason_txt = str(parsed.get("reason") or "")
        if (
            isinstance(vlm_aud, (int, float))
            and 0.40 <= float(vlm_aud) <= 0.62
            and any(tok in reason_txt for tok in ("无音轨", "缺少音轨", "无法确认音画同步", "暂按中性分"))
        ):
            parsed["scores"]["audio_sync"] = obj_aud

    verdict = _normalize_verdict(parsed, stage=stage, shot_id=shot.id, diagnostics=diagnostics)
    if isinstance(parsed.get("triage_category"), str) and parsed["triage_category"].strip():
        verdict["triage_category"] = parsed["triage_category"].strip()
    return verdict


def reconcile_panel_verdicts(
    primary_verdict: dict[str, Any],
    tiebreaker_verdict: dict[str, Any] | None,
    *,
    policy: str,
    escalation_reasons: list[str],
) -> dict[str, Any]:
    """Deterministically reconcile primary + optional tie-breaker verdicts into one final verdict."""
    if tiebreaker_verdict is None:
        merged = dict(primary_verdict)
        merged["panel"] = {
            "schema": "external-critic-panel/v1",
            "policy": policy,
            "tiebreaker_triggered": False,
            "escalation_reasons": list(escalation_reasons),
            "verdicts": [primary_verdict],
        }
        return merged

    p_dec = str(primary_verdict.get("decision") or "unavailable").lower()
    t_dec = str(tiebreaker_verdict.get("decision") or "unavailable").lower()
    p_code = str(primary_verdict.get("failure_code") or "NONE").upper()
    t_code = str(tiebreaker_verdict.get("failure_code") or "NONE").upper()

    # If one critic failed to connect/parse, fall back to the available one.
    if p_dec == "unavailable" and t_dec != "unavailable":
        chosen = dict(tiebreaker_verdict)
        chosen["reason"] = f"[主审不可用，采用二审] {tiebreaker_verdict.get('reason', '')}".strip()
    elif t_dec == "unavailable" and p_dec != "unavailable":
        chosen = dict(primary_verdict)
        chosen["reason"] = f"[二审不可用，保留主审] {primary_verdict.get('reason', '')}".strip()
    elif p_dec == "unavailable" and t_dec == "unavailable":
        chosen = dict(primary_verdict)
    elif p_dec == t_dec:
        # Both critics agree on decision; keep primary structure and merge failures/scores.
        chosen = dict(primary_verdict)
        chosen["reason"] = (
            f"[双审一致:{p_dec}] 主审({primary_verdict.get('supervisor_id')}): {primary_verdict.get('reason', '')} | "
            f"二审({tiebreaker_verdict.get('supervisor_id')}): {tiebreaker_verdict.get('reason', '')}"
        ).strip()
    else:
        # Disagreement resolution:
        # 1) Any hard defect code flagged as failed -> conservative failed.
        if (p_dec == "failed" and p_code in HARD_DEFECT_CODES) or (t_dec == "failed" and t_code in HARD_DEFECT_CODES):
            failed_side = primary_verdict if (p_dec == "failed" and p_code in HARD_DEFECT_CODES) else tiebreaker_verdict
            chosen = dict(failed_side)
            chosen["decision"] = "failed"
            chosen["reason"] = (
                f"[双审分歧-硬伤保守拦截] 主审({primary_verdict.get('supervisor_id')}={p_dec}/{p_code}) vs "
                f"二审({tiebreaker_verdict.get('supervisor_id')}={t_dec}/{t_code})：{failed_side.get('reason', '')}"
            ).strip()
        # 2) warning + failed -> failed (conservative).
        elif {p_dec, t_dec} == {"warning", "failed"}:
            failed_side = primary_verdict if p_dec == "failed" else tiebreaker_verdict
            chosen = dict(failed_side)
            chosen["decision"] = "failed"
            chosen["reason"] = (
                f"[双审分歧-降级拦截] 主审({primary_verdict.get('supervisor_id')}={p_dec}) vs "
                f"二审({tiebreaker_verdict.get('supervisor_id')}={t_dec})：{failed_side.get('reason', '')}"
            ).strip()
        # 3) warning + passed -> trust tie-breaker when tie-breaker explicitly passed; otherwise warning.
        elif p_dec == "warning" and t_dec == "passed":
            chosen = dict(tiebreaker_verdict)
            chosen["decision"] = "passed"
            chosen["reason"] = (
                f"[二审放行] 主审({primary_verdict.get('supervisor_id')}=warning) -> "
                f"二审({tiebreaker_verdict.get('supervisor_id')}=passed)：{tiebreaker_verdict.get('reason', '')}"
            ).strip()
        else:
            # passed vs failed (non-hard defect) or passed vs warning -> warning for takeover queue review.
            chosen = dict(primary_verdict if p_dec != "passed" else tiebreaker_verdict)
            chosen["decision"] = "warning"
            chosen["reason"] = (
                f"[双审分歧-转人工/接管复核] 主审({primary_verdict.get('supervisor_id')}={p_dec}/{p_code}) vs "
                f"二审({tiebreaker_verdict.get('supervisor_id')}={t_dec}/{t_code})"
            ).strip()

    # Merge unique failures from both critics
    merged_failures: list[str] = []
    for v in (primary_verdict, tiebreaker_verdict):
        for item in v.get("failures") or []:
            s = str(item).strip()
            if s and s not in merged_failures:
                merged_failures.append(s)
    chosen["failures"] = merged_failures
    chosen["supervisor_id"] = (
        f"panel:{primary_verdict.get('supervisor_id', 'primary')}+"
        f"{tiebreaker_verdict.get('supervisor_id', 'tiebreaker')}"
    )
    chosen["panel"] = {
        "schema": "external-critic-panel/v1",
        "policy": policy,
        "tiebreaker_triggered": True,
        "escalation_reasons": list(escalation_reasons),
        "verdicts": [primary_verdict, tiebreaker_verdict],
    }
    return chosen


def run_external_critic_panel(
    project_dir: str | Path,
    shot_id: str,
    *,
    stage: str = "video",
    policy: str = "escalate",
    escalate_after_reworks: int = 2,
    force_tiebreaker: bool = False,
    refresh_diagnostics: bool = True,
    persist_verdict: bool = False,
    lease_token: str | None = None,
    primary_critic: CriticSpec | Callable[..., dict[str, Any]] | None = None,
    tiebreaker_critic: CriticSpec | Callable[..., dict[str, Any]] | None = None,
    server_flywheel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run objective hard gates + primary critic + conditional tie-breaker critic on a shot."""
    clean_stage = (stage or "video").strip().lower()
    if clean_stage not in {"image", "video"}:
        raise ValueError(f"unsupported critic stage: {stage!r}")
    clean_policy = (policy or "escalate").strip().lower()
    if clean_policy not in VALID_PANEL_POLICIES:
        raise ValueError(f"unsupported panel policy: {policy!r}; expected one of {VALID_PANEL_POLICIES}")

    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)

    if refresh_diagnostics:
        refresh_shot_diagnostics(manifest, root, shot.id)

    if clean_stage == "image":
        raw_asset = (shot.assets or {}).get("image")
        artifact_path = Path(str(raw_asset)) if raw_asset else None
        obj_diag = extract_image_diagnostics(artifact_path)
    else:
        vid_diag = (shot.diagnostics or {}).get("video") if isinstance(shot.diagnostics, dict) else None
        if not isinstance(vid_diag, dict) or not vid_diag:
            raw_vid = (shot.assets or {}).get("video")
            vid_diag = extract_video_diagnostics(Path(str(raw_vid)), shot_id=shot.id) if raw_vid else {}
        sheet_path = vid_diag.get("contact_sheet_path") if isinstance(vid_diag, dict) else None
        raw_vid = (shot.assets or {}).get("video")
        artifact_path = Path(str(sheet_path)) if sheet_path and Path(str(sheet_path)).is_file() else (Path(str(raw_vid)) if raw_vid else None)
        obj_diag = dict(vid_diag) if isinstance(vid_diag, dict) else {}

    # 1) Objective fast-fail before spending any external model tokens.
    if artifact_path is None or not artifact_path.is_file():
        verdict = _normalize_verdict(
            {
                "decision": "failed",
                "failure_code": "LOCAL_DEFECT" if clean_stage == "image" else "VIDEO_MOTION_FAIL",
                "scores": {"visual_h3_grade": 0.0},
                "reason": "待审产物文件缺失（客观硬门拦截，未调用外部大模型）",
                "supervisor_id": "objective-hard-gate",
            },
            stage=clean_stage,
            shot_id=shot.id,
            diagnostics=obj_diag,
        )
        verdict["panel"] = {
            "schema": "external-critic-panel/v1",
            "policy": clean_policy,
            "tiebreaker_triggered": False,
            "escalation_reasons": ["objective_fast_fail"],
            "verdicts": [verdict],
        }
        return _maybe_persist_panel_verdict(
            root,
            shot.id,
            verdict,
            persist=persist_verdict,
            lease_token=lease_token,
            server_flywheel_dir=server_flywheel_dir,
        )

    qa_profile = resolve_shot_qa_profile(manifest, shot)
    eval_diag = shot.diagnostics if isinstance(shot.diagnostics, dict) and shot.diagnostics else obj_diag
    obj_qa = evaluate_objective_qa_flags(eval_diag, qa_profile, stage=clean_stage)
    if obj_qa.get("hard_fails"):
        fail = obj_qa["hard_fails"][0]
        code = str(fail.get("code") or "")
        verdict = _normalize_verdict(
            {
                "decision": "failed",
                "failure_code": "LOCAL_DEFECT" if clean_stage == "image" or code == "DARK_OR_BLANK_IMAGE" else "VIDEO_MOTION_FAIL",
                "scores": {"visual_h3_grade": 0.15},
                "reason": str(fail.get("message") or "客观质检硬门失败（未调用外部大模型）"),
                "failures": [str(item.get("code")) for item in obj_qa["hard_fails"] if item.get("code")],
                "supervisor_id": "objective-hard-gate",
            },
            stage=clean_stage,
            shot_id=shot.id,
            diagnostics={**obj_diag, "objective_qa": obj_qa},
        )
        verdict["panel"] = {
            "schema": "external-critic-panel/v1",
            "policy": clean_policy,
            "tiebreaker_triggered": False,
            "escalation_reasons": ["objective_hard_gate"],
            "verdicts": [verdict],
        }
        return _maybe_persist_panel_verdict(
            root,
            shot.id,
            verdict,
            persist=persist_verdict,
            lease_token=lease_token,
            server_flywheel_dir=server_flywheel_dir,
        )

    if clean_stage == "image":
        split_reason = _detect_split_panel_bars(artifact_path)
        if split_reason:
            verdict = _normalize_verdict(
                {
                    "decision": "failed",
                    "failure_code": "SPLIT_PANEL_TEXT",
                    "scores": {"visual_h3_grade": 0.15},
                    "reason": split_reason,
                    "supervisor_id": "objective-hard-gate",
                },
                stage=clean_stage,
                shot_id=shot.id,
                diagnostics=obj_diag,
            )
            verdict["panel"] = {
                "schema": "external-critic-panel/v1",
                "policy": clean_policy,
                "tiebreaker_triggered": False,
                "escalation_reasons": ["split_panel_fast_fail"],
                "verdicts": [verdict],
            }
            return _maybe_persist_panel_verdict(
                root,
                shot.id,
                verdict,
                persist=persist_verdict,
                lease_token=lease_token,
                server_flywheel_dir=server_flywheel_dir,
            )

    # 2) Resolve external critics
    env_primary, env_tiebreaker = resolve_critic_panel_from_env()
    primary = primary_critic if primary_critic is not None else env_primary
    tiebreaker = tiebreaker_critic if tiebreaker_critic is not None else env_tiebreaker

    if primary is None and tiebreaker is None:
        return {
            "schema": "quality-evaluation/v1",
            "schema_version": "quality-evaluation/v2",
            "stage": clean_stage,
            "shot_id": shot.id,
            "decision": "unavailable",
            "failure_code": "NONE",
            "scores": {},
            "overall_score": 0.0,
            "reason": "未配置外部监督模型（请配置 GEMINI_API_KEY / XAI_API_KEY / STORY_PRIMARY_CRITIC_* 或经 MCP 人工接管）",
            "failures": [],
            "diagnostics": obj_diag,
            "panel": {
                "schema": "external-critic-panel/v1",
                "policy": clean_policy,
                "tiebreaker_triggered": False,
                "escalation_reasons": ["no_critic_configured"],
                "verdicts": [],
            },
        }

    def _run_one(critic: CriticSpec | Callable[..., dict[str, Any]], role_label: str) -> dict[str, Any]:
        if callable(critic):
            raw = critic(
                manifest=manifest,
                shot=shot,
                stage=clean_stage,
                artifact_path=artifact_path,
                diagnostics=obj_diag,
            )
            return _normalize_verdict(raw, stage=clean_stage, shot_id=shot.id, diagnostics=obj_diag)
        try:
            return _invoke_critic_spec(
                critic,
                manifest=manifest,
                shot=shot,
                stage=clean_stage,
                artifact_path=artifact_path,
                diagnostics=obj_diag,
            )
        except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, ValueError, KeyError, IndexError, OSError) as exc:
            return {
                "schema": "quality-evaluation/v1",
                "schema_version": "quality-evaluation/v2",
                "stage": clean_stage,
                "shot_id": shot.id,
                "decision": "unavailable",
                "failure_code": "NONE",
                "scores": {},
                "overall_score": 0.0,
                "reason": f"{role_label}({critic.critic_id}) 调用异常: {type(exc).__name__}",
                "failures": [],
                "supervisor_id": critic.critic_id,
                "diagnostics": obj_diag,
            }

    if primary is not None:
        primary_verdict = _run_one(primary, "primary")
    else:
        # Only tie-breaker configured -> treat as primary
        assert tiebreaker is not None
        primary_verdict = _run_one(tiebreaker, "tiebreaker")
        tiebreaker = None

    # 3) Decide whether to invoke tie-breaker
    escalation_reasons: list[str] = []
    if tiebreaker is not None and clean_policy != "primary_only":
        if clean_policy == "always":
            escalation_reasons.append("policy_always")
        if force_tiebreaker:
            escalation_reasons.append("force_tiebreaker")
        if primary_verdict.get("decision") == "warning":
            escalation_reasons.append("primary_warning")
        if primary_verdict.get("decision") == "unavailable":
            escalation_reasons.append("primary_unavailable")
        if len(shot.rework_history or []) >= max(1, int(escalate_after_reworks)):
            escalation_reasons.append(f"rework_round_ge_{int(escalate_after_reworks)}")

    tiebreaker_verdict = _run_one(tiebreaker, "tiebreaker") if (tiebreaker is not None and escalation_reasons) else None
    merged = reconcile_panel_verdicts(
        primary_verdict,
        tiebreaker_verdict,
        policy=clean_policy,
        escalation_reasons=escalation_reasons,
    )
    return _maybe_persist_panel_verdict(
        root,
        shot.id,
        merged,
        persist=persist_verdict,
        lease_token=lease_token,
        server_flywheel_dir=server_flywheel_dir,
    )


def _maybe_persist_panel_verdict(
    root: Path,
    shot_id: str,
    verdict: dict[str, Any],
    *,
    persist: bool,
    lease_token: str | None,
    server_flywheel_dir: str | Path | None,
) -> dict[str, Any]:
    if not persist or verdict.get("decision") == "unavailable":
        return verdict
    from .supervisor import submit_quality_verdict

    persisted = submit_quality_verdict(
        root,
        shot_id,
        stage=str(verdict.get("stage") or "video"),
        decision=str(verdict.get("decision") or "warning"),
        scores=verdict.get("scores") if isinstance(verdict.get("scores"), dict) else None,
        failure_code=str(verdict.get("failure_code") or "NONE"),
        reason=str(verdict.get("reason") or ""),
        failures=verdict.get("failures") if isinstance(verdict.get("failures"), list) else None,
        supervisor_id=str(verdict.get("supervisor_id") or "external-critic-panel"),
        lease_token=lease_token,
        server_flywheel_dir=server_flywheel_dir,
    )
    return {**verdict, "persisted": persisted}


def _shot_already_passed(shot: Shot, stage: str) -> bool:
    qr = shot.quality_report if isinstance(shot.quality_report, dict) else {}
    if str(qr.get("decision") or "").lower() != "passed":
        return False
    qr_stage = str(qr.get("stage") or "").lower()
    if not qr_stage:
        return True
    if stage == "image" and qr_stage in {"image", "video", "final"}:
        return True
    return qr_stage == stage


def run_project_critic_panel(
    project_dir: str | Path,
    *,
    stage: str = "video",
    policy: str = "escalate",
    shot_ids: list[str] | None = None,
    only_pending: bool = False,
    escalate_after_reworks: int = 2,
    force_tiebreaker: bool = False,
    refresh_diagnostics: bool = True,
    persist_verdict: bool = False,
    auto_claim_critic_lease: bool = True,
    lease_token: str | None = None,
    supervisor_id: str = "external-critic-panel",
    primary_critic: CriticSpec | Callable[..., dict[str, Any]] | None = None,
    tiebreaker_critic: CriticSpec | Callable[..., dict[str, Any]] | None = None,
    server_flywheel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run unattended multi-model external critic sweep across a project (or selected shots)."""
    root = Path(project_dir).resolve()
    manifest_path = root / "project.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    manifest = ProjectManifest.load(manifest_path)

    selected_set = {s.strip() for s in (shot_ids or []) if str(s).strip()}
    target_shots: list[Shot] = []
    skipped_shots: list[str] = []
    for shot in manifest.shots:
        if selected_set and shot.id not in selected_set:
            continue
        if only_pending and _shot_already_passed(shot, stage):
            skipped_shots.append(shot.id)
            continue
        target_shots.append(shot)

    acquired_lease: dict[str, Any] | None = None
    active_token = lease_token
    if persist_verdict and not active_token and auto_claim_critic_lease and target_shots:
        from .supervisor import claim_takeover

        acquired_lease = claim_takeover(
            root,
            supervisor_id=supervisor_id,
            role="critic",
        )
        active_token = acquired_lease["lease_token"]

    results: list[dict[str, Any]] = []
    counts = {"passed": 0, "warning": 0, "failed": 0, "unavailable": 0}
    escalated_shots: list[str] = []
    attention_shots: list[str] = []

    try:
        for shot in target_shots:
            verdict = run_external_critic_panel(
                root,
                shot.id,
                stage=stage,
                policy=policy,
                escalate_after_reworks=escalate_after_reworks,
                force_tiebreaker=force_tiebreaker,
                refresh_diagnostics=refresh_diagnostics,
                persist_verdict=persist_verdict,
                lease_token=active_token,
                primary_critic=primary_critic,
                tiebreaker_critic=tiebreaker_critic,
                server_flywheel_dir=server_flywheel_dir,
            )
            dec = str(verdict.get("decision") or "unavailable").lower()
            counts[dec] = counts.get(dec, 0) + 1
            if dec != "passed":
                attention_shots.append(shot.id)
            panel_meta = verdict.get("panel") if isinstance(verdict.get("panel"), dict) else {}
            if panel_meta.get("tiebreaker_triggered"):
                escalated_shots.append(shot.id)
            results.append(verdict)
    finally:
        if acquired_lease is not None:
            from .supervisor import release_takeover

            release_takeover(root, acquired_lease["lease_token"], role="critic")

    all_ok = counts["failed"] == 0 and counts["unavailable"] == 0 and counts["warning"] == 0
    return {
        "schema": "external-critic-project-report/v1",
        "project_dir": str(root),
        "stage": stage,
        "policy": policy,
        "ok": all_ok,
        "total_shots": len(manifest.shots),
        "evaluated_count": len(results),
        "skipped_passed_shots": skipped_shots,
        "counts": counts,
        "escalated_shots": escalated_shots,
        "attention_shots": attention_shots,
        "persisted": bool(persist_verdict),
        "results": results,
    }
