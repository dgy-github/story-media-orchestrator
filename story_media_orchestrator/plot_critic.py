"""Multimodal VLM & objective plot/quality critic outputting quality-evaluation/v2 verdicts."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import re
from typing import Any
from urllib import error as urllib_error, request as urllib_request

from .frame_decomposer import extract_image_diagnostics, extract_video_diagnostics
from .manifest import ProjectManifest, Shot
from .qa_profiles import (
    build_vlm_qa_rubric_prompt,
    compute_objective_audio_sync_score,
    evaluate_objective_qa_flags,
    resolve_shot_qa_profile,
)
from .quality import build_supervisor_verdict

PLOT_FAILURE_CODES = (
    "PASS",
    "NONE",
    "IMAGE_PROMPT_DRIFT",
    "CHARACTER_INCONSISTENT",
    "LOCAL_DEFECT",
    "VIDEO_MOTION_FAIL",
    "VFX_QUALITY_DEFECT",
    "SPLIT_PANEL_TEXT",
    "STATIC_OR_BLACK_VIDEO",
    "OTHER",
)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_DEFAULT_DASHSCOPE_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _collect_character_ref_uris(manifest: ProjectManifest, shot: Shot, *, limit: int = 2) -> list[dict[str, str]]:
    from .character_bible import resolve_shot_characters

    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for profile in resolve_shot_characters(manifest, shot):
        if not isinstance(profile, dict):
            continue
        raw_path = profile.get("ref_image")
        if not isinstance(raw_path, str) or not raw_path.strip() or raw_path in seen:
            continue
        uri = _encode_image_for_vlm(Path(raw_path), max_edge=512)
        if not uri:
            continue
        seen.add(raw_path)
        refs.append(
            {
                "id": str(profile.get("key") or profile.get("id") or ""),
                "name": str(profile.get("name") or profile.get("key") or profile.get("id") or ""),
                "data_uri": uri,
            }
        )
        if len(refs) >= limit:
            break
    return refs


def _detect_split_panel_bars(path: Path) -> str | None:
    """Detect unmistakable internal black/white collage/split-screen gutters without flagging normal photos."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
    except Exception:
        return None

    h, w = gray.shape
    if h < 120 or w < 120:
        return None

    def _has_internal_gutter(arr: np.ndarray) -> bool:
        length = arr.shape[0]
        lo = int(length * 0.20)
        hi = int(length * 0.80)
        if hi - lo < 4:
            return False
        band_means = arr.mean(axis=1)
        band_stds = arr.std(axis=1)
        uniform_gutter = (band_stds < 3.5) & ((band_means < 14.0) | (band_means > 240.0))
        run = 0
        for idx in range(lo, hi):
            if uniform_gutter[idx]:
                run += 1
                if run >= 3:
                    g_mean = float(band_means[idx])
                    top = arr[: max(1, idx - 3), :]
                    bottom = arr[min(length - 1, idx + 2) :, :]
                    if top.size >= 64 and bottom.size >= 64:
                        if float(top.std()) > 12.0 and float(bottom.std()) > 12.0:
                            if abs(float(top.mean()) - g_mean) >= 25.0 and abs(float(bottom.mean()) - g_mean) >= 25.0:
                                return True
            else:
                run = 0
        return False

    if _has_internal_gutter(gray):
        return "检测到画面内部存在上下分屏分隔白边/黑边条，疑似多格拼贴废图"
    if _has_internal_gutter(gray.T):
        return "检测到画面内部存在左右分屏分隔白边/黑边条，疑似多格分镜拼贴废图"
    return None


def _encode_image_for_vlm(path: Path, *, max_edge: int = 1024) -> str | None:
    if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
        return None
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if max(rgb.size) > max_edge:
            rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buf = BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("VLM critic response did not contain a valid JSON object")


def _normalize_verdict(
    raw: dict[str, Any],
    *,
    stage: str,
    shot_id: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("critic output must be a JSON object")

    raw_dec = str(raw.get("decision") or "").strip().lower()
    raw_code = str(raw.get("failure_code") or "NONE").strip().upper()
    if raw_dec in {"pass", "passed", "ok"} or (not raw_dec and raw_code == "PASS"):
        decision = "passed"
    elif raw_dec in {"warn", "warning"}:
        decision = "warning"
    elif raw_dec in {"fail", "failed", "reject"}:
        decision = "failed"
    elif raw_dec == "unavailable":
        return {
            "schema": "quality-evaluation/v1",
            "schema_version": "quality-evaluation/v2",
            "stage": stage,
            "shot_id": shot_id,
            "decision": "unavailable",
            "failure_code": "NONE",
            "scores": {},
            "overall_score": 0.0,
            "reason": str(raw.get("reason") or "质量评估暂不可用"),
            "failures": list(raw.get("failures") or []) if isinstance(raw.get("failures"), list) else [],
            "diagnostics": diagnostics,
            "rework_action": {"strategy": "none"},
        }
    else:
        raise ValueError(f"unsupported critic decision: {raw.get('decision')!r}")

    if decision == "passed" and raw_code in {"", "NONE"}:
        failure_code = "PASS"
    elif raw_code in PLOT_FAILURE_CODES:
        failure_code = raw_code
    else:
        failure_code = "OTHER"

    if decision == "failed" and failure_code in {"NONE", "PASS"}:
        failure_code = "IMAGE_PROMPT_DRIFT" if stage == "image" else "VIDEO_MOTION_FAIL"

    raw_scores = raw.get("scores")
    scores: dict[str, float] | None = None
    if isinstance(raw_scores, dict) and raw_scores:
        scores = {}
        for k, v in raw_scores.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                scores[str(k)] = max(0.0, min(1.0, float(v)))

    failures = [str(item) for item in (raw.get("failures") or []) if str(item).strip()] if isinstance(raw.get("failures"), list) else []
    verdict = build_supervisor_verdict(
        stage=stage,
        decision=decision,
        scores=scores,
        failure_code=failure_code,
        reason=str(raw.get("reason") or ""),
        failures=failures,
        supervisor_id=str(raw.get("supervisor_id") or "plot-critic"),
        round_index=int(raw.get("round_index") or 0),
    )

    action = raw.get("rework_action")
    if not isinstance(action, dict):
        action = {}
    default_strategy = "none"
    if decision == "failed":
        if stage == "video":
            default_strategy = "reroute_video"
        elif failure_code == "CHARACTER_INCONSISTENT":
            default_strategy = "patch_character_dna"
        elif failure_code == "LOCAL_DEFECT":
            default_strategy = "regenerate_with_negative"
        else:
            default_strategy = "regenerate_first_frame"

    verdict["shot_id"] = shot_id
    verdict["diagnostics"] = diagnostics
    verdict["rework_action"] = {
        "strategy": str(action.get("strategy") or default_strategy),
        "revised_prompt": action.get("revised_prompt"),
        "negative_addition": action.get("negative_addition"),
        "inpaint_bbox": action.get("inpaint_bbox"),
        "target_video_route": action.get("target_video_route"),
    }
    return verdict


def _call_http_evaluator(
    url: str,
    *,
    manifest: ProjectManifest,
    shot: Shot,
    stage: str,
    artifact_path: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    image_data_uri = _encode_image_for_vlm(artifact_path)
    character_ref_images = _collect_character_ref_uris(manifest, shot)
    qa_profile = resolve_shot_qa_profile(manifest, shot)
    body = json.dumps(
        {
            "stage": stage,
            "shot_id": shot.id,
            "text": shot.text,
            "visual_action": ctx.get("visual_action") or shot.text,
            "qa_profile": qa_profile,
            "character_bible": manifest.character_bible,
            "character_ref_images": character_ref_images,
            "diagnostics": diagnostics,
            "image_data_uri": image_data_uri,
            "artifact": {
                "shot_id": shot.id,
                "path": str(artifact_path),
                "diagnostics": diagnostics,
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib_request.Request(url, body, {"Content-Type": "application/json"})
    with urllib_request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _normalize_verdict(payload, stage=stage, shot_id=shot.id, diagnostics=diagnostics)


def _call_dashscope_vlm(
    api_key: str,
    *,
    manifest: ProjectManifest,
    shot: Shot,
    stage: str,
    artifact_path: Path,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    image_data_uri = _encode_image_for_vlm(artifact_path)
    if image_data_uri is None:
        return {
            "schema": "quality-evaluation/v1",
            "schema_version": "quality-evaluation/v2",
            "stage": stage,
            "shot_id": shot.id,
            "decision": "unavailable",
            "failure_code": "NONE",
            "scores": {},
            "overall_score": 0.0,
            "reason": "待审产物不是有效图像或拆帧宫格图，无法调用视觉裁判",
            "failures": [],
            "diagnostics": diagnostics,
            "rework_action": {"strategy": "none"},
        }

    base_url = os.environ.get("STORY_PLOT_CRITIC_BASE_URL", _DEFAULT_DASHSCOPE_COMPAT_URL).strip()
    model = os.environ.get("STORY_PLOT_CRITIC_MODEL", "qwen-vl-max").strip()
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    from .character_bible import resolve_shot_characters

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
    scene_rubric_parts = [p for p in (canonical_action, f"场景：{location_hint}" if location_hint else "", f"景别：{framing_hint}" if framing_hint else "", f"阶段连续性：{continuity_hint}" if continuity_hint else "") if p]
    framing_note = (
        "（注：当前为空镜或群像镜头，无具名主角入画，请勿将路人/群像与主角外貌强行比对）\n"
        if not char_summary
        else (
            "（注：远景/大远景镜头中人物占画面比例较小，请重点核查宏观场景奇观、发色衣袍大轮廓与动作气势，勿苛求微距五官细节）\n"
            if "远景" in framing_hint
            else ""
        )
    )
    qa_profile = resolve_shot_qa_profile(manifest, shot)
    full_diag = shot.diagnostics if isinstance(shot.diagnostics, dict) and shot.diagnostics else diagnostics
    from .lipsync import classify_shot_speech_delivery

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
        f"你是AI动漫短剧质检监督员。请评估 stage={stage!r} 镜头 {shot.id} 是否达到 H3 电影级标准。\n"
        f"{rubric_block}\n"
        f"剧本动作：{'；'.join(scene_rubric_parts)}\n"
        f"本镜出场角色设定：{json.dumps(char_summary, ensure_ascii=False)}\n"
        f"{framing_note}"
        f"客观诊断指标：{json.dumps(diag_for_prompt, ensure_ascii=False)}\n"
        "硬性失败：画面像素里出现汉字、字幕、对白条或水印，记 LOCAL_DEFECT（注：青铜古剑/石碑/陶罐上的非文字装饰纹样不属于汉字字幕）。"
        f"{speech_rule}"
        "特效硬伤（廉价2D荧光油漆刀光条、平面线框斑马光柱、离地假棉花团云层、无体积受光的静态贴片光效、中段火焰/光球骤膨糊死主体或法阵空间严重穿帮）记 VFX_QUALITY_DEFECT。"
        "景别与人景比例硬伤（远景/建立镜却让前景人群或主角半身霸屏超过30%遮挡环境，或雪地/山路行进镜切掉小腿与双脚导致看不到脚下积雪与接地足迹）记 IMAGE_PROMPT_DRIFT。"
        "发色以本镜角色设定和阶段连续性为准，不要用其他章节的发色母版推翻当前阶段。\n"
        "严格输出单个 JSON 对象（不要额外解释）：\n"
        '{"decision":"passed|warning|failed","failure_code":"PASS|IMAGE_PROMPT_DRIFT|CHARACTER_INCONSISTENT|LOCAL_DEFECT|VIDEO_MOTION_FAIL|VFX_QUALITY_DEFECT",'
        '"scores":{"character":0.0-1.0,"motion":0.0-1.0,"expression":0.0-1.0,"camera_displacement":0.0-1.0,"lighting_style":0.0-1.0,"vfx_quality":0.0-1.0,"audio_sync":0.0-1.0,'
        '"plot_fidelity":0.0-1.0,"character_consistency":0.0-1.0,"visual_h3_grade":0.0-1.0},'
        '"reason":"简短中文原因",'
        '"rework_action":{"strategy":"regenerate_first_frame|patch_character_dna|reroute_video|none","revised_prompt":null,"negative_addition":null,"target_video_route":null}}'
    )
    char_refs = _collect_character_ref_uris(manifest, shot)
    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for ref in char_refs:
        label = ref.get("name") or ref.get("id") or "角色"
        content_blocks.append({"type": "text", "text": f"角色母版参考图（{label}）："})
        content_blocks.append({"type": "image_url", "image_url": {"url": ref["data_uri"]}})
    if char_refs:
        content_blocks.append({"type": "text", "text": "待审产物画面（请严格对照角色母版核查五官、发型与服饰一致性）："})
    content_blocks.append({"type": "image_url", "image_url": {"url": image_data_uri}})
    if stage == "video" and isinstance(diagnostics, dict):
        dense_path = diagnostics.get("dense_strip_path")
        if isinstance(dense_path, str) and dense_path.strip() and Path(dense_path).is_file():
            dense_uri = _encode_image_for_vlm(Path(dense_path), max_edge=1280)
            if dense_uri:
                content_blocks.append({"type": "text", "text": "8帧密集时序拆帧图（请核查全片动作连贯性、是否静止、光影色温是否漂移、是否有肢体穿模）："})
                content_blocks.append({"type": "image_url", "image_url": {"url": dense_uri}})
        roi_path = diagnostics.get("roi_zoom_path")
        if isinstance(roi_path, str) and roi_path.strip() and Path(roi_path).is_file():
            roi_uri = _encode_image_for_vlm(Path(roi_path), max_edge=1280)
            if roi_uri:
                content_blocks.append({"type": "text", "text": "人物局部放大对比条（请重点核查微表情、眼神、对白口型及面部五官稳定性）："})
                content_blocks.append({"type": "image_url", "image_url": {"url": roi_uri}})

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    }
    candidate_models = [model]
    raw_fallbacks = os.environ.get("STORY_PLOT_CRITIC_FALLBACK_MODELS", "").strip()
    if raw_fallbacks:
        for fb_m in (m.strip() for m in raw_fallbacks.split(",") if m.strip()):
            if fb_m not in candidate_models:
                candidate_models.append(fb_m)
    elif model.startswith("gpt-"):
        for fb_m in ("gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol"):
            if fb_m not in candidate_models:
                candidate_models.append(fb_m)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "openai-python/1.50.0",
    }
    parsed: dict[str, Any] | None = None
    used_model = model
    last_err: Exception | None = None
    for cand_model in candidate_models:
        cand_payload = {**payload, "model": cand_model}
        req_bytes = json.dumps(cand_payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(2):
            try:
                req = urllib_request.Request(base_url, req_bytes, headers)
                with urllib_request.urlopen(req, timeout=75) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    )
                parsed = _parse_json_response(str(content))
                used_model = cand_model
                break
            except urllib_error.HTTPError as exc:
                last_err = exc
                if exc.code in {400, 429, 500, 502, 503, 504}:
                    if attempt == 0 and cand_model == candidate_models[-1]:
                        import time

                        time.sleep(1.2)
                        continue
                    break
                raise
            except (urllib_error.URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
                last_err = exc
                if attempt == 0:
                    import time

                    time.sleep(1.2)
                    continue
                break
        if parsed is not None:
            break
    if parsed is None:
        assert last_err is not None
        raise last_err
    parsed.setdefault("supervisor_id", used_model)
    # Calibrate blind VLM audio_sync=0.50 placeholder using objective local av_sync/audio telemetry when no lip violation exists.
    obj_aud = compute_objective_audio_sync_score(
        full_diag,
        qa_profile,
        has_dialogue=(delivery == "on_screen_dialogue"),
    )
    if obj_aud is not None and isinstance(parsed.get("scores"), dict):
        vlm_aud = parsed["scores"].get("audio_sync")
        reason_txt = str(parsed.get("reason") or "")
        blind_audio_placeholder = (
            isinstance(vlm_aud, (int, float))
            and 0.40 <= float(vlm_aud) <= 0.62
            and any(
                tok in reason_txt
                for tok in ("无音轨", "缺少音轨", "未提供可听音轨", "无法确认音画同步", "无法仅凭", "暂按中性分", "缺少原始音轨", "无可听音轨")
            )
        )
        has_lip_defect = any(
            tok in reason_txt
            for tok in ("违反旁白", "违反闭口", "触发旁白镜头张嘴", "全程闭嘴", "近乎闭合", "基本闭合", "嚼口", "持续张口")
        )
        if blind_audio_placeholder and not has_lip_defect:
            parsed["scores"]["audio_sync"] = obj_aud
    return _normalize_verdict(parsed, stage=stage, shot_id=shot.id, diagnostics=diagnostics)


def critique_shot(
    manifest: ProjectManifest,
    shot: Shot,
    stage: str,
    artifact_path_or_sheet: str | Path | None,
    *,
    evaluator: Any = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a shot image or video contact sheet into a normalized quality-evaluation/v2 verdict."""
    clean_stage = (stage or "image").strip().lower()
    if clean_stage not in {"image", "video"}:
        raise ValueError(f"unsupported critic stage: {stage!r}")

    artifact_path = Path(artifact_path_or_sheet) if artifact_path_or_sheet else None
    if clean_stage == "image":
        obj_diag = extract_image_diagnostics(artifact_path)
    else:
        existing_vid_diag = (
            ((shot.diagnostics or {}).get("video") or {})
            if isinstance(shot.diagnostics, dict)
            else {}
        )
        if not existing_vid_diag:
            vid_asset = (getattr(shot, "assets", None) or {}).get("video")
            if vid_asset and Path(str(vid_asset)).is_file():
                existing_vid_diag = extract_video_diagnostics(Path(str(vid_asset)), shot_id=shot.id)
        if artifact_path and artifact_path.suffix.lower() in _IMAGE_SUFFIXES:
            sheet_diag = extract_image_diagnostics(artifact_path)
            obj_diag = {**existing_vid_diag, "contact_sheet_image": sheet_diag} if existing_vid_diag else sheet_diag
        elif existing_vid_diag:
            obj_diag = existing_vid_diag
        else:
            obj_diag = extract_video_diagnostics(artifact_path, shot_id=shot.id)

    if callable(evaluator):
        raw = evaluator(
            manifest=manifest,
            shot=shot,
            stage=clean_stage,
            artifact_path=artifact_path,
            diagnostics=obj_diag,
        )
        return _normalize_verdict(raw, stage=clean_stage, shot_id=shot.id, diagnostics=obj_diag)

    # Objective fast-fail before calling any remote VLM token endpoint.
    if artifact_path is None or not artifact_path.is_file():
        return _normalize_verdict(
            {
                "decision": "failed",
                "failure_code": "LOCAL_DEFECT" if clean_stage == "image" else "VIDEO_MOTION_FAIL",
                "scores": {"visual_h3_grade": 0.0},
                "reason": "待审产物文件缺失",
            },
            stage=clean_stage,
            shot_id=shot.id,
            diagnostics=obj_diag,
        )

    qa_profile = resolve_shot_qa_profile(manifest, shot)
    qa_thresholds = qa_profile.get("thresholds") or {}

    has_dialogue = bool(getattr(shot, "dialogue_lines", None))
    try:
        from .lipsync import classify_shot_speech_delivery

        bible = getattr(manifest, "character_bible", None)
        has_dialogue = classify_shot_speech_delivery(shot, bible) == "on_screen_dialogue"
    except Exception:
        pass
    shot_bundle = shot.diagnostics if isinstance(getattr(shot, "diagnostics", None), dict) else None
    if isinstance(shot_bundle, dict) and any(k in shot_bundle for k in ("video", "image", "objective_qa", "audio")):
        eval_diag = shot_bundle
    else:
        eval_diag = obj_diag
    obj_qa = evaluate_objective_qa_flags(
        eval_diag,
        qa_profile,
        stage=clean_stage,
        has_dialogue=has_dialogue,
    )
    if obj_qa.get("hard_fails"):
        fail = obj_qa["hard_fails"][0]
        code = str(fail.get("code") or "")
        failure_code = (
            "LOCAL_DEFECT"
            if clean_stage == "image" or code == "DARK_OR_BLANK_IMAGE"
            else "VIDEO_MOTION_FAIL"
        )
        return _normalize_verdict(
            {
                "decision": "failed",
                "failure_code": failure_code,
                "scores": {"visual_h3_grade": 0.1 if code == "DARK_OR_BLANK_IMAGE" else 0.25},
                "reason": str(fail.get("message") or "客观质检硬门失败"),
                "failures": [str(item.get("code")) for item in obj_qa["hard_fails"] if item.get("code")],
            },
            stage=clean_stage,
            shot_id=shot.id,
            diagnostics={**obj_diag, "objective_qa": obj_qa} if isinstance(obj_diag, dict) else obj_diag,
        )

    dark_floor = float(qa_thresholds.get("dark_luma_floor", 8.0))
    mean_luma = obj_diag.get("mean_luma")
    is_dark = (
        float(mean_luma) < dark_floor
        if isinstance(mean_luma, (int, float))
        else bool(obj_diag.get("is_blank_or_dark"))
    )
    sharp = float(obj_diag.get("sharpness") if obj_diag.get("sharpness") is not None else 999.0)
    if obj_diag.get("available") is True and is_dark and sharp < 3.0:
        return _normalize_verdict(
            {
                "decision": "failed",
                "failure_code": "LOCAL_DEFECT" if clean_stage == "image" else "VIDEO_MOTION_FAIL",
                "scores": {"visual_h3_grade": 0.1},
                "reason": "画面过暗或纯黑废帧，未达到可用清晰度",
            },
            stage=clean_stage,
            shot_id=shot.id,
            diagnostics=obj_diag,
        )

    if clean_stage == "image":
        split_reason = _detect_split_panel_bars(artifact_path)
        if split_reason:
            return _normalize_verdict(
                {
                    "decision": "failed",
                    "failure_code": "SPLIT_PANEL_TEXT",
                    "scores": {"visual_h3_grade": 0.15},
                    "reason": split_reason,
                    "rework_action": {
                        "strategy": "regenerate_first_frame",
                        "negative_addition": "multi-panel, split screen, collage grid, white borders, dividing gutter",
                    },
                },
                stage=clean_stage,
                shot_id=shot.id,
                diagnostics=obj_diag,
            )

    has_remote_vlm = bool(
        os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip()
        or os.environ.get("STORY_PLOT_CRITIC_API_KEY", "").strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
    )
    is_kinetic_action_shot = bool(
        (
            isinstance(getattr(shot, "scene_context", None), dict)
            and (shot.scene_context.get("action_beat") or shot.scene_context.get("impact_beat"))
        )
        or qa_thresholds.get("allow_impact_motion_blur")
    )
    if clean_stage == "video" and obj_diag.get("is_clipping_suspect") is True and not obj_diag.get("motion_blur_likely"):
        clip_info = obj_diag.get("clipping_suspect_frame") or {}
        clip_mode = obj_diag.get("clipping_mode") or clip_info.get("clipping_mode")
        retain_ratio = float(clip_info.get("sharpness_retain_ratio", 1.0))
        anomaly_score = float(clip_info.get("anomaly_score", 1.0))
        genre_clip_limit = float(qa_thresholds.get("clipping_retain_ratio", 0.42))
        effective_retain_limit = min(0.34, genre_clip_limit) if is_kinetic_action_shot else genre_clip_limit
        severe_without_vlm = not has_remote_vlm and (
            (clip_mode == "sharp_topology_tear" and not is_kinetic_action_shot)
            or retain_ratio < effective_retain_limit
        )
        severe_even_with_vlm = has_remote_vlm and (retain_ratio < 0.30 or anomaly_score >= 1.85)
        if severe_without_vlm or severe_even_with_vlm:
            reason = (
                "视频中段检测到锐利画面撕裂或肢体拓扑错位，疑似严重穿模崩坏"
                if clip_mode == "sharp_topology_tear"
                else "视频中段插值清晰度骤降，疑似发生人物穿模或重影崩坏"
            )
            return _normalize_verdict(
                {
                    "decision": "failed",
                    "failure_code": "VIDEO_MOTION_FAIL",
                    "scores": {"visual_h3_grade": 0.25},
                    "reason": reason,
                },
                stage=clean_stage,
                shot_id=shot.id,
                diagnostics=obj_diag,
            )

    custom_url = os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip()
    if custom_url:
        try:
            return _call_http_evaluator(
                custom_url,
                manifest=manifest,
                shot=shot,
                stage=clean_stage,
                artifact_path=artifact_path,
                diagnostics=obj_diag,
            )
        except Exception as exc:
            return {
                "schema": "quality-evaluation/v1",
                "schema_version": "quality-evaluation/v2",
                "stage": clean_stage,
                "shot_id": shot.id,
                "decision": "unavailable",
                "failure_code": "NONE",
                "scores": {},
                "overall_score": 0.0,
                "reason": f"质量模型服务异常: {type(exc).__name__}",
                "failures": [],
                "diagnostics": obj_diag,
                "rework_action": {"strategy": "none"},
            }

    api_key = (
        os.environ.get("STORY_PLOT_CRITIC_API_KEY", "").strip()
        or os.environ.get("DASHSCOPE_API_KEY", "").strip()
    )
    if api_key:
        try:
            return _call_dashscope_vlm(
                api_key,
                manifest=manifest,
                shot=shot,
                stage=clean_stage,
                artifact_path=artifact_path,
                diagnostics=obj_diag,
            )
        except Exception as exc:
            return {
                "schema": "quality-evaluation/v1",
                "schema_version": "quality-evaluation/v2",
                "stage": clean_stage,
                "shot_id": shot.id,
                "decision": "unavailable",
                "failure_code": "NONE",
                "scores": {},
                "overall_score": 0.0,
                "reason": f"DashScope VLM 裁判调用异常: {type(exc).__name__}",
                "failures": [],
                "diagnostics": obj_diag,
                "rework_action": {"strategy": "none"},
            }

    return {
        "schema": "quality-evaluation/v1",
        "schema_version": "quality-evaluation/v2",
        "stage": clean_stage,
        "shot_id": shot.id,
        "decision": "unavailable",
        "failure_code": "NONE",
        "scores": {},
        "overall_score": 0.0,
        "reason": "未配置 VLM 评估服务或 DASHSCOPE_API_KEY",
        "failures": [],
        "diagnostics": obj_diag,
        "rework_action": {"strategy": "none"},
    }
