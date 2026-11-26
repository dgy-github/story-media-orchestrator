"""Multi-tier automated rework engine gating paid video generation and bounding per-shot retries."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .character_bible import record_auto_anchor_frame, resolve_shot_characters
from .cost_ledger import check_and_record_cost
from .editing import probe_duration, resolve_elastic_shot_duration
from .flywheel import (
    record_intervention_case,
    resolve_flywheel_dir,
    resolve_pending_case_on_verdict,
)
from .frame_decomposer import mark_downstream_relay_staleness, refresh_shot_diagnostics
from .manifest import ProjectManifest, Shot
from .plot_critic import critique_shot


def _merge_negative(existing: str | None, addition: str | None) -> str | None:
    parts: list[str] = []
    seen: set[str] = set()
    for raw in (existing, addition):
        if not raw:
            continue
        for token in re.split(r"[，,、;；]+", str(raw)):
            clean = token.strip()
            if clean and clean not in seen:
                seen.add(clean)
                parts.append(clean)
    return "，".join(parts) if parts else None


def _capture_snapshot(manifest: ProjectManifest, shot: Shot) -> dict[str, Any]:
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    report = shot.quality_report if isinstance(shot.quality_report, dict) else {}
    return {
        "before_score": report.get("overall_score"),
        "before_prompt": shot.prompt_override or ctx.get("visual_action") or shot.text,
        "before_negative": shot.negative_override or ctx.get("negative") or "",
        "before_route": ctx.get("_video_route") or shot.video_route or manifest.video_routing_policy,
        "before_seed": shot.seed_override,
    }


def _apply_image_rework_strategy(
    manifest: ProjectManifest,
    shot: Shot,
    verdict: dict[str, Any],
    round_idx: int,
) -> dict[str, Any]:
    failure_code = str(verdict.get("failure_code") or "IMAGE_PROMPT_DRIFT").upper()
    action = verdict.get("rework_action") if isinstance(verdict.get("rework_action"), dict) else {}
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    base_action = str(shot.prompt_override or ctx.get("visual_action") or shot.text or "").strip()
    revised = str(action.get("revised_prompt") or "").strip()
    action_guidance: str | None = None

    if failure_code in {"IMAGE_PROMPT_DRIFT", "PLOT_DRIFT", "SPLIT_PANEL_TEXT"}:
        shot.prompt_override = revised or f"{base_action}；单帧完整构图，准确呈现关键剧情动作，无分屏无文字"
        shot.negative_override = _merge_negative(
            shot.negative_override,
            str(action.get("negative_addition") or "多格分屏，拼贴，多余文字"),
        )
        action_guidance = revised or "准确呈现关键剧情动作，严禁分屏或文字"
    elif failure_code == "CHARACTER_INCONSISTENT":
        resolved = resolve_shot_characters(manifest, shot)
        traits: list[str] = []
        locks: list[str] = list(shot.character_locks or [])
        for prof in resolved:
            name = str(prof.get("name") or prof.get("key") or "").strip()
            dna = str(prof.get("visual_dna") or prof.get("appearance") or prof.get("clothing") or "").strip()
            if name and name not in locks:
                locks.append(name)
            if name and dna:
                traits.append(f"{name}:{dna}")
        if locks:
            shot.character_locks = locks
        trait_text = "；".join(traits) if traits else "保持人物外貌与服装严格一致"
        head = revised or base_action
        shot.prompt_override = f"{head}；严格保持角色特征一致（{trait_text[:120]}）"
        if action.get("negative_addition"):
            shot.negative_override = _merge_negative(shot.negative_override, str(action["negative_addition"]))
        action_guidance = f"严格锁定角色外貌特征：{trait_text[:80]}"
    else:
        if revised:
            shot.prompt_override = revised
        shot.negative_override = _merge_negative(
            shot.negative_override,
            str(action.get("negative_addition") or "局部崩坏，畸形肢体，模糊残影，多余文字水印"),
        )
        if action.get("inpaint_bbox") is not None and isinstance(shot.scene_context, dict):
            shot.scene_context["_last_inpaint_bbox"] = action["inpaint_bbox"]
        action_guidance = "抑制手部与面部局部崩坏"

    shot.seed_override = (int(shot.seed_override) if shot.seed_override is not None else 1000) + round_idx * 7919
    return {
        "stage": "image",
        "prompt_override": shot.prompt_override,
        "negative_override": shot.negative_override,
        "seed_override": shot.seed_override,
        "action_guidance": action_guidance,
    }


def run_image_pre_audit_and_rework(
    manifest: ProjectManifest,
    root: str | Path,
    shot: Shot,
    frame_path: Path,
    image_provider: Any,
    *,
    shot_index: int,
    evaluator: Any = None,
) -> tuple[Path, bool]:
    """Audit generated keyframe and perform up to max_rework_rounds image repairs before video spend."""
    project_root = Path(root)
    manifest_path = project_root / "project.json"
    verdict = critique_shot(manifest, shot, "image", frame_path, evaluator=evaluator, root=project_root)
    shot.quality_report = verdict
    require_critic = bool(
        isinstance(manifest.supervisor_policy, dict)
        and manifest.supervisor_policy.get("require_vlm_critic")
    )
    if verdict.get("decision") == "unavailable" and require_critic:
        shot.review = "needs_takeover"
        shot.error = f"critic_unavailable:image:{verdict.get('reason', '')}"
        return frame_path, False
    if verdict.get("decision") != "failed":
        return frame_path, True

    fly_dir = resolve_flywheel_dir(manifest, project_root)
    max_rounds = max(0, int(getattr(manifest, "max_rework_rounds", 2)))
    current_path = frame_path

    for round_idx in range(1, max_rounds + 1):
        before_snap = _capture_snapshot(manifest, shot)
        failure_code = str(verdict.get("failure_code") or "IMAGE_PROMPT_DRIFT").upper()
        intervention = _apply_image_rework_strategy(manifest, shot, verdict, round_idx)

        if fly_dir is not None:
            record_intervention_case(
                fly_dir,
                manifest=manifest,
                shot=shot,
                failure_code=failure_code,
                failure_reason=str(verdict.get("reason") or ""),
                strategy="auto_image_rework",
                intervention=intervention,
                supervisor_id=str(verdict.get("supervisor_id") or "plot-critic"),
                before_snapshot=before_snap,
            )

        shot.rework_history.append(
            {
                "source": "auto_rework",
                "stage": "image",
                "round": round_idx,
                "failure_code": failure_code,
                "prompt_override": shot.prompt_override,
                "negative_override": shot.negative_override,
                "seed_override": shot.seed_override,
            }
        )

        check_and_record_cost(
            manifest,
            shot.id,
            "image",
            manifest.image_source,
            round_idx=round_idx,
            reason=f"auto_rework:{failure_code}",
            save_path=manifest_path,
        )

        shot.image_task.clear()
        try:
            current_path = Path(image_provider.generate(shot, current_path))
        except Exception as img_exc:
            if current_path.is_file() and any(
                tok in str(img_exc) for tok in ("auth_unavailable", "usage_limit_reached", "HTTP 503", "Codex")
            ):
                return current_path, True
            raise
        shot.assets["image"] = str(current_path)
        if isinstance(shot.scene_context, dict):
            shot.scene_context.pop("_relay_continuity_stale", None)
            shot.scene_context.pop("_relay_stale_upstream_shot_id", None)
        mark_downstream_relay_staleness(manifest, shot.id)
        record_auto_anchor_frame(manifest, project_root, shot, current_path)

        verdict = critique_shot(manifest, shot, "image", current_path, evaluator=evaluator, root=project_root)
        shot.quality_report = verdict
        if verdict.get("decision") == "unavailable" and require_critic:
            if fly_dir is not None:
                resolve_pending_case_on_verdict(fly_dir, shot, verdict)
            shot.review = "needs_takeover"
            shot.error = f"critic_unavailable:image:{verdict.get('reason', '')}"
            return current_path, False
        if verdict.get("decision") in {"passed", "warning"}:
            if fly_dir is not None:
                resolve_pending_case_on_verdict(fly_dir, shot, verdict)
            return current_path, True

    if fly_dir is not None:
        resolve_pending_case_on_verdict(fly_dir, shot, verdict)
    shot.review = "needs_takeover"
    shot.error = f"image_rework_exhausted:{verdict.get('failure_code', 'OTHER')}:{verdict.get('reason', '')}"
    return current_path, False


def _select_upgraded_video_route(
    shot: Shot,
    verdict: dict[str, Any],
    *,
    manifest: ProjectManifest | None = None,
) -> str:
    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    provider_setting = getattr(manifest, "video_provider", "") if manifest is not None else ""
    default_fallback = "comfy_i2v" if provider_setting == "comfyui" else "i2v_fast"
    current_route = str(ctx.get("_video_route") or shot.video_route or default_fallback)
    action = verdict.get("rework_action") if isinstance(verdict.get("rework_action"), dict) else {}
    target_route = str(action.get("target_video_route") or "").strip()
    has_last = bool(shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file())

    is_comfy = current_route.startswith("comfy_") or (
        provider_setting == "comfyui" and current_route in {"preview", "auto", ""}
    )
    if is_comfy:
        if target_route in {"kf2v", "i2v_quality", "comfy_fl2v"}:
            return "comfy_fl2v"
        if target_route in {"comfy_i2v", "comfy_t2v", "preview"}:
            return target_route
        if current_route in {"preview", "comfy_t2v"} and not has_last:
            return "comfy_i2v"
        return "comfy_fl2v"

    if target_route == "kf2v" and not has_last:
        target_route = "i2v_quality"
    if target_route:
        return target_route
    if has_last and current_route != "kf2v":
        return "kf2v"
    if current_route in {"i2v_fast", "t2v", "comfy_t2v", "preview"}:
        return "i2v_quality"
    return current_route


def run_video_post_audit_and_rework(
    manifest: ProjectManifest,
    root: str | Path,
    shot: Shot,
    frame_path: Path,
    video_path: Path,
    video_provider: Any,
    *,
    audio_dur: float,
    alignment_mode: str,
    evaluator: Any = None,
    ffmpeg_resolver: Any = None,
    image_provider: Any = None,
    shot_index: int = 1,
) -> Path:
    """Extract 2x2 contact sheet, audit motion fidelity, and upgrade video route/seed if needed."""
    project_root = Path(root)
    manifest_path = project_root / "project.json"
    diag = refresh_shot_diagnostics(manifest, project_root, shot.id)
    sheet_or_video = (diag.get("video") or {}).get("contact_sheet_path") or video_path
    verdict = critique_shot(manifest, shot, "video", sheet_or_video, evaluator=evaluator, root=project_root)
    shot.quality_report = verdict
    require_critic = bool(
        isinstance(manifest.supervisor_policy, dict)
        and manifest.supervisor_policy.get("require_vlm_critic")
    )
    if verdict.get("decision") == "unavailable" and require_critic:
        shot.review = "needs_takeover"
        shot.error = f"critic_unavailable:video:{verdict.get('reason', '')}"
        return video_path
    if verdict.get("decision") != "failed":
        return video_path

    fly_dir = resolve_flywheel_dir(manifest, project_root)
    used_rounds = len(
        [item for item in shot.rework_history if isinstance(item, dict) and item.get("source") == "auto_rework"]
    )
    remaining = max(0, int(getattr(manifest, "max_rework_rounds", 2)) - used_rounds)
    current_video = video_path

    for step in range(1, remaining + 1):
        round_num = used_rounds + step
        before_snap = _capture_snapshot(manifest, shot)
        target_route = _select_upgraded_video_route(shot, verdict, manifest=manifest)
        failure_code = str(verdict.get("failure_code") or "VIDEO_MOTION_FAIL").upper()

        if (
            target_route == "comfy_fl2v"
            and not (shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file())
            and image_provider is not None
        ):
            from .codex_image import ensure_shot_end_keyframe

            try:
                ensure_shot_end_keyframe(
                    manifest,
                    project_root,
                    shot,
                    frame_path,
                    image_provider,
                    shot_index=shot_index,
                    reason=f"auto_rework:{failure_code}:end_keyframe",
                )
            except Exception as end_exc:
                if any(tok in str(end_exc) for tok in ("auth_unavailable", "usage_limit_reached", "HTTP 503", "Codex")):
                    target_route = "comfy_i2v"
                else:
                    raise

        shot.video_route = target_route
        shot.seed_override = (int(shot.seed_override) if shot.seed_override is not None else 1000) + round_num * 7919

        if fly_dir is not None:
            record_intervention_case(
                fly_dir,
                manifest=manifest,
                shot=shot,
                failure_code=failure_code,
                failure_reason=str(verdict.get("reason") or ""),
                strategy="reroute_video",
                intervention={
                    "stage": "video",
                    "video_route": target_route,
                    "seed_override": shot.seed_override,
                },
                supervisor_id=str(verdict.get("supervisor_id") or "plot-critic"),
                before_snapshot=before_snap,
            )

        shot.rework_history.append(
            {
                "source": "auto_rework",
                "stage": "video",
                "round": round_num,
                "failure_code": failure_code,
                "video_route": target_route,
                "seed_override": shot.seed_override,
            }
        )

        check_and_record_cost(
            manifest,
            shot.id,
            "video",
            target_route,
            round_idx=round_num,
            reason=f"auto_rework:{failure_code}",
            save_path=manifest_path,
        )

        # Keep first frame and audio track untouched; only regenerate the video clip.
        current_video = Path(video_provider.generate(shot, frame_path, current_video))
        shot.assets["video"] = str(current_video)
        shot.mode = "render"
        if isinstance(shot.scene_context, dict):
            shot.scene_context.pop("_relay_continuity_stale", None)
            shot.scene_context.pop("_relay_stale_upstream_shot_id", None)
        mark_downstream_relay_staleness(manifest, shot.id)
        if alignment_mode == "elastic" and manifest.tts_source != "silent" and callable(ffmpeg_resolver):
            opt_ffmpeg = ffmpeg_resolver()
            if opt_ffmpeg:
                vid_dur = probe_duration(opt_ffmpeg, current_video)
                if vid_dur is not None:
                    shot.scene_context["_video_duration"] = vid_dur
                    shot.duration = resolve_elastic_shot_duration(
                        audio_dur,
                        vid_dur,
                        tts_source=manifest.tts_source,
                        alignment_mode=alignment_mode,
                    )

        diag = refresh_shot_diagnostics(manifest, project_root, shot.id)
        sheet_or_video = (diag.get("video") or {}).get("contact_sheet_path") or current_video
        verdict = critique_shot(manifest, shot, "video", sheet_or_video, evaluator=evaluator, root=project_root)
        shot.quality_report = verdict
        if verdict.get("decision") == "unavailable" and require_critic:
            if fly_dir is not None:
                resolve_pending_case_on_verdict(fly_dir, shot, verdict)
            shot.review = "needs_takeover"
            shot.error = f"critic_unavailable:video:{verdict.get('reason', '')}"
            return current_video
        if verdict.get("decision") in {"passed", "warning"}:
            if fly_dir is not None:
                resolve_pending_case_on_verdict(fly_dir, shot, verdict)
            return current_video

    if fly_dir is not None:
        resolve_pending_case_on_verdict(fly_dir, shot, verdict)
    shot.review = "needs_takeover"
    shot.error = f"video_rework_exhausted:{verdict.get('failure_code', 'VIDEO_MOTION_FAIL')}:{verdict.get('reason', '')}"
    return current_video
