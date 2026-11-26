"""Domain service for external Supervisor takeover, inspection, surgical rework, and finalization."""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4
import json
import wave

from .candidates import select_candidate
from .character_bible import (
    build_consistent_image_scene,
    build_photo_to_3d_anime_prompt,
    generate_character_portrait,
    invalidate_character_shots,
    record_auto_anchor_frame,
    resolve_shot_characters,
    safe_character_filename,
    sync_character_bible,
    upsert_character_profile,
)
from .cost_ledger import (
    BudgetExceededError,
    check_rework_budget,
    estimate_operation_cost,
    record_cost_entry,
    summarize_cost_ledger,
)
from .editing import (
    MOTIONS,
    default_image_size_for_aspect,
    probe_duration,
    resolve_canvas_size,
    resolve_elastic_shot_duration,
)
from .flywheel import (
    FAILURE_CODES,
    get_flywheel_stats,
    load_active_playbook,
    record_intervention_case,
    resolve_flywheel_dir,
    resolve_pending_case_on_verdict,
)
from .frame_decomposer import mark_downstream_relay_staleness, refresh_shot_diagnostics
from .manifest import ProjectManifest, Shot
from .preview import _shot_cache_key, render_preview
from .qa_profiles import build_vlm_qa_rubric_prompt, resolve_shot_qa_profile
from .quality import build_supervisor_verdict
from .registry import ArtifactRegistry
from .review_gate import approve, fingerprint
from .tts import FakeTTSProvider, synthesize_shot_audio
from .video_router import resolve_shot_video_route, validate_shot_video_route

_LOCK_FILENAME = ".takeover.lock"
_VALID_TAKEOVER_ROLES = ("supervisor", "asset_producer", "critic")


def _lock_path_for_role(root: Path, role: str = "supervisor") -> Path:
    clean_role = (role or "supervisor").strip().lower()
    if clean_role not in _VALID_TAKEOVER_ROLES:
        raise ValueError(f"unsupported takeover role: {role!r}; expected one of {_VALID_TAKEOVER_ROLES}")
    if clean_role == "supervisor":
        return root / _LOCK_FILENAME
    return root / f".takeover.{clean_role}.lock"
_VALID_STRATEGIES = (
    "rewrite_image_prompt",
    "patch_character_dna",
    "reroute_video",
    "fallback_25d_motion",
    "select_candidate",
    "reset_shot",
)
_ROLE_TO_DIR_EXT = {
    "image": ("frames", "{idx:04d}.png"),
    "last_frame": ("frames", "{idx:04d}_last.png"),
    "video": ("video", "{idx:04d}.mp4"),
    "audio": ("audio", "{idx:04d}.wav"),
}


class TakeoverConflictError(RuntimeError):
    """Raised when an operation conflicts with an active Supervisor takeover lease."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _manifest_path(project_dir: str | Path) -> tuple[Path, Path]:
    root = Path(project_dir).resolve()
    manifest_file = root / "project.json"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"project.json not found in {root}")
    return root, manifest_file


def _read_lock(lock_path: Path) -> dict[str, Any] | None:
    if not lock_path.is_file():
        return None
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _is_lock_expired(lock: dict[str, Any], *, now: datetime | None = None) -> bool:
    raw_exp = lock.get("expires_at")
    if not isinstance(raw_exp, str) or not raw_exp.strip():
        return True
    try:
        expires_at = datetime.fromisoformat(raw_exp)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return expires_at <= (now or _utcnow())


def claim_takeover(
    project_dir: str | Path,
    *,
    supervisor_id: str = "gemini-4-carbon",
    lease_seconds: float = 300.0,
    force: bool = False,
    role: str = "supervisor",
) -> dict[str, Any]:
    """Acquire a supervisor (exclusive) or role-scoped (asset_producer / critic) takeover lease."""
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, (int, float)) or lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive number")
    clean_role = (role or "supervisor").strip().lower()
    if clean_role not in _VALID_TAKEOVER_ROLES:
        raise ValueError(f"unsupported takeover role: {role!r}; expected one of {_VALID_TAKEOVER_ROLES}")

    root, _ = _manifest_path(project_dir)
    now = _utcnow()

    # Exclusive supervisor lock blocks role-scoped leases unless force=True.
    if clean_role != "supervisor":
        sup_lock = _read_lock(root / _LOCK_FILENAME)
        if sup_lock is not None and not _is_lock_expired(sup_lock, now=now) and not force:
            raise TakeoverConflictError(
                f"project is exclusively locked by supervisor {sup_lock.get('supervisor_id')} until {sup_lock.get('expires_at')}"
            )

    lock_path = _lock_path_for_role(root, clean_role)
    existing = _read_lock(lock_path)
    if existing is not None and not _is_lock_expired(existing, now=now) and not force:
        raise TakeoverConflictError(
            f"project role {clean_role!r} is already locked by {existing.get('supervisor_id')} until {existing.get('expires_at')}"
        )

    clean_supervisor = str(supervisor_id or "gemini-4-carbon").strip() or "gemini-4-carbon"
    expires_at = now + timedelta(seconds=float(lease_seconds))
    lock_data = {
        "schema": "takeover-lease/v1",
        "project_dir": str(root),
        "role": clean_role,
        "supervisor_id": clean_supervisor,
        "lease_token": f"lease_{uuid4().hex}",
        "acquired_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "lease_seconds": float(lease_seconds),
    }
    tmp = lock_path.with_name(f".takeover.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(lock_data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(lock_path)
    finally:
        tmp.unlink(missing_ok=True)
    return lock_data


def verify_takeover(
    project_dir: str | Path,
    lease_token: str | None = None,
    *,
    allowed_roles: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Validate an active exclusive supervisor lock or matching role-scoped lock; expired locks are cleaned up."""
    root = Path(project_dir).resolve()
    sup_path = root / _LOCK_FILENAME
    sup_lock = _read_lock(sup_path)
    if sup_lock is not None:
        if _is_lock_expired(sup_lock):
            sup_path.unlink(missing_ok=True)
        else:
            if not lease_token or lease_token != sup_lock.get("lease_token"):
                raise TakeoverConflictError(
                    f"active takeover lease belongs to {sup_lock.get('supervisor_id')} until {sup_lock.get('expires_at')}; "
                    "pass matching lease_token or wait for expiration"
                )
            return sup_lock

    for role_name in allowed_roles:
        if role_name == "supervisor":
            continue
        r_path = _lock_path_for_role(root, role_name)
        r_lock = _read_lock(r_path)
        if r_lock is None:
            continue
        if _is_lock_expired(r_lock):
            r_path.unlink(missing_ok=True)
            continue
        if not lease_token or lease_token != r_lock.get("lease_token"):
            raise TakeoverConflictError(
                f"active {role_name!r} lease belongs to {r_lock.get('supervisor_id')} until {r_lock.get('expires_at')}; "
                "pass matching lease_token or wait for expiration"
            )
        return r_lock
    return None


def release_takeover(
    project_dir: str | Path,
    lease_token: str | None = None,
    *,
    role: str | None = None,
) -> dict[str, Any]:
    """Release an active takeover lease when lease_token matches or lock is expired."""
    root = Path(project_dir).resolve()
    roles_to_check = (role,) if role is not None else _VALID_TAKEOVER_ROLES
    matched_path: Path | None = None
    matched_lock: dict[str, Any] | None = None
    mismatched_active = False

    for r_name in roles_to_check:
        lock_path = _lock_path_for_role(root, r_name)
        existing = _read_lock(lock_path)
        if existing is None:
            continue
        if _is_lock_expired(existing) or existing.get("lease_token") == lease_token:
            matched_path = lock_path
            matched_lock = existing
            break
        mismatched_active = True

    if matched_path is None or matched_lock is None:
        if mismatched_active:
            raise TakeoverConflictError("lease_token does not match active takeover lock")
        return {"released": False, "reason": "no_active_lock"}

    matched_path.unlink(missing_ok=True)
    return {
        "released": True,
        "role": matched_lock.get("role", "supervisor"),
        "supervisor_id": matched_lock.get("supervisor_id"),
        "project_dir": str(root),
    }


def _load_visual_attachment(label: str, file_path: str | Path | None) -> tuple[str, bytes, str] | None:
    if not file_path:
        return None
    path = Path(file_path)
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return None

    from io import BytesIO
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        max_dim = max(rgb.size)
        if max_dim > 1280:
            scale = 1280.0 / max_dim
            rgb = rgb.resize(
                (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buf = BytesIO()
        if suffix in {".jpg", ".jpeg"}:
            rgb.save(buf, format="JPEG", quality=88)
            fmt = "jpeg"
        else:
            rgb.save(buf, format="PNG")
            fmt = "png"
        return label, buf.getvalue(), fmt


def _collect_shot_visuals(
    shot: Shot,
    *,
    include_candidates: bool = False,
) -> list[tuple[str, bytes, str]]:
    visuals: list[tuple[str, bytes, str]] = []
    seen_paths: set[str] = set()

    def maybe_add(label: str, raw_path: str | Path | None) -> None:
        if not raw_path:
            return
        resolved = str(Path(raw_path).resolve())
        if resolved in seen_paths:
            return
        loaded = _load_visual_attachment(label, resolved)
        if loaded is not None:
            seen_paths.add(resolved)
            visuals.append(loaded)

    diag = shot.diagnostics if isinstance(shot.diagnostics, dict) else {}
    vid_diag = diag.get("video") if isinstance(diag.get("video"), dict) else {}
    maybe_add("video_contact_sheet", vid_diag.get("contact_sheet_path"))
    maybe_add("video_dense_strip", vid_diag.get("dense_strip_path"))
    maybe_add("video_roi_zoom", vid_diag.get("roi_zoom_path"))
    maybe_add("image", shot.assets.get("image"))
    maybe_add("last_frame", shot.assets.get("last_frame"))
    if include_candidates:
        for idx, cand in enumerate(shot.candidates[:2], 1):
            if isinstance(cand, dict):
                maybe_add(f"candidate_{idx}_{cand.get('id')}", cand.get("path"))
    return visuals


def get_pipeline_snapshot(
    project_dir: str | Path,
    *,
    server_flywheel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return full 5-stage pipeline status, shot matrix, review gates, ledger, and flywheel telemetry."""
    root, manifest_path = _manifest_path(project_dir)
    manifest = ProjectManifest.load(manifest_path)
    total = max(1, len(manifest.shots))

    shots_summary: list[dict[str, Any]] = []
    for idx, shot in enumerate(manifest.shots):
        route_info = resolve_shot_video_route(
            manifest,
            shot,
            has_first_frame=bool(shot.assets.get("image")),
            has_last_frame=bool(shot.assets.get("last_frame")),
            shot_index=idx,
            total_shots=total,
        )
        qr = shot.quality_report if isinstance(shot.quality_report, dict) else {}
        diag = shot.diagnostics if isinstance(shot.diagnostics, dict) else {}
        vid_diag = diag.get("video") if isinstance(diag.get("video"), dict) else {}
        img_diag = diag.get("image") if isinstance(diag.get("image"), dict) else {}
        shots_summary.append(
            {
                "shot_id": shot.id,
                "scene_id": shot.scene_id,
                "text": shot.text,
                "status": shot.status,
                "mode": shot.mode,
                "duration": shot.duration,
                "motion": shot.motion,
                "transition": shot.transition,
                "video_route": shot.video_route or route_info.get("route"),
                "assets_present": sorted(shot.assets.keys()),
                "quality_decision": qr.get("decision") or "unreviewed",
                "quality_score": qr.get("overall_score"),
                "failure_code": qr.get("failure_code"),
                "rework_count": len(shot.rework_history),
                "has_diagnostics": bool(img_diag.get("available") or vid_diag.get("available")),
                "image_sharpness": img_diag.get("sharpness"),
                "video_is_static_suspect": vid_diag.get("is_static_suspect"),
                "video_motion_mean_diff": vid_diag.get("motion_mean_diff"),
                "error": shot.error,
            }
        )

    lock = _read_lock(root / _LOCK_FILENAME)
    active_lock = None
    if lock is not None and not _is_lock_expired(lock):
        active_lock = {
            "supervisor_id": lock.get("supervisor_id"),
            "expires_at": lock.get("expires_at"),
        }

    fly_dir = resolve_flywheel_dir(manifest, root, server_default_dir=server_flywheel_dir)
    flywheel_stats = None
    if fly_dir is not None and (fly_dir / "playbook.json").is_file():
        flywheel_stats = get_flywheel_stats(fly_dir)["playbook"].get("stats")

    return {
        "schema": "pipeline-snapshot/v1",
        "project_id": manifest.project_id,
        "project_dir": str(root),
        "status": manifest.status,
        "mode": manifest.mode,
        "output": manifest.output,
        "jianying_draft": manifest.jianying_draft,
        "settings": {
            "aspect_ratio": manifest.aspect_ratio,
            "scale_mode": manifest.scale_mode,
            "alignment_mode": manifest.alignment_mode,
            "subtitle_style": manifest.subtitle_style,
            "speech_mode": manifest.speech_mode,
            "tts_source": manifest.tts_source,
            "image_source": manifest.image_source,
            "image_model": manifest.image_model,
            "image_size": manifest.image_size,
            "video_provider": manifest.video_provider,
            "video_routing_policy": manifest.video_routing_policy,
            "consistency_mode": manifest.consistency_mode,
            "style_preset": manifest.style_preset,
            "genre": getattr(manifest, "genre", "general"),
        },
        "review_gates": {
            "review_enforced": manifest.review_enforced,
            "pending_gate": manifest.status if manifest.status in {"awaiting_storyboard_review", "awaiting_asset_review"} else None,
            "approvals": dict(manifest.approvals),
        },
        "active_takeover_lock": active_lock,
        "character_count": len(manifest.character_bible),
        "shots": shots_summary,
        "cost_ledger": summarize_cost_ledger(manifest),
        "flywheel_stats": flywheel_stats,
    }


def inspect_shot_bundle(
    project_dir: str | Path,
    shot_id: str,
    *,
    refresh_diagnostics: bool = True,
    include_candidates: bool = False,
    server_flywheel_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str]]]:
    """Build complete multimodal inspection payload + visual frames for a single shot."""
    root, manifest_path = _manifest_path(project_dir)
    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)

    if refresh_diagnostics:
        refresh_shot_diagnostics(manifest, root, shot_id)
        manifest.save(manifest_path)

    visual_action = (shot.scene_context or {}).get("visual_action") or shot.text
    base_scene = {
        "summary": shot.text,
        "description": visual_action,
        "action": visual_action + "。仅绘制当前动作的一张完整画面、一个时间点。不要分屏、拼贴、上下多格、漫画分镜、字幕或文字。",
        "location": (shot.scene_context or {}).get("location", "以动作描述中的地点为准"),
        "lighting": (shot.scene_context or {}).get("lighting") or (shot.scene_context or {}).get("time") or "以动作描述中的时间与光线为准",
        "shot_id": shot.id,
        **{
            k: shot.scene_context[k]
            for k in ("framing", "mood", "negative", "continuity")
            if isinstance(shot.scene_context, dict) and k in shot.scene_context
        },
    }
    effective_scene, conditioning = build_consistent_image_scene(
        manifest,
        shot,
        base_scene,
        project_root=root if (root / "flywheel").is_dir() else None,
    )
    resolved_chars = resolve_shot_characters(manifest, shot)

    shot_idx = next((i for i, item in enumerate(manifest.shots) if item.id == shot.id), 0)
    route_info = resolve_shot_video_route(
        manifest,
        shot,
        has_first_frame=bool(shot.assets.get("image")),
        has_last_frame=bool(shot.assets.get("last_frame")),
        has_character_refs=any(bool(c.get("ref_image")) for c in resolved_chars),
        shot_index=shot_idx,
        total_shots=max(1, len(manifest.shots)),
    )

    playbook = load_active_playbook(
        manifest,
        root if (root / "flywheel").is_dir() else None,
        server_default_dir=server_flywheel_dir,
    )
    exemplars = (playbook.get("exemplars") or [])[:5] if playbook else []

    visuals = _collect_shot_visuals(shot, include_candidates=include_candidates)
    qa_profile = resolve_shot_qa_profile(manifest, shot)
    qa_stage = "video" if bool(shot.assets.get("video")) else "image"
    qa_rubric_prompt = build_vlm_qa_rubric_prompt(
        qa_profile,
        stage=qa_stage,
        shot=shot,
        diagnostics=shot.diagnostics if isinstance(shot.diagnostics, dict) else None,
    )
    bundle = {
        "schema": "shot-inspection/v1",
        "project_id": manifest.project_id,
        "project_dir": str(root),
        "project_settings": {
            "aspect_ratio": manifest.aspect_ratio,
            "style_preset": manifest.style_preset,
            "genre": getattr(manifest, "genre", "general"),
            "consistency_mode": manifest.consistency_mode,
            "video_routing_policy": manifest.video_routing_policy,
            "budget_limit_cny": manifest.budget_limit_cny,
        },
        "qa_profile": qa_profile,
        "qa_rubric_prompt": qa_rubric_prompt,
        "shot": {
            "id": shot.id,
            "scene_id": shot.scene_id,
            "text": shot.text,
            "narration": shot.narration,
            "dialogue_lines": shot.dialogue_lines,
            "duration": shot.duration,
            "status": shot.status,
            "mode": shot.mode,
            "motion": shot.motion,
            "transition": shot.transition,
            "video_route": shot.video_route,
            "resolved_video_route": route_info,
            "character_locks": shot.character_locks,
            "prompt_override": shot.prompt_override,
            "negative_override": shot.negative_override,
            "seed_override": shot.seed_override,
            "assets": shot.assets,
            "rework_history": shot.rework_history,
        },
        "resolved_characters": [
            {
                "key": c.get("key"),
                "name": c.get("name"),
                "gender": c.get("gender"),
                "visual_dna": c.get("visual_dna"),
                "seed": c.get("seed"),
                "has_ref_image": bool(c.get("ref_image")),
            }
            for c in resolved_chars
        ],
        "effective_image_scene": {
            **{k: v for k, v in effective_scene.items() if isinstance(v, (str, int, float, bool))},
            "conditioning_seed": conditioning.get("seed"),
            "uses_ref_img": bool(conditioning.get("ref_img")),
            "applied_flywheel_rules": conditioning.get("applied_flywheel_rules", []),
        },
        "diagnostics": shot.diagnostics,
        "quality_report": shot.quality_report,
        "cost_ledger": summarize_cost_ledger(manifest),
        "flywheel_exemplars": exemplars,
        "attached_visuals": [label for label, _, _ in visuals],
    }
    return bundle, visuals


def submit_quality_verdict(
    project_dir: str | Path,
    shot_id: str,
    *,
    stage: str = "image",
    decision: str,
    scores: dict[str, float] | None = None,
    failure_code: str = "NONE",
    reason: str = "",
    failures: list[str] | None = None,
    supervisor_id: str = "gemini-4-carbon",
    lease_token: str | None = None,
    server_flywheel_dir: str | Path | None = None,
    genre: str | None = None,
) -> dict[str, Any]:
    """Record a supervisor quality verdict and resolve any pending Quality Flywheel trajectory."""
    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token, allowed_roles=("critic",))
    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)

    effective_genre = genre
    if effective_genre is None:
        m_genre = getattr(manifest, "genre", "general")
        if isinstance(m_genre, str) and m_genre.strip() and m_genre.strip().lower() != "general":
            effective_genre = m_genre.strip()

    verdict = build_supervisor_verdict(
        stage=stage,
        decision=decision,
        scores=scores,
        failure_code=failure_code,
        reason=reason,
        failures=failures,
        supervisor_id=supervisor_id,
        round_index=len(shot.rework_history),
        genre=effective_genre,
    )
    shot.quality_report = verdict
    shot.review = {
        "passed": "approved",
        "warning": "needs_review",
        "failed": "rejected",
    }[verdict["decision"]]
    if verdict["decision"] == "passed" and isinstance(shot.scene_context, dict):
        shot.scene_context.pop("_relay_continuity_stale", None)
        shot.scene_context.pop("_relay_stale_upstream_shot_id", None)

    fly_dir = resolve_flywheel_dir(manifest, root, server_default_dir=server_flywheel_dir)
    flywheel_update = None
    if fly_dir is not None:
        flywheel_update = resolve_pending_case_on_verdict(
            fly_dir,
            shot,
            verdict,
            project_id=manifest.project_id,
        )

    manifest.save(manifest_path)
    return {
        "shot_id": shot.id,
        "verdict": verdict,
        "review": shot.review,
        "flywheel_update": flywheel_update,
    }


def _resolve_single_shot_image_provider(manifest: ProjectManifest, root: Path):
    if manifest.image_source in {"dashscope", "codex"}:
        from .providers import build_project_image_provider

        return build_project_image_provider(manifest, root)
    if manifest.real_preview:
        from .local_media import StoryCardProvider

        return StoryCardProvider(size=resolve_canvas_size(manifest.aspect_ratio))
    from .providers import TextFrameProvider

    return TextFrameProvider()


def _resolve_single_shot_tts(manifest: ProjectManifest):
    if manifest.tts_source == "windows":
        from .tts import WindowsTTSProvider

        return WindowsTTSProvider()
    if manifest.tts_source == "cosyvoice":
        from .tts import CosyVoiceTTSProvider

        return CosyVoiceTTSProvider()
    if manifest.tts_source == "edge":
        from .tts import EdgeNeuralTTSProvider

        return EdgeNeuralTTSProvider()
    return FakeTTSProvider()


def rework_single_shot(
    manifest: ProjectManifest,
    root: str | Path,
    shot_id: str,
    *,
    image_provider: Any = None,
    video_provider: Any = None,
    tts: Any = None,
    round_index: int = 1,
    reason: str = "supervisor_rework",
    use_last_frame: bool | None = None,
    final_prompt_override: str | None = None,
) -> dict[str, Any]:
    """Regenerate only invalidated/missing assets for `shot_id` without touching sibling shots."""
    project_root = Path(root).resolve()
    shot_idx, shot = next(
        ((i, item) for i, item in enumerate(manifest.shots, 1) if item.id == shot_id),
        (None, None),
    )
    if shot_idx is None or shot is None:
        raise KeyError(shot_id)

    consistency_mode = getattr(manifest, "consistency_mode", "auto")
    if consistency_mode != "off":
        sync_character_bible(manifest, project_root=project_root if (project_root / "flywheel").is_dir() else None)

    executed_stages: list[str] = []
    round_cost_cny = 0.0

    img_existing = shot.assets.get("image")
    if not img_existing or not Path(img_existing).is_file():
        img_prov = image_provider or _resolve_single_shot_image_provider(manifest, project_root)
        frames_dir = project_root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        out_img = img_prov.generate(shot, frames_dir / f"{shot_idx:04d}.png")
        shot.assets["image"] = str(out_img)
        record_auto_anchor_frame(manifest, project_root, shot, out_img)
        cost = estimate_operation_cost("image", manifest.image_source, manifest=manifest)
        record_cost_entry(
            manifest,
            shot_id=shot.id,
            stage="image",
            provider_or_route=f"image:{manifest.image_source}",
            cost_cny=cost,
            reason=reason,
            round_index=round_index,
        )
        round_cost_cny += cost
        executed_stages.append("image")

    audio_existing = shot.assets.get("audio")
    if not audio_existing or not Path(audio_existing).is_file():
        tts_prov = tts or _resolve_single_shot_tts(manifest)
        audio_dir = project_root / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        out_audio = synthesize_shot_audio(
            tts_prov,
            shot,
            audio_dir / f"{shot_idx:04d}.wav",
            speech_mode=getattr(manifest, "speech_mode", "auto"),
            voice_cast=getattr(manifest, "voice_cast", {}) or {},
            character_bible=manifest.character_bible,
        )
        shot.assets["audio"] = str(out_audio)
        try:
            with wave.open(str(out_audio), "rb") as source:
                shot.duration = max(1.0, source.getnframes() / source.getframerate())
        except (wave.Error, OSError):
            pass
        cost = estimate_operation_cost("tts", manifest.tts_source, manifest=manifest)
        record_cost_entry(
            manifest,
            shot_id=shot.id,
            stage="audio",
            provider_or_route=f"tts:{manifest.tts_source}",
            cost_cny=cost,
            reason=reason,
            round_index=round_index,
        )
        round_cost_cny += cost
        executed_stages.append("audio")

    route_info = resolve_shot_video_route(
        manifest,
        shot,
        has_first_frame=bool(shot.assets.get("image")),
        has_last_frame=bool(shot.assets.get("last_frame")),
        available_providers=(
            set(getattr(video_provider, "adapters_by_provider", {}).keys())
            or ({"dashscope", "comfyui"} if video_provider is not None else None)
        ),
        shot_index=shot_idx - 1,
        total_shots=max(1, len(manifest.shots)),
    )

    if route_info["route"] == "preview" or manifest.mode != "render":
        shot.assets.pop("video", None)
        shot.mode = "preview"
        shot.scene_context["_video_route"] = "preview"
        shot.scene_context["_video_model"] = "preview"
        if shot.motion == "none" and route_info.get("suggested_motion") not in (None, "none"):
            shot.motion = route_info["suggested_motion"]
    else:
        vid_existing = shot.assets.get("video")
        if not vid_existing or not Path(vid_existing).is_file():
            vid_prov = video_provider
            if vid_prov is None:
                from .project_video import build_project_video_provider

                vid_prov = build_project_video_provider(manifest, project_root)
            if vid_prov is None or route_info["route"] == "preview":
                shot.assets.pop("video", None)
                shot.mode = "preview"
            else:
                from .project_video import parse_optional_bool

                use_last = parse_optional_bool(use_last_frame, name="use_last_frame")
                # Explicit False: do not generate or require a last frame just
                # because the route label is comfy_fl2v. The on-disk asset is kept.
                if (
                    use_last is not False
                    and route_info["route"] == "comfy_fl2v"
                    and not (
                        shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file()
                    )
                ):
                    from .codex_image import ensure_shot_end_keyframe

                    end_img_prov = image_provider or _resolve_single_shot_image_provider(manifest, project_root)
                    ensure_shot_end_keyframe(
                        manifest,
                        project_root,
                        shot,
                        Path(shot.assets["image"]),
                        end_img_prov,
                        shot_index=shot_idx,
                        reason=f"{reason}:end_keyframe",
                    )
                video_dir = project_root / "video"
                video_dir.mkdir(parents=True, exist_ok=True)
                out_vid = vid_prov.generate(
                    shot,
                    Path(shot.assets["image"]),
                    video_dir / f"{shot_idx:04d}.mp4",
                    use_last_frame=use_last,
                    final_prompt_override=final_prompt_override,
                )
                shot.assets["video"] = str(out_vid)
                shot.mode = "render"
                shot.scene_context["_video_route"] = route_info["route"]
                shot.scene_context["_video_model"] = route_info.get("model") or route_info["provider"]
                cost = estimate_operation_cost("video", route_info["route"], manifest=manifest)
                record_cost_entry(
                    manifest,
                    shot_id=shot.id,
                    stage="video",
                    provider_or_route=f"video:{route_info['route']}",
                    cost_cny=cost,
                    reason=reason,
                    round_index=round_index,
                )
                round_cost_cny += cost
                executed_stages.append("video")

    shot.cache_key = _shot_cache_key(manifest, shot)
    shot.status = "generated"
    shot.error = None
    if isinstance(shot.scene_context, dict):
        shot.scene_context.pop("_relay_continuity_stale", None)
        shot.scene_context.pop("_relay_stale_upstream_shot_id", None)
    stale_downstream = mark_downstream_relay_staleness(manifest, shot.id)
    diag = refresh_shot_diagnostics(manifest, project_root, shot.id)
    return {
        "shot_id": shot.id,
        "executed_stages": executed_stages,
        "round_cost_cny": round(round_cost_cny, 4),
        "video_route": route_info["route"],
        "diagnostics": diag,
        "stale_downstream_shots": stale_downstream,
    }


def _estimate_rework_plan_cost(
    manifest: ProjectManifest,
    shot: Shot,
    *,
    strategy: str,
    video_route: str | None,
) -> float:
    total = 0.0
    if strategy in {"rewrite_image_prompt", "patch_character_dna", "reset_shot"}:
        total += estimate_operation_cost("image", manifest.image_source, manifest=manifest)
    if strategy == "reset_shot":
        total += estimate_operation_cost("tts", manifest.tts_source, manifest=manifest)
    if manifest.mode == "render" and strategy != "fallback_25d_motion" and strategy != "select_candidate":
        effective_route = validate_shot_video_route(video_route) or shot.video_route
        if not effective_route:
            route_info = resolve_shot_video_route(
                manifest,
                shot,
                has_first_frame=True,
                has_last_frame=bool(shot.assets.get("last_frame")),
            )
            effective_route = route_info["route"]
        if effective_route != "preview" and (
            strategy in {"reroute_video", "reset_shot"} or bool(shot.assets.get("video")) or shot.mode == "render"
        ):
            total += estimate_operation_cost("video", effective_route, manifest=manifest)
    return round(total, 4)


def intervene_and_rework_shot(
    project_dir: str | Path,
    shot_id: str,
    *,
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
    max_cost_cny: float | None = 5.0,
    supervisor_id: str = "gemini-4-carbon",
    lease_token: str | None = None,
    image_provider: Any = None,
    video_provider: Any = None,
    tts: Any = None,
    server_flywheel_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, bytes, str]]]:
    """Apply surgical supervisor intervention, guard budget, run single-shot rework, and log flywheel case."""
    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token)
    clean_strategy = (strategy or "").strip()
    if clean_strategy not in _VALID_STRATEGIES:
        raise ValueError(f"unsupported intervention strategy: {strategy!r}")

    if motion is not None and motion not in MOTIONS:
        raise ValueError(f"unsupported motion: {motion!r}; expected one of {MOTIONS}")

    from .project_video import parse_optional_bool

    use_last = parse_optional_bool(use_last_frame, name="use_last_frame")
    frozen_prompt = (
        final_prompt_override.strip()
        if isinstance(final_prompt_override, str) and final_prompt_override.strip()
        else None
    )

    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)

    policy = manifest.supervisor_policy if isinstance(manifest.supervisor_policy, dict) else {}
    max_rounds = int(policy.get("max_rework_rounds", 3))
    if len(shot.rework_history) >= max_rounds:
        raise ValueError(
            f"shot {shot.id} reached max_rework_rounds ({max_rounds}); "
            "raise supervisor_policy.max_rework_rounds or use inject_external_asset / manual review"
        )

    estimated_cost = _estimate_rework_plan_cost(
        manifest,
        shot,
        strategy=clean_strategy,
        video_route=video_route,
    )
    if execute_now:
        check_rework_budget(manifest, estimated_cost, max_cost_cny=max_cost_cny)

    ctx_before = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    report_before = shot.quality_report if isinstance(shot.quality_report, dict) else {}
    before_snapshot = {
        "before_score": report_before.get("overall_score"),
        "before_prompt": shot.prompt_override or ctx_before.get("visual_action") or shot.text,
        "before_negative": shot.negative_override or ctx_before.get("negative") or "",
        "before_route": ctx_before.get("_video_route") or shot.video_route or manifest.video_routing_policy,
        "before_seed": shot.seed_override,
    }

    if clean_strategy == "select_candidate":
        if not candidate_id:
            raise ValueError("select_candidate strategy requires candidate_id")
        select_candidate(manifest, shot.id, candidate_id, role=frame_role)
    elif clean_strategy == "patch_character_dna":
        if not character_name or not isinstance(character_dna_patch, dict):
            raise ValueError("patch_character_dna requires character_name and character_dna_patch")
        upsert_character_profile(
            manifest,
            character_name,
            appearance=character_dna_patch.get("appearance"),
            clothing=character_dna_patch.get("clothing"),
            visual_dna=character_dna_patch.get("visual_dna"),
            seed=character_dna_patch.get("seed"),
            gender=character_dna_patch.get("gender"),
            role=character_dna_patch.get("role"),
        )
        if prompt_override is not None:
            shot.prompt_override = prompt_override.strip() or None
        if negative_override is not None:
            shot.negative_override = negative_override.strip() or None
        if seed_override is not None:
            shot.seed_override = int(seed_override)
        shot.assets.pop("image", None)
        shot.assets.pop("video", None)
        shot.image_task.clear()
        shot.mode = "preview"
    elif clean_strategy == "rewrite_image_prompt":
        if prompt_override is not None:
            shot.prompt_override = prompt_override.strip() or None
        if negative_override is not None:
            shot.negative_override = negative_override.strip() or None
        if seed_override is not None:
            shot.seed_override = int(seed_override)
        shot.assets.pop("image", None)
        shot.assets.pop("video", None)
        shot.image_task.clear()
        shot.mode = "preview"
    elif clean_strategy == "reroute_video":
        next_route = validate_shot_video_route(video_route)
        if next_route is not None:
            shot.video_route = next_route
        if motion is not None:
            shot.motion = motion
        if prompt_override is not None:
            shot.prompt_override = prompt_override.strip() or None
        if negative_override is not None:
            shot.negative_override = negative_override.strip() or None
        if seed_override is not None:
            shot.seed_override = int(seed_override)
        shot.assets.pop("video", None)
        shot.mode = "preview"
    elif clean_strategy == "fallback_25d_motion":
        shot.video_route = "preview"
        shot.motion = motion or ("ken_burns" if shot.motion == "none" else shot.motion)
        shot.assets.pop("video", None)
        shot.mode = "preview"
    elif clean_strategy == "reset_shot":
        shot.assets.clear()
        shot.image_task.clear()
        shot.cache_key = None
        shot.mode = "preview"
        shot.prompt_override = prompt_override.strip() if isinstance(prompt_override, str) and prompt_override.strip() else None
        shot.negative_override = negative_override.strip() if isinstance(negative_override, str) and negative_override.strip() else None
        shot.seed_override = int(seed_override) if seed_override is not None else None

    manifest.approvals.pop("assets", None)
    manifest.output = None
    shot.error = None

    round_index = len(shot.rework_history) + 1
    inferred_stage = (
        "video"
        if clean_strategy in {"reroute_video", "fallback_25d_motion"}
        else "image"
    )
    fly_dir = resolve_flywheel_dir(manifest, root, server_default_dir=server_flywheel_dir)
    flywheel_case = None
    if fly_dir is not None:
        intervention_payload: dict[str, Any] = {
            "stage": inferred_stage,
            "prompt_override": shot.prompt_override,
            "negative_override": shot.negative_override,
            "action_guidance": action_guidance,
            "trigger_keywords": trigger_keywords,
            "character_name": character_name,
            "character_dna_patch": character_dna_patch,
            "video_route": shot.video_route if clean_strategy in {"reroute_video", "fallback_25d_motion"} else None,
            "motion": motion or (shot.motion if clean_strategy == "fallback_25d_motion" else None),
            "seed_override": shot.seed_override,
        }
        flywheel_case = record_intervention_case(
            fly_dir,
            manifest=manifest,
            shot=shot,
            failure_code=failure_code or (shot.quality_report or {}).get("failure_code") or "OTHER",
            failure_reason=failure_reason or (shot.quality_report or {}).get("reason") or "",
            strategy=clean_strategy,
            intervention=intervention_payload,
            supervisor_id=supervisor_id,
            before_snapshot=before_snapshot,
        )

    execution_summary = None
    if execute_now and clean_strategy != "select_candidate":
        execution_summary = rework_single_shot(
            manifest,
            root,
            shot.id,
            image_provider=image_provider,
            video_provider=video_provider,
            tts=tts,
            round_index=round_index,
            reason=f"supervisor:{clean_strategy}",
            use_last_frame=use_last,
            final_prompt_override=frozen_prompt,
        )
    else:
        shot.cache_key = _shot_cache_key(manifest, shot)
        refresh_shot_diagnostics(manifest, root, shot.id)

    history_entry = {
        "round_index": round_index,
        "strategy": clean_strategy,
        "supervisor_id": supervisor_id,
        "estimated_cost_cny": estimated_cost,
        "actual_cost_cny": execution_summary["round_cost_cny"] if execution_summary else 0.0,
        "flywheel_case_id": flywheel_case["case_id"] if flywheel_case else None,
        "executed": bool(execute_now),
        "recorded_at": _utcnow().isoformat(),
    }
    if use_last is not None:
        history_entry["use_last_frame"] = use_last
    if frozen_prompt is not None:
        history_entry["final_prompt_override"] = True
    shot.rework_history.append(history_entry)

    manifest.save(manifest_path)
    visuals = _collect_shot_visuals(shot)
    return (
        {
            "shot_id": shot.id,
            "round_index": round_index,
            "strategy": clean_strategy,
            "estimated_cost_cny": estimated_cost,
            "execution": execution_summary,
            "flywheel_case_id": flywheel_case["case_id"] if flywheel_case else None,
            "cost_ledger": summarize_cost_ledger(manifest),
            "attached_visuals": [label for label, _, _ in visuals],
            "stale_downstream_shots": (execution_summary or {}).get("stale_downstream_shots", []),
            "use_last_frame": use_last,
            "final_prompt_override": bool(frozen_prompt),
        },
        visuals,
    )


def inject_external_asset(
    project_dir: str | Path,
    shot_id: str,
    *,
    asset_role: str,
    source_path: str | Path | None = None,
    content_base64: str | None = None,
    provenance: str = "external-agent",
    lease_token: str | None = None,
) -> dict[str, Any]:
    """Allow an external supervisor or specialized agent to inject an asset directly into a shot."""
    role = (asset_role or "").strip()
    if role not in _ROLE_TO_DIR_EXT:
        raise ValueError(f"unsupported asset_role {asset_role!r}; expected one of {tuple(_ROLE_TO_DIR_EXT)}")
    if bool(source_path) == bool(content_base64):
        raise ValueError("provide exactly one of source_path or content_base64")

    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token, allowed_roles=("asset_producer",))
    manifest = ProjectManifest.load(manifest_path)
    shot_idx, shot = next(
        ((i, item) for i, item in enumerate(manifest.shots, 1) if item.id == shot_id),
        (None, None),
    )
    if shot_idx is None or shot is None:
        raise KeyError(shot_id)

    if source_path is not None:
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(f"source_path not found: {source_path}")
        raw_bytes = src.read_bytes()
    else:
        raw_bytes = base64.b64decode(content_base64 or "", validate=True)
        if not raw_bytes:
            raise ValueError("decoded asset bytes must not be empty")

    registry = ArtifactRegistry(root / "artifacts")
    artifact_ref = registry.put_bytes(raw_bytes)

    subdir, template = _ROLE_TO_DIR_EXT[role]
    target = root / subdir / template.format(idx=shot_idx)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw_bytes)

    shot.assets[role] = str(target)
    if role == "image":
        shot.assets.pop("video", None)
        shot.mode = "preview"
    elif role == "video":
        registry.video_preview(artifact_ref)
        shot.mode = "render"
        try:
            from .local_media import find_ffmpeg

            ffmpeg = find_ffmpeg()
            vid_dur = probe_duration(ffmpeg, target)
            if vid_dur is not None:
                shot.scene_context["_video_duration"] = vid_dur
                shot.duration = resolve_elastic_shot_duration(
                    shot.duration,
                    vid_dur,
                    tts_source=manifest.tts_source,
                    alignment_mode=manifest.alignment_mode,
                )
        except Exception:
            pass
    elif role == "audio":
        try:
            with wave.open(str(target), "rb") as source:
                shot.duration = max(1.0, source.getnframes() / source.getframerate())
        except (wave.Error, OSError):
            pass

    shot.scene_context[f"_external_{role}_provenance"] = {
        "provenance": provenance,
        "artifact_ref": artifact_ref,
        "text": shot.text,
        "injected_at": _utcnow().isoformat(),
    }
    shot.status = "generated"
    shot.error = None
    if role in {"image", "last_frame", "video"} and isinstance(shot.scene_context, dict):
        shot.scene_context.pop("_relay_continuity_stale", None)
        shot.scene_context.pop("_relay_stale_upstream_shot_id", None)
    stale_downstream = (
        mark_downstream_relay_staleness(manifest, shot.id)
        if role in {"image", "last_frame", "video"}
        else []
    )
    shot.cache_key = _shot_cache_key(manifest, shot)
    manifest.approvals.pop("assets", None)
    manifest.output = None

    diag = refresh_shot_diagnostics(manifest, root, shot.id)
    manifest.save(manifest_path)
    return {
        "shot_id": shot.id,
        "asset_role": role,
        "path": str(target),
        "artifact_ref": artifact_ref,
        "provenance": provenance,
        "diagnostics": diag,
        "stale_downstream_shots": stale_downstream,
    }


def run_pipeline_stage(
    project_dir: str | Path,
    *,
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
    """Execute one or all pipeline stages under MCP supervisor governance and lease checks."""
    clean_stage = str(stage or "all").strip().lower()
    if clean_stage not in {"all", "keyframes", "tts", "video", "assemble"}:
        raise ValueError(f"unsupported target stage: {stage!r}")

    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token)
    manifest = ProjectManifest.load(manifest_path)

    if mode is not None:
        clean_mode = str(mode).strip().lower()
        if clean_mode not in {"preview", "render"}:
            raise ValueError(f"unsupported mode: {mode!r}")
        manifest.mode = clean_mode
    if execution_strategy is not None:
        clean_strategy = str(execution_strategy).strip().lower()
        if clean_strategy not in {"interleaved", "stage_batched"}:
            raise ValueError(f"unsupported execution_strategy: {execution_strategy!r}")
        manifest.execution_strategy = clean_strategy
    if image_source is not None:
        clean_img = str(image_source).strip().lower()
        if clean_img not in {"local", "dashscope", "codex"}:
            raise ValueError(f"unsupported image_source: {image_source!r}")
        manifest.image_source = clean_img
    if tts_source is not None:
        clean_tts = str(tts_source).strip().lower()
        if clean_tts not in {"silent", "windows", "cosyvoice", "edge"}:
            raise ValueError(f"unsupported tts_source: {tts_source!r}")
        if clean_tts != manifest.tts_source:
            manifest.tts_source = clean_tts
            for s in manifest.shots:
                s.assets.pop("audio", None)
    if video_provider is not None or video_routing_policy is not None:
        from .video_router import validate_routing_settings

        next_vp, next_pol, _ = validate_routing_settings(
            video_provider or manifest.video_provider,
            video_routing_policy or manifest.video_routing_policy,
            manifest.video_fallback_provider,
        )
        if next_vp != manifest.video_provider or next_pol != manifest.video_routing_policy:
            manifest.video_provider = next_vp
            manifest.video_routing_policy = next_pol
            for s in manifest.shots:
                s.assets.pop("video", None)
                s.mode = "preview"
                s.error = None
    if lipsync_provider is not None:
        clean_lip = str(lipsync_provider).strip().lower()
        if clean_lip not in {"off", "http", "comfyui", "command"}:
            raise ValueError(f"unsupported lipsync_provider: {lipsync_provider!r}")
        manifest.lipsync_provider = clean_lip
    if auto_split_shots is not None:
        manifest.auto_split_shots = bool(auto_split_shots)
    if max_shot_seconds is not None:
        if float(max_shot_seconds) <= 1.0:
            raise ValueError("max_shot_seconds must be > 1.0")
        manifest.max_shot_seconds = float(max_shot_seconds)
    if relay_tail_frames is not None:
        manifest.relay_tail_frames = bool(relay_tail_frames)
    if auto_rework is not None:
        manifest.auto_rework = bool(auto_rework)
    if budget_limit_cny is not None:
        if float(budget_limit_cny) <= 0:
            raise ValueError("budget_limit_cny must be positive")
        manifest.budget_limit_cny = float(budget_limit_cny)

    if approve_gates and manifest.status in {"awaiting_storyboard_review", "awaiting_asset_review"}:
        approve(manifest)
    enforce_review = False if approve_gates else manifest.review_enforced
    if approve_gates and manifest.review_enforced:
        manifest.approvals["storyboard"] = fingerprint(manifest, "storyboard")

    video_provider_obj = None
    if manifest.mode == "render" and clean_stage in {"all", "video"}:
        from .project_video import build_project_video_provider, needs_video_generation

        if needs_video_generation(manifest):
            video_provider_obj = build_project_video_provider(manifest, root)

    output = render_preview(
        manifest,
        root,
        video_provider=video_provider_obj,
        enforce_review=enforce_review,
        target_stage=clean_stage,
    )
    if approve_gates and manifest.review_enforced and output:
        manifest.approvals["assets"] = fingerprint(manifest, "assets")

    manifest.save(manifest_path)
    return {
        "schema": "pipeline-stage-run/v1",
        "project_id": manifest.project_id,
        "stage": clean_stage,
        "execution_strategy": manifest.execution_strategy,
        "status": manifest.status,
        "output": str(output) if output else None,
        "shots_total": len(manifest.shots),
        "keyframes_ready": sum(1 for s in manifest.shots if (s.assets or {}).get("image")),
        "audio_ready": sum(1 for s in manifest.shots if (s.assets or {}).get("audio")),
        "videos_ready": sum(1 for s in manifest.shots if (s.assets or {}).get("video")),
        "cost_ledger": summarize_cost_ledger(manifest),
    }


def finalize_pipeline(
    project_dir: str | Path,
    *,
    approve_gates: bool = True,
    export_jianying: bool = True,
    lease_token: str | None = None,
) -> dict[str, Any]:
    """Approve pending gates, assemble final timeline without re-billing, and export Jianying draft."""
    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token)
    manifest = ProjectManifest.load(manifest_path)

    from .assembly_quality_gate import assert_no_failed_shots_for_assembly

    assert_no_failed_shots_for_assembly(manifest, output_kind="final")

    if approve_gates and manifest.status in {"awaiting_storyboard_review", "awaiting_asset_review"}:
        approve(manifest)

    enforce_review = False if approve_gates else manifest.review_enforced
    if approve_gates and manifest.review_enforced:
        manifest.approvals["storyboard"] = fingerprint(manifest, "storyboard")

    assets_ready = bool(manifest.shots) and all(
        s.assets.get("image")
        and Path(str(s.assets["image"])).is_file()
        and s.assets.get("audio")
        and Path(str(s.assets["audio"])).is_file()
        for s in manifest.shots
    )
    if assets_ready:
        for s in manifest.shots:
            s.cache_key = _shot_cache_key(manifest, s)
            if s.status not in {"generated", "done"}:
                s.status = "generated"
        output = render_preview(manifest, root, enforce_review=enforce_review, target_stage="assemble")
    else:
        output = render_preview(manifest, root, enforce_review=enforce_review)
    if approve_gates and manifest.review_enforced:
        manifest.approvals["assets"] = fingerprint(manifest, "assets")

    draft_dir = None
    if export_jianying and output:
        from .jianying import export_jianying_draft

        draft_dir = str(export_jianying_draft(manifest, root))

    manifest.save(manifest_path)
    return {
        "project_id": manifest.project_id,
        "status": manifest.status,
        "output": str(output) if output else None,
        "jianying_draft": draft_dir or manifest.jianying_draft,
        "timeline": manifest.timeline,
        "cost_ledger": summarize_cost_ledger(manifest),
    }


def list_takeover_queue(
    project_dir: str | Path,
    *,
    filter_status: str = "needs_attention",
    server_flywheel_dir: str | Path | None = None,
) -> dict[str, Any]:
    """List shots requiring supervisor attention (failed, warning, needs_takeover, or all)."""
    root, manifest_path = _manifest_path(project_dir)
    clean_filter = (filter_status or "needs_attention").strip().lower()
    if clean_filter not in {"needs_attention", "failed", "warning", "needs_takeover", "all"}:
        raise ValueError(
            f"unsupported filter_status: {filter_status!r}; expected one of needs_attention, failed, warning, needs_takeover, all"
        )
    manifest = ProjectManifest.load(manifest_path)
    items: list[dict[str, Any]] = []
    for shot in manifest.shots:
        report = shot.quality_report if isinstance(shot.quality_report, dict) else {}
        vid_diag = (shot.diagnostics or {}).get("video") if isinstance(shot.diagnostics, dict) else {}
        decision = str(report.get("decision") or "").lower()
        ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
        is_stale_relay = bool(ctx.get("_relay_continuity_stale"))
        is_failed = shot.status == "failed" or decision == "failed"
        is_warning = (
            decision == "warning"
            or is_stale_relay
            or bool(
                isinstance(vid_diag, dict)
                and (vid_diag.get("is_static_suspect") or vid_diag.get("is_clipping_suspect"))
            )
        )
        is_takeover = getattr(shot, "review", None) == "needs_takeover"
        matched = (
            clean_filter == "all"
            or (clean_filter == "failed" and is_failed)
            or (clean_filter == "warning" and is_warning)
            or (clean_filter == "needs_takeover" and is_takeover)
            or (clean_filter == "needs_attention" and (is_failed or is_warning or is_takeover))
        )
        if not matched:
            continue
        items.append(
            {
                "shot_id": shot.id,
                "scene_id": shot.scene_id,
                "status": shot.status,
                "review": shot.review,
                "decision": report.get("decision"),
                "failure_code": report.get("failure_code"),
                "reason": report.get("reason"),
                "error": shot.error,
                "relay_continuity_stale": is_stale_relay,
                "relay_stale_upstream_shot_id": ctx.get("_relay_stale_upstream_shot_id"),
                "rework_rounds": len(shot.rework_history),
                "contact_sheet_ref": vid_diag.get("contact_sheet_ref") if isinstance(vid_diag, dict) else None,
                "contact_sheet_path": vid_diag.get("contact_sheet_path") if isinstance(vid_diag, dict) else None,
            }
        )
    return {
        "schema": "takeover-queue/v1",
        "project_id": manifest.project_id,
        "project_dir": str(root),
        "project_status": manifest.status,
        "filter": clean_filter,
        "total_shots": len(manifest.shots),
        "queue_count": len(items),
        "items": items,
        "cost_ledger": summarize_cost_ledger(manifest),
    }


def get_shot_contact_sheet_bytes(
    project_dir: str | Path,
    shot_id: str,
    *,
    refresh_if_missing: bool = True,
) -> bytes:
    """Return JPEG bytes for a shot's 2x2 diagnostic contact sheet (or fallback keyframe image)."""
    root, manifest_path = _manifest_path(project_dir)
    manifest = ProjectManifest.load(manifest_path)
    shot = manifest.shot(shot_id)
    vid_diag = (shot.diagnostics or {}).get("video") if isinstance(shot.diagnostics, dict) else {}
    sheet_path = vid_diag.get("contact_sheet_path") if isinstance(vid_diag, dict) else None

    if shot.assets.get("video") and refresh_if_missing and (not sheet_path or not Path(sheet_path).is_file()):
        diag = refresh_shot_diagnostics(manifest, root, shot_id)
        manifest.save(manifest_path)
        sheet_path = (diag.get("video") or {}).get("contact_sheet_path")

    if sheet_path and Path(sheet_path).is_file():
        return Path(sheet_path).read_bytes()

    img_path = shot.assets.get("image")
    if img_path and Path(img_path).is_file() and Path(img_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        from PIL import Image

        with Image.open(img_path) as img:
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=88)
            return buf.getvalue()

    raise FileNotFoundError(f"no visual diagnostic or image available for shot {shot_id}")


def get_keyframe_generation_tasks(
    project_dir: str | Path,
    *,
    include_generated: bool = False,
) -> dict[str, Any]:
    """Return structured keyframe prompts and ref paths for free Codex Chat MCP generation."""
    from .codex_image import build_codex_keyframe_spec

    root, manifest_path = _manifest_path(project_dir)
    manifest = ProjectManifest.load(manifest_path)
    if getattr(manifest, "consistency_mode", "auto") != "off":
        sync_character_bible(manifest, project_root=root if (root / "flywheel").is_dir() else None)

    total = max(1, len(manifest.shots))
    avail_providers = {"comfyui"} if getattr(manifest, "video_provider", "comfyui") == "comfyui" else None
    raster_exts = {".png", ".jpg", ".jpeg", ".webp"}
    tasks: list[dict[str, Any]] = []

    for idx, shot in enumerate(manifest.shots, 1):
        route_info = resolve_shot_video_route(
            manifest,
            shot,
            has_first_frame=True,
            has_last_frame=bool((shot.assets or {}).get("last_frame")),
            available_providers=avail_providers,
            shot_index=idx - 1,
            total_shots=total,
        )
        route_name = str(route_info.get("route") or "")
        motion_level = str((route_info.get("motion_analysis") or {}).get("level") or "medium")
        requires_last = route_name in {"comfy_fl2v", "kf2v"} or motion_level == "high"

        first_existing = (shot.assets or {}).get("image")
        first_exists = bool(
            isinstance(first_existing, str)
            and first_existing.strip()
            and Path(first_existing).is_file()
            and Path(first_existing).suffix.lower() in raster_exts
        )
        recommended_first = str(root / "frames" / f"{idx:04d}.png")
        if include_generated or not first_exists:
            spec_first = build_codex_keyframe_spec(manifest, shot, role="first", project_root=root)
            tasks.append(
                {
                    "shot_index": idx,
                    "asset_role": "image",
                    "frame_role": "first",
                    "already_exists": first_exists,
                    "existing_path": str(first_existing) if first_exists else None,
                    "recommended_output_path": recommended_first,
                    "video_route": route_name,
                    "motion_level": motion_level,
                    "inject_tool": "story_inject_external_asset",
                    **spec_first,
                }
            )

        if requires_last:
            last_existing = (shot.assets or {}).get("last_frame")
            last_exists = bool(
                isinstance(last_existing, str)
                and last_existing.strip()
                and Path(last_existing).is_file()
                and Path(last_existing).suffix.lower() in raster_exts
            )
            if include_generated or not last_exists:
                ref_first = str(first_existing) if first_exists else None
                spec_last = build_codex_keyframe_spec(
                    manifest,
                    shot,
                    role="last",
                    first_frame_path=ref_first,
                    project_root=root,
                )
                tasks.append(
                    {
                        "shot_index": idx,
                        "asset_role": "last_frame",
                        "frame_role": "last",
                        "already_exists": last_exists,
                        "existing_path": str(last_existing) if last_exists else None,
                        "recommended_output_path": str(root / "frames" / f"{idx:04d}_last.png"),
                        "video_route": route_name,
                        "motion_level": motion_level,
                        "depends_on_shot_asset": None if first_exists else f"{shot.id}:image",
                        "inject_tool": "story_inject_external_asset",
                        **spec_last,
                    }
                )

    return {
        "schema": "keyframe-tasks/v1",
        "project_id": manifest.project_id,
        "project_dir": str(root),
        "aspect_ratio": manifest.aspect_ratio,
        "image_size": manifest.image_size or default_image_size_for_aspect(manifest.aspect_ratio),
        "image_source": manifest.image_source,
        "video_provider": manifest.video_provider,
        "video_routing_policy": manifest.video_routing_policy,
        "total_tasks": len(tasks),
        "pending_tasks": sum(1 for t in tasks if not t["already_exists"]),
        "tasks": tasks,
        "workflow_hint": (
            "Draw each pending `image` (and `last_frame` for comfy_fl2v/high-motion shots) using Codex built-in `image_gen`, "
            "attaching listed `reference_images`, then call `story_inject_external_asset` with `asset_role` (`image` or `last_frame`). "
            "Generate `image` before `last_frame` on the same shot so `last_frame` can use `image` as its visual lock reference."
        ),
    }


def cast_character_3d_from_photo(
    project_dir: str | Path,
    character_name: str,
    *,
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
    """Cast a de-identified 3D bone-structure anime character master from a real photo or bind Codex output."""
    from .character_bible import verify_portrait_deidentification
    from .codex_image import _is_valid_raster_image

    root, manifest_path = _manifest_path(project_dir)
    verify_takeover(root, lease_token)
    manifest = ProjectManifest.load(manifest_path)
    sync_character_bible(manifest, project_root=root if (root / "flywheel").is_dir() else None)

    profile = upsert_character_profile(
        manifest,
        character_name,
        appearance=appearance,
        clothing=clothing,
        visual_dna=visual_dna,
    )
    if not getattr(manifest, "style_preset", None):
        manifest.style_preset = "stylized_3d_anime"

    key = str(profile.get("key") or character_name)
    safe_file = safe_character_filename(key)
    portrait_dir = root / "artifacts" / "characters"
    portrait_dir.mkdir(parents=True, exist_ok=True)
    target_path = portrait_dir / f"{safe_file}.png"

    photo_path: Path | None = None
    if isinstance(source_photo_path, str) and source_photo_path.strip():
        photo_path = Path(source_photo_path).expanduser().resolve()
        if not photo_path.is_file():
            raise FileNotFoundError(f"source portrait photo not found: {source_photo_path}")

    if isinstance(stylized_portrait_path, str) and stylized_portrait_path.strip():
        stylized_src = Path(stylized_portrait_path).expanduser().resolve()
        if not stylized_src.is_file():
            raise FileNotFoundError(f"stylized portrait not found: {stylized_portrait_path}")
        raw_bytes = stylized_src.read_bytes()
        if not _is_valid_raster_image(raw_bytes):
            raise ValueError("stylized_portrait_path must point to a valid PNG/JPEG/WebP image")
        biometric_audit = None
        if photo_path is not None:
            biometric_audit = verify_portrait_deidentification(
                photo_path,
                stylized_src,
                max_similarity=max_identity_similarity,
            )
            profile["biometric_audit"] = biometric_audit
        tmp = target_path.with_suffix(".png.tmp")
        tmp.write_bytes(raw_bytes)
        tmp.replace(target_path)
        ArtifactRegistry(root / "artifacts").put_bytes(raw_bytes)
        profile["ref_image"] = str(target_path)
        profile["anchor_shot_id"] = f"char-{safe_file}"
        profile["stylized_from_photo"] = True
        if photo_path is not None:
            profile["source_photo_sha256"] = hashlib.sha256(photo_path.read_bytes()).hexdigest()[:16]
        if isinstance(deidentify_traits, str) and deidentify_traits.strip():
            profile["deidentify_traits"] = deidentify_traits.strip()
        invalidated = invalidate_character_shots(manifest, character_name)
        manifest.save(manifest_path)
        return {
            "status": "bound",
            "character_key": key,
            "profile": profile,
            "invalidated_shots": invalidated,
            "ref_image": str(target_path),
            "style_preset": manifest.style_preset,
            "biometric_audit": biometric_audit,
        }

    if execute_now:
        profile = generate_character_portrait(
            manifest,
            root,
            character_name,
            from_photo=photo_path,
            deidentify_traits=deidentify_traits,
            three_d_contour=True,
            max_identity_similarity=max_identity_similarity,
        )
        invalidated = invalidate_character_shots(manifest, character_name)
        manifest.save(manifest_path)
        return {
            "status": "generated",
            "character_key": key,
            "profile": profile,
            "invalidated_shots": invalidated,
            "ref_image": profile.get("ref_image"),
            "style_preset": manifest.style_preset,
            "biometric_audit": profile.get("biometric_audit"),
        }

    spec = build_photo_to_3d_anime_prompt(
        profile,
        deidentify_traits=deidentify_traits,
        style_preset=manifest.style_preset,
    )
    manifest.save(manifest_path)
    return {
        "status": "ready_for_codex_image_gen",
        "character_key": key,
        "profile": profile,
        "prompt": spec["prompt"],
        "negative_prompt": spec["negative"],
        "lighting": spec["lighting"],
        "framing": spec["framing"],
        "reference_images": [str(photo_path)] if photo_path is not None else [],
        "recommended_output_path": str(target_path),
        "style_preset": manifest.style_preset,
        "bind_hint": (
            "Call Codex built-in `image_gen` with `reference_images` and `prompt`, "
            "then call `story_cast_character_3d_from_photo` with `stylized_portrait_path` pointing to the generated image."
        ),
    }


def doctor_check(
    project_dir: str | Path | None = None,
    *,
    probe_network: bool = False,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Run local toolchain, sibling repository, closed-source git lock, and H3/ComfyUI preflight diagnostics."""
    from .doctor import run_preflight_doctor

    return run_preflight_doctor(project_dir, probe_network=probe_network, timeout=timeout)


def plan_parent_segment_group(project_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    from .parent_segment_group import plan_parent_segment_group as _plan

    return _plan(project_dir, plan)


def dryrun_parent_segment_group(project_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    from .parent_segment_group import dryrun_parent_segment_group as _dryrun

    return _dryrun(project_dir, plan)


def execute_parent_segment_group(
    project_dir: str | Path,
    plan: dict[str, Any],
    *,
    execute_now: bool | None = None,
    generation_authorized: bool | None = None,
    video_provider: Any = None,
    lease_token: str | None = None,
    supervisor_id: str = "grok-bot-t50u",
) -> dict[str, Any]:
    from .parent_segment_group import execute_parent_segment_group as _execute

    return _execute(
        project_dir,
        plan,
        execute_now=execute_now,
        generation_authorized=generation_authorized,
        video_provider=video_provider,
        lease_token=lease_token,
        supervisor_id=supervisor_id,
    )


def assemble_parent_segment_group(
    project_dir: str | Path,
    plan: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    from .parent_segment_group import assemble_parent_segment_group as _assemble

    return _assemble(project_dir, plan, **kwargs)
