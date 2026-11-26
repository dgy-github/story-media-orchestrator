"""Dependency-light preview renderer; uses ffmpeg when installed."""
from __future__ import annotations
import json
import hashlib, shutil, subprocess, wave
from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from .manifest import ProjectManifest
from .tts import FakeTTSProvider, TTSProvider, synthesize_shot_audio
from .providers import ImageProvider, TextFrameProvider, VideoProvider
from .cost_ledger import (
    BudgetExceededError,
    check_rework_budget,
    estimate_operation_cost,
    record_cost_entry,
)
try:
    from .codex_image import CodexQuotaExceededError
except ImportError:  # Open-core build without optional codex_image adapter
    class CodexQuotaExceededError(RuntimeError):  # type: ignore[no-redef]
        """Fallback exception when optional codex_image module is absent."""
from .editing import (
    XFADE_TRANSITIONS,
    build_subtitle_cues,
    compose_xfade_segments,
    format_ass,
    format_srt,
    mix_audio_with_bgm,
    probe_duration,
    resolve_canvas_size,
    resolve_elastic_shot_duration,
    resolve_safe_shot_transition,
    validate_edit_settings,
)

def _cache_key(shot, mode: str) -> str:
    public_context = {k: v for k, v in shot.scene_context.items() if not k.startswith("_")} if shot.scene_context else {}
    context = "\0" + json.dumps(public_context, sort_keys=True, ensure_ascii=False) if public_context else ""
    return hashlib.sha256(f"v1\0{mode}\0{shot.text}{context}".encode()).hexdigest()


def _concat_file(path: str) -> str:
    escaped = Path(path).resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


def _shot_cache_key(manifest, shot, mode="preview"):
    aspect_ratio = getattr(manifest, "aspect_ratio", "16:9")
    if manifest.image_source in {"dashscope", "codex"}:
        profile = f":{manifest.image_source}:{manifest.image_model}:{manifest.image_size}:v2"
    elif manifest.real_preview:
        profile = ":story-card-v1" + (f":ar:{aspect_ratio}" if aspect_ratio != "16:9" else "")
    else:
        profile = ""
    from .character_bible import shot_consistency_fingerprint
    cc_fp = shot_consistency_fingerprint(manifest, shot)
    if cc_fp:
        profile += f":cc:{cc_fp}"
    return _cache_key(shot, mode + profile)


def render_preview(manifest: ProjectManifest, root: str | Path, tts: TTSProvider | None = None,
                   image_provider: ImageProvider | None = None,
                   video_provider: VideoProvider | None = None,
                   *, max_attempts: int = 2, real_preview: bool | None = None,
                   enforce_review: bool = False,
                   critic_evaluator: object | None = None,
                   target_stage: str = "all") -> Path | None:
    clean_stage = str(target_stage or "all").strip().lower()
    if clean_stage not in {"all", "keyframes", "tts", "video", "assemble"}:
        raise ValueError(f"unsupported target_stage: {target_stage!r}")
    execution_strategy = str(getattr(manifest, "execution_strategy", "interleaved") or "interleaved").strip().lower()
    if execution_strategy not in {"interleaved", "stage_batched"}:
        raise ValueError(f"unsupported execution_strategy: {execution_strategy!r}")
    do_keyframes = clean_stage in {"all", "keyframes"}
    do_tts = clean_stage in {"all", "tts"}
    do_video = clean_stage in {"all", "video"}
    do_assemble = clean_stage in {"all", "assemble"}

    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    aspect_ratio = getattr(manifest, "aspect_ratio", "16:9")
    scale_mode = getattr(manifest, "scale_mode", "contain")
    alignment_mode = getattr(manifest, "alignment_mode", "pad")
    subtitle_style = getattr(manifest, "subtitle_style", "soft")
    speech_mode = getattr(manifest, "speech_mode", "auto")
    voice_cast = getattr(manifest, "voice_cast", {}) or {}
    bgm_source = getattr(manifest, "bgm_source", None)
    bgm_volume = float(getattr(manifest, "bgm_volume", 0.22))
    bgm_ducking = bool(getattr(manifest, "bgm_ducking", True))
    validate_edit_settings(
        aspect_ratio=aspect_ratio,
        scale_mode=scale_mode,
        alignment_mode=alignment_mode,
        subtitle_style=subtitle_style,
        speech_mode=speech_mode,
        bgm_volume=bgm_volume,
    )
    from .character_bible import sync_character_bible, validate_consistency_settings
    consistency_mode, _ = validate_consistency_settings(
        getattr(manifest, "consistency_mode", "auto"),
        getattr(manifest, "style_preset", None),
    )
    if consistency_mode != "off":
        sync_character_bible(manifest, project_root=root if (root / "flywheel").is_dir() else None)
    if getattr(manifest, "auto_split_shots", False):
        from .lipsync import split_long_dialogue_shots

        split_long_dialogue_shots(
            manifest,
            max_shot_seconds=float(getattr(manifest, "max_shot_seconds", 4.8) or 4.8),
        )
    from .review_gate import block_unapproved
    if enforce_review and block_unapproved(manifest, root, 'storyboard'):
        return None
    if real_preview is not None:
        manifest.real_preview = real_preview
    if manifest.image_source in {"dashscope", "codex"}:
        manifest.real_preview = True
    real_preview = manifest.real_preview
    # Older render projects included composition mode in the image/audio key.
    # Migrate only an exact match; content/model changes still invalidate assets.
    for shot in manifest.shots:
        if shot.cache_key == _shot_cache_key(manifest, shot, "render"):
            shot.cache_key = _shot_cache_key(manifest, shot)
    manifest.output = None
    manifest.error = None
    frames = root / "frames"; frames.mkdir(exist_ok=True)
    audio = root / "audio"; audio.mkdir(exist_ok=True)
    try:
        if manifest.image_source in {"dashscope", "codex"} and image_provider is None and (do_keyframes or do_video):
            needs_image = do_keyframes and any(
                s.cache_key != _shot_cache_key(manifest, s)
                or not Path(s.assets.get("image", "")).is_file()
                for s in manifest.shots
            )
            needs_end_frame = False
            if manifest.mode == "render" and (video_provider is not None or execution_strategy == "stage_batched" or do_keyframes):
                from .video_router import resolve_shot_video_route

                avail_vid = (
                    set(getattr(video_provider, "adapters_by_provider", {}).keys())
                    or {"dashscope", "comfyui"}
                )
                needs_end_frame = bool(getattr(manifest, "auto_rework", False)) or any(
                    resolve_shot_video_route(
                        manifest,
                        s,
                        has_first_frame=True,
                        has_last_frame=bool(s.assets.get("last_frame")),
                        available_providers=avail_vid,
                        shot_index=idx,
                        total_shots=len(manifest.shots),
                    ).get("route")
                    == "comfy_fl2v"
                    and not Path(s.assets.get("last_frame", "")).is_file()
                    for idx, s in enumerate(manifest.shots)
                )
            if needs_image or needs_end_frame:
                from .providers import build_project_image_provider
                image_provider = build_project_image_provider(manifest, root)
        elif manifest.image_source not in {"local", "dashscope", "codex"}:
            raise ValueError("unsupported image source")
        if real_preview and do_keyframes:
            from .local_media import StoryCardProvider
            image_provider = image_provider or StoryCardProvider(size=resolve_canvas_size(aspect_ratio))
    except Exception as exc:
        manifest.status = "failed"
        manifest.error = f"image provider setup failed: {type(exc).__name__}: {exc}"
        manifest.save(root / "project.json")
        raise
    if tts is None and do_tts:
        if manifest.tts_source == 'windows':
            from .tts import WindowsTTSProvider
            tts = WindowsTTSProvider()
        elif manifest.tts_source == 'cosyvoice':
            from .tts import CosyVoiceTTSProvider
            tts = CosyVoiceTTSProvider()
        elif manifest.tts_source == 'edge':
            from .tts import EdgeNeuralTTSProvider
            tts = EdgeNeuralTTSProvider()
        elif manifest.tts_source == 'silent':
            tts = FakeTTSProvider()
        else:
            raise ValueError('unsupported TTS source')
    image_provider = image_provider or TextFrameProvider()
    def _resolve_optional_ffmpeg() -> str | None:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        if real_preview:
            try:
                from .local_media import find_ffmpeg
                return find_ffmpeg()
            except Exception:
                return None
        return None
    from .video_router import resolve_shot_video_route
    total_shots = len(manifest.shots)
    auto_rework = bool(getattr(manifest, "auto_rework", False))
    relay_tails = bool(getattr(manifest, "relay_tail_frames", False))

    def _invalidate_stale_shot_assets(shot_obj, expected_key: str) -> None:
        if shot_obj.cache_key is not None and shot_obj.cache_key != expected_key:
            if not do_keyframes and not do_video:
                if do_tts:
                    shot_obj.assets.pop("audio", None)
                return
            preserved_external = {}
            ctx = shot_obj.scene_context if isinstance(shot_obj.scene_context, dict) else {}
            for ext_role in ("image", "last_frame", "audio", "video"):
                prov = ctx.get(f"_external_{ext_role}_provenance")
                ext_path = shot_obj.assets.get(ext_role)
                if (
                    isinstance(prov, dict)
                    and prov.get("text") == shot_obj.text
                    and isinstance(ext_path, str)
                    and ext_path.strip()
                    and Path(ext_path).is_file()
                ):
                    preserved_external[ext_role] = ext_path
            shot_obj.assets.clear()
            shot_obj.assets.update(preserved_external)
            shot_obj.image_task.clear()

    def _probe_audio_duration(audio_file: Path | str | None, fallback: float) -> float:
        if audio_file and Path(audio_file).is_file():
            try:
                with wave.open(str(audio_file), "rb") as source:
                    return max(1.0, source.getnframes() / source.getframerate())
            except (wave.Error, OSError):
                pass
        return fallback

    def _sync_elastic_shot_duration(shot_obj, audio_dur: float, video_file: Path | str | None) -> None:
        if alignment_mode != "elastic" or manifest.tts_source == "silent" or not video_file:
            return
        video_dur = (shot_obj.scene_context or {}).get("_video_duration")
        if video_dur is None:
            opt_ffmpeg = _resolve_optional_ffmpeg()
            if opt_ffmpeg:
                video_dur = probe_duration(opt_ffmpeg, Path(video_file))
                if video_dur is not None:
                    shot_obj.scene_context["_video_duration"] = video_dur
        if video_dur is not None:
            shot_obj.duration = resolve_elastic_shot_duration(
                audio_dur, video_dur, tts_source=manifest.tts_source, alignment_mode=alignment_mode
            )

    def _maybe_apply_lipsync(shot_obj, video_file: Path) -> Path:
        if getattr(manifest, "lipsync_provider", "off") != "off" and shot_obj.assets.get("audio"):
            from .lipsync import apply_shot_lipsync

            updated = apply_shot_lipsync(manifest, root, shot_obj, video_file, shot_obj.assets["audio"])
            shot_obj.assets["video"] = str(updated)
            return updated
        return video_file

    avail_providers = (
        set(getattr(video_provider, "adapters_by_provider", {}).keys())
        or ({"dashscope", "comfyui"} if (video_provider is not None or execution_strategy == "stage_batched" or do_keyframes) else None)
    )

    if execution_strategy == "interleaved" and clean_stage == "all":
        for i, shot in enumerate(manifest.shots, 1):
            if (relay_tails or auto_rework) and i > 1:
                from .frame_decomposer import apply_tail_frame_relay
                apply_tail_frame_relay(manifest, root, manifest.shots[i - 2], shot)
            key = _shot_cache_key(manifest, shot)
            pre_route = resolve_shot_video_route(
                manifest,
                shot,
                has_first_frame=True,
                has_last_frame=bool(shot.assets.get("last_frame")),
                available_providers=(
                    set(getattr(video_provider, "adapters_by_provider", {}).keys())
                    or ({"dashscope", "comfyui"} if video_provider is not None else None)
                ),
                shot_index=i - 1,
                total_shots=total_shots,
            )
            expects_video = (
                manifest.mode == "render"
                and video_provider is not None
                and pre_route["route"] != "preview"
                and (shot.scene_context or {}).get("_video_route") != "blocked:image_gate_failed"
            )
            cached = (shot.cache_key == key and shot.status in {"generated", "done"}
                      and {"image", "audio"} <= shot.assets.keys()
                      and (not expects_video
                           or (shot.mode == "render" and "video" in shot.assets))
                      and all(Path(p).is_file() for p in shot.assets.values()))
            if cached:
                if manifest.mode == "render" and shot.mode == "render" and shot.assets.get("video"):
                    audio_dur = _probe_audio_duration(shot.assets.get("audio"), shot.duration)
                    _sync_elastic_shot_duration(shot, audio_dur, shot.assets["video"])
                continue
            _invalidate_stale_shot_assets(shot, key)
            shot.cache_key = key
            shot.status = "generating"; shot.error = None
            manifest.status = "generating"
            manifest.save(root / "project.json")
            # A cloud task may already have been accepted when the connection fails.
            # Leave resubmission to explicit resume/retry rather than bill twice.
            attempts_limit = 1 if manifest.image_source in {"dashscope", "codex"} or video_provider is not None else max_attempts
            for attempt in range(attempts_limit):
                shot.attempts += 1
                try:
                    image = shot.assets.get("image")
                    generating_new_image = not (image and Path(image).exists())
                    img_route = f"image:{manifest.image_source}"
                    img_cost = estimate_operation_cost("image", img_route, manifest=manifest)
                    if generating_new_image and img_cost > 0:
                        check_rework_budget(manifest, img_cost)
                    frame_path = Path(image) if image and Path(image).exists() else image_provider.generate(shot, frames / f"{i:04d}.png")
                    shot.assets["image"] = str(frame_path)
                    if generating_new_image and (img_cost > 0 or auto_rework):
                        record_cost_entry(
                            manifest,
                            shot_id=shot.id,
                            stage="image",
                            provider_or_route=img_route,
                            cost_cny=img_cost,
                            reason="pipeline",
                        )
                    from .character_bible import record_auto_anchor_frame
                    record_auto_anchor_frame(manifest, root, shot, frame_path)
                    allow_paid_video = True
                    if auto_rework:
                        from .rework_engine import run_image_pre_audit_and_rework
                        frame_path, allow_paid_video = run_image_pre_audit_and_rework(
                            manifest,
                            root,
                            shot,
                            frame_path,
                            image_provider,
                            shot_index=i,
                            evaluator=critic_evaluator,
                        )
                        key = _shot_cache_key(manifest, shot)
                    shot.cache_key = key
                    manifest.save(root / "project.json")
                    existing_audio = shot.assets.get("audio")
                    generating_new_audio = not (existing_audio and Path(existing_audio).is_file())
                    tts_route = f"tts:{manifest.tts_source}"
                    tts_cost = estimate_operation_cost("tts", tts_route, manifest=manifest)
                    if generating_new_audio and tts_cost > 0:
                        check_rework_budget(manifest, tts_cost)
                    audio_path = (
                        Path(existing_audio)
                        if existing_audio and Path(existing_audio).is_file()
                        else synthesize_shot_audio(
                            tts,
                            shot,
                            audio / f"{i:04d}.wav",
                            speech_mode=speech_mode,
                            voice_cast=voice_cast,
                            character_bible=manifest.character_bible,
                        )
                    )
                    shot.assets["audio"] = str(audio_path)
                    if generating_new_audio and (tts_cost > 0 or auto_rework):
                        record_cost_entry(
                            manifest,
                            shot_id=shot.id,
                            stage="tts",
                            provider_or_route=tts_route,
                            cost_cny=tts_cost,
                            reason="pipeline",
                        )
                    audio_dur = _probe_audio_duration(audio_path, shot.duration)
                    shot.duration = audio_dur
                    if manifest.mode == "render":
                        if not allow_paid_video:
                            shot.scene_context["_video_route"] = "blocked:image_gate_failed"
                            shot.scene_context["_video_model"] = "preview"
                            if shot.motion == "none" and pre_route.get("suggested_motion") and pre_route["suggested_motion"] != "none":
                                shot.motion = pre_route["suggested_motion"]
                            shot.mode = "preview"
                        elif pre_route["route"] == "preview":
                            shot.scene_context["_video_route"] = "preview"
                            shot.scene_context["_video_model"] = "preview"
                            if shot.motion == "none" and pre_route.get("suggested_motion") and pre_route["suggested_motion"] != "none":
                                shot.motion = pre_route["suggested_motion"]
                            shot.mode = "preview"
                        elif video_provider is not None:
                            try:
                                if pre_route["route"] == "comfy_fl2v" and not (
                                    shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file()
                                ):
                                    from .codex_image import ensure_shot_end_keyframe

                                    ensure_shot_end_keyframe(
                                        manifest,
                                        root,
                                        shot,
                                        frame_path,
                                        image_provider,
                                        shot_index=i,
                                        reason="pipeline:comfy_fl2v_end_keyframe",
                                    )
                                    key = _shot_cache_key(manifest, shot)
                                planned_vid_cost = estimate_operation_cost("video", pre_route["route"], manifest=manifest)
                                if planned_vid_cost > 0:
                                    check_rework_budget(manifest, planned_vid_cost)
                                video_path = Path(video_provider.generate(shot, frame_path, root / "video" / f"{i:04d}.mp4"))
                                shot.assets["video"] = str(video_path)
                                shot.mode = "render"
                                actual_route = (shot.scene_context or {}).get("_video_route") or pre_route["route"]
                                actual_vid_cost = estimate_operation_cost("video", actual_route, manifest=manifest)
                                if actual_vid_cost > 0 or auto_rework:
                                    record_cost_entry(
                                        manifest,
                                        shot_id=shot.id,
                                        stage="video",
                                        provider_or_route=actual_route,
                                        cost_cny=actual_vid_cost,
                                        reason="pipeline",
                                    )
                                _sync_elastic_shot_duration(shot, audio_dur, video_path)
                                if auto_rework:
                                    from .rework_engine import run_video_post_audit_and_rework
                                    video_path = run_video_post_audit_and_rework(
                                        manifest,
                                        root,
                                        shot,
                                        frame_path,
                                        video_path,
                                        video_provider,
                                        audio_dur=audio_dur,
                                        alignment_mode=alignment_mode,
                                        evaluator=critic_evaluator,
                                        ffmpeg_resolver=_resolve_optional_ffmpeg,
                                        image_provider=image_provider,
                                        shot_index=i,
                                    )
                                    key = _shot_cache_key(manifest, shot)
                                elif relay_tails:
                                    from .frame_decomposer import refresh_shot_diagnostics
                                    refresh_shot_diagnostics(manifest, root, shot.id)
                                _maybe_apply_lipsync(shot, Path(shot.assets["video"]))
                            except (BudgetExceededError, CodexQuotaExceededError):
                                raise
                            except Exception as exc:
                                shot.error = f"video fallback: {type(exc).__name__}: {exc}"
                                shot.mode = "preview"
                                shot.scene_context["_video_route"] = "fallback:preview"
                                if shot.motion == "none" and pre_route.get("suggested_motion") and pre_route["suggested_motion"] != "none":
                                    shot.motion = pre_route["suggested_motion"]
                    shot.cache_key = key; shot.status = "generated"
                    break
                except CodexQuotaExceededError as exc:
                    shot.status = "failed"
                    shot.review = "needs_takeover"
                    shot.error = str(exc)
                    manifest.status = "awaiting_agent_takeover"
                    manifest.error = str(exc)
                    manifest.save(root / "project.json")
                    return None
                except BudgetExceededError as exc:
                    shot.status = "failed"
                    shot.review = "needs_takeover"
                    shot.error = str(exc)
                    manifest.status = "awaiting_agent_takeover"
                    manifest.error = str(exc)
                    manifest.save(root / "project.json")
                    raise
                except Exception as exc:
                    shot.error = f"{type(exc).__name__}: {exc}"
                    if attempt + 1 == attempts_limit: shot.status = "failed"
            manifest.save(root / "project.json")
            if shot.status == "failed": manifest.status = "failed"; continue
    elif do_keyframes or do_tts or do_video:
        # Stage-batched execution: complete all offline/free keyframes (+ end frames) and TTS before cloud video.
        if do_keyframes or do_tts:
            for i, shot in enumerate(manifest.shots, 1):
                if do_keyframes and (relay_tails or auto_rework) and i > 1:
                    from .frame_decomposer import apply_tail_frame_relay
                    apply_tail_frame_relay(manifest, root, manifest.shots[i - 2], shot)
                key = _shot_cache_key(manifest, shot)
                _invalidate_stale_shot_assets(shot, key)
                pre_route = resolve_shot_video_route(
                    manifest,
                    shot,
                    has_first_frame=True,
                    has_last_frame=bool(shot.assets.get("last_frame")),
                    available_providers=avail_providers,
                    shot_index=i - 1,
                    total_shots=total_shots,
                )
                img_ready = (not do_keyframes) or (
                    bool(shot.assets.get("image"))
                    and Path(shot.assets["image"]).is_file()
                    and (
                        manifest.mode != "render"
                        or pre_route["route"] != "comfy_fl2v"
                        or (bool(shot.assets.get("last_frame")) and Path(shot.assets["last_frame"]).is_file())
                    )
                )
                aud_ready = (not do_tts) or (bool(shot.assets.get("audio")) and Path(shot.assets["audio"]).is_file())
                if shot.cache_key == key and shot.status in {"generated", "done"} and img_ready and aud_ready:
                    continue

                shot.cache_key = key
                shot.status = "generating"; shot.error = None
                manifest.status = "generating"
                manifest.save(root / "project.json")
                attempts_limit = 1 if manifest.image_source in {"dashscope", "codex"} else max_attempts
                for attempt in range(attempts_limit):
                    shot.attempts += 1
                    try:
                        if do_keyframes:
                            image = shot.assets.get("image")
                            generating_new_image = not (image and Path(image).exists())
                            img_route = f"image:{manifest.image_source}"
                            img_cost = estimate_operation_cost("image", img_route, manifest=manifest)
                            if generating_new_image and img_cost > 0:
                                check_rework_budget(manifest, img_cost)
                            frame_path = Path(image) if image and Path(image).exists() else image_provider.generate(shot, frames / f"{i:04d}.png")
                            shot.assets["image"] = str(frame_path)
                            if generating_new_image and (img_cost > 0 or auto_rework):
                                record_cost_entry(
                                    manifest,
                                    shot_id=shot.id,
                                    stage="image",
                                    provider_or_route=img_route,
                                    cost_cny=img_cost,
                                    reason="pipeline",
                                )
                            from .character_bible import record_auto_anchor_frame
                            record_auto_anchor_frame(manifest, root, shot, frame_path)
                            allow_paid_video = True
                            if auto_rework:
                                from .rework_engine import run_image_pre_audit_and_rework
                                frame_path, allow_paid_video = run_image_pre_audit_and_rework(
                                    manifest,
                                    root,
                                    shot,
                                    frame_path,
                                    image_provider,
                                    shot_index=i,
                                    evaluator=critic_evaluator,
                                )
                            if not allow_paid_video:
                                shot.scene_context["_video_route"] = "blocked:image_gate_failed"
                                shot.scene_context["_video_model"] = "preview"
                                shot.mode = "preview"
                            elif (
                                manifest.mode == "render"
                                and pre_route["route"] == "comfy_fl2v"
                                and not (shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file())
                            ):
                                from .codex_image import ensure_shot_end_keyframe

                                try:
                                    ensure_shot_end_keyframe(
                                        manifest,
                                        root,
                                        shot,
                                        frame_path,
                                        image_provider,
                                        shot_index=i,
                                        reason="pipeline:comfy_fl2v_end_keyframe",
                                    )
                                except Exception as end_exc:
                                    if any(tok in str(end_exc) for tok in ("auth_unavailable", "usage_limit_reached", "HTTP 503", "Codex")):
                                        shot.video_route = "comfy_i2v"
                                    else:
                                        raise
                            key = _shot_cache_key(manifest, shot)
                            shot.cache_key = key

                        if do_tts:
                            existing_audio = shot.assets.get("audio")
                            generating_new_audio = not (existing_audio and Path(existing_audio).is_file())
                            tts_route = f"tts:{manifest.tts_source}"
                            tts_cost = estimate_operation_cost("tts", tts_route, manifest=manifest)
                            if generating_new_audio and tts_cost > 0:
                                check_rework_budget(manifest, tts_cost)
                            audio_path = (
                                Path(existing_audio)
                                if existing_audio and Path(existing_audio).is_file()
                                else synthesize_shot_audio(
                                    tts,
                                    shot,
                                    audio / f"{i:04d}.wav",
                                    speech_mode=speech_mode,
                                    voice_cast=voice_cast,
                                    character_bible=manifest.character_bible,
                                )
                            )
                            shot.assets["audio"] = str(audio_path)
                            if generating_new_audio and (tts_cost > 0 or auto_rework):
                                record_cost_entry(
                                    manifest,
                                    shot_id=shot.id,
                                    stage="tts",
                                    provider_or_route=tts_route,
                                    cost_cny=tts_cost,
                                    reason="pipeline",
                                )
                            shot.duration = _probe_audio_duration(audio_path, shot.duration)
                        shot.status = "generated"
                        break
                    except CodexQuotaExceededError as exc:
                        shot.status = "failed"
                        shot.review = "needs_takeover"
                        shot.error = str(exc)
                        manifest.status = "awaiting_agent_takeover"
                        manifest.error = str(exc)
                        manifest.save(root / "project.json")
                        return None
                    except BudgetExceededError as exc:
                        shot.status = "failed"
                        shot.review = "needs_takeover"
                        shot.error = str(exc)
                        manifest.status = "awaiting_agent_takeover"
                        manifest.error = str(exc)
                        manifest.save(root / "project.json")
                        raise
                    except Exception as exc:
                        shot.error = f"{type(exc).__name__}: {exc}"
                        if attempt + 1 == attempts_limit:
                            shot.status = "failed"
                manifest.save(root / "project.json")
                if shot.status == "failed":
                    manifest.status = "failed"

        if do_video:
            for i, shot in enumerate(manifest.shots, 1):
                if shot.status == "failed":
                    continue
                if do_keyframes and (relay_tails or auto_rework) and i > 1:
                    from .frame_decomposer import apply_tail_frame_relay
                    apply_tail_frame_relay(manifest, root, manifest.shots[i - 2], shot)
                key = _shot_cache_key(manifest, shot)
                if manifest.mode != "render":
                    shot.cache_key = key
                    shot.status = "generated"
                    continue
                if (shot.scene_context or {}).get("_video_route") == "blocked:image_gate_failed":
                    shot.mode = "preview"
                    shot.cache_key = key
                    shot.status = "generated"
                    manifest.save(root / "project.json")
                    continue

                pre_route = resolve_shot_video_route(
                    manifest,
                    shot,
                    has_first_frame=bool(shot.assets.get("image")),
                    has_last_frame=bool(shot.assets.get("last_frame")),
                    available_providers=(
                        set(getattr(video_provider, "adapters_by_provider", {}).keys())
                        or ({"dashscope", "comfyui"} if video_provider is not None else None)
                    ),
                    shot_index=i - 1,
                    total_shots=total_shots,
                )
                expects_video = video_provider is not None and pre_route["route"] != "preview"
                cached_vid = (
                    shot.cache_key == key
                    and shot.status in {"generated", "done"}
                    and (not expects_video or (shot.mode == "render" and shot.assets.get("video") and Path(shot.assets["video"]).is_file()))
                )
                if cached_vid:
                    if shot.mode == "render" and shot.assets.get("video"):
                        audio_dur = _probe_audio_duration(shot.assets.get("audio"), shot.duration)
                        _sync_elastic_shot_duration(shot, audio_dur, shot.assets["video"])
                    continue

                frame_str = shot.assets.get("image")
                frame_path = Path(frame_str) if frame_str and Path(frame_str).is_file() else None
                audio_dur = _probe_audio_duration(shot.assets.get("audio"), shot.duration)
                shot.duration = audio_dur
                try:
                    if pre_route["route"] == "preview":
                        shot.scene_context["_video_route"] = "preview"
                        shot.scene_context["_video_model"] = "preview"
                        if shot.motion == "none" and pre_route.get("suggested_motion") and pre_route["suggested_motion"] != "none":
                            shot.motion = pre_route["suggested_motion"]
                        shot.mode = "preview"
                    elif video_provider is not None:
                        try:
                            if pre_route["route"] == "comfy_fl2v" and frame_path is not None and not (
                                shot.assets.get("last_frame") and Path(shot.assets["last_frame"]).is_file()
                            ):
                                from .codex_image import ensure_shot_end_keyframe

                                try:
                                    ensure_shot_end_keyframe(
                                        manifest,
                                        root,
                                        shot,
                                        frame_path,
                                        image_provider,
                                        shot_index=i,
                                        reason="pipeline:comfy_fl2v_end_keyframe",
                                    )
                                except Exception as end_exc:
                                    if any(tok in str(end_exc) for tok in ("auth_unavailable", "usage_limit_reached", "HTTP 503", "Codex")):
                                        shot.video_route = "comfy_i2v"
                                    else:
                                        raise
                                key = _shot_cache_key(manifest, shot)
                            planned_vid_cost = estimate_operation_cost("video", pre_route["route"], manifest=manifest)
                            if planned_vid_cost > 0:
                                check_rework_budget(manifest, planned_vid_cost)
                            video_path = Path(video_provider.generate(shot, frame_path, root / "video" / f"{i:04d}.mp4"))
                            shot.assets["video"] = str(video_path)
                            shot.mode = "render"
                            actual_route = (shot.scene_context or {}).get("_video_route") or pre_route["route"]
                            actual_vid_cost = estimate_operation_cost("video", actual_route, manifest=manifest)
                            if actual_vid_cost > 0 or auto_rework:
                                record_cost_entry(
                                    manifest,
                                    shot_id=shot.id,
                                    stage="video",
                                    provider_or_route=actual_route,
                                    cost_cny=actual_vid_cost,
                                    reason="pipeline",
                                )
                            _sync_elastic_shot_duration(shot, audio_dur, video_path)
                            if auto_rework:
                                from .rework_engine import run_video_post_audit_and_rework
                                video_path = run_video_post_audit_and_rework(
                                    manifest,
                                    root,
                                    shot,
                                    frame_path,
                                    video_path,
                                    video_provider,
                                    audio_dur=audio_dur,
                                    alignment_mode=alignment_mode,
                                    evaluator=critic_evaluator,
                                    ffmpeg_resolver=_resolve_optional_ffmpeg,
                                    image_provider=image_provider,
                                    shot_index=i,
                                )
                                key = _shot_cache_key(manifest, shot)
                            elif relay_tails:
                                from .frame_decomposer import refresh_shot_diagnostics
                                refresh_shot_diagnostics(manifest, root, shot.id)
                            _maybe_apply_lipsync(shot, Path(shot.assets["video"]))
                        except (BudgetExceededError, CodexQuotaExceededError):
                            raise
                        except Exception as exc:
                            shot.error = f"video fallback: {type(exc).__name__}: {exc}"
                            shot.mode = "preview"
                            shot.scene_context["_video_route"] = "fallback:preview"
                            if shot.motion == "none" and pre_route.get("suggested_motion") and pre_route["suggested_motion"] != "none":
                                shot.motion = pre_route["suggested_motion"]
                    shot.cache_key = key
                    shot.status = "generated"
                except CodexQuotaExceededError as exc:
                    shot.status = "failed"
                    shot.review = "needs_takeover"
                    shot.error = str(exc)
                    manifest.status = "awaiting_agent_takeover"
                    manifest.error = str(exc)
                    manifest.save(root / "project.json")
                    return None
                except BudgetExceededError as exc:
                    shot.status = "failed"
                    shot.review = "needs_takeover"
                    shot.error = str(exc)
                    manifest.status = "awaiting_agent_takeover"
                    manifest.error = str(exc)
                    manifest.save(root / "project.json")
                    raise
                manifest.save(root / "project.json")

    if not do_assemble:
        if any(getattr(s, "review", None) == "needs_takeover" for s in manifest.shots):
            manifest.status = "awaiting_agent_takeover"
        elif any(s.status == "failed" for s in manifest.shots):
            manifest.status = "failed"
        else:
            manifest.status = f"staged:{clean_stage}"
        manifest.save(root / "project.json")
        return None
    from .assembly_quality_gate import AssemblyQualityBlocked, assert_no_failed_shots_for_assembly
    try:
        assert_no_failed_shots_for_assembly(manifest, output_kind="preview")
    except AssemblyQualityBlocked as exc:
        manifest.status = "failed"
        manifest.error = str(exc)
        manifest.output = None
        manifest.save(root / "project.json")
        raise
    effective_transitions = [
        resolve_safe_shot_transition(
            s,
            manifest.shots[idx + 1] if idx + 1 < len(manifest.shots) else None,
            character_bible=manifest.character_bible,
        )
        for idx, s in enumerate(manifest.shots)
    ]
    for s, eff_trans in zip(manifest.shots, effective_transitions):
        if eff_trans != getattr(s, "transition", "cut") and isinstance(s.scene_context, dict):
            s.scene_context["_effective_transition"] = eff_trans
    needs_xfade = any(t in XFADE_TRANSITIONS for t in effective_transitions[:-1])
    needs_segments = (
        (manifest.mode == "render" and any(shot.mode == "render" and shot.assets.get("video") for shot in manifest.shots))
        or any(s.motion != 'none' or eff_trans != 'cut' for s, eff_trans in zip(manifest.shots, effective_transitions))
        or aspect_ratio != "16:9"
        or scale_mode != "contain"
        or needs_xfade
    )
    if needs_segments or alignment_mode != "pad" or subtitle_style == "burn_in" or bgm_source:
        from .segments import normalize_durations
        normalize_durations(manifest.shots)
    if enforce_review and all(s.status != 'failed' for s in manifest.shots):
        if block_unapproved(manifest, root, 'assets'):
            return None
    ffmpeg = shutil.which("ffmpeg")
    cues = build_subtitle_cues(manifest.shots, aspect_ratio)
    clock = sum(shot.duration for shot in manifest.shots)
    subtitle_file = root / "subtitles.srt"
    subtitle_file.write_text(format_srt(cues), encoding="utf-8")
    if subtitle_style == "burn_in":
        (root / "subtitles.ass").write_text(format_ass(cues, aspect_ratio), encoding="utf-8")
    output = root / "preview.mp4"
    if real_preview:
        from .local_media import find_ffmpeg
        try:
            ffmpeg = find_ffmpeg()
            if any(s.status == "failed" for s in manifest.shots):
                failures = "; ".join(f"{s.id}: {s.error}" for s in manifest.shots if s.status == "failed")
                raise RuntimeError(f"Shot generation failed; {failures}")
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.save(root / "project.json")
            raise
    images = [shot.assets.get("image") for shot in manifest.shots]
    if ffmpeg and images and all(image and Path(image).is_file() and Path(image).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} for image in images):
        concat = root / "timeline.txt"
        audio_concat = root / "audio-timeline.txt"
        lines = []
        for shot in manifest.shots:
            lines += [_concat_file(shot.assets['image']), f"duration {shot.duration}"]
        lines.append(lines[-2]); concat.write_text("\n".join(lines), encoding="utf-8")
        audio_concat.write_text("\n".join(_concat_file(shot.assets['audio']) for shot in manifest.shots), encoding="utf-8")
        manifest.status = "assembling"
        manifest.save(root / "project.json")
        temporary_output = root / f".preview-{uuid4().hex}.mp4"
        segment_directory = None
        try:
            from .segments import render_segments, align_audio_segments
            segment_directory = TemporaryDirectory(prefix=".segments-", dir=root)
            use_stereo = bool(bgm_source or manifest.tts_source in ("windows", "cosyvoice", "edge"))
            aligned_audio = align_audio_segments(
                ffmpeg,
                manifest.shots,
                segment_directory.name,
                use_video_audio=manifest.mode == 'render' and manifest.tts_source == 'silent',
                sample_rate=44100 if use_stereo else 16000,
                channels=2 if use_stereo else 1,
            )
            audio_concat.write_text("\n".join(_concat_file(str(segment)) for segment in aligned_audio), encoding="utf-8")
            audio_input_args = ["-f", "concat", "-safe", "0", "-i", str(audio_concat.resolve())]
            if bgm_source:
                voice_master = Path(segment_directory.name) / "voice_master.wav"
                subprocess.run(
                    [ffmpeg, "-y", "-xerror", "-f", "concat", "-safe", "0", "-i", str(audio_concat.resolve()),
                     "-c:a", "pcm_s16le", str(voice_master.resolve())],
                    check=True,
                    capture_output=True,
                )
                mixed_master = mix_audio_with_bgm(
                    ffmpeg,
                    voice_master,
                    bgm_source,
                    Path(segment_directory.name) / "mixed_master.wav",
                    duration=clock,
                    volume=bgm_volume,
                    ducking=bgm_ducking,
                    work_dir=root,
                )
                audio_input_args = ["-i", str(mixed_master.resolve())]

            video_input_args = ["-f", "concat", "-safe", "0", "-i", str(concat.resolve())]
            if needs_segments:
                segments = render_segments(
                    ffmpeg,
                    manifest.shots,
                    segment_directory.name,
                    aspect_ratio=aspect_ratio,
                    scale_mode=scale_mode,
                    alignment_mode=alignment_mode,
                    character_bible=manifest.character_bible,
                )
                if needs_xfade:
                    xfade_video = compose_xfade_segments(
                        ffmpeg,
                        segments,
                        manifest.shots,
                        Path(segment_directory.name) / "xfade_master.mp4",
                        character_bible=manifest.character_bible,
                    )
                    video_input_args = ["-i", str(xfade_video.resolve())]
                else:
                    concat.write_text("\n".join(_concat_file(str(segment)) for segment in segments), encoding="utf-8")

            width, height = resolve_canvas_size(aspect_ratio)
            vf_chain = (
                f"fps=25,scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p"
            )
            if subtitle_style == "burn_in":
                vf_chain += ",ass=subtitles.ass"
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-xerror",
                    *video_input_args,
                    *audio_input_args,
                    "-i",
                    str(subtitle_file.resolve()),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-map",
                    "2:0",
                    "-c:s",
                    "mov_text",
                    "-metadata:s:s:0",
                    "language=zho",
                    "-t",
                    str(clock),
                    "-vf",
                    vf_chain,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(temporary_output.resolve()),
                ],
                check=True,
                capture_output=True,
                cwd=str(root),
            )
            temporary_output.replace(output)
        except (OSError, subprocess.SubprocessError) as exc:
            manifest.status = "failed"
            manifest.error = f"assembly failed: {type(exc).__name__}: {exc}"
            manifest.save(root / "project.json")
            raise
        finally:
            temporary_output.unlink(missing_ok=True)
            if segment_directory is not None:
                segment_directory.cleanup()
    else:
        output = root / "preview.txt"
        output.write_text("\n".join(shot.text for shot in manifest.shots), encoding="utf-8")
    manifest.output = str(output)
    if any(getattr(s, "review", None) == "needs_takeover" for s in manifest.shots):
        manifest.status = "awaiting_agent_takeover"
    else:
        manifest.status = "done" if all(s.status != "failed" for s in manifest.shots) else "partial"
    manifest.timeline = [{"shot_id": shot.id, "start": sum(s.duration for s in manifest.shots[:i]), "duration": shot.duration, "transition": shot.transition, "motion": shot.motion} for i, shot in enumerate(manifest.shots)]
    for shot in manifest.shots:
        if shot.status != "failed": shot.status = "done"
    manifest.save(root / "project.json")
    return output
