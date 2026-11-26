"""Smart video model routing, motion intensity scoring, and fallback resolution."""

from __future__ import annotations

import os
from typing import Any

from .manifest import ProjectManifest, Shot
from .wan_video import is_wan_i2v_model, is_wan_kf2v_model, is_wan_r2v_model

VIDEO_PROVIDERS = ("dashscope", "comfyui", "auto")
ROUTING_POLICIES = ("manual", "smart_i2v", "hybrid_economy", "quality_first")
FALLBACK_PROVIDERS = ("dashscope", "comfyui", "preview")
SHOT_VIDEO_ROUTES = (
    "auto",
    "preview",
    "i2v_fast",
    "i2v_quality",
    "kf2v",
    "r2v",
    "t2v",
    "comfy_i2v",
    "comfy_fl2v",
    "comfy_t2v",
)

VIDEO_MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "wan2.2-i2v-flash": {
        "provider": "dashscope",
        "mode": "i2v",
        "tier": "fast",
        "default_resolution": "720P",
        "supported_durations": (5,),
        "label": "Wan 2.2 极速图生视频 (I2V Flash)",
    },
    "wan2.2-i2v-plus": {
        "provider": "dashscope",
        "mode": "i2v",
        "tier": "quality",
        "default_resolution": "1080P",
        "supported_durations": (5,),
        "label": "Wan 2.2 专业图生视频 (I2V Plus)",
    },
    "wanx2.1-i2v-turbo": {
        "provider": "dashscope",
        "mode": "i2v",
        "tier": "fast",
        "default_resolution": "720P",
        "supported_durations": (3, 4, 5),
        "label": "Wanx 2.1 极速图生视频 (3-5s)",
    },
    "wanx2.1-i2v-plus": {
        "provider": "dashscope",
        "mode": "i2v",
        "tier": "quality",
        "default_resolution": "720P",
        "supported_durations": (5,),
        "label": "Wanx 2.1 专业图生视频",
    },
    "wan2.2-kf2v-flash": {
        "provider": "dashscope",
        "mode": "kf2v",
        "tier": "fast",
        "default_resolution": "720P",
        "supported_durations": (5,),
        "label": "Wan 2.2 首尾帧插值视频 (KF2V)",
    },
    "wan2.1-vace-plus": {
        "provider": "dashscope",
        "mode": "r2v",
        "tier": "quality",
        "default_resolution": "720P",
        "supported_durations": (5,),
        "label": "Wan 2.1 VACE 多图参考生视频 (R2V)",
    },
    "wanx2.1-t2v-turbo": {
        "provider": "dashscope",
        "mode": "t2v",
        "tier": "fast",
        "default_resolution": "720P",
        "supported_durations": (5,),
        "label": "Wanx 2.1 极速文生视频 (T2V)",
    },
}

_ASPECT_TO_WAN_SIZE = {
    "16:9": "1280*720",
    "9:16": "720*1280",
    "1:1": "960*960",
    "3:4": "720*960",
    "4:3": "960*720",
}

_HIGH_MOTION_KEYWORDS = (
    "跑", "奔", "追", "冲", "跳", "跃", "飞", "打", "撞", "摔", "跌", "炸", "爆",
    "翻滚", "拔剑", "挥", "疾驰", "飙车", "推门", "转身", "争执", "撕", "雨中狂奔",
    "run", "chase", "fight", "jump", "dash", "explode", "crash", "sprint", "spin",
)
_LOW_MOTION_KEYWORDS = (
    "说", "问", "回答", "低声", "呢喃", "轻声", "微笑", "叹气", "沉默", "注视", "看着",
    "盯着", "特写", "近景", "静静", "坐着", "站着", "凝视", "回忆", "思考", "对视",
    "dialogue", "whisper", "smile", "sigh", "close-up", "stare", "sit", "quiet",
)


def default_wan_size_for_aspect(aspect_ratio: str | None = "16:9") -> str:
    return _ASPECT_TO_WAN_SIZE.get((aspect_ratio or "16:9").strip(), "1280*720")


def validate_routing_settings(
    video_provider: str = "comfyui",
    routing_policy: str = "manual",
    fallback_provider: str | None = None,
) -> tuple[str, str, str | None]:
    provider = (video_provider or "comfyui").strip()
    if provider not in VIDEO_PROVIDERS:
        raise ValueError("unsupported video provider")
    policy = (routing_policy or "manual").strip()
    if policy not in ROUTING_POLICIES:
        raise ValueError(f"unsupported video routing policy: {routing_policy}")
    fallback = fallback_provider.strip() if isinstance(fallback_provider, str) and fallback_provider.strip() else None
    if fallback is not None and fallback not in FALLBACK_PROVIDERS:
        raise ValueError(f"unsupported video fallback provider: {fallback_provider}")
    return provider, policy, fallback


def validate_shot_video_route(route: str | None) -> str | None:
    if route is None or not str(route).strip():
        return None
    cleaned = str(route).strip()
    if cleaned not in SHOT_VIDEO_ROUTES:
        raise ValueError(f"unsupported shot video route: {route}")
    return None if cleaned == "auto" else cleaned


def analyze_shot_motion_intensity(shot: Shot) -> dict[str, Any]:
    """Score visual motion complexity of `shot` to guide hybrid video routing."""
    action = ""
    framing = ""
    if isinstance(shot.scene_context, dict):
        action = str(shot.scene_context.get("visual_action") or "")
        framing = str(shot.scene_context.get("framing") or "")
    text_blob = f"{shot.text} {action} {framing}".lower()

    high_hits = sum(1 for kw in _HIGH_MOTION_KEYWORDS if kw in text_blob)
    low_hits = sum(1 for kw in _LOW_MOTION_KEYWORDS if kw in text_blob)
    dialogue_heavy = bool(shot.dialogue_lines) or bool(shot.sub_captions)

    score = 0.45
    score += min(0.45, high_hits * 0.22)
    score -= min(0.30, low_hits * 0.12)
    if dialogue_heavy and high_hits == 0:
        score -= 0.18
    if shot.duration >= 5.0 and high_hits > 0:
        score += 0.08
    score = round(max(0.05, min(0.95, score)), 2)

    if score < 0.35:
        level = "low"
    elif score >= 0.65:
        level = "high"
    else:
        level = "medium"

    if shot.motion and shot.motion != "none":
        suggested_motion = shot.motion
    elif level == "low":
        suggested_motion = "push_in"
    elif level == "medium":
        suggested_motion = "ken_burns"
    else:
        suggested_motion = "pull_out"

    return {
        "score": score,
        "level": level,
        "high_hits": high_hits,
        "low_hits": low_hits,
        "dialogue_heavy": dialogue_heavy,
        "suggested_motion": suggested_motion,
    }


def detect_available_video_providers() -> set[str]:
    available: set[str] = set()
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        available.add("dashscope")
    if (
        os.environ.get("MINIMAX_H3_COMFYUI_BASE_URL", "").strip()
        or os.environ.get("COMFYUI_URL", "").strip()
        or os.environ.get("COMFYUI_HOST", "").strip()
    ):
        available.add("comfyui")
    return available


def _select_backend_provider(
    configured_provider: str,
    available_providers: set[str],
    fallback_provider: str | None,
) -> str:
    if configured_provider != "auto":
        return configured_provider
    if "dashscope" in available_providers:
        return "dashscope"
    if "comfyui" in available_providers:
        return "comfyui"
    if fallback_provider in ("dashscope", "comfyui"):
        return fallback_provider
    return "preview" if fallback_provider == "preview" else "dashscope"


def resolve_shot_video_route(
    manifest: ProjectManifest,
    shot: Shot,
    *,
    has_first_frame: bool = False,
    has_last_frame: bool = False,
    has_character_refs: bool = False,
    available_providers: set[str] | None = None,
    shot_index: int = 0,
    total_shots: int = 1,
) -> dict[str, Any]:
    """Choose provider, workflow mode, and concrete model for a single shot."""
    provider_setting, policy, fallback = validate_routing_settings(
        getattr(manifest, "video_provider", "comfyui"),
        getattr(manifest, "video_routing_policy", "manual"),
        getattr(manifest, "video_fallback_provider", None),
    )
    explicit_route = validate_shot_video_route(getattr(shot, "video_route", None))
    analysis = analyze_shot_motion_intensity(shot)
    avail = available_providers if available_providers is not None else detect_available_video_providers()
    video_options = getattr(manifest, "video_options", {}) or {}

    if explicit_route is not None:
        if explicit_route == "preview":
            return {
                "route": "preview",
                "provider": "preview",
                "model": None,
                "workflow_mode": "preview",
                "suggested_motion": analysis["suggested_motion"],
                "reason": "shot_override:preview",
                "motion_analysis": analysis,
                "fallback_provider": fallback,
            }
        if explicit_route in ("comfy_i2v", "comfy_fl2v", "comfy_t2v"):
            wf_mode = "t2v" if explicit_route == "comfy_t2v" else "i2v"
            return {
                "route": explicit_route,
                "provider": "comfyui",
                "model": None,
                "workflow_mode": wf_mode,
                "needs_end_frame": explicit_route == "comfy_fl2v",
                "suggested_motion": shot.motion,
                "reason": f"shot_override:{explicit_route}",
                "motion_analysis": analysis,
                "fallback_provider": fallback,
            }
        route_model_map = {
            "i2v_fast": ("wan2.2-i2v-flash", "i2v"),
            "i2v_quality": ("wan2.2-i2v-plus", "i2v"),
            "kf2v": ("wan2.2-kf2v-flash", "kf2v"),
            "r2v": ("wan2.1-vace-plus", "r2v"),
            "t2v": ("wanx2.1-t2v-turbo", "t2v"),
        }
        model_name, wf_mode = route_model_map[explicit_route]
        return {
            "route": explicit_route,
            "provider": "dashscope",
            "model": model_name,
            "workflow_mode": wf_mode,
            "suggested_motion": shot.motion,
            "reason": f"shot_override:{explicit_route}",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }

    backend = _select_backend_provider(provider_setting, avail, fallback)
    if backend == "preview":
        return {
            "route": "preview",
            "provider": "preview",
            "model": None,
            "workflow_mode": "preview",
            "suggested_motion": analysis["suggested_motion"],
            "reason": "auto_no_video_backend_preview",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }

    if policy == "manual":
        if backend == "comfyui":
            wf_mode = str(video_options.get("workflow_mode") or ("i2v" if provider_setting == "auto" and has_first_frame else "t2v"))
            route = "comfy_i2v" if wf_mode == "i2v" else "comfy_t2v"
            return {
                "route": route,
                "provider": "comfyui",
                "model": None,
                "workflow_mode": wf_mode,
                "suggested_motion": shot.motion,
                "reason": "manual:comfyui",
                "motion_analysis": analysis,
                "fallback_provider": fallback,
            }
        configured_model = video_options.get("model")
        if configured_model:
            model_name = str(configured_model)
        elif provider_setting == "auto" and has_first_frame:
            model_name = "wan2.2-i2v-flash"
        else:
            model_name = "wanx2.1-t2v-turbo"
        if is_wan_kf2v_model(model_name):
            wf_mode, route = "kf2v", "kf2v"
        elif is_wan_r2v_model(model_name):
            wf_mode, route = "r2v", "r2v"
        elif is_wan_i2v_model(model_name):
            wf_mode = "i2v"
            route = "i2v_quality" if "plus" in model_name else "i2v_fast"
        else:
            wf_mode, route = "t2v", "t2v"
        return {
            "route": route,
            "provider": "dashscope",
            "model": model_name,
            "workflow_mode": wf_mode,
            "suggested_motion": shot.motion,
            "reason": "manual:dashscope",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }

    if policy == "hybrid_economy" and analysis["level"] == "low":
        decision = {
            "route": "preview",
            "provider": "preview",
            "model": None,
            "workflow_mode": "preview",
            "suggested_motion": analysis["suggested_motion"],
            "reason": "hybrid_economy:low_motion_preview",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }
        return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)

    if backend == "comfyui":
        if has_first_frame and (has_last_frame or policy == "quality_first" or analysis["level"] == "high"):
            wf_mode = "i2v"
            route = "comfy_fl2v"
            needs_end_frame = True
        elif has_first_frame:
            wf_mode = "i2v"
            route = "comfy_i2v"
            needs_end_frame = False
        else:
            wf_mode = "t2v"
            route = "comfy_t2v"
            needs_end_frame = False
        decision = {
            "route": route,
            "provider": "comfyui",
            "model": None,
            "workflow_mode": wf_mode,
            "needs_end_frame": needs_end_frame,
            "suggested_motion": shot.motion,
            "reason": f"{policy}:{route}",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }
        return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)

    # DashScope smart/hybrid/quality routing
    if has_first_frame and has_last_frame and (policy in ("smart_i2v", "quality_first") or analysis["level"] == "high"):
        decision = {
            "route": "kf2v",
            "provider": "dashscope",
            "model": "wan2.2-kf2v-flash",
            "workflow_mode": "kf2v",
            "suggested_motion": shot.motion,
            "reason": f"{policy}:kf2v",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }
        return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)
    if has_character_refs and not has_first_frame:
        decision = {
            "route": "r2v",
            "provider": "dashscope",
            "model": "wan2.1-vace-plus",
            "workflow_mode": "r2v",
            "suggested_motion": shot.motion,
            "reason": f"{policy}:r2v_character_refs",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }
        return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)
    if has_first_frame:
        use_quality = policy == "quality_first" or (analysis["level"] == "high" and policy in ("smart_i2v", "hybrid_economy"))
        model_name = "wan2.2-i2v-plus" if use_quality else "wan2.2-i2v-flash"
        route = "i2v_quality" if use_quality else "i2v_fast"
        decision = {
            "route": route,
            "provider": "dashscope",
            "model": model_name,
            "workflow_mode": "i2v",
            "suggested_motion": shot.motion,
            "reason": f"{policy}:{route}",
            "motion_analysis": analysis,
            "fallback_provider": fallback,
        }
        return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)
    decision = {
        "route": "t2v",
        "provider": "dashscope",
        "model": "wanx2.1-t2v-turbo",
        "workflow_mode": "t2v",
        "suggested_motion": shot.motion,
        "reason": f"{policy}:t2v_fallback",
        "motion_analysis": analysis,
        "fallback_provider": fallback,
    }
    return _maybe_apply_flywheel_route(manifest, shot, decision, has_first_frame=has_first_frame, has_last_frame=has_last_frame)


def _maybe_apply_flywheel_route(
    manifest: ProjectManifest,
    shot: Shot,
    decision: dict[str, Any],
    *,
    has_first_frame: bool,
    has_last_frame: bool,
) -> dict[str, Any]:
    from .flywheel import apply_flywheel_to_video_route

    ctx = shot.scene_context if isinstance(shot.scene_context, dict) else {}
    text_blob = " ".join(
        str(p)
        for p in (
            getattr(shot, "prompt_override", None) or "",
            shot.text,
            ctx.get("visual_action") or "",
            ctx.get("framing") or "",
        )
        if p
    )
    return apply_flywheel_to_video_route(
        manifest,
        shot,
        text_blob,
        decision,
        has_first_frame=has_first_frame,
        has_last_frame=has_last_frame,
    )
