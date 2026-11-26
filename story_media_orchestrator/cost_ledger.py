"""Per-minute budget ledger and circuit breaker for cost-aware pipeline rework."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .manifest import ProjectManifest

DEFAULT_UNIT_COSTS_CNY: dict[str, float] = {
    "image:local": 0.0,
    "image:codex": 0.0,
    "image:dashscope": 0.14,
    "tts:silent": 0.0,
    "tts:windows": 0.0,
    "tts:edge": 0.0,
    "tts:cosyvoice": 0.05,
    "lipsync:off": 0.0,
    "lipsync:http": 0.15,
    "lipsync:comfyui": 0.12,
    "lipsync:command": 0.0,
    "video:preview": 0.0,
    "video:comfy_i2v": 0.35,
    "video:comfy_fl2v": 0.45,
    "video:comfy_t2v": 0.35,
    "video:t2v": 0.60,
    "video:i2v_fast": 0.70,
    "video:kf2v": 0.85,
    "video:r2v": 1.80,
    "video:i2v_quality": 2.80,
}

_STAGE_DEFAULTS: dict[str, str] = {
    "image": "image:dashscope",
    "tts": "tts:cosyvoice",
    "audio": "tts:cosyvoice",
    "lipsync": "lipsync:http",
    "video": "video:i2v_fast",
}


class BudgetExceededError(RuntimeError):
    """Raised when a planned rework exceeds per-call or project per-minute budget ceiling."""


def _env_unit_costs() -> dict[str, float]:
    rates: dict[str, float] = {}
    raw_json = os.environ.get("STORY_MEDIA_UNIT_COSTS_JSON", "").strip()
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    try:
                        rates[str(k).strip().lower()] = max(0.0, float(v))
                    except (TypeError, ValueError):
                        continue
        except (ValueError, TypeError):
            pass

    known_env_map = {f"STORY_COST_{k.replace(':', '_').upper()}": k for k in DEFAULT_UNIT_COSTS_CNY}
    for env_key, env_val in os.environ.items():
        if not env_key.startswith("STORY_COST_") or not str(env_val).strip():
            continue
        try:
            cost_val = max(0.0, float(env_val))
        except ValueError:
            continue
        if env_key in known_env_map:
            rates[known_env_map[env_key]] = cost_val
        else:
            suffix = env_key[len("STORY_COST_") :].lower()
            if "_" in suffix:
                stage_part, route_part = suffix.split("_", 1)
                rates[f"{stage_part}:{route_part}"] = cost_val
    return rates


def estimate_operation_cost(
    stage: str,
    provider_or_route: str,
    *,
    count: int = 1,
    manifest: ProjectManifest | None = None,
) -> float:
    """Estimate CNY cost for generating or reworking a shot artifact."""
    norm_stage = (stage or "").strip().lower()
    norm_route = (provider_or_route or "").strip().lower()
    if norm_route in {"comfyui", "comfy"} and norm_stage == "video":
        norm_route = "comfy_i2v"
    key = norm_route if ":" in norm_route else f"{norm_stage}:{norm_route}"

    custom_rates = {}
    if manifest is not None and isinstance(manifest.supervisor_policy, dict):
        raw_rates = manifest.supervisor_policy.get("unit_costs_cny")
        if isinstance(raw_rates, dict):
            custom_rates = {str(k).lower(): float(v) for k, v in raw_rates.items()}

    env_rates = _env_unit_costs()

    if key in custom_rates:
        unit = custom_rates[key]
    elif norm_route in custom_rates:
        unit = custom_rates[norm_route]
    elif key in env_rates:
        unit = env_rates[key]
    elif norm_route in env_rates:
        unit = env_rates[norm_route]
    elif key in DEFAULT_UNIT_COSTS_CNY:
        unit = DEFAULT_UNIT_COSTS_CNY[key]
    else:
        fallback_key = _STAGE_DEFAULTS.get(norm_stage, "")
        unit = DEFAULT_UNIT_COSTS_CNY.get(fallback_key, 0.0)

    return round(max(0.0, float(unit)) * max(1, int(count)), 4)


def effective_project_budget_cny(manifest: ProjectManifest) -> float:
    """Compute effective project spend ceiling in CNY from budget_limit_cny (CNY/min) and duration."""
    policy = manifest.supervisor_policy if isinstance(manifest.supervisor_policy, dict) else {}
    if policy.get("hard_cap_cny") is not None:
        return round(max(0.0, float(policy["hard_cap_cny"])), 4)
    total_sec = sum(max(0.5, float(getattr(s, "duration", 3.0) or 3.0)) for s in manifest.shots)
    if total_sec <= 0.0:
        total_sec = 12.0
    minutes_floor = max(total_sec / 60.0, 0.20)
    limit_per_min = max(0.0, float(getattr(manifest, "budget_limit_cny", 100.0) or 100.0))
    return round(limit_per_min * minutes_floor, 4)


def _ensure_ledger_dict(manifest: ProjectManifest) -> dict[str, Any]:
    ledger = manifest.cost_ledger if isinstance(manifest.cost_ledger, dict) else {}
    if not isinstance(ledger.get("entries"), list):
        ledger["entries"] = []
    manifest.cost_ledger = ledger
    return ledger


def record_cost_entry(
    manifest: ProjectManifest,
    *,
    shot_id: str,
    stage: str,
    provider_or_route: str,
    cost_cny: float,
    reason: str = "pipeline",
    round_index: int = 0,
) -> dict[str, Any]:
    """Append a spend record into manifest.cost_ledger and refresh summary totals."""
    ledger = _ensure_ledger_dict(manifest)
    entry = {
        "shot_id": str(shot_id),
        "stage": str(stage),
        "provider_or_route": str(provider_or_route),
        "cost_cny": round(max(0.0, float(cost_cny)), 4),
        "reason": str(reason),
        "round_index": max(0, int(round_index)),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    ledger["entries"].append(entry)
    summary = summarize_cost_ledger(manifest)
    ledger["spent_cny"] = summary["spent_cny"]
    ledger["rework_spent_cny"] = summary["rework_spent_cny"]
    ledger["cny_per_minute"] = summary["cny_per_minute"]
    ledger["effective_ceiling_cny"] = summary["effective_ceiling_cny"]
    return entry


def summarize_cost_ledger(manifest: ProjectManifest) -> dict[str, Any]:
    """Aggregate spend by stage/shot and evaluate compliance with target CNY/minute."""
    ledger = manifest.cost_ledger if isinstance(manifest.cost_ledger, dict) else {}
    entries = list(ledger.get("entries") or [])
    by_stage: dict[str, float] = {}
    by_shot: dict[str, float] = {}
    spent_cny = 0.0
    rework_spent_cny = 0.0

    for item in entries:
        if not isinstance(item, dict):
            continue
        cost = float(item.get("cost_cny") or 0.0)
        stage = str(item.get("stage") or "unknown")
        shot_id = str(item.get("shot_id") or "project")
        reason = str(item.get("reason") or "pipeline")
        spent_cny += cost
        if reason != "pipeline" or int(item.get("round_index") or 0) > 0:
            rework_spent_cny += cost
        by_stage[stage] = round(by_stage.get(stage, 0.0) + cost, 4)
        by_shot[shot_id] = round(by_shot.get(shot_id, 0.0) + cost, 4)

    spent_cny = round(spent_cny, 4)
    rework_spent_cny = round(rework_spent_cny, 4)
    total_sec = sum(max(0.5, float(getattr(s, "duration", 3.0) or 3.0)) for s in manifest.shots) or 12.0
    minutes_actual = max(total_sec / 60.0, 1.0 / 60.0)
    cny_per_minute = round(spent_cny / minutes_actual, 2)
    budget_limit = round(float(getattr(manifest, "budget_limit_cny", 100.0) or 100.0), 2)
    ceiling_cny = effective_project_budget_cny(manifest)
    remaining_cny = round(max(0.0, ceiling_cny - spent_cny), 4)

    if remaining_cny >= 2.8:
        recommended_max_tier = "i2v_quality"
    elif remaining_cny >= 0.7:
        recommended_max_tier = "i2v_fast"
    elif remaining_cny >= 0.14:
        recommended_max_tier = "image_only"
    else:
        recommended_max_tier = "preview_25d_only"

    codex_calls = sum(
        1
        for item in entries
        if isinstance(item, dict) and str(item.get("provider_or_route") or "") == "image:codex"
    )
    return {
        "schema": "cost-ledger/v1",
        "spent_cny": spent_cny,
        "rework_spent_cny": rework_spent_cny,
        "estimated_duration_seconds": round(total_sec, 2),
        "cny_per_minute": cny_per_minute,
        "budget_limit_cny": budget_limit,
        "effective_ceiling_cny": ceiling_cny,
        "remaining_ceiling_cny": remaining_cny,
        "within_target": cny_per_minute <= budget_limit and spent_cny <= ceiling_cny + 1e-6,
        "by_stage": by_stage,
        "by_shot": by_shot,
        "entry_count": len(entries),
        "recommended_max_tier": recommended_max_tier,
        "execution_strategy": getattr(manifest, "execution_strategy", "interleaved"),
        "codex_quota_telemetry": {
            "image_calls": codex_calls,
            "estimated_quota_burn": codex_calls,
        },
    }


def check_rework_budget(
    manifest: ProjectManifest,
    estimated_cost_cny: float,
    *,
    max_cost_cny: float | None = None,
) -> dict[str, Any]:
    """Fail closed when rework would breach per-call max_cost_cny or project ceiling."""
    cost = round(max(0.0, float(estimated_cost_cny)), 4)
    summary = summarize_cost_ledger(manifest)
    if cost <= 0.0:
        return {
            "allowed": True,
            "estimated_cost_cny": 0.0,
            "remaining_before_cny": summary["remaining_ceiling_cny"],
            "remaining_after_cny": summary["remaining_ceiling_cny"],
        }

    if max_cost_cny is not None and cost > float(max_cost_cny) + 1e-6:
        raise BudgetExceededError(
            f"planned rework cost {cost:.2f} CNY exceeds per-call limit {float(max_cost_cny):.2f} CNY; "
            f"consider fallback_25d_motion (0.00 CNY), rewrite_image_prompt (0.14 CNY), or i2v_fast (0.70 CNY)"
        )

    ceiling = summary["effective_ceiling_cny"]
    spent = summary["spent_cny"]
    if spent + cost > ceiling + 1e-6:
        raise BudgetExceededError(
            f"planned rework cost {cost:.2f} CNY exceeds remaining project budget "
            f"{summary['remaining_ceiling_cny']:.2f} CNY (spent={spent:.2f}, ceiling={ceiling:.2f}); "
            f"use fallback_25d_motion (0.00 CNY) or adjust budget_limit_cny"
        )

    return {
        "allowed": True,
        "estimated_cost_cny": cost,
        "remaining_before_cny": summary["remaining_ceiling_cny"],
        "remaining_after_cny": round(max(0.0, ceiling - (spent + cost)), 4),
    }


def check_and_record_cost(
    manifest: ProjectManifest,
    shot_id: str,
    stage: str,
    model: str,
    round_idx: int = 0,
    *,
    reason: str | None = None,
    max_cost_cny: float | None = None,
    save_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify remaining budget before spending, halt to awaiting_agent_takeover if exceeded, else record entry."""
    cost = estimate_operation_cost(stage, model, manifest=manifest)
    try:
        check_rework_budget(manifest, cost, max_cost_cny=max_cost_cny)
    except BudgetExceededError:
        manifest.status = "awaiting_agent_takeover"
        if save_path is not None:
            manifest.save(save_path)
        raise
    return record_cost_entry(
        manifest,
        shot_id=shot_id,
        stage=stage,
        provider_or_route=model,
        cost_cny=cost,
        reason=reason or ("rework" if round_idx > 0 else "pipeline"),
        round_index=round_idx,
    )
