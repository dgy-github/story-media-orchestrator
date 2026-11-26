from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import os
from typing import Any
from urllib import request

QUALITY_DECISIONS = ("passed", "warning", "failed", "unavailable")


def evaluate_artifact(artifact: dict, stage: str) -> dict:
    """Call the configured quality model; fail closed when it is unavailable."""
    url = os.environ.get("STORY_QUALITY_EVALUATOR_URL", "").strip()
    if not url:
        return {"schema": "quality-evaluation/v1", "decision": "unavailable", "stage": stage, "reason": "未配置质量模型服务"}
    try:
        body = json.dumps({"stage": stage, "artifact": artifact}, ensure_ascii=False).encode()
        req = request.Request(url, body, {"Content-Type": "application/json"})
        with request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict) or result.get("schema") != "quality-evaluation/v1":
            raise ValueError("quality evaluator returned an incompatible schema")
        if result.get("decision") not in {"passed", "warning", "failed", "unavailable"}:
            raise ValueError("quality evaluator returned an invalid decision")
        if result.get("stage", stage) != stage:
            raise ValueError("quality result belongs to another stage")
        if "reason" in result and not isinstance(result["reason"], str):
            raise ValueError("quality reason must be text")
        if "failures" in result and not isinstance(result["failures"], list):
            raise ValueError("quality failures must be a list")
        return {**result, "stage": stage}
    except Exception as exc:
        return {"schema": "quality-evaluation/v1", "decision": "unavailable", "stage": stage, "reason": f"质量服务不可用: {type(exc).__name__}"}


def build_supervisor_verdict(
    *,
    stage: str,
    decision: str,
    scores: dict[str, float] | None = None,
    failure_code: str = "NONE",
    reason: str = "",
    failures: list[str] | None = None,
    supervisor_id: str = "gemini-4-carbon",
    round_index: int = 0,
    genre: str | None = None,
) -> dict[str, Any]:
    """Build and validate a structured supervisor verdict (quality-evaluation/v2)."""
    clean_stage = (stage or "image").strip().lower()
    if clean_stage not in {"storyboard", "image", "video", "audio", "final"}:
        raise ValueError(f"unsupported stage for quality verdict: {stage}")
    clean_decision = (decision or "").strip().lower()
    if clean_decision not in {"passed", "warning", "failed"}:
        raise ValueError(f"decision must be one of passed, warning, failed; got {decision!r}")

    normalized_scores: dict[str, float] = {}
    if scores is not None:
        if not isinstance(scores, dict):
            raise ValueError("scores must be a dict mapping metric names to [0.0, 1.0]")
        for key, raw_val in scores.items():
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)):
                raise ValueError(f"score {key!r} must be numeric")
            val = float(raw_val)
            if not math.isfinite(val) or not 0.0 <= val <= 1.0:
                raise ValueError(f"score {key!r} must be within [0.0, 1.0]; got {val}")
            normalized_scores[str(key).strip()] = round(val, 4)

    qa_profile_id: str | None = None
    if genre is not None and str(genre).strip():
        from .qa_profiles import compute_weighted_score, get_qa_profile

        profile = get_qa_profile(str(genre))
        qa_profile_id = profile["id"]
        if normalized_scores:
            overall_score = compute_weighted_score(normalized_scores, profile)
        else:
            overall_score = {"passed": 1.0, "warning": 0.65, "failed": 0.25}[clean_decision]
    elif normalized_scores:
        overall_score = round(sum(normalized_scores.values()) / len(normalized_scores), 4)
    else:
        overall_score = {"passed": 1.0, "warning": 0.65, "failed": 0.25}[clean_decision]

    clean_failures: list[str] = []
    if failures is not None:
        if not isinstance(failures, list):
            raise ValueError("failures must be a list of strings")
        clean_failures = [str(item).strip() for item in failures if str(item).strip()]

    clean_code = (failure_code or "NONE").strip().upper()
    if clean_decision == "failed" and clean_code == "NONE":
        clean_code = "OTHER"

    verdict: dict[str, Any] = {
        "schema": "quality-evaluation/v1",
        "schema_version": "quality-evaluation/v2",
        "stage": clean_stage,
        "decision": clean_decision,
        "reason": str(reason or "").strip(),
        "failures": clean_failures,
        "failure_code": clean_code,
        "scores": normalized_scores,
        "overall_score": overall_score,
        "supervisor_id": str(supervisor_id or "gemini-4-carbon").strip() or "gemini-4-carbon",
        "round_index": max(0, int(round_index)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if qa_profile_id is not None:
        verdict["qa_profile"] = qa_profile_id
    return verdict
