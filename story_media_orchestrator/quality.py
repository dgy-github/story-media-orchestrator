from __future__ import annotations
import json, os
from urllib import request

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
