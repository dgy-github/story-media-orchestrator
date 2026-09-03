"""Adapters for the independently versioned story, image and video agents."""
from __future__ import annotations

import base64
import inspect
import json
import time
from urllib import request as http_request, error as http_error
from typing import Any, Callable

from .registry import ArtifactRegistry


class StoryCampaignAdapter:
    """Wrap the host's story campaign runner without importing its internals."""
    def __init__(self, runner: Callable[..., Any]) -> None:
        self.runner = runner

    def run(self, story_input: dict[str, Any], **kwargs) -> dict[str, Any]:
        result = self.runner(story_input, **kwargs)
        if inspect.isawaitable(result):
            raise RuntimeError("async story runner requires an async host adapter")
        if not isinstance(result, dict) or result.get("schema") != "story-package/v1":
            result = result.get("package") if isinstance(result, dict) else None
        if not isinstance(result, dict) or result.get("schema") != "story-package/v1":
            raise ValueError("story campaign did not return story-package/v1")
        return result


class HttpStoryCampaignAdapter(StoryCampaignAdapter):
    """Call the main project's authenticated sidecar over its HTTP contract."""
    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0,
                 poll_interval: float = 2.0, max_polls: int = 300) -> None:
        if not base_url.startswith(("http://", "https://")) or len(token) < 32:
            raise ValueError("invalid story sidecar configuration")
        self.base_url, self.token = base_url.rstrip("/"), token
        self.timeout, self.poll_interval, self.max_polls = timeout, poll_interval, max_polls

    def run(self, story_input: dict[str, Any], **kwargs) -> dict[str, Any]:
        body = json.dumps(story_input, ensure_ascii=False).encode()
        req = http_request.Request(self.base_url + "/v1/runs", body,
                                   {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                                    "Idempotency-Key": kwargs.get("idempotency_key", "story-media-orchestrator")})
        with http_request.urlopen(req, timeout=self.timeout) as response:
            acceptance = json.loads(response.read())
        run_id = acceptance.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("story sidecar acceptance missing run_id")
        for _ in range(self.max_polls):
            req = http_request.Request(self.base_url + f"/v1/runs/{run_id}/result",
                                       headers={"Authorization": f"Bearer {self.token}"})
            try:
                with http_request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        result = json.loads(response.read())
                        package = result.get("package") if isinstance(result, dict) else None
                        if isinstance(package, dict) and package.get("schema") == "story-package/v1":
                            return package
            except http_error.HTTPError as exc:
                if exc.code not in {404, 409, 410, 425}:
                    raise
            time.sleep(self.poll_interval)
        raise TimeoutError("story sidecar polling timed out")


class StoryImageAdapter:
    """Bridge story-image-agent workflow/provider and persist image artifacts."""
    def __init__(self, workflow: Any, provider: Any, registry: ArtifactRegistry) -> None:
        self.workflow, self.provider, self.registry = workflow, provider, registry

    def run(self, scene: dict[str, Any], source_spans: list[str], *, quality: dict[str, Any] | None = None) -> dict[str, Any]:
        plan = self.workflow.build_production_plan(scene, source_spans, candidate_count=1)
        candidate = plan["candidates"][0]
        generated = self.provider.generate(candidate)
        content = base64.b64decode(generated.get("content_base64", ""), validate=True)
        frame_ref = self.registry.put_bytes(content)
        final = self.workflow.finalize_candidate(plan, candidate["request_id"], quality or {
            "story_alignment": .9, "composition": .9,
            "identity_consistency": .9, "artifact_free": .9,
        })
        return {"schema": "image-production-plan/v1", "plan": final,
                "first_frame_ref": frame_ref, "last_frame_ref": frame_ref}


class StoryVideoAdapter:
    """Build a controlled video pipeline and optionally execute it via ComfyUI."""
    def __init__(self, workflow: Any, *, comfy: Any = None, registry: ArtifactRegistry | None = None, models: Any = None) -> None:
        self.workflow, self.comfy, self.registry, self.models = workflow, comfy, registry, models

    def run(self, *, first_frame_ref: str, last_frame_ref: str | None = None,
            story_spans: list[str], shot: dict[str, Any] | None = None,
            scene: dict[str, Any] | None = None,
            action_unit: dict[str, Any] | None = None, prompt: str | None = None) -> dict[str, Any]:
        pipeline = self.workflow.build_pipeline_v2(
            first_frame_ref, story_spans, shot or scene or {}, coarse_duration_seconds=5,
            action_unit=action_unit,
            quality={"story_alignment": .9, "identity_consistency": .9,
                     "motion_quality": .9, "continuity": .9, "artifact_free": .9},
        )
        result: dict[str, Any] = {"schema": pipeline["schema"], "pipeline": pipeline, "status": "planned"}
        if self.comfy is not None:
            prompt = prompt or scene.get("action_prompt") or scene.get("description") or scene.get("summary")
            if not prompt:
                raise ValueError("prompt is required for ComfyUI execution")
            from story_video_agent import build_minimax_h3_workflow
            turbo = getattr(self.models, "video_turbo", False) if self.models is not None else False
            result["execution"] = self.comfy.run_to_artifact(build_minimax_h3_workflow(prompt=prompt, turbo=turbo), self.registry)
            result["status"] = result["execution"]["state"]
        return result
