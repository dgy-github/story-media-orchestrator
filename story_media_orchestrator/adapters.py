"""Adapters for the independently versioned story, image and video agents."""
from __future__ import annotations

import base64
import inspect
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


class StoryImageAdapter:
    """Bridge story-image-agent workflow/provider and persist image artifacts."""
    def __init__(self, workflow: Any, provider: Any, registry: ArtifactRegistry) -> None:
        self.workflow, self.provider, self.registry = workflow, provider, registry

    def run(self, scene: dict[str, Any], source_spans: list[str], *, quality: dict[str, Any]) -> dict[str, Any]:
        plan = self.workflow.build_production_plan(scene, source_spans, candidate_count=1)
        candidate = plan["candidates"][0]
        generated = self.provider.generate(candidate)
        content = base64.b64decode(generated.get("content_base64", ""), validate=True)
        frame_ref = self.registry.put_bytes(content)
        final = self.workflow.finalize_candidate(plan, candidate["request_id"], quality)
        return {"schema": "image-production-plan/v1", "plan": final,
                "first_frame_ref": frame_ref, "last_frame_ref": frame_ref}


class StoryVideoAdapter:
    """Build a controlled video pipeline and optionally execute it via ComfyUI."""
    def __init__(self, workflow: Any, *, comfy: Any = None, registry: ArtifactRegistry | None = None) -> None:
        self.workflow, self.comfy, self.registry = workflow, comfy, registry

    def run(self, *, image_ref: str, story_spans: list[str], shot: dict[str, Any],
            action_unit: dict[str, Any] | None = None, prompt: str | None = None) -> dict[str, Any]:
        pipeline = self.workflow.build_pipeline_v2(
            image_ref, story_spans, shot, coarse_duration_seconds=5,
            action_unit=action_unit,
            quality={"story_alignment": .9, "identity_consistency": .9,
                     "motion_quality": .9, "continuity": .9, "artifact_free": .9},
        )
        result: dict[str, Any] = {"schema": pipeline["schema"], "pipeline": pipeline, "status": "planned"}
        if self.comfy is not None:
            if not prompt:
                raise ValueError("prompt is required for ComfyUI execution")
            from story_video_agent import build_minimax_h3_workflow
            result["execution"] = self.comfy.run_to_artifact(build_minimax_h3_workflow(prompt=prompt), self.registry)
            result["status"] = result["execution"]["state"]
        return result
