"""Minimal single-scene orchestration across the three independent agents."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class OrchestrationError(ValueError):
    """A contract or stage failure that stops the media run."""


def _require_schema(value: Any, schema: str, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise OrchestrationError(f"{label} must be {schema}")
    return value


class SingleSceneOrchestrator:
    """Run one story scene through image planning and video planning.

    Stage implementations are injected so this package never imports or merges
    the three agent repositories. Each callable is expected to be deterministic
    in contract tests and may be replaced by real providers by the host runtime.
    """

    def __init__(self, story_agent: Callable[..., dict[str, Any]],
                 image_agent: Callable[..., dict[str, Any]],
                 video_agent: Callable[..., dict[str, Any]]) -> None:
        self.story_agent = story_agent
        self.image_agent = image_agent
        self.video_agent = video_agent

    def run(self, *, story_input: dict[str, Any], scene_index: int = 0,
            image_quality: dict[str, float] | None = None,
            video_quality: dict[str, float] | None = None) -> dict[str, Any]:
        story = _require_schema(self.story_agent(story_input), "story-package/v1", "story output")
        scenes = story.get("scenes")
        if not isinstance(scenes, list) or not 0 <= scene_index < len(scenes):
            raise OrchestrationError("story package has no requested scene")
        scene = scenes[scene_index]
        if not isinstance(scene, dict):
            raise OrchestrationError("story scene must be an object")
        spans = scene.get("source_spans") or scene.get("spans")
        if not isinstance(spans, list) or not spans:
            raise OrchestrationError("scene must carry source spans")

        image_plan = _require_schema(
            self.image_agent(scene=scene, source_spans=spans),
            "image-production-plan/v1", "image output",
        )
        if image_quality is not None:
            image_plan = self.image_agent(plan=image_plan, quality=image_quality)
            _require_schema(image_plan, "image-production-plan/v1", "final image output")
        refs = image_plan.get("final_artifacts") or image_plan.get("artifacts")
        if not isinstance(refs, list) or len(refs) < 2 or not all(isinstance(r, str) for r in refs[:2]):
            raise OrchestrationError("image plan must provide first and last frame artifacts")

        video_plan = _require_schema(
            self.video_agent(scene=scene, source_spans=spans,
                             first_frame_ref=refs[0], last_frame_ref=refs[-1]),
            "video-generation-pipeline/v2", "video output",
        )
        if video_quality is not None:
            video_plan = self.video_agent(plan=video_plan, quality=video_quality)
            _require_schema(video_plan, "video-generation-pipeline/v2", "final video output")
        return {
            "schema": "story-media-run/v1",
            "status": "succeeded",
            "scene_index": scene_index,
            "story": story,
            "image_plan": image_plan,
            "video_plan": video_plan,
        }
