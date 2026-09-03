"""Explicit runtime assembly for the three sibling agent projects."""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters import StoryCampaignAdapter, StoryImageAdapter, StoryVideoAdapter
from .pipeline import SingleSceneOrchestrator
from .registry import ArtifactRegistry
from .config import OrchestratorConfig


@dataclass(frozen=True)
class RuntimeConfig:
    image_root: Path
    video_root: Path
    artifact_root: Path

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        image = Path(os.environ.get("STORY_IMAGE_AGENT_ROOT", "")).resolve()
        video = Path(os.environ.get("STORY_VIDEO_AGENT_ROOT", "")).resolve()
        artifact = Path(os.environ.get("STORY_MEDIA_ARTIFACT_ROOT", ".artifacts")).resolve()
        if not image.is_dir() or not video.is_dir():
            raise RuntimeError("STORY_IMAGE_AGENT_ROOT and STORY_VIDEO_AGENT_ROOT must be existing directories")
        return cls(image, video, artifact)


def _import_from_root(root: Path, module: str) -> Any:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(module)


def build_runtime(*, story_runner: Callable[..., dict[str, Any]], config: RuntimeConfig | None = None,
                  image_provider: Any | None = None, video_client: Any | None = None) -> SingleSceneOrchestrator:
    """Assemble real sibling adapters without executing a task.

    ``story_runner`` must be supplied by the host's Rust-capability-backed
    campaign runtime. Providers can be injected for tests; otherwise image
    and video providers are constructed from their own environment/config APIs.
    """
    cfg = config or RuntimeConfig.from_environment()
    image_pkg = _import_from_root(cfg.image_root, "story_image_agent")
    video_pkg = _import_from_root(cfg.video_root, "story_video_agent")
    image_workflow = image_pkg.ImagePromptWorkflow("story-media-orchestrator")
    provider = image_provider or image_pkg.DashScopeImageProvider.from_nanocodex_config()
    if image_provider is None:
        provider.model = cfg.models.image_model
        provider.size = provider._normalize_size(cfg.models.image_size)
    video_workflow = video_pkg.VideoPromptWorkflow("story-media-orchestrator")
    comfy = video_client or video_pkg.ComfyUIAdapter.from_environment()
    registry = ArtifactRegistry(cfg.artifact_root)
    return SingleSceneOrchestrator(
        StoryCampaignAdapter(story_runner).run,
        StoryImageAdapter(image_workflow, provider, registry).run,
        StoryVideoAdapter(video_workflow, comfy=comfy, registry=registry, models=cfg.models).run,
    )


def build_runtime_from_environment(*, story_runner: Callable[..., dict[str, Any]],
                                   image_provider: Any | None = None,
                                   video_client: Any | None = None) -> SingleSceneOrchestrator:
    cfg = OrchestratorConfig.from_environment()
    return build_runtime(story_runner=story_runner,
                         config=RuntimeConfig(cfg.image_root, cfg.video_root, cfg.artifact_root),
                         image_provider=image_provider, video_client=video_client)
