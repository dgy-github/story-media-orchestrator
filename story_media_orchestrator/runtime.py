"""Explicit runtime assembly for the three sibling agent projects."""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import url2pathname

from .adapters import HttpStoryCampaignAdapter, StoryCampaignAdapter, StoryImageAdapter, StoryVideoAdapter
from .pipeline import SingleSceneOrchestrator
from .registry import ArtifactRegistry
from .config import ModelConfig, OrchestratorConfig


@dataclass(frozen=True)
class RuntimeConfig:
    image_root: Path
    video_root: Path
    artifact_root: Path
    models: ModelConfig = ModelConfig()

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        image = Path(os.environ.get("STORY_IMAGE_AGENT_ROOT", "")).resolve()
        video = Path(os.environ.get("STORY_VIDEO_AGENT_ROOT", "")).resolve()
        artifact = Path(os.environ.get("STORY_MEDIA_ARTIFACT_ROOT", ".artifacts")).resolve()
        if not image.is_dir() or not video.is_dir():
            raise RuntimeError("STORY_IMAGE_AGENT_ROOT and STORY_VIDEO_AGENT_ROOT must be existing directories")
        return cls(image, video, artifact)


def resolve_sibling_root(repo_name: str, dist_name: str = "", strip_suffix: str = "", module_name: str = "") -> Path:
    """Resolve a sibling repository root from workspace, editable metadata, or installed spec."""
    workspace_root = Path(__file__).resolve().parents[2]
    sibling = workspace_root / repo_name
    if sibling.is_dir():
        return sibling
    if dist_name:
        try:
            dist = importlib.metadata.distribution(dist_name)
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                info = json.loads(direct_url)
                url = info.get("url", "")
                if url.startswith("file://"):
                    candidate = Path(url2pathname(urlparse(url).path))
                    if strip_suffix and candidate.name == strip_suffix:
                        candidate = candidate.parent
                    if candidate.is_dir():
                        return candidate
            for rel_file in dist.files or ():
                if str(rel_file).endswith(".pth"):
                    pth_path = Path(dist.locate_file(rel_file))
                    if pth_path.is_file():
                        for line in pth_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                            line = line.strip()
                            if line and not line.startswith("#") and not line.startswith("import "):
                                candidate = Path(line)
                                if strip_suffix and candidate.name == strip_suffix:
                                    candidate = candidate.parent
                                if candidate.is_dir():
                                    return candidate
        except importlib.metadata.PackageNotFoundError:
            pass
    if module_name:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                candidate = Path(spec.origin).resolve().parents[1]
                if "site-packages" not in candidate.parts and candidate.is_dir():
                    return candidate
        except (ImportError, ValueError, OSError):
            pass
    return sibling


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
    if video_client is None and comfy is not None and hasattr(comfy, "timeout"):
        comfy.timeout = float(os.environ.get("MINIMAX_H3_COMFYUI_TIMEOUT") or "180.0")
    registry = ArtifactRegistry(cfg.artifact_root)
    return SingleSceneOrchestrator(
        StoryCampaignAdapter(story_runner).run,
        StoryImageAdapter(image_workflow, provider, registry).run,
        StoryVideoAdapter(video_workflow, comfy=comfy, registry=registry, models=cfg.models).run,
    )


def build_runtime_from_environment(*, story_runner: Callable[..., dict[str, Any]] | None = None,
                                   image_provider: Any | None = None,
                                   video_client: Any | None = None) -> SingleSceneOrchestrator:
    cfg = OrchestratorConfig.from_environment()
    if story_runner is None:
        if not cfg.story_sidecar_url or not cfg.story_sidecar_token:
            raise RuntimeError("provide story_runner or STORY_SIDECAR_URL/STORY_SIDECAR_TOKEN")
        story_runner = HttpStoryCampaignAdapter(cfg.story_sidecar_url, cfg.story_sidecar_token).run
    return build_runtime(story_runner=story_runner,
                         config=RuntimeConfig(cfg.image_root, cfg.video_root, cfg.artifact_root, cfg.models),
                         image_provider=image_provider, video_client=video_client)
