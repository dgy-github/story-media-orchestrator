"""Unified, secret-safe model configuration for the orchestrator."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    story_model: str = "configured-by-story-runtime"
    image_provider: str = "dashscope"
    image_model: str = "wan2.2-t2i-flash"
    image_size: str = "720*1280"
    video_provider: str = "comfyui"
    video_model: str = "minimax_h3_fl2va"
    video_turbo: bool = False
    video_steps: int = 20

@dataclass(frozen=True)
class OrchestratorConfig:
    image_root: Path
    video_root: Path
    artifact_root: Path
    comfyui_base_url: str | None
    comfyui_token: str | None
    story_sidecar_url: str | None
    story_sidecar_token: str | None
    models: ModelConfig

    @classmethod
    def from_environment(cls) -> "OrchestratorConfig":
        image = Path(os.environ.get("STORY_IMAGE_AGENT_ROOT", "")).resolve()
        video = Path(os.environ.get("STORY_VIDEO_AGENT_ROOT", "")).resolve()
        if not image.is_dir() or not video.is_dir():
            raise RuntimeError("STORY_IMAGE_AGENT_ROOT and STORY_VIDEO_AGENT_ROOT must be existing directories")
        turbo = os.environ.get("STORY_VIDEO_TURBO", "false").lower()
        if turbo not in {"true", "false"}:
            raise ValueError("STORY_VIDEO_TURBO must be true or false")
        steps = int(os.environ.get("STORY_VIDEO_STEPS", "8" if turbo == "true" else "20"))
        if not 1 <= steps <= 100:
            raise ValueError("STORY_VIDEO_STEPS must be between 1 and 100")
        models = ModelConfig(
            story_model=os.environ.get("STORY_MODEL", "configured-by-story-runtime"),
            image_provider=os.environ.get("STORY_IMAGE_PROVIDER", "dashscope"),
            image_model=os.environ.get("STORY_IMAGE_MODEL", "wan2.2-t2i-flash"),
            image_size=os.environ.get("STORY_IMAGE_SIZE", "720*1280"),
            video_provider=os.environ.get("STORY_VIDEO_PROVIDER", "comfyui"),
            video_model=os.environ.get("STORY_VIDEO_MODEL", "minimax_h3_fl2va"),
            video_turbo=turbo == "true", video_steps=steps,
        )
        return cls(image, video, Path(os.environ.get("STORY_MEDIA_ARTIFACT_ROOT", ".artifacts")).resolve(),
                   os.environ.get("MINIMAX_H3_COMFYUI_BASE_URL"),
                   os.environ.get("MINIMAX_H3_COMFYUI_TOKEN"),
                   os.environ.get("STORY_SIDECAR_URL"), os.environ.get("STORY_SIDECAR_TOKEN"), models)
