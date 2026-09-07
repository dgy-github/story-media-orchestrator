"""Provider contracts used by the resumable project workflow."""
from __future__ import annotations

from pathlib import Path
import os
from urllib.error import HTTPError
from typing import Protocol

from .manifest import Shot
from .registry import ArtifactRegistry


class ImageProvider(Protocol):
    def generate(self, shot: Shot, output: Path) -> Path: ...


class VideoProvider(Protocol):
    def generate(self, shot: Shot, image: Path | None, output: Path) -> Path: ...


class TextFrameProvider:
    """Credential-free fallback that keeps a project inspectable."""

    def generate(self, shot: Shot, output: Path) -> Path:
        target = output.with_suffix(".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(shot.text, encoding="utf-8")
        return target


class StoryImageProvider:
    """Materialize images produced by the existing story-image adapter."""

    def __init__(self, adapter, registry: ArtifactRegistry) -> None:
        self.adapter, self.registry = adapter, registry

    @staticmethod
    def _visual_characters(characters):
        descriptions = []
        for character in characters or []:
            if not isinstance(character, dict):
                continue
            fields = [character.get(key) for key in ('name', 'appearance', 'clothing')]
            description = '，'.join(value.strip() for value in fields if isinstance(value, str) and value.strip())
            if description:
                descriptions.append(description)
        return '；'.join(descriptions) or '以动作描述中的人物为准'

    def generate(self, shot: Shot, output: Path) -> Path:
        visual_action = shot.scene_context.get("visual_action") or shot.text
        try:
            result = self.adapter.run(
                {"summary": shot.text, "description": visual_action, "action": visual_action + "。仅绘制当前动作的一张完整画面、一个时间点。不要分屏、拼贴、上下多格、漫画分镜、字幕或文字。",
                 "location": shot.scene_context.get("location", "以动作描述中的地点为准"),
                 "characters": self._visual_characters(shot.scene_context.get("characters")),
                 "lighting": shot.scene_context.get("lighting") or shot.scene_context.get("time") or "以动作描述中的时间与光线为准", "shot_id": shot.id,
                 **{key: shot.scene_context[key] for key in ("framing", "mood", "negative", "continuity") if key in shot.scene_context}},
                shot.scene_context.get("source_spans") or [shot.scene_id, shot.id],
            )
        except Exception as exc:
            cause = exc
            for _ in range(8):
                if isinstance(cause, HTTPError):
                    message = {
                        401: "阿里云 API Key 无效（HTTP 401），请更新 DashScope 凭据后恢复",
                        403: "阿里云拒绝访问（HTTP 403），请检查模型权限及账户状态",
                        429: "阿里云请求限流（HTTP 429），请稍后恢复",
                    }.get(cause.code, f"阿里云请求失败（HTTP {cause.code}）")
                    raise RuntimeError(message) from None
                if cause.__cause__ is None:
                    break
                cause = cause.__cause__
            raise
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.registry.get_bytes(result["first_frame_ref"]))
        return output


def build_project_image_provider(manifest, root: Path) -> StoryImageProvider:
    """Reuse the sibling DashScope adapter without requiring story/video services.

    Model and size are pinned in the manifest on first use. Credentials remain
    in the existing provider's environment/config loader, never in the project.
    """
    from .runtime import _import_from_root
    from .adapters import StoryImageAdapter
    image_root = Path(os.environ.get("STORY_IMAGE_AGENT_ROOT",
                                    str(Path(__file__).resolve().parents[2] / "story-image-agent")))
    pkg = _import_from_root(image_root, "story_image_agent")
    provider = pkg.DashScopeImageProvider.from_nanocodex_config()
    provider.model = manifest.image_model or os.environ.get("STORY_IMAGE_MODEL") or provider.model
    provider.size = provider._normalize_size(
        manifest.image_size or os.environ.get("STORY_IMAGE_SIZE") or provider.size)
    manifest.image_model, manifest.image_size = provider.model, provider.size
    registry = ArtifactRegistry(root / "artifacts")
    adapter = StoryImageAdapter(pkg.ImagePromptWorkflow(manifest.project_id), provider, registry)
    from .cloud_checkpoint import CheckpointedStoryImageProvider
    return CheckpointedStoryImageProvider(adapter, registry, manifest, root)
