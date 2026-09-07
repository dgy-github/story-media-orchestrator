"""Provider contracts used by the resumable project workflow."""
from __future__ import annotations

from pathlib import Path
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

    def generate(self, shot: Shot, output: Path) -> Path:
        result = self.adapter.run(
            {"summary": shot.text, "description": shot.text, "shot_id": shot.id},
            [shot.scene_id, shot.id],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.registry.get_bytes(result["first_frame_ref"]))
        return output
