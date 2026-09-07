"""Provider contracts used by the resumable project workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .manifest import Shot


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

