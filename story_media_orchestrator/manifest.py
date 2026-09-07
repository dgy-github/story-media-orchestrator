"""Persistent project manifest for resumable preview workflows."""
from __future__ import annotations
import json
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

STATES = ("planned", "generating", "generated", "assembled", "done", "failed")

@dataclass
class Shot:
    id: str
    text: str
    status: str = "planned"
    assets: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    subtitle: str | None = None
    duration: float = 3.0
    scene_id: str = "scene-01"
    review: str = "pending"
    attempts: int = 0
    cache_key: str | None = None
    mode: str = "preview"

@dataclass
class ProjectManifest:
    project_id: str
    story: str
    mode: str = "preview"
    status: str = "planned"
    shots: list[Shot] = field(default_factory=list)
    output: str | None = None
    review_required: bool = True
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["shots"] = [Shot(**shot) for shot in data.get("shots", [])]
        return cls(**data)

    @classmethod
    def create(cls, project_id: str, story: str) -> "ProjectManifest":
        import re
        chunks = [part.strip() for part in re.split(r"[。！？.!?\n]+", story) if part.strip()]
        shots = [Shot(f"shot-{i:02d}", text, scene_id=f"scene-{(i-1)//3+1:02d}") for i, text in enumerate(chunks or [story], 1)]
        return cls(project_id, story, shots=shots)

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id: return shot
        raise KeyError(shot_id)
