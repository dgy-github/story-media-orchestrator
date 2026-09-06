"""Persistent project manifest for resumable preview workflows."""
from __future__ import annotations
import json
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

@dataclass
class ProjectManifest:
    project_id: str
    story: str
    mode: str = "preview"
    status: str = "planned"
    shots: list[Shot] = field(default_factory=list)
    output: str | None = None

    def save(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ProjectManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["shots"] = [Shot(**shot) for shot in data.get("shots", [])]
        return cls(**data)

    @classmethod
    def create(cls, project_id: str, story: str) -> "ProjectManifest":
        chunks = [part.strip() for part in story.replace("。", ".").split(".") if part.strip()]
        return cls(project_id, story, shots=[Shot(f"shot-{i:02d}", text) for i, text in enumerate(chunks or [story], 1)])

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id: return shot
        raise KeyError(shot_id)
