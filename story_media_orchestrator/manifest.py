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
    motion: str = 'none'
    transition: str = 'cut'
    candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_batch: str | None = None

    image_task: dict[str, str] = field(default_factory=dict)
    scene_context: dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectManifest:
    project_id: str
    story: str
    mode: str = "preview"
    status: str = "planned"
    shots: list[Shot] = field(default_factory=list)
    output: str | None = None
    review_required: bool = True
    review_enforced: bool = False
    approvals: dict[str, str] = field(default_factory=dict)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    real_preview: bool = False
    tts_source: str = 'silent'
    image_source: str = "local"
    image_model: str | None = None
    image_size: str | None = None
    video_provider: str = "comfyui"
    video_options: dict[str, Any] = field(default_factory=dict)
    story_package: dict[str, Any] | None = None
    story_job: dict[str, Any] | None = None

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

    @classmethod
    def from_story_package(cls, project_id: str, package: dict[str, Any]) -> "ProjectManifest":
        if not isinstance(package, dict) or package.get("schema") != "story-package/v1":
            raise ValueError("expected story-package/v1")
        scenes = package.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError("story package requires scenes")
        expanded = []
        for scene_index, scene in enumerate(scenes, 1):
            if not isinstance(scene, dict):
                raise ValueError("invalid scene")
            lines = scene.get("lines", [])
            if not isinstance(lines, list):
                raise ValueError("scene lines must be a list")
            groups, current = [], []
            for line in lines:
                if not isinstance(line, dict) or not isinstance(line.get("text"), str) or not line["text"].strip():
                    continue
                if line.get("kind") == "action" and current:
                    groups.append(current)
                    current = []
                current.append(line)
            if current:
                groups.append(current)
            for group in groups or [[]]:
                expanded.append({**scene, "node_id": scene.get("node_id") or f"scene-{scene_index:02d}", "lines": group, "_cast_lines": lines})
        shots = []
        for i, scene in enumerate(expanded, 1):
            if not isinstance(scene, dict):
                raise ValueError("invalid scene")
            lines = scene.get("lines", [])
            text = " ".join(line["text"].strip() for line in lines
                            if isinstance(line, dict) and isinstance(line.get("text"), str) and line["text"].strip())
            text = text or scene.get("summary", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("scene requires text")
            # Keep scene cast across beats, including later pronoun-only actions.
            cast_lines = scene["_cast_lines"]
            speakers = {line["speaker"].rsplit("/", 1)[-1] for line in cast_lines
                        if isinstance(line, dict) and isinstance(line.get("speaker"), str)}
            actions = " ".join(line.get("text", "") for line in cast_lines
                               if isinstance(line, dict) and line.get("kind") == "action"
                               and isinstance(line.get("text"), str))
            def named_in_action(character):
                import re
                name = character.get("name")
                if not isinstance(name, str) or not name.strip():
                    return False
                # Avoid matching e.g. Ann inside Anna. Chinese names have no
                # whitespace boundary; preserve this as an explicit heuristic.
                pattern = re.escape(name.strip())
                if name.isascii():
                    pattern = r"(?<!\w)" + pattern + r"(?!\w)"
                return re.search(pattern, actions) is not None
            characters = [character for character in package.get("characters", [])
                          if isinstance(character, dict) and (character.get("node_id") in speakers or named_in_action(character))]
            context = {key: scene[key] for key in ("location", "lighting", "time", "episode_ref", "framing", "mood", "negative", "continuity") if key in scene}
            visual_action = " ".join(line["text"].strip() for line in lines
                                     if isinstance(line, dict) and line.get("kind") == "action"
                                     and isinstance(line.get("text"), str) and line["text"].strip())
            if visual_action:
                context["visual_action"] = visual_action
            if isinstance(scene.get("node_id"), str) and scene["node_id"]:
                scene_ref = "story-package/" + scene["node_id"]
                context["source_spans"] = [scene_ref] + [scene_ref + "/" + line["node_id"]
                    for line in lines if isinstance(line, dict) and isinstance(line.get("node_id"), str) and line["node_id"]]
            if characters:
                context["characters"] = characters
                context["character_sources"] = {character["node_id"]:
                    "speaker_reference" if character["node_id"] in speakers else "action_name_match"
                    for character in characters if isinstance(character.get("node_id"), str)}
            shots.append(Shot(f"shot-{i:02d}", text.strip(), scene_id=scene.get("node_id") or f"scene-{i:02d}", scene_context=context))
        return cls(project_id, "\n".join(s.text for s in shots), shots=shots, story_package=package)

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id: return shot
        raise KeyError(shot_id)
