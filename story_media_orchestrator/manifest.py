"""Persistent project manifest for resumable preview workflows."""
from __future__ import annotations
import json
import re
from uuid import uuid4
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any

STATES = ("planned", "generating", "generated", "assembled", "done", "partial", "awaiting_agent_takeover", "failed")


def _strip_outer_quotes(text: str) -> str:
    pairs = (("“", "”"), ("「", "」"), ('"', '"'), ("'", "'"))
    stripped = text.strip()
    for left, right in pairs:
        if stripped.startswith(left) and stripped.endswith(right) and len(stripped) >= 2:
            return stripped[1:-1].strip()
    for left, right in pairs:
        if stripped.startswith(left) and right not in stripped[1:]:
            stripped = stripped[1:].strip()
        if stripped.endswith(right) and left not in stripped[:-1]:
            stripped = stripped[:-1].strip()
    return stripped


_PRONOUN_STOP_SPEAKERS = {
    "他", "她", "它", "我", "你", "他们", "她们", "众人", "大家", "有人",
    "旁人", "行人", "人们", "一人", "两人", "二人", "心中", "心里", "暗自",
    "旁白", "画外音", "声音", "耳边", "听说", "据说", "常说", "只听",
}
_ADDRESSEE_SPEECH_RE = re.compile(
    r"(?:^|[，,、\s])(?:于是|随后|接着|只见|却见)?(?P<speaker>[\u4e00-\u9fff]{2,4}?)"
    r"(?:转身|抬头|低头|停步|上前|停下脚步){0,2}(?:对|向|朝|冲|望着|看着)[\u4e00-\u9fff]{1,6}?"
    r"(?:不禁|连忙|急忙|冷冷|轻声|低声|大声|沉声|笑着|微笑着|焦急地|平静地){0,2}"
    r"(?:说道|喊道|问道|答道|叫道|喝道|笑道|叹道|低语道|喃喃道|回答道|质问道|惊呼道|开口道|"
    r"轻声道|低声道|大声道|沉声道|冷声道|接口道|接着道|急道|忙道|说|问|喊|答)"
    r"\s*[：:,，]?\s*$"
)
_DIRECT_SPEECH_RE = re.compile(
    r"(?:^|[，,、\s])(?:于是|随后|接着|只见|却见)?(?P<speaker>[\u4e00-\u9fff]{2,5}?)"
    r"(?:转身|抬头|低头|停步|上前){0,2}"
    r"(?:不禁|连忙|急忙|忽然|突然|微微|冷冷|轻声|低声|大声|沉声|笑着|微笑着|焦急地|愤怒地|平静地){0,2}"
    r"(?:说道|喊道|问道|答道|叫道|喝道|笑道|叹道|低语道|喃喃道|回答道|质问道|惊呼道|开口道|"
    r"轻声道|低声道|大声道|沉声道|冷声道|接口道|接着道|急道|忙道|说|问|喊|答|低语|喃喃|惊呼|质问|回答)"
    r"\s*[：:,，]?\s*$"
)


_SPEECH_VERB_SUFFIX_RE = re.compile(
    r"(?:说道|喊道|问道|答道|叫道|喝道|笑道|叹道|低语道|喃喃道|回答道|质问道|惊呼道|开口道|"
    r"轻声道|低声道|大声道|沉声道|冷声道|接口道|接着道|急道|忙道|说|问|喊|答|低语|喃喃|惊呼|质问|回答)$"
)


def _infer_quote_speaker(prefix: str) -> str:
    clause = re.split(r"[。！？；\n]", prefix)[-1].strip()
    if not clause:
        return "角色"
    for pattern in (_ADDRESSEE_SPEECH_RE, _DIRECT_SPEECH_RE):
        match = pattern.search(clause)
        if not match:
            continue
        cand = match.group("speaker").strip()
        if (
            cand
            and cand not in _PRONOUN_STOP_SPEAKERS
            and not cand.startswith(("对", "向", "朝", "冲", "把", "被", "在", "和", "与"))
            and not cand.endswith(("不", "没", "都", "还", "却"))
        ):
            return cand
    return "角色"


def _parse_inline_dialogue(text: str) -> tuple[str | None, list[dict[str, str]]]:
    """Extract optional inline speaker dialogue while leaving shot.text intact."""
    cleaned = text.strip()
    if not cleaned:
        return None, []
    labeled = re.match(r"^(?P<speaker>[\u4e00-\u9fffA-Za-z0-9_·-]{1,16})\s*[:：]\s*(?P<utterance>.+)$", cleaned)
    if labeled:
        speaker = labeled.group("speaker").strip()
        raw_utterance = labeled.group("utterance").strip()
        utterance = _strip_outer_quotes(raw_utterance)
        has_inline_quote_tail = bool(re.search(r'[“"「][^”"」]+[”"」].*\S', raw_utterance))
        is_speech_attribution = bool(_SPEECH_VERB_SUFFIX_RE.search(speaker) and re.search(r'[“"「]', cleaned))
        if not is_speech_attribution and not has_inline_quote_tail:
            if speaker in {"旁白", "画外音", "narrator", "Narrator"}:
                return utterance or cleaned, []
            if utterance:
                return None, [{"speaker": speaker, "text": utterance}]
    quote_pattern = r'[“"「]([^”"」]+)(?:[”"」]|$)'
    quote_matches = [m for m in re.finditer(quote_pattern, cleaned) if m.group(1).strip()]
    if quote_matches:
        narrated = re.sub(quote_pattern, "", cleaned).strip(" ：:,，")
        dialogue: list[dict[str, str]] = []
        prev_end = 0
        for match in quote_matches:
            speaker = _infer_quote_speaker(cleaned[prev_end:match.start()])
            dialogue.append({"speaker": speaker, "text": match.group(1).strip()})
            prev_end = match.end()
        return (narrated or None), dialogue
    return cleaned, []

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
    narration: str | None = None
    dialogue_lines: list[dict[str, str]] = field(default_factory=list)
    sub_captions: list[dict[str, Any]] = field(default_factory=list)
    voice: str | None = None
    video_route: str | None = None
    character_locks: list[str] = field(default_factory=list)
    prompt_override: str | None = None
    negative_override: str | None = None
    seed_override: int | None = None
    quality_report: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    rework_history: list[dict[str, Any]] = field(default_factory=list)

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
    aspect_ratio: str = "16:9"
    scale_mode: str = "contain"
    alignment_mode: str = "pad"
    subtitle_style: str = "soft"
    speech_mode: str = "auto"
    voice_cast: dict[str, str] = field(default_factory=dict)
    bgm_source: str | None = None
    bgm_volume: float = 0.22
    bgm_ducking: bool = True
    jianying_draft: str | None = None
    character_bible: dict[str, dict[str, Any]] = field(default_factory=dict)
    consistency_mode: str = "auto"
    style_preset: str | None = None
    genre: str = "general"
    video_routing_policy: str = "manual"
    video_fallback_provider: str | None = None
    budget_limit_cny: float = 100.0
    cost_ledger: dict[str, Any] = field(default_factory=dict)
    supervisor_policy: dict[str, Any] = field(default_factory=dict)
    flywheel_path: str | None = None
    auto_rework: bool = False
    max_rework_rounds: int = 2
    relay_tail_frames: bool = False
    execution_strategy: str = "interleaved"
    lipsync_provider: str = "off"
    auto_split_shots: bool = False
    max_shot_seconds: float = 4.8

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
        shot_keys = {f.name for f in fields(Shot)}
        manifest_keys = {f.name for f in fields(cls)}
        data["shots"] = [Shot(**{k: v for k, v in shot.items() if k in shot_keys}) for shot in data.get("shots", [])]
        return cls(**{k: v for k, v in data.items() if k in manifest_keys})

    @classmethod
    def create(
        cls,
        project_id: str,
        story: str,
        *,
        auto_split_shots: bool = False,
        max_shot_seconds: float = 4.8,
    ) -> "ProjectManifest":
        chunks = [part.strip() for part in re.split(r"(?:[。！？.!?]+[”\"」]?|\n)+", story) if part.strip()]
        shots = []
        for i, text in enumerate(chunks or [story], 1):
            narration, dialogue_lines = _parse_inline_dialogue(text)
            shots.append(Shot(
                f"shot-{i:02d}",
                text,
                scene_id=f"scene-{(i-1)//3+1:02d}",
                narration=narration,
                dialogue_lines=dialogue_lines,
            ))
        manifest = cls(
            project_id,
            story,
            shots=shots,
            auto_split_shots=bool(auto_split_shots),
            max_shot_seconds=float(max_shot_seconds),
        )
        if auto_split_shots:
            from .lipsync import split_long_dialogue_shots

            split_long_dialogue_shots(manifest, max_shot_seconds=float(max_shot_seconds))
        return manifest

    @classmethod
    def from_story_package(
        cls,
        project_id: str,
        package: dict[str, Any],
        *,
        auto_split_shots: bool = False,
        max_shot_seconds: float = 4.8,
    ) -> "ProjectManifest":
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
            char_names = {
                c["node_id"]: (c.get("name") or c["node_id"])
                for c in package.get("characters", [])
                if isinstance(c, dict) and isinstance(c.get("node_id"), str)
            }
            narration_parts: list[str] = []
            dialogue_lines: list[dict[str, str]] = []
            for line in lines:
                if not isinstance(line, dict) or not isinstance(line.get("text"), str) or not line["text"].strip():
                    continue
                line_text = line["text"].strip()
                kind = str(line.get("kind") or "").lower()
                raw_speaker = line.get("speaker")
                speaker_id = raw_speaker.rsplit("/", 1)[-1] if isinstance(raw_speaker, str) and raw_speaker.strip() else ""
                if kind == "action" or (kind == "narration" and not speaker_id):
                    narration_parts.append(line_text)
                else:
                    speaker_name = char_names.get(speaker_id, speaker_id) or "旁白"
                    if speaker_name == "旁白" and kind != "dialogue":
                        narration_parts.append(line_text)
                    else:
                        dialogue_lines.append({"speaker": speaker_name, "text": line_text})
            narration_text = (" ".join(narration_parts) if lines else text.strip()) or None
            shots.append(Shot(
                f"shot-{i:02d}",
                text.strip(),
                scene_id=scene.get("node_id") or f"scene-{i:02d}",
                scene_context=context,
                narration=narration_text,
                dialogue_lines=dialogue_lines,
            ))
        manifest = cls(
            project_id,
            "\n".join(s.text for s in shots),
            shots=shots,
            story_package=package,
            auto_split_shots=bool(auto_split_shots),
            max_shot_seconds=float(max_shot_seconds),
        )
        if auto_split_shots:
            from .lipsync import split_long_dialogue_shots

            split_long_dialogue_shots(manifest, max_shot_seconds=float(max_shot_seconds))
        return manifest

    def shot(self, shot_id: str) -> Shot:
        for shot in self.shots:
            if shot.id == shot_id: return shot
        raise KeyError(shot_id)
