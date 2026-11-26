"""Asset contract, pre-flight linter, and ComfyUI receipt verification for MiniMax H3 I2VA bundles."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageStat

from .lipsync import is_minimax_h3_grid_length

H3_BUNDLE_SCHEMA = "h3-chapter-bundle/v1"
H3_DEFAULT_MAX_FRAMES = 175
VALID_DIALOGUE_MODES = frozenset({"visible", "on_screen_dialogue", "off_screen_voiceover"})

REQUIRED_SHOT_KEYS = ("id", "title", "frames", "first_frame", "h3_prompt", "action", "audio")

_LANG_MARKER_RE = re.compile(r"^\s*\[(?:Chinese|Mandarin|English|zh|en)\]\s*", re.IGNORECASE)
_CLOSED_LIPS_RE = re.compile(
    r"lips?\s+(?:remain|stay|kept)\s+closed|closed\s+lips?|muzzle\s+(?:remains?|stays?)\s+closed|mouth\s+(?:remains?|stays?)\s+closed",
    re.IGNORECASE,
)
_COLOR_TRAP_RE = re.compile(
    r"(?:replaces?|replaced\s+by|turns?\s+into|shifts?\s+to)[^.\n]{0,80}\b(?:gold|golden|crimson|red|white|silver|blue)\b[^.\n]{0,80}\b(?:robe|cloak|coat|garment|sleeve|armor)\b"
    r"|(?:robe|cloak|coat|garment|sleeve|armor)[^.\n]{0,80}(?:replaces?|replaced\s+by)",
    re.IGNORECASE,
)

SILENT_SPEECH_TRIGGER_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("paper_whisper", re.compile(r"\bpaper\s+whisper\b", re.IGNORECASE), "paper rustle"),
    (
        "indistinct_conversation",
        re.compile(r",?\s*(?:and\s+)?low\s+indistinct\s+conversation(?:\s+without\s+intelligible\s+words)?", re.IGNORECASE),
        "",
    ),
    (
        "indistinct_conversation_bare",
        re.compile(r"\bindistinct\s+conversation\b", re.IGNORECASE),
        "",
    ),
    (
        "without_intelligible_speech",
        re.compile(r",?\s*without\s+intelligible\s+(?:words|speech|dialogue)", re.IGNORECASE),
        "",
    ),
    (
        "leave_room_for_line",
        re.compile(r",?\s*(?:soft\s+enough\s+to\s+)?leave\s+room\s+for\s+the\s+next\s+line", re.IGNORECASE),
        "",
    ),
    ("murmuring_voices", re.compile(r"\bmurmuring\s+voices?\b", re.IGNORECASE), ""),
    ("whispered_words", re.compile(r"\bwhispered\s+words?\b", re.IGNORECASE), ""),
)


@dataclass(frozen=True)
class LintIssue:
    """Single validation error or warning produced during H3 bundle inspection."""

    severity: str
    code: str
    shot_id: str | None
    message: str
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "shot_id": self.shot_id,
            "message": self.message,
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass
class BundleLintReport:
    """Structured pre-flight report for an H3 chapter bundle directory."""

    schema: str
    bundle_dir: str
    chapter: str | None
    title: str | None
    ok: bool
    errors: list[LintIssue] = field(default_factory=list)
    warnings: list[LintIssue] = field(default_factory=list)
    shot_count: int = 0
    bundle_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bundle_dir": self.bundle_dir,
            "chapter": self.chapter,
            "title": self.title,
            "ok": self.ok,
            "shot_count": self.shot_count,
            "bundle_fingerprint": self.bundle_fingerprint,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
        }


def slow_breathing_frames(
    script_frames: int,
    *,
    step: int = 17,
    max_frames: int = H3_DEFAULT_MAX_FRAMES,
) -> int:
    """Compute slow-breathing extended shot frame length clamped to the H3 17k+5 grid."""
    if isinstance(script_frames, bool) or not isinstance(script_frames, int):
        raise ValueError(f"script_frames must be an int, got {script_frames!r}")
    if not is_minimax_h3_grid_length(script_frames):
        raise ValueError(f"script_frames {script_frames} is not on the MiniMax H3 17k+5 grid")
    if step % 17 != 0:
        raise ValueError(f"step {step} must be a multiple of 17 to preserve the 17k+5 grid")
    target = min(int(max_frames), int(script_frames) + int(step))
    if not is_minimax_h3_grid_length(target):
        raise ValueError(f"extended target {target} is not on the MiniMax H3 17k+5 grid")
    return target


def extract_dialogue_tags(h3_prompt: str) -> list[str]:
    """Extract dialogue strings from `<d>...</d>` tags with optional language markers stripped."""
    raw_tags = re.findall(r"<d>(.*?)</d>", str(h3_prompt or ""), flags=re.DOTALL)
    return [_LANG_MARKER_RE.sub("", tag).strip() for tag in raw_tags]


def normalize_dialogue_text(text: str) -> str:
    """Normalize dialogue text to CJK/alphanumeric characters for exact drift comparison."""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(text or ""))


def _split_overall_soundscape(h3_prompt: str) -> tuple[str, str, str, str] | None:
    if "overall_soundscape:" not in h3_prompt:
        return None
    head, rest = h3_prompt.split("overall_soundscape:", 1)
    if "non_diegetic_music:" in rest:
        sc, tail = rest.split("non_diegetic_music:", 1)
        return head, sc, "non_diegetic_music:", tail
    return head, rest, "", ""


def sanitize_silent_soundscape(h3_prompt: str, *, has_dialogue: bool) -> tuple[str, list[str]]:
    """Strip speech-trigger phrases in `overall_soundscape:` for non-dialogue shots."""
    if has_dialogue or not isinstance(h3_prompt, str):
        return h3_prompt, []
    split = _split_overall_soundscape(h3_prompt)
    if split is None:
        return h3_prompt, []

    head, sc, tail_prefix, tail = split
    cleaned = sc.strip()
    applied: list[str] = []
    for rule_name, pattern, repl in SILENT_SPEECH_TRIGGER_PATTERNS:
        new_val, n = pattern.subn(repl, cleaned)
        if n > 0:
            cleaned = new_val
            applied.append(rule_name)

    if not applied:
        return h3_prompt, []

    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,])", r"\1", cleaned)
    cleaned = cleaned.strip().rstrip(",")
    if cleaned and not cleaned.endswith("."):
        cleaned += "."

    if tail_prefix:
        rebuilt = f"{head}overall_soundscape: {cleaned}\n\n{tail_prefix}{tail}"
    else:
        rebuilt = f"{head}overall_soundscape: {cleaned}"
    return rebuilt, applied


def compute_shot_input_fingerprint(
    h3_prompt: str,
    first_frame: Path,
    *,
    target_frames: int,
    width: int = 1024,
    height: int = 576,
    steps: int = 20,
    turbo: bool = False,
    resolution: str = "1080P",
) -> str:
    """Compute a deterministic 20-char SHA256 fingerprint for shot render inputs."""
    h = hashlib.sha256()
    h.update(str(h3_prompt).strip().encode("utf-8"))
    h.update(
        f"|len={target_frames}|w={width}|h={height}|steps={steps}|turbo={int(bool(turbo))}|res={resolution}|".encode(
            "utf-8"
        )
    )
    h.update(Path(first_frame).read_bytes())
    return h.hexdigest()[:20]


def _lint_image(img_path: Path, shot_id: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    if not img_path.is_file():
        return [LintIssue("error", "MISSING_KEYFRAME", shot_id, f"Missing keyframe image: {img_path}")]
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            mean_luma = float(ImageStat.Stat(img.convert("L")).mean[0])
    except Exception as exc:
        return [LintIssue("error", "CORRUPT_KEYFRAME", shot_id, f"Failed to open keyframe {img_path}: {exc}")]

    if w < 640 or h < 360:
        issues.append(
            LintIssue("error", "LOW_RES_KEYFRAME", shot_id, f"Keyframe resolution {w}x{h} below 640x360", {"width": w, "height": h})
        )
    if h <= 0 or abs((w / float(h)) - (16.0 / 9.0)) > 0.09:
        issues.append(
            LintIssue(
                "error",
                "ASPECT_RATIO_MISMATCH",
                shot_id,
                f"Keyframe aspect ratio {w}:{h} deviates from 16:9",
                {"width": w, "height": h},
            )
        )
    if mean_luma < 3.0:
        issues.append(
            LintIssue("error", "BLANK_KEYFRAME", shot_id, f"Keyframe is nearly black (mean luma={mean_luma:.2f})", {"mean_luma": round(mean_luma, 3)})
        )
    return issues


def validate_h3_bundle(
    bundle_dir: Path,
    *,
    max_frames: int = H3_DEFAULT_MAX_FRAMES,
    check_images: bool = True,
    require_chapter_field: bool = False,
) -> BundleLintReport:
    """Validate an H3 chapter bundle directory or explicit prompt JSON file against h3-chapter-bundle/v1."""
    raw_target = Path(bundle_dir)
    if raw_target.is_file() and raw_target.suffix.lower() == ".json":
        prompts_path = raw_target
        bdir = raw_target.parent
    else:
        bdir = raw_target
        prompts_path = bdir / "h3_prompts.json"
    errors: list[LintIssue] = []
    warnings: list[LintIssue] = []

    if not prompts_path.is_file():
        errors.append(LintIssue("error", "MISSING_PROMPTS_JSON", None, f"Missing {prompts_path.name} in {bdir}"))
        return BundleLintReport(H3_BUNDLE_SCHEMA, str(raw_target), None, None, False, errors, warnings, 0, None)

    try:
        raw_bytes = prompts_path.read_bytes()
        data = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        errors.append(LintIssue("error", "INVALID_JSON", None, f"Invalid h3_prompts.json: {exc}"))
        return BundleLintReport(H3_BUNDLE_SCHEMA, str(bdir), None, None, False, errors, warnings, 0, None)

    if not isinstance(data, dict):
        errors.append(LintIssue("error", "INVALID_JSON_ROOT", None, "h3_prompts.json root must be a JSON object"))
        return BundleLintReport(H3_BUNDLE_SCHEMA, str(bdir), None, None, False, errors, warnings, 0, None)

    title = str(data.get("title") or "").strip() or None
    if not title:
        errors.append(LintIssue("error", "MISSING_TITLE", None, "Top-level 'title' must be a non-empty string"))

    chapter_raw = data.get("chapter")
    chapter = str(chapter_raw).strip() if chapter_raw is not None and str(chapter_raw).strip() else None
    if not chapter:
        issue = LintIssue("error" if require_chapter_field else "warning", "MISSING_CHAPTER_FIELD", None, "Top-level 'chapter' field is missing")
        (errors if require_chapter_field else warnings).append(issue)

    fps = data.get("fps")
    if fps is not None and float(fps) != 24.0:
        errors.append(LintIssue("error", "INVALID_FPS", None, f"Expected fps=24, got {fps!r}"))

    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        errors.append(LintIssue("error", "MISSING_SHOTS", None, "Top-level 'shots' must be a non-empty list"))
        return BundleLintReport(H3_BUNDLE_SCHEMA, str(bdir), chapter, title, False, errors, warnings, 0, None)

    seen_ids: set[str] = set()
    keyframe_hashes: list[tuple[str, str]] = []

    for idx, sdata in enumerate(shots):
        if not isinstance(sdata, dict):
            errors.append(LintIssue("error", "INVALID_SHOT_ENTRY", f"#{idx}", "Shot entry must be an object"))
            continue

        sid = str(sdata.get("id") or f"#{idx}").strip()
        for key in REQUIRED_SHOT_KEYS:
            val = sdata.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(LintIssue("error", "MISSING_SHOT_FIELD", sid, f"Shot missing required field {key!r}"))

        if sid in seen_ids:
            errors.append(LintIssue("error", "DUPLICATE_SHOT_ID", sid, f"Duplicate shot id {sid!r}"))
        seen_ids.add(sid)

        frames = sdata.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
            errors.append(LintIssue("error", "INVALID_FRAMES", sid, f"Shot 'frames' must be positive int, got {frames!r}"))
        else:
            if not is_minimax_h3_grid_length(frames):
                errors.append(LintIssue("error", "INVALID_H3_FRAME_GRID", sid, f"Shot frames={frames} is not on 17k+5 grid"))
            if frames > max_frames:
                errors.append(LintIssue("error", "FRAMES_EXCEED_MAX", sid, f"Shot frames={frames} exceeds max_frames={max_frames}"))

        h3_prompt = str(sdata.get("h3_prompt") or "")
        tags = extract_dialogue_tags(h3_prompt)
        dlg = sdata.get("dialogue")

        if isinstance(dlg, dict) and str(dlg.get("line") or "").strip():
            line_txt = str(dlg["line"])
            if not tags:
                errors.append(LintIssue("error", "MISSING_D_TAG", sid, "Dialogue shot has dialogue.line but no <d>...</d> tag in h3_prompt"))
            elif normalize_dialogue_text(line_txt) != normalize_dialogue_text("".join(tags)):
                errors.append(
                    LintIssue(
                        "error",
                        "DIALOGUE_PROMPT_MISMATCH",
                        sid,
                        "dialogue.line differs from <d> tag text in h3_prompt",
                        {"dialogue_line": line_txt, "prompt_d_tags": tags},
                    )
                )
            mode = str(dlg.get("mode") or "visible").strip()
            if mode not in VALID_DIALOGUE_MODES:
                errors.append(LintIssue("error", "INVALID_DIALOGUE_MODE", sid, f"Unsupported dialogue mode {mode!r}"))
            elif mode == "off_screen_voiceover" and not _CLOSED_LIPS_RE.search(h3_prompt):
                warnings.append(
                    LintIssue(
                        "warning",
                        "VOICEOVER_MISSING_CLOSED_LIPS_GUARD",
                        sid,
                        "Voiceover shot h3_prompt lacks explicit closed-lips guard phrase",
                    )
                )
        else:
            if tags:
                errors.append(
                    LintIssue(
                        "error",
                        "UNEXPECTED_D_TAG",
                        sid,
                        "Non-dialogue shot contains unexpected <d>...</d> tag in h3_prompt",
                        {"prompt_d_tags": tags},
                    )
                )
            sc_split = _split_overall_soundscape(h3_prompt)
            if sc_split is not None:
                sc_text = sc_split[1]
                for rule_name, pat, _ in SILENT_SPEECH_TRIGGER_PATTERNS:
                    m = pat.search(sc_text)
                    if m:
                        errors.append(
                            LintIssue(
                                "error",
                                "SILENT_SOUNDSCAPE_SPEECH_TRIGGER",
                                sid,
                                f"Silent shot overall_soundscape contains speech-trigger phrase ({rule_name})",
                                {"rule": rule_name, "matched": m.group(0).strip()},
                            )
                        )

        trap_match = _COLOR_TRAP_RE.search(h3_prompt)
        if trap_match:
            warnings.append(
                LintIssue(
                    "warning",
                    "CONTRASTIVE_COLOR_TRAP",
                    sid,
                    "Prompt contains contrastive garment-color phrasing that may cause costume drift",
                    {"matched": trap_match.group(0).strip()},
                )
            )

        ff_rel = sdata.get("first_frame")
        if ff_rel:
            img_path = (bdir / str(ff_rel)).resolve()
            if check_images:
                for img_issue in _lint_image(img_path, sid):
                    (errors if img_issue.severity == "error" else warnings).append(img_issue)
            if img_path.is_file():
                keyframe_hashes.append((sid, hashlib.sha256(img_path.read_bytes()).hexdigest()))

    fp_hash = hashlib.sha256()
    fp_hash.update(raw_bytes)
    for sid, kh in sorted(keyframe_hashes):
        fp_hash.update(f"|{sid}:{kh}".encode("utf-8"))
    bundle_fp = fp_hash.hexdigest()[:24]

    return BundleLintReport(
        schema=H3_BUNDLE_SCHEMA,
        bundle_dir=str(raw_target),
        chapter=chapter,
        title=title,
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        shot_count=len(shots),
        bundle_fingerprint=bundle_fp,
    )


def verify_comfyui_history_entry(
    history: Mapping[str, Any],
    *,
    expected_prompt: str,
    expected_length: int,
    expected_width: int = 1024,
    expected_height: int = 576,
    expected_steps: int = 20,
    expected_turbo: bool = False,
    expected_upscale: tuple[int, int] = (1920, 1080),
) -> dict[str, Any]:
    """Verify ComfyUI /history contains a matching H3 graph with exact dimensions and quality parameters."""
    if not isinstance(history, Mapping):
        raise ValueError("ComfyUI history must be a mapping")

    matches: list[tuple[int, dict[str, Any]]] = []
    target_prompt = str(expected_prompt).strip()

    for idx, (pid, entry) in enumerate(history.items()):
        if not isinstance(entry, Mapping):
            continue
        prompt_tuple = entry.get("prompt")
        if not isinstance(prompt_tuple, (list, tuple)) or len(prompt_tuple) < 3:
            continue
        pg = prompt_tuple[2]
        if not isinstance(pg, Mapping) or "5" not in pg:
            continue
        n5 = (pg["5"] or {}).get("inputs") or {}
        if str(n5.get("prompt") or "").strip() != target_prompt or n5.get("length") != expected_length:
            continue

        n8 = (pg.get("8") or {}).get("inputs") or {}
        n22 = (pg.get("22") or {}).get("inputs") or {}
        seq = prompt_tuple[0]
        turbo = "15" in pg
        info = {
            "prompt_id": pid,
            "node5_width": n5.get("width"),
            "node5_height": n5.get("height"),
            "node5_length": n5.get("length"),
            "steps": n8.get("steps"),
            "turbo": turbo,
            "upscale_width": n22.get("width"),
            "upscale_height": n22.get("height"),
            "prompt_exact_match": True,
        }
        if (info["node5_width"], info["node5_height"], info["node5_length"]) != (
            expected_width,
            expected_height,
            expected_length,
        ):
            raise ValueError(
                f"ComfyUI node 5 dims mismatch: {info} (expected {expected_width}x{expected_height}x{expected_length})"
            )
        if info["steps"] != expected_steps or turbo != expected_turbo:
            raise ValueError(
                f"ComfyUI sampling quality mismatch: {info} (expected steps={expected_steps}, turbo={expected_turbo})"
            )
        if (info["upscale_width"], info["upscale_height"]) != expected_upscale:
            raise ValueError(
                f"ComfyUI node 22 upscale mismatch: {info} (expected {expected_upscale[0]}x{expected_upscale[1]})"
            )
        matches.append((int(seq) if isinstance(seq, int) else idx, info))

    if not matches:
        raise ValueError(f"Could not find matching verbatim prompt in ComfyUI history for length={expected_length}")
    matches.sort(key=lambda item: item[0])
    return matches[-1][1]
