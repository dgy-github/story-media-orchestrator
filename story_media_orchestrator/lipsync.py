"""Speech delivery classification, dialogue-length shot splitting, and lip-sync post-processing."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable
from urllib import parse as urllib_parse, request as urllib_request
from uuid import uuid4

from .editing import shot_has_character_presence

SPEECH_DELIVERY_MODES = ("silent", "voiceover", "on_screen_dialogue")
LIPSYNC_PROVIDERS = ("off", "http", "comfyui", "command")

_VOICEOVER_SPEAKERS = frozenset(
    {"旁白", "画外音", "narrator", "os", "内心独白", "独白", "传音", "inner_monologue", "voiceover"}
)
_VOICEOVER_SUFFIX_RE = re.compile(
    r"(?:\(\s*(?:os|独白|内心|画外音|传音)\s*\)|（\s*(?:os|画外音|内心|独白|内心独白|传音)\s*）)\s*$",
    re.IGNORECASE,
)
_HIDDEN_FACE_RE = re.compile(r"(?:背影|背对镜头|远景剪影|未露脸|遮挡面部|背身)")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[。！？；，,!?;\n])")


def _is_voiceover_speaker(speaker: str) -> bool:
    clean = (speaker or "").strip()
    if not clean:
        return True
    if clean.lower() in _VOICEOVER_SPEAKERS:
        return True
    return bool(_VOICEOVER_SUFFIX_RE.search(clean))


def classify_shot_speech_delivery(
    shot: Any,
    character_bible: dict[str, Any] | None = None,
) -> str:
    """Classify a shot into 'silent', 'voiceover', or 'on_screen_dialogue'."""
    ctx = getattr(shot, "scene_context", None)
    if isinstance(ctx, dict):
        explicit = str(ctx.get("speech_delivery") or "").strip().lower()
        if explicit in SPEECH_DELIVERY_MODES:
            return explicit
        if ctx.get("is_voiceover") is True:
            return "voiceover"

    dialogue_lines = getattr(shot, "dialogue_lines", None) or []
    narration = str(getattr(shot, "narration", None) or "").strip()
    text = str(getattr(shot, "text", None) or "").strip()
    if not dialogue_lines and not narration and not text:
        return "silent"

    on_screen_lines = [
        line
        for line in dialogue_lines
        if isinstance(line, dict)
        and str(line.get("text") or "").strip()
        and not _is_voiceover_speaker(str(line.get("speaker") or ""))
    ]
    if not on_screen_lines:
        return "voiceover" if (narration or text) else "silent"

    action_blob = " ".join(
        part
        for part in (
            str(getattr(shot, "prompt_override", None) or ""),
            str(ctx.get("visual_action") or "") if isinstance(ctx, dict) else "",
            str(ctx.get("framing") or "") if isinstance(ctx, dict) else "",
        )
        if part
    )
    if _HIDDEN_FACE_RE.search(action_blob) or not shot_has_character_presence(shot, character_bible):
        return "voiceover"
    return "on_screen_dialogue"


def build_mouth_articulation_guidance(
    shot: Any,
    character_bible: dict[str, Any] | None = None,
    *,
    tts_source: str = "silent",
) -> dict[str, str]:
    """Return video prompt/negative hints to prevent mouth-flapping or closed-mouth ventriloquism."""
    delivery = classify_shot_speech_delivery(shot, character_bible)
    ctx = getattr(shot, "scene_context", None)
    has_explicit_delivery = isinstance(ctx, dict) and (
        bool(ctx.get("speech_delivery")) or ctx.get("is_voiceover") is True
    )
    if tts_source == "silent" and not has_explicit_delivery and delivery != "on_screen_dialogue":
        delivery = "silent"

    if delivery == "silent":
        return {
            "delivery": "silent",
            "video_prompt_hint": "人物双唇与在场灵兽动物口部均自然微抿闭合，不张嘴说话或鸣叫",
            "negative_hint": "无声乱动嘴型，动物张嘴鸣叫，夸张咧嘴",
        }
    if delivery == "voiceover":
        return {
            "delivery": "voiceover",
            "video_prompt_hint": "画外音镜头：画面人物双唇与在场灵兽动物口部均保持自然闭合不开口，不鸣叫，仅以眼神与微表情传达情绪",
            "negative_hint": "人物张嘴说话，动物张嘴鸣叫，口型乱动",
        }

    speakers: list[str] = []
    for line in getattr(shot, "dialogue_lines", None) or ():
        if not isinstance(line, dict):
            continue
        sp = str(line.get("speaker") or "").strip()
        if sp and not _is_voiceover_speaker(sp) and sp not in speakers:
            speakers.append(sp)
    speakers_str = "、".join(speakers) if speakers else "说话角色"
    return {
        "delivery": "on_screen_dialogue",
        "video_prompt_hint": f"角色{speakers_str}自然开口说台词，唇齿开合节奏清晰稳定",
        "negative_hint": "闭嘴腹语，夸张咧嘴畸变，双嘴重影",
    }


def recommend_video_frame_length(
    audio_duration: float | None,
    *,
    fps: float = 24.0,
    default_length: int = 124,
    min_frames: int = 49,
    max_frames: int = 169,
) -> int:
    """Snap target video frame count to the 4k+1 causal VAE grid covering audio_duration."""
    if audio_duration is None or audio_duration <= 0 or fps <= 0:
        return int(default_length)
    raw = max(1, math.ceil(float(audio_duration) * float(fps)))
    snapped = 4 * math.ceil(max(0, raw - 1) / 4) + 1
    return max(int(min_frames), min(int(max_frames), int(snapped)))


def recommend_minimax_h3_frame_length(
    audio_duration: float | None,
    *,
    fps: float = 24.0,
    default_length: int = 124,
    min_frames: int = 73,
    max_frames: int = 260,
) -> int:
    """Snap target video frame count to MiniMax H3's trained 17k+5 grid.

    Choose the shortest on-grid length that still covers the audio duration so
    official generation does not trim, resample, or require a 124-frame (5.167s)
    floor when a shorter grid point already covers the wav. The Comfy
    MiniMaxH3ImageToVideo node accepts length >= 1; 17k+5 is the official grid.
    """
    def _grid(k: int) -> int:
        return 17 * int(k) + 5

    min_allowed = max(_grid(4), int(min_frames))  # 73 == 17*4+5
    max_allowed = min(_grid(15), int(max_frames))  # 260 == 17*15+5
    if max_allowed < min_allowed:
        return int(min_allowed)
    if audio_duration is None or audio_duration <= 0 or fps <= 0:
        snapped = int(default_length)
    else:
        raw = max(1, math.ceil(float(audio_duration) * float(fps)))
        k = math.ceil(max(0, raw - 5) / 17)
        snapped = _grid(k)
    if snapped < min_allowed:
        snapped = min_allowed
    if snapped > max_allowed:
        snapped = _grid(math.floor((max_allowed - 5) / 17))
    return int(snapped)


def is_minimax_h3_grid_length(length: int) -> bool:
    """Return True when length is on the official MiniMax H3 17k+5 grid."""
    if type(length) is not int or length < 5:
        return False
    return (length - 5) % 17 == 0


def resolve_explicit_minimax_h3_frame_length(
    length: int,
    *,
    min_frames: int = 73,
    max_frames: int = 260,
) -> int:
    """Accept an exact on-grid visual length. Do not snap from a rounded second value.

    Unlike recommend_minimax_h3_frame_length, this does not take audio duration and
    will not turn 3.042s into 90 frames. Lengths below the official floor stay rejected
    until a shorter official range is separately verified.
    """
    if type(length) is not int or length <= 0:
        raise ValueError("H3 frame length must be a positive int (not a rounded duration)")
    if not is_minimax_h3_grid_length(length):
        raise ValueError(f"H3 frame length {length} is not on the 17k+5 grid")
    min_allowed = max(17 * 4 + 5, int(min_frames))
    max_allowed = min(17 * 15 + 5, int(max_frames))
    if length < min_allowed:
        raise ValueError(
            f"H3 frame length {length} is below the official floor {min_allowed}; "
            "shorter grid points are unverified in this pipeline"
        )
    if length > max_allowed:
        raise ValueError(f"H3 frame length {length} exceeds the official max {max_allowed}")
    return int(length)


def exact_minimax_h3_duration_seconds(length: int, fps: float = 24.0) -> float:
    """Return length/fps exactly. Do not round to 3.042 and re-ceil."""
    frames = resolve_explicit_minimax_h3_frame_length(length)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be a positive finite number")
    return frames / float(fps)


def assert_duration_matches_h3_frame_count(
    duration_seconds: float,
    frame_count: int,
    *,
    fps: float = 24.0,
) -> float:
    """Fail closed when a supplied duration is not exactly frame_count/fps."""
    exact = exact_minimax_h3_duration_seconds(frame_count, fps=fps)
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, (int, float)) or not math.isfinite(float(duration_seconds)):
        raise ValueError("duration_seconds must be a positive finite number")
    if abs(float(duration_seconds) - exact) > 1e-12:
        raise ValueError(
            f"duration {duration_seconds} does not exactly equal {frame_count}/{fps} "
            f"({exact}); refusing to snap a rounded second value onto a different grid point"
        )
    return exact


def _spoken_char_units(text: str) -> float:
    cleaned = re.sub(r"[\s，,。！？.!?;；：:\"'“”‘’「」『』（）()—\-]+", "", text or "")
    if not cleaned:
        return 0.0
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    non_cjk = re.sub(r"[\u4e00-\u9fff]", " ", cleaned)
    ascii_words = [w for w in re.findall(r"[A-Za-z0-9]+", non_cjk) if w]
    other_chars = len(re.sub(r"[A-Za-z0-9\s]", "", non_cjk))
    return float(cjk_count) + len(ascii_words) * 1.3 + float(other_chars)


def estimate_shot_speech_seconds(shot: Any, *, chars_per_second: float = 4.2) -> float:
    """Estimate spoken audio duration in seconds from narration + dialogue_lines (or text)."""
    cps = max(1.0, float(chars_per_second))
    narration = str(getattr(shot, "narration", None) or "").strip()
    dialogue_lines = [
        line
        for line in (getattr(shot, "dialogue_lines", None) or [])
        if isinstance(line, dict) and str(line.get("text") or "").strip()
    ]
    if narration or dialogue_lines:
        units = _spoken_char_units(narration) + sum(
            _spoken_char_units(str(line.get("text") or "")) for line in dialogue_lines
        )
        pauses = 0.22 * max(0, len(dialogue_lines) - 1)
        if narration and dialogue_lines:
            pauses += 0.22
    else:
        units = _spoken_char_units(str(getattr(shot, "text", None) or ""))
        pauses = 0.0
    return round(max(1.0, (units / cps) + pauses), 3)


def _split_text_to_budget(text: str, max_units: int) -> list[str]:
    cleaned = text.strip()
    if not cleaned or _spoken_char_units(cleaned) <= max_units:
        return [cleaned] if cleaned else []

    raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(cleaned) if c and c.strip()]
    chunks: list[str] = []
    current = ""
    for clause in raw_clauses:
        if _spoken_char_units(clause) > max_units:
            if current:
                chunks.append(current)
                current = ""
            # Hard-chunk unusually long punctuation-free spans by character count.
            step = max(6, int(max_units))
            for idx in range(0, len(clause), step):
                part = clause[idx : idx + step].strip()
                if part:
                    chunks.append(part)
            continue
        candidate = f"{current}{clause}" if current else clause
        if not current or _spoken_char_units(candidate) <= max_units:
            current = candidate
        else:
            chunks.append(current)
            current = clause
    if current:
        chunks.append(current)
    return chunks or [cleaned]


def _filter_context_characters_for_speaker(ctx: dict[str, Any], speaker: str) -> dict[str, Any]:
    updated = deepcopy(ctx)
    chars = updated.get("characters")
    if isinstance(chars, list) and speaker:
        matched = []
        sp_low = speaker.strip().lower()
        for item in chars:
            if isinstance(item, dict):
                name = str(item.get("name") or item.get("node_id") or item.get("key") or "").strip().lower()
                if name == sp_low:
                    matched.append(item)
            elif isinstance(item, str) and item.strip().lower() == sp_low:
                matched.append(item)
        if matched:
            updated["characters"] = matched
    return updated


def split_long_dialogue_shots(
    manifest: Any,
    *,
    max_shot_seconds: float = 4.8,
    chars_per_second: float = 4.2,
) -> list[str]:
    """Split shots exceeding max_shot_seconds into speaker-isolated / clause-bounded sub-shots."""
    max_sec = max(2.0, float(max_shot_seconds))
    cps = max(1.0, float(chars_per_second))
    max_units = max(6, int(max_sec * cps))

    new_shots: list[Any] = []
    created_sub_ids: list[str] = []

    for shot in manifest.shots:
        est = estimate_shot_speech_seconds(shot, chars_per_second=cps)
        if est <= max_sec:
            new_shots.append(shot)
            continue

        narration = str(getattr(shot, "narration", None) or "").strip()
        dialogue_lines = [
            line
            for line in (getattr(shot, "dialogue_lines", None) or [])
            if isinstance(line, dict) and str(line.get("text") or "").strip()
        ]
        units: list[tuple[str, str | None, str]] = []
        if narration or dialogue_lines:
            if narration:
                for piece in _split_text_to_budget(narration, max_units):
                    units.append(("narration", None, piece))
            for line in dialogue_lines:
                speaker = str(line.get("speaker") or "").strip() or "角色"
                line_text = str(line.get("text") or "").strip()
                for piece in _split_text_to_budget(line_text, max_units):
                    units.append(("dialogue", speaker, piece))
        else:
            for piece in _split_text_to_budget(str(getattr(shot, "text", None) or ""), max_units):
                units.append(("narration", None, piece))

        if len(units) <= 1:
            new_shots.append(shot)
            continue

        base_ctx = deepcopy(shot.scene_context) if isinstance(shot.scene_context, dict) else {}
        for part_idx, (kind, speaker, piece_text) in enumerate(units, 1):
            sub_id = f"{shot.id}-p{part_idx}"
            sub_ctx = deepcopy(base_ctx)
            sub_ctx["_split_from_shot_id"] = shot.id
            if kind == "dialogue" and speaker and not _is_voiceover_speaker(speaker) and speaker != "角色":
                sub_locks = [speaker]
                sub_ctx = _filter_context_characters_for_speaker(sub_ctx, speaker)
                sub_narration = None
                sub_dialogue = [{"speaker": speaker, "text": piece_text}]
                full_text = f"{speaker}：{piece_text}"
            elif kind == "dialogue":
                sub_locks = list(getattr(shot, "character_locks", None) or [])
                sp_label = speaker or "角色"
                sub_narration = None
                sub_dialogue = [{"speaker": sp_label, "text": piece_text}]
                full_text = f"{sp_label}：{piece_text}" if sp_label != "角色" else piece_text
            else:
                sub_locks = list(getattr(shot, "character_locks", None) or [])
                sub_narration = piece_text
                sub_dialogue = []
                full_text = piece_text

            sub_dur = round(max(1.5, min(max_sec, _spoken_char_units(piece_text) / cps + 0.35)), 2)
            sub_shot = replace(
                shot,
                id=sub_id,
                text=full_text,
                duration=sub_dur,
                narration=sub_narration,
                dialogue_lines=sub_dialogue,
                character_locks=sub_locks,
                scene_context=sub_ctx,
                assets={},
                image_task={},
                quality_report={},
                diagnostics={},
                rework_history=[],
                cache_key=None,
                error=None,
            )
            new_shots.append(sub_shot)
            created_sub_ids.append(sub_id)

    manifest.shots = new_shots
    return created_sub_ids


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    """Keep HTTP and ComfyUI lip-sync calls scoped to the configured origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _include_base64_payload() -> bool:
    raw = os.environ.get("STORY_LIPSYNC_INCLUDE_BASE64", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _validate_comfy_media_name(name: str, *, allow_slash: bool = False) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ComfyUI media filename must not be blank")
    cleaned = name.strip()
    if "\\" in cleaned or ":" in cleaned or ".." in cleaned or cleaned.startswith("/"):
        raise ValueError(f"invalid ComfyUI media filename: {name!r}")
    if not allow_slash and "/" in cleaned:
        raise ValueError(f"invalid ComfyUI media filename: {name!r}")
    return cleaned


def _substitute_workflow_tokens(obj: Any, tokens: dict[str, Any]) -> Any:
    if isinstance(obj, dict):
        return {k: _substitute_workflow_tokens(v, tokens) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_workflow_tokens(item, tokens) for item in obj]
    if isinstance(obj, str):
        if obj in tokens:
            return tokens[obj]
        replaced = obj
        for key, value in tokens.items():
            replaced = replaced.replace(key, str(value))
        return replaced
    return obj


def build_comfy_lipsync_workflow(
    *,
    video_filename: str,
    audio_filename: str,
    seed: int = 0,
    fps: float = 25.0,
) -> dict[str, Any]:
    """Build a ComfyUI API graph for LatentSync lip-syncing or load STORY_LIPSYNC_COMFY_WORKFLOW_JSON."""
    safe_video = _validate_comfy_media_name(video_filename, allow_slash=True)
    safe_audio = _validate_comfy_media_name(audio_filename, allow_slash=True)
    custom_spec = os.environ.get("STORY_LIPSYNC_COMFY_WORKFLOW_JSON", "").strip()
    if custom_spec:
        spec_path = Path(custom_spec)
        raw_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else custom_spec
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("STORY_LIPSYNC_COMFY_WORKFLOW_JSON must decode to a non-empty workflow dict")
        tokens = {
            "{{VIDEO_FILENAME}}": safe_video,
            "{{AUDIO_FILENAME}}": safe_audio,
            "{{SEED}}": int(seed),
            "{{FPS}}": float(fps),
        }
        return _substitute_workflow_tokens(parsed, tokens)

    return {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": safe_video,
                "force_rate": float(fps),
                "force_size": "Disabled",
                "custom_width": 512,
                "custom_height": 512,
                "frame_load_cap": 0,
                "skip_first_frames": 0,
                "select_every_nth": 1,
            },
        },
        "2": {
            "class_type": "LoadAudio",
            "inputs": {"audio": safe_audio},
        },
        "3": {
            "class_type": "LatentSyncVideoLengthAdjuster",
            "inputs": {
                "images": ["1", 0],
                "audio": ["2", 0],
                "mode": "normal",
                "fps": float(fps),
                "silent_padding_sec": 0.2,
            },
        },
        "4": {
            "class_type": "LatentSyncNode",
            "inputs": {
                "images": ["3", 0],
                "audio": ["3", 1],
                "seed": int(seed),
                "lips_expression": 1.5,
                "inference_steps": 20,
            },
        },
        "5": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["4", 0],
                "audio": ["4", 1],
                "frame_rate": float(fps),
                "loop_count": 0,
                "filename_prefix": "lipsync/LatentSync",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }


def _append_comfy_token(path: str, token: str | None) -> str:
    if not token:
        return path
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}token={urllib_parse.quote(token, safe='')}"


def _upload_comfy_media(
    opener: urllib_request.OpenerDirector,
    base_url: str,
    media_bytes: bytes,
    filename: str,
    content_type: str,
    *,
    token: str | None = None,
    timeout: float = 60.0,
) -> str:
    boundary = f"----StoryLipSyncBoundary{uuid4().hex}"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            media_bytes,
            b"\r\n",
            f"--{boundary}\r\n".encode("ascii"),
            b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n',
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    upload_path = _append_comfy_token("/upload/image", token)
    req = urllib_request.Request(
        f"{base_url}{upload_path}",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with opener.open(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8") or "{}")
    uploaded_name = _validate_comfy_media_name(str(payload.get("name") or filename))
    subfolder = str(payload.get("subfolder") or "").strip("/")
    if subfolder:
        _validate_comfy_media_name(subfolder, allow_slash=True)
        return f"{subfolder}/{uploaded_name}"
    return uploaded_name


def _extract_comfy_video_location(history_entry: dict[str, Any]) -> tuple[str, str, str] | None:
    outputs = history_entry.get("outputs") if isinstance(history_entry, dict) else None
    if not isinstance(outputs, dict):
        return None
    for node in outputs.values():
        if not isinstance(node, dict):
            continue
        for field in ("gifs", "videos", "images"):
            items = node.get(field)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                fname = item.get("filename")
                if not isinstance(fname, str) or not fname.strip():
                    continue
                subfolder = str(item.get("subfolder") or "")
                folder_type = str(item.get("type") or "output")
                _validate_comfy_media_name(fname)
                if subfolder:
                    _validate_comfy_media_name(subfolder, allow_slash=True)
                if folder_type not in {"output", "temp"}:
                    raise ValueError(f"invalid ComfyUI output type: {folder_type!r}")
                return fname, subfolder, folder_type
    return None


def _run_comfy_lipsync_workflow(
    base_url: str,
    vid: Path,
    aud: Path,
    *,
    shot_id: str,
    token: str | None = None,
) -> bytes:
    clean_base = base_url.rstrip("/")
    opener = urllib_request.build_opener(_NoRedirect())
    timeout = float(os.environ.get("STORY_LIPSYNC_TIMEOUT", "180"))

    safe_shot = re.sub(r"[^A-Za-z0-9_-]+", "_", str(shot_id or "shot")).strip("_") or "shot"
    video_name = f"lipsync_{safe_shot}_{uuid4().hex[:8]}{vid.suffix or '.mp4'}"
    audio_name = f"lipsync_{safe_shot}_{uuid4().hex[:8]}{aud.suffix or '.wav'}"
    uploaded_video = _upload_comfy_media(
        opener,
        clean_base,
        vid.read_bytes(),
        video_name,
        "video/mp4",
        token=token,
        timeout=timeout,
    )
    uploaded_audio = _upload_comfy_media(
        opener,
        clean_base,
        aud.read_bytes(),
        audio_name,
        "audio/wav",
        token=token,
        timeout=timeout,
    )

    workflow = build_comfy_lipsync_workflow(
        video_filename=uploaded_video,
        audio_filename=uploaded_audio,
    )
    submit_req = urllib_request.Request(
        f"{clean_base}{_append_comfy_token('/prompt', token)}",
        data=json.dumps({"prompt": workflow}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with opener.open(submit_req, timeout=timeout) as resp:
        submit_payload = json.loads(resp.read().decode("utf-8") or "{}")
    prompt_id = str(submit_payload.get("prompt_id") or "").strip()
    if not prompt_id:
        raise RuntimeError("ComfyUI /prompt response missing prompt_id for lipsync workflow")

    max_polls = max(1, int(os.environ.get("STORY_LIPSYNC_COMFY_MAX_POLLS", "180")))
    poll_interval = max(0.0, float(os.environ.get("STORY_LIPSYNC_COMFY_POLL_INTERVAL", "1.0")))
    for poll_idx in range(max_polls):
        hist_path = _append_comfy_token(f"/history/{urllib_parse.quote(prompt_id, safe='')}", token)
        hist_req = urllib_request.Request(f"{clean_base}{hist_path}", method="GET")
        with opener.open(hist_req, timeout=timeout) as resp:
            raw_hist = json.loads(resp.read().decode("utf-8") or "{}")
        entry = raw_hist.get(prompt_id, raw_hist) if isinstance(raw_hist, dict) else {}
        status = entry.get("status", {}) if isinstance(entry, dict) else {}
        if status.get("status_str") in {"error", "failed"}:
            raise RuntimeError(f"ComfyUI lipsync execution failed for prompt_id={prompt_id}")
        location = _extract_comfy_video_location(entry) if (
            status.get("completed") or (isinstance(entry, dict) and entry.get("outputs"))
        ) else None
        if location is not None:
            fname, subfolder, folder_type = location
            view_query = urllib_parse.urlencode(
                {"filename": fname, "subfolder": subfolder, "type": folder_type}
            )
            view_path = _append_comfy_token(f"/view?{view_query}", token)
            view_req = urllib_request.Request(f"{clean_base}{view_path}", method="GET")
            with opener.open(view_req, timeout=timeout) as resp:
                return resp.read()
        if poll_idx + 1 < max_polls and poll_interval > 0:
            time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI lipsync poll timeout for prompt_id={prompt_id}")


def _run_http_lipsync_endpoint(
    url: str,
    vid: Path,
    aud: Path,
    tmp_out: Path,
    *,
    shot_id: str,
    token: str | None = None,
    timeout: float = 120.0,
) -> None:
    body: dict[str, Any] = {
        "shot_id": shot_id,
        "video_path": str(vid.resolve()),
        "audio_path": str(aud.resolve()),
        "output_path": str(tmp_out.resolve()),
        "video_filename": vid.name,
        "audio_filename": aud.name,
    }
    if _include_base64_payload():
        body["video_base64"] = base64.b64encode(vid.read_bytes()).decode("ascii")
        body["audio_base64"] = base64.b64encode(aud.read_bytes()).decode("ascii")

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    opener = urllib_request.build_opener(_NoRedirect())
    req = urllib_request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with opener.open(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if isinstance(payload, dict) and payload.get("video_base64"):
        tmp_out.write_bytes(base64.b64decode(payload["video_base64"], validate=True))
    elif isinstance(payload, dict) and payload.get("output_path") and Path(payload["output_path"]).is_file():
        tmp_out.write_bytes(Path(payload["output_path"]).read_bytes())


def apply_shot_lipsync(
    manifest: Any,
    root: str | Path,
    shot: Any,
    video_path: str | Path,
    audio_path: str | Path,
    *,
    runner: Callable[[Path, Path, Path], Path | bytes | None] | None = None,
) -> Path:
    """Optionally refine on-screen dialogue shots with an audio-driven lip-sync backend."""
    provider = str(getattr(manifest, "lipsync_provider", "off") or "off").strip().lower()
    if provider not in LIPSYNC_PROVIDERS:
        raise ValueError(f"unsupported lipsync_provider: {provider!r}; expected one of {LIPSYNC_PROVIDERS}")

    vid = Path(video_path)
    aud = Path(audio_path)
    if provider == "off":
        return vid

    delivery = classify_shot_speech_delivery(shot, getattr(manifest, "character_bible", None))
    if delivery != "on_screen_dialogue":
        return vid
    if not vid.is_file() or not aud.is_file():
        return vid

    from .cost_ledger import check_and_record_cost

    project_root = Path(root)
    check_and_record_cost(
        manifest,
        shot.id,
        "lipsync",
        provider,
        reason="pipeline:lipsync",
        save_path=project_root / "project.json",
    )

    tmp_out = vid.with_name(f".{vid.stem}.lipsync.{uuid4().hex[:8]}{vid.suffix or '.mp4'}")
    try:
        if callable(runner):
            produced = runner(vid, aud, tmp_out)
            if isinstance(produced, (bytes, bytearray)):
                tmp_out.write_bytes(bytes(produced))
            elif isinstance(produced, (str, Path)) and Path(produced).is_file() and Path(produced).resolve() != tmp_out.resolve():
                tmp_out.write_bytes(Path(produced).read_bytes())
        elif provider == "http":
            url = os.environ.get("STORY_LIPSYNC_URL", "").strip()
            if not url:
                raise RuntimeError("lipsync_provider='http' requires STORY_LIPSYNC_URL")
            _run_http_lipsync_endpoint(
                url,
                vid,
                aud,
                tmp_out,
                shot_id=shot.id,
                token=os.environ.get("STORY_LIPSYNC_TOKEN", "").strip() or None,
                timeout=float(os.environ.get("STORY_LIPSYNC_TIMEOUT", "120")),
            )
        elif provider == "command":
            cmd_tmpl = os.environ.get("STORY_LIPSYNC_CMD", "").strip()
            if not cmd_tmpl:
                raise RuntimeError("lipsync_provider='command' requires STORY_LIPSYNC_CMD")
            cmd_str = cmd_tmpl.format(
                video=str(vid.resolve()),
                audio=str(aud.resolve()),
                output=str(tmp_out.resolve()),
            )
            subprocess.run(cmd_str, shell=True, check=True, capture_output=True)
        elif provider == "comfyui":
            custom_url = os.environ.get("STORY_LIPSYNC_URL", "").strip()
            if custom_url:
                token = (
                    os.environ.get("STORY_LIPSYNC_TOKEN", "").strip()
                    or os.environ.get("MINIMAX_H3_COMFYUI_TOKEN", "").strip()
                    or None
                )
                _run_http_lipsync_endpoint(
                    custom_url,
                    vid,
                    aud,
                    tmp_out,
                    shot_id=shot.id,
                    token=token,
                    timeout=float(os.environ.get("STORY_LIPSYNC_TIMEOUT", "180")),
                )
            else:
                base = (
                    os.environ.get("MINIMAX_H3_COMFYUI_BASE_URL", "").strip()
                    or os.environ.get("COMFYUI_URL", "").strip()
                )
                if not base:
                    raise RuntimeError("lipsync_provider='comfyui' requires STORY_LIPSYNC_URL or MINIMAX_H3_COMFYUI_BASE_URL")
                comfy_token = os.environ.get("MINIMAX_H3_COMFYUI_TOKEN", "").strip() or None
                video_bytes = _run_comfy_lipsync_workflow(
                    base,
                    vid,
                    aud,
                    shot_id=shot.id,
                    token=comfy_token,
                )
                tmp_out.write_bytes(video_bytes)

        if not tmp_out.is_file() or tmp_out.stat().st_size == 0:
            raise RuntimeError(f"lip-sync backend ({provider}) produced no valid output video")
        tmp_out.replace(vid)
    finally:
        tmp_out.unlink(missing_ok=True)

    if isinstance(getattr(shot, "scene_context", None), dict):
        shot.scene_context["_lipsync_applied"] = True
        shot.scene_context["_lipsync_provider"] = provider
    return vid
