"""Export project timelines into self-contained Jianying Pro (CapCut CN) draft directories."""
from __future__ import annotations

import json
import shutil
import time
import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

from .editing import build_subtitle_cues, ensure_bgm_track, resolve_canvas_size, resolve_safe_shot_transition

_TRANSITION_LABELS = {
    "fade": "叠化",
    "xfade_dissolve": "叠化",
    "xfade_wipeleft": "向左擦除",
    "xfade_slideup": "向上滑动",
}


def _us(seconds: float) -> int:
    return max(0, round(float(seconds) * 1_000_000))


def _wav_duration(path: Path, fallback: float) -> float:
    try:
        with wave.open(str(path), "rb") as wav_in:
            if wav_in.getframerate() > 0:
                return max(0.0, wav_in.getnframes() / wav_in.getframerate())
    except (wave.Error, OSError):
        pass
    return fallback


def _stage_asset(source: Path, dest_name: str, assets_dir: Path, copy_assets: bool) -> Path:
    resolved = source.resolve()
    if not copy_assets:
        return resolved
    dest = assets_dir / dest_name
    shutil.copy2(resolved, dest)
    return dest.resolve()


def export_jianying_draft(
    manifest: Any,
    project_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    copy_assets: bool = True,
) -> Path:
    """Create a self-contained Jianying Pro draft folder and record its path on manifest."""
    root = Path(project_root).resolve()
    draft_dir = Path(output_dir).resolve() if output_dir else (root / "jianying_draft")
    assets_dir = draft_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    aspect_ratio = getattr(manifest, "aspect_ratio", "16:9")
    width, height = resolve_canvas_size(aspect_ratio)
    alignment_mode = getattr(manifest, "alignment_mode", "pad")
    total_seconds = sum(float(shot.duration) for shot in manifest.shots)
    total_us = _us(total_seconds)

    videos_mat: list[dict[str, Any]] = []
    audios_mat: list[dict[str, Any]] = []
    texts_mat: list[dict[str, Any]] = []
    speeds_mat: list[dict[str, Any]] = []
    transitions_mat: list[dict[str, Any]] = []
    canvases_mat: list[dict[str, Any]] = []

    video_segments: list[dict[str, Any]] = []
    voice_segments: list[dict[str, Any]] = []
    text_segments: list[dict[str, Any]] = []
    meta_files: list[dict[str, Any]] = []

    canvas_id = uuid4().hex
    canvases_mat.append({
        "id": canvas_id,
        "type": "canvas_color",
        "color": "#101a2b",
        "album_image": "",
        "blur": 0.0,
    })

    cursor_seconds = 0.0
    for idx, shot in enumerate(manifest.shots, 1):
        shot_dur = float(shot.duration)
        start_us = _us(cursor_seconds)
        dur_us = _us(shot_dur)

        raw_video = shot.assets.get("video") if getattr(shot, "mode", "preview") == "render" else None
        visual_src = Path(raw_video) if raw_video and Path(raw_video).is_file() else Path(shot.assets["image"])
        is_video = bool(raw_video and Path(raw_video).is_file())
        visual_name = f"shot_{idx:02d}_{'video' if is_video else 'image'}{visual_src.suffix or '.png'}"
        staged_visual = _stage_asset(visual_src, visual_name, assets_dir, copy_assets)

        raw_video_dur = (getattr(shot, "scene_context", {}) or {}).get("_video_duration")
        if is_video and alignment_mode == "elastic" and raw_video_dur and float(raw_video_dur) > 0:
            speed_ratio = round(float(raw_video_dur) / shot_dur, 4) if shot_dur > 0 else 1.0
            source_dur_us = _us(float(raw_video_dur))
        else:
            speed_ratio = 1.0
            source_dur_us = _us(float(raw_video_dur)) if (is_video and raw_video_dur) else dur_us

        vid_mat_id = uuid4().hex
        videos_mat.append({
            "id": vid_mat_id,
            "type": "video" if is_video else "photo",
            "path": str(staged_visual),
            "material_name": staged_visual.name,
            "duration": source_dur_us,
            "width": width,
            "height": height,
            "has_audio": False,
        })
        meta_files.append({"id": vid_mat_id, "file_Path": str(staged_visual), "duration": source_dur_us, "type": 0 if is_video else 2})

        speed_id = uuid4().hex
        speeds_mat.append({
            "id": speed_id,
            "type": "speed",
            "mode": 0,
            "speed": speed_ratio,
        })

        extra_refs = [speed_id, canvas_id]
        next_shot = manifest.shots[idx] if idx < len(manifest.shots) else None
        trans = resolve_safe_shot_transition(
            shot,
            next_shot,
            character_bible=getattr(manifest, "character_bible", None),
        )
        if trans != "cut" and idx < len(manifest.shots):
            trans_id = uuid4().hex
            transitions_mat.append({
                "id": trans_id,
                "type": "transition",
                "name": _TRANSITION_LABELS.get(trans, trans),
                "effect_id": trans,
                "duration": _us(0.36),
                "is_overlap": trans.startswith("xfade_"),
            })
            extra_refs.append(trans_id)

        video_segments.append({
            "id": uuid4().hex,
            "material_id": vid_mat_id,
            "target_timerange": {"start": start_us, "duration": dur_us},
            "source_timerange": {"start": 0, "duration": min(dur_us, source_dur_us) if speed_ratio == 1.0 else source_dur_us},
            "speed": speed_ratio,
            "volume": 0.0 if shot.assets.get("audio") else 1.0,
            "extra_material_refs": extra_refs,
            "clip": {"scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": 0.0}, "rotation": 0.0},
        })

        audio_asset = shot.assets.get("audio")
        if audio_asset and Path(audio_asset).is_file():
            audio_src = Path(audio_asset)
            staged_audio = _stage_asset(audio_src, f"shot_{idx:02d}_voice{audio_src.suffix or '.wav'}", assets_dir, copy_assets)
            actual_audio_dur = _wav_duration(audio_src, shot_dur)
            audio_src_us = min(dur_us, _us(actual_audio_dur)) if actual_audio_dur > 0 else dur_us
            aud_mat_id = uuid4().hex
            audios_mat.append({
                "id": aud_mat_id,
                "type": "extract_music",
                "name": f"{shot.id}_voice",
                "path": str(staged_audio),
                "duration": _us(actual_audio_dur),
            })
            meta_files.append({"id": aud_mat_id, "file_Path": str(staged_audio), "duration": _us(actual_audio_dur), "type": 1})
            voice_segments.append({
                "id": uuid4().hex,
                "material_id": aud_mat_id,
                "target_timerange": {"start": start_us, "duration": audio_src_us},
                "source_timerange": {"start": 0, "duration": audio_src_us},
                "speed": 1.0,
                "volume": 1.0,
                "extra_material_refs": [],
            })

        cursor_seconds += shot_dur

    tracks: list[dict[str, Any]] = [
        {"id": uuid4().hex, "type": "video", "attribute": 0, "flag": 0, "segments": video_segments},
        {"id": uuid4().hex, "type": "audio", "attribute": 0, "flag": 0, "segments": voice_segments},
    ]

    bgm_source = getattr(manifest, "bgm_source", None)
    if bgm_source:
        bgm_src = ensure_bgm_track(str(bgm_source), root, duration=max(12.0, total_seconds))
        staged_bgm = _stage_asset(bgm_src, f"bgm{bgm_src.suffix or '.wav'}", assets_dir, copy_assets)
        bgm_src_dur = _wav_duration(bgm_src, max(12.0, total_seconds))
        bgm_mat_id = uuid4().hex
        audios_mat.append({
            "id": bgm_mat_id,
            "type": "music",
            "name": bgm_src.stem,
            "path": str(staged_bgm),
            "duration": _us(bgm_src_dur),
        })
        meta_files.append({"id": bgm_mat_id, "file_Path": str(staged_bgm), "duration": _us(bgm_src_dur), "type": 1})
        bgm_seg_dur_us = total_us
        tracks.append({
            "id": uuid4().hex,
            "type": "audio",
            "attribute": 0,
            "flag": 0,
            "segments": [{
                "id": uuid4().hex,
                "material_id": bgm_mat_id,
                "target_timerange": {"start": 0, "duration": bgm_seg_dur_us},
                "source_timerange": {"start": 0, "duration": min(bgm_seg_dur_us, _us(bgm_src_dur))},
                "speed": 1.0,
                "volume": float(getattr(manifest, "bgm_volume", 0.22)),
                "extra_material_refs": [],
            }],
        })

    cues = build_subtitle_cues(manifest.shots, aspect_ratio)
    for cue in cues:
        cue_start_us = _us(cue["start"])
        cue_dur_us = max(100_000, _us(cue["end"] - cue["start"]))
        text_mat_id = uuid4().hex
        text_content = json.dumps(
            {
                "text": cue["text"],
                "styles": [
                    {
                        "fill": {"content": {"solid": {"color": [1.0, 1.0, 1.0]}}},
                        "stroke": {"content": {"solid": {"color": [0.06, 0.08, 0.11]}}, "width": 0.08},
                        "size": 9.0,
                        "bold": True,
                        "range": [0, len(cue["text"])],
                    }
                ],
            },
            ensure_ascii=False,
        )
        texts_mat.append({
            "id": text_mat_id,
            "type": "subtitle",
            "content": text_content,
            "recognize_text": cue["text"],
            "font_size": 9.0,
            "alignment": 1,
            "text_color": "#ffffff",
        })
        text_segments.append({
            "id": uuid4().hex,
            "material_id": text_mat_id,
            "target_timerange": {"start": cue_start_us, "duration": cue_dur_us},
            "source_timerange": {"start": 0, "duration": cue_dur_us},
            "speed": 1.0,
            "volume": 1.0,
            "extra_material_refs": [],
            "clip": {"scale": {"x": 1.0, "y": 1.0}, "transform": {"x": 0.0, "y": -0.72}, "rotation": 0.0},
        })
    tracks.append({"id": uuid4().hex, "type": "text", "attribute": 0, "flag": 0, "segments": text_segments})

    draft_id = str(uuid4()).upper()
    draft_content = {
        "id": draft_id,
        "name": manifest.project_id,
        "version": 360000,
        "fps": 25.0,
        "duration": total_us,
        "canvas_config": {"width": width, "height": height, "ratio": aspect_ratio},
        "platform": {"os": "windows"},
        "materials": {
            "videos": videos_mat,
            "audios": audios_mat,
            "texts": texts_mat,
            "speeds": speeds_mat,
            "transitions": transitions_mat,
            "canvases": canvases_mat,
        },
        "tracks": tracks,
    }
    now_us = int(time.time() * 1_000_000)
    draft_meta = {
        "draft_id": draft_id,
        "draft_name": manifest.project_id,
        "draft_fold_path": str(draft_dir),
        "tm_draft_create": now_us,
        "tm_draft_modified": now_us,
        "tm_duration": total_us,
        "draft_materials": [{"type": 0, "value": meta_files}],
    }

    (draft_dir / "draft_content.json").write_text(json.dumps(draft_content, ensure_ascii=False, indent=2), encoding="utf-8")
    (draft_dir / "draft_meta_info.json").write_text(json.dumps(draft_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest.jianying_draft = str(draft_dir)
    return draft_dir
