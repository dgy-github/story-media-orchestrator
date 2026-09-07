"""Dependency-light preview renderer; uses ffmpeg when installed."""
from __future__ import annotations
import json
import hashlib, shutil, subprocess, wave
from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
from .manifest import ProjectManifest
from .tts import FakeTTSProvider, TTSProvider
from .providers import ImageProvider, TextFrameProvider, VideoProvider

def _cache_key(shot, mode: str) -> str:
    context = "\0" + json.dumps(shot.scene_context, sort_keys=True, ensure_ascii=False) if shot.scene_context else ""
    return hashlib.sha256(f"v1\0{mode}\0{shot.text}{context}".encode()).hexdigest()


def _concat_file(path: str) -> str:
    escaped = Path(path).resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'"


def _shot_cache_key(manifest, shot, mode="preview"):
    profile = (f":dashscope:{manifest.image_model}:{manifest.image_size}:v2"
               if manifest.image_source == "dashscope" else ":story-card-v1" if manifest.real_preview else "")
    return _cache_key(shot, mode + profile)


def render_preview(manifest: ProjectManifest, root: str | Path, tts: TTSProvider | None = None,
                   image_provider: ImageProvider | None = None,
                   video_provider: VideoProvider | None = None,
                   *, max_attempts: int = 2, real_preview: bool | None = None,
                   enforce_review: bool = False) -> Path | None:
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    from .review_gate import block_unapproved
    if enforce_review and block_unapproved(manifest, root, 'storyboard'):
        return None
    if real_preview is not None:
        manifest.real_preview = real_preview
    if manifest.image_source == "dashscope":
        manifest.real_preview = True
    real_preview = manifest.real_preview
    # Older render projects included composition mode in the image/audio key.
    # Migrate only an exact match; content/model changes still invalidate assets.
    for shot in manifest.shots:
        if shot.cache_key == _shot_cache_key(manifest, shot, "render"):
            shot.cache_key = _shot_cache_key(manifest, shot)
    manifest.output = None
    manifest.error = None
    frames = root / "frames"; frames.mkdir(exist_ok=True)
    audio = root / "audio"; audio.mkdir(exist_ok=True)
    try:
        if manifest.image_source == "dashscope" and image_provider is None:
            needs_image = any(s.cache_key != _shot_cache_key(manifest, s)
                              or not Path(s.assets.get("image", "")).is_file() for s in manifest.shots)
            if needs_image:
                from .providers import build_project_image_provider
                image_provider = build_project_image_provider(manifest, root)
        elif manifest.image_source not in {"local", "dashscope"}:
            raise ValueError("unsupported image source")
        if real_preview:
            from .local_media import StoryCardProvider
            image_provider = image_provider or StoryCardProvider()
    except Exception as exc:
        manifest.status = "failed"
        manifest.error = f"image provider setup failed: {type(exc).__name__}: {exc}"
        manifest.save(root / "project.json")
        raise
    if tts is None:
        if manifest.tts_source == 'windows':
            from .tts import WindowsTTSProvider
            tts = WindowsTTSProvider()
        elif manifest.tts_source == 'silent':
            tts = FakeTTSProvider()
        else:
            raise ValueError('unsupported TTS source')
    image_provider = image_provider or TextFrameProvider()
    for i, shot in enumerate(manifest.shots, 1):
        key = _shot_cache_key(manifest, shot)
        cached = (shot.cache_key == key and shot.status in {"generated", "done"}
                  and {"image", "audio"} <= shot.assets.keys()
                  and (manifest.mode != "render" or video_provider is None
                       or (shot.mode == "render" and "video" in shot.assets))
                  and all(Path(p).is_file() for p in shot.assets.values()))
        if cached:
            continue
        if shot.cache_key is not None and shot.cache_key != key:
            shot.assets.clear()
            shot.image_task.clear()
        shot.cache_key = key
        shot.status = "generating"; shot.error = None
        manifest.status = "generating"
        manifest.save(root / "project.json")
        # A cloud task may already have been accepted when the connection fails.
        # Leave resubmission to explicit resume/retry rather than bill twice.
        attempts_limit = 1 if manifest.image_source == "dashscope" or video_provider is not None else max_attempts
        for attempt in range(attempts_limit):
            shot.attempts += 1
            try:
                image = shot.assets.get("image")
                frame_path = Path(image) if image and Path(image).exists() else image_provider.generate(shot, frames / f"{i:04d}.png")
                shot.assets["image"] = str(frame_path)
                shot.cache_key = key
                manifest.save(root / "project.json")
                existing_audio = shot.assets.get("audio")
                audio_path = Path(existing_audio) if existing_audio and Path(existing_audio).is_file() else tts.synthesize(shot.text, audio / f"{i:04d}.wav")
                shot.assets["audio"] = str(audio_path)
                try:
                    with wave.open(str(audio_path), "rb") as source:
                        shot.duration = max(1.0, source.getnframes() / source.getframerate())
                except (wave.Error, OSError):
                    pass
                if manifest.mode == "render" and video_provider:
                    try:
                        shot.assets["video"] = str(video_provider.generate(shot, frame_path, root / "video" / f"{i:04d}.mp4"))
                        shot.mode = "render"
                    except Exception as exc:
                        shot.error = f"video fallback: {type(exc).__name__}: {exc}"
                        shot.mode = "preview"
                shot.cache_key = key; shot.status = "generated"
                break
            except Exception as exc:
                shot.error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 == attempts_limit: shot.status = "failed"
        manifest.save(root / "project.json")
        if shot.status == "failed": manifest.status = "failed"; continue
    if manifest.mode == 'render' or any(s.motion != 'none' or s.transition != 'cut' for s in manifest.shots):
        from .segments import normalize_durations
        normalize_durations(manifest.shots)
    if enforce_review and all(s.status != 'failed' for s in manifest.shots):
        if block_unapproved(manifest, root, 'assets'):
            return None
    ffmpeg = shutil.which("ffmpeg")
    subtitle_file = root / "subtitles.srt"
    clock = 0.0; subtitle_lines = []
    for index, shot in enumerate(manifest.shots, 1):
        end = clock + shot.duration
        fmt = lambda value: f"{int(value//3600):02d}:{int(value%3600//60):02d}:{value%60:06.3f}".replace('.', ',')
        subtitle_lines += [str(index), f"{fmt(clock)} --> {fmt(end)}", shot.subtitle or shot.text, ""]
        clock = end
    subtitle_file.write_text("\n".join(subtitle_lines), encoding="utf-8")
    output = root / "preview.mp4"
    if real_preview:
        from .local_media import find_ffmpeg
        try:
            ffmpeg = find_ffmpeg()
            if any(s.status == "failed" for s in manifest.shots):
                failures = "; ".join(f"{s.id}: {s.error}" for s in manifest.shots if s.status == "failed")
                raise RuntimeError(f"Shot generation failed; {failures}")
        except Exception as exc:
            manifest.status = "failed"
            manifest.error = str(exc)
            manifest.save(root / "project.json")
            raise
    images = [shot.assets.get("image") for shot in manifest.shots]
    if ffmpeg and images and all(image and Path(image).is_file() and Path(image).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} for image in images):
        concat = root / "timeline.txt"
        audio_concat = root / "audio-timeline.txt"
        lines = []
        for shot in manifest.shots:
            lines += [_concat_file(shot.assets['image']), f"duration {shot.duration}"]
        lines.append(lines[-2]); concat.write_text("\n".join(lines), encoding="utf-8")
        audio_concat.write_text("\n".join(_concat_file(shot.assets['audio']) for shot in manifest.shots), encoding="utf-8")
        manifest.status = "assembling"
        manifest.save(root / "project.json")
        temporary_output = root / f".preview-{uuid4().hex}.mp4"
        segment_directory = None
        try:
            from .segments import render_segments, align_audio_segments
            segment_directory = TemporaryDirectory(prefix=".segments-", dir=root)
            aligned_audio = align_audio_segments(ffmpeg, manifest.shots, segment_directory.name,
                                                 use_video_audio=manifest.mode == 'render' and manifest.tts_source == 'silent')
            audio_concat.write_text("\n".join(_concat_file(str(segment)) for segment in aligned_audio), encoding="utf-8")
            if ((manifest.mode == "render" and any(shot.mode == "render" and shot.assets.get("video") for shot in manifest.shots))
                    or any(s.motion != 'none' or s.transition != 'cut' for s in manifest.shots)):
                segments = render_segments(ffmpeg, manifest.shots, segment_directory.name)
                concat.write_text("\n".join(_concat_file(str(segment)) for segment in segments), encoding="utf-8")
            subprocess.run([ffmpeg, "-y", "-xerror", "-f", "concat", "-safe", "0", "-i", str(concat), "-f", "concat", "-safe", "0", "-i", str(audio_concat), "-i", str(subtitle_file), "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=zho", "-t", str(clock), "-vf", "fps=25,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p", "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(temporary_output)], check=True, capture_output=True)
            temporary_output.replace(output)
        except (OSError, subprocess.SubprocessError) as exc:
            manifest.status = "failed"
            manifest.error = f"assembly failed: {type(exc).__name__}: {exc}"
            manifest.save(root / "project.json")
            raise
        finally:
            temporary_output.unlink(missing_ok=True)
            if segment_directory is not None:
                segment_directory.cleanup()
    else:
        output = root / "preview.txt"
        output.write_text("\n".join(shot.text for shot in manifest.shots), encoding="utf-8")
    manifest.output = str(output); manifest.status = "done" if all(s.status != "failed" for s in manifest.shots) else "partial"
    manifest.timeline = [{"shot_id": shot.id, "start": sum(s.duration for s in manifest.shots[:i]), "duration": shot.duration, "transition": shot.transition, "motion": shot.motion} for i, shot in enumerate(manifest.shots)]
    for shot in manifest.shots:
        if shot.status != "failed": shot.status = "done"
    manifest.save(root / "project.json")
    return output
