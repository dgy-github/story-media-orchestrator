"""Dependency-light preview renderer; uses ffmpeg when installed."""
from __future__ import annotations
import hashlib, shutil, subprocess
from pathlib import Path
from .manifest import ProjectManifest
from .tts import FakeTTSProvider, TTSProvider
from .providers import ImageProvider, TextFrameProvider, VideoProvider

def _cache_key(shot, mode: str) -> str:
    return hashlib.sha256(f"v1\0{mode}\0{shot.text}\0{shot.duration}".encode()).hexdigest()


def render_preview(manifest: ProjectManifest, root: str | Path, tts: TTSProvider | None = None,
                   image_provider: ImageProvider | None = None,
                   video_provider: VideoProvider | None = None,
                   *, max_attempts: int = 2) -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    frames = root / "frames"; frames.mkdir(exist_ok=True)
    audio = root / "audio"; audio.mkdir(exist_ok=True)
    tts = tts or FakeTTSProvider(); image_provider = image_provider or TextFrameProvider()
    for i, shot in enumerate(manifest.shots, 1):
        key = _cache_key(shot, manifest.mode)
        cached = shot.cache_key == key and shot.status == "done" and all(Path(p).exists() for p in shot.assets.values())
        if cached:
            continue
        shot.status = "generating"; shot.error = None
        for attempt in range(max_attempts):
            shot.attempts += 1
            try:
                image = shot.assets.get("image")
                frame_path = Path(image) if image and Path(image).exists() else image_provider.generate(shot, frames / f"{i:04d}.png")
                shot.assets["image"] = str(frame_path)
                shot.assets["audio"] = str(tts.synthesize(shot.text, audio / f"{i:04d}.wav"))
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
                if attempt + 1 == max_attempts: shot.status = "failed"
        if shot.status == "failed": manifest.status = "failed"; continue
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
    images = [shot.assets.get("image") for shot in manifest.shots]
    if ffmpeg and all(image and Path(image).exists() for image in images):
        concat = root / "timeline.txt"
        lines = []
        for shot in manifest.shots:
            lines += [f"file '{Path(shot.assets['image']).resolve().as_posix()}'", f"duration {shot.duration}"]
        lines.append(lines[-2]); concat.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-shortest", "-vf", "scale=1280:720,format=yuv420p", "-c:v", "libx264", "-c:a", "aac", str(output)], check=True, capture_output=True)
    else:
        output = root / "preview.txt"
        output.write_text("\n".join(shot.text for shot in manifest.shots), encoding="utf-8")
    manifest.output = str(output); manifest.status = "done" if all(s.status != "failed" for s in manifest.shots) else "partial"
    manifest.timeline = [{"shot_id": shot.id, "start": sum(s.duration for s in manifest.shots[:i]), "duration": shot.duration, "transition": "cut"} for i, shot in enumerate(manifest.shots)]
    for shot in manifest.shots:
        if shot.status != "failed": shot.status = "done"
    return output
