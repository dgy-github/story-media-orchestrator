"""Dependency-light preview renderer; uses ffmpeg when installed."""
from __future__ import annotations
import shutil, subprocess
from pathlib import Path
from .manifest import ProjectManifest
from .tts import FakeTTSProvider, TTSProvider

def render_preview(manifest: ProjectManifest, root: str | Path, tts: TTSProvider | None = None) -> Path:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    frames = root / "frames"; frames.mkdir(exist_ok=True)
    audio = root / "audio"; audio.mkdir(exist_ok=True)
    tts = tts or FakeTTSProvider()
    for i, shot in enumerate(manifest.shots, 1):
        shot.status = "generated"
        image = shot.assets.get("image")
        if image and Path(image).exists():
            frame_path = Path(image)
        else:
            frame_path = frames / f"{i:04d}.txt"
            frame_path.write_text(shot.text, encoding="utf-8")
        tts.synthesize(shot.text, audio / f"{i:04d}.wav")
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
    manifest.output = str(output); manifest.status = "done"
    for shot in manifest.shots: shot.status = "done"
    return output
