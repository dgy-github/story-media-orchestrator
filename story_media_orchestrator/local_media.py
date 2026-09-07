"""Local storyboard cards and an optional bundled FFmpeg for playable previews."""
from pathlib import Path
import os
import shutil


def find_ffmpeg() -> str:
    """Resolve an explicit encoder, PATH encoder, or installed preview extra."""
    configured = os.environ.get("STORY_MEDIA_FFMPEG")
    if configured:
        return configured
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('Install playable preview dependencies: pip install -e ".[preview]"') from exc


class StoryCardProvider:
    """Render story text as a PNG card; this is not AI-generated imagery."""

    def generate(self, shot, output: Path) -> Path:
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB", (1280, 720), "#101a2b")
        draw = ImageDraw.Draw(image)
        candidates = [os.environ.get("STORY_MEDIA_FONT", ""),
                      "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        font_path = next((p for p in candidates if p and Path(p).is_file()), None)
        font = ImageFont.truetype(font_path, 42) if font_path else ImageFont.load_default(size=42)
        draw.rectangle((0, 0, 16, 720), fill="#63b5ef")
        draw.text((72, 65), f"STORY PREVIEW / {shot.id}", font=font, fill="#63b5ef")
        lines, line = [], ""
        for char in shot.text:
            if char == "\n" or draw.textlength(line + char, font=font) > 1120:
                lines.append(line)
                line = "" if char == "\n" else char
            else:
                line += char
        lines.append(line)
        if len(lines) > 8:
            lines = lines[:8]
            lines[-1] = lines[-1][:-3] + "..."
        draw.multiline_text((72, 180), "\n".join(lines), font=font, fill="#eef4ff", spacing=15)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
        return output
