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

    def __init__(self, size: tuple[int, int] = (1280, 720)) -> None:
        self.size = (int(size[0]), int(size[1]))

    def generate(self, shot, output: Path) -> Path:
        from PIL import Image, ImageDraw, ImageFont
        width, height = self.size
        font_size = 42 if width >= 1000 else 36
        margin_x = 72 if width >= 1000 else 48
        max_text_width = 1120 if width == 1280 else width - margin_x * 2
        body_y = 180 if height == 720 else max(180, int(height * 0.2))
        line_spacing = 15 if height == 720 else 18
        max_lines = 8 if height == 720 else max(8, (height - body_y - 160) // (font_size + line_spacing))
        image = Image.new("RGB", (width, height), "#101a2b")
        draw = ImageDraw.Draw(image)
        candidates = [os.environ.get("STORY_MEDIA_FONT", ""),
                      "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        font_path = next((p for p in candidates if p and Path(p).is_file()), None)
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(size=font_size)
        draw.rectangle((0, 0, 16, height), fill="#63b5ef")
        draw.text((margin_x, 65), f"STORY PREVIEW / {shot.id}", font=font, fill="#63b5ef")
        lines, line = [], ""
        for char in shot.text:
            if char == "\n" or draw.textlength(line + char, font=font) > max_text_width:
                lines.append(line)
                line = "" if char == "\n" else char
            else:
                line += char
        lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1][:-3] + "..."
        draw.multiline_text((margin_x, body_y), "\n".join(lines), font=font, fill="#eef4ff", spacing=line_spacing)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
        return output
