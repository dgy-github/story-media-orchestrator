from pathlib import Path
import subprocess

import pytest

from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.preview import render_preview


def test_real_preview_encodes_and_recovers_without_regeneration(tmp_path):
    from story_media_orchestrator.local_media import find_ffmpeg
    root = tmp_path / "中文 project's preview"
    manifest = ProjectManifest.create("preview", "信使抵达。城市苏醒。")
    output = render_preview(manifest, root, real_preview=True)
    assert output.suffix == ".mp4"
    subprocess.run([find_ffmpeg(), "-v", "error", "-i", str(output),
                    "-f", "null", "-"], check=True, capture_output=True)
    attempts = [s.attempts for s in manifest.shots]
    previous_video = output.read_bytes()
    image = Path(manifest.shots[0].assets["image"])
    original = image.read_bytes()
    image.write_bytes(b"invalid png")
    with pytest.raises(subprocess.CalledProcessError):
        render_preview(manifest, root, real_preview=True)
    failed = ProjectManifest.load(root / "project.json")
    assert failed.status == "failed"
    assert failed.output is None
    assert failed.error
    assert output.read_bytes() == previous_video
    image.write_bytes(original)
    render_preview(failed, root, real_preview=True)
    assert failed.status == "done"
    assert failed.error is None
    assert [s.attempts for s in failed.shots] == attempts


def test_real_preview_upgrades_text_cache(tmp_path):
    manifest = ProjectManifest.create("preview", "A scene.")
    render_preview(manifest, tmp_path)
    render_preview(manifest, tmp_path, real_preview=True)
    assert Path(manifest.shots[0].assets["image"]).suffix == ".png"
    assert Path(manifest.output).suffix == ".mp4"


def test_cli_resume_remembers_playable_mode(tmp_path):
    from story_media_orchestrator.cli import main
    main(["create", str(tmp_path), "A scene."])
    main(["run", str(tmp_path), "--real-preview"])
    first = ProjectManifest.load(tmp_path / "project.json")
    main(["resume", str(tmp_path)])
    resumed = ProjectManifest.load(tmp_path / "project.json")
    assert resumed.output.endswith(".mp4")
    assert resumed.shots[0].attempts == first.shots[0].attempts


def test_single_shot_video_contains_frames_for_its_full_duration(tmp_path):
    from story_media_orchestrator.local_media import find_ffmpeg
    manifest = ProjectManifest.create("preview", "A courier walks along the road at sunrise.")
    output = render_preview(manifest, tmp_path, real_preview=True)
    decoded = subprocess.run([find_ffmpeg(), "-v", "error", "-i", str(output),
                              "-an", "-vf", "scale=1:1,format=gray", "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-"], check=True, capture_output=True)
    assert len(decoded.stdout) == round(manifest.shots[0].duration * 25)
