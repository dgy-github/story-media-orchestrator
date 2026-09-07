from pathlib import Path
import pytest

from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.preview import render_preview


class BrokenVideo:
    def generate(self, shot, image, output):
        raise RuntimeError("provider unavailable")


def test_resume_uses_cached_assets(tmp_path: Path):
    manifest = ProjectManifest.create("demo", "First shot. Second shot.")
    render_preview(manifest, tmp_path)
    attempts = [shot.attempts for shot in manifest.shots]
    render_preview(manifest, tmp_path)
    assert [shot.attempts for shot in manifest.shots] == attempts
    assert manifest.timeline[1]["start"] == manifest.shots[0].duration
    assert all("audio" in shot.assets for shot in manifest.shots)


def test_render_failure_falls_back_to_preview(tmp_path: Path):
    manifest = ProjectManifest.create("demo", "One shot.")
    manifest.mode = "render"
    render_preview(manifest, tmp_path, video_provider=BrokenVideo())
    assert manifest.status == "done"
    assert manifest.shots[0].mode == "preview"
    assert manifest.shots[0].error.startswith("video fallback:")


def test_text_frames_do_not_enter_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("story_media_orchestrator.preview.shutil.which", lambda _: "not-a-real-ffmpeg")
    manifest = ProjectManifest.create("demo", "A courier arrives.")
    assert render_preview(manifest, tmp_path).read_text(encoding="utf-8") == "A courier arrives"


def test_edit_regenerates_visual(tmp_path):
    manifest = ProjectManifest.create("demo", "Old scene.")
    render_preview(manifest, tmp_path)
    manifest.shots[0].text = "New scene"
    render_preview(manifest, tmp_path)
    assert Path(manifest.shots[0].assets["image"]).read_text(encoding="utf-8") == "New scene"


def test_interruption_preserves_completed_shot(tmp_path):
    from story_media_orchestrator.tts import FakeTTSProvider
    class InterruptedTTS(FakeTTSProvider):
        def synthesize(self, text, output):
            if text == "Second":
                raise KeyboardInterrupt()
            return super().synthesize(text, output)
    manifest = ProjectManifest.create("demo", "First. Second.")
    with pytest.raises(KeyboardInterrupt):
        render_preview(manifest, tmp_path, tts=InterruptedTTS())
    restored = ProjectManifest.load(tmp_path / "project.json")
    assert restored.shots[0].status == "generated"
    render_preview(restored, tmp_path)
    assert restored.shots[0].attempts == 1


def test_failed_manifest_write_preserves_previous_file(tmp_path, monkeypatch):
    manifest = ProjectManifest.create("demo", "Original.")
    target = tmp_path / "project.json"
    manifest.save(target)
    manifest.story = "Changed"
    write_text = Path.write_text
    def disk_full(path, text, **kwargs):
        write_text(path, text[:10], **kwargs)
        raise OSError("disk full")
    monkeypatch.setattr(Path, "write_text", disk_full)
    with pytest.raises(OSError):
        manifest.save(target)
    assert ProjectManifest.load(target).story == "Original."


def test_retry_regenerates_only_selected_visual(tmp_path):
    from story_media_orchestrator.cli import main
    manifest = ProjectManifest.create("demo", "First. Second.")
    render_preview(manifest, tmp_path)
    selected = Path(manifest.shots[0].assets["image"])
    selected.write_text("stale", encoding="utf-8")
    main(["retry", str(tmp_path), "shot-01"])
    restored = ProjectManifest.load(tmp_path / "project.json")
    assert selected.read_text(encoding="utf-8") == "First"
    assert restored.shots[1].attempts == 1


def test_create_does_not_overwrite_existing_project(tmp_path):
    from story_media_orchestrator.cli import main
    main(["create", str(tmp_path), "Original."])
    with pytest.raises(SystemExit):
        main(["create", str(tmp_path), "Replacement."])
    assert ProjectManifest.load(tmp_path / "project.json").story == "Original."
