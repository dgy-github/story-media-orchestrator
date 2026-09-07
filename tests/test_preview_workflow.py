from pathlib import Path

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
    assert manifest.timeline[1]["start"] == 3.0


def test_render_failure_falls_back_to_preview(tmp_path: Path):
    manifest = ProjectManifest.create("demo", "One shot.")
    manifest.mode = "render"
    render_preview(manifest, tmp_path, video_provider=BrokenVideo())
    assert manifest.status == "done"
    assert manifest.shots[0].mode == "preview"
    assert manifest.shots[0].error.startswith("video fallback:")
