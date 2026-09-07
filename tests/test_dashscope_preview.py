import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from story_media_orchestrator.cli import main
from story_media_orchestrator.manifest import ProjectManifest


@pytest.fixture
def cloud(monkeypatch):
    calls = []
    class Provider:
        model = "wan2.2-t2i-flash"
        size = "720*1280"
        base_url = "https://dashscope.aliyuncs.com/api/v1"
        @classmethod
        def from_nanocodex_config(cls):
            return cls()
        @staticmethod
        def _normalize_size(size):
            return size
        def _read_json(self, request):
            return {"output": {"task_id": "test-task"}}
        def generate(self, request):
            calls.append((self.model, self.size, request))
            buffer = BytesIO()
            Image.new("RGB", (72, 128), "blue").save(buffer, format="PNG")
            return {"content_base64": base64.b64encode(buffer.getvalue()).decode()}
    class Workflow:
        def __init__(self, project_id):
            self.project_id = project_id
        def build_production_plan(self, scene, spans, candidate_count):
            return {"candidates": [{"request_id": "test", "prompt": scene.get("action", "default action")}]}
        def finalize_candidate(self, plan, candidate_id, quality):
            return plan
    monkeypatch.setattr("story_media_orchestrator.runtime._import_from_root", lambda *args:
                        SimpleNamespace(DashScopeImageProvider=Provider, ImagePromptWorkflow=Workflow))
    return calls, Provider


def test_cloud_cli_persists_choice_and_resume_reuses_image(tmp_path, cloud):
    calls, _ = cloud
    main(["create", str(tmp_path), "A courier at sunrise."])
    main(["run", str(tmp_path), "--image-source", "dashscope"])
    manifest = ProjectManifest.load(tmp_path / "project.json")
    assert manifest.image_source == "dashscope"
    assert manifest.image_model == "wan2.2-t2i-flash"
    assert manifest.output.endswith(".mp4")
    assert len(calls) == 1
    assert calls[0][2]["prompt"].startswith("A courier at sunrise")
    assert "不要分屏、拼贴" in calls[0][2]["prompt"]
    main(["resume", str(tmp_path)])
    assert len(calls) == 1
    assert ProjectManifest.load(tmp_path / "project.json").shots[0].attempts == 1


def test_switching_from_local_and_changing_model_invalidates_cache(tmp_path, cloud):
    calls, _ = cloud
    main(["create", str(tmp_path), "A courier."])
    main(["run", str(tmp_path), "--real-preview"])
    main(["run", str(tmp_path), "--image-source", "dashscope"])
    assert len(calls) == 1
    main(["run", str(tmp_path), "--image-model", "another-model"])
    assert len(calls) == 2
    assert calls[-1][0] == "another-model"


def test_cloud_failure_does_not_fall_back_or_resubmit(tmp_path, cloud):
    calls, provider = cloud
    def fail(self, request):
        calls.append(request)
        raise RuntimeError("provider unavailable")
    provider.generate = fail
    main(["create", str(tmp_path), "A courier."])
    with pytest.raises(RuntimeError):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    failed = ProjectManifest.load(tmp_path / "project.json")
    assert len(calls) == 1
    assert failed.status == "failed"
    assert failed.shots[0].status == "failed"
    assert "provider unavailable" in failed.shots[0].error
    assert failed.output is None
    assert failed.image_source == "dashscope"


def test_cloud_setup_failure_is_saved(tmp_path, cloud):
    _, provider = cloud
    def missing():
        raise RuntimeError("no configured key")
    provider.from_nanocodex_config = missing
    main(["create", str(tmp_path), "A courier."])
    with pytest.raises(RuntimeError, match="no configured key"):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    failed = ProjectManifest.load(tmp_path / "project.json")
    assert failed.status == "failed"
    assert "no configured key" in failed.error
    assert failed.image_source == "dashscope"


def test_downloaded_image_survives_audio_failure(tmp_path, cloud):
    from story_media_orchestrator.preview import render_preview
    calls, _ = cloud
    class BrokenTTS:
        def synthesize(self, text, output):
            raise RuntimeError("audio unavailable")
    manifest = ProjectManifest.create("demo", "A courier.")
    manifest.image_source = "dashscope"
    with pytest.raises(RuntimeError):
        render_preview(manifest, tmp_path, tts=BrokenTTS())
    restored = ProjectManifest.load(tmp_path / "project.json")
    render_preview(restored, tmp_path)
    assert len(calls) == 1
    assert restored.status == "done"


def test_auth_error_is_actionable_and_does_not_expose_url(tmp_path, cloud):
    from urllib.error import HTTPError
    _, provider = cloud
    def unauthorized(self, request):
        try:
            raise HTTPError("https://example.invalid/?token=secret", 401, "unauthorized", {}, None)
        except HTTPError as exc:
            raise RuntimeError("provider request failed") from exc
    provider.generate = unauthorized
    main(["create", str(tmp_path), "A courier."])
    with pytest.raises(RuntimeError):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    error = ProjectManifest.load(tmp_path / "project.json").shots[0].error
    assert "401" in error
    assert "API Key" in error
    assert "secret" not in error


def test_cached_images_can_resume_without_credentials(tmp_path, cloud):
    _, provider = cloud
    main(["create", str(tmp_path), "A courier."])
    main(["run", str(tmp_path), "--image-source", "dashscope"])
    def missing():
        raise RuntimeError("no configured key")
    provider.from_nanocodex_config = missing
    main(["resume", str(tmp_path)])
    assert ProjectManifest.load(tmp_path / "project.json").status == "done"


def test_resume_reuses_accepted_cloud_task_after_poll_failure(tmp_path, cloud):
    from urllib.request import Request
    _, provider = cloud
    original_generate = provider.generate
    submissions = []
    def submit(self, request):
        submissions.append(request)
        return {"output": {"task_id": "accepted-task"}}
    provider._read_json = submit
    should_fail = [True]
    def generate(self, request):
        receipt = self._read_json(Request(self.base_url + "/submit", b'fixed body'))
        assert receipt["output"]["task_id"] == "accepted-task"
        if should_fail[0]:
            raise TimeoutError("poll timed out")
        return original_generate(self, request)
    provider.generate = generate
    main(["create", str(tmp_path), "A courier."])
    main(["run", str(tmp_path), "--real-preview"])
    with pytest.raises(RuntimeError):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    should_fail[0] = False
    main(["resume", str(tmp_path)])
    assert len(submissions) == 1
    assert ProjectManifest.load(tmp_path / "project.json").status == "done"


def test_unknown_submit_blocks_resume_until_explicit_retry(tmp_path, cloud):
    from urllib.request import Request
    _, provider = cloud
    submissions = []
    def disconnect(self, request):
        submissions.append(request)
        raise ConnectionError("connection lost before receipt")
    provider._read_json = disconnect
    def generate(self, request):
        return self._read_json(Request(self.base_url + "/submit", b'fixed body'))
    provider.generate = generate
    main(["create", str(tmp_path), "A courier."])
    with pytest.raises(RuntimeError):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    with pytest.raises(RuntimeError):
        main(["resume", str(tmp_path)])
    assert len(submissions) == 1
    with pytest.raises(RuntimeError):
        main(["retry", str(tmp_path), "shot-01"])
    assert len(submissions) == 2


def test_rejected_submit_can_resume_after_credentials_are_fixed(tmp_path, cloud):
    from urllib.error import HTTPError
    from urllib.request import Request
    _, provider = cloud
    original_generate = provider.generate
    submissions = []
    rejected = [True]
    def submit(self, request):
        submissions.append(request)
        if rejected[0]:
            raise HTTPError(self.base_url, 401, "unauthorized", {}, None)
        return {"output": {"task_id": "accepted-task"}}
    provider._read_json = submit
    def generate(self, request):
        self._read_json(Request(self.base_url + "/submit", b'fixed body'))
        return original_generate(self, request)
    provider.generate = generate
    main(["create", str(tmp_path), "A courier."])
    with pytest.raises(RuntimeError):
        main(["run", str(tmp_path), "--image-source", "dashscope"])
    assert ProjectManifest.load(tmp_path / "project.json").shots[0].image_task["state"] == "rejected"
    rejected[0] = False
    main(["resume", str(tmp_path)])
    assert len(submissions) == 2
    assert ProjectManifest.load(tmp_path / "project.json").status == "done"
