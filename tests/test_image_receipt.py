import base64
import json
from urllib.request import Request

import pytest

from story_media_orchestrator.image_receipt import generate_candidate
from story_media_orchestrator.registry import ArtifactRegistry


class Provider:
    base_url = "https://example.test"
    model = "test"
    size = "720*1280"

    def __init__(self):
        self.calls = []
        self.fail_poll = True
        self.unknown = False

    def _read_json(self, request):
        self.calls.append(request.get_method())
        if self.unknown:
            raise TimeoutError("submit response lost")
        return {"output": {"task_id": "accepted"}}

    def generate(self, candidate):
        result = self._read_json(Request(self.base_url, json.dumps(candidate['prompt']).encode()))
        assert result['output']['task_id'] == 'accepted'
        if self.fail_poll:
            raise TimeoutError("poll timed out")
        return {"content_base64": base64.b64encode(b"image").decode(), "mime_type": "image/png"}


def test_accepted_candidate_resumes_and_completed_candidate_is_cached(tmp_path):
    provider, store = Provider(), ArtifactRegistry(tmp_path)
    candidate = {"prompt": "scene"}
    with pytest.raises(TimeoutError):
        generate_candidate(provider, candidate, store, "batch", 0)
    provider.fail_poll = False
    first = generate_candidate(provider, candidate, store, "batch", 0)
    assert generate_candidate(provider, candidate, store, "batch", 0) == first
    assert provider.calls == ["POST"]
    # Identical prompts in different candidate slots still generate distinct images.
    generate_candidate(provider, candidate, store, "batch", 1)
    assert provider.calls == ["POST", "POST"]
    with pytest.raises(RuntimeError, match="参数已改变"):
        generate_candidate(provider, {"prompt": "changed"}, store, "batch", 0)


def test_unknown_submission_never_resubmits(tmp_path):
    provider, store = Provider(), ArtifactRegistry(tmp_path)
    provider.unknown = True
    with pytest.raises(TimeoutError):
        generate_candidate(provider, {"prompt": "scene"}, store, "batch", 0)
    with pytest.raises(RuntimeError, match="提交结果未知"):
        generate_candidate(provider, {"prompt": "scene"}, store, "batch", 0)
    assert provider.calls == ["POST"]


def test_corrupted_cached_artifact_never_resubmits(tmp_path):
    provider, store = Provider(), ArtifactRegistry(tmp_path)
    provider.fail_poll = False
    generate_candidate(provider, {"prompt": "scene"}, store, "batch", 0)
    receipt = json.loads(next((tmp_path / "image-tasks").glob("*.json")).read_text())
    (tmp_path / receipt['artifact_ref'].rsplit('/', 1)[-1]).write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match="integrity"):
        generate_candidate(provider, {"prompt": "scene"}, store, "batch", 0)
    assert provider.calls == ["POST"]


def test_original_dashscope_batch_resumes_with_fresh_workflow(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "story-image-agent"))
    from story_image_agent import DashScopeImageProvider, ImagePromptWorkflow
    import story_image_agent.provider as original
    from story_media_orchestrator.adapters import StoryImageAdapter

    submitted, polled = [], []
    interrupted = True

    def transport(self, request):
        if request.get_method() == "POST":
            task = f"task-{len(submitted) + 1}"
            submitted.append(task)
            return {"output": {"task_id": task}}
        task = request.full_url.rsplit("/", 1)[-1]
        polled.append(task)
        if interrupted and task == "task-2":
            return {"output": {"task_status": "RUNNING"}}
        return {"output": {"task_status": "SUCCEEDED", "results": [{"url": f"https://example.test/{task}.png"}]}}

    monkeypatch.setattr(DashScopeImageProvider, "_read_json", transport)
    monkeypatch.setattr(original, "download_content", lambda request, **kwargs: ("image/png", request.full_url.encode()))

    def run():
        # New instances emulate a fresh CLI process, including random request IDs.
        provider = DashScopeImageProvider("test-key", max_polls=1, poll_interval_seconds=0)
        return StoryImageAdapter(ImagePromptWorkflow("project"), provider, ArtifactRegistry(tmp_path)).run(
            {"action": "walk"}, ["story-package/scene-1"], candidate_count=3, batch_id="original-batch")

    first = run()
    assert first["status"] == "partial"
    assert submitted == ["task-1", "task-2", "task-3"]
    interrupted = False
    second = run()
    assert second["status"] == "succeeded"
    assert submitted == ["task-1", "task-2", "task-3"]
    assert polled == ["task-1", "task-2", "task-3", "task-2"]
    assert first["candidate_results"][0]["artifact_ref"] == second["candidate_results"][0]["artifact_ref"]
    assert first["candidate_results"][2]["artifact_ref"] == second["candidate_results"][2]["artifact_ref"]
