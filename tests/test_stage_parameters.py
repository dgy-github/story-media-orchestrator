import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from story_media_orchestrator.cli import _legacy


def test_image_size_reaches_provider(monkeypatch, tmp_path):
    provider = SimpleNamespace(size="old", _normalize_size=lambda s: "normalized:" + s)
    package = SimpleNamespace(ImagePromptWorkflow=Mock(), DashScopeImageProvider=SimpleNamespace(from_nanocodex_config=lambda: provider))
    monkeypatch.setitem(sys.modules, "story_image_agent", package)
    adapter = Mock()
    adapter.return_value.run.return_value = {}
    monkeypatch.setattr("story_media_orchestrator.adapters.StoryImageAdapter", adapter)
    monkeypatch.setenv("STORY_MEDIA_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"_stage":"image", "scene":{}, "image_size":"720*1280"})))
    _legacy()
    assert provider.size == "normalized:720*1280"
    assert adapter.call_args.args[1] is provider


def test_video_turbo_reaches_adapter(monkeypatch, tmp_path):
    package = SimpleNamespace(VideoPromptWorkflow=Mock(), ComfyUIAdapter=SimpleNamespace(from_environment=Mock()))
    monkeypatch.setitem(sys.modules, "story_video_agent", package)
    adapter = Mock()
    adapter.return_value.run.return_value = {}
    monkeypatch.setattr("story_media_orchestrator.adapters.StoryVideoAdapter", adapter)
    monkeypatch.setenv("STORY_MEDIA_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"_stage":"video", "first_frame_ref":"artifact://sha256/test", "turbo":True})))
    _legacy()
    models = adapter.call_args.kwargs["models"]
    assert models.video_turbo is True
    assert models.video_steps == 8


def test_original_comfy_adapter_can_store_output(monkeypatch, tmp_path):
    from pathlib import Path
    from story_media_orchestrator.registry import ArtifactRegistry
    sibling = Path(__file__).resolve().parents[2] / "story-video-agent"
    monkeypatch.syspath_prepend(str(sibling))
    from story_video_agent.http_adapter import ComfyUIAdapter
    client = object.__new__(ComfyUIAdapter)
    client.submit = lambda *a, **k: "prompt-1"
    client.history = lambda *a: {"status":{"completed":True}}
    client.output_reference = lambda *a: ({"filename":"preview.mp4"}, "video")
    client.download = lambda *a: b"video-content"
    store = ArtifactRegistry(tmp_path)
    result = client.run_to_artifact({}, store)
    assert result["state"] == "succeeded"
    assert store.get_bytes(result["artifact_ref"]) == b"video-content"


def test_comfy_polling_is_paced_and_keeps_task_id(monkeypatch, tmp_path):
    from pathlib import Path
    import pytest
    from story_media_orchestrator.adapters import StoryVideoAdapter
    from story_media_orchestrator.registry import ArtifactRegistry
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "story-video-agent"))
    from story_video_agent.http_adapter import ComfyUIAdapter
    client = object.__new__(ComfyUIAdapter)
    client.base_url = "http://localhost:8188"
    client.submit = Mock(return_value="prompt-pending")
    client.history = Mock(return_value={})
    sleeps = []
    monkeypatch.setattr("story_media_orchestrator.adapters.time.sleep", sleeps.append)
    adapter = StoryVideoAdapter(None, comfy=client, registry=ArtifactRegistry(tmp_path))
    with pytest.raises(TimeoutError, match="prompt-pending"):
        adapter._execute_comfy({}, poll_interval=2, max_polls=3)
    assert sleeps == [2, 2]
    assert client.submit.call_count == 1
    assert client.history.call_count == 3

def test_video_seed_reaches_original_graph(monkeypatch, tmp_path):
    from pathlib import Path
    from unittest.mock import Mock
    from story_media_orchestrator.adapters import StoryVideoAdapter
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-video-agent'))
    workflow = Mock()
    workflow.build_pipeline.return_value = {'schema':'video-generation-pipeline/v2'}
    adapter = StoryVideoAdapter(workflow, comfy=object())
    adapter._execute_comfy = Mock(return_value={'state':'succeeded'})
    adapter.run(first_frame_ref='artifact://sha256/a',story_spans=['scene'],prompt='walk',seed=456)
    graph = adapter._execute_comfy.call_args.args[0]
    assert graph['6']['inputs']['noise_seed'] == 456


def test_video_dimensions_and_duration_reach_original_graph(monkeypatch):
    from pathlib import Path
    from story_media_orchestrator.adapters import StoryVideoAdapter
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-video-agent'))
    workflow = Mock()
    workflow.build_pipeline.return_value = {'schema':'video-generation-pipeline/v2'}
    adapter = StoryVideoAdapter(workflow, comfy=object())
    adapter._execute_comfy = Mock(return_value={'state':'succeeded'})
    adapter.run(first_frame_ref='artifact://sha256/a', story_spans=['scene'], prompt='walk',
                width=480, height=864, length=192, fps=24)
    graph = adapter._execute_comfy.call_args.args[0]
    assert graph['5']['inputs']['width'] == 480
    assert graph['5']['inputs']['height'] == 864
    assert graph['5']['inputs']['length'] == 192
    assert graph['13']['inputs']['fps'] == 24
    assert workflow.build_pipeline.call_args.kwargs['coarse_duration_seconds'] == 8

def test_original_action_plan_is_preserved_but_unsupported_execution_is_rejected(monkeypatch):
    from pathlib import Path
    from unittest.mock import Mock
    import pytest
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-video-agent'))
    from story_video_agent import VideoPromptWorkflow
    from story_video_agent.action_units import classic_action_unit
    from story_media_orchestrator.adapters import StoryVideoAdapter
    unit = classic_action_unit('turn_and_leave_v1')
    arguments = dict(first_frame_ref='artifact://sha256/' + 'a' * 64, story_spans=['story-package/scene'],
                     prompt='turn and leave', action_unit=unit, length=120, fps=24)
    plan = StoryVideoAdapter(VideoPromptWorkflow('p')).run(**arguments)
    assert plan['schema'] == 'video-generation-pipeline/v2'
    assert plan['status'] == 'planned'
    adapter = StoryVideoAdapter(VideoPromptWorkflow('p'), comfy=object())
    adapter._execute_comfy = Mock()
    with pytest.raises(ValueError, match='未接入动作控制'):
        adapter.run(**arguments)
    adapter._execute_comfy.assert_not_called()


def test_image_revision_uses_original_workflow_and_preserves_history(monkeypatch, tmp_path):
    import base64
    from pathlib import Path
    from story_media_orchestrator.adapters import StoryImageAdapter
    from story_media_orchestrator.registry import ArtifactRegistry
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-image-agent'))
    from story_image_agent import ImagePromptWorkflow
    workflow = ImagePromptWorkflow('revision-project')
    provider = Mock()
    provider.generate.return_value = {'content_base64': base64.b64encode(b'image').decode()}
    registry = ArtifactRegistry(tmp_path)
    result = StoryImageAdapter(workflow, provider, registry).run(
        {'action': 'open door', 'negative': 'no watermark'}, ['scene-original'],
        candidate_count=2, prompt_revision='A wide view of the clock tower')
    assert provider.generate.call_count == 2
    for call in provider.generate.call_args_list:
        assert call.args[0]['prompt'] == 'A wide view of the clock tower'
        assert call.args[0]['source_spans'] == ['scene-original']
        assert call.args[0]['negative_prompt'] == 'no watermark'
    history = json.loads(registry.get_bytes(result['prompt_history_ref']))['revisions']
    assert json.loads(json.dumps(result['prompt_history'])) == history
    assert len(history) == 2
    assert history[1]['parent_revision_id'] == history[0]['revision_id']
    assert history[1]['revision_id'] == result['plan']['prompt_revision_id']


def test_candidate_error_details_never_expose_provider_secrets():
    from urllib.error import HTTPError
    from story_media_orchestrator.errors import image_error_detail
    failure = RuntimeError('secret-provider-body')
    failure.__cause__ = HTTPError('https://example.invalid?key=secret', 401, 'secret-key', {}, None)
    assert '认证失败' in image_error_detail(failure)
    assert 'secret' not in image_error_detail(failure)
    assert 'secret' not in image_error_detail(RuntimeError('secret-key'))
    assert '恢复原批次' in image_error_detail(TimeoutError('secret-key'))
