from unittest.mock import Mock

import pytest

from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.wan_video import run_wan_video


def test_wan_resumes_original_task_and_caches_playable_result(tmp_path):
    provider = Mock(base_url="https://example.test")
    provider.submit.return_value = "wan-task"
    provider.wait.side_effect = [TimeoutError("poll interrupted"),
        {"task_id": "wan-task", "state": "succeeded", "urls": ["https://example.test/video.mp4"]}]
    store = ArtifactRegistry(tmp_path)
    download = Mock(return_value=b'\x00\x00\x00\x18ftypisomvideo')
    with pytest.raises(TimeoutError):
        run_wan_video(provider, store, prompt="walk", downloader=download)
    result = run_wan_video(provider, store, prompt="walk", downloader=download)
    assert run_wan_video(provider, store, prompt="walk", downloader=download) == result
    assert provider.submit.call_count == 1
    assert provider.wait.call_count == 2
    download.assert_called_once()
    assert store.video_preview(result['execution']['artifact_ref']).endswith('.mp4')


def test_wan_unknown_submission_is_not_repeated(tmp_path):
    provider = Mock(base_url="https://example.test")
    provider.submit.side_effect = TimeoutError("submit interrupted")
    store = ArtifactRegistry(tmp_path)
    with pytest.raises(TimeoutError):
        run_wan_video(provider, store, prompt="walk")
    with pytest.raises(RuntimeError, match="outcome unknown"):
        run_wan_video(provider, store, prompt="walk")
    assert provider.submit.call_count == 1


def test_invalid_wan_duration_never_submits(tmp_path):
    provider = Mock()
    for duration in [0, 11, True, 1.5]:
        with pytest.raises(ValueError):
            run_wan_video(provider, ArtifactRegistry(tmp_path), prompt="walk", duration_seconds=duration)
    provider.submit.assert_not_called()


def test_new_batch_generates_new_task_but_resume_does_not(tmp_path):
    provider = Mock(base_url="https://example.test")
    provider.submit.side_effect = ['first', 'second']
    provider.wait.side_effect = lambda task: {'state':'succeeded', 'task_id':task, 'urls':['https://example.test/' + task]}
    for batch in ['one', 'one', 'two']:
        run_wan_video(provider, ArtifactRegistry(tmp_path), prompt='same prompt', batch_id=batch,
                      downloader=lambda url: url.encode())
    assert provider.submit.call_count == 2
    assert all('batch_id' not in call.kwargs for call in provider.submit.call_args_list)


def test_original_wan_parameters_and_resume(monkeypatch, tmp_path):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'story-video-agent'))
    from story_video_agent.dashscope import DashScopeWanAdapter
    submitted = []
    completed = False
    def call(self, method, path, payload=None):
        if method == 'POST':
            submitted.append(payload)
            return {'output': {'task_id': 'original-wan'}}
        assert path == '/tasks/original-wan'
        return {'output': {'task_status': 'SUCCEEDED' if completed else 'RUNNING',
                           'video_url': 'https://example.test/video.mp4'}}
    monkeypatch.setattr(DashScopeWanAdapter, '_call', call)
    def run():
        return run_wan_video(DashScopeWanAdapter('test', max_polls=1), ArtifactRegistry(tmp_path),
            prompt='walk', model='wan2.1-t2v-turbo', size='720*1280', duration_seconds=5,
            negative_prompt='blur', downloader=lambda url: b'\x00\x00\x00\x18ftypisomvideo')
    with pytest.raises(TimeoutError):
        run()
    completed = True
    assert run()['status'] == 'succeeded'
    assert submitted == [{'model': 'wan2.1-t2v-turbo', 'input': {'prompt': 'walk', 'negative_prompt': 'blur'},
                          'parameters': {'size': '720*1280', 'duration': 5, 'prompt_extend': True}}]


def test_default_wan_model_rejects_unsupported_duration_before_submission(tmp_path):
    import pytest
    from unittest.mock import Mock
    from story_media_orchestrator.wan_video import run_wan_video
    from story_media_orchestrator.registry import ArtifactRegistry
    provider = Mock()
    with pytest.raises(ValueError, match='5'):
        run_wan_video(provider, ArtifactRegistry(tmp_path), prompt='test', duration_seconds=7)
    provider.submit.assert_not_called()
    assert not (tmp_path / 'video-tasks').exists()
