from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.video_checkpoint import execute_with_receipt


def test_resume_uses_original_submission_and_cached_result(tmp_path):
    store = ArtifactRegistry(tmp_path)
    client = SimpleNamespace(base_url="http://localhost:8188", submit=Mock(return_value="task-1"))
    def pending(transport):
        assert transport.submit({}) == "task-1"
        raise TimeoutError("waiting")
    with pytest.raises(TimeoutError): execute_with_receipt(client, {}, store, pending)
    def complete(transport):
        assert transport.submit({}) == "task-1"
        return {"state":"succeeded", "prompt_id":"task-1", "artifact_ref":store.put(b"video")}
    result = execute_with_receipt(client, {}, store, complete)
    assert client.submit.call_count == 1
    assert execute_with_receipt(client, {}, store, Mock(side_effect=AssertionError("cached"))) == result


def test_unknown_submission_never_auto_resubmits(tmp_path):
    client = SimpleNamespace(base_url="http://localhost:8188", submit=Mock(side_effect=TimeoutError("network")))
    store = ArtifactRegistry(tmp_path)
    with pytest.raises(TimeoutError): execute_with_receipt(client, {}, store, lambda t: t.submit({}))
    with pytest.raises(RuntimeError, match="outcome unknown"):
        execute_with_receipt(client, {}, store, lambda t: t.submit({}))
    assert client.submit.call_count == 1

def test_concurrent_execution_does_not_submit_twice(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    store = ArtifactRegistry(tmp_path)
    client = SimpleNamespace(base_url="http://localhost:8188", submit=Mock(return_value="task-one"))
    entered, release = Event(), Event()
    def running(transport):
        transport.submit({})
        entered.set()
        assert release.wait(5)
        return {"state":"succeeded", "prompt_id":"task-one", "artifact_ref":store.put(b"video")}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(execute_with_receipt, client, {}, store, running)
        assert entered.wait(5)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                execute_with_receipt(client, {}, store, running)
        finally:
            release.set()
        first.result(timeout=5)
    assert client.submit.call_count == 1


def test_corrupt_cached_video_does_not_trigger_resubmission(tmp_path):
    store = ArtifactRegistry(tmp_path)
    client = SimpleNamespace(base_url="http://localhost:8188", submit=Mock(return_value="task-one"))
    def complete(transport):
        transport.submit({})
        return {"state":"succeeded", "prompt_id":"task-one", "artifact_ref":store.put(b"video")}
    result = execute_with_receipt(client, {}, store, complete)
    (tmp_path / result['artifact_ref'].rsplit('/', 1)[-1]).write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match="integrity"):
        execute_with_receipt(client, {}, store, Mock(side_effect=AssertionError("must not execute")))
    assert client.submit.call_count == 1

def test_unverified_result_keeps_receipt_resumable(tmp_path):
    import json
    store = ArtifactRegistry(tmp_path)
    client = SimpleNamespace(base_url="http://localhost:8188", submit=Mock(return_value="task-one"))
    def wrong(transport):
        transport.submit({})
        return {"state":"succeeded", "prompt_id":"another-task", "artifact_ref":store.put(b"video")}
    with pytest.raises(RuntimeError, match="does not match"):
        execute_with_receipt(client, {}, store, wrong)
    receipt = json.loads(next((tmp_path/'video-tasks').glob('*.json')).read_text())
    assert receipt['state'] == 'accepted'
    assert receipt['prompt_id'] == 'task-one'


@pytest.mark.parametrize('status', [400, 401, 403, 429])
def test_explicit_rejection_can_resume_after_correction(tmp_path, status):
    import json
    from urllib.error import HTTPError
    error = RuntimeError('provider rejected')
    error.__cause__ = HTTPError('https://example.invalid', status, 'rejected', {}, None)
    client = SimpleNamespace(base_url='wan:https://example.invalid', submit=Mock(side_effect=[error, 'accepted']))
    store = ArtifactRegistry(tmp_path)
    with pytest.raises(RuntimeError):
        execute_with_receipt(client, {}, store, lambda t: t.submit({}))
    receipt = json.loads(next((tmp_path / 'video-tasks').glob('*.json')).read_text())
    assert receipt['state'] == 'rejected'
    assert receipt['http_status'] == status
    def complete(transport):
        task = transport.submit({})
        return {'state': 'succeeded', 'prompt_id': task, 'artifact_ref': store.put(b'video')}
    execute_with_receipt(client, {}, store, complete)
    assert client.submit.call_count == 2


@pytest.mark.parametrize('status', [408, 500, 502])
def test_ambiguous_http_failure_blocks_resubmission(tmp_path, status):
    from urllib.error import HTTPError
    client = SimpleNamespace(base_url='wan:https://example.invalid', submit=Mock(side_effect=HTTPError('https://example.invalid', status, 'error', {}, None)))
    store = ArtifactRegistry(tmp_path)
    with pytest.raises(HTTPError):
        execute_with_receipt(client, {}, store, lambda t: t.submit({}))
    with pytest.raises(RuntimeError, match='outcome unknown'):
        execute_with_receipt(client, {}, store, lambda t: t.submit({}))
    assert client.submit.call_count == 1
