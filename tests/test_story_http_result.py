import json
import pytest
from story_media_orchestrator.adapters import HttpStoryCampaignAdapter

class Response:
    status = 200
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.body).encode()

@pytest.mark.parametrize("wrapped", [False, True])
def test_accepts_native_and_wrapped_result(monkeypatch, wrapped):
    package = {"schema":"story-package/v1", "scenes":[]}
    responses = iter([Response({"run_id":"run_test"}), Response({"package":package} if wrapped else package)])
    monkeypatch.setattr("story_media_orchestrator.adapters.http_request.urlopen", lambda *a, **k: next(responses))
    assert HttpStoryCampaignAdapter("http://localhost:8765", "x"*32, poll_interval=0).run({}) == package

def test_invalid_success_response_fails_without_polling(monkeypatch):
    responses = iter([Response({"run_id":"run_test"}), Response({"unexpected":True})])
    monkeypatch.setattr("story_media_orchestrator.adapters.http_request.urlopen", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="invalid story package"):
        HttpStoryCampaignAdapter("http://localhost:8765", "x"*32, poll_interval=0).run({})


def test_terminal_failure_stops_result_poll(monkeypatch):
    import io
    from urllib.error import HTTPError
    event = {"run_id":"run_test","seq":1,"event_type":"run.failed"}
    class Stream(Response):
        def __init__(self): self.stream = io.BytesIO(b"data: " + json.dumps(event).encode() + b"\n\n: replay-complete\n\n")
        def readline(self, limit): return self.stream.readline(limit)
    calls = []
    def request(req, **kwargs):
        calls.append(req.full_url)
        if len(calls) == 1: return Response({"run_id":"run_test"})
        if req.full_url.endswith("/result"): raise HTTPError(req.full_url,404,"pending",{},None)
        return Stream()
    monkeypatch.setattr("story_media_orchestrator.adapters.http_request.urlopen", request)
    with pytest.raises(RuntimeError, match="run.failed"):
        HttpStoryCampaignAdapter("http://localhost:8765", "x"*32, poll_interval=0).run({})
    assert len(calls) == 3

def test_story_budget_can_wait_beyond_default_poll_limit(monkeypatch):
    from urllib.error import HTTPError
    polls = 0
    def request(req, **kwargs):
        nonlocal polls
        if req.full_url.endswith('/v1/runs'): return Response({'run_id':'run_budget'})
        polls += 1
        if polls <= 301: raise HTTPError(req.full_url,404,'pending',{},None)
        return Response({'schema':'story-package/v1','scenes':[]})
    monkeypatch.setattr('story_media_orchestrator.adapters.http_request.urlopen', request)
    monkeypatch.setattr('story_media_orchestrator.adapters.time.sleep', lambda _: None)
    adapter = HttpStoryCampaignAdapter('http://localhost:8765','x'*32)
    monkeypatch.setattr(adapter,'_check_terminal_events',lambda run, cursor:cursor)
    result = adapter.run({'job':{'job_id':'job_budget','budget':{'deadline_seconds':900}}})
    assert result['schema'] == 'story-package/v1'
    assert polls == 302


def test_timeout_reports_task_id_without_resubmit(monkeypatch):
    from urllib.error import HTTPError
    submitted = []
    def request(req, **kwargs):
        if req.full_url.endswith('/v1/runs'):
            submitted.append(req)
            return Response({'run_id':'run_pending'})
        raise HTTPError(req.full_url,404,'pending',{},None)
    monkeypatch.setattr('story_media_orchestrator.adapters.http_request.urlopen',request)
    adapter = HttpStoryCampaignAdapter('http://localhost:8765','x'*32,max_polls=1,poll_interval=0)
    monkeypatch.setattr(adapter,'_check_terminal_events',lambda run,cursor:cursor)
    with pytest.raises(TimeoutError,match='run_id=run_pending'):
        adapter.run({'job':{'job_id':'original-job'}})
    assert len(submitted) == 1
    assert submitted[0].get_header('Idempotency-key') == 'original-job'


def test_terminal_failure_preserves_node_and_provider_reason(monkeypatch):
    import io
    event = {"run_id": "run_test", "seq": 1, "event_type": "run.failed",
             "payload": {"error": "fixed workflow failed: provider_or_task_failure at t09: RuntimeError: Rust capability rejected request: HTTP 502"}}
    class Stream(Response):
        def __init__(self): self.stream = io.BytesIO(b"data: " + json.dumps(event).encode() + b"\n\n: replay-complete\n\n")
        def readline(self, limit): return self.stream.readline(limit)
    monkeypatch.setattr("story_media_orchestrator.adapters.http_request.urlopen", lambda *a, **k: Stream())
    adapter = HttpStoryCampaignAdapter("http://localhost:8765", "x" * 32)
    with pytest.raises(RuntimeError, match="t09.*HTTP 502"):
        adapter._check_terminal_events("run_test", 0)
