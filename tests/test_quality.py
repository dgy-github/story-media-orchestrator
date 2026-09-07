import json
import pytest
from story_media_orchestrator.quality import evaluate_artifact

class Response:
    def __init__(self, value): self.value = value
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps(self.value).encode()

@pytest.mark.parametrize('patch', [
    {'decision':'approved'}, {'stage':'video'}, {'reason':{}}, {'failures':'none'}
])
def test_invalid_quality_result_is_unavailable(monkeypatch, patch):
    monkeypatch.setenv('STORY_QUALITY_EVALUATOR_URL','http://localhost/evaluate')
    value = {'schema':'quality-evaluation/v1','decision':'passed','stage':'image',**patch}
    monkeypatch.setattr('story_media_orchestrator.quality.request.urlopen',lambda *a,**k:Response(value))
    result = evaluate_artifact({},'image')
    assert result['decision'] == 'unavailable'
    assert result['stage'] == 'image'


def test_legacy_result_without_stage_gets_current_stage(monkeypatch):
    monkeypatch.setenv('STORY_QUALITY_EVALUATOR_URL','http://localhost/evaluate')
    value = {'schema':'quality-evaluation/v1','decision':'warning','reason':'needs review'}
    monkeypatch.setattr('story_media_orchestrator.quality.request.urlopen',lambda *a,**k:Response(value))
    assert evaluate_artifact({},'image') == {**value,'stage':'image'}
