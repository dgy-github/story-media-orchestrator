from PIL import Image
import pytest
from story_media_orchestrator.manifest import ProjectManifest


def test_selection_preserves_audio_and_invalidates_video_and_approval(tmp_path):
    from story_media_orchestrator.candidates import select_candidate
    manifest = ProjectManifest.create('test', 'walk')
    shot = manifest.shots[0]
    image = tmp_path/'chosen.png'
    Image.new('RGB',(64,64),'red').save(image)
    shot.candidates = [{'id':'one', 'path':str(image), 'quality':{'decision':'unavailable'}}]
    shot.assets = {'audio':'voice.wav', 'video':'old.mp4'}
    manifest.approvals['assets'] = 'old'
    select_candidate(manifest, shot.id, 'one')
    assert shot.assets == {'audio':'voice.wav','image':str(image)}
    assert 'assets' not in manifest.approvals
    shot.candidates[0]['quality'] = {'decision':'failed'}
    with pytest.raises(ValueError):
        select_candidate(manifest, shot.id, 'one')


def test_generated_candidates_bind_to_story_and_keep_quality(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from story_media_orchestrator.candidates import generate_candidates, select_candidate
    from story_media_orchestrator.registry import ArtifactRegistry
    from io import BytesIO
    store = ArtifactRegistry(tmp_path/'artifacts')
    data = BytesIO(); Image.new('RGB',(64,64),'blue').save(data,format='PNG')
    ref = store.put_bytes(data.getvalue())
    class Adapter:
        def run(self, scene, spans, **kwargs):
            assert kwargs['candidate_count'] == 2
            return {'candidate_results':[{'status':'succeeded','artifact_ref':ref}, {'status':'failed'}]}
    monkeypatch.setattr('story_media_orchestrator.providers.build_project_image_provider', lambda *a: SimpleNamespace(adapter=Adapter(),registry=store))
    monkeypatch.setattr('story_media_orchestrator.quality.evaluate_artifact', lambda *a: {'decision':'warning'})
    manifest = ProjectManifest.create('test','walk')
    manifest.image_source = 'dashscope'
    generate_candidates(manifest,tmp_path,'shot-01',2)
    shot = manifest.shots[0]
    assert len(shot.candidates) == 1
    assert shot.candidates[0]['quality']['decision'] == 'warning'
    assert shot.candidate_batch  # Partial batch must remain resumable.
    shot.text = 'different scene'
    with pytest.raises(ValueError, match='changed'):
        select_candidate(manifest,shot.id,shot.candidates[0]['id'])
