from pathlib import Path
import base64
from PIL import Image
from story_media_orchestrator.registry import ArtifactRegistry
from story_media_orchestrator.wan_video import run_wan_video


def test_keyframes_are_sent_and_changed_frame_invalidates_receipt(tmp_path):
    first, last = tmp_path/'first.png', tmp_path/'last.png'
    Image.new('RGB', (64,64), 'red').save(first)
    Image.new('RGB', (64,64), 'blue').save(last)
    class Provider:
        base_url = 'https://example.test'
        calls = []
        def _call(self, method, path, payload):
            self.calls.append((path, payload))
            return {'output': {'task_id': str(len(self.calls))}}
        def wait(self, task_id):
            return {'state':'succeeded','task_id':task_id,'urls':['https://example.test/video']}
    provider = Provider()
    def run():
        return run_wan_video(provider, ArtifactRegistry(tmp_path/'artifacts'), prompt='turn',
            model='wan2.2-kf2v-flash', first_frame=first, last_frame=last,
            downloader=lambda _: b'video')
    assert run()['conditioning_mode'] == 'first_last_frame'
    run()
    assert len(provider.calls) == 1
    path, payload = provider.calls[0]
    assert path == '/services/aigc/image2video/video-synthesis'
    assert base64.b64decode(payload['input']['first_frame_url'].split(',')[1]) == first.read_bytes()
    assert base64.b64decode(payload['input']['last_frame_url'].split(',')[1]) == last.read_bytes()
    Image.new('RGB', (64,64), 'green').save(last)
    run()
    assert len(provider.calls) == 2
