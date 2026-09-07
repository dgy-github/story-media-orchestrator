from pathlib import Path
import subprocess

from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.preview import render_preview


def test_render_output_uses_video_instead_of_source_image(tmp_path):
    from PIL import Image
    ffmpeg = find_ffmpeg()
    class Images:
        def generate(self, shot, output):
            Image.new('RGB', (64, 64), 'blue').save(output)
            return output
    class Videos:
        def generate(self, shot, image, output):
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=red:s=1280x720:r=25',
                            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                            '-t', '1', '-c:v', 'libx264', '-c:a', 'aac', '-pix_fmt', 'yuv420p', str(output)],
                           check=True, capture_output=True)
            return output
    manifest = ProjectManifest.create('p', '测试。')
    manifest.mode = 'render'
    output = render_preview(manifest, tmp_path, image_provider=Images(), video_provider=Videos(), real_preview=True)
    pixel = subprocess.run([ffmpeg, '-i', str(output), '-vf', 'scale=1:1', '-frames:v', '1',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
    assert pixel[0] > 200 and pixel[2] < 40, 'output used blue source image instead of red video'
    assert not list(tmp_path.glob('.segments-*'))
    assert manifest.status == 'done'
    import array
    def audio_samples(path):
        data = subprocess.run([ffmpeg, '-i', str(path), '-map', '0:a:0', '-f', 's16le',
                               '-ac', '1', '-ar', '16000', '-'], check=True, capture_output=True).stdout
        return array.array('h', data)
    assert max(abs(value) for value in audio_samples(output)[:16000]) > 1000, 'final MP4 lost model audio'

    # Changing only output mode must retain assets but actually use images.
    from unittest.mock import Mock
    manifest.mode = 'preview'
    image_provider = Mock()
    image_provider.generate.side_effect = AssertionError('mode change regenerated image')
    output = render_preview(manifest, tmp_path, image_provider=image_provider, real_preview=True)
    pixel = subprocess.run([ffmpeg, '-i', str(output), '-vf', 'crop=100:100,scale=1:1', '-frames:v', '1',
                            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True).stdout
    assert pixel[2] > 200 and pixel[0] < 40, 'preview retained old rendered video'
    image_provider.generate.assert_not_called()
    assert not any(audio_samples(output)), 'image preview retained model audio'
