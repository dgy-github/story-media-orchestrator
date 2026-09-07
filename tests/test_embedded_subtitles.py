import subprocess

from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.manifest import ProjectManifest
from story_media_orchestrator.preview import render_preview


def test_mp4_contains_story_subtitles_with_timeline(tmp_path):
    manifest = ProjectManifest.create('p', '信使推开门。老人递出钥匙。')
    output = render_preview(manifest, tmp_path, real_preview=True)
    extracted = subprocess.run([find_ffmpeg(), '-i', str(output), '-map', '0:s:0',
                                '-f', 'srt', '-'], check=True, capture_output=True).stdout.decode('utf-8')
    assert '信使推开门' in extracted
    assert '老人递出钥匙' in extracted
    assert '00:00:00,000 --> 00:00:03,000' in extracted
    assert '00:00:03,000 --> 00:00:06,000' in extracted
