import subprocess
from PIL import Image
from story_media_orchestrator.manifest import Shot
from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.segments import render_segments


def test_motion_changes_pixels_and_fade_keeps_duration(tmp_path):
    source = tmp_path / 'image.png'
    image = Image.new('RGB', (320, 180), 'white')
    for x in range(110, 140):
        for y in range(60, 120):
            image.putpixel((x, y), (255, 0, 0))
    image.save(source)
    shot = Shot('shot-01', 'test', duration=2, assets={'image': str(source)})
    shot.motion = 'push_in'
    shot.transition = 'fade'
    output = render_segments(find_ffmpeg(), [shot], tmp_path)[0]
    frames = subprocess.run([find_ffmpeg(), '-v', 'error', '-i', str(output),
        '-vf', 'scale=32:18,format=gray', '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    assert len(frames) == 50 * 32 * 18
    assert max(frames[:576]) < 20  # Allow codec/range rounding near black.
    assert frames[15*576:16*576] != frames[30*576:31*576]


def test_fractional_timeline_uses_same_frame_boundaries():
    from story_media_orchestrator.segments import normalize_durations
    shots = [Shot(str(i),'test',duration=1.01) for i in range(100)]
    normalize_durations(shots)
    assert abs(sum(s.duration for s in shots) - 101) < 0.000001
    assert all(abs(s.duration*25-round(s.duration*25)) < 0.000001 for s in shots)
