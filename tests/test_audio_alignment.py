import wave
from story_media_orchestrator.local_media import find_ffmpeg
from story_media_orchestrator.manifest import Shot
from story_media_orchestrator.segments import align_audio_segments


def test_short_cached_audio_is_padded_without_rewriting_source(tmp_path):
    source = tmp_path / 'original.wav'
    with wave.open(str(source), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\x01\x00' * 16000)
    original = source.read_bytes()
    directory = tmp_path / 'aligned'
    directory.mkdir()
    paths = align_audio_segments(find_ffmpeg(), [Shot('s', 'text', duration=3, assets={'audio':str(source)})], directory)
    with wave.open(str(paths[0])) as aligned:
        assert aligned.getnframes() == 48000
        samples = aligned.readframes(48000)
        assert samples[:32000] == b'\x01\x00' * 16000
        assert samples[32000:] == b'\x00\x00' * 32000
    assert source.read_bytes() == original


def test_model_audio_is_retained_only_for_render_shots(tmp_path):
    import subprocess
    import array
    ffmpeg = find_ffmpeg()
    silent = tmp_path / 'silent.wav'
    with wave.open(str(silent), 'wb') as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000)
        output.writeframes(bytes(32000))
    video = tmp_path / 'model.mp4'
    subprocess.run([ffmpeg, '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:d=1',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(video)],
                   check=True, capture_output=True)
    original = video.read_bytes()
    directory = tmp_path / 'aligned'; directory.mkdir()
    shot = Shot('s', 'text', duration=2, mode='render', assets={'audio':str(silent), 'video':str(video)})
    target = align_audio_segments(ffmpeg, [shot], directory)[0]
    with wave.open(str(target)) as result:
        assert result.getnframes() == 32000
        samples = array.array('h', result.readframes(32000))
        assert max(abs(value) for value in samples[:16000]) > 1000
        assert all(value == 0 for value in samples[20000:])
    shot.mode = 'preview'
    target = align_audio_segments(ffmpeg, [shot], directory)[0]
    with wave.open(str(target)) as result:
        assert not any(result.readframes(32000))
    assert video.read_bytes() == original
