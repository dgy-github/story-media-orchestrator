import sys
import wave
import pytest


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows speech integration')
def test_windows_speech_produces_audible_wav(tmp_path):
    from story_media_orchestrator.tts import WindowsTTSProvider
    target = WindowsTTSProvider().synthesize('城市苏醒，信使踏上新的旅程。', tmp_path / '中文 voice.wav')
    with wave.open(str(target), 'rb') as stream:
        assert stream.getnframes() > stream.getframerate()
        assert any(stream.readframes(stream.getnframes()))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows speech integration')
def test_cli_speech_setting_survives_resume(tmp_path):
    from story_media_orchestrator.cli import main
    from story_media_orchestrator.manifest import ProjectManifest
    main(['create', str(tmp_path), '城市苏醒。'])
    main(['run', str(tmp_path), '--real-preview', '--tts-source', 'windows'])
    first = ProjectManifest.load(tmp_path / 'project.json')
    assert first.tts_source == 'windows'
    with wave.open(first.shots[0].assets['audio'], 'rb') as stream:
        assert any(stream.readframes(stream.getnframes()))
    main(['resume', str(tmp_path)])
    restored = ProjectManifest.load(tmp_path / 'project.json')
    assert restored.shots[0].attempts == first.shots[0].attempts
