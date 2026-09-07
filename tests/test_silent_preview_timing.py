import wave

from story_media_orchestrator.tts import FakeTTSProvider


def test_short_storyboard_action_is_not_one_second(tmp_path):
    output = FakeTTSProvider().synthesize('推开门。', tmp_path / 'short.wav')
    with wave.open(str(output)) as audio:
        assert audio.getnframes() / audio.getframerate() == 3


def test_long_storyboard_text_still_extends_duration(tmp_path):
    output = FakeTTSProvider().synthesize('字' * 60, tmp_path / 'long.wav')
    with wave.open(str(output)) as audio:
        assert audio.getnframes() / audio.getframerate() == 5
