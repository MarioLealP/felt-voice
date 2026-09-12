from felt_voice import tts
from felt_voice.config import VoiceConfig


class _FakeVoice:
    def __init__(self, path):
        self.path = path

    def synthesize_wav(self, text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 8)


def _reset_cache():
    tts._voice = None
    tts._loaded_name = None


def test_get_voice_reused_for_same_name(monkeypatch, tmp_path):
    created = []

    def fake_load(path):
        instance = _FakeVoice(path)
        created.append(instance)
        return instance

    monkeypatch.setattr(tts.PiperVoice, "load", staticmethod(fake_load))
    _reset_cache()

    config = VoiceConfig(piper_voice="en_US-lessac-medium", piper_voices_dir=str(tmp_path))
    first = tts._get_voice(config)
    second = tts._get_voice(config)

    assert first is second
    assert len(created) == 1


def test_get_voice_reloaded_when_voice_name_changes(monkeypatch, tmp_path):
    created = []

    def fake_load(path):
        instance = _FakeVoice(path)
        created.append(instance)
        return instance

    monkeypatch.setattr(tts.PiperVoice, "load", staticmethod(fake_load))
    _reset_cache()

    config_a = VoiceConfig(piper_voice="en_US-lessac-medium", piper_voices_dir=str(tmp_path))
    config_b = VoiceConfig(piper_voice="en_US-amy-medium", piper_voices_dir=str(tmp_path))

    first = tts._get_voice(config_a)
    second = tts._get_voice(config_b)

    assert first is not second
    assert len(created) == 2


def test_synthesize_returns_nonempty_wav_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(tts.PiperVoice, "load", staticmethod(lambda path: _FakeVoice(path)))
    _reset_cache()

    config = VoiceConfig(piper_voice="en_US-lessac-medium", piper_voices_dir=str(tmp_path))
    audio = tts.synthesize("hello there", config)

    assert isinstance(audio, bytes)
    assert audio.startswith(b"RIFF")
    assert len(audio) > 0
