from felt_voice import tts
from felt_voice.config import VoiceConfig


class _FakeVoice:
    def __init__(self, path):
        self.path = path

    def synthesize_wav(self, text, wav_file):
        self.last_text = text
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 8)


def _reset_cache():
    tts._voice = None
    tts._loaded_name = None
    tts._chatterbox = None
    tts._chatterbox_failed = False


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

    config = VoiceConfig(piper_voice="en_US-lessac-medium", piper_voices_dir=str(tmp_path), tts_engine="piper")
    audio, engine = tts.synthesize("hello there", config)

    assert engine == "piper"

    assert isinstance(audio, bytes)
    assert audio.startswith(b"RIFF")
    assert len(audio) > 0


class _FakeChatterbox:
    def __init__(self, fail=False):
        self.fail = fail
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return b"RIFF-chatterbox"


def _config(tmp_path, **kwargs):
    return VoiceConfig(piper_voice="v", piper_voices_dir=str(tmp_path), **kwargs)


def _patch(monkeypatch, loader):
    monkeypatch.setattr(tts.PiperVoice, "load", staticmethod(lambda path: _FakeVoice(path)))
    monkeypatch.setattr(tts, "_load_chatterbox", loader)
    _reset_cache()


def test_chatterbox_used_when_selected(monkeypatch, tmp_path):
    fake = _FakeChatterbox()
    _patch(monkeypatch, lambda config: fake)

    audio, engine = tts.synthesize("[laugh] Hi there.", _config(tmp_path, tts_engine="chatterbox"))

    assert (audio, engine) == (b"RIFF-chatterbox", "chatterbox")
    assert fake.texts == ["Hi there."]


def test_chatterbox_loaded_once(monkeypatch, tmp_path):
    loads = []
    _patch(monkeypatch, lambda config: loads.append(1) or _FakeChatterbox())
    config = _config(tmp_path, tts_engine="chatterbox")

    tts.init_engines(config)
    tts.synthesize("One.", config)
    tts.synthesize("Two.", config)

    assert len(loads) == 1


def test_piper_fallback_when_chatterbox_fails_to_load(monkeypatch, tmp_path, caplog):
    def broken(config):
        raise FileNotFoundError("no reference clip")

    _patch(monkeypatch, broken)
    config = _config(tmp_path, tts_engine="chatterbox")

    audio, engine = tts.synthesize("Hello.", config)

    assert engine == "piper"
    assert audio.startswith(b"RIFF")
    assert tts.active_engine(config) == "piper"
    assert "Chatterbox failed to load" in caplog.text


def test_piper_fallback_when_chatterbox_errors_on_a_request(monkeypatch, tmp_path, caplog):
    fake = _FakeChatterbox(fail=True)
    _patch(monkeypatch, lambda config: fake)
    config = _config(tmp_path, tts_engine="chatterbox", allow_tags=True)

    audio, engine = tts.synthesize("[sigh] Fine.", config)

    assert engine == "piper"
    assert audio.startswith(b"RIFF")
    assert fake.texts == ["[sigh] Fine."]
    assert tts._voice.last_text == "Fine."  # Piper never gets a tag
    assert "Chatterbox errored on a request" in caplog.text


def test_piper_engine_never_loads_chatterbox(monkeypatch, tmp_path):
    def must_not_load(config):
        raise AssertionError("chatterbox loaded")

    _patch(monkeypatch, must_not_load)

    _, engine = tts.synthesize("Hello.", _config(tmp_path, tts_engine="piper"))

    assert engine == "piper"


class _FakeResp:
    def __init__(self, content=b"RIFF-indextts", fail=False):
        self.content = content
        self.fail = fail

    def raise_for_status(self):
        if self.fail:
            raise RuntimeError("503")


def _fake_client(post):
    """httpx.Client stand-in (tts._indextts opens one per request since B5.7)."""

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json, extensions=None):
            return post(url, json)

    return Client


def test_indextts_used_when_selected_and_gets_the_emotion(monkeypatch, tmp_path):
    calls = []
    _patch(monkeypatch, lambda config: _FakeChatterbox())
    monkeypatch.setattr(tts.httpx, "Client", _fake_client(lambda url, json: calls.append((url, json)) or _FakeResp()))
    config = _config(tmp_path, tts_engine="indextts", indextts_url="http://tts:1")

    audio, engine = tts.synthesize("[sigh] Go to bed.", config, emotion="sleepy")

    assert (audio, engine) == (b"RIFF-indextts", "indextts")
    assert calls == [("http://tts:1/tts", {"text": "Go to bed.", "emotion": "sleepy"})]
    assert tts.active_engine(config) == "indextts"


def test_piper_fallback_when_indextts_errors(monkeypatch, tmp_path, caplog):
    _patch(monkeypatch, lambda config: _FakeChatterbox())
    monkeypatch.setattr(tts.httpx, "Client", _fake_client(lambda url, json: _FakeResp(fail=True)))

    audio, engine = tts.synthesize("Hello.", _config(tmp_path, tts_engine="indextts"))

    assert engine == "piper"
    assert audio.startswith(b"RIFF")
    assert "IndexTTS errored" in caplog.text


def test_indextts_engine_never_loads_chatterbox(monkeypatch, tmp_path):
    loads = []
    _patch(monkeypatch, lambda config: loads.append(1) or _FakeChatterbox())
    monkeypatch.setattr(tts.httpx, "Client", _fake_client(lambda url, json: _FakeResp()))
    config = _config(tmp_path, tts_engine="indextts")

    tts.init_engines(config)
    tts.synthesize("Hi.", config)

    assert loads == []
