from felt_voice import stt
from felt_voice.config import VoiceConfig


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeWhisperModel:
    def __init__(self, model_name, device, compute_type):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path):
        return ([_FakeSegment("  hello  "), _FakeSegment("world  ")], None)


def _reset_cache():
    stt._model = None
    stt._loaded_key = None


def test_get_model_reused_for_same_config(monkeypatch):
    created = []

    def fake_ctor(model_name, device, compute_type):
        instance = _FakeWhisperModel(model_name, device, compute_type)
        created.append(instance)
        return instance

    monkeypatch.setattr(stt, "WhisperModel", fake_ctor)
    _reset_cache()

    config = VoiceConfig(whisper_model="base.en", whisper_device="cpu", whisper_compute_type="int8")
    first = stt._get_model(config)
    second = stt._get_model(config)

    assert first is second
    assert len(created) == 1


def test_get_model_reloaded_when_config_changes(monkeypatch):
    created = []

    def fake_ctor(model_name, device, compute_type):
        instance = _FakeWhisperModel(model_name, device, compute_type)
        created.append(instance)
        return instance

    monkeypatch.setattr(stt, "WhisperModel", fake_ctor)
    _reset_cache()

    small = VoiceConfig(whisper_model="base.en", whisper_device="cpu", whisper_compute_type="int8")
    large = VoiceConfig(whisper_model="small.en", whisper_device="cpu", whisper_compute_type="int8")

    first = stt._get_model(small)
    second = stt._get_model(large)

    assert first is not second
    assert len(created) == 2


def test_transcribe_joins_and_strips_segment_text(monkeypatch):
    monkeypatch.setattr(stt, "WhisperModel", lambda model_name, device, compute_type: _FakeWhisperModel(model_name, device, compute_type))
    _reset_cache()

    config = VoiceConfig(whisper_model="base.en", whisper_device="cpu", whisper_compute_type="int8")
    result = stt.transcribe("irrelevant.wav", config)

    assert result == "hello world"


class _FakeParakeet:
    def recognize(self, audio, sample_rate):
        assert sample_rate == 16000
        return " Bora lá jogar. "


def test_parakeet_engine(monkeypatch):
    loads = []
    monkeypatch.setattr(stt, "_parakeet", None)
    monkeypatch.setattr(stt, "_parakeet_key", None)

    def fake_get(config):
        loads.append(config.parakeet_model)
        return _FakeParakeet()

    monkeypatch.setattr(stt, "_get_parakeet", fake_get)
    monkeypatch.setattr(stt, "_decode_16k", lambda path: b"")

    assert stt.transcribe("x.wav", VoiceConfig(stt_engine="parakeet")) == "Bora lá jogar."
    assert loads == ["nemo-parakeet-tdt-0.6b-v3"]
