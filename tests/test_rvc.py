from felt_voice import rvc
from felt_voice.config import VoiceConfig


class _FakeConverter:
    def __init__(self, only_cpu, hubert_path, rmvpe_path):
        self.only_cpu = only_cpu
        self.confs = []
        self.calls = []

    def apply_conf(self, **kwargs):
        self.confs.append(kwargs)

    def __call__(self, path, tag, overwrite=False):
        # Mimics the real library's overwrite=True behavior (confirmed
        # 2026-09-12 via a live spike): converted audio is written back
        # over the same path, not a new file.
        self.calls.append((path, tag, overwrite))
        with open(path, "r+b") as f:
            data = f.read()
            f.seek(0)
            f.write(data + b"-converted")
            f.truncate()
        return [path]


def _reset_cache():
    rvc._converter = None
    rvc._loaded_key = None


def test_get_converter_reused_for_same_model_and_index(monkeypatch):
    created = []

    def fake_ctor(only_cpu, hubert_path, rmvpe_path):
        instance = _FakeConverter(only_cpu, hubert_path, rmvpe_path)
        created.append(instance)
        return instance

    monkeypatch.setattr(rvc, "BaseLoader", fake_ctor)
    _reset_cache()

    config = VoiceConfig(rvc_model_path="model.pth", rvc_index_path="model.index")
    first = rvc._get_converter(config)
    second = rvc._get_converter(config)

    assert first is second
    assert len(created) == 1


def test_get_converter_reloaded_when_model_path_changes(monkeypatch):
    created = []

    def fake_ctor(only_cpu, hubert_path, rmvpe_path):
        instance = _FakeConverter(only_cpu, hubert_path, rmvpe_path)
        created.append(instance)
        return instance

    monkeypatch.setattr(rvc, "BaseLoader", fake_ctor)
    _reset_cache()

    config_a = VoiceConfig(rvc_model_path="a.pth", rvc_index_path="a.index")
    config_b = VoiceConfig(rvc_model_path="b.pth", rvc_index_path="b.index")

    first = rvc._get_converter(config_a)
    second = rvc._get_converter(config_b)

    assert first is not second
    assert len(created) == 2


def test_convert_returns_the_converted_bytes(monkeypatch):
    monkeypatch.setattr(
        rvc, "BaseLoader", lambda only_cpu, hubert_path, rmvpe_path: _FakeConverter(only_cpu, hubert_path, rmvpe_path)
    )
    _reset_cache()

    config = VoiceConfig(rvc_model_path="model.pth", rvc_index_path="model.index")
    result = rvc.convert(b"RIFF-fake-wav-bytes", config)

    assert result == b"RIFF-fake-wav-bytes-converted"


def test_convert_passes_only_cpu_setting_through(monkeypatch):
    seen = {}

    def fake_ctor(only_cpu, hubert_path, rmvpe_path):
        seen["only_cpu"] = only_cpu
        return _FakeConverter(only_cpu, hubert_path, rmvpe_path)

    monkeypatch.setattr(rvc, "BaseLoader", fake_ctor)
    _reset_cache()

    config = VoiceConfig(rvc_model_path="model.pth", rvc_index_path="model.index", rvc_only_cpu=False)
    rvc.convert(b"data", config)

    assert seen["only_cpu"] is False
