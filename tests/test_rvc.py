import io

import numpy as np
import pytest
import soundfile as sf

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


def test_rvc_pitch_is_applied_and_changing_it_reloads(monkeypatch):
    created = []

    def fake_ctor(only_cpu, hubert_path, rmvpe_path):
        instance = _FakeConverter(only_cpu, hubert_path, rmvpe_path)
        created.append(instance)
        return instance

    monkeypatch.setattr(rvc, "BaseLoader", fake_ctor)
    _reset_cache()

    rvc._get_converter(VoiceConfig(rvc_model_path="m.pth", rvc_pitch=-4))
    rvc._get_converter(VoiceConfig(rvc_model_path="m.pth", rvc_pitch=0))

    assert len(created) == 2
    assert created[0].confs[0]["pitch_lvl"] == -4
    assert created[1].confs[0]["pitch_lvl"] == 0


# -- long audio in pieces (2026-10-21: one long sentence OOMed RVC on the 1660)


def _tone(seconds, rate=16000, gaps=()):
    """A loud tone with silent 100 ms gaps at the given seconds."""
    t = np.arange(int(seconds * rate)) / rate
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    for g in gaps:
        samples[int(g * rate) : int((g + 0.1) * rate)] = 0
    return samples


def test_split_points_cut_in_the_pause_before_each_limit():
    rate = 16000
    cuts = rvc.split_points(_tone(20, rate, gaps=(7.0, 14.5)), rate, 8)
    assert len(cuts) == 2
    assert 7.0 * rate <= cuts[0] < 7.1 * rate
    assert 14.5 * rate <= cuts[1] < 14.6 * rate


def test_short_audio_is_never_split():
    assert rvc.split_points(_tone(5), 16000, 8) == []


class _ResampleConverter(_FakeConverter):
    """Converts like RVC does: reads the WAV, writes it back at 40 kHz."""

    def __call__(self, path, tag, overwrite=False):
        self.calls.append((path, tag, overwrite))
        samples, rate = sf.read(path, dtype="float32")
        out = np.repeat(samples, 40000 // rate)
        sf.write(path, out, 40000, format="WAV", subtype="PCM_16")
        return [path]


def test_convert_in_pieces_joins_them_back(monkeypatch):
    fake = _ResampleConverter(False, None, None)
    monkeypatch.setattr(rvc, "BaseLoader", lambda only_cpu, hubert_path, rmvpe_path: fake)
    _reset_cache()
    buffer = io.BytesIO()
    sf.write(buffer, _tone(20, 8000, gaps=(7.0, 14.5)), 8000, format="WAV", subtype="PCM_16")

    out = rvc.convert(buffer.getvalue(), VoiceConfig(rvc_model_path="m.pth", rvc_max_seconds=8))

    assert len(fake.calls) == 3
    samples, rate = sf.read(io.BytesIO(out))
    assert rate == 40000
    assert abs(len(samples) / rate - 20) < 0.01


class _DyingConverter(_FakeConverter):
    """The real library when its worker thread crashes: nothing converted, []."""

    def __call__(self, path, tag, overwrite=False):
        self.calls.append((path, tag, overwrite))
        return []


def test_a_failed_conversion_raises_instead_of_returning_the_base_voice(monkeypatch):
    monkeypatch.setattr(rvc, "BaseLoader", lambda only_cpu, hubert_path, rmvpe_path: _DyingConverter(False, None, None))
    _reset_cache()
    with pytest.raises(RuntimeError, match="RVC conversion failed"):
        rvc.convert(b"RIFF-fake", VoiceConfig(rvc_model_path="m.pth"))

