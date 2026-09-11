"""Speech-to-text via faster-whisper. CPU by default (2026-09-11 finding
in assistant-voice.md — the Nitro5's 6GB card is already tight with
Ollama loaded)."""

from faster_whisper import WhisperModel

from felt_voice.config import VoiceConfig

_model = None
_loaded_key = None


def _get_model(config: VoiceConfig) -> WhisperModel:
    global _model, _loaded_key
    key = (config.whisper_model, config.whisper_device, config.whisper_compute_type)
    if _model is None or _loaded_key != key:
        _model = WhisperModel(
            config.whisper_model, device=config.whisper_device, compute_type=config.whisper_compute_type
        )
        _loaded_key = key
    return _model


def transcribe(audio_path: str, config: VoiceConfig) -> str:
    model = _get_model(config)
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments)
