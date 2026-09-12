"""Text-to-speech via Piper, using its Python API directly (no CLI
shell-out) — verified against the same voice used in the spike."""

import io
import wave
from pathlib import Path

from piper import PiperVoice

from felt_voice.config import VoiceConfig
from felt_voice.text_normalize import normalize_for_speech

_voice = None
_loaded_name = None


def _get_voice(config: VoiceConfig) -> PiperVoice:
    global _voice, _loaded_name
    if _voice is None or _loaded_name != config.piper_voice:
        model_path = Path(config.piper_voices_dir) / f"{config.piper_voice}.onnx"
        _voice = PiperVoice.load(str(model_path))
        _loaded_name = config.piper_voice
    return _voice


def synthesize(text: str, config: VoiceConfig) -> bytes:
    voice = _get_voice(config)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(normalize_for_speech(text), wav_file)
    return buffer.getvalue()
