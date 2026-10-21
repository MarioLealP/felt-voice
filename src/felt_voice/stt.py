"""Speech-to-text: faster-whisper (CPU by default, 2026-09-11 finding in
assistant-voice.md - the Nitro5's 6GB card is already tight with Ollama
loaded) or Parakeet TDT 0.6B v3 through onnx-asr, for English + Portuguese
(stt_engine, config.py)."""

import subprocess

import numpy as np
from faster_whisper import WhisperModel

from felt_voice.config import VoiceConfig

_model = None
_loaded_key = None
_parakeet = None
_parakeet_key = None


def _get_model(config: VoiceConfig) -> WhisperModel:
    global _model, _loaded_key
    key = (config.whisper_model, config.whisper_device, config.whisper_compute_type)
    if _model is None or _loaded_key != key:
        _model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        _loaded_key = key
    return _model


def _get_parakeet(config: VoiceConfig):
    global _parakeet, _parakeet_key
    key = (config.parakeet_model, config.parakeet_quantization, config.stt_threads)
    if _parakeet is None or _parakeet_key != key:
        import onnx_asr
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.stt_threads
        _parakeet = onnx_asr.load_model(
            config.parakeet_model,
            quantization=config.parakeet_quantization,
            sess_options=options,
        )
        _parakeet_key = key
    return _parakeet


def warm_up(config: VoiceConfig) -> None:
    """Loads the configured model at startup, so the first question isn't slow."""
    if config.stt_engine == "parakeet":
        _get_parakeet(config)
    else:
        _get_model(config)


def _decode_16k(audio_path: str) -> np.ndarray:
    """Any audio ffmpeg reads (wav, flac, ogg, webm...) as 16 kHz mono float32 -
    Parakeet's own loader only takes WAV, and Whisper took anything."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            audio_path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe(audio_path: str, config: VoiceConfig) -> str:
    if config.stt_engine == "parakeet":
        return (
            _get_parakeet(config)
            .recognize(_decode_16k(audio_path), sample_rate=16000)
            .strip()
        )
    model = _get_model(config)
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments)
