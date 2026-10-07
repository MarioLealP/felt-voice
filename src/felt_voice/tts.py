"""Text-to-speech: Chatterbox-Turbo, IndexTTS-2.5 (via the felt-indextts
service) or Piper, selected by FELT_VOICE_TTS_ENGINE. Piper is also the
automatic fallback whenever the configured engine fails to load or errors
on a request, so a broken engine degrades to the old voice path instead of
a silent Felt."""

import io
import logging
import ssl
import wave
from pathlib import Path

import httpx
from piper import PiperVoice

from felt_voice import mtls
from felt_voice.config import VoiceConfig
from felt_voice.tags import strip_tags
from felt_voice.text_normalize import normalize_for_speech

log = logging.getLogger(__name__)

_voice = None
_loaded_name = None

_chatterbox = None
_chatterbox_failed = False


def _get_voice(config: VoiceConfig) -> PiperVoice:
    global _voice, _loaded_name
    if _voice is None or _loaded_name != config.piper_voice:
        model_path = Path(config.piper_voices_dir) / f"{config.piper_voice}.onnx"
        _voice = PiperVoice.load(str(model_path))
        _loaded_name = config.piper_voice
    return _voice


def _load_chatterbox(config: VoiceConfig):
    from felt_voice.chatterbox_engine import ChatterboxEngine

    return ChatterboxEngine(config)


def init_engines(config: VoiceConfig) -> None:
    """Loads the configured engine once (at startup), plus Piper so the
    fallback is instant. A Chatterbox load failure is logged, not raised."""
    global _chatterbox, _chatterbox_failed
    _get_voice(config)
    if config.tts_engine == "chatterbox" and _chatterbox is None and not _chatterbox_failed:
        try:
            _chatterbox = _load_chatterbox(config)
        except Exception as exc:
            _chatterbox_failed = True
            log.warning("Chatterbox failed to load, falling back to Piper: %r", exc)


def active_engine(config: VoiceConfig) -> str:
    if config.tts_engine == "indextts":
        return "indextts"
    return "chatterbox" if config.tts_engine == "chatterbox" and _chatterbox is not None else "piper"


def prepare_text(text: str, config: VoiceConfig) -> str:
    return normalize_for_speech(strip_tags(text, allow=config.allow_tags))


def silence_wav(seconds: float = 0.2, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * int(seconds * rate))
    return buffer.getvalue()


def _piper(text: str, config: VoiceConfig) -> bytes:
    voice = _get_voice(config)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        # Piper would read a tag out loud, so it never gets one.
        voice.synthesize_wav(strip_tags(text), wav_file)
    return buffer.getvalue()


def _indextts(text: str, emotion: str | None, config: VoiceConfig, overrides: dict | None = None) -> bytes:
    verify: ssl.SSLContext | bool = True
    extensions = {}
    if config.indextts_url.startswith("https://"):  # WP-62 M2b B5.7 (D11)
        if not (config.tls_client_bundle and config.tls_ca):
            raise RuntimeError("an https indextts_url needs FELT_VOICE_TLS_CLIENT_BUNDLE + FELT_VOICE_TLS_CA")
        verify = mtls.client_context(config.tls_client_bundle, config.tls_ca)
        extensions = {"sni_hostname": config.indextts_tls_name}
    with httpx.Client(verify=verify, timeout=config.indextts_timeout_s) as client:
        resp = client.post(
            f"{config.indextts_url}/tts",
            json={"text": text, "emotion": emotion, **(overrides or {})},
            extensions=extensions,
        )
    resp.raise_for_status()
    return resp.content


def synthesize(
    text: str, config: VoiceConfig, emotion: str | None = None, overrides: dict | None = None
) -> tuple[bytes, str]:
    """Returns (wav_bytes, engine_used). Pre-RVC audio. `emotion` (one of
    felt-core's mood words) is used by the indextts engine only."""
    text = prepare_text(text, config)
    if config.tts_engine == "indextts":
        try:
            return _indextts(text, emotion, config, overrides), "indextts"
        except Exception as exc:
            log.warning("IndexTTS errored on a request, falling back to Piper: %r", exc)
    if config.tts_engine == "chatterbox":
        init_engines(config)  # no-op once loaded (or once it has failed)
        if _chatterbox is not None:
            try:
                return _chatterbox.synthesize(text), "chatterbox"
            except Exception as exc:
                log.warning("Chatterbox errored on a request, falling back to Piper: %r", exc)
    return _piper(text, config), "piper"
