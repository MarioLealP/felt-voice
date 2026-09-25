"""Config — same no-hardcoding discipline as felt-core."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FELT_VOICE_", env_file=".env", extra="ignore")

    # STT — CPU by default (2026-09-11 finding: the Nitro5's 6GB card is
    # already tight with Ollama loaded; CPU int8 is fast enough for short
    # utterances). Override to "cuda" on a box with GPU headroom to spare.
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # TTS engine. "chatterbox" = Chatterbox-Turbo zero-shot cloning a Teto
    # reference clip (Mario approved it over Piper after an A/B listen,
    # 2026-09-25: "much better"); "piper" = the original engine. Either way
    # the output goes through the same RVC Teto pass (one voice). Piper stays
    # loaded as the automatic fallback if Chatterbox fails to load or errors.
    tts_engine: Literal["chatterbox", "piper"] = "chatterbox"
    chatterbox_ref_wav: str = "./models/chatterbox/teto_ref.wav"
    chatterbox_device: str = "cuda"
    # Fit next to the chat model on the Nitro5's 6 GB card: T3 in bf16, the
    # conditioning-only encoders parked on CPU after the reference is baked.
    chatterbox_low_vram: bool = True

    # Keep curatable Chatterbox tags ([sigh], [gasp], ...) instead of
    # stripping every one. Off for live speech; for WP-19's offline batch.
    # [laugh]/[chuckle] are stripped regardless. See tags.py.
    allow_tags: bool = False

    # Piper (fallback engine, or primary with tts_engine=piper)
    piper_voice: str = "en_US-lessac-medium"
    piper_voices_dir: str = "./voices"

    # Optional Kasane Teto voice-conversion pass over Piper's raw output
    # (Mario, 2026-09-12: "get the teto voice when possible also" —
    # assistant-voice.md already settled on RVC, retrieval-based voice
    # conversion, as the realistic path since Teto is a singing voicebank,
    # not a speech model). Unset = disabled, plain Piper audio returned
    # unchanged — the documented fallback if RVC quality doesn't hold up.
    rvc_model_path: str | None = None
    rvc_index_path: str | None = None
    rvc_index_influence: float = 0.66
    # CPU by default, same reasoning as Whisper above: a real inference
    # spike (2026-09-12) measured ~3s steady-state CPU latency for a
    # ~3.5s clip — acceptable added latency for a synchronous TTS
    # request, so there's no need to fight Ollama for the Nitro5's tight
    # 6GB card the way the wake-word *training* job would have.
    rvc_only_cpu: bool = True

    host: str = "0.0.0.0"
    port: int = 8421


def load_config() -> VoiceConfig:
    return VoiceConfig()
