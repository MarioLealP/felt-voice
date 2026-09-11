"""Config — same no-hardcoding discipline as felt-core."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FELT_VOICE_", env_file=".env", extra="ignore")

    # STT — CPU by default (2026-09-11 finding: the Nitro5's 6GB card is
    # already tight with Ollama loaded; CPU int8 is fast enough for short
    # utterances). Override to "cuda" on a box with GPU headroom to spare.
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # TTS
    piper_voice: str = "en_US-lessac-medium"
    piper_voices_dir: str = "./voices"

    host: str = "0.0.0.0"
    port: int = 8421


def load_config() -> VoiceConfig:
    return VoiceConfig()
