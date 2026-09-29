"""Optional Kasane Teto voice-conversion pass over Piper's TTS output
(Mario, 2026-09-12: "get the teto voice when possible also"). Teto is a
UTAU singing voicebank, not a speech model — assistant-voice.md already
settled on RVC (retrieval-based voice conversion) as the realistic path:
Piper generates the actual speech, a pretrained RVC model re-voices it.

`infer_rvc_python` chosen after a real inference spike (2026-09-12) on
the Nitro5: takes a WAV file path in, converts in place, auto-downloads
its own HuBERT/RMVPE feature-extraction models on first use. CPU-only
inference of a ~3.5s clip measured ~3s steady-state once the model's
warm — acceptable added latency for a synchronous TTS request, and no
need to compete with Ollama for GPU the way the wake-word *training*
job would.
"""

import tempfile
from pathlib import Path

from infer_rvc_python import BaseLoader

from felt_voice.config import VoiceConfig

_converter = None
_loaded_key = None


def _get_converter(config: VoiceConfig) -> BaseLoader:
    global _converter, _loaded_key
    # Keyed on model+index path so switching RVC models (or turning it
    # off/on across config changes) reloads rather than silently reusing
    # stale weights — same cache-invalidation pattern as tts.py's
    # _get_voice and stt.py's _get_model.
    key = (config.rvc_model_path, config.rvc_index_path, config.rvc_pitch)
    if _converter is None or _loaded_key != key:
        _converter = BaseLoader(only_cpu=config.rvc_only_cpu, hubert_path=None, rmvpe_path=None)
        _converter.apply_conf(
            tag="rvc",
            file_model=config.rvc_model_path,
            pitch_algo="rmvpe+",
            pitch_lvl=config.rvc_pitch,
            file_index=config.rvc_index_path or "",
            index_influence=config.rvc_index_influence,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )
        _loaded_key = key
    return _converter


def convert(wav_bytes: bytes, config: VoiceConfig) -> bytes:
    """Runs `wav_bytes` (Piper's raw output) through the configured RVC
    model. Only meant to be called when `config.rvc_model_path` is set —
    see api.py's tts_endpoint for the gating."""
    converter = _get_converter(config)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        # overwrite=True: confirmed via a real spike (2026-09-12, file
        # size/duration/sample-rate changed) that this writes the
        # converted audio back over the same path rather than a
        # differently-named output file.
        converter(tmp.name, "rvc", overwrite=True)
        return Path(tmp.name).read_bytes()
