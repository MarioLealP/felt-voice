"""Sopro V2 Turbo TTS on the CPU, cloning a Portuguese woman's voice (2026-10-21,
Mario's pick for English and European Portuguese - workspace
felt-voice-ptpt-spike.md). Its output goes through the same RVC Teto pass as
the other engines (api.py): the reference only brings the accent, Teto's
timbre comes from RVC.

The reference is encoded once at load. Each line gets `lang` from
guess_lang(): Sopro reads ambiguous text better when told the language, and
Felt speaks English or Portuguese.
"""

import io
from pathlib import Path

import soundfile as sf

from felt_voice.config import VoiceConfig
from felt_voice.lang import guess_lang


class SoproEngine:
    def __init__(self, config: VoiceConfig):
        import torch
        from sopro import SoproTTS

        ref = Path(config.sopro_ref_wav)
        if not ref.is_file():
            raise FileNotFoundError(f"Sopro reference clip not found: {ref}")
        torch.set_num_threads(config.sopro_threads)
        self.tts = SoproTTS.from_pretrained(config.sopro_model, device="cpu")
        self.ref = self.tts.prepare_reference(ref_audio_path=str(ref))

    def synthesize(self, text: str) -> bytes:
        wav = self.tts.synthesize(text, ref=self.ref, lang=guess_lang(text))
        wav = wav.detach().cpu().float()
        if wav.dim() == 2:
            wav = wav[0] if wav.shape[0] == 1 else wav.mean(dim=0)
        buffer = io.BytesIO()
        sf.write(
            buffer, wav.numpy(), self.tts.sample_rate, format="WAV", subtype="PCM_16"
        )
        return buffer.getvalue()
