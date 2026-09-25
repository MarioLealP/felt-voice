"""Chatterbox-Turbo TTS, zero-shot cloning a Kasane Teto reference clip.

Ported from the felt-voice-next spike (2026-09-25), which Mario approved
over Piper after an A/B listen ("much better"). Its output still goes
through the same RVC Teto pass as Piper's did (api.py): one voice.

Low-VRAM mode fits it next to Ollama's chat model on the Nitro5's 6 GB card
(measured peak 5472/6144 MiB with qwen2.5:3b resident): T3 (the 427M GPT-2
backbone) in bf16, and the S3 tokenizer, S3Gen speaker encoder and voice
encoder, which only `prepare_conditionals` uses, parked on CPU once the
reference conditioning is baked. Perth watermarking is switched off: it's
irrelevant for a local assistant and costs CPU time per request.
"""

import io
from pathlib import Path

import soundfile as sf
import torch

from felt_voice.config import VoiceConfig


class ChatterboxEngine:
    def __init__(self, config: VoiceConfig):
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        ref = Path(config.chatterbox_ref_wav)
        if not ref.is_file():
            raise FileNotFoundError(f"Chatterbox reference clip not found: {ref}")
        device = config.chatterbox_device

        self.tts = ChatterboxTurboTTS.from_pretrained(device=device)
        self.tts.watermarker.apply_watermark = lambda wav, sample_rate: wav
        self.tts.prepare_conditionals(str(ref))

        if config.chatterbox_low_vram and device == "cuda":
            s3 = self.tts.s3gen
            # S3Gen.device is derived from the tokenizer's params; pin it to
            # cuda before the tokenizer moves to CPU.
            s3.__class__ = type(
                "S3GenPinned", (type(s3),), {"device": property(lambda _s: torch.device("cuda"))}
            )
            s3.tokenizer.to("cpu")
            s3.speaker_encoder.to("cpu")
            self.tts.ve.to("cpu")
            self.tts.t3.to(dtype=torch.bfloat16)
            self.tts.conds.t3 = self.tts.conds.t3.to(dtype=torch.bfloat16)
            torch.cuda.empty_cache()

    def synthesize(self, text: str) -> bytes:
        with torch.inference_mode():
            wav = self.tts.generate(text)
        buffer = io.BytesIO()
        sf.write(buffer, wav.squeeze(0).float().cpu().numpy(), self.tts.sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()
