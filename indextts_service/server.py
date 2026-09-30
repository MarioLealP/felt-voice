"""felt-indextts: IndexTTS-2.5 as a small local HTTP service, the base voice
for felt-voice's `indextts` engine.

Mario picked "IndexTTS reading with Teto's own reference clip, then the Teto
RVC model at pitch -4" in the 2026-09-29 voice A/B ("-4 is the way, idx -4").
This service does the first half: text plus a mood -> base WAV. felt-voice
applies the RVC pass itself (api.py), exactly as it does for Piper.

It runs in IndexTTS's own venv (its torch/transformers pins clash with
felt-voice's), so it lives here as a single file with its own entry point:

    cd ~/felt-indextts/index-tts && .venv/bin/python ~/felt-voice/indextts_service/server.py

Fitting the laptop's 6 GB RTX 3060 next to felt-voice's GPU RVC:
- `semantic_model` (w2v-BERT, 2.2 GB fp32) only encodes the reference clip,
  and IndexTTS caches the result per reference. It runs on the CPU (once,
  ~1-2 s), which takes the service from 5.5 GB to 2.8 GB of VRAM.
- The GPT runs in bf16. The optional accel engine (flash_attn) gained ~5 %
  and cost 1.5 GB, and the fused BigVGAN kernel doesn't build here, so both
  are off.

Moods: felt-core's emotion words map onto IndexTTS's 8-float emotion
vector [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
An unknown or missing emotion speaks neutrally (the reference's own tone).
"""

import io
import logging
import threading
import time
from contextlib import asynccontextmanager

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger("felt_indextts")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FELT_INDEXTTS_", extra="ignore")

    model_dir: str = "checkpoints"
    ref_wav: str = "/home/benylin/felt-voice/models/indextts/teto_ref_ja.wav"
    lang: str = "EN"
    # Scales every emotion vector (after normalizing, below). 0.6 = Mario's
    # pick by ear, 2026-09-30 ("60% for normal mood"), from now / 100% / 60%
    # / 35% takes of all eight moods: enough mood to hear, still mostly
    # Teto's own delivery instead of IndexTTS's stock emotion presets.
    emotion_strength: float = 0.6
    # Pass the vector through IndexTTS's own normalize_emo_vec first (what
    # its web UI does): de-emphasises surprised/calm, caps the sum at 0.8.
    # The sum is how much of the delivery comes from IndexTTS's stock
    # emotion presets instead of the reference voice - raw table vectors
    # summing to ~0.9 made every mood line sound like an exclaiming stock
    # actor (Mario, 2026-09-30).
    emotion_normalize: bool = True
    host: str = "127.0.0.1"
    port: int = 8431


# [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
EMOTIONS: dict[str, list[float]] = {
    "happy": [0.6, 0, 0, 0, 0, 0, 0, 0.2],
    "excited": [0.7, 0, 0, 0, 0, 0, 0.3, 0],
    "amused": [0.5, 0, 0, 0, 0, 0, 0.2, 0.2],
    "sad": [0, 0, 0.6, 0, 0, 0.2, 0, 0],
    "annoyed": [0, 0.5, 0, 0, 0.3, 0, 0, 0],
    "surprised": [0.2, 0, 0, 0, 0, 0, 0.6, 0],
    "thinking": [0, 0, 0, 0, 0, 0, 0, 0.6],
    "sleepy": [0, 0, 0, 0, 0, 0.4, 0, 0.5],
}


def emotion_vector(emotion: str | None, strength: float) -> list[float] | None:
    vec = EMOTIONS.get((emotion or "").strip().lower())
    return None if vec is None else [round(v * strength, 3) for v in vec]


class _CpuSemanticModel:
    """Wraps w2v-BERT so IndexTTS's `.to(device)` is a no-op and every call
    runs on the CPU, handing its hidden states back on the GPU."""

    def __init__(self, model):
        self.model = model

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, input_features=None, attention_mask=None, **kwargs):
        out = self.model(
            input_features=input_features.cpu().float(), attention_mask=attention_mask.cpu(), **kwargs
        )
        out.hidden_states = tuple(h.to("cuda") for h in out.hidden_states)
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


def load_tts(settings: Settings):
    import indextts.infer_v2_5 as iv

    original = iv.Wav2Vec2BertModel.from_pretrained
    iv.Wav2Vec2BertModel.from_pretrained = classmethod(
        lambda cls, *a, **k: _CpuSemanticModel(original(*a, **k))
    )
    return iv.IndexTTS2(
        cfg_path=f"{settings.model_dir}/config.yaml",
        model_dir=settings.model_dir,
        use_bf16=True,
        use_cuda_kernel=False,
    )


settings = Settings()
_tts = None
_lock = threading.Lock()  # one GPU, one request at a time


def synthesize(text: str, emotion: str | None, strength: float | None = None, normalize: bool | None = None) -> bytes:
    vec = emotion_vector(emotion, 1.0)
    if vec is not None:
        if settings.emotion_normalize if normalize is None else normalize:
            vec = _tts.normalize_emo_vec(vec, apply_bias=True)
        s = settings.emotion_strength if strength is None else strength
        vec = [round(v * s, 3) for v in vec]
    kwargs = {"emo_vector": vec, "use_random": False} if vec else {}
    with _lock:
        t0 = time.perf_counter()
        # output_path=None returns (sampling_rate, int16 samples) in memory.
        result = _tts.infer(spk_audio_prompt=settings.ref_wav, text=text, lang=settings.lang, output_path=None, **kwargs)
        sr, samples = result
        buf = io.BytesIO()
        sf.write(buf, samples.reshape(-1) if hasattr(samples, "reshape") else samples, sr, format="WAV", subtype="PCM_16")
        log.info(
            "indextts chars=%d emotion=%s vec=%s synth_s=%.3f audio_s=%.2f",
            len(text), emotion or "neutral", vec, time.perf_counter() - t0, len(samples.reshape(-1)) / sr,
        )
    return buf.getvalue()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _tts
    t0 = time.perf_counter()
    _tts = load_tts(settings)
    log.info("IndexTTS loaded in %.1fs", time.perf_counter() - t0)
    # The first calls compile kernels and encode the reference: ~13-23 s.
    # Pay that here, not on her first real line.
    for warm_text, warm_emotion in (("Warm up.", None), ("Warming up a little more.", "excited")):
        try:
            synthesize(warm_text, warm_emotion)
        except Exception as exc:
            log.warning("warm-up failed: %r", exc)
    log.info("ready, VRAM %.0f MiB", torch.cuda.memory_allocated() / 2**20)
    yield


app = FastAPI(title="felt-indextts", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str
    emotion: str | None = None
    # Per-request overrides of the settings above (A/B listening tests).
    emotion_strength: float | None = None
    emotion_normalize: bool | None = None


@app.post("/tts")
def tts_endpoint(req: TTSRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    return Response(content=synthesize(req.text, req.emotion, req.emotion_strength, req.emotion_normalize), media_type="audio/wav")


@app.get("/health")
def health() -> dict:
    return {"ok": _tts is not None}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(app, host=settings.host, port=settings.port)
