"""felt-voice's frontend API — what felt-core or a dashboard/overlay
client actually calls for speech in/out."""

import os

# Must run before anything imports huggingface_hub: it reads HF_HUB_OFFLINE
# once, at import time. config.py itself only imports pydantic.
from felt_voice.config import load_config as _load_config_early

if _load_config_early().hf_offline:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
import tempfile
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from felt_voice import tts
from felt_voice.config import VoiceConfig, load_config
from felt_voice.gpu_stats import get_gpu_stats
from felt_voice.rvc import convert as rvc_convert
from felt_voice.stt import transcribe

log = logging.getLogger("felt_voice.api")


def _speak(text: str, config: VoiceConfig) -> bytes:
    """TTS then the RVC Teto pass, logging per-stage timing."""
    if not tts.prepare_text(text, config):
        # Nothing speakable (e.g. a tag-only request): short silence, no RVC.
        return tts.silence_wav()
    t0 = time.perf_counter()
    audio, engine = tts.synthesize(text, config)
    t1 = time.perf_counter()
    if config.rvc_model_path:
        audio = rvc_convert(audio, config)
    t2 = time.perf_counter()
    log.info(
        "tts engine=%s chars=%d synth_s=%.3f rvc_s=%.3f total_s=%.3f",
        engine, len(text), t1 - t0, t2 - t1, t2 - t0,
    )
    return audio


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config = load_config()
    t0 = time.perf_counter()
    tts.init_engines(config)
    log.info("tts engines loaded in %.1fs, active=%s", time.perf_counter() - t0, tts.active_engine(config))
    try:
        _speak("Warm up.", config)  # loads RVC and CUDA kernels before the first real request
    except Exception as exc:
        log.warning("TTS warm-up failed: %r", exc)
    yield


app = FastAPI(title="felt-voice", version="0.1.0", lifespan=lifespan)


class TTSRequest(BaseModel):
    text: str


class STTResponse(BaseModel):
    text: str


@app.post("/v1/tts")
async def tts_endpoint(req: TTSRequest) -> Response:
    # Deliberately synchronous inside the async handler: requests are
    # serialized, so the GPU models never run concurrently.
    return Response(content=_speak(req.text, load_config()), media_type="audio/wav")


@app.post("/v1/stt", response_model=STTResponse)
async def stt_endpoint(file: UploadFile = File(...)) -> STTResponse:
    config = load_config()
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        text = transcribe(tmp.name, config)
    return STTResponse(text=text)


@app.get("/v1/gpu-status")
async def gpu_status_endpoint() -> dict:
    stats = get_gpu_stats()
    if stats is None:
        return {"available": False}
    return {"available": True, **stats}


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port)
