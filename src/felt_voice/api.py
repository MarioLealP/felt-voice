"""felt-voice's frontend API — what felt-core or a dashboard/overlay
client actually calls for speech in/out."""

import tempfile

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from felt_voice.config import load_config
from felt_voice.gpu_stats import get_gpu_stats
from felt_voice.stt import transcribe
from felt_voice.tts import synthesize

app = FastAPI(title="felt-voice", version="0.1.0")


class TTSRequest(BaseModel):
    text: str


class STTResponse(BaseModel):
    text: str


@app.post("/v1/tts")
async def tts_endpoint(req: TTSRequest) -> Response:
    config = load_config()
    audio = synthesize(req.text, config)
    return Response(content=audio, media_type="audio/wav")


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
    config = load_config()
    uvicorn.run(app, host=config.host, port=config.port)
