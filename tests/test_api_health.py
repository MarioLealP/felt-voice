"""A long synthesis must not starve the health check: the kubelet killed the pod mid-clip (liveness 5 s)."""

import threading
import time
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from felt_voice import api


@asynccontextmanager
async def _no_warmup(_app):
    yield


def test_health_answers_while_a_clip_is_being_made(monkeypatch):
    started = threading.Event()

    def slow_speak(*_a, **_k):
        started.set()
        time.sleep(2)
        return b"RIFF"

    monkeypatch.setattr(api, "_speak", slow_speak)
    monkeypatch.setattr(api.app.router, "lifespan_context", _no_warmup)
    with TestClient(api.app) as client:  # one event loop for both requests, like uvicorn
        out = {}
        t = threading.Thread(target=lambda: out.setdefault("tts", client.post("/v1/tts", json={"text": "hi"})))
        t.start()
        assert started.wait(5)
        t0 = time.monotonic()
        assert client.get("/openapi.json").status_code == 200
        assert time.monotonic() - t0 < 1.0
        t.join()
    assert out["tts"].content == b"RIFF"
