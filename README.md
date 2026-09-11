# felt-voice

Felt's voice — local speech-to-text (`faster-whisper`) and
text-to-speech (Piper). Part of the [Felt](https://git.clockmost.com/Benylin/Felt)
project; see that repo for the overall architecture.

Both engines are local — no audio ever leaves the LAN, matching Felt's
local-first principle. Config (never hardcoded) via `.env` — see
`.env.example`.

## Quickstart

```
uv sync
uv run python -m piper.download_voices --download-dir voices en_US-lessac-medium
cp .env.example .env
uv run felt-voice-serve
```

## API

- `POST /v1/tts` — `{"text": "..."}` → WAV audio bytes.
- `POST /v1/stt` — multipart file upload (audio) → `{"text": "..."}`.

## Status

Built (2026-09-11): both endpoints verified end-to-end via a real TTS →
STT round trip through the actual HTTP API, not just a script test. STT
runs on CPU by default — the reference deployment (Nitro5, 6 GB VRAM) is
already tight with Ollama loaded for `felt-core`; CPU int8 is fast enough
for short utterances. Not yet built: mic capture (that's a client's job —
the dashboard or overlay companion — not this service's), wake-word
detection, RVC voice conversion.
