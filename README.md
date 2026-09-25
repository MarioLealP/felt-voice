# felt-voice

Felt's voice — local speech-to-text (`faster-whisper`) and
text-to-speech (Chatterbox-Turbo by default, Piper as the fallback), both
re-voiced through the same Kasane Teto RVC model. Part of the [Felt](https://git.clockmost.com/Benylin/Felt)
project; see that repo for the overall architecture.

Both engines are local — no audio ever leaves the LAN, matching Felt's
local-first principle. Config (never hardcoded) via `.env` — see
`.env.example`.

## Quickstart

```
uv sync
uv run python -m piper.download_voices --download-dir voices en_US-lessac-medium
# Chatterbox reference clip (not committed, models/ is ignored):
#   models/chatterbox/teto_ref.wav
cp .env.example .env
uv run felt-voice-serve
```

## API

- `POST /v1/tts` — `{"text": "..."}` → WAV audio bytes.
- `POST /v1/stt` — multipart file upload (audio) → `{"text": "..."}`.

## TTS engines

`FELT_VOICE_TTS_ENGINE=chatterbox|piper`. Chatterbox-Turbo is loaded once
at startup in low-VRAM mode (T3 in bf16, conditioning encoders on CPU,
watermark off); Piper is loaded too, and takes over automatically, with a
warning in the log, if Chatterbox fails to load or errors on a request.
Every bracketed tag (`[laugh]`, `[sigh]`, ...) is stripped before
synthesis unless `FELT_VOICE_ALLOW_TAGS=true` (see `tags.py`). Each
request logs `tts engine=... synth_s=... rvc_s=... total_s=...`.

## Status

Built (2026-09-11): both endpoints verified end-to-end via a real TTS →
STT round trip through the actual HTTP API, not just a script test. STT
runs on CPU by default — the reference deployment (Nitro5, 6 GB VRAM) is
already tight with Ollama loaded for `felt-core`; CPU int8 is fast enough
for short utterances. Not yet built: mic capture (that's a client's job —
the dashboard or overlay companion — not this service's), wake-word
detection, RVC voice conversion.
