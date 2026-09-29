# felt-voice

Felt's voice — local speech-to-text (`faster-whisper`) and
text-to-speech (IndexTTS-2.5 with per-line moods, Chatterbox-Turbo or Piper,
with Piper as the automatic fallback), all re-voiced through the same Kasane
Teto RVC model. Part of the [Felt](https://git.clockmost.com/Benylin/Felt)
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

- `POST /v1/tts` — `{"text": "...", "emotion": "sleepy"}` → WAV audio bytes.
  `emotion` is optional (felt-core's mood words: happy, excited, amused, sad,
  annoyed, surprised, thinking, sleepy, neutral); only `indextts` uses it.
- `POST /v1/stt` — multipart file upload (audio) → `{"text": "..."}`.

## TTS engines

### indextts (live since 2026-09-29)

Mario's pick from the 2026-09-29 voice A/B: **IndexTTS-2.5 reading with
Teto's own (Japanese) reference clip, then the Teto RVC model at pitch -4**
(`FELT_VOICE_RVC_PITCH=-4`). IndexTTS supplies natural delivery and the
mood; RVC gives her the familiar timbre.

IndexTTS runs as a separate local service, **felt-indextts**
(`indextts_service/server.py`, `127.0.0.1:8431`), because its torch and
transformers pins clash with this venv. It returns base audio; RVC stays in
this process. Setup on the Nitro5:

```
deploy/install-indextts.sh                       # clone + venv + weights + reference
sudo cp deploy/felt-indextts.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now felt-indextts
```

Measured on the RTX 3060 6 GB (2026-09-29): ~3.8 GB VRAM for IndexTTS + GPU
RVC together (its 2.2 GB reference encoder runs on the CPU, once per
reference), and a line takes about its own duration to synthesise (a 3 s
first sentence ~3.7 s, a 10 s line ~10.5 s). The service warms up at start
(the first call otherwise takes 13-23 s).

### chatterbox / piper

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
