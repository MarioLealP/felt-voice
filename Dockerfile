# felt-voice on the server (2026-10-21): STT (faster-whisper, CPU) + TTS
# (Chatterbox-Turbo "lite" on the P104) + the Teto RVC pass (on the 1660).
# Models are not baked in - the stack mounts them read-only (HF cache,
# rmvpe.pt, the RVC model, the Turbo reference clip) and runs offline.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app /app
ENV PATH=/app/.venv/bin:$PATH
EXPOSE 8421
CMD ["felt-voice-serve"]
