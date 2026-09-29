#!/bin/bash
# Idempotent install of the felt-indextts service's runtime on the Nitro5:
# IndexTTS pinned clone + its uv venv + model weights + the Teto reference.
# Run as benylin. The systemd unit itself needs sudo (see README).
set -euo pipefail
PIN=d9e41aa            # index-tts commit benched 2026-09-29
D=~/felt-indextts; mkdir -p $D; cd $D
[ -d index-tts ] || git clone -q https://github.com/index-tts/index-tts.git
cd index-tts && git fetch -q && git checkout -q $PIN
uv sync -q
VIRTUAL_ENV=.venv uv pip install -q "pydantic-settings>=2.10"
[ -f checkpoints/config.yaml ] || uv tool run --from huggingface_hub hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints
# Teto reference clip (Apache-2.0, NeatAvocado14/KasaneTeto-GPTSoVITS-V2)
R=~/felt-voice/models/indextts/teto_ref_ja.wav
mkdir -p "$(dirname $R)"
[ -s $R ] || curl -sfL -o $R https://huggingface.co/NeatAvocado14/KasaneTeto-GPTSoVITS-V2/resolve/main/ref_audio.wav
echo "felt-indextts runtime ready in $D"
