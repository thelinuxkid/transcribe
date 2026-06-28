#!/bin/bash
# setup.sh — whispermlx transcription environment (conda)
# M3 MacBook Air · Spanish meeting audio · multi-speaker
# Run once: bash setup.sh

set -e

ENV_NAME="transcribe"

# ── Check conda ──────────────────────────────────────────────────────────────
if ! command -v conda &> /dev/null; then
  echo ""
  echo "ERROR: conda not found."
  echo ""
  echo "Install Miniforge (recommended for Apple Silicon):"
  echo "  brew install miniforge"
  echo "  conda init zsh   # or bash"
  echo "  # then open a new terminal and re-run this script"
  echo ""
  echo "Or download directly:"
  echo "  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
  exit 1
fi

# ── Check ffmpeg ─────────────────────────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
  echo "WARNING: ffmpeg not found. Installing via conda-forge..."
  # will be installed into the env below
  INSTALL_FFMPEG=true
else
  echo "ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
  INSTALL_FFMPEG=false
fi

# ── Create or update conda env ───────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
  echo ""
  echo "==> Conda env '${ENV_NAME}' already exists. Updating..."
  conda activate "${ENV_NAME}"
else
  echo ""
  echo "==> Creating conda env '${ENV_NAME}' with Python 3.11..."
  # Python 3.11: safest for pyannote-audio 4.x deps
  # torch 2.8 (pinned by whispermlx) is available on conda-forge for arm64
  conda create -n "${ENV_NAME}" python=3.11 -c conda-forge -y
fi

# Activate — works in both bash and zsh
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "==> Activated: $(which python) ($(python --version))"

# ── Install ffmpeg into env if missing system-wide ───────────────────────────
if [ "$INSTALL_FFMPEG" = true ]; then
  echo ""
  echo "==> Installing ffmpeg..."
  conda install -n "${ENV_NAME}" ffmpeg -c conda-forge -y --quiet
fi

# ── Install pip deps ─────────────────────────────────────────────────────────
# mlx-whisper is Apple Silicon only — not on conda-forge, must come from pip
# torch 2.8 is pulled by whispermlx; conda-forge torch would conflict, so pip handles it
echo ""
echo "==> Upgrading pip..."
pip install --upgrade pip --quiet

echo ""
echo "==> Installing whispermlx + all deps (mlx-whisper, pyannote, torch 2.8)..."
echo "    This may take a few minutes on first run..."
pip install whispermlx

echo ""
echo "==> Installing huggingface_hub CLI..."
pip install huggingface_hub soundfile --quiet

echo ""
echo "==> Installing enric-transcribe package (editable)..."
pip install -e "$(dirname "$0")" --quiet

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Verifying installation..."
python -c "import whispermlx; print('  whispermlx OK')"
python -c "import mlx_whisper; print('  mlx_whisper OK')"
python -c "import pyannote.audio; print('  pyannote.audio OK')"
python -c "import torch; print(f'  torch {torch.__version__} OK')"

echo ""
echo "============================================"
echo " Setup complete."
echo "============================================"
echo ""
echo "NEXT STEP — HuggingFace token (required for diarization):"
echo ""
echo "  1. Create a free account at https://huggingface.co"
echo "  2. Go to https://huggingface.co/settings/tokens"
echo "     → New token → name it anything → Access: Read"
echo "  3. Accept pyannote model terms at (one-time, in browser):"
echo "     → https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "     → https://huggingface.co/pyannote/segmentation-3.0"
echo "  4. Save your token:"
echo "     conda activate ${ENV_NAME}"
echo "     huggingface-cli login"
echo ""
echo "Then transcribe:"
echo "  conda activate ${ENV_NAME}"
echo "  transcribe meeting.m4a"
echo "  transcribe meeting.m4a --speakers 4 --diarize"
echo "  diarize meeting.m4a transcripts/meeting.json --speakers 4"
echo ""
