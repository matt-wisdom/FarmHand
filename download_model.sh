#!/usr/bin/env bash
# ==============================================================================
# FarmHand AI: Model Download Script
# ==============================================================================
# Downloads the Qwen 2.5 3B Instruct (Q4_K_M GGUF, ~1.92 GB) model file required
# for local on-device inference. Idempotent: skips download if already present.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/backend/models"
MODEL_FILE="$MODEL_DIR/qwen2.5-3b-instruct.Q4_K_M.gguf"
# Default official Hugging Face repository URL
MODEL_URL="https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf/resolve/main/qwen2.5-3b-instruct.Q4_K_M.gguf"

mkdir -p "$MODEL_DIR"

if [[ -f "$MODEL_FILE" ]]; then
  echo "Model already present at $MODEL_FILE, skipping download."
  exit 0
fi

echo "Downloading FarmHand model from $MODEL_URL -> $MODEL_FILE (~1.92 GB)..."
if command -v curl >/dev/null 2>&1; then
  curl -L --fail --progress-bar -o "$MODEL_FILE.partial" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
  wget --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL"
else
  echo "Error: neither curl nor wget available" >&2
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "Download complete: $MODEL_FILE"
