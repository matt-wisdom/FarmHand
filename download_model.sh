#!/usr/bin/env bash
# ==============================================================================
# FarmHand AI: Model Download Script
# ==============================================================================
# Invokes Python huggingface_hub downloader to fetch quantized GGUF weights
# into backend/models/. Idempotent: skips download if already present.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="python3"

if command -v python >/dev/null 2>&1; then
  PYTHON_EXEC="python"
fi

exec "$PYTHON_EXEC" "$HERE/download_model.py" "$@"
