#!/usr/bin/env bash
# ==============================================================================
# FarmHand AI: Model Download Script
# ==============================================================================
# Downloads GGUF model and Modelfile directly from:
# https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf
# Fails immediately if download cannot be completed. No fallbacks.
# ==============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${HERE}/backend/models"

mkdir -p "${TARGET_DIR}"

MODEL_URL="https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf/resolve/main/qwen2.5-1.5b-instruct.Q4_K_M.gguf?download=true"
MODEL_FILE="${TARGET_DIR}/qwen2.5-1.5b-instruct.Q4_K_M.gguf"

MODELFILE_URL="https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf/resolve/main/Modelfile?download=true"
MODELFILE_DEST="${TARGET_DIR}/Modelfile"

AUTH_HEADER=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

echo "================================================================================"
echo " FarmHand AI - Model Downloader"
echo " Target Directory: ${TARGET_DIR}"
echo "================================================================================"

# 1. Download Model GGUF
if [[ -f "${MODEL_FILE}" ]] && [[ $(stat -c%s "${MODEL_FILE}" 2>/dev/null || stat -f%z "${MODEL_FILE}" 2>/dev/null || echo 0) -gt 100000000 ]]; then
  echo "[✓] Model file already exists: ${MODEL_FILE}"
else
  echo "[+] Downloading model from: ${MODEL_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar ${AUTH_HEADER:+"${AUTH_HEADER[@]}"} -o "${MODEL_FILE}.partial" "${MODEL_URL}"
    mv "${MODEL_FILE}.partial" "${MODEL_FILE}"
  elif command -v wget >/dev/null 2>&1; then
    wget --progress=bar:force -O "${MODEL_FILE}.partial" "${MODEL_URL}"
    mv "${MODEL_FILE}.partial" "${MODEL_FILE}"
  else
    echo "[-] Error: Neither curl nor wget found in PATH." >&2
    exit 1
  fi
  echo "[✓] Model downloaded successfully to ${MODEL_FILE}"
fi

# 2. Download Modelfile
echo "[+] Downloading Modelfile from: ${MODELFILE_URL}"
if command -v curl >/dev/null 2>&1; then
  curl -fL ${AUTH_HEADER:+"${AUTH_HEADER[@]}"} -s -S -o "${MODELFILE_DEST}" "${MODELFILE_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q -O "${MODELFILE_DEST}" "${MODELFILE_URL}"
fi
echo "[✓] Modelfile saved to ${MODELFILE_DEST}"

echo "================================================================================"
echo " All model assets downloaded successfully."
echo "================================================================================"
