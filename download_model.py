#!/usr/bin/env python3
"""
FarmHand AI - Model Download Utility
------------------------------------
Downloads quantized GGUF models directly from Hugging Face Hub.
Strict download with zero fallback behavior.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIR = ROOT_DIR / "backend" / "models"
MODEL_URL = "https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf/resolve/main/qwen2.5-1.5b-instruct.Q4_K_M.gguf?download=true"
MODELFILE_URL = "https://huggingface.co/matt-wisdom/qwen_farm_agent_gguf/resolve/main/Modelfile?download=true"


def download_file(url: str, dest: Path, token: str | None = None) -> None:
    req = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "FarmHand-Downloader/1.0"}
    )
    auth_token = token or os.getenv("HF_TOKEN")
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")

    print(f"[+] Downloading from {url} -> {dest}...")
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            total_bytes = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            temp_dest = dest.with_suffix(dest.suffix + ".partial")
            with open(temp_dest, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = (downloaded / total_bytes) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        sys.stdout.write(
                            f"\rProgress: {mb_done:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)"
                        )
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(
                            f"\rDownloaded: {downloaded / (1024 * 1024):.1f} MB"
                        )
                        sys.stdout.flush()

            sys.stdout.write("\n")
            temp_dest.replace(dest)
            print(f"[✓] Saved to {dest}")
    except Exception as e:
        print(f"[-] ERROR: Failed to download from {url}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    model_dest = TARGET_DIR / "qwen2.5-1.5b-instruct.Q4_K_M.gguf"
    modelfile_dest = TARGET_DIR / "Modelfile"

    token = os.getenv("HF_TOKEN")

    if model_dest.exists() and model_dest.stat().st_size > 100_000_000:
        print(
            f"[✓] Model file already exists at {model_dest} ({model_dest.stat().st_size / 1e9:.2f} GB)."
        )
    else:
        download_file(MODEL_URL, model_dest, token=token)

    download_file(MODELFILE_URL, modelfile_dest, token=token)
    print("\n[✓] All model assets ready.")


if __name__ == "__main__":
    main()
