#!/usr/bin/env python3
"""
FarmHand AI - Model Download Utility
------------------------------------
Downloads quantized GGUF models directly from Hugging Face Hub using
the official `huggingface_hub` library or a pure Python urllib fallback.

Usage:
  python3 download_model.py
  python3 download_model.py --repo-id matt-wisdom/qwen_farm_agent_gguf --filename qwen2.5-3b-instruct.Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIR = ROOT_DIR / "backend" / "models"
DEFAULT_REPO = "matt-wisdom/qwen_farm_agent_gguf"
DEFAULT_3B = "qwen2.5-3b-instruct.Q4_K_M.gguf"
DEFAULT_1_5B = "qwen2.5-1.5b-instruct.Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FarmHand GGUF model from Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO,
        help=f"Hugging Face repository ID (default: '{DEFAULT_REPO}')",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=DEFAULT_3B,
        help=f"Target model file to download (default: '{DEFAULT_3B}')",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=TARGET_DIR,
        help="Target local directory to store model file (default: 'backend/models')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token (or HF_TOKEN env var)",
    )
    return parser.parse_args()


def download_via_urllib(url: str, dest: Path, token: str | None = None) -> bool:
    req = urllib.request.Request(url)  # noqa: S310
    auth_token = token or os.getenv("HF_TOKEN")
    if auth_token:
        req.add_header("Authorization", f"Bearer {auth_token}")

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
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def download_model():
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.local_dir / args.filename

    if target_path.exists() and target_path.stat().st_size > 100_000_000:
        print(
            f"Model file already exists at {target_path} ({target_path.stat().st_size / 1e9:.2f} GB). Skipping download."
        )
        return

    # Try huggingface_hub first if installed
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        print(
            f"Downloading '{args.filename}' from Hugging Face repository '{args.repo_id}' via huggingface_hub..."
        )
        try:
            downloaded_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                local_dir=args.local_dir,
                token=args.token or os.getenv("HF_TOKEN"),
            )
            print(f"Successfully downloaded model to: {downloaded_path}")
            return
        except EntryNotFoundError:
            if args.filename == DEFAULT_3B:
                print(
                    f"Notice: '{DEFAULT_3B}' not found on repo '{args.repo_id}'. Attempting fallback to '{DEFAULT_1_5B}'..."
                )
                fallback_target = args.local_dir / DEFAULT_1_5B
                if (
                    fallback_target.exists()
                    and fallback_target.stat().st_size > 100_000_000
                ):
                    print(f"Fallback model already exists at {fallback_target}.")
                    return
                downloaded_path = hf_hub_download(
                    repo_id=args.repo_id,
                    filename=DEFAULT_1_5B,
                    local_dir=args.local_dir,
                    token=args.token or os.getenv("HF_TOKEN"),
                )
                print(f"Successfully downloaded fallback model to: {downloaded_path}")
                return
            else:
                raise
    except ImportError:
        pass

    # Pure Python urllib fallback
    primary_url = f"https://huggingface.co/{args.repo_id}/resolve/main/{args.filename}"
    print(f"Downloading '{args.filename}' from {primary_url}...")
    success = download_via_urllib(primary_url, target_path, token=args.token)

    if not success and args.filename == DEFAULT_3B:
        fallback_url = (
            f"https://huggingface.co/{args.repo_id}/resolve/main/{DEFAULT_1_5B}"
        )
        fallback_path = args.local_dir / DEFAULT_1_5B
        print(
            f"Notice: '{args.filename}' returned 404. Attempting fallback download '{DEFAULT_1_5B}' from {fallback_url}..."
        )
        fallback_success = download_via_urllib(
            fallback_url, fallback_path, token=args.token
        )
        if fallback_success:
            print(f"Successfully downloaded fallback model to: {fallback_path}")
            return
        else:
            print(
                f"Error: Unable to download model from {args.repo_id}", file=sys.stderr
            )
            sys.exit(1)


if __name__ == "__main__":
    download_model()
