#!/usr/bin/env python3
"""
FarmHand AI - Hugging Face Model Upload Utility
------------------------------------------------
Uploads the quantized Qwen 2.5 3B Instruct GGUF model and associated model card
to Hugging Face Hub under the specified repository ID.

Usage:
  # Using CLI token or interactive prompt:
  python scripts/upload_to_hf.py --repo-id <username>/FarmHand-Qwen2.5-3B-GGUF

  # Using explicit token argument or environment variable HF_TOKEN:
  python scripts/upload_to_hf.py --repo-id <username>/FarmHand-Qwen2.5-3B-GGUF --token hf_xxx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo, upload_file
except ImportError:
    print(
        "Error: 'huggingface_hub' is required. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "backend" / "models" / "qwen2.5-3b-instruct.Q4_K_M.gguf"
DOWNLOAD_SCRIPT = ROOT_DIR / "download_model.sh"

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- gguf
- llama.cpp
- qwen2.5
- agriculture
- veterinary
- feed-formulation
- nigeria
- hausa
- pidgin
- adtc2026
- on-device-ai
pipeline_tag: text-generation
language:
- en
- ha
- pcm
---

# FarmHand AI: Qwen 2.5 3B Instruct (Q4_K_M GGUF)

Official quantized model release for **FarmHand AI**, an offline agricultural advisory and flock management system built for the **Africa Deep Tech Challenge 2026 (ADTC 2026)**.

## Model Highlights
- **Base Architecture**: Qwen 2.5 3B Instruct
- **Quantization**: 4-bit Medium (`Q4_K_M`), quantized via `llama.cpp`
- **File Size**: ~1.93 GB
- **Target Hardware**: Commodity 8 GB laptops with Intel Core i5/i7 (integrated graphics, zero discrete GPU)
- **Memory Footprint**: Operates within ~2.3 GB Peak RSS (far below the 7 GB ADTC RAM limit)
- **Languages**: English, Nigerian Pidgin (authentic conversational vernacular), Hausa (bilingual agricultural domain mapping)

## Key Integrations & Capabilities
1. **Clinical Veterinary Triage**: Outbreak symptom diagnosis, quarantine protocols, and sudden mortality investigation (e.g. PPR, African Swine Fever, Newcastle).
2. **Linear Programming Least-Cost Feed Formulation**: Formulates balanced livestock rations across 22 local Nigerian ingredients (maize, soy meal, PKC, bone meal, wheat offal, rice bran, etc.) with custom batch sizing.
3. **Flock & Financial Operations Ledger**: Deterministic multi-farm inventory and expenditure accounting with SQLite persistence.
4. **Hybrid Offline RAG**: Grounded on ~1,397 chunks of Nigerian agricultural extension bulletins from IITA, NAERLS, CGIAR, and FAO.

## Usage with `llama.cpp` / `llama-cpp-python`

```bash
# Run via llama.cpp
./llama-cli -m qwen2.5-3b-instruct.Q4_K_M.gguf -p "Farmer: 4 of my goats died sudden-sudden and foam dey commot their mouth. Wetin fit cause am?" -n 256 -t 2
```

## Challenge Context
- **Competition**: Africa Deep Tech Challenge 2026 (ADTC 2026)
- **Track**: Agriculture
- **Team**: FarmHand AI Team
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload FarmHand GGUF model to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="matt-wisdom/qwen_farm_agent_gguf",
        help="Target Hugging Face repository ID (default: 'matt-wisdom/qwen_farm_agent_gguf')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face User Access Token (with write permission). Defaults to $HF_TOKEN env var.",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the GGUF model file (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private (default is public)",
    )
    parser.add_argument(
        "--skip-download-script-update",
        action="store_true",
        help="Skip automatic updating of download_model.sh with the new Hugging Face URL",
    )
    return parser.parse_args()


def update_download_script(repo_id: str, filename: str) -> None:
    """Updates download_model.sh with the resolved Hugging Face URL."""
    if not DOWNLOAD_SCRIPT.exists():
        return

    hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    content = DOWNLOAD_SCRIPT.read_text(encoding="utf-8")

    # Replace MODEL_URL line
    lines = content.splitlines()
    updated_lines = []
    for line in lines:
        if line.startswith("MODEL_URL="):
            updated_lines.append(f'MODEL_URL="{hf_url}"')
        else:
            updated_lines.append(line)

    DOWNLOAD_SCRIPT.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"[upload_to_hf] Updated {DOWNLOAD_SCRIPT} with MODEL_URL={hf_url}")


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    model_file: Path = args.model_file
    if not model_file.exists():
        print(f"Error: Model file not found at '{model_file}'.", file=sys.stderr)
        sys.exit(1)

    file_size_gb = model_file.stat().st_size / (1024**3)
    print(f"Found model file: {model_file.name} ({file_size_gb:.2f} GB)")

    api = HfApi(token=token)

    # Verify authentication
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname")
        print(f"Authenticated with Hugging Face as: @{username}")
    except Exception as e:
        print(f"Warning: Authentication check: {e}")
        if not token:
            print(
                "Note: If upload fails, pass --token <your_write_token> or set HF_TOKEN environment variable."
            )

    # 1. Create Repository
    print(
        f"Creating / verifying model repository: '{args.repo_id}' (private={args.private})..."
    )
    try:
        repo_url = create_repo(
            repo_id=args.repo_id,
            token=token,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository ready at: {repo_url}")
    except Exception as e:
        print(f"Repository creation warning: {e}")

    # 2. Upload Model Card README.md
    print("Uploading Model Card (README.md)...")
    model_card_path = ROOT_DIR / "MODEL_CARD.md"
    card_content = (
        model_card_path.read_text(encoding="utf-8")
        if model_card_path.exists()
        else MODEL_CARD_TEMPLATE
    )
    temp_readme = ROOT_DIR / "temp_MODEL_CARD.md"
    try:
        temp_readme.write_text(card_content, encoding="utf-8")
        upload_file(
            path_or_fileobj=str(temp_readme),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=token,
            repo_type="model",
            commit_message="docs: add FarmHand AI model card and specifications",
        )
        print("Model Card README.md uploaded successfully.")
    except Exception as e:
        print(f"Error uploading README.md: {e}", file=sys.stderr)
    finally:
        if temp_readme.exists():
            temp_readme.unlink()

    # 3. Upload GGUF Model File
    print(
        f"Uploading '{model_file.name}' ({file_size_gb:.2f} GB) to '{args.repo_id}'..."
    )
    print("This may take a few minutes depending on your internet connection...")
    try:
        upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=model_file.name,
            repo_id=args.repo_id,
            token=token,
            repo_type="model",
            commit_message=f"feat: upload {model_file.name} for ADTC 2026",
        )
        print(f"Model file '{model_file.name}' uploaded successfully!")
    except Exception as e:
        print(f"Error uploading model file: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Update download_model.sh
    if not args.skip_download_script_update:
        update_download_script(args.repo_id, model_file.name)

    resolved_url = (
        f"https://huggingface.co/{args.repo_id}/resolve/main/{model_file.name}"
    )
    print("\n" + "=" * 70)
    print("✓ Upload Complete!")
    print(f"Hugging Face Model Page: https://huggingface.co/{args.repo_id}")
    print(f"Direct Resolve Download URL: {resolved_url}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
