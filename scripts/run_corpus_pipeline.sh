#!/usr/bin/env bash
# Run the full Nigeria Agro Knowledge Corpus pipeline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> Seed downloads"
python scripts/download_sources.py

echo "==> NAERLS harvest (animals first)"
python scripts/harvest_naerls.py --max-docs 80

echo "==> IITA harvest"
python scripts/harvest_iita.py --max-docs 40

echo "==> CGSpace harvest (expanded animal queries)"
python scripts/harvest_cgspace.py --max-docs 150 --max-pages 3 --sleep 1.0

echo "==> Hugging Face Q&A"
python scripts/harvest_hf_qa.py

echo "==> Extract text"
python scripts/extract_text.py

echo "==> Chunk + Q&A normalize"
python scripts/chunk_and_tag.py

echo "==> Done"
echo "Manifest: data/manifests/sources.csv"
echo "Chunks:   data/processed/chunks.jsonl"
echo "Q&A:      data/processed/qa_pairs.jsonl"
