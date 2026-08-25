#!/usr/bin/env python3
"""Chunk extracted texts and normalize HF Q&A into processed JSONL files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_utils import (
    EXTRACTED_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    ROOT,
    ensure_dirs,
)

CROP_TERMS = [
    "rice",
    "maize",
    "corn",
    "cassava",
    "cowpea",
    "soybean",
    "groundnut",
    "peanut",
    "tomato",
    "onion",
    "pepper",
    "yam",
    "sorghum",
    "millet",
    "cocoa",
    "oil palm",
    "plantain",
    "banana",
]

LIVESTOCK_TERMS = [
    "cattle",
    "cow",
    "bull",
    "goat",
    "sheep",
    "ram",
    "poultry",
    "chicken",
    "duck",
    "guinea fowl",
    "fish",
    "aquaculture",
]

DISEASE_PEST_TERMS = [
    "disease",
    "pest",
    "virus",
    "bacteria",
    "fungus",
    "blight",
    "mosaic",
    "rust",
    "wilt",
    "rot",
    "striga",
    "armyworm",
    "borer",
    "aphid",
    "whitefly",
    "rosette",
    "newcastle",
    "ppr",
    "worm",
    "tick",
    "quarantine",
    "vaccinat",
]

ACTION_TERMS = [
    "spray",
    "apply",
    "treat",
    "treatment",
    "vaccinate",
    "rogue",
    "remove",
    "quarantine",
    "isolate",
    "mulch",
    "irrigate",
    "fertiliz",
    "weed",
    "plant",
    "harvest",
    "feed",
    "deworm",
    "disinfect",
    "rotate",
    "burn",
    "prune",
]


def approx_tokens(text: str) -> int:
    return max(1, len(text.split()))


def chunk_text(
    text: str, target_tokens: int = 650, overlap_tokens: int = 80
) -> list[str]:
    # Prefer paragraph / section splits
    parts = re.split(r"\n\s*\n+", text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf = []
            buf_tokens = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        t = approx_tokens(part)
        if t > target_tokens * 1.5:
            flush()
            words = part.split()
            step = max(1, target_tokens - overlap_tokens)
            for i in range(0, len(words), step):
                window = " ".join(words[i : i + target_tokens])
                if window.strip():
                    chunks.append(window.strip())
            continue
        if buf_tokens + t > target_tokens and buf:
            flush()
            # overlap: keep last fragment words
            if chunks and overlap_tokens > 0:
                prev_words = chunks[-1].split()[-overlap_tokens:]
                buf = [" ".join(prev_words)]
                buf_tokens = len(prev_words)
        buf.append(part)
        buf_tokens += t
    flush()
    return [c for c in chunks if c]


def tag_text(text: str) -> dict:
    low = text.lower()
    crops = [c for c in CROP_TERMS if c in low]
    livestock = [c for c in LIVESTOCK_TERMS if c in low]
    disease = [c for c in DISEASE_PEST_TERMS if c in low]
    actions = [c for c in ACTION_TERMS if c in low]
    tags = []
    if crops:
        tags.append("crop")
    if livestock:
        tags.append("livestock")
    if disease:
        tags.append("disease_or_pest")
    if actions:
        tags.append("actionable")
    if "nigeria" in low:
        tags.append("nigeria")
    return {
        "crops": crops,
        "livestock": livestock,
        "disease_pest_terms": disease,
        "action_terms": actions,
        "tags": tags,
    }


def load_extract_meta() -> dict[str, dict]:
    meta_path = EXTRACTED_DIR / "_extract_meta.jsonl"
    mapping: dict[str, dict] = {}
    if not meta_path.exists():
        return mapping
    with meta_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            ef = row.get("extracted_file")
            if ef:
                mapping[ef] = row
    return mapping


def write_chunks(target_tokens: int, overlap_tokens: int) -> int:
    meta = load_extract_meta()
    out_path = PROCESSED_DIR / "chunks.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted(EXTRACTED_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = str(path.relative_to(ROOT))
            info = meta.get(rel, {})
            chunks = chunk_text(
                text, target_tokens=target_tokens, overlap_tokens=overlap_tokens
            )
            for i, chunk in enumerate(chunks):
                tags = tag_text(chunk)
                record = {
                    "id": f"{path.stem}::{i}",
                    "text": chunk,
                    "char_count": len(chunk),
                    "approx_tokens": approx_tokens(chunk),
                    "source_file": info.get("source_file", ""),
                    "extracted_file": rel,
                    "title": info.get("title", path.stem),
                    "url": info.get("url", ""),
                    "license": info.get("license", ""),
                    "source": info.get("source", ""),
                    "topics": info.get("topics", ""),
                    **tags,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
    print(f"[chunks] wrote {n} chunks -> {out_path.relative_to(ROOT)}")
    return n


def guess_language(text: str, default_langs: list[str] | None = None) -> str:
    if not text:
        return (default_langs or ["unknown"])[0]
    # very light heuristic
    if re.search(r"\b(wetin|dey|una|abi|naf|no be)\b", text, re.IGNORECASE):
        return "pcm"
    if re.search(r"[ẹọṣẸỌṢ]", text) or re.search(
        r"\b(jẹ|ṣe|ni|awọn)\b", text, re.IGNORECASE
    ):
        return "yo"
    if re.search(
        r"\b(ina|ne|da|ba|wannan|shin)\b", text, re.IGNORECASE
    ) and not re.search(r"\b(the|and|what|how)\b", text, re.IGNORECASE):
        return "ha"
    if default_langs:
        return default_langs[0]
    return "en"


def write_qa_pairs() -> int:
    out_path = PROCESSED_DIR / "qa_pairs.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for path in sorted((RAW_DIR / "hf_qa").glob("*.jsonl")):
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    q = row.get("question")
                    a = row.get("answer")
                    raw = row.get("raw") or {}
                    if not q or not a:
                        # try common alternates inside raw
                        q = (
                            q
                            or raw.get("enhanced_prompt")
                            or raw.get("instruction")
                            or raw.get("question")
                        )
                        a = (
                            a
                            or raw.get("enhanced_completion")
                            or raw.get("output")
                            or raw.get("answer")
                        )
                    if not q or not a:
                        continue
                    langs = row.get("languages") or []
                    lang = (
                        raw.get("dialect")
                        or raw.get("language")
                        or guess_language(str(q), langs)
                    )
                    record = {
                        "id": f"{row.get('dataset')}::{row.get('split')}::{n}",
                        "question": str(q).strip(),
                        "answer": str(a).strip(),
                        "language": str(lang).lower() if lang else "unknown",
                        "source": row.get("dataset"),
                        "split": row.get("split"),
                        "tags": tag_text(f"{q}\n{a}")["tags"],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
    print(f"[qa   ] wrote {n} pairs -> {out_path.relative_to(ROOT)}")
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-tokens", type=int, default=650)
    parser.add_argument("--overlap-tokens", type=int, default=80)
    args = parser.parse_args()

    ensure_dirs()
    write_chunks(args.target_tokens, args.overlap_tokens)
    write_qa_pairs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
