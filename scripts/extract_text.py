#!/usr/bin/env python3
"""Extract UTF-8 text from downloaded PDFs into data/extracted/."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_utils import EXTRACTED_DIR, MANIFEST_PATH, ROOT, ensure_dirs


def extract_with_pymupdf(path: Path) -> tuple[str, str]:
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    note = "pymupdf"
    if len(text) < 200:
        note = "needs_ocr_or_scanned"
    return text, note


def extract_with_pypdf(path: Path) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    text = "\n".join(parts).strip()
    note = "pypdf"
    if len(text) < 200:
        note = "needs_ocr_or_scanned"
    return text, note


def extract_pdf(path: Path) -> tuple[str, str]:
    try:
        return extract_with_pymupdf(path)
    except Exception:
        try:
            return extract_with_pypdf(path)
        except Exception as exc:  # noqa: BLE001
            return "", f"extract_error:{exc.__class__.__name__}:{exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=0, help="Optional max PDFs to process"
    )
    args = parser.parse_args()

    ensure_dirs()
    if not MANIFEST_PATH.exists():
        print("No manifest found. Run download/harvest scripts first.")
        return 1

    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    processed = 0
    needs_ocr = 0
    failed = 0
    meta_path = EXTRACTED_DIR / "_extract_meta.jsonl"
    meta_fh = meta_path.open("w", encoding="utf-8")

    for row in rows:
        if row.get("status") != "downloaded":
            continue
        local = row.get("local_path") or ""
        if not local:
            continue
        src = ROOT / local
        if not src.exists():
            continue
        if src.suffix.lower() != ".pdf":
            continue
        if args.limit and processed >= args.limit:
            break

        rel = Path(local)
        out = EXTRACTED_DIR / rel.with_suffix(".txt").name
        # keep source subdir name in filename prefix
        out = EXTRACTED_DIR / f"{row.get('source', 'doc')}__{src.stem}.txt"

        print(f"[extract] {src.name}")
        text, note = extract_pdf(src)
        if not text:
            failed += 1
            meta = {
                "source_file": local,
                "extracted_file": "",
                "chars": 0,
                "note": note,
                "title": row.get("title"),
            }
            meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            print(f"         FAIL {note}")
            continue

        out.write_text(text, encoding="utf-8")
        processed += 1
        if note == "needs_ocr_or_scanned":
            needs_ocr += 1
        meta = {
            "source_file": local,
            "extracted_file": str(out.relative_to(ROOT)),
            "chars": len(text),
            "note": note,
            "title": row.get("title"),
            "topics": row.get("topics"),
            "license": row.get("license"),
            "url": row.get("url"),
            "source": row.get("source"),
        }
        meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        print(f"         -> {out.relative_to(ROOT)} ({len(text)} chars) [{note}]")

    meta_fh.close()
    print(
        f"\nExtraction done. ok={processed} needs_ocr_flag={needs_ocr} failed={failed}"
    )
    print(f"Meta: {meta_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
