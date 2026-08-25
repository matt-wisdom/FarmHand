#!/usr/bin/env python3
"""Harvest NAERLS extension bulletin PDFs, prioritizing livestock/animal topics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_utils import (
    RAW_DIR,
    ROOT,
    download_file,
    ensure_dirs,
    load_manifest,
    make_session,
    safe_filename,
    save_manifest,
    upsert_manifest,
    url_hash,
)

BULLETINS_URL = "https://naerls.gov.ng/bulletins/"

ANIMAL_KEYS = (
    "cattle",
    "sheep",
    "goat",
    "poultry",
    "livestock",
    "herd",
    "worm",
    "bull",
    "ram",
    "duck",
    "guinea",
    "fish",
    "feed",
    "hatchery",
    "harchery",
    "animal",
    "fowl",
    "snail",
    "aquaculture",
    "restrain",
    "transport",
    "pond",
    "dairy",
    "pig",
    "bee",
    "rabbit",
)


def collect_pdfs(session) -> list[tuple[str, str, bool]]:
    resp = session.get(BULLETINS_URL, timeout=60)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    out: list[tuple[str, str, bool]] = []
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(BULLETINS_URL, a["href"]).split("?")[0]
        if not href.lower().endswith(".pdf"):
            continue
        if href in seen:
            continue
        seen.add(href)
        title = a.get_text(" ", strip=True) or Path(href).name
        blob = f"{title} {href}".lower()
        is_animal = any(k in blob for k in ANIMAL_KEYS)
        out.append((title, href, is_animal))
    # animal first
    out.sort(key=lambda x: (0 if x[2] else 1, x[0].lower()))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-docs", type=int, default=80)
    parser.add_argument("--animals-only", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.6)
    args = parser.parse_args()

    ensure_dirs()
    session = make_session()
    rows = load_manifest()
    dest_dir = RAW_DIR / "naerls"
    candidates = collect_pdfs(session)
    if args.animals_only:
        candidates = [c for c in candidates if c[2]]
    print(f"Found {len(candidates)} NAERLS PDFs (animals_only={args.animals_only})")

    downloaded = failed = 0
    for title, url, is_animal in candidates:
        if downloaded >= args.max_docs:
            break
        h = url_hash(url)
        if h in rows and rows[h].get("status") == "downloaded":
            continue
        fname = safe_filename(title, url)
        dest = dest_dir / fname
        print(f"[get ] {'[animal] ' if is_animal else ''}{title[:80]}\n       {url}")
        status, size, notes = download_file(session, url, dest, sleep_s=args.sleep)
        if status == 200 and size > 0:
            downloaded += 1
            local_path = str(dest.relative_to(ROOT))
            record_status = "downloaded"
            print(f"       -> {local_path} ({size} bytes)")
        else:
            failed += 1
            local_path = ""
            record_status = "failed"
            dest.unlink(missing_ok=True)
            print(f"       FAIL status={status} notes={notes}")

        topics = "livestock|animal|husbandry" if is_animal else "extension|nigeria"
        upsert_manifest(
            rows,
            {
                "url": url,
                "title": title,
                "source": "naerls",
                "category": "livestock_husbandry" if is_animal else "extension_guide",
                "topics": topics,
                "license": "NAERLS extension bulletin (institutional)",
                "local_path": local_path,
                "status": record_status,
                "http_status": status,
                "bytes": size if record_status == "downloaded" else 0,
                "notes": notes if notes != "ok" else "harvest_naerls",
            },
        )
        save_manifest(rows)
        time.sleep(0.05)

    print(f"\nNAERLS harvest done. downloaded={downloaded} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
