#!/usr/bin/env python3
"""Crawl IITA listing pages for Nigeria production-guide PDFs."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_utils import (
    RAW_DIR,
    ROOT,
    download_file,
    ensure_dirs,
    is_allowlisted,
    load_manifest,
    make_session,
    safe_filename,
    save_manifest,
    upsert_manifest,
    url_hash,
)

NIGERIA_HINT = re.compile(r"nigeria|northern nigeria|west africa", re.IGNORECASE)
GUIDE_HINT = re.compile(
    r"guide|production|handbook|manual|extension|bulletin", re.IGNORECASE
)


def collect_pdf_links(session, page_url: str) -> list[tuple[str, str]]:
    resp = session.get(page_url, timeout=60)
    if resp.status_code != 200:
        print(f"[warn] listing failed {resp.status_code}: {page_url}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    found: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(" ", strip=True)
        full = urljoin(page_url, href).split("?")[0]
        if not full.lower().endswith(".pdf"):
            continue
        if not is_allowlisted(full):
            continue
        label = text or Path(urlparse(full).path).name
        blob = f"{label} {full}"
        if NIGERIA_HINT.search(blob) or GUIDE_HINT.search(blob):
            found.append((label, full))
    # also follow a few same-domain child pages that look like publication lists
    child_pages = []
    for a in soup.find_all("a", href=True):
        full = urljoin(page_url, a["href"]).split("#")[0]
        if (
            urlparse(full).netloc.endswith("iita.org")
            and GUIDE_HINT.search(a.get_text(" ", strip=True) + full)
            and full not in child_pages
            and full != page_url
        ):
            child_pages.append(full)
    for child in child_pages[:8]:
        time.sleep(0.5)
        resp = session.get(child, timeout=60)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            full = urljoin(child, a["href"]).split("?")[0]
            if full.lower().endswith(".pdf") and is_allowlisted(full):
                label = a.get_text(" ", strip=True) or Path(urlparse(full).path).name
                found.append((label, full))
    # dedupe
    out = []
    seen = set()
    for t, u in found:
        if u not in seen:
            seen.add(u)
            out.append((t, u))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, default=Path(__file__).with_name("sources_seed.yaml")
    )
    parser.add_argument("--max-docs", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.75)
    args = parser.parse_args()

    ensure_dirs()
    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    listing_urls = seed.get("iita_listing_urls") or []
    # Always include known IITA upload directory index is not browsable; rely on listings + seed.
    session = make_session()
    rows = load_manifest()
    dest_dir = RAW_DIR / "iita"

    candidates: list[tuple[str, str]] = []
    for page in listing_urls:
        print(f"[crawl] {page}")
        candidates.extend(collect_pdf_links(session, page))
        time.sleep(args.sleep)

    # Prefer Nigeria-titled guides first
    candidates.sort(
        key=lambda x: (0 if NIGERIA_HINT.search(x[0] + x[1]) else 1, x[0].lower())
    )
    print(f"Found {len(candidates)} candidate IITA PDFs")

    downloaded = failed = 0
    for title, url in candidates:
        if downloaded >= args.max_docs:
            break
        h = url_hash(url)
        if h in rows and rows[h].get("status") == "downloaded":
            continue
        fname = safe_filename(title, url)
        dest = dest_dir / fname
        print(f"[get ] {title[:80]}\n       {url}")
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

        upsert_manifest(
            rows,
            {
                "url": url,
                "title": title,
                "source": "iita",
                "category": "extension_guide",
                "topics": "iita|nigeria|production",
                "license": "IITA open publication (institutional)",
                "local_path": local_path,
                "status": record_status,
                "http_status": status,
                "bytes": size if record_status == "downloaded" else 0,
                "notes": notes if notes != "ok" else "harvest_iita",
            },
        )
        save_manifest(rows)

    print(f"\nIITA harvest done. downloaded={downloaded} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
