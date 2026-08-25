#!/usr/bin/env python3
"""Harvest open-access Nigeria agro documents from CGSpace via DSpace 7 REST API."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import yaml

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

SEARCH_URL = (
    "https://cgspace.cgiar.org/server/api/discover/search/objects"
    "?query={query}&dsoType=Item&size={size}&page={page}"
)


def search_items(
    session, query: str, max_pages: int = 2, page_size: int = 20
) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for page in range(max_pages):
        url = SEARCH_URL.format(query=quote_plus(query), size=page_size, page=page)
        resp = session.get(url, timeout=90)
        if resp.status_code == 429:
            print("[warn] rate limited; sleeping 20s")
            time.sleep(20)
            resp = session.get(url, timeout=90)
        if resp.status_code != 200:
            print(f"[warn] search failed {resp.status_code}: {query} page={page}")
            break
        data = resp.json()
        embedded = (
            data.get("_embedded", {})
            .get("searchResult", {})
            .get("_embedded", {})
            .get("objects", [])
        )
        if not embedded:
            break
        for obj in embedded:
            item = obj.get("_embedded", {}).get("indexableObject", {})
            uuid = item.get("uuid")
            if not uuid or uuid in seen:
                continue
            seen.add(uuid)
            items.append(item)
        time.sleep(1.2)
    return items


def original_pdfs(session, item: dict) -> list[tuple[str, str]]:
    title = item.get("name") or "cgspace_item"
    bundles_url = item.get("_links", {}).get("bundles", {}).get("href")
    if not bundles_url:
        return []
    resp = session.get(bundles_url, timeout=60)
    if resp.status_code != 200:
        return []
    out: list[tuple[str, str]] = []
    for bundle in resp.json().get("_embedded", {}).get("bundles", []):
        if (bundle.get("name") or "").upper() != "ORIGINAL":
            continue
        bs_url = bundle.get("_links", {}).get("bitstreams", {}).get("href")
        if not bs_url:
            continue
        bs_resp = session.get(bs_url, timeout=60)
        if bs_resp.status_code != 200:
            continue
        for bit in bs_resp.json().get("_embedded", {}).get("bitstreams", []):
            name = bit.get("name") or ""
            content = bit.get("_links", {}).get("content", {}).get("href")
            if content and name.lower().endswith(".pdf"):
                out.append((f"{title} - {name}", content))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, default=Path(__file__).with_name("sources_seed.yaml")
    )
    parser.add_argument("--max-docs", type=int, default=40)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    ensure_dirs()
    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    queries = seed.get("cgspace_queries") or ["Nigeria agriculture"]
    rows = load_manifest()
    session = make_session()
    dest_dir = RAW_DIR / "cgspace"

    items: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        print(f"[search] {q}")
        for item in search_items(session, q, max_pages=args.max_pages):
            uuid = item.get("uuid")
            if uuid and uuid not in seen:
                seen.add(uuid)
                items.append(item)
        time.sleep(args.sleep)

    print(f"Found {len(items)} unique CGSpace items")
    # Prefer animal/livestock-titled items first
    animal_hint = (
        "livestock",
        "cattle",
        "goat",
        "sheep",
        "poultry",
        "animal",
        "ppr",
        "disease",
        "ruminant",
        "veterinary",
        "dairy",
        "avian",
        "fish",
        "pig",
    )

    def animal_rank(item: dict) -> tuple[int, str]:
        name = (item.get("name") or "").lower()
        return (0 if any(h in name for h in animal_hint) else 1, name)

    items.sort(key=animal_rank)
    downloaded = failed = 0

    for item in items:
        if downloaded >= args.max_docs:
            break
        pdfs = original_pdfs(session, item)
        time.sleep(args.sleep)
        item_url = item.get("_links", {}).get("self", {}).get("href", "")
        for title, pdf_url in pdfs:
            if downloaded >= args.max_docs:
                break
            h = url_hash(pdf_url)
            if h in rows and rows[h].get("status") == "downloaded":
                continue
            fname = safe_filename(title, pdf_url)
            dest = dest_dir / fname
            print(f"[get ] {title[:90]}\n       {pdf_url}")
            status, size, notes = download_file(
                session, pdf_url, dest, sleep_s=args.sleep
            )
            if status == 200 and size > 0 and dest.read_bytes()[:5] == b"%PDF-":
                downloaded += 1
                record_status = "downloaded"
                local_path = str(dest.relative_to(ROOT))
                print(f"       -> {local_path} ({size} bytes)")
            else:
                failed += 1
                record_status = "failed"
                local_path = ""
                if notes == "ok" and status == 200:
                    notes = "not_a_pdf"
                dest.unlink(missing_ok=True)
                print(f"       FAIL status={status} notes={notes}")

            upsert_manifest(
                rows,
                {
                    "url": pdf_url,
                    "title": title,
                    "source": "cgspace",
                    "category": "research",
                    "topics": "nigeria|cgiar|open_access",
                    "license": "CGIAR open access (verify per item)",
                    "local_path": local_path,
                    "status": record_status,
                    "http_status": status,
                    "bytes": size if record_status == "downloaded" else 0,
                    "notes": notes if notes != "ok" else f"item:{item_url}",
                },
            )
            save_manifest(rows)

    print(f"\nCGSpace harvest done. downloaded={downloaded} failed={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
