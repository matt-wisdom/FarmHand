#!/usr/bin/env python3
"""Download seed open-access Nigeria agro sources listed in sources_seed.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    topics_to_str,
    upsert_manifest,
    url_hash,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=Path,
        default=Path(__file__).with_name("sources_seed.yaml"),
        help="Path to seed YAML",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.75, help="Delay between downloads"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if present"
    )
    args = parser.parse_args()

    ensure_dirs()
    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    sources = seed.get("sources") or []
    rows = load_manifest()
    session = make_session()

    ok = skipped = failed = 0
    for item in sources:
        url = item["url"]
        h = url_hash(url)
        source = item.get("source", "other")
        title = item.get("title", "document")
        dest_dir = RAW_DIR / source
        dest_dir.mkdir(parents=True, exist_ok=True)

        fname = safe_filename(title, url)
        dest = dest_dir / fname

        existing = rows.get(h)
        if (
            existing
            and existing.get("status") == "downloaded"
            and Path(existing.get("local_path", "")).exists()
            and not args.force
        ):
            skipped += 1
            print(f"[skip] {title}")
            continue

        print(f"[get ] {title}\n       {url}")
        status, size, notes = download_file(session, url, dest, sleep_s=args.sleep)
        if status == 200 and size > 0:
            record_status = "downloaded"
            ok += 1
            local_path = str(dest.relative_to(ROOT))
            print(f"       -> {local_path} ({size} bytes)")
        else:
            record_status = "failed"
            failed += 1
            local_path = ""
            if dest.exists() and dest.stat().st_size == 0:
                dest.unlink(missing_ok=True)
            print(f"       FAIL status={status} notes={notes}")

        upsert_manifest(
            rows,
            {
                "url": url,
                "title": title,
                "source": source,
                "category": item.get("category", ""),
                "topics": topics_to_str(item.get("topics")),
                "license": item.get("license", ""),
                "local_path": local_path,
                "status": record_status,
                "http_status": status,
                "bytes": size,
                "notes": notes if notes != "ok" else (item.get("notes") or ""),
            },
        )
        save_manifest(rows)

    print(f"\nDone. downloaded={ok} skipped={skipped} failed={failed}")
    print("Manifest: data/manifests/sources.csv")
    return 0 if failed == 0 or ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
