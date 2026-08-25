"""Shared helpers for Nigeria Agro Knowledge Corpus downloaders."""

from __future__ import annotations

import csv
import hashlib
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifests" / "sources.csv"

USER_AGENT = "AgroAI-CorpusBot/1.0 (+https://github.com/local/Agro-AI; open-access research corpus)"

MANIFEST_FIELDS = [
    "url_hash",
    "url",
    "title",
    "source",
    "category",
    "topics",
    "license",
    "local_path",
    "status",
    "http_status",
    "bytes",
    "notes",
]

ALLOWLISTED_DOMAINS = {
    "www.iita.org",
    "iita.org",
    "biblio.iita.org",
    "naerls.gov.ng",
    "www.naerls.gov.ng",
    "cgspace.cgiar.org",
    "hdl.handle.net",
    "www.fao.org",
    "fao.org",
    "openknowledge.fao.org",
    "www.icrisat.org",
    "icrisat.org",
    "oar.icrisat.org",
    "plantwiseplusknowledgebank.org",
    "www.plantwiseplusknowledgebank.org",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "www.njaat.com.ng",
    "ija.oauife.edu.ng",
    "www.openjournalsnigeria.org.ng",
    "openjournalsnigeria.org.ng",
    "khub.africacdc.org",
    "africacdc.org",
    "www.ilri.org",
    "ilri.org",
}


def ensure_dirs() -> None:
    for path in (
        RAW_DIR / "iita",
        RAW_DIR / "naerls",
        RAW_DIR / "cgspace",
        RAW_DIR / "plantwise",
        RAW_DIR / "fao",
        RAW_DIR / "journals",
        RAW_DIR / "hf_qa",
        RAW_DIR / "icrisat",
        RAW_DIR / "africacdc",
        EXTRACTED_DIR,
        PROCESSED_DIR,
        MANIFEST_PATH.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def is_allowlisted(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    bare = host[4:] if host.startswith("www.") else host
    return host in ALLOWLISTED_DOMAINS or bare in ALLOWLISTED_DOMAINS


def safe_filename(title: str, url: str, default_ext: str = ".pdf") -> str:
    base = re.sub(r"[^\w\-.]+", "_", title.strip())[:80].strip("._") or "document"
    path = urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext not in {".pdf", ".html", ".htm", ".json", ".jsonl", ".txt", ".xml"}:
        ext = default_ext
    return f"{base}__{url_hash(url)}{ext}"


def load_manifest() -> dict[str, dict[str, str]]:
    ensure_dirs()
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["url_hash"]: row for row in reader if row.get("url_hash")}


def save_manifest(rows: dict[str, dict[str, str]]) -> None:
    ensure_dirs()
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for key in sorted(rows.keys()):
            row = {field: rows[key].get(field, "") for field in MANIFEST_FIELDS}
            writer.writerow(row)


def upsert_manifest(rows: dict[str, dict[str, str]], record: dict[str, Any]) -> None:
    h = record.get("url_hash") or url_hash(record["url"])
    record = {**record, "url_hash": h}
    normalized = {field: str(record.get(field, "") or "") for field in MANIFEST_FIELDS}
    rows[h] = normalized


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return session


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    timeout: int = 60,
    sleep_s: float = 0.5,
) -> tuple[int, int, str]:
    """Download url to dest. Returns (http_status, bytes_written, notes)."""
    if not is_allowlisted(url):
        return 0, 0, "blocked: domain not allowlisted"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(
            url, stream=True, timeout=timeout, allow_redirects=True
        ) as resp:
            status = resp.status_code
            if status != 200:
                return status, 0, f"http_error:{status}"
            final_host = urlparse(resp.url).netloc.lower()
            if not is_allowlisted(resp.url) and final_host not in ALLOWLISTED_DOMAINS:
                return status, 0, "blocked: redirect domain not allowlisted"
            size = 0
            with dest.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        out.write(chunk)
                        size += len(chunk)
            time.sleep(sleep_s)
            return status, size, "ok"
    except requests.RequestException as exc:
        return 0, 0, f"request_error:{exc.__class__.__name__}:{exc}"


def topics_to_str(topics: list[str] | str | None) -> str:
    if topics is None:
        return ""
    if isinstance(topics, str):
        return topics
    return "|".join(topics)
