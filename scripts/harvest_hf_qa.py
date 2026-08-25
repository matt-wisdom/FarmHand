#!/usr/bin/env python3
"""Download Hugging Face Nigeria agro Q&A datasets into data/raw/hf_qa/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corpus_utils import (
    RAW_DIR,
    ROOT,
    ensure_dirs,
    load_manifest,
    save_manifest,
    topics_to_str,
    upsert_manifest,
)


def detect_columns(example: dict) -> tuple[str | None, str | None]:
    keys = {k.lower(): k for k in example}
    q_candidates = [
        "question",
        "instruction",
        "prompt",
        "input",
        "query",
        "enhanced_prompt",
    ]
    a_candidates = [
        "answer",
        "output",
        "response",
        "completion",
        "enhanced_completion",
        "target",
    ]
    q = next((keys[c] for c in q_candidates if c in keys), None)
    a = next((keys[c] for c in a_candidates if c in keys), None)
    return q, a


def iter_rows(ds_id: str):
    from datasets import load_dataset

    # Load each split independently to tolerate schema drift across splits.
    for split_name in ("train", "test", "validation"):
        try:
            split = load_dataset(ds_id, split=split_name)
            for row in split:
                yield split_name, dict(row)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] split={split_name} load failed for {ds_id}: {exc}")

        try:
            split = load_dataset(ds_id, split=split_name, streaming=True)
            for row in split:
                clean = dict(row)
                clean.pop("messages", None)
                yield split_name, clean
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] split={split_name} stream failed for {ds_id}: {exc}")
            continue


def export_via_hub_files(
    ds_id: str, languages: list[str], out_dir: Path
) -> tuple[Path, int]:
    """Fallback: download raw dataset files from the Hub and normalize to JSONL."""
    from huggingface_hub import hf_hub_download, list_repo_files

    print(f"[hf  ] hub-file fallback for {ds_id}")
    files = list_repo_files(ds_id, repo_type="dataset")
    cache_dir = out_dir / f"_{ds_id.replace('/', '__')}_files"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_files: list[Path] = []
    for name in files:
        if name.endswith((".json", ".jsonl", ".parquet", ".csv")) or name.startswith(
            "data/"
        ):
            if name.endswith((".md", ".txt")):
                continue
            path = Path(
                hf_hub_download(
                    ds_id, name, repo_type="dataset", local_dir=str(cache_dir)
                )
            )
            local_files.append(path)

    safe_name = ds_id.replace("/", "__")
    out_path = out_dir / f"{safe_name}.jsonl"
    n = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for path in local_files:
            if path.suffix == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            elif path.suffix == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
                rows = obj if isinstance(obj, list) else [obj]
            else:
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                clean = {k: v for k, v in row.items() if k != "messages"}
                q_col, a_col = detect_columns(clean)
                record = {
                    "dataset": ds_id,
                    "split": path.stem,
                    "languages": languages,
                    "raw": clean,
                }
                if q_col and a_col:
                    record["question"] = clean.get(q_col)
                    record["answer"] = clean.get(a_col)
                if not record.get("question") or not record.get("answer"):
                    continue
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                n += 1
    if n == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("hub-file fallback produced no rows")
    return out_path, n


def export_dataset(ds_id: str, languages: list[str], out_dir: Path) -> tuple[Path, int]:
    print(f"[hf  ] loading {ds_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = ds_id.replace("/", "__")
    out_path = out_dir / f"{safe_name}.jsonl"

    n = 0
    q_col = a_col = None
    with out_path.open("w", encoding="utf-8") as fh:
        for split_name, row in iter_rows(ds_id):
            if q_col is None:
                q_col, a_col = detect_columns(row)
            record = {
                "dataset": ds_id,
                "split": split_name,
                "languages": languages,
                "raw": row,
            }
            if q_col and a_col:
                record["question"] = row.get(q_col)
                record["answer"] = row.get(a_col)
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            n += 1
    if n == 0:
        out_path.unlink(missing_ok=True)
        return export_via_hub_files(ds_id, languages, out_dir)
    return out_path, n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, default=Path(__file__).with_name("sources_seed.yaml")
    )
    args = parser.parse_args()

    ensure_dirs()
    seed = yaml.safe_load(args.seed.read_text(encoding="utf-8"))
    datasets_cfg = seed.get("hf_datasets") or []
    rows = load_manifest()
    out_dir = RAW_DIR / "hf_qa"

    for cfg in datasets_cfg:
        ds_id = cfg["id"]
        languages = cfg.get("languages") or []
        url = f"https://huggingface.co/datasets/{ds_id}"
        try:
            path, n = export_dataset(ds_id, languages, out_dir)
            status = "downloaded"
            notes = f"rows={n}"
            local_path = str(path.relative_to(ROOT))
            print(f"       -> {local_path} ({n} rows)")
            http_status = 200
            size = path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            try:
                path, n = export_via_hub_files(ds_id, languages, out_dir)
                status = "downloaded"
                notes = f"rows={n};hub_fallback"
                local_path = str(path.relative_to(ROOT))
                print(f"       -> {local_path} ({n} rows) [hub fallback]")
                http_status = 200
                size = path.stat().st_size
            except Exception as exc2:  # noqa: BLE001
                status = "failed"
                notes = f"hf_error:{exc.__class__.__name__}:{exc}|fallback:{exc2}"
                local_path = ""
                http_status = 0
                size = 0
                print(f"       FAIL {notes}")

        upsert_manifest(
            rows,
            {
                "url": url,
                "title": ds_id,
                "source": "hf_qa",
                "category": "qa_dataset",
                "topics": topics_to_str(languages + ["qa", "agriculture"]),
                "license": "See dataset card on Hugging Face",
                "local_path": local_path,
                "status": status,
                "http_status": http_status,
                "bytes": size,
                "notes": notes,
            },
        )
        save_manifest(rows)

    print("HF Q&A harvest done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
