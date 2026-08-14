#!/usr/bin/env python3
"""
Build BM25 index from document_chunks for hybrid search.
Run this after rebuilding FAISS embeddings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import BM25
from database import DB_PATH, get_db_connection

MODELS_DIR = Path(__file__).parent / "models"
BM25_INDEX_PATH = MODELS_DIR / "bm25_index"


def build_bm25_index():
    """Build and save BM25 index from database chunks."""
    print("[build_bm25] Loading chunks from database...")

    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM document_chunks")
        corpus = [row["text"] for row in cursor.fetchall()]

    if not corpus:
        print("[build_bm25] No chunks found in database!")
        return

    print(f"[build_bm25] Building index for {len(corpus)} documents...")

    retriever = BM25.index(corpus)

    BM25_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    retriever.save(str(BM25_INDEX_PATH))

    print(f"[build_bm25] Index saved to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    build_bm25_index()