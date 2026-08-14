#!/usr/bin/env python3
"""
Build BM25 index from document_chunks for hybrid search.
Run this after rebuilding FAISS embeddings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import bm25s
except ImportError:
    print("bm25s not installed. Run: pip install bm25s")
    sys.exit(1)

import Stemmer
from database import DB_PATH, get_db_connection

MODELS_DIR = Path(__file__).parent / "models"


def build_bm25_index():
    """Build BM25 index from database chunks."""
    print("[build_bm25] Loading chunks from database...")

    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM document_chunks")
        corpus = [row["text"] for row in cursor.fetchall()]

    if not corpus:
        print("[build_bm25] No chunks found in database!")
        return

    print(f"[build_bm25] Building index for {len(corpus)} documents...")

    # Tokenize
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stemmer=stemmer)

    # Build model
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # Save
    save_path = MODELS_DIR / "bm25_model"
    save_path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(save_path))

    print(f"[build_bm25] Index saved to {save_path}")


if __name__ == "__main__":
    build_bm25_index()