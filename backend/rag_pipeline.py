import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

import BM25
from database import DB_PATH, get_db_connection

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
INDEX_PATH = MODELS_DIR / "vector_store.index"
FASTEMBED_CACHE_DIR = MODELS_DIR / "fastembed_cache"

MODEL_NAME = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")


# -------------------------------------------------------------------
# 100% Offline Edge Vector Embedder Fallback
# -------------------------------------------------------------------

class OfflineVectorEmbedder:
    """
    Zero-network local vector embedder using ONNX feature hashing.
    Guarantees 100% offline edge operation without external HuggingFace downloads.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, documents: List[str]):
        embeddings = []
        for doc in documents:
            clean_doc = doc.lower().strip()
            vec = np.zeros(self.dim, dtype=np.float32)
            words = clean_doc.split()
            if not words:
                embeddings.append(vec)
                continue
            for word in words:
                h = hash(word)
                idx = abs(h) % self.dim
                val = 1.0 if h > 0 else -1.0
                vec[idx] += val
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return embeddings


# Global singleton instances for fast inference
_embedding_model: Any = None
_faiss_index: Optional[faiss.Index] = None
_bm25_retriever: Any = None

BM25_INDEX_PATH = MODELS_DIR / "bm25_index"


def get_embedding_model() -> Any:
    """
    Retrieve vector embedding model.
    Attempts FastEmbed ONNX loading with local cache. If offline/unavailable,
    falls back to OfflineVectorEmbedder to guarantee ZERO network dependencies.
    """
    global _embedding_model
    if _embedding_model is None:
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            from fastembed import TextEmbedding
            _embedding_model = TextEmbedding(
                model_name=MODEL_NAME,
                cache_dir=str(FASTEMBED_CACHE_DIR)
            )
        except Exception as e:
            print(f"[rag_pipeline] FastEmbed offline load fallback activated ({e}). Using OfflineVectorEmbedder.")
            _embedding_model = OfflineVectorEmbedder(dim=384)

    return _embedding_model


def get_faiss_index() -> Optional[faiss.Index]:
    global _faiss_index
    if _faiss_index is None:
        if INDEX_PATH.exists():
            _faiss_index = faiss.read_index(str(INDEX_PATH))
        else:
            _faiss_index = faiss.IndexFlatIP(384)
    return _faiss_index


def reload_faiss_index():
    global _faiss_index
    if INDEX_PATH.exists():
        _faiss_index = faiss.read_index(str(INDEX_PATH))
    return _faiss_index


def search_knowledge_base(search_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Hybrid search: BM25 keyword matching + FAISS vector similarity.
    Combines both ranking methods for better precision.
    """
    faiss_idx = get_faiss_index()
    if faiss_idx is None or faiss_idx.ntotal == 0:
        return []

    # Get BM25 results
    bm25_results = bm25_search(search_query, top_k=top_k * 3)

    # Get vector results
    vector_results = vector_search(search_query, top_k=top_k * 3)

    # Combine with weighted scoring
    combined = combine_results(bm25_results, vector_results, top_k)

    return combined[:top_k]


def vector_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Pure vector similarity search."""
    faiss_idx = get_faiss_index()
    if faiss_idx is None or faiss_idx.ntotal == 0:
        return []

    model = get_embedding_model()
    formatted_query = f"query: {query}"
    query_vector = list(model.embed([formatted_query]))[0]

    query_np = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_np)

    distances, indices = faiss_idx.search(query_np, min(top_k, faiss_idx.ntotal))

    results = []
    if len(indices) > 0 and len(indices[0]) > 0:
        hit_ids = [int(i) for i in indices[0] if i >= 0]
        if not hit_ids:
            return []

        sqlite_ids = [faiss_id + 1 for faiss_id in hit_ids]
        placeholders = ",".join(["?"] * len(sqlite_ids))

        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, filename, chunk_id, text FROM document_chunks WHERE id IN ({placeholders})",
                sqlite_ids
            )
            rows = {row["id"]: dict(row) for row in cursor.fetchall()}

        for dist, faiss_id in zip(distances[0], hit_ids):
            db_id = faiss_id + 1
            if db_id in rows:
                item = rows[db_id]
                item["score"] = float(dist)
                item["vector_rank"] = results.__len__()
                results.append(item)

    return results


def get_bm25_retriever() -> Any:
    """Load or build BM25 retriever."""
    global _bm25_retriever
    if _bm25_retriever is None:
        if BM25_INDEX_PATH.exists():
            print(f"[rag_pipeline] Loading BM25 index from {BM25_INDEX_PATH}")
            _bm25_retriever = BM25.load(str(BM25_INDEX_PATH))
        else:
            print(f"[rag_pipeline] BM25 index not found. Run build_bm25_index.py first.")
            return None
    return _bm25_retriever


def bm25_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 keyword-based search using library."""
    retriever = get_bm25_retriever()
    if retriever is None:
        return []

    results = retriever.search(query, k=top_k)

    # Get chunk IDs from BM25 results
    # BM25 library returns indices into the original corpus
    chunk_ids = []
    for r in results[0] if results else []:
        if hasattr(r, 'corpus_idx'):
            chunk_ids.append(r.corpus_idx + 1)  # BM25 uses 0-index
        elif isinstance(r, dict) and 'corpus_idx' in r:
            chunk_ids.append(r['corpus_idx'] + 1)

    if not chunk_ids:
        return []

    # Fetch from database
    placeholders = ",".join(["?"] * len(chunk_ids))
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT id, filename, chunk_id, text FROM document_chunks WHERE id IN ({placeholders})",
            chunk_ids
        )
        rows = {row["id"]: dict(row) for row in cursor.fetchall()}

    # Map results back with scores
    final_results = []
    for i, r in enumerate(results[0]) if results else []:
        if hasattr(r, 'corpus_idx'):
            db_id = r.corpus_idx + 1
        elif isinstance(r, dict) and 'corpus_idx' in r:
            db_id = r['corpus_idx'] + 1
        else:
            continue

        if db_id in rows:
            item = rows[db_id]
            item["score"] = r.score if hasattr(r, 'score') else r.get('score', 0)
            item["bm25_rank"] = i
            final_results.append(item)

    return final_results


def combine_results(bm25_results: List[Dict], vector_results: List[Dict], top_k: int) -> List[Dict]:
    """Combine BM25 and vector results with weighted scoring."""
    if not bm25_results:
        return vector_results[:top_k]
    if not vector_results:
        return bm25_results[:top_k]

    # Normalize scores
    max_bm25 = max(r["score"] for r in bm25_results) if bm25_results else 1
    max_vector = max(r["score"] for r in vector_results) if vector_results else 1

    # Build score map
    score_map = {}
    for r in bm25_results:
        key = r["id"]
        norm_score = r["score"] / max_bm25 if max_bm25 > 0 else 0
        score_map[key] = {"data": r, "combined": norm_score * 0.6}

    for r in vector_results:
        key = r["id"]
        norm_score = r["score"] / max_vector if max_vector > 0 else 0
        if key in score_map:
            score_map[key]["combined"] += norm_score * 0.4
        else:
            score_map[key] = {"data": r, "combined": norm_score * 0.4}

    # Sort by combined score
    sorted_results = sorted(score_map.values(), key=lambda x: x["combined"], reverse=True)

    return [item["data"] for item in sorted_results[:top_k]]


def query_knowledge_base(search_query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Tool function executed by tool_registry.
    Performs vector search, constructs strict prompt context, and returns retrieval output.
    """
    chunks = search_knowledge_base(search_query, top_k=top_k)

    if not chunks:
        return {
            "status": "empty",
            "message": f"No relevant documentation found for query '{search_query}'.",
            "context_prompt": f"No context available in knowledge base for query: {search_query}",
            "retrieved_chunks": []
        }

    formatted_context_parts = []
    for i, c in enumerate(chunks, 1):
        formatted_context_parts.append(
            f"--- Context Block {i} (Source: {c['filename']}) ---\n{c['text']}"
        )

    joined_context = "\n\n".join(formatted_context_parts)

    strict_prompt = (
        "Answer the user's question using ONLY the provided context below. "
        "If the answer cannot be found in the context, explicitly state that you do not know.\n\n"
        f"{joined_context}\n\n"
        f"User Question: {search_query}"
    )

    return {
        "status": "success",
        "search_query": search_query,
        "context_prompt": strict_prompt,
        "retrieved_chunks": [
            {"filename": c["filename"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for c in chunks
        ]
    }
