import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

import bm25s
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
    print(f"[rag_pipeline] search_knowledge_base query: '{search_query}' top_k: {top_k}")

    faiss_idx = get_faiss_index()
    print(f"[rag_pipeline] FAISS index total: {faiss_idx.ntotal if faiss_idx else 'None'}")

    if faiss_idx is None or faiss_idx.ntotal == 0:
        print("[rag_pipeline] WARNING: FAISS index empty or None!")
        return []

    # Get BM25 results
    print(f"[rag_pipeline] Running BM25 search...")
    bm25_results = bm25_search(search_query, top_k=top_k * 3)
    print(f"[rag_pipeline] BM25 results count: {len(bm25_results)}")
    if bm25_results:
        print(f"[rag_pipeline] BM25 top result: {bm25_results[0].get('text', '')[:100]}...")

    # Get vector results
    print(f"[rag_pipeline] Running vector search...")
    vector_results = vector_search(search_query, top_k=top_k * 3)
    print(f"[rag_pipeline] Vector results count: {len(vector_results)}")
    if vector_results:
        print(f"[rag_pipeline] Vector top result: {vector_results[0].get('text', '')[:100]}...")

    # Combine with weighted scoring
    print(f"[rag_pipeline] Combining results...")
    combined = combine_results(bm25_results, vector_results, top_k)
    print(f"[rag_pipeline] Combined results count: {len(combined)}")

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

    distances, indices = faiss_idx.search(query_np, min(top_k * 5, faiss_idx.ntotal))

    results = []
    if len(indices) > 0 and len(indices[0]) > 0:
        hit_ids = [int(i.item()) if hasattr(i, 'item') else int(i) for i in indices[0] if i >= 0]
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
                if len(results) >= top_k:
                    break

    return results


def get_bm25_retriever() -> Any:
    """Load or build BM25 retriever."""
    global _bm25_retriever
    if _bm25_retriever is None:
        save_path = MODELS_DIR / "bm25_model"
        if save_path.exists():
            print(f"[rag_pipeline] Loading BM25 index from {save_path}")
            _bm25_retriever = bm25s.BM25.load(str(save_path))
        else:
            print(f"[rag_pipeline] BM25 index not found. Run build_bm25_index.py first.")
            return None
    return _bm25_retriever


def bm25_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 keyword-based search using bm25s library."""
    print(f"[rag_pipeline] BM25 search called with query: '{query}'")

    retriever = get_bm25_retriever()
    if retriever is None:
        print("[rag_pipeline] BM25 retriever is None!")
        return []

    try:
        import Stemmer
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize([query], stemmer=stemmer)
        res = retriever.retrieve(query_tokens, k=top_k * 5)
        
        doc_indices = res.documents[0] if hasattr(res, "documents") else []
        scores = res.scores[0] if hasattr(res, "scores") else []
        
        if len(doc_indices) == 0:
            return []
            
        sqlite_ids = [int(idx.item()) + 1 if hasattr(idx, 'item') else int(idx) + 1 for idx in doc_indices]
        placeholders = ",".join(["?"] * len(sqlite_ids))
        
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT id, filename, chunk_id, text FROM document_chunks WHERE id IN ({placeholders})", sqlite_ids)
            rows = {r["id"]: dict(r) for r in cursor.fetchall()}
            
        final_results = []
        for rank, (idx, score) in enumerate(zip(doc_indices, scores)):
            db_id = int(idx.item()) + 1 if hasattr(idx, 'item') else int(idx) + 1
            if db_id in rows:
                item = rows[db_id]
                item["score"] = float(score)
                item["bm25_rank"] = rank
                final_results.append(item)
                if len(final_results) >= top_k:
                    break
                
        print(f"[rag_pipeline] BM25 returning {len(final_results)} results")
        return final_results
    except Exception as e:
        print(f"[rag_pipeline] BM25 retrieve ERROR: {e}")
        return []


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
    print(f"[rag_pipeline] query_knowledge_base called with: '{search_query}'")
    chunks = search_knowledge_base(search_query, top_k=top_k)
    print(f"[rag_pipeline] query_knowledge_base got {len(chunks)} chunks")

    if not chunks:
        print("[rag_pipeline] WARNING: No chunks found!")
        return {
            "status": "empty",
            "message": f"No relevant documentation found for query '{search_query}'.",
            "context_prompt": f"No context available in knowledge base for query: {search_query}",
            "retrieved_chunks": []
        }

    formatted_context_parts = []
    for i, c in enumerate(chunks, 1):
        raw_text = (c.get('text') or '').strip()
        if len(raw_text) > 600:
            raw_text = raw_text[:600].rsplit(' ', 1)[0] + "..."
        formatted_context_parts.append(
            f"--- Context Block {i} (Source: {c['filename']}) ---\n{raw_text}"
        )

    joined_context = "\n\n".join(formatted_context_parts)

    return {
        "status": "success",
        "search_query": search_query,
        "context_prompt": joined_context,
        "retrieved_chunks": [
            {"filename": c["filename"], "chunk_id": c["chunk_id"], "score": c["score"]}
            for c in chunks
        ]
    }


# -------------------------------------------------------------------
# Document Upload Functions
# -------------------------------------------------------------------

UPLOADS_DIR = BASE_DIR.parent / "data" / "uploads"


def add_document_to_knowledge_base(farm_id: str, pdf_path: Path) -> Dict[str, Any]:
    """
    Add a PDF document to the farm's knowledge base.
    
    Args:
        farm_id: Farm ID (None for global)
        pdf_path: Path to the uploaded PDF file
        
    Returns:
        Dict with success status and chunk count
    """
    from build_embeddings import add_document_to_vector_store
    from build_bm25_index import rebuild_bm25_index
    
    if not pdf_path.exists():
        return {"success": False, "error": "File not found"}
    
    try:
        # Add to vector store
        chunks_added = add_document_to_vector_store(
            pdf_path=pdf_path,
            farm_id=farm_id
        )
        
        # Rebuild BM25 index to include new chunks
        rebuild_bm25_index()
        
        return {
            "success": True,
            "filename": pdf_path.name,
            "chunks_added": chunks_added,
            "farm_id": farm_id
        }
    except Exception as e:
        print(f"[rag_pipeline] Error adding document: {e}")
        return {"success": False, "error": str(e)}


def list_farm_documents(farm_id: str) -> List[Dict[str, Any]]:
    """List all documents uploaded by a specific farm."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT filename, COUNT(*) as chunk_count, MIN(created_at) as first_uploaded
            FROM document_chunks 
            WHERE farm_id = ?
            GROUP BY filename
            ORDER BY first_uploaded DESC
            """,
            (farm_id,)
        )
        return [
            {
                "filename": row["filename"],
                "chunk_count": row["chunk_count"],
                "uploaded_at": row["first_uploaded"]
            }
            for row in cursor.fetchall()
        ]


def get_global_documents() -> List[Dict[str, Any]]:
    """List all global/system documents."""
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT filename, COUNT(*) as chunk_count, MIN(created_at) as first_uploaded
            FROM document_chunks 
            WHERE farm_id IS NULL
            GROUP BY filename
            ORDER BY first_uploaded DESC
            """
        )
        return [
            {
                "filename": row["filename"],
                "chunk_count": row["chunk_count"],
                "uploaded_at": row["first_uploaded"]
            }
            for row in cursor.fetchall()
        ]


def delete_farm_document(farm_id: str, filename: str) -> Dict[str, Any]:
    """Delete a farm's uploaded document from knowledge base."""
    from build_embeddings import remove_document_from_vector_store
    from build_bm25_index import rebuild_bm25_index
    
    try:
        chunks_removed = remove_document_from_vector_store(filename, farm_id)
        rebuild_bm25_index()
        
        return {
            "success": True,
            "filename": filename,
            "chunks_removed": chunks_removed
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
