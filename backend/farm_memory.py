"""
FarmHand AI - Semantic Farm Memory & Clinical Observation Engine.
Provides persistent vector-embedded observation logging, semantic memory retrieval,
and clinical context formatting for RAG and Anomaly Detection.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import database
from database import (
    DB_PATH,
    get_active_farm_memories,
    get_all_farm_memories,
    record_health_log,
    resolve_farm_memory,
    save_farm_memory,
)


# -------------------------------------------------------------------
# Embedding & Semantic Math Helpers
# -------------------------------------------------------------------

def get_embedding_model():
    """Reuses the FastEmbed model singleton from rag_pipeline."""
    from rag_pipeline import get_embedding_model as rag_get_embedding_model
    return rag_get_embedding_model()


def embed_text(text: str) -> List[float]:
    """Generates a dense normalized vector embedding for observation text."""
    if not text or not text.strip():
        return []
    try:
        model = get_embedding_model()
        # FastEmbed returns a generator of numpy arrays
        embeddings = list(model.embed([text.strip()]))
        vec = embeddings[0].tolist()
        return vec
    except Exception as e:
        print(f"[farm_memory] Embedding generation exception: {e}")
        return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two dense vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# -------------------------------------------------------------------
# Memory Ingestion & Management
# -------------------------------------------------------------------

def log_and_embed_observation(
    farm_id: str,
    species: str,
    category: str,
    observation: str,
    source: str = "chat",
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Embeds a clinical observation or symptom, persists to farm_memories,
    and cross-records to health_logs.
    """
    text_to_embed = observation.strip()
    emb = embed_text(text_to_embed)

    # Persist structured memory
    memory = save_farm_memory(
        farm_id=farm_id,
        species=species,
        category=category,
        observation=observation,
        embedding=emb,
        source=source,
        db_path=db_path
    )

    # Cross-record in health_logs for unified medical history
    try:
        record_health_log(
            farm_id=farm_id,
            animal_id=f"{species.capitalize()}-Flock",
            event_type=f"memory_{category.lower()}",
            notes=f"[{category.capitalize()}] {observation}",
            db_path=db_path
        )
    except Exception as e:
        print(f"[farm_memory] Health log cross-record notice: {e}")

    print(f"[farm_memory] Logged active memory ID {memory.get('id')} for Farm '{farm_id}': {species} ({category})")
    return memory


# -------------------------------------------------------------------
# Semantic Retrieval for RAG & Chat
# -------------------------------------------------------------------

def search_farm_memories(
    farm_id: str,
    query: str,
    top_k: int = 3,
    threshold: float = 0.35,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """
    Retrieves the most semantically relevant active clinical memories for a farm query.
    Uses dense vector similarity with FastEmbed + keyword heuristic fallback.
    """
    active_mems = get_active_farm_memories(farm_id=farm_id, limit=30, db_path=db_path)
    if not active_mems:
        return []

    q_clean = query.lower().strip()
    q_emb = embed_text(query)

    scored_memories = []
    for mem in active_mems:
        m_emb = mem.get("embedding", [])
        m_obs = (mem.get("observation") or "").lower()
        m_sp = (mem.get("species") or "").lower()
        m_cat = (mem.get("category") or "").lower()

        # Pure dense vector similarity score
        v_score = cosine_similarity(q_emb, m_emb) if q_emb and m_emb else 0.0

        if v_score >= threshold:
            scored_memories.append({
                "id": mem.get("id"),
                "species": mem.get("species"),
                "category": mem.get("category"),
                "observation": mem.get("observation"),
                "created_at": mem.get("created_at"),
                "score": round(v_score, 3)
            })

    # Sort descending by relevance score
    scored_memories.sort(key=lambda x: x["score"], reverse=True)
    return scored_memories[:top_k]


def format_memories_for_rag(memories: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved memories into a clean markdown block for LLM prompt context.
    """
    if not memories:
        return ""

    lines = ["ACTIVE FARM OBSERVATIONS & LONG-TERM MEMORY:"]
    for m in memories:
        date_str = (m.get("created_at") or "")[:10]
        sp = m.get("species", "General")
        cat = m.get("category", "general")
        obs = m.get("observation", "")
        lines.append(f"- [{date_str}] {sp} ({cat}): {obs}")

    return "\n".join(lines)
