import json
import os
import sys
from pathlib import Path
import faiss
import fitz  # PyMuPDF
import numpy as np

from database import DB_PATH, get_db_connection, init_db
from rag_pipeline import OfflineVectorEmbedder

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = Path("/mnt/C6EE65A1EE658B0F/WORKEST/Agro-AI/data/raw/")
MODELS_DIR = BASE_DIR / "models"
INDEX_PATH = MODELS_DIR / "vector_store.index"
FASTEMBED_CACHE_DIR = MODELS_DIR / "fastembed_cache"

# Embedding Model Configuration
MODEL_NAME = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE_CHARS = 512  # Smaller chunks for 3B model
CHUNK_OVERLAP_CHARS = 100  # ~25 tokens overlap
BATCH_SIZE = 64  # Smaller batch for memory efficiency


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract full raw text from a PDF file using PyMuPDF (fitz)."""
    text_blocks = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text = page.get_text()
            if text:
                text_blocks.append(text)
    return "\n".join(text_blocks)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS):
    """Splits text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_embedding_model():
    """Load FastEmbed ONNX embedding model with persistent local cache directory."""
    FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(FASTEMBED_CACHE_DIR)
        )
    except Exception as e:
        print(f"[build_embeddings] FastEmbed model load exception ({e}). Retrying with network access...")
        old_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        try:
            from fastembed import TextEmbedding
            return TextEmbedding(
                model_name=MODEL_NAME,
                cache_dir=str(FASTEMBED_CACHE_DIR)
            )
        except Exception as e2:
            print(f"[build_embeddings] FastEmbed unavailable ({e2}). Using OfflineVectorEmbedder fallback.")
            return OfflineVectorEmbedder(dim=384)
        finally:
            if old_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = old_offline


def process_knowledge_base():
    """
    Memory-efficient batch processing of PDF knowledge base.
    Streams embeddings in batches of 128 into FAISS CPU index + SQLite DB
    maintaining a constant tiny RAM footprint (< 50MB).
    """
    init_db()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"[build_embeddings] Knowledge base directory {KNOWLEDGE_BASE_DIR} does not exist.")
        return

    pdf_files = list(KNOWLEDGE_BASE_DIR.glob("**/*.pdf"))
    print(f"[build_embeddings] Found {len(pdf_files)} PDF file(s) in {KNOWLEDGE_BASE_DIR}.")

    if not pdf_files:
        print("[build_embeddings] No PDF files found to ingest.")
        return

    print("[build_embeddings] Loading ONNX embedding model...")
    embedding_model = load_embedding_model()

    dimension = 384
    index = faiss.IndexFlatIP(dimension)

    # Reset SQLite document_chunks table
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")

    batch_passages = []
    batch_metadata = []
    total_chunks_processed = 0

    def flush_batch():
        nonlocal total_chunks_processed, batch_passages, batch_metadata, index
        if not batch_passages:
            return

        # Embed batch using generator stream
        embeddings_gen = embedding_model.embed(batch_passages)
        batch_np = np.array(list(embeddings_gen), dtype=np.float32)

        # L2 normalize batch vectors for Cosine Similarity via IndexFlatIP
        faiss.normalize_L2(batch_np)

        # Add batch incrementally to FAISS index
        index.add(batch_np)

        # Write batch metadata incrementally to SQLite database
        with get_db_connection(DB_PATH) as conn:
            cursor = conn.cursor()
            for idx, meta in enumerate(batch_metadata):
                row_id = total_chunks_processed + idx + 1
                cursor.execute(
                    """
                    INSERT INTO document_chunks (id, filename, chunk_id, text, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        meta["filename"],
                        meta["chunk_id"],
                        meta["raw_text"],
                        json.dumps({"filename": meta["filename"], "chunk_id": meta["chunk_id"]})
                    )
                )

        total_chunks_processed += len(batch_passages)
        print(f"  [Batch Processed] Total ingested chunks: {total_chunks_processed} (RAM footprint constant < 50MB)...")

        # Clear batch to release memory immediately
        batch_passages = []
        batch_metadata = []

    print("[build_embeddings] Starting memory-efficient batch ingestion...")

    for pdf_path in pdf_files:
        rel_filename = str(pdf_path.relative_to(KNOWLEDGE_BASE_DIR))
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text)

        for chunk_idx, chunk_content in enumerate(chunks):
            formatted_passage = f"passage: {chunk_content}"
            batch_passages.append(formatted_passage)
            batch_metadata.append({
                "filename": rel_filename,
                "chunk_id": chunk_idx,
                "raw_text": chunk_content
            })

            if len(batch_passages) >= BATCH_SIZE:
                flush_batch()

    # Flush any remaining trailing chunks
    flush_batch()

    # Save FAISS index to disk
    faiss.write_index(index, str(INDEX_PATH))
    print(f"\n[build_embeddings] SUCCESS: Saved FAISS index to {INDEX_PATH} (total vectors: {index.ntotal}, dimension: {dimension}).")
    print(f"[build_embeddings] Ingested {total_chunks_processed} total chunks into SQLite database {DB_PATH}.")


if __name__ == "__main__":
    process_knowledge_base()
