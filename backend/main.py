import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import (
    DB_PATH,
    add_animal_record,
    add_chat_message,
    create_chat_thread,
    create_farm,
    delete_chat_thread,
    delete_farm,
    get_all_animals,
    get_all_expenditures,
    get_all_health_logs,
    get_chat_threads,
    get_farm_by_id,
    get_farms,
    get_thread_by_id,
    get_thread_messages,
    init_db,
    update_farm,
)
from llm_engine import chat_completion, get_llm
import rag_pipeline

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FarmHandBackend")


# -------------------------------------------------------------------
# FastAPI Lifespan Context Manager
# -------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler.
    Performs startup initializations (database tables, warm-up FastEmbed & FAISS)
    and clean shutdown releases.
    """
    logger.info("Initializing SQLite database tables (including multi-farm profiles & threads)...")
    init_db()

    logger.info("Warming up FastEmbed ONNX embedding session...")
    try:
        rag_pipeline.get_embedding_model()
        logger.info("FastEmbed model loaded successfully.")
    except Exception as e:
        logger.warning(f"FastEmbed initialization warning: {e}")

    logger.info("Loading FAISS vector index...")
    try:
        rag_pipeline.get_faiss_index()
        logger.info("FAISS vector index loaded.")
    except Exception as e:
        logger.warning(f"FAISS index load warning: {e}")

    logger.info("Checking llama.cpp Qwen model status...")
    llm = get_llm()
    if llm is not None:
        logger.info("llama.cpp LLM engine loaded successfully.")
    else:
        logger.warning("llama.cpp model file not found in backend/models/. Edge backend will use fallback router.")

    logger.info("FarmHand edge backend startup complete.")
    yield
    logger.info("FarmHand edge backend shutdown.")


# -------------------------------------------------------------------
# FastAPI Application Declaration
# -------------------------------------------------------------------

app = FastAPI(
    title="FarmHand AI Edge System API",
    description="Edge offline farm AI management system supporting multi-farm context isolation and ChatML tool calls.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for browser local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------------------------------------------------
# Pydantic Schemas
# -------------------------------------------------------------------

class FarmCreateRequest(BaseModel):
    name: str = Field(..., description="Farm name (e.g. Green Valley Poultry Pen)")
    farm_type: str = Field("General", description="Species/Type constraint ('Goat', 'Poultry', 'Cattle', 'Crops', 'Fish', 'General')")
    description: Optional[str] = Field("", description="Custom farm details and description")


class FarmUpdateRequest(BaseModel):
    name: str
    farm_type: str
    description: Optional[str] = ""


class CreateThreadRequest(BaseModel):
    title: Optional[str] = Field("New Chat", description="Thread title")
    farm_id: Optional[str] = Field("default_farm", description="Active farm ID")


class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="UUID of active chat thread")
    user_input: str = Field(..., description="User text prompt")
    farm_id: Optional[str] = Field("default_farm", description="Active farm ID")


class EditMessageRequest(BaseModel):
    thread_id: str = Field(..., description="UUID of active chat thread")
    message_index: int = Field(..., description="0-indexed position of message to edit")
    new_content: str = Field(..., description="Updated text prompt")
    farm_id: Optional[str] = Field("default_farm", description="Active farm ID")


class ChatResponse(BaseModel):
    status: str = "success"
    thread_id: str = Field(..., description="UUID of the chat thread")
    response: str = Field(..., description="Assistant final response string")


# -------------------------------------------------------------------
# Multi-Farm Management Endpoints
# -------------------------------------------------------------------

@app.get("/farms", tags=["Multi-Farm Management"])
def list_farms_endpoint():
    """GET /farms: Retrieve list of all registered farm profiles."""
    farms = get_farms()
    return {"status": "success", "count": len(farms), "farms": farms}


@app.post("/farms", tags=["Multi-Farm Management"])
def create_farm_endpoint(payload: FarmCreateRequest):
    """POST /farms: Create a new farm profile."""
    farm = create_farm(name=payload.name, farm_type=payload.farm_type, description=payload.description or "")
    return {"status": "success", "farm": farm}


@app.get("/farms/{farm_id}", tags=["Multi-Farm Management"])
def get_farm_endpoint(farm_id: str):
    """GET /farms/{farm_id}: Fetch specific farm profile details."""
    farm = get_farm_by_id(farm_id)
    if not farm:
        raise HTTPException(status_code=404, detail="Farm profile not found")
    return {"status": "success", "farm": farm}


@app.put("/farms/{farm_id}", tags=["Multi-Farm Management"])
def update_farm_endpoint(farm_id: str, payload: FarmUpdateRequest):
    """PUT /farms/{farm_id}: Update an existing farm profile."""
    updated = update_farm(farm_id, name=payload.name, farm_type=payload.farm_type, description=payload.description or "")
    if not updated:
        raise HTTPException(status_code=404, detail="Farm profile not found")
    return {"status": "success", "farm": updated}


@app.delete("/farms/{farm_id}", tags=["Multi-Farm Management"])
def delete_farm_endpoint(farm_id: str):
    """DELETE /farms/{farm_id}: Delete a farm profile."""
    if farm_id == "default_farm":
        raise HTTPException(status_code=400, detail="Cannot delete default farm profile")
    success = delete_farm(farm_id)
    if not success:
        raise HTTPException(status_code=404, detail="Farm profile not found")
    return {"status": "success", "message": f"Farm '{farm_id}' deleted."}


# -------------------------------------------------------------------
# Core Web & Session Endpoints
# -------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def read_root():
    """Serve single-page HTML/JS frontend at http://127.0.0.1:8000/."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "FarmHand AI Edge API running."}


@app.get("/health", tags=["System"])
@app.get("/metrics", tags=["System"])
def health_check():
    """Health check & real-time system/process RAM memory profiling endpoint."""
    faiss_idx = rag_pipeline.get_faiss_index()
    farms = get_farms()
    threads = get_chat_threads()

    process_ram_mb = 0.0
    sys_ram_total_mb = 0.0
    sys_ram_used_mb = 0.0
    sys_ram_percent = 0.0

    try:
        import psutil
        process = psutil.Process()
        process_ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
        vm = psutil.virtual_memory()
        sys_ram_total_mb = round(vm.total / (1024 * 1024), 2)
        sys_ram_used_mb = round(vm.used / (1024 * 1024), 2)
        sys_ram_percent = vm.percent
    except Exception as e:
        logger.warning(f"Error reading psutil memory metrics: {e}")

    return {
        "status": "healthy",
        "database_path": str(DB_PATH),
        "database_ready": DB_PATH.exists(),
        "vector_store_ready": rag_pipeline.INDEX_PATH.exists(),
        "vector_count": faiss_idx.ntotal if faiss_idx else 0,
        "llm_model_ready": get_llm() is not None,
        "process_ram_mb": process_ram_mb,
        "system_ram_total_mb": sys_ram_total_mb,
        "system_ram_used_mb": sys_ram_used_mb,
        "system_ram_percent": sys_ram_percent,
        "total_farms": len(farms),
        "total_threads": len(threads)
    }


@app.post("/threads", tags=["Session Management"])
def create_thread_endpoint(payload: Optional[CreateThreadRequest] = None):
    """POST /threads: Initializes and returns a new empty chat thread/session UUID for a farm."""
    title = payload.title if payload and payload.title else "New Chat"
    farm_id = payload.farm_id if payload and payload.farm_id else "default_farm"
    thread_id = create_chat_thread(title=title, farm_id=farm_id)
    return {
        "status": "success",
        "thread_id": thread_id,
        "farm_id": farm_id,
        "title": title
    }


@app.get("/threads", tags=["Session Management"])
def get_threads_endpoint(farm_id: str = "default_farm"):
    """GET /threads: Returns list of threads for the active farm sorted by updated_at DESC."""
    threads = get_chat_threads(farm_id=farm_id)
    return {
        "status": "success",
        "farm_id": farm_id,
        "count": len(threads),
        "threads": threads
    }


@app.get("/threads/{thread_id}", tags=["Session Management"])
def get_thread_history_endpoint(thread_id: str):
    """GET /threads/{thread_id}: Returns message history for a specific thread."""
    messages = get_thread_messages(thread_id)
    return {
        "status": "success",
        "thread_id": thread_id,
        "count": len(messages),
        "messages": messages
    }


@app.delete("/threads/{thread_id}", tags=["Session Management"])
def delete_thread_endpoint(thread_id: str):
    """DELETE /threads/{thread_id}: Deletes a thread and all associated messages."""
    delete_chat_thread(thread_id)
    return {
        "status": "success",
        "message": f"Thread '{thread_id}' deleted."
    }


@app.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
def chat_endpoint(payload: ChatRequest):
    """POST /chat: Process user input for an active thread within the active farm context."""
    thread_id = payload.thread_id
    user_input = payload.user_input.strip()
    farm_id = payload.farm_id or "default_farm"

    if not user_input:
        raise HTTPException(status_code=400, detail="user_input cannot be empty.")

    # 1. Fetch thread history from DB
    history_rows = get_thread_messages(thread_id)
    messages: List[Dict[str, str]] = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    # 2. Append new user prompt
    messages.append({"role": "user", "content": user_input})
    add_chat_message(thread_id, "user", user_input)

    # 3. Process LLM completion with farm_id context
    try:
        assistant_response = chat_completion(messages, farm_id=farm_id, thread_id=thread_id)
    except Exception as e:
        logger.error(f"Error in chat_completion: {e}")
        assistant_response = f"Internal system error: {str(e)}"

    # 4. Save assistant response to DB
    add_chat_message(thread_id, "assistant", assistant_response)

    return ChatResponse(
        status="success",
        thread_id=thread_id,
        response=assistant_response
    )


@app.post("/threads/chat", response_model=ChatResponse, tags=["AI Chat"])
def thread_chat_alias(payload: ChatRequest):
    """POST /threads/chat: Alias endpoint for POST /chat."""
    return chat_endpoint(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
