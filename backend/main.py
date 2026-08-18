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
    record_expenditure,
    get_all_health_logs,
    record_health_log,
    get_telemetry_data,
    get_chat_threads,
    get_current_flock_totals,
    get_farm_by_id,
    get_farms,
    get_flock_count_on_date,
    get_flock_ledger_history,
    get_latest_ledger_anomaly,
    get_ledger_anomaly_history,
    get_active_farm_memories,
    get_all_farm_memories,
    resolve_farm_memory,
    delete_farm_memory,
    get_thread_by_id,
    get_thread_messages,
    init_db,
    record_flock_event,
    update_farm,
)
from llm_engine import chat_completion, get_llm
import rag_pipeline
import anomaly_detector
import farm_memory

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FarmHandBackend")


class HealthLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(HealthLogFilter())


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
    language: Optional[str] = Field("english", description="Language: english, hausa, pidgin")


class EditMessageRequest(BaseModel):
    thread_id: str = Field(..., description="UUID of active chat thread")
    message_index: int = Field(..., description="0-indexed position of message to edit")
    new_content: str = Field(..., description="Updated text prompt")
    farm_id: Optional[str] = Field("default_farm", description="Active farm ID")


class ChatResponse(BaseModel):
    status: str = "success"
    thread_id: str = Field(..., description="UUID of the chat thread")
    response: str = Field(..., description="Assistant final response string")


class FlockEventRequest(BaseModel):
    species: str = Field("Poultry", description="Species name")
    count_change: Optional[int] = Field(0, description="Change in count (positive for additions, negative for losses)")
    exact_total: Optional[int] = Field(None, description="Direct new total count (for initial setup or reset)")
    event_type: str = Field("count_update", description="Event type: initial_count, purchase, mortality, sale, count_update")
    notes: Optional[str] = Field("", description="Optional event description or notes")


class FarmMemoryRequest(BaseModel):
    species: str = Field("General", description="Species observed")
    category: str = Field("symptom", description="Category: symptom, behavior, treatment, feeding, general")
    observation: str = Field(..., description="Clinical or behavioral observation description")


class ExpenditureRequest(BaseModel):
    category: str = Field("feed", description="Category of expenditure (e.g. feed, veterinary, equipment, operations)")
    amount: float = Field(..., gt=0, description="Amount spent in NGN")
    description: Optional[str] = Field("", description="Description of the expenditure")


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
# Flock & Herd Count Ledger Endpoints
# -------------------------------------------------------------------

@app.get("/api/farms/{farm_id}/flock-ledger", tags=["Flock Ledger"])
def get_flock_ledger_endpoint(farm_id: str):
    """GET /api/farms/{farm_id}/flock-ledger: Returns current flock totals and recent ledger history."""
    totals = get_current_flock_totals(farm_id=farm_id)
    history = get_flock_ledger_history(farm_id=farm_id, limit=50)
    return {
        "status": "success",
        "farm_id": farm_id,
        "current_totals": totals,
        "total_flock_size": sum(totals.values()),
        "history": history
    }


@app.post("/api/farms/{farm_id}/flock-ledger", tags=["Flock Ledger"])
def record_flock_event_endpoint(farm_id: str, payload: FlockEventRequest):
    """POST /api/farms/{farm_id}/flock-ledger: Records a new count event and evaluates ledger anomalies."""
    entry = record_flock_event(
        farm_id=farm_id,
        species=payload.species,
        count_change=payload.count_change or 0,
        exact_total=payload.exact_total,
        event_type=payload.event_type,
        notes=payload.notes or ""
    )
    # Automatically trigger anomaly detection upon ledger update
    anomaly_eval = anomaly_detector.run_flock_anomaly_detection(farm_id=farm_id, trigger_source="api_ledger_post")
    return {
        "status": "success",
        "farm_id": farm_id,
        "entry": entry,
        "anomaly_evaluation": anomaly_eval
    }


@app.get("/api/farms/{farm_id}/flock-ledger/anomalies", tags=["Flock Ledger"])
def get_flock_anomalies_endpoint(farm_id: str):
    """GET /api/farms/{farm_id}/flock-ledger/anomalies: Returns latest anomaly evaluation and history."""
    latest = get_latest_ledger_anomaly(farm_id=farm_id)
    history = get_ledger_anomaly_history(farm_id=farm_id, limit=10)
    return {
        "status": "success",
        "farm_id": farm_id,
        "latest": latest,
        "history": history
    }


@app.post("/api/farms/{farm_id}/flock-ledger/anomalies/run", tags=["Flock Ledger"])
def trigger_flock_anomalies_endpoint(farm_id: str):
    """POST /api/farms/{farm_id}/flock-ledger/anomalies/run: Manually triggers anomaly analysis."""
    record = anomaly_detector.run_flock_anomaly_detection(farm_id=farm_id, trigger_source="manual_ui_trigger")
    return {
        "status": "success",
        "farm_id": farm_id,
        "anomaly_report": record
    }


@app.get("/api/farms/{farm_id}/flock-ledger/count", tags=["Flock Ledger"])
def get_flock_count_endpoint(farm_id: str, species: Optional[str] = None, target_date: Optional[str] = None):
    """GET /api/farms/{farm_id}/flock-ledger/count: Queries count on a specific historical date (YYYY-MM-DD)."""
    result = get_flock_count_on_date(farm_id=farm_id, species=species, target_date=target_date)
    return {
        "status": "success",
        "farm_id": farm_id,
        "data": result
    }


# -------------------------------------------------------------------
# Persistent Farm Memory & Clinical Observation Endpoints
# -------------------------------------------------------------------

@app.get("/api/farms/{farm_id}/memories", tags=["Farm Memory"])
def get_farm_memories_endpoint(farm_id: str, status: Optional[str] = None):
    """GET /api/farms/{farm_id}/memories: Retrieve all persistent memories for the active farm."""
    if status == "active":
        memories = get_active_farm_memories(farm_id=farm_id, limit=50)
    else:
        memories = get_all_farm_memories(farm_id=farm_id, limit=50)
    return {
        "status": "success",
        "farm_id": farm_id,
        "count": len(memories),
        "memories": memories
    }


@app.post("/api/farms/{farm_id}/memories", tags=["Farm Memory"])
def create_farm_memory_endpoint(farm_id: str, payload: FarmMemoryRequest):
    """POST /api/farms/{farm_id}/memories: Manually record and embed a new clinical observation."""
    mem = farm_memory.log_and_embed_observation(
        farm_id=farm_id,
        species=payload.species,
        category=payload.category,
        observation=payload.observation,
        source="manual_ui"
    )
    return {
        "status": "success",
        "farm_id": farm_id,
        "memory": mem
    }


@app.put("/api/farms/{farm_id}/memories/{memory_id}/resolve", tags=["Farm Memory"])
def resolve_farm_memory_endpoint(farm_id: str, memory_id: int):
    """PUT /api/farms/{farm_id}/memories/{memory_id}/resolve: Mark observation as resolved."""
    ok = resolve_farm_memory(memory_id=memory_id, farm_id=farm_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory record not found or already resolved")
    return {"status": "success", "message": f"Memory {memory_id} marked as resolved"}


@app.delete("/api/farms/{farm_id}/memories/{memory_id}", tags=["Farm Memory"])
def delete_farm_memory_endpoint(farm_id: str, memory_id: int):
    """DELETE /api/farms/{farm_id}/memories/{memory_id}: Permanently remove memory."""
    ok = delete_farm_memory(memory_id=memory_id, farm_id=farm_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory record not found")
    return {"status": "success", "message": f"Memory {memory_id} deleted"}


# -------------------------------------------------------------------
# Operational Data Endpoints (Expenditures, Health Logs, Telemetry)
# -------------------------------------------------------------------

@app.get("/api/farms/{farm_id}/expenditures", tags=["Expenditures"])
def get_farm_expenditures_endpoint(farm_id: str):
    """GET /api/farms/{farm_id}/expenditures: Returns recorded expenditures for the farm."""
    records = get_all_expenditures(farm_id=farm_id)
    total_amount = sum(r.get("amount", 0) for r in records)
    return {
        "status": "success",
        "farm_id": farm_id,
        "count": len(records),
        "total_amount": total_amount,
        "expenditures": records
    }


@app.post("/api/farms/{farm_id}/expenditures", tags=["Expenditures"])
def create_farm_expenditure_endpoint(farm_id: str, payload: ExpenditureRequest):
    """POST /api/farms/{farm_id}/expenditures: Log a new expenditure."""
    rec = record_expenditure(
        farm_id=farm_id,
        category=payload.category,
        amount=payload.amount,
        description=payload.description or ""
    )
    return {
        "status": "success",
        "farm_id": farm_id,
        "expenditure": rec
    }


@app.get("/api/farms/{farm_id}/health-logs", tags=["Health Records"])
def get_farm_health_logs_endpoint(farm_id: str):
    """GET /api/farms/{farm_id}/health-logs: Returns animal medical & health check records."""
    logs = get_all_health_logs(farm_id=farm_id)
    return {
        "status": "success",
        "farm_id": farm_id,
        "count": len(logs),
        "health_logs": logs
    }


@app.get("/api/farms/{farm_id}/telemetry", tags=["IoT Telemetry"])
def get_farm_telemetry_endpoint(farm_id: str):
    """GET /api/farms/{farm_id}/telemetry: Returns recent IoT sensor readings."""
    data = get_telemetry_data(farm_id=farm_id, limit=50)
    return {
        "status": "success",
        "farm_id": farm_id,
        "count": len(data),
        "telemetry": data
    }


# -------------------------------------------------------------------
def normalize_language(lang: Optional[str]) -> str:
    if not lang:
        return "english"
    l = str(lang).strip().lower()
    if l in ["ha", "hausa"] or l.startswith("ha-") or "hausa" in l:
        return "hausa"
    if l in ["pg", "pidgin", "pcm"] or l.startswith("pid") or "pidgin" in l:
        return "pidgin"
    return "english"


@app.get("/", include_in_schema=False)
def read_root():
    """Serve single-page HTML/JS frontend at http://127.0.0.1:8000/."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
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
    import time
    req_start = time.time()
    thread_id = payload.thread_id
    user_input = payload.user_input.strip()
    farm_id = payload.farm_id or "default_farm"
    language = normalize_language(payload.language)

    logger.info(f"[API /chat] Incoming Request | thread_id='{thread_id}' | farm_id='{farm_id}' | language='{language}' | prompt='{user_input}'")

    if not user_input:
        logger.warning(f"[API /chat] Empty user_input received for thread '{thread_id}'")
        raise HTTPException(status_code=400, detail="user_input cannot be empty.")

    # 1. Fetch thread history from DB
    history_rows = get_thread_messages(thread_id)
    messages: List[Dict[str, str]] = [{"role": r["role"], "content": r["content"]} for r in history_rows]
    logger.info(f"[API /chat] Loaded {len(messages)} prior messages from database for thread '{thread_id}'")

    # 2. Append new user prompt
    messages.append({"role": "user", "content": user_input})
    add_chat_message(thread_id, "user", user_input)

    # 3. Process LLM completion with farm_id context
    try:
        assistant_response = chat_completion(messages, farm_id=farm_id, thread_id=thread_id, language=language)
    except Exception as e:
        logger.error(f"[API /chat] Exception in chat_completion for thread '{thread_id}': {e}", exc_info=True)
        assistant_response = f"Internal system error: {str(e)}"

    # 4. Save assistant response to DB
    add_chat_message(thread_id, "assistant", assistant_response)
    total_req_time = time.time() - req_start
    logger.info(f"[API /chat] Completed successfully in {total_req_time:.2f}s | Response length: {len(assistant_response)} chars")

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
