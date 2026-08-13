import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Database file path setup
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "data"
DB_PATH = DB_DIR / "farm_local.db"


def ensure_dirs():
    """Ensure data directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Establishes and returns a SQLite database connection with row_factory set
    to sqlite3.Row for dictionary-like access.
    """
    ensure_dirs()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH):
    """
    Initialize SQLite database tables for farms, farm records, telemetry, document chunks,
    and relational chat session management (chat_threads & chat_messages).
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Farms Table (Multi-farm profile management)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farms (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                farm_type TEXT NOT NULL DEFAULT 'General',
                description TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Chat Threads Table (Linked to farm_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL DEFAULT 'default_farm',
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (farm_id) REFERENCES farms(id) ON DELETE CASCADE
            )
        """)

        # Chat Messages Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
            )
        """)

        # Expenditures Table (Linked to farm_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expenditures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL DEFAULT 'default_farm',
                category TEXT NOT NULL,
                amount REAL NOT NULL,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Health Logs Table (Linked to farm_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL DEFAULT 'default_farm',
                animal_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                notes TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Telemetry Data Table (Linked to farm_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL DEFAULT 'default_farm',
                node_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Animals Table (Linked to farm_id)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS animals (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL DEFAULT 'default_farm',
                name TEXT NOT NULL,
                species TEXT NOT NULL,
                breed TEXT,
                status TEXT DEFAULT 'Active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Document Chunks Table (RAG text passage & metadata storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        # Auto-migrate missing columns for existing databases
        for tbl in ["chat_threads", "expenditures", "health_logs", "telemetry_data", "animals"]:
            try:
                cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN farm_id TEXT DEFAULT 'default_farm'")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Seed Default Farm if no farms exist
        cursor.execute("SELECT COUNT(*) as cnt FROM farms")
        if cursor.fetchone()["cnt"] == 0:
            cursor.execute(
                "INSERT INTO farms (id, name, farm_type, description) VALUES (?, ?, ?, ?)",
                ("default_farm", "My Main Farm", "General", "Default general farm profile")
            )


# -------------------------------------------------------------------
# Multi-Farm Management Functions
# -------------------------------------------------------------------

def create_farm(name: str, farm_type: str = "General", description: str = "", db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Create a new farm profile."""
    farm_id = f"farm_{uuid.uuid4().hex[:8]}"
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO farms (id, name, farm_type, description) VALUES (?, ?, ?, ?)",
            (farm_id, name, farm_type, description)
        )
    return {"id": farm_id, "name": name, "farm_type": farm_type, "description": description}


def get_farms(db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all farm profiles."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farms ORDER BY created_at ASC")
        return [dict(r) for r in cursor.fetchall()]


def get_farm_by_id(farm_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a farm profile by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM farms WHERE id = ?", (farm_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_farm(farm_id: str, name: str, farm_type: str, description: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Update an existing farm profile."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE farms SET name = ?, farm_type = ?, description = ? WHERE id = ?",
            (name, farm_type, description, farm_id)
        )
        if cursor.rowcount > 0:
            return {"id": farm_id, "name": name, "farm_type": farm_type, "description": description}
    return None


def delete_farm(farm_id: str, db_path: Path = DB_PATH) -> bool:
    """Delete a farm profile and associated threads."""
    if farm_id == "default_farm":
        return False  # Prevent deleting default farm
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM farms WHERE id = ?", (farm_id,))
        return cursor.rowcount > 0


# -------------------------------------------------------------------
# Relational Chat Thread Functions
# -------------------------------------------------------------------

def create_chat_thread(title: str = "New Chat", farm_id: str = "default_farm", db_path: Path = DB_PATH) -> str:
    thread_id = str(uuid.uuid4())
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_threads (id, farm_id, title) VALUES (?, ?, ?)",
            (thread_id, farm_id, title)
        )
    return thread_id


def get_chat_threads(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM chat_threads WHERE farm_id = ? ORDER BY updated_at DESC",
            (farm_id,)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_thread_by_id(thread_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chat_threads WHERE id = ?", (thread_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_thread_title(thread_id: str, title: str, db_path: Path = DB_PATH):
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_threads SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (title, thread_id)
        )


def delete_chat_thread(thread_id: str, db_path: Path = DB_PATH):
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_threads WHERE id = ?", (thread_id,))


def add_chat_message(thread_id: str, role: str, content: str, db_path: Path = DB_PATH) -> int:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content)
        )
        msg_id = cursor.lastrowid
        cursor.execute("UPDATE chat_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (thread_id,))
        return msg_id


def get_thread_messages(thread_id: str, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, thread_id, role, content, created_at FROM chat_messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,)
        )
        return [dict(r) for r in cursor.fetchall()]


# -------------------------------------------------------------------
# Operational Record Helper Functions
# -------------------------------------------------------------------

def get_all_expenditures(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM expenditures WHERE farm_id = ? ORDER BY timestamp DESC", (farm_id,))
        return [dict(r) for r in cursor.fetchall()]


def get_all_health_logs(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM health_logs WHERE farm_id = ? ORDER BY timestamp DESC", (farm_id,))
        return [dict(r) for r in cursor.fetchall()]


def get_all_animals(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM animals WHERE farm_id = ? ORDER BY created_at DESC", (farm_id,))
        return [dict(r) for r in cursor.fetchall()]


def add_animal_record(animal_id: str, name: str, species: str, breed: str = "", status: str = "Active", farm_id: str = "default_farm", db_path: Path = DB_PATH) -> str:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO animals (id, farm_id, name, species, breed, status) VALUES (?, ?, ?, ?, ?, ?)",
            (animal_id, farm_id, name, species, breed, status)
        )
        return animal_id


def get_system_context_summary(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> str:
    """
    Retrieves a live summary of the active farm profile, species constraints, custom description,
    and database records to dynamically ground the LLM system prompt.
    """
    try:
        farm = get_farm_by_id(farm_id, db_path)
        farm_name = farm["name"] if farm else "My Main Farm"
        farm_type = farm["farm_type"] if farm else "General"
        farm_desc = farm["description"] if farm and farm.get("description") else "No custom description provided."

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, name, species FROM animals WHERE farm_id = ? LIMIT 10", (farm_id,))
            animals = [f"{r['id']} ({r['species']} - {r['name']})" for r in cursor.fetchall()]
            animals_str = ", ".join(animals) if animals else "NONE (0 animals currently registered)"
            
            cursor.execute("SELECT COUNT(*) as cnt, SUM(amount) as total FROM expenditures WHERE farm_id = ?", (farm_id,))
            exp_row = cursor.fetchone()
            exp_cnt = exp_row['cnt'] if exp_row else 0
            exp_total = exp_row['total'] or 0.0

        return (
            f"ACTIVE FARM PROFILE (READ-ONLY TRUTH FROM farm_local.db):\n"
            f"- Farm Name: {farm_name}\n"
            f"- Target Species Scope: {farm_type}\n"
            f"- Farmer's Custom Details: \"{farm_desc}\"\n"
            f"- Registered Animals: {animals_str}\n"
            f"- Total Recorded Expenditures: {exp_cnt} records (Total: NGN {exp_total:,.2f})\n"
        )
    except Exception as e:
        return "ACTIVE FARM PROFILE: General Farm (0 registered animals)."


if __name__ == "__main__":
    init_db()
    print(f"Database successfully initialized with multi-farm tables at {DB_PATH}")
