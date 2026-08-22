import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Database file path setup
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "farm_local.db"


def ensure_dirs():
    """Ensure database parent directory exists."""
    pass


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
                farm_id TEXT,  -- NULL = global/system, otherwise farm-specific upload
                chunk_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add index for farm_id filtering (run once)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_farm_id ON document_chunks(farm_id)")
        except Exception:
            pass

        # Flock Ledger Table (Append-only time-series animal count ledger)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flock_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL,
                species TEXT NOT NULL,
                count_change INTEGER NOT NULL,
                new_total INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(farm_id) REFERENCES farms(id) ON DELETE CASCADE
            )
        """)

        # Flock Ledger Anomalies Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ledger_anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL,
                severity TEXT NOT NULL, -- 'NORMAL', 'INFO', 'WARNING', 'CRITICAL'
                title TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                report_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(farm_id) REFERENCES farms(id) ON DELETE CASCADE
            )
        """)

        # Persistent Farm Memories & Clinical Observations Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS farm_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                farm_id TEXT NOT NULL,
                species TEXT NOT NULL,
                category TEXT NOT NULL,
                observation TEXT NOT NULL,
                embedding_json TEXT DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'chat',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP DEFAULT NULL,
                FOREIGN KEY(farm_id) REFERENCES farms(id) ON DELETE CASCADE
            )
        """)

        # Auto-migrate missing columns for existing databases
        for tbl in ["chat_threads", "expenditures", "health_logs", "telemetry_data", "animals", "flock_ledger", "ledger_anomalies", "farm_memories"]:
            try:
                cursor.execute(f"ALTER TABLE {tbl} ADD COLUMN farm_id TEXT DEFAULT 'default_farm'")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Migrate document_chunks table
        try:
            cursor.execute("ALTER TABLE document_chunks ADD COLUMN farm_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        
        try:
            cursor.execute("ALTER TABLE document_chunks ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Migration complete


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
    """Delete a farm profile and associated records."""
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


def truncate_thread_messages_from(thread_id: str, from_message_id: int, db_path: Path = DB_PATH) -> int:
    """Deletes all messages in a thread with id >= from_message_id (inclusive) to reset history upon edit."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM chat_messages WHERE thread_id = ? AND id >= ?",
            (thread_id, from_message_id)
        )
        deleted_count = cursor.rowcount
        cursor.execute("UPDATE chat_threads SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (thread_id,))
        return deleted_count


def truncate_thread_messages_by_index(thread_id: str, from_index: int, db_path: Path = DB_PATH) -> int:
    """Deletes all messages in a thread starting from 0-based from_index onwards."""
    messages = get_thread_messages(thread_id, db_path=db_path)
    if 0 <= from_index < len(messages):
        target_msg_id = messages[from_index]["id"]
        return truncate_thread_messages_from(thread_id, target_msg_id, db_path=db_path)
    return 0


# -------------------------------------------------------------------
# Operational Record Helper Functions
# -------------------------------------------------------------------

def get_all_expenditures(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM expenditures WHERE farm_id = ? ORDER BY timestamp DESC", (farm_id,))
        return [dict(r) for r in cursor.fetchall()]


def record_expenditure(
    farm_id: str,
    category: str,
    amount: float,
    description: str = "",
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """Records a new operational or financial expenditure."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO expenditures (farm_id, category, amount, description) VALUES (?, ?, ?, ?)",
            (farm_id, category.strip().lower(), float(amount), description.strip())
        )
        exp_id = cursor.lastrowid
        cursor.execute("SELECT * FROM expenditures WHERE id = ?", (exp_id,))
        return dict(cursor.fetchone())


def get_telemetry_data(
    farm_id: str = "default_farm",
    limit: int = 50,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves recent IoT sensor readings for a farm."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM telemetry_data WHERE farm_id = ? ORDER BY timestamp DESC LIMIT ?",
            (farm_id, limit)
        )
        return [dict(r) for r in cursor.fetchall()]


def get_all_health_logs(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM health_logs WHERE farm_id = ? ORDER BY timestamp DESC", (farm_id,))
        return [dict(r) for r in cursor.fetchall()]


def record_health_log(farm_id: str, animal_id: str, event_type: str, notes: str = "", db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Records a health check, vaccination, or disease log into health_logs."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO health_logs (farm_id, animal_id, event_type, notes) VALUES (?, ?, ?, ?)",
            (farm_id, animal_id, event_type, notes)
        )
        log_id = cursor.lastrowid
        cursor.execute("SELECT * FROM health_logs WHERE id = ?", (log_id,))
        return dict(cursor.fetchone())


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


# -------------------------------------------------------------------
# Flock & Herd Count Ledger Functions
# -------------------------------------------------------------------

def normalize_species_name(species: str) -> str:
    """Normalize species colloquialisms to canonical species names."""
    s = str(species).lower().strip()
    if s in ["chicken", "chickens", "hen", "hens", "broiler", "broilers", "layer", "layers", "poultry", "bird", "birds", "kaza", "kaji"]:
        return "Poultry"
    if s in ["goat", "goats", "buck", "doe", "kid", "kids", "akuya", "awaki"]:
        return "Goat"
    if s in ["sheep", "ram", "rams", "ewe", "ewes", "lamb", "lambs", "tinkiya", "tumaki"]:
        return "Sheep"
    if s in ["cow", "cows", "cattle", "bull", "bulls", "calf", "calves", "saniya", "shanu"]:
        return "Cattle"
    if s in ["pig", "pigs", "swine", "alhanzir"]:
        return "Pig"
    return s.capitalize() if s else "Poultry"


def get_current_flock_totals(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> Dict[str, int]:
    """Returns the latest count total for each species on the active farm."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT species, new_total
            FROM flock_ledger
            WHERE farm_id = ? AND id IN (
                SELECT MAX(id) FROM flock_ledger WHERE farm_id = ? GROUP BY species
            )
        """, (farm_id, farm_id))
        return {r["species"]: r["new_total"] for r in cursor.fetchall()}


def record_flock_event(
    farm_id: str = "default_farm",
    species: str = "Poultry",
    count_change: int = 0,
    event_type: str = "initial_count",
    notes: str = "",
    exact_total: Optional[int] = None,
    created_at: Optional[str] = None,
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Appends a new count event to the flock ledger and calculates the running total.
    """
    norm_species = normalize_species_name(species)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        # Find current latest total
        cursor.execute(
            "SELECT new_total FROM flock_ledger WHERE farm_id = ? AND species = ? ORDER BY id DESC LIMIT 1",
            (farm_id, norm_species)
        )
        row = cursor.fetchone()
        previous_total = row["new_total"] if row else 0

        if exact_total is not None:
            new_total = max(0, exact_total)
            count_change = new_total - previous_total
        else:
            new_total = max(0, previous_total + count_change)

        if created_at:
            cursor.execute(
                "INSERT INTO flock_ledger (farm_id, species, count_change, new_total, event_type, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (farm_id, norm_species, count_change, new_total, event_type, notes, created_at)
            )
        else:
            cursor.execute(
                "INSERT INTO flock_ledger (farm_id, species, count_change, new_total, event_type, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (farm_id, norm_species, count_change, new_total, event_type, notes)
            )
        ledger_id = cursor.lastrowid
        cursor.execute("SELECT * FROM flock_ledger WHERE id = ?", (ledger_id,))
        return dict(cursor.fetchone())


def get_flock_count_on_date(
    farm_id: str = "default_farm",
    species: Optional[str] = None,
    target_date: Optional[str] = None,
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """
    Retrieves the flock count balance as of a specific date (YYYY-MM-DD).
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if species:
            norm_species = normalize_species_name(species)
            if target_date:
                cursor.execute("""
                    SELECT species, new_total, created_at
                    FROM flock_ledger
                    WHERE farm_id = ? AND species = ? AND date(created_at) <= date(?)
                    ORDER BY id DESC LIMIT 1
                """, (farm_id, norm_species, target_date))
            else:
                cursor.execute("""
                    SELECT species, new_total, created_at
                    FROM flock_ledger
                    WHERE farm_id = ? AND species = ?
                    ORDER BY id DESC LIMIT 1
                """, (farm_id, norm_species))
            row = cursor.fetchone()
            return {
                "species": norm_species,
                "count": row["new_total"] if row else 0,
                "as_of_date": target_date or "current",
                "last_record_date": row["created_at"] if row else None
            }
        else:
            # All species
            if target_date:
                cursor.execute("""
                    SELECT species, new_total, created_at
                    FROM flock_ledger
                    WHERE farm_id = ? AND date(created_at) <= date(?) AND id IN (
                        SELECT MAX(id) FROM flock_ledger WHERE farm_id = ? AND date(created_at) <= date(?) GROUP BY species
                    )
                """, (farm_id, target_date, farm_id, target_date))
            else:
                cursor.execute("""
                    SELECT species, new_total, created_at
                    FROM flock_ledger
                    WHERE farm_id = ? AND id IN (
                        SELECT MAX(id) FROM flock_ledger WHERE farm_id = ? GROUP BY species
                    )
                """, (farm_id, farm_id))
            rows = cursor.fetchall()
            totals = {r["species"]: r["new_total"] for r in rows}
            return {
                "species_counts": totals,
                "total_flock_size": sum(totals.values()),
                "as_of_date": target_date or "current"
            }


def get_flock_ledger_history(
    farm_id: str = "default_farm",
    species: Optional[str] = None,
    limit: int = 50,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Returns the time-series audit log of flock count changes."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if species:
            norm_species = normalize_species_name(species)
            cursor.execute(
                "SELECT * FROM flock_ledger WHERE farm_id = ? AND species = ? ORDER BY id DESC LIMIT ?",
                (farm_id, norm_species, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM flock_ledger WHERE farm_id = ? ORDER BY id DESC LIMIT ?",
                (farm_id, limit)
            )
        return [dict(r) for r in cursor.fetchall()]


# -------------------------------------------------------------------
# Ledger Anomaly Management Functions
# -------------------------------------------------------------------

def save_ledger_anomaly(
    farm_id: str,
    severity: str,
    title: str,
    metrics: Dict[str, Any],
    report_text: str,
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """Persists a new flock ledger anomaly detection report."""
    metrics_str = json.dumps(metrics) if isinstance(metrics, dict) else str(metrics)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ledger_anomalies (farm_id, severity, title, metrics_json, report_text) VALUES (?, ?, ?, ?, ?)",
            (farm_id, severity.upper(), title, metrics_str, report_text)
        )
        anomaly_id = cursor.lastrowid
        cursor.execute("SELECT * FROM ledger_anomalies WHERE id = ?", (anomaly_id,))
        row = cursor.fetchone()
        res = dict(row)
        try:
            res["metrics"] = json.loads(res["metrics_json"])
        except Exception:
            res["metrics"] = {}
        return res


def get_latest_ledger_anomaly(
    farm_id: str = "default_farm",
    db_path: Path = DB_PATH
) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent anomaly evaluation for a farm."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM ledger_anomalies WHERE farm_id = ? ORDER BY id DESC LIMIT 1",
            (farm_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        try:
            res["metrics"] = json.loads(res["metrics_json"])
        except Exception:
            res["metrics"] = {}
        return res


def get_ledger_anomaly_history(
    farm_id: str = "default_farm",
    limit: int = 20,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves anomaly history logs for a farm."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM ledger_anomalies WHERE farm_id = ? ORDER BY id DESC LIMIT ?",
            (farm_id, limit)
        )
        results = []
        for r in cursor.fetchall():
            item = dict(r)
            try:
                item["metrics"] = json.loads(item["metrics_json"])
            except Exception:
                item["metrics"] = {}
            results.append(item)
        return results


# -------------------------------------------------------------------
# Persistent Farm Memory & Clinical Observation Functions
# -------------------------------------------------------------------

def save_farm_memory(
    farm_id: str,
    species: str,
    category: str,
    observation: str,
    embedding: Optional[List[float]] = None,
    source: str = "chat",
    db_path: Path = DB_PATH
) -> Dict[str, Any]:
    """Persists a new clinical or behavioral observation into farm_memories."""
    emb_str = json.dumps(embedding) if embedding is not None else "[]"
    norm_species = normalize_species_name(species)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO farm_memories (farm_id, species, category, observation, embedding_json, source, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
            (farm_id, norm_species, category.lower().strip(), observation.strip(), emb_str, source)
        )
        mem_id = cursor.lastrowid
        cursor.execute("SELECT * FROM farm_memories WHERE id = ?", (mem_id,))
        row = cursor.fetchone()
        res = dict(row)
        try:
            res["embedding"] = json.loads(res.get("embedding_json", "[]"))
        except Exception:
            res["embedding"] = []
        return res


def get_active_farm_memories(
    farm_id: str = "default_farm",
    species: Optional[str] = None,
    limit: int = 20,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves all active clinical observations for a farm, optionally filtered by species."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if species:
            norm_species = normalize_species_name(species)
            cursor.execute(
                """
                SELECT * FROM farm_memories 
                WHERE farm_id = ? AND status = 'active' AND species = ?
                ORDER BY id DESC LIMIT ?
                """,
                (farm_id, norm_species, limit)
            )
        else:
            cursor.execute(
                """
                SELECT * FROM farm_memories 
                WHERE farm_id = ? AND status = 'active'
                ORDER BY id DESC LIMIT ?
                """,
                (farm_id, limit)
            )
        results = []
        for r in cursor.fetchall():
            item = dict(r)
            try:
                item["embedding"] = json.loads(item.get("embedding_json", "[]"))
            except Exception:
                item["embedding"] = []
            results.append(item)
        return results


def get_all_farm_memories(
    farm_id: str = "default_farm",
    limit: int = 50,
    db_path: Path = DB_PATH
) -> List[Dict[str, Any]]:
    """Retrieves all memories (active and resolved) for audit and history."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM farm_memories WHERE farm_id = ? ORDER BY id DESC LIMIT ?",
            (farm_id, limit)
        )
        results = []
        for r in cursor.fetchall():
            item = dict(r)
            try:
                item["embedding"] = json.loads(item.get("embedding_json", "[]"))
            except Exception:
                item["embedding"] = []
            results.append(item)
        return results


def resolve_farm_memory(
    memory_id: int,
    farm_id: str = "default_farm",
    db_path: Path = DB_PATH
) -> bool:
    """Marks an active clinical observation as resolved/cured."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE farm_memories 
            SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP
            WHERE id = ? AND farm_id = ?
            """,
            (memory_id, farm_id)
        )
        return cursor.rowcount > 0


def delete_farm_memory(
    memory_id: int,
    farm_id: str = "default_farm",
    db_path: Path = DB_PATH
) -> bool:
    """Deletes a memory record permanently."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM farm_memories WHERE id = ? AND farm_id = ?",
            (memory_id, farm_id)
        )
        return cursor.rowcount > 0


def get_system_context_summary(farm_id: str = "default_farm", db_path: Path = DB_PATH) -> str:
    """
    Retrieves a live summary of the active farm profile, species constraints, custom description,
    and flock ledger counts to dynamically ground the LLM system prompt.
    """
    try:
        farm = get_farm_by_id(farm_id, db_path)
        farm_name = farm["name"] if farm else "General Farm"
        farm_type = farm["farm_type"] if farm else "General"
        farm_desc = farm["description"] if farm and farm.get("description") else "No custom description provided."

        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Flock Ledger Totals
            cursor.execute("""
                SELECT species, new_total
                FROM flock_ledger
                WHERE farm_id = ? AND id IN (
                    SELECT MAX(id) FROM flock_ledger WHERE farm_id = ? GROUP BY species
                )
            """, (farm_id, farm_id))
            flock_rows = cursor.fetchall()
            if flock_rows:
                flock_items = [f"{r['species']}: {r['new_total']}" for r in flock_rows]
                flock_str = ", ".join(flock_items)
                total_animals = sum(r['new_total'] for r in flock_rows)
            else:
                flock_str = "0 animals recorded in ledger"
                total_animals = 0
            
            # 2. Recent Ledger Event
            cursor.execute("SELECT species, count_change, new_total, event_type, created_at FROM flock_ledger WHERE farm_id = ? ORDER BY id DESC LIMIT 3", (farm_id,))
            recent_events = cursor.fetchall()
            if recent_events:
                events_str = "; ".join([f"{e['created_at'][:10]}: {e['event_type']} ({e['count_change']:+d} {e['species']} -> Total: {e['new_total']})" for e in recent_events])
            else:
                events_str = "No events logged yet"

            # 3. Expenditures
            cursor.execute("SELECT COUNT(*) as cnt, SUM(amount) as total FROM expenditures WHERE farm_id = ?", (farm_id,))
            exp_row = cursor.fetchone()
            exp_cnt = exp_row['cnt'] if exp_row else 0
            exp_total = exp_row['total'] or 0.0

        return (
            f"ACTIVE FARM PROFILE (READ-ONLY TRUTH FROM farm_local.db):\n"
            f"- Farm Name: {farm_name}\n"
            f"- Target Species Scope: {farm_type}\n"
            f"- Farmer's Custom Profile Notes: \"{farm_desc}\"\n"
            f"- Current Flock Ledger Counts: {flock_str} (Total: {total_animals})\n"
            f"- Recent Flock Ledger Events: {events_str}\n"
            f"- Total Recorded Expenditures: {exp_cnt} records (Total: NGN {exp_total:,.2f})\n"
        )
    except Exception as e:
        return "ACTIVE FARM PROFILE: General Farm (0 recorded flock animals)."


if __name__ == "__main__":
    init_db()
    print(f"Database successfully initialized with multi-farm tables at {DB_PATH}")
