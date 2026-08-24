#!/usr/bin/env python3
"""One-time migration script to fix document_chunks table schema."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "farm_local.db"


def migrate():
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Add farm_id column
    try:
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN farm_id TEXT")
        print("Added farm_id column")
    except sqlite3.OperationalError as e:
        print(f"farm_id: {e}")

    # Add created_at column (without default for migration)
    try:
        cursor.execute("ALTER TABLE document_chunks ADD COLUMN created_at TEXT")
        print("Added created_at column")
    except sqlite3.OperationalError as e:
        print(f"created_at: {e}")

    # Create index
    try:
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_document_chunks_farm_id ON document_chunks(farm_id)"
        )
        print("Created index")
    except sqlite3.OperationalError as e:
        print(f"index: {e}")

    conn.commit()

    # Verify
    cursor.execute("PRAGMA table_info(document_chunks)")
    cols = [row[1] for row in cursor.fetchall()]
    print(f"Columns: {cols}")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
