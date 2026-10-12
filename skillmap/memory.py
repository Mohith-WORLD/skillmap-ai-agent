"""
Persistent SQLite checkpointer integration for LangGraph.
"""

import os
import sqlite3
from typing import Any, List, Optional
from langgraph.checkpoint.sqlite import SqliteSaver


def get_sqlite_saver(db_path: str = "data/checkpoints.db") -> SqliteSaver:
    """
    Initialize and return a persistent SqliteSaver instance.
    
    Ensures directory exists, sets robust SQLite pragmas, and executes
    database setup migrations.
    """
    # Ensure directory exists for database file
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    # Open persistent connection
    conn = sqlite3.connect(db_path, check_same_thread=False)

    # Enable WAL mode for better concurrency and crash resilience
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Instantiate checkpointer and ensure schema is initialized
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def list_saved_threads(db_path: str = "data/checkpoints.db") -> List[str]:
    """Retrieve all unique conversation thread IDs stored in the SQLite checkpoints database."""
    if not os.path.exists(db_path):
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id ASC;")
        threads = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return threads
    except Exception:
        return []


def get_thread_messages(saver: SqliteSaver, thread_id: str) -> List[Any]:
    """Retrieve messages for a thread directly from SQLite checkpointer."""
    try:
        checkpoint_tuple = saver.get_tuple({"configurable": {"thread_id": thread_id}})
        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            return []
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        return channel_values.get("messages", [])
    except Exception:
        return []
