"""
AIO v3 – Feature 7: Mémoire persistante par agent (SQLite)
Chaque agent stocke et relit son contexte entre sessions.
"""
from __future__ import annotations
import json, sqlite3, time, hashlib
from typing import List, Dict, Any, Optional


class AgentMemory:
    """
    Mémoire persistante SQLite par agent.
    Stocke messages, faits et résumés de session.
    """

    def __init__(self, db_path: str = ":memory:", max_messages: int = 100):
        self.db_path = db_path
        self.max_messages = max_messages
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                UNIQUE(agent_id, key)
            );
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_msg_agent ON messages(agent_id, session_id);
            CREATE INDEX IF NOT EXISTS idx_facts_agent ON facts(agent_id);
        """)
        self._conn.commit()

    # ── Messages ──────────────────────────────────────────────────────────────
    def add_message(self, agent_id: str, session_id: str, role: str, content: str):
        self._conn.execute(
            "INSERT INTO messages (agent_id, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (agent_id, session_id, role, content, time.time())
        )
        # Trim LRU
        count = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE agent_id=?", (agent_id,)
        ).fetchone()[0]
        if count > self.max_messages:
            self._conn.execute(
                "DELETE FROM messages WHERE agent_id=? AND id IN "
                "(SELECT id FROM messages WHERE agent_id=? ORDER BY created_at ASC LIMIT ?)",
                (agent_id, agent_id, count - self.max_messages)
            )
        self._conn.commit()

    def get_history(self, agent_id: str, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE agent_id=? AND session_id=? "
            "ORDER BY created_at DESC LIMIT ?",
            (agent_id, session_id, limit)
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    # ── Faits persistants ─────────────────────────────────────────────────────
    def set_fact(self, agent_id: str, key: str, value: Any):
        self._conn.execute(
            "INSERT INTO facts (agent_id, key, value, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(agent_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (agent_id, key, json.dumps(value), time.time())
        )
        self._conn.commit()

    def get_fact(self, agent_id: str, key: str) -> Optional[Any]:
        row = self._conn.execute(
            "SELECT value FROM facts WHERE agent_id=? AND key=?", (agent_id, key)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_all_facts(self, agent_id: str) -> Dict[str, Any]:
        rows = self._conn.execute(
            "SELECT key, value FROM facts WHERE agent_id=?", (agent_id,)
        ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    # ── Résumés de session ────────────────────────────────────────────────────
    def store_summary(self, agent_id: str, session_id: str, summary: str):
        self._conn.execute(
            "INSERT INTO summaries (agent_id, session_id, summary, created_at) VALUES (?,?,?,?)",
            (agent_id, session_id, summary, time.time())
        )
        self._conn.commit()

    def get_last_summary(self, agent_id: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT summary FROM summaries WHERE agent_id=? ORDER BY created_at DESC LIMIT 1",
            (agent_id,)
        ).fetchone()
        return row[0] if row else None

    def stats(self, agent_id: str) -> Dict[str, int]:
        return {
            "messages": self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE agent_id=?", (agent_id,)
            ).fetchone()[0],
            "facts": self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE agent_id=?", (agent_id,)
            ).fetchone()[0],
            "summaries": self._conn.execute(
                "SELECT COUNT(*) FROM summaries WHERE agent_id=?", (agent_id,)
            ).fetchone()[0],
        }
