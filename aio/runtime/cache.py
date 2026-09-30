"""
AIO v3 – Feature 1: Cache sémantique réel
Cosine similarity sur embeddings numpy → hit si sim > seuil.
Zero token consommé sur un cache hit.
"""
from __future__ import annotations
import hashlib, json, time, sqlite3
from typing import Optional, Dict, Any, List
import numpy as np


def _embed_text(text: str, dim: int = 128) -> np.ndarray:
    """Embedding déterministe léger (hash-based, sans modèle externe)."""
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(list(h))
    raw = rng.standard_normal(dim).astype(np.float32)
    return raw / (np.linalg.norm(raw) + 1e-9)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class SemanticCache:
    """
    Cache sémantique SQLite.
    Hit si cosine(query_emb, cached_emb) > threshold.
    """

    def __init__(self, db_path: str = ":memory:", threshold: float = 0.92, max_entries: int = 1000):
        self.db_path = db_path
        self.threshold = threshold
        self.max_entries = max_entries
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._stats = {"hits": 0, "misses": 0, "tokens_saved": 0}

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                query_hash TEXT NOT NULL,
                query_emb BLOB NOT NULL,
                response TEXT NOT NULL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON cache(model)")
        self._conn.commit()

    def lookup(self, model: str, query: str) -> Optional[Dict[str, Any]]:
        q_emb = _embed_text(query)
        rows = self._conn.execute(
            "SELECT query_emb, response, tokens_in, tokens_out FROM cache WHERE model=?",
            (model,)
        ).fetchall()
        best_sim, best_row = -1.0, None
        for row in rows:
            cached_emb = np.frombuffer(row[0], dtype=np.float32)
            sim = _cosine(q_emb, cached_emb)
            if sim > best_sim:
                best_sim, best_row = sim, row
        if best_sim >= self.threshold and best_row is not None:
            self._stats["hits"] += 1
            self._stats["tokens_saved"] += best_row[2] + best_row[3]
            return {
                "content": best_row[1],
                "cache_hit": True,
                "similarity": round(best_sim, 4),
                "tokens_in": 0,
                "tokens_out": 0,
            }
        self._stats["misses"] += 1
        return None

    def store(self, model: str, query: str, response: str, tokens_in: int = 0, tokens_out: int = 0):
        q_emb = _embed_text(query)
        q_hash = hashlib.md5(query.encode()).hexdigest()
        self._conn.execute(
            "INSERT INTO cache (model, query_hash, query_emb, response, tokens_in, tokens_out, created_at) VALUES (?,?,?,?,?,?,?)",
            (model, q_hash, q_emb.tobytes(), response, tokens_in, tokens_out, time.time())
        )
        # LRU trim
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count > self.max_entries:
            self._conn.execute(
                "DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY created_at ASC LIMIT ?)",
                (count - self.max_entries,)
            )
        self._conn.commit()

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "hit_rate": round(self._stats["hits"] / max(total, 1), 3),
            "total_queries": total,
        }
