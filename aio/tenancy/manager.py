"""
AIO v3 – Feature 9: Multi-tenancy
API keys par tenant · quotas · rate limiting · isolation SQLite
"""
from __future__ import annotations
import hashlib, secrets, sqlite3, time
from typing import Dict, Any, Optional


class TenantManager:
    """Gestion complète multi-tenant avec API keys, quotas et isolation."""

    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tenants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                plan TEXT DEFAULT 'starter',
                quota_req_month INTEGER DEFAULT 10000,
                quota_tokens_month INTEGER DEFAULT 1000000,
                created_at REAL NOT NULL,
                active INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0,
                period TEXT NOT NULL,
                UNIQUE(tenant_id, period)
            );
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                event TEXT NOT NULL,
                detail TEXT,
                ts REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_tenant ON usage(tenant_id, period);
        """)
        self._conn.commit()

    # ── Gestion tenants ───────────────────────────────────────────────────────
    def create_tenant(self, name: str, plan: str = "starter") -> Dict[str, Any]:
        """Crée un tenant et retourne sa clé API en clair (une seule fois)."""
        api_key = "aio-" + secrets.token_urlsafe(32)
        tenant_id = "t_" + hashlib.md5(name.encode()).hexdigest()[:12]
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        quotas = {"starter": (10_000, 1_000_000), "pro": (100_000, 10_000_000), "enterprise": (10_000_000, 1_000_000_000)}
        q_req, q_tok = quotas.get(plan, quotas["starter"])
        self._conn.execute(
            "INSERT OR IGNORE INTO tenants (id, name, api_key_hash, plan, quota_req_month, quota_tokens_month, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (tenant_id, name, api_key_hash, plan, q_req, q_tok, time.time())
        )
        self._conn.commit()
        self._audit(tenant_id, "created", f"plan={plan}")
        return {"tenant_id": tenant_id, "name": name, "plan": plan, "api_key": api_key}

    def authenticate(self, api_key: str) -> Optional[str]:
        """Retourne tenant_id si clé valide et tenant actif, sinon None."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT id FROM tenants WHERE api_key_hash=? AND active=1", (key_hash,)
        ).fetchone()
        return row[0] if row else None

    # ── Quotas et usage ───────────────────────────────────────────────────────
    def _period(self) -> str:
        t = time.gmtime()
        return f"{t.tm_year}-{t.tm_mon:02d}"

    def check_quota(self, tenant_id: str) -> Dict[str, Any]:
        """Vérifie si le tenant peut encore faire des requêtes ce mois."""
        tenant = self._conn.execute(
            "SELECT quota_req_month, quota_tokens_month, plan FROM tenants WHERE id=?", (tenant_id,)
        ).fetchone()
        if not tenant:
            return {"allowed": False, "reason": "unknown_tenant"}
        q_req, q_tok, plan = tenant
        period = self._period()
        usage = self._conn.execute(
            "SELECT requests, tokens_in+tokens_out FROM usage WHERE tenant_id=? AND period=?",
            (tenant_id, period)
        ).fetchone()
        used_req = usage[0] if usage else 0
        used_tok = usage[1] if usage else 0
        if used_req >= q_req:
            return {"allowed": False, "reason": "quota_requests_exceeded",
                    "used": used_req, "limit": q_req}
        if used_tok >= q_tok:
            return {"allowed": False, "reason": "quota_tokens_exceeded",
                    "used": used_tok, "limit": q_tok}
        return {"allowed": True, "used_requests": used_req, "limit_requests": q_req,
                "used_tokens": used_tok, "limit_tokens": q_tok, "plan": plan}

    def record_usage(self, tenant_id: str, requests: int = 1, tokens_in: int = 0, tokens_out: int = 0):
        period = self._period()
        self._conn.execute(
            "INSERT INTO usage (tenant_id, period, requests, tokens_in, tokens_out) VALUES (?,?,?,?,?) "
            "ON CONFLICT(tenant_id, period) DO UPDATE SET "
            "requests=requests+excluded.requests, "
            "tokens_in=tokens_in+excluded.tokens_in, "
            "tokens_out=tokens_out+excluded.tokens_out",
            (tenant_id, period, requests, tokens_in, tokens_out)
        )
        self._conn.commit()

    def get_usage(self, tenant_id: str) -> Dict[str, Any]:
        period = self._period()
        row = self._conn.execute(
            "SELECT requests, tokens_in, tokens_out FROM usage WHERE tenant_id=? AND period=?",
            (tenant_id, period)
        ).fetchone()
        return {
            "tenant_id": tenant_id, "period": period,
            "requests": row[0] if row else 0,
            "tokens_in": row[1] if row else 0,
            "tokens_out": row[2] if row else 0,
        }

    def list_tenants(self) -> list:
        rows = self._conn.execute(
            "SELECT id, name, plan, active, created_at FROM tenants ORDER BY created_at DESC"
        ).fetchall()
        return [{"id": r[0], "name": r[1], "plan": r[2], "active": bool(r[3])} for r in rows]

    def deactivate_tenant(self, tenant_id: str):
        self._conn.execute("UPDATE tenants SET active=0 WHERE id=?", (tenant_id,))
        self._conn.commit()
        self._audit(tenant_id, "deactivated", "")

    def _audit(self, tenant_id: str, event: str, detail: str = ""):
        self._conn.execute(
            "INSERT INTO audit_log (tenant_id, event, detail, ts) VALUES (?,?,?,?)",
            (tenant_id, event, detail, time.time())
        )
        self._conn.commit()

    def get_audit_log(self, tenant_id: str, limit: int = 50) -> list:
        rows = self._conn.execute(
            "SELECT event, detail, ts FROM audit_log WHERE tenant_id=? ORDER BY ts DESC LIMIT ?",
            (tenant_id, limit)
        ).fetchall()
        return [{"event": r[0], "detail": r[1], "ts": r[2]} for r in rows]
