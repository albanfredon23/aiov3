"""
AIO v3 – Feature 12: Rollback automatique des mutations
Surveille N requêtes post-déploiement.
Rollback si taux d'erreur > seuil ou dégradation latence > seuil.
"""
from __future__ import annotations
import time
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class MutationStatus(Enum):
    CANDIDATE = "candidate"
    PROMOTED   = "promoted"
    ROLLED_BACK = "rolled_back"
    STABLE      = "stable"


class MutationRecord:
    def __init__(self, mutation_id: str, name: str):
        self.mutation_id = mutation_id
        self.name = name
        self.status = MutationStatus.CANDIDATE
        self.promoted_at: Optional[float] = None
        self.latencies: List[float] = []
        self.errors: int = 0
        self.requests: int = 0
        self.rollback_reason: Optional[str] = None

    def record(self, latency_ms: float, success: bool):
        self.requests += 1
        self.latencies.append(latency_ms)
        if not success:
            self.errors += 1

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.requests, 1)

    @property
    def median_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[len(s) // 2]


class RollbackManager:
    """
    Surveille les mutations déployées.
    Rollback automatique si dégradation détectée.
    """

    def __init__(
        self,
        error_rate_threshold: float = 0.10,   # 10% d'erreurs → rollback
        latency_degradation: float = 1.30,     # 30% de latence en plus → rollback
        min_requests: int = 10,                # fenêtre minimale d'observation
        on_rollback: Optional[Callable[[str, str], None]] = None,
    ):
        self.error_thr = error_rate_threshold
        self.latency_deg = latency_degradation
        self.min_req = min_requests
        self.on_rollback = on_rollback
        self._mutations: Dict[str, MutationRecord] = {}
        self._baseline_latency: Optional[float] = None
        self._history: List[Dict[str, Any]] = []

    def set_baseline(self, median_latency_ms: float):
        """Définit la latence de référence avant promotion."""
        self._baseline_latency = median_latency_ms

    def promote(self, mutation_id: str, name: str) -> MutationRecord:
        """Marque une mutation comme active en production."""
        rec = MutationRecord(mutation_id, name)
        rec.status = MutationStatus.PROMOTED
        rec.promoted_at = time.time()
        self._mutations[mutation_id] = rec
        self._history.append({
            "event": "promoted", "mutation_id": mutation_id,
            "name": name, "ts": time.time()
        })
        return rec

    def record(self, mutation_id: str, latency_ms: float, success: bool) -> Optional[str]:
        """
        Enregistre une observation et déclenche rollback si nécessaire.
        Retourne la raison du rollback ou None si OK.
        """
        rec = self._mutations.get(mutation_id)
        if not rec or rec.status != MutationStatus.PROMOTED:
            return None
        rec.record(latency_ms, success)
        if rec.requests < self.min_req:
            return None
        # Vérification taux d'erreurs
        if rec.error_rate > self.error_thr:
            reason = f"error_rate={rec.error_rate:.2%} > threshold={self.error_thr:.2%}"
            self._do_rollback(rec, reason)
            return reason
        # Vérification dégradation latence
        if self._baseline_latency and self._baseline_latency > 0:
            if rec.median_latency > self._baseline_latency * self.latency_deg:
                reason = (f"latency={rec.median_latency:.1f}ms > "
                          f"baseline*{self.latency_deg}={self._baseline_latency*self.latency_deg:.1f}ms")
                self._do_rollback(rec, reason)
                return reason
        # Stable
        if rec.requests >= self.min_req * 3:
            rec.status = MutationStatus.STABLE
        return None

    def _do_rollback(self, rec: MutationRecord, reason: str):
        rec.status = MutationStatus.ROLLED_BACK
        rec.rollback_reason = reason
        self._history.append({
            "event": "rollback", "mutation_id": rec.mutation_id,
            "name": rec.name, "reason": reason, "ts": time.time()
        })
        if self.on_rollback:
            self.on_rollback(rec.mutation_id, reason)

    def status(self, mutation_id: str) -> Optional[Dict[str, Any]]:
        rec = self._mutations.get(mutation_id)
        if not rec:
            return None
        return {
            "mutation_id": rec.mutation_id,
            "name": rec.name,
            "status": rec.status.value,
            "requests": rec.requests,
            "error_rate": round(rec.error_rate, 4),
            "median_latency_ms": round(rec.median_latency, 2),
            "rollback_reason": rec.rollback_reason,
        }

    def all_statuses(self) -> List[Dict[str, Any]]:
        return [self.status(mid) for mid in self._mutations]

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)
