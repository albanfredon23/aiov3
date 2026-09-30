"""
AIO v2 – RobustBenchmarkGate (I7)
Protocole ABBA + bootstrap non-paramétrique sur médiane(Δ).
"""
from __future__ import annotations
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn


class RobustBenchmarkGate:
    """
    Valide qu'un candidat est statistiquement plus rapide que la référence.
    Protocole :
      1. Mesures appariées ABBA (neutralise dérive thermique)
      2. Bootstrap 2000x sur médiane(Δ = t_cand – t_ref)
      3. Admission si CI_upper(α) < required_Δ
    """

    def __init__(self, bootstrap_iters: int = 2000, alpha: float = 0.05, warmup: int = 3):
        self.n_boot = bootstrap_iters
        self.alpha = alpha
        self.warmup = warmup

    # ── Mesure ABBA ──────────────────────────────────────────────────────────
    def _time_module(self, m: nn.Module, x: torch.Tensor, n: int = 10) -> np.ndarray:
        m.eval()
        times = []
        with torch.no_grad():
            for _ in range(self.warmup):        # chauffe cache
                _ = m(x)
            for _ in range(n):
                t0 = time.perf_counter()
                _ = m(x)
                times.append((time.perf_counter() - t0) * 1e3)
        return np.array(times)

    def _abba_deltas(
        self, cand: nn.Module, ref: nn.Module, x: torch.Tensor, pairs: int = 10
    ) -> np.ndarray:
        """Retourne Δ_i = t_cand_i − t_ref_i issus du schéma AB/BA alterné."""
        deltas = []
        for i in range(pairs):
            if i % 2 == 0:          # schéma A→B
                t0 = time.perf_counter(); ref(x); t_r = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter(); cand(x); t_c = (time.perf_counter() - t0) * 1e3
            else:                   # schéma B→A
                t0 = time.perf_counter(); cand(x); t_c = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter(); ref(x); t_r = (time.perf_counter() - t0) * 1e3
            deltas.append(t_c - t_r)
        return np.array(deltas)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    def _bootstrap_ci(self, deltas: np.ndarray) -> Tuple[float, float]:
        boot = np.array([
            np.median(np.random.choice(deltas, size=len(deltas), replace=True))
            for _ in range(self.n_boot)
        ])
        lo, hi = np.percentile(boot, [100 * self.alpha / 2, 100 * (1 - self.alpha / 2)])
        return float(lo), float(hi)

    # ── Validation principale ─────────────────────────────────────────────────
    def validate(
        self,
        cand: nn.Module,
        ref: nn.Module,
        x: torch.Tensor,
        target_speedup: float = 1.05,
    ) -> Dict[str, Any]:
        ref_times = self._time_module(ref, x)
        cand_times = self._time_module(cand, x)
        med_ref = float(np.median(ref_times))
        med_cand = float(np.median(cand_times))

        deltas = self._abba_deltas(cand, ref, x)
        ci_lo, ci_hi = self._bootstrap_ci(deltas)

        # Δ requis : t_cand < t_ref / speedup  ⟹  Δ < t_ref * (1/speedup - 1)
        required_delta = med_ref * (1.0 / target_speedup - 1.0)   # négatif
        is_valid = ci_hi < required_delta

        return {
            "is_valid": is_valid,
            "med_ref_ms": round(med_ref, 3),
            "med_cand_ms": round(med_cand, 3),
            "observed_speedup": round(med_ref / max(med_cand, 1e-6), 4),
            "delta_median_ms": round(float(np.median(deltas)), 3),
            "ci_lower_ms": round(ci_lo, 3),
            "ci_upper_ms": round(ci_hi, 3),
            "required_delta_ms": round(required_delta, 3),
            "target_speedup": target_speedup,
        }
