"""Outils statistiques sans dépendance lourde (pas de SciPy).

- ``chi2_sf`` : fonction de survie de la loi du χ² (p-value), via la fonction
  gamma incomplète régularisée (séries + fraction continue de Lentz).
- ``historical_var`` / ``historical_cvar`` : VaR et Expected Shortfall historiques.
- ``max_drawdown`` : drawdown maximal d'une courbe de valeur.
"""
from __future__ import annotations

import math

import numpy as np

_EPS = 1e-15
_TINY = 1e-300
_MAX_ITER = 10_000


def _gamma_p_series(a: float, x: float) -> float:
    """Gamma incomplète inférieure régularisée P(a, x) par développement en série."""
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_MAX_ITER):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_continued_fraction(a: float, x: float) -> float:
    """Gamma incomplète supérieure régularisée Q(a, x) par fraction continue (Lentz)."""
    b = x + 1.0 - a
    c = 1.0 / _TINY
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _TINY:
            d = _TINY
        c = b + an / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(statistic: float, df: int) -> float:
    """P(X >= statistic) pour X ~ χ²(df)."""
    if df <= 0:
        raise ValueError("df doit être strictement positif")
    if statistic <= 0:
        return 1.0
    a = df / 2.0
    x = statistic / 2.0
    if x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(a, x)))
    return max(0.0, min(1.0, _gamma_q_continued_fraction(a, x)))


def portfolio_returns(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rendements d'un portefeuille à poids constants (le reste est en cash à 0 %)."""
    returns = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if returns.ndim != 2 or returns.shape[1] != weights.shape[0]:
        raise ValueError("returns doit être de forme (T, N) et weights de forme (N,)")
    return returns @ weights


def historical_var(pnl: np.ndarray, confidence: float = 0.99) -> float:
    """VaR historique (perte positive) au niveau ``confidence``."""
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    return float(max(0.0, np.quantile(-pnl, confidence)))


def historical_cvar(pnl: np.ndarray, confidence: float = 0.99) -> float:
    """Expected Shortfall historique : perte moyenne au-delà de la VaR."""
    pnl = np.asarray(pnl, dtype=float)
    if pnl.size == 0:
        return 0.0
    losses = -pnl
    threshold = np.quantile(losses, confidence)
    tail = losses[losses >= threshold]
    return float(max(0.0, tail.mean())) if tail.size else 0.0


def max_drawdown(values: np.ndarray) -> float:
    """Drawdown maximal (fraction positive) d'une série de valeurs de portefeuille."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))
