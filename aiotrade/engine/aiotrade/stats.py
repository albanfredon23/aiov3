"""Outils statistiques sans dépendance lourde (pas de SciPy).

- ``chi2_sf`` : fonction de survie de la loi du χ² (p-value), via la fonction
  gamma incomplète régularisée (séries + fraction continue de Lentz).
- ``historical_var`` / ``historical_cvar`` : VaR et Expected Shortfall historiques.
- ``chi2_ppf`` : quantile de la loi du χ² (seuil critique), par bissection ;
- ``max_drawdown``, ``sortino_ratio``, ``calmar_ratio`` : indicateurs de performance nets.
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


def chi2_ppf(probability: float, df: int) -> float:
    """Quantile x tel que P(X <= x) = probability pour X ~ χ²(df). Ex. chi2_ppf(0.99, 4) = 13,28."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability doit être dans ]0, 1[")
    target = 1.0 - probability
    lo, hi = 0.0, max(1.0, float(df))
    while chi2_sf(hi, df) > target:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


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


def sortino_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    """Ratio de Sortino annualisé (cible 0) sur des rendements nets par période."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    downside = np.minimum(returns, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0.0:
        return 0.0 if returns.mean() <= 0 else float("inf")
    return float(returns.mean() / dd * np.sqrt(periods_per_year))


def annualized_return(values: np.ndarray, periods_per_year: float) -> float:
    """Rendement annualisé géométrique d'une courbe de valeur."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values[0] <= 0 or values[-1] <= 0:
        return 0.0
    years = (values.size - 1) / periods_per_year
    return float((values[-1] / values[0]) ** (1.0 / years) - 1.0)


def calmar_ratio(values: np.ndarray, periods_per_year: float) -> float:
    """Ratio de Calmar : rendement annualisé / drawdown maximal."""
    mdd = max_drawdown(values)
    ann = annualized_return(values, periods_per_year)
    if mdd == 0.0:
        return 0.0 if ann <= 0 else float("inf")
    return float(ann / mdd)
