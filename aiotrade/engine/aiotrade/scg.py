"""SCG — garde-fou financier (Safety Constraint Guard).

Chaque allocation cible ou ordre est évalué *avant* exécution contre trois
contraintes dures. Le moindre dépassement entraîne le rejet, sans arbitrage
avec le rendement espéré :

1. drawdown maximal autorisé (drawdown courant + perte de stress projetée) ;
2. budget de Value-at-Risk (VaR historique 1 pas, en fraction des fonds propres) ;
3. marge disponible (marge requise par l'exposition brute).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .stats import historical_var, portfolio_returns


def _pct(value: float) -> str:
    """Pourcentage au format français : 0.1655 -> « 16,55 % »."""
    return f"{value * 100:.2f}\u00a0%".replace(".", ",")


@dataclass(frozen=True)
class RiskLimits:
    max_drawdown: float = 0.10
    var_budget: float = 0.02
    var_confidence: float = 0.99
    max_margin_usage: float = 0.30
    margin_rates: tuple[float, ...] = (0.20, 0.25, 0.05, 0.15)
    stress_horizon: int = 10

    def validate(self, n_assets: int) -> None:
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown doit être dans ]0, 1[")
        if not 0 < self.var_budget < 1:
            raise ValueError("var_budget doit être dans ]0, 1[")
        if not 0.5 < self.var_confidence < 1:
            raise ValueError("var_confidence doit être dans ]0.5, 1[")
        if not 0 < self.max_margin_usage <= 1:
            raise ValueError("max_margin_usage doit être dans ]0, 1]")
        if len(self.margin_rates) != n_assets:
            raise ValueError("margin_rates doit contenir un taux par actif")


@dataclass
class Violation:
    rule: str
    value: float
    limit: float
    message: str


@dataclass
class GuardDecision:
    approved: bool
    violations: list[Violation] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "violations": [asdict(v) for v in self.violations],
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
        }


class SCGGuard:
    def __init__(self, limits: RiskLimits | None = None, n_assets: int = 4) -> None:
        self.limits = limits or RiskLimits()
        self.limits.validate(n_assets)
        self.n_assets = n_assets

    def check_allocation(
        self,
        weights: np.ndarray,
        history: np.ndarray,
        equity: float,
        peak_equity: float,
        stress_loss: float | None = None,
    ) -> GuardDecision:
        """Vérifie une allocation cible.

        ``stress_loss`` : perte projetée sur l'horizon (ex. quantile 5 % des
        trajectoires TAP). À défaut, VaR 1 pas × sqrt(horizon).
        """
        lim = self.limits
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (self.n_assets,):
            raise ValueError(f"weights doit contenir {self.n_assets} valeurs")
        if equity <= 0 or peak_equity <= 0:
            raise ValueError("equity et peak_equity doivent être positifs")
        peak_equity = max(peak_equity, equity)

        pnl = portfolio_returns(history, weights)
        var = historical_var(pnl, lim.var_confidence)
        if stress_loss is None:
            stress_loss = var * np.sqrt(lim.stress_horizon)
        stress_loss = float(np.clip(stress_loss, 0.0, 1.0))
        current_dd = 1.0 - equity / peak_equity
        projected_dd = 1.0 - equity * (1.0 - stress_loss) / peak_equity
        margin_usage = float(np.sum(np.abs(weights) * np.asarray(lim.margin_rates)))

        violations: list[Violation] = []
        is_flat = bool(np.allclose(weights, 0.0))
        if not is_flat and projected_dd > lim.max_drawdown:
            violations.append(
                Violation(
                    "max_drawdown",
                    projected_dd,
                    lim.max_drawdown,
                    f"Drawdown projeté {_pct(projected_dd)} > limite {_pct(lim.max_drawdown)}",
                )
            )
        if var > lim.var_budget:
            violations.append(
                Violation("var_budget", var, lim.var_budget, f"VaR {_pct(var)} > budget {_pct(lim.var_budget)}")
            )
        if margin_usage > lim.max_margin_usage:
            violations.append(
                Violation(
                    "margin",
                    margin_usage,
                    lim.max_margin_usage,
                    f"Marge requise {_pct(margin_usage)} > disponible {_pct(lim.max_margin_usage)}",
                )
            )
        return GuardDecision(
            approved=not violations,
            violations=violations,
            metrics={
                "var": var,
                "stress_loss": stress_loss,
                "current_drawdown": current_dd,
                "projected_drawdown": projected_dd,
                "margin_usage": margin_usage,
                "gross_exposure": float(np.abs(weights).sum()),
            },
        )

    def check_order(
        self,
        current_weights: np.ndarray,
        order: dict[int, float],
        history: np.ndarray,
        equity: float,
        peak_equity: float,
    ) -> GuardDecision:
        """Vérifie un ordre unitaire : ``order`` = {indice d'actif: variation de poids}."""
        post_trade = np.asarray(current_weights, dtype=float).copy()
        for index, delta in order.items():
            if not 0 <= index < self.n_assets:
                raise ValueError(f"actif inconnu : {index}")
            post_trade[index] += delta
        return self.check_allocation(post_trade, history, equity, peak_equity)
