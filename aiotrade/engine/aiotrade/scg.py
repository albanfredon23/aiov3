"""SCG — Spherical Constraint Graph : garde-fou financier.

Pour une exposition candidate w (fraction signée de l'équité), chaque
trajectoire du faisceau TAP est projetée en courbe d'équité

    E(i, τ) = 1 + w · R(i, τ) − frictions,

où R est le rendement cumulé, le stop loss coupe la trajectoire (avec le gap
éventuel au-delà du stop) et les frictions réelles (demi-spread, commissions,
slippage, impact) sont déduites à l'entrée et à la sortie.

Contraintes par trajectoire (chaque trajectoire rejetée est attribuée à la
première contrainte violée) : levier maximal, drawdown du compte et drawdown
intra-horizon, perte maximale par trade, perte finale au-delà du budget CVaR,
puis du budget VaR. Les budgets VaR / CVaR du mandat sont journaliers et
ramenés à l'horizon de décision par la règle de la racine du temps.

La décision n'est admissible que si la part des trajectoires survivantes
atteint ``min_admissibility`` (75 % par défaut) et que la VaR et la CVaR du
faisceau respectent leur budget. Sinon : élagage dur.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .tap import TrajectoryBundle

REJECTION_KEYS = ("leverage_violation", "drawdown_violation", "loss_violation", "cvar_violation", "var_violation")


@dataclass(frozen=True)
class RiskMandate:
    max_drawdown: float = 0.10  # drawdown maximal du compte (mandat)
    max_path_drawdown: float = 0.02  # drawdown maximal à l'intérieur d'une trajectoire
    max_loss_per_trade: float = 0.01  # 1 % de l'équité
    max_leverage: float = 1.5
    var_confidence: float = 0.95
    var_budget: float = 0.01  # VaR journalière à 95 % : 1 % de l'équité
    cvar_budget: float = 0.015  # CVaR journalière à 95 % : 1,5 %
    min_admissibility: float = 0.75

    def __post_init__(self) -> None:
        checks = {
            "max_drawdown": 0 < self.max_drawdown < 1,
            "max_path_drawdown": 0 < self.max_path_drawdown < 1,
            "max_loss_per_trade": 0 < self.max_loss_per_trade < 0.2,
            "max_leverage": 0 < self.max_leverage <= 10,
            "var_confidence": 0.5 < self.var_confidence < 1,
            "var_budget": 0 < self.var_budget < 1,
            "cvar_budget": 0 < self.cvar_budget < 1,
            "min_admissibility": 0 < self.min_admissibility <= 1,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"paramètres de mandat invalides : {', '.join(bad)}")

    @staticmethod
    def horizon_scale(horizon_bars: int, bar_minutes: int) -> float:
        """Facteur racine du temps pour ramener un budget journalier à l'horizon de décision."""
        return float(np.sqrt(horizon_bars * bar_minutes / (24 * 60)))

    def constraint_labels(self) -> list[str]:
        return [
            f"MAX_LEVERAGE_{self.max_leverage:g}X",
            f"MAX_LOSS_PER_TRADE_{self.max_loss_per_trade * 100:g}%",
            f"MAX_DRAWDOWN_{self.max_drawdown * 100:g}%",
            f"MIN_ADMISSIBILITY_{self.min_admissibility * 100:g}%",
        ]


@dataclass
class SCGReport:
    exposure: float
    admissibility_ratio: float
    admitted: bool
    rejections: dict[str, int]
    var: float
    cvar: float
    expected_net: float
    stop_distance: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "exposure": round(self.exposure, 4),
            "admissibility_ratio": round(self.admissibility_ratio, 4),
            "admitted": self.admitted,
            "rejections": dict(self.rejections),
            "var": round(self.var, 6),
            "cvar": round(self.cvar, 6),
            "expected_net": round(self.expected_net, 6),
            "stop_distance": round(self.stop_distance, 6),
            "reasons": self.reasons,
        }


class SphericalConstraintGraph:
    def __init__(self, mandate: RiskMandate | None = None) -> None:
        self.mandate = mandate or RiskMandate()

    def equity_paths(
        self, bundle: TrajectoryBundle, exposure: float, stop_distance: float, cost_per_unit: float
    ) -> np.ndarray:
        """Courbes d'équité relatives (N, H + 1), stop loss et frictions inclus."""
        direction = 1 if exposure >= 0 else -1
        move = direction * bundle.cumulative  # rendement favorable > 0
        hit = move <= -stop_distance
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), bundle.horizon)
        steps = np.arange(bundle.horizon)[None, :]
        # Après le stop, la trajectoire est figée à sa valeur de sortie (gap compris).
        frozen = np.take_along_axis(move, np.minimum(first, bundle.horizon - 1)[:, None], axis=1)
        move = np.where(steps > first[:, None], frozen, move)
        entry_cost = abs(exposure) * cost_per_unit / 2.0
        curves = 1.0 + abs(exposure) * move - entry_cost
        curves[:, -1] -= abs(exposure) * cost_per_unit / 2.0  # coût de sortie
        stopped = first < bundle.horizon
        curves[stopped, -1] = (1.0 + abs(exposure) * frozen[stopped, 0] - abs(exposure) * cost_per_unit)
        return np.hstack([np.full((bundle.n_paths, 1), 1.0 - entry_cost), curves])

    def evaluate(
        self,
        bundle: TrajectoryBundle,
        exposure: float,
        equity: float,
        peak_equity: float,
        stop_distance: float,
        cost_per_unit: float,
        budget_scale: float = 1.0,
    ) -> SCGReport:
        m = self.mandate
        if equity <= 0 or peak_equity <= 0:
            raise ValueError("equity et peak_equity doivent être positifs")
        if not 0 < budget_scale <= 1:
            raise ValueError("budget_scale doit être dans ]0 ; 1]")
        peak_equity = max(peak_equity, equity)
        n = bundle.n_paths
        rejections = dict.fromkeys(REJECTION_KEYS, 0)
        reasons: list[str] = []
        if exposure == 0.0:
            return SCGReport(0.0, 1.0, True, rejections, 0.0, 0.0, 0.0, stop_distance, ["CASH : aucune exposition"])
        var_budget, cvar_budget = m.var_budget * budget_scale, m.cvar_budget * budget_scale

        curves = self.equity_paths(bundle, exposure, stop_distance, cost_per_unit)
        trough = curves.min(axis=1)
        running_peak = np.maximum.accumulate(curves, axis=1)
        path_dd = (1.0 - curves / running_peak).max(axis=1)
        account_dd = 1.0 - equity * trough / peak_equity
        losses = 1.0 - curves[:, -1]

        checks = {
            "leverage_violation": np.full(n, abs(exposure) > m.max_leverage + 1e-9),
            "drawdown_violation": (account_dd > m.max_drawdown) | (path_dd > m.max_path_drawdown),
            "loss_violation": (1.0 - trough) > m.max_loss_per_trade + 1e-12,
            "cvar_violation": losses > cvar_budget,
            "var_violation": losses > var_budget,
        }
        rejected = np.zeros(n, dtype=bool)
        for key in REJECTION_KEYS:  # attribution à la première contrainte violée
            fresh = checks[key] & ~rejected
            rejections[key] = int(fresh.sum())
            rejected |= fresh
        ratio = float(1.0 - rejected.mean())

        threshold = np.quantile(losses, m.var_confidence)
        var = float(max(0.0, threshold))
        tail = losses[losses >= threshold]
        cvar = float(max(0.0, tail.mean())) if tail.size else 0.0
        var_ok, cvar_ok = var <= var_budget, cvar <= cvar_budget
        if not var_ok:
            reasons.append(f"VaR {var:.4%} > budget {var_budget:.4%}")
        if not cvar_ok:
            reasons.append(f"CVaR {cvar:.4%} > budget {cvar_budget:.4%}")
        if ratio < m.min_admissibility:
            reasons.append(f"Ratio admissible {ratio:.1%} < seuil {m.min_admissibility:.0%}")
        admitted = ratio >= m.min_admissibility and var_ok and cvar_ok
        return SCGReport(
            exposure=exposure,
            admissibility_ratio=ratio,
            admitted=admitted,
            rejections=rejections,
            var=var,
            cvar=cvar,
            expected_net=float(-losses.mean()),
            stop_distance=stop_distance,
            reasons=reasons,
        )
