"""Co-pilote AIOTrade : orchestre TAP → SCG → filtre χ² sur un marché simulé.

Boucle de décision à chaque pas :

1. le filtre χ² surveille la distribution récente du marché ; en cas d'alerte,
   le portefeuille est immédiatement ramené en cash (mode sécurité) ;
2. hors mode sécurité, tous les ``rebalance_every`` pas, le moteur TAP propose
   ses scénarios d'allocation et leurs faisceaux de trajectoires ;
3. le garde-fou SCG rejette chaque scénario qui violerait une contrainte ;
4. parmi les scénarios admissibles, le meilleur score est retenu ; le cash est
   toujours admissible, il n'y a donc jamais d'impasse.

Toutes les décisions sont journalisées pour l'audit (conformité, surveillance).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chi2 import Chi2Config, Chi2Filter, market_features
from .market import MarketConfig, MarketData, generate_market
from .scg import RiskLimits, SCGGuard
from .stats import max_drawdown
from .tap import TAPConfig, TAPEngine


@dataclass(frozen=True)
class CopilotConfig:
    rebalance_every: int = 5
    initial_equity: float = 1_000_000.0
    cost_multiplier: float = 1.0  # coût = demi-spread × rotation × multiplicateur


@dataclass
class AuditEvent:
    step: int
    kind: str  # rebalance | rejected | safe_mode_on | safe_mode_off | hold
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"step": self.step, "kind": self.kind, "message": self.message, "details": self.details}


@dataclass
class SimulationResult:
    assets: tuple[str, ...]
    start_step: int
    equity: np.ndarray
    benchmark: np.ndarray
    weights: np.ndarray  # (pas simulés, N)
    safe_mode: np.ndarray  # (pas simulés,) bool
    chi2_p_values: np.ndarray
    events: list[AuditEvent]
    last_scenarios: list[dict]
    limits: RiskLimits
    shock: str
    shock_window: tuple[int, int] | None

    def summary(self) -> dict:
        eq, bm = self.equity, self.benchmark
        rejected = [e for e in self.events if e.kind == "rejected"]
        return {
            "final_return": float(eq[-1] / eq[0] - 1.0),
            "benchmark_return": float(bm[-1] / bm[0] - 1.0),
            "max_drawdown": max_drawdown(eq),
            "benchmark_max_drawdown": max_drawdown(bm),
            "drawdown_limit": self.limits.max_drawdown,
            "rejected_allocations": len(rejected),
            "rebalances": sum(1 for e in self.events if e.kind == "rebalance"),
            "safe_mode_steps": int(self.safe_mode.sum()),
            "safe_mode_activations": sum(1 for e in self.events if e.kind == "safe_mode_on"),
        }

    def to_dict(self, max_points: int = 400) -> dict:
        n = self.equity.size
        idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
        steps = (idx + self.start_step).tolist()
        return {
            "assets": list(self.assets),
            "shock": self.shock,
            "shock_window": list(self.shock_window) if self.shock_window else None,
            "summary": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.summary().items()},
            "series": {
                "step": steps,
                "equity": np.round(self.equity[idx], 2).tolist(),
                "benchmark": np.round(self.benchmark[idx], 2).tolist(),
                "safe_mode": self.safe_mode[idx].astype(bool).tolist(),
                "chi2_p_value": [float(f"{p:.4g}") for p in self.chi2_p_values[idx]],
                "gross_exposure": np.round(np.abs(self.weights[idx]).sum(axis=1), 4).tolist(),
            },
            "events": [e.to_dict() for e in self.events[-200:]],
            "last_scenarios": self.last_scenarios,
        }


class AIOTradeCopilot:
    def __init__(
        self,
        limits: RiskLimits | None = None,
        tap_config: TAPConfig | None = None,
        chi2_config: Chi2Config | None = None,
        config: CopilotConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.tap = TAPEngine(tap_config, seed=seed)
        self.chi2_config = chi2_config or Chi2Config()
        self.config = config or CopilotConfig()

    def run(self, market: MarketData) -> SimulationResult:
        n = len(market.assets)
        guard = SCGGuard(self.limits, n_assets=n)
        features, names = market_features(market)
        chi2 = Chi2Filter(self.chi2_config, series_names=names)
        start = chi2.min_history - 1
        if start + 2 > market.steps:
            raise ValueError("série trop courte pour la référence du filtre χ²")
        cost_rate = market.spreads_bps / 2.0 / 10_000.0 * self.config.cost_multiplier

        equity = self.config.initial_equity
        peak = equity
        weights = np.zeros(n)
        eq_curve, bm_curve, w_hist, safe_hist, p_hist = [], [], [], [], []
        benchmark = equity
        bm_weights = np.full(n, 1.0 / n)
        events: list[AuditEvent] = []
        last_scenarios: list[dict] = []

        for t in range(start, market.steps - 1):
            # 1) surveillance χ² sur l'information disponible jusqu'à t inclus
            reading = chi2.update(features[: t + 1], t)
            history = market.returns[: t + 1]
            target = weights
            if reading.safe_mode:
                target = np.zeros(n)
                if reading.alert and (not safe_hist or not safe_hist[-1]):
                    events.append(
                        AuditEvent(
                            t,
                            "safe_mode_on",
                            "Rupture détectée par le filtre χ² : portefeuille mis en sécurité (cash).",
                            reading.to_dict(),
                        )
                    )
            else:
                if safe_hist and safe_hist[-1]:
                    events.append(AuditEvent(t, "safe_mode_off", "Marché revenu à la normale : reprise.", {}))
                due = (t - start) % self.config.rebalance_every == 0
                if not due and not np.allclose(weights, 0.0):
                    # surveillance continue : l'allocation en place reste-t-elle admissible ?
                    held = guard.check_allocation(weights, history[-self.tap.config.lookback * 2 :], equity, peak)
                    if not held.approved:
                        events.append(
                            AuditEvent(
                                t,
                                "rejected",
                                "SCG : l'allocation en place viole une contrainte, réallocation immédiate.",
                                {"scenario": "position_courante", "violations": [v.rule for v in held.violations]},
                            )
                        )
                        due = True
                if due:
                    target, scenario_report = self._decide(t, history, equity, peak, guard, events, market)
                    last_scenarios = scenario_report

            # 2) exécution au prix de clôture t, avec coût de demi-spread
            turnover = np.abs(target - weights)
            equity -= equity * float(np.sum(turnover * cost_rate[t]))
            weights = target

            # 3) le marché évolue de t à t + 1
            r = market.returns[t + 1]
            equity *= 1.0 + float(weights @ r)
            benchmark *= 1.0 + float(bm_weights @ r)
            peak = max(peak, equity)

            eq_curve.append(equity)
            bm_curve.append(benchmark)
            w_hist.append(weights.copy())
            safe_hist.append(reading.safe_mode)
            p_hist.append(reading.p_value)

        return SimulationResult(
            assets=market.assets,
            start_step=start + 1,
            equity=np.array(eq_curve),
            benchmark=np.array(bm_curve),
            weights=np.array(w_hist),
            safe_mode=np.array(safe_hist, dtype=bool),
            chi2_p_values=np.array(p_hist),
            events=events,
            last_scenarios=last_scenarios,
            limits=self.limits,
            shock=market.shock.value,
            shock_window=market.shock_window,
        )

    def _decide(
        self,
        t: int,
        history: np.ndarray,
        equity: float,
        peak: float,
        guard: SCGGuard,
        events: list[AuditEvent],
        market: MarketData,
    ) -> tuple[np.ndarray, list[dict]]:
        scenarios = self.tap.propose(history)
        report: list[dict] = []
        admissible = []
        for sc in scenarios:
            decision = guard.check_allocation(
                sc.weights, history[-self.tap.config.lookback * 2 :], equity, peak, stress_loss=max(0.0, -sc.p05)
            )
            entry = sc.to_dict(market.assets)
            entry["guard"] = decision.to_dict()
            report.append(entry)
            if decision.approved:
                admissible.append(sc)
            else:
                events.append(
                    AuditEvent(
                        t,
                        "rejected",
                        f"SCG rejette « {sc.label} » : " + " ; ".join(v.message for v in decision.violations),
                        {"scenario": sc.name, "violations": [v.rule for v in decision.violations]},
                    )
                )
        chosen = max(admissible, key=lambda s: s.score)
        events.append(
            AuditEvent(
                t,
                "rebalance",
                f"Allocation « {chosen.label} » retenue (score {chosen.score:+.4f}).",
                {"scenario": chosen.name, "weights": {a: round(float(w), 4) for a, w in zip(market.assets, chosen.weights)}},
            )
        )
        return chosen.weights.copy(), report


def simulate(
    market_config: MarketConfig,
    limits: RiskLimits | None = None,
    tap_config: TAPConfig | None = None,
    chi2_config: Chi2Config | None = None,
) -> SimulationResult:
    market = generate_market(market_config)
    copilot = AIOTradeCopilot(limits, tap_config, chi2_config, seed=market_config.seed)
    return copilot.run(market)
