"""Co-pilote AIOTrade : enchaîne les étapes du pipeline pour une décision.

1. Filtre d'intégrité χ² / Mahalanobis : FREEZE → CASH.
2. Macro Gate : BLACKOUT → CASH ; DERISK → exposition réduite.
3. Prévision : N trajectoires multi-pas (essaim d'agents ou Kronos).
4. TAP : faisceau par direction candidate (LONG / SHORT), stop calculé.
5. Kelly fractionnaire sur le faisceau, plafonné par le mandat.
6. SCG : la taille est divisée par deux jusqu'à admissibilité (≥ 75 % de
   trajectoires survivantes, VaR / CVaR dans le budget), sinon CASH.
7. Décision LONG / SHORT / CASH + enregistrement XAI prêt pour le registre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .execution import ExecutionConfig, ExecutionModel
from .forecast import Forecaster, MarketContext
from .integrity import IntegrityReading, IntegrityStatus
from .macro_gate import GateDecision, GateState
from .market import iso_utc
from .scg import REJECTION_KEYS, RiskMandate, SCGReport, SphericalConstraintGraph
from .sizing import ZERO_SIZE, FractionalKellySizer, KellyEstimate, PositionSize, SizingConfig
from .tap import TAPConfig, TAPPlanner, TrajectoryBundle


class Action(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    CASH = "CASH"


@dataclass(frozen=True)
class CopilotConfig:
    tap: TAPConfig = field(default_factory=TAPConfig)
    mandate: RiskMandate = field(default_factory=RiskMandate)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    max_halvings: int = 4
    stop_quantile: float = 0.8
    min_kelly: float = 0.0005  # avantage minimal (part d'équité risquée) pour s'exposer
    allow_short: bool = True


@dataclass(frozen=True)
class MarketQuote:
    price: float
    spread: float  # en prix
    avg_volume: float  # notionnel moyen par barre


@dataclass
class Decision:
    action: Action
    direction: int
    size: PositionSize
    stop_distance: float
    stop_price: float | None
    report: SCGReport | None
    kelly: KellyEstimate | None
    round_trip_cost: float
    scg_evaluations: int
    trajectories_pruned: int
    record: dict[str, Any]
    cash_reason: str | None = None  # freeze | warmup | macro | edge | scg | size

    @property
    def target_units(self) -> float:
        return self.direction * self.size.units

    @property
    def exposure(self) -> float:
        return self.direction * self.size.exposure


class Copilot:
    def __init__(self, forecaster: Forecaster, config: CopilotConfig | None = None, symbol: str = "EURUSD") -> None:
        self.config = config or CopilotConfig()
        self.forecaster = forecaster
        self.symbol = symbol
        self.planner = TAPPlanner(self.config.tap)
        self.scg = SphericalConstraintGraph(self.config.mandate)
        self.sizer = FractionalKellySizer(self.config.sizing)
        self.execution = ExecutionModel(self.config.execution)

    # ------------------------------------------------------------------
    def decide(
        self,
        ctx: MarketContext,
        quote: MarketQuote,
        integrity: IntegrityReading,
        gate: GateDecision,
        equity: float,
        peak_equity: float,
        rng: np.random.Generator,
    ) -> Decision:
        cfg = self.config
        reasons: list[str] = []
        if integrity.status == IntegrityStatus.FREEZE:
            return self.cash_decision(ctx, integrity, gate, ["FREEZE INTÉGRITÉ : flux de marché anormal, portefeuille en cash"],
                                      cash_reason="freeze")
        if integrity.status == IntegrityStatus.WARMUP:
            return self.cash_decision(ctx, integrity, gate, ["PRÉCHAUFFAGE DU FILTRE D'INTÉGRITÉ"], cash_reason="warmup")
        if gate.state == GateState.BLACKOUT or not gate.allow_new_positions:
            return self.cash_decision(ctx, integrity, gate, [gate.reason], cash_reason="macro")
        if gate.state == GateState.DERISK:
            reasons.append(gate.reason)

        bundle = self.planner.plan(self.forecaster, ctx, rng)
        cost = self.execution.round_trip(equity, quote.price, quote.spread, ctx.sigma, quote.avg_volume)
        directions = (1, -1) if cfg.allow_short else (1,)
        best: tuple[int, float, KellyEstimate] | None = None
        for d in directions:
            stop = self.planner.stop_distance(bundle, d, ctx.sigma, cfg.stop_quantile)
            outcomes = self.scg.equity_paths(bundle, float(d), stop, cost)[:, -1] - 1.0
            kelly = self.sizer.estimate(outcomes)
            if best is None or kelly.fraction > best[2].fraction:
                best = (d, stop, kelly)
        assert best is not None
        direction, stop, kelly = best
        if kelly.fraction <= cfg.min_kelly:
            return self.cash_decision(
                ctx, integrity, gate, reasons + ["AVANTAGE INSUFFISANT : Kelly fractionnaire nul sur le faisceau"],
                bundle=bundle, kelly=kelly, cost=cost, cash_reason="edge",
            )

        loss_at_stop = stop + cost
        budget_scale = RiskMandate.horizon_scale(bundle.horizon, ctx.bar_minutes)
        size = self.sizer.size(
            kelly, equity, quote.price, loss_at_stop,
            max_risk=cfg.mandate.max_loss_per_trade, max_leverage=cfg.mandate.max_leverage,
            exposure_multiplier=gate.exposure_multiplier,
        )
        evaluations, pruned = 0, 0
        report: SCGReport | None = None
        halvings = 0
        while size.lots > 0:
            report = self.scg.evaluate(bundle, direction * size.exposure, equity, peak_equity, stop, cost, budget_scale)
            evaluations += 1
            pruned += sum(report.rejections.values())
            if report.admitted:
                break
            if halvings >= cfg.max_halvings:
                size = ZERO_SIZE
                break
            size = self.sizer.shrink(size, equity, quote.price, loss_at_stop)
            halvings += 1
        if size.lots == 0 or report is None or not report.admitted:
            why = "ÉLAGAGE SCG : aucune taille admissible" if report is not None else f"TAILLE NULLE ({size.capped_by})"
            return self.cash_decision(
                ctx, integrity, gate, reasons + [why] + (report.reasons if report else []),
                bundle=bundle, kelly=kelly, cost=cost, report=report, evaluations=evaluations, pruned=pruned,
                cash_reason="scg" if report is not None else "size",
            )

        if halvings:
            reasons.append(f"Taille divisée par {2 ** halvings} pour satisfaire le SCG")
        action = Action.LONG if direction > 0 else Action.SHORT
        stop_price = quote.price * (1.0 - direction * stop)
        record = self._record(ctx, integrity, gate, bundle, report, action, direction * size.exposure, kelly, size,
                              cost, stop_price, evaluations, pruned, reasons)
        return Decision(action, direction, size, stop, stop_price, report, kelly, cost, evaluations, pruned, record)

    # ------------------------------------------------------------------
    def cash_decision(
        self,
        ctx: MarketContext,
        integrity: IntegrityReading,
        gate: GateDecision,
        reasons: list[str],
        bundle: TrajectoryBundle | None = None,
        kelly: KellyEstimate | None = None,
        cost: float = 0.0,
        report: SCGReport | None = None,
        evaluations: int = 0,
        pruned: int = 0,
        cash_reason: str = "freeze",
    ) -> Decision:
        record = self._record(ctx, integrity, gate, bundle, report, Action.CASH, 0.0, kelly, ZERO_SIZE, cost, None,
                              evaluations, pruned, reasons)
        return Decision(Action.CASH, 0, ZERO_SIZE, 0.0, None, report, kelly, cost, evaluations, pruned, record,
                        cash_reason)

    def _record(
        self,
        ctx: MarketContext,
        integrity: IntegrityReading,
        gate: GateDecision,
        bundle: TrajectoryBundle | None,
        report: SCGReport | None,
        action: Action,
        allocation: float,
        kelly: KellyEstimate | None,
        size: PositionSize,
        cost: float,
        stop_price: float | None,
        evaluations: int,
        pruned: int,
        reasons: list[str],
    ) -> dict[str, Any]:
        m = self.config.mandate
        rejections = dict(report.rejections) if report else dict.fromkeys(REJECTION_KEYS, 0)
        constraints = [f"MAX_LEVERAGE_{m.max_leverage:g}X", f"MAX_LOSS_PER_TRADE_{m.max_loss_per_trade * 100:g}PCT"]
        if stop_price is not None:
            constraints.append(f"STOP_LOSS_{stop_price:.5f}")
        if gate.state != GateState.NOMINAL:
            constraints.append(f"MACRO_{gate.state.value}_{gate.exposure_multiplier:g}X")
        if integrity.status == IntegrityStatus.FREEZE:
            constraints.append("INTEGRITY_FREEZE")
        describe = getattr(self.forecaster, "describe", None)
        return {
            "timestamp_utc": iso_utc(ctx.now),
            "symbol": self.symbol,
            "market_integrity_d2": round(integrity.d2, 2),
            "chi2_threshold": round(integrity.threshold, 2),
            "status": integrity.status.value,
            "macro_gate": gate.state.value,
            "tap_trajectories_tested": bundle.n_paths if bundle is not None else 0,
            "tap_horizon_bars": bundle.horizon if bundle is not None else 0,
            "scg_rejections": rejections,
            "admissibility_ratio": round(report.admissibility_ratio, 4) if report else None,
            "selected_allocation": round(allocation, 4),
            "decision": action.value,
            "active_constraints": constraints,
            "scg_evaluations": evaluations,
            "trajectories_pruned": pruned,
            "kelly": kelly.to_dict() if kelly else None,
            "position": size.to_dict(),
            "round_trip_cost_bps": round(cost * 1e4, 3),
            "var": round(report.var, 6) if report else None,
            "cvar": round(report.cvar, 6) if report else None,
            "forecaster": describe(ctx) if callable(describe) else {"model": getattr(self.forecaster, "name", "?")},
            "reasons": reasons,
        }
