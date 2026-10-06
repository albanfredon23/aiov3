"""Backtest walk-forward à quatre bras (protocole expérimental AIOTrade).

- Bras A : momentum classique (rétrospective choisie sur la fenêtre de calibration), 1×.
- Bras B : prédictif non contraint (signe de la prévision de l'essaim), L_max, toujours investi.
- Bras C : TAP seul (direction du faisceau, 1× si la majorité des trajectoires gagne), sans SCG.
- Bras D : AIOTrade complet : intégrité χ² + Macro Gate + TAP + SCG + Kelly + maker/taker + stops.

Tous les bras paient les frictions (spread, slippage, impact Almgren-Chriss,
commissions) : les indicateurs sont nets. Les décisions sont prises à la
clôture d'une barre et exécutées sur la barre suivante (aucune donnée future).
Le walk-forward calibre sur 12 mois et teste hors échantillon sur 3 mois,
fenêtre glissante.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import numpy as np

from .copilot import Action, Copilot, CopilotConfig, MarketQuote
from .execution import ExecutionModel
from .forecast import AgentSwarmForecaster, MarketContext
from .integrity import IntegrityConfig, IntegrityFilter, IntegrityStatus, ewma_volatility, microstructure_features
from .ledger import XAILedger
from .macro_gate import GateState, MacroCalendarGate, MacroGateConfig, StaticCalendar
from .market import MarketConfig, MarketData, generate_market, iso_utc, macro_event_mask
from .stats import calmar_ratio, max_drawdown, sortino_ratio

ARMS = ("A", "B", "C", "D")
ARM_LABELS = {
    "A": "Momentum classique",
    "B": "Prédictif non contraint",
    "C": "TAP sans SCG",
    "D": "AIOTrade complet",
}
TARGETS = {
    "max_drawdown_reduction_vs_B": 0.40,
    "sortino_net": 2.0,
    "calmar_net": 1.5,
    "scg_pruning_rate": (0.15, 0.45),
}
CONTEXT_WINDOW = 2_048


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 100_000.0
    decision_every: int = 4  # une décision par heure en barres de 15 min
    calibration_months: int = 12
    test_months: int = 3
    folds: int = 4
    trading_days_per_month: int = 21
    momentum_lookbacks: tuple[int, ...] = (24, 48, 96, 192)
    hysteresis: float = 0.25
    seed: int = 11
    copilot: CopilotConfig = field(default_factory=CopilotConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    gate: MacroGateConfig = field(default_factory=MacroGateConfig)
    ledger_in_memory: int | None = None


# ---------------------------------------------------------------- comptes
@dataclass
class Account:
    cash: float
    units: float = 0.0
    costs: float = 0.0
    trades: int = 0
    turnover: float = 0.0
    stop: float | None = None
    stops_hit: int = 0
    derisked: bool = False
    opened_at: int = 0  # barre de la dernière décision confirmant la position

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    def fill(self, delta: float, price: float, cost_frac: float) -> None:
        if delta == 0.0:
            return
        notional = abs(delta) * price
        self.cash -= delta * price + notional * cost_frac
        self.units += delta
        self.costs += notional * cost_frac
        self.turnover += notional
        self.trades += 1
        if self.units == 0.0:
            self.stop = None
            self.derisked = False


@dataclass
class Order:
    target: float
    maker: bool = False
    limit: float | None = None
    stop: float | None = None
    keep_stop: bool = False


# ---------------------------------------------------------------- indicateurs
def performance(equity: np.ndarray, bars_per_year: int) -> dict[str, float]:
    equity = np.asarray(equity, dtype=float)
    r = equity[1:] / equity[:-1] - 1.0
    years = max(r.size / bars_per_year, 1e-9)
    total = float(equity[-1] / equity[0] - 1.0)
    cagr = float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0) if equity[-1] > 0 else -1.0
    return {
        "total_return": total,
        "annualized_return": cagr,
        "annualized_volatility": float(r.std() * np.sqrt(bars_per_year)) if r.size else 0.0,
        "max_drawdown": float(max_drawdown(equity)),
        "sortino": float(sortino_ratio(r, bars_per_year)),
        "calmar": float(calmar_ratio(equity, bars_per_year)),
    }


def _clean(v: Any, nd: int = 4) -> Any:
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    return v


def _round(d: dict[str, Any], nd: int = 4) -> dict[str, Any]:
    return {k: _clean(v, nd) for k, v in d.items()}


# ---------------------------------------------------------------- moteur
class _Series:
    """Grandeurs précalculées sur toute la série (causales)."""

    def __init__(self, market: MarketData) -> None:
        self.m = market
        r = market.returns
        self.returns = r
        vol = ewma_volatility(r)
        self.sigma_close = np.sqrt(0.94 * vol**2 + 0.06 * r**2)  # vol connue à la clôture t
        self.features = microstructure_features(market)
        kernel = np.ones(96) / 96.0
        avg = np.convolve(market.volume, kernel)[: market.bars]
        avg[:96] = np.cumsum(market.volume[:96]) / np.arange(1, 97)
        self.avg_volume = avg
        self.step = timedelta(minutes=market.bar_minutes)
        self.event_times = [e.time_utc for e in market.events]

    def context(self, t: int) -> MarketContext:
        m = self.m
        lo = max(0, t + 1 - CONTEXT_WINDOW)
        now = m.timestamps[t] + self.step
        i = bisect.bisect_left(self.event_times, now)
        j = bisect.bisect_right(self.event_times, now + timedelta(hours=4))
        return MarketContext(
            timestamps=m.timestamps[lo : t + 1],
            open=m.open[lo : t + 1],
            high=m.high[lo : t + 1],
            low=m.low[lo : t + 1],
            close=m.close[lo : t + 1],
            volume=m.volume[lo : t + 1],
            returns=self.returns[lo : t + 1],
            sigma=float(self.sigma_close[t]),
            bar_minutes=m.bar_minutes,
            upcoming_events=self.m.events[i:j],
        )


def _choose_momentum(returns: np.ndarray, lookbacks: tuple[int, ...], every: int) -> int:
    """Rétrospective de momentum au meilleur ratio de Sharpe brut sur la calibration."""
    best, best_score = lookbacks[0], -np.inf
    csum = np.concatenate([[0.0], np.cumsum(returns)])
    for lb in lookbacks:
        idx = np.arange(lb, returns.size - every, every)
        signal = np.sign(csum[idx + 1] - csum[idx + 1 - lb])
        fwd = csum[idx + 1 + every] - csum[idx + 1]
        pnl = signal * fwd
        score = pnl.mean() / (pnl.std() + 1e-12)
        if score > best_score:
            best, best_score = lb, score
    return best


@dataclass
class FoldResult:
    calibration: dict[str, Any]
    test_start: int
    test_end: int
    equity: dict[str, np.ndarray]
    accounts: dict[str, Account]
    integrity_d2: np.ndarray
    integrity_status: list[str]
    gate_states: list[str]
    decisions: list[str]
    scg_stats: dict[str, float]
    ledger: XAILedger
    momentum_lookback: int
    exposure_d: np.ndarray


def run_fold(
    market: MarketData,
    cal_start: int,
    test_start: int,
    test_end: int,
    config: BacktestConfig | None = None,
    rng_seed: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    ledger: XAILedger | None = None,
) -> FoldResult:
    cfg = config or BacktestConfig()
    if test_start - cal_start < 800 or test_end <= test_start:
        raise ValueError("au moins 800 barres de calibration et une fenêtre de test non vide sont requises")
    s = _Series(market)
    rng = np.random.default_rng(cfg.seed if rng_seed is None else rng_seed)
    execm = ExecutionModel(cfg.copilot.execution)
    every = cfg.decision_every
    horizon = cfg.copilot.tap.horizon
    lmax = cfg.copilot.mandate.max_leverage

    # ---- calibration (fenêtre d'entraînement uniquement)
    integrity = IntegrityFilter(cfg.integrity)
    gate = MacroCalendarGate(StaticCalendar(market.events), cfg.gate)
    event_mask = macro_event_mask(market)
    calib = integrity.calibrate(s.features[cal_start:test_start], exclude=event_mask[cal_start:test_start])
    forecaster = AgentSwarmForecaster()
    swarm_fit = forecaster.calibrate(
        market.close[cal_start:test_start], s.returns[cal_start:test_start], s.sigma_close[cal_start:test_start]
    )
    for t in range(max(cal_start, test_start - 4_000), test_start - 1):  # apprentissage en ligne des poids
        if t + 1 - cal_start > 1_000:
            forecaster.observe(s.context(t), float(s.returns[t + 1]))
    lookback = _choose_momentum(s.returns[cal_start:test_start], cfg.momentum_lookbacks, every)
    copilot = Copilot(forecaster, cfg.copilot, symbol=market.symbol)
    ledger = ledger if ledger is not None else XAILedger(keep_in_memory=cfg.ledger_in_memory)

    accounts = {a: Account(cfg.initial_equity) for a in ARMS}
    pending: dict[str, Order | None] = dict.fromkeys(ARMS, None)
    peak_d = cfg.initial_equity
    n = test_end - test_start
    equity = {a: np.empty(n + 1) for a in ARMS}
    for a in ARMS:
        equity[a][0] = cfg.initial_equity
    d2 = np.zeros(n)
    exposure_d = np.zeros(n)
    statuses: list[str] = []
    gates: list[str] = []
    decisions: list[str] = []
    scg = {"evaluations": 0, "tested": 0, "pruned": 0, "decisions": 0, "scg_vetoes": 0, "resized": 0,
           "freeze_exits": 0, "macro_exits": 0, "macro_derisks": 0, "bars_in_market": 0}

    def taker_cost(t: int, notional: float, price: float) -> float:
        return execm.cost(notional, price, market.spread[t], s.sigma_close[t], s.avg_volume[t]).total

    last_ctx: MarketContext | None = None
    for k, t in enumerate(range(test_start, test_end)):
        o, h, lo_, c = market.open[t], market.high[t], market.low[t], market.close[t]
        # 1) exécution des ordres de la barre précédente
        for a in ARMS:
            acc, order = accounts[a], pending[a]
            if order is not None and not order.maker:
                delta = order.target - acc.units
                acc.fill(delta, o, taker_cost(t, abs(delta) * o, o))
                if acc.units != 0.0 and not order.keep_stop:
                    acc.stop = order.stop
                pending[a] = None
        # 2) stops (bras D)
        acc = accounts["D"]
        if acc.units != 0.0 and acc.stop is not None:
            hit = (acc.units > 0 and lo_ <= acc.stop) or (acc.units < 0 and h >= acc.stop)
            if hit:
                px = min(o, acc.stop) if acc.units > 0 else max(o, acc.stop)
                acc.fill(-acc.units, px, taker_cost(t, abs(acc.units) * px, px))
                acc.stops_hit += 1
                pending["D"] = None
        # 3) ordres maker (limite passive), repli taker à la clôture
        order = pending["D"]
        if order is not None and order.maker:
            delta = order.target - acc.units
            side = 1 if delta > 0 else -1
            if delta != 0.0 and execm.maker_fill(side, order.limit, lo_, h, rng):
                acc.fill(delta, order.limit, execm.cost(0, order.limit, 0, 0, 1, maker=True).total)
            elif delta != 0.0:
                acc.fill(delta, c, taker_cost(t, abs(delta) * c, c))
            if acc.units != 0.0:
                acc.stop = order.stop
            pending["D"] = None
        # 4) valorisation
        for a in ARMS:
            equity[a][k + 1] = accounts[a].equity(c)
        peak_d = max(peak_d, equity["D"][k + 1])
        scg["bars_in_market"] += int(accounts["D"].units != 0.0)
        exposure_d[k] = accounts["D"].units * c / equity["D"][k + 1]

        # 5) état du marché à la clôture
        reading = integrity.update(s.features[t])
        now = market.timestamps[t] + s.step
        gd = gate.evaluate(now)
        d2[k] = reading.d2
        statuses.append(reading.status.value)
        gates.append(gd.state.value)
        ctx = s.context(t)
        if last_ctx is not None:
            forecaster.observe(last_ctx, float(s.returns[t]))
        last_ctx = ctx
        eq_d = equity["D"][k + 1]

        # 6) garde-fous permanents du bras D (entre deux décisions)
        acc = accounts["D"]
        forced = None
        if acc.units != 0.0 and reading.status == IntegrityStatus.FREEZE:
            forced, scg["freeze_exits"] = "freeze", scg["freeze_exits"] + 1
            pending["D"] = Order(0.0)
        elif acc.units != 0.0 and gd.state == GateState.BLACKOUT:
            forced, scg["macro_exits"] = "blackout", scg["macro_exits"] + 1
            pending["D"] = Order(0.0)
        elif acc.units != 0.0 and gd.state == GateState.DERISK and not acc.derisked:
            acc.derisked = True
            scg["macro_derisks"] += 1
            pending["D"] = Order(acc.units * gd.exposure_multiplier, keep_stop=True)
        if forced:
            rec = copilot.cash_decision(ctx, reading, gd, [f"SORTIE FORCÉE ({forced.upper()}) : {gd.reason if forced == 'blackout' else 'flux anormal'}"]).record
            ledger.append(rec)

        # 7) décisions périodiques
        if (t - test_start) % every == 0 and t + 1 < test_end:
            price, spread, vol = c, market.spread[t], s.avg_volume[t]
            # A : momentum
            mom = np.sign(s.returns[t - lookback + 1 : t + 1].sum())
            pending["A"] = Order(mom * equity["A"][k + 1] / price)
            # B : prédictif non contraint, toujours investi au levier max
            exp_ret = forecaster.expected_return(ctx, horizon)
            pending["B"] = Order((1.0 if exp_ret >= 0 else -1.0) * lmax * equity["B"][k + 1] / price)
            # C : TAP sans SCG
            bundle = copilot.planner.plan(forecaster, ctx, rng)
            finals = bundle.cumulative[:, -1]
            direction = 1.0 if finals.mean() >= 0 else -1.0
            p_win = float((direction * finals > 0).mean())
            pending["C"] = Order(direction * equity["C"][k + 1] / price if p_win > 0.5 else 0.0)
            # D : AIOTrade complet
            if forced is None:
                quote = MarketQuote(price, spread, vol)
                dec = copilot.decide(ctx, quote, reading, gd, eq_d, peak_d, rng)
                ledger.append(dec.record)
                scg["decisions"] += 1
                scg["evaluations"] += dec.scg_evaluations
                scg["tested"] += dec.scg_evaluations * cfg.copilot.tap.n_paths
                scg["pruned"] += dec.trajectories_pruned
                if dec.scg_evaluations and dec.action == Action.CASH:
                    scg["scg_vetoes"] += 1
                elif dec.scg_evaluations > 1:
                    scg["resized"] += 1
                decisions.append(dec.action.value)
                target = dec.target_units
                cur = acc.units
                same_side = np.sign(target) == np.sign(cur) and cur != 0.0
                if target == 0.0:
                    # Avantage insuffisant : la position déjà admise vit jusqu'au bout de son horizon.
                    # Toute autre raison (risque, macro, intégrité, SCG) la coupe immédiatement.
                    if cur != 0.0 and not (dec.cash_reason == "edge" and k - acc.opened_at < horizon):
                        pending["D"] = Order(0.0)
                elif same_side and abs(target - cur) <= cfg.hysteresis * abs(cur):
                    acc.opened_at = k  # hystérésis : position reconduite, stop inchangé
                else:
                    acc.opened_at = k
                    limit = price - np.sign(target - cur) * spread / 2.0
                    if np.sign(target) == np.sign(target - cur):
                        pending["D"] = Order(target, maker=True, limit=limit, stop=dec.stop_price)
                    else:
                        pending["D"] = Order(target, stop=dec.stop_price)
            else:
                decisions.append(Action.CASH.value)
        if progress and k % 500 == 0:
            progress(k, n)

    return FoldResult(
        calibration={
            **calib,
            "momentum_lookback": lookback,
            "vol_regime_edges": [float(x) for x in forecaster.vol_edges],
            "swarm": swarm_fit,
        },
        test_start=test_start,
        test_end=test_end,
        equity=equity,
        accounts=accounts,
        integrity_d2=d2,
        integrity_status=statuses,
        gate_states=gates,
        decisions=decisions,
        scg_stats=scg,
        ledger=ledger,
        momentum_lookback=lookback,
        exposure_d=exposure_d,
    )


def arm_summary(fold: FoldResult, arm: str, bars_per_year: int, initial: float) -> dict[str, Any]:
    acc = fold.accounts[arm]
    eq = fold.equity[arm]
    perf = performance(eq, bars_per_year)
    perf.update(
        {
            "trades": acc.trades,
            "costs_paid_pct": acc.costs / initial,
            "turnover": acc.turnover / initial,
        }
    )
    if arm == "D":
        perf["stops_hit"] = acc.stops_hit
    return perf


def scg_rates(stats: dict[str, float]) -> dict[str, float]:
    tested = stats["tested"] or 1
    decisions = stats["decisions"] or 1
    return {
        "scg_pruning_rate": stats["pruned"] / tested,
        "scg_veto_rate": stats["scg_vetoes"] / decisions,
        "scg_resize_rate": stats["resized"] / decisions,
    }


def walk_forward(
    config: BacktestConfig | None = None,
    market_config: MarketConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    cfg = config or BacktestConfig()
    bars_per_day = 24 * 60 // (market_config.bar_minutes if market_config else 15)
    month = cfg.trading_days_per_month * bars_per_day
    cal, test = cfg.calibration_months * month, cfg.test_months * month
    total = cal + cfg.folds * test
    mc = market_config or MarketConfig(bars=total, seed=cfg.seed)
    if mc.bars < total:
        raise ValueError(f"la série doit contenir au moins {total} barres")
    market = generate_market(mc)
    bpy = market.bars_per_year
    folds_out = []
    chained = {a: [cfg.initial_equity] for a in ARMS}
    stats_total: dict[str, float] = {}
    ledger_heads = []
    sample_records: list[dict[str, Any]] = []
    for f in range(cfg.folds):
        test_start = cal + f * test
        fold = run_fold(market, test_start - cal, test_start, test_start + test, cfg, rng_seed=cfg.seed + f)
        arms = {a: _round(arm_summary(fold, a, bpy, cfg.initial_equity)) for a in ARMS}
        for a in ARMS:
            growth = fold.equity[a][1:] / fold.equity[a][0]
            chained[a].extend((chained[a][-1] * growth).tolist())
        for key, v in fold.scg_stats.items():
            stats_total[key] = stats_total.get(key, 0) + v
        verify = fold.ledger.verify()
        ledger_heads.append({"fold": f, "records": len(fold.ledger), "head_hash": fold.ledger.head_hash, "verified": verify.ok})
        if not sample_records:
            sample_records = [r for r in fold.ledger.records() if r["decision"] != "CASH"][:2] + [
                r for r in fold.ledger.records() if r["decision"] == "CASH" and r["scg_evaluations"]
            ][:1]
        folds_out.append(
            {
                "fold": f,
                "test_window": [iso_utc(market.timestamps[test_start]), iso_utc(market.timestamps[test_start + test - 1])],
                "calibration": fold.calibration,
                "arms": arms,
                "scg": _round({**fold.scg_stats, **scg_rates(fold.scg_stats)}),
                "time_in_market_D": round(fold.scg_stats["bars_in_market"] / test, 4),
            }
        )
        if progress:
            progress(f"pli {f + 1}/{cfg.folds} terminé")

    aggregate = {a: _round(performance(np.array(chained[a]), bpy)) for a in ARMS}
    rates = scg_rates(stats_total)
    dd_b, dd_d = aggregate["B"]["max_drawdown"], aggregate["D"]["max_drawdown"]
    reduction = 1.0 - dd_d / dd_b if dd_b > 0 else 0.0
    lo, hi = TARGETS["scg_pruning_rate"]
    checks = {
        "max_drawdown_reduction_vs_B": {"target": ">= 40 %", "measured": round(reduction, 4), "met": reduction >= 0.40},
        "sortino_net": {"target": "> 2,0", "measured": aggregate["D"]["sortino"], "met": aggregate["D"]["sortino"] > 2.0},
        "calmar_net": {"target": "> 1,5", "measured": aggregate["D"]["calmar"], "met": aggregate["D"]["calmar"] > 1.5},
        "scg_pruning_rate": {
            "target": "15 à 45 %",
            "measured": round(rates["scg_pruning_rate"], 4),
            "met": lo <= rates["scg_pruning_rate"] <= hi,
        },
    }
    return {
        "protocol": {
            "data": "marché synthétique (GARCH + régimes + annonces macro), aucune donnée réelle",
            "symbol": market.symbol,
            "bar_minutes": market.bar_minutes,
            "calibration_months": cfg.calibration_months,
            "test_months": cfg.test_months,
            "folds": cfg.folds,
            "bars": market.bars,
            "seed": mc.seed,
            "decision_every_bars": cfg.decision_every,
            "costs": "nets de spread, slippage, impact Almgren-Chriss et commissions pour les 4 bras",
            "mandate": {k: getattr(cfg.copilot.mandate, k) for k in cfg.copilot.mandate.__dataclass_fields__},
        },
        "arm_labels": ARM_LABELS,
        "aggregate": aggregate,
        "scg": _round({**stats_total, **rates}),
        "targets": checks,
        "folds": folds_out,
        "ledger": ledger_heads,
        "sample_records": sample_records,
    }
