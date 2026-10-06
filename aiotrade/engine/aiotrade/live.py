"""Exécution en temps réel sur chandeliers d'un courtier (papier, testnet ou live).

À chaque clôture de barre : historique → calibrage walk-forward (filtre
d'intégrité, essaim d'agents) → décision du co-pilote → enregistrement XAI →
ordre éventuel. Au comptant, un signal SHORT ramène la position à zéro.

Sur des chandeliers, le spread et la profondeur L1 ne sont pas historisés :
le filtre d'intégrité utilise des proxys (amplitude relative de la barre pour
le spread, taille moyenne des transactions pour la profondeur).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .broker.base import Broker, BrokerError, Candle, OrderRequest, Side
from .copilot import Action, Copilot, CopilotConfig, MarketQuote
from .forecast import AgentSwarmForecaster, MarketContext
from .integrity import IntegrityConfig, IntegrityFilter, ewma_volatility, microstructure_features
from .ledger import XAILedger
from .macro_gate import MacroCalendarGate, StaticCalendar
from .market import MarketData
from .sizing import SizingConfig

INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def market_from_candles(candles: list[Candle], symbol: str, book_spread: float) -> MarketData:
    if len(candles) < 2:
        raise ValueError("historique insuffisant")
    bar_minutes = int((candles[1].open_time - candles[0].open_time).total_seconds() // 60)
    close = np.array([c.close for c in candles])
    high = np.array([c.high for c in candles])
    low = np.array([c.low for c in candles])
    trades = np.maximum(np.array([c.trades for c in candles], dtype=float), 1.0)
    quote_volume = np.maximum(np.array([c.quote_volume for c in candles]), 1e-9)
    rng = np.maximum(high - low, close * 1e-6)
    spread = np.maximum(book_spread, 0.05 * rng)  # proxy de spread effectif
    depth = 20.0 * quote_volume / trades  # proxy de profondeur : taille moyenne des transactions
    return MarketData(
        symbol=symbol,
        bar_minutes=bar_minutes,
        timestamps=[c.open_time for c in candles],
        open=np.array([c.open for c in candles]),
        high=high,
        low=low,
        close=close,
        volume=quote_volume,
        spread=spread,
        depth=depth,
    )


@dataclass
class LiveStep:
    record: dict[str, Any]
    order: dict[str, Any] | None
    equity: float
    price: float


class LiveRunner:
    def __init__(
        self,
        broker: Broker,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        history_bars: int = 2_000,
        ledger_path: str | Path = "ledger/decisions.jsonl",
        calendar_path: str | Path | None = None,
        config: CopilotConfig | None = None,
        send_orders: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        if interval not in INTERVAL_MINUTES:
            raise ValueError(f"intervalle non pris en charge : {interval}")
        if history_bars < 1_500:
            raise ValueError("au moins 1 500 barres d'historique sont nécessaires au calibrage")
        self.broker = broker
        self.symbol = symbol
        self.interval = interval
        self.history_bars = history_bars
        self.ledger = XAILedger(ledger_path, keep_in_memory=50)
        self.calendar = StaticCalendar.from_json(calendar_path) if calendar_path else StaticCalendar()
        self.config = config or CopilotConfig()
        self.send_orders = send_orders
        self.log = log
        self.rng = np.random.default_rng()
        self._peak = max((float(r.get("equity", 0.0)) for r in self.ledger.records()), default=0.0)

    def _sizing_for(self, price: float) -> SizingConfig:
        rules = self.broker.rules(self.symbol)
        base = self.config.sizing
        need = max(rules.min_qty, rules.step_size, rules.min_notional / price)
        min_lot = math.ceil(need / rules.step_size - 1e-9) * rules.step_size
        return replace(base, contract_size=1.0, lot_step=rules.step_size, min_lot=min_lot,
                       max_lot=max(min_lot, rules.max_qty))

    def step(self) -> LiveStep:
        history = self.broker.history(self.symbol, self.interval, self.history_bars)  # type: ignore[attr-defined]
        quote = self.broker.get_quote(self.symbol)
        market = market_from_candles(history, self.symbol, quote.spread)
        n = market.bars
        r = market.returns
        vol = ewma_volatility(r)
        sigma = np.sqrt(0.94 * vol**2 + 0.06 * r**2)
        features = microstructure_features(market)

        integrity = IntegrityFilter(IntegrityConfig())
        integrity.calibrate(features[:-1])
        reading = integrity.update(features[-1])
        forecaster = AgentSwarmForecaster()
        forecaster.calibrate(market.close[:-1], r[:-1], sigma[:-1])
        for t in range(max(1_000, n - 800), n - 1):
            forecaster.observe(self._ctx(market, r, sigma, t), float(r[t + 1]))

        price = quote.mid
        copilot = Copilot(forecaster, replace(self.config, sizing=self._sizing_for(price), allow_short=False), self.symbol)
        now = market.timestamps[-1] + (market.timestamps[-1] - market.timestamps[-2])
        gate = MacroCalendarGate(self.calendar).evaluate(now)
        balances = self.broker.balances()
        rules = self.broker.rules(self.symbol)
        held = balances.get(rules.base_asset, 0.0)
        equity = balances.get(rules.quote_asset, 0.0) + held * price
        if equity <= 0:
            raise RuntimeError("équité nulle : compte vide")
        self._peak = max(self._peak, equity)
        ctx = self._ctx(market, r, sigma, n - 1)
        avg_volume = float(market.volume[-96:].mean())
        decision = copilot.decide(
            ctx, MarketQuote(price, quote.spread, avg_volume), reading, gate, equity, self._peak, self.rng
        )
        record = dict(decision.record)
        record["broker_mode"] = getattr(self.broker, "mode", "?")
        record["equity"] = round(equity, 2)

        target = decision.target_units if decision.action == Action.LONG else 0.0
        delta = rules.floor_qty(abs(target - held)) * (1 if target >= held else -1)
        order_info = None
        if abs(delta) >= rules.min_qty and abs(delta) * price >= rules.min_notional:
            side = Side.BUY if delta > 0 else Side.SELL
            request = OrderRequest(
                symbol=self.symbol,
                side=side,
                quantity=abs(delta),
                order_type="MARKET",
                stop_loss=decision.stop_price if side == Side.BUY else None,
                client_id=f"aiotrade-{len(self.ledger)}",
            )
            if self.send_orders:
                if side == Side.SELL or decision.action != Action.LONG:
                    cancel = getattr(self.broker, "cancel_all", None)
                    if callable(cancel):
                        cancel(self.symbol)  # retire le stop de la position avant de la réduire
                order_info = self.broker.place_order(request).to_dict()
            else:
                order_info = {"intended": True, "side": side.value, "quantity": abs(delta), "sent": False}
        record["order"] = order_info
        entry = self.ledger.append(record)
        return LiveStep(entry, order_info, equity, price)

    def _ctx(self, market: MarketData, r: np.ndarray, sigma: np.ndarray, t: int) -> MarketContext:
        lo = max(0, t + 1 - 2_048)
        now = market.timestamps[t]
        return MarketContext(
            timestamps=market.timestamps[lo : t + 1],
            open=market.open[lo : t + 1],
            high=market.high[lo : t + 1],
            low=market.low[lo : t + 1],
            close=market.close[lo : t + 1],
            volume=market.volume[lo : t + 1],
            returns=r[lo : t + 1],
            sigma=float(sigma[t]),
            bar_minutes=market.bar_minutes,
            upcoming_events=self.calendar.events_between(now, now + (market.timestamps[1] - market.timestamps[0]) * 16),
        )

    def run_forever(self, max_steps: int | None = None) -> None:
        minutes = INTERVAL_MINUTES[self.interval]
        steps = 0
        while max_steps is None or steps < max_steps:
            try:
                step = self.step()
            except (BrokerError, OSError) as exc:
                # Panne réseau ou refus du courtier : aucune décision, nouvel essai à la
                # prochaine clôture. Le stop déjà posé chez le courtier protège la position.
                self.log(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} ERREUR courtier : {exc}")
            else:
                rec = step.record
                self.log(
                    f"{rec['timestamp_utc']} {rec['decision']:<5} alloc={rec['selected_allocation']:+.4f} "
                    f"d2={rec['market_integrity_d2']} gate={rec['macro_gate']} ordre={step.order}"
                )
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            now = datetime.now(timezone.utc).timestamp()
            period = minutes * 60
            time.sleep(period - now % period + 5)  # juste après la clôture de la prochaine barre
