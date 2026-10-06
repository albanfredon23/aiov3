"""Courtier papier : ordres simulés sur des cotations réelles ou synthétiques, aucun ordre envoyé."""
from __future__ import annotations

import json
from pathlib import Path

from ..execution import ExecutionModel
from .base import BrokerError, Candle, OrderRequest, OrderResult, Quote, Side, SymbolRules


class PaperBroker:
    """Portefeuille simulé (comptant, sans vente à découvert), état éventuellement persisté en JSON.

    ``data`` fournit les cotations et chandeliers (par exemple les données de
    marché publiques de Binance) ; les exécutions sont simulées localement avec
    le modèle de coûts du moteur.
    """

    name = "paper"
    mode = "paper"

    def __init__(
        self,
        data: object,
        quote_asset: str = "USDT",
        initial_cash: float = 10_000.0,
        state_path: str | Path | None = None,
        execution: ExecutionModel | None = None,
    ) -> None:
        self.data = data
        self.quote_asset = quote_asset
        self.state_path = Path(state_path) if state_path else None
        self.execution = execution or ExecutionModel()
        self._balances = {quote_asset: float(initial_cash)}
        if self.state_path and self.state_path.exists():
            self._balances = {k: float(v) for k, v in json.loads(self.state_path.read_text())["balances"].items()}

    def get_quote(self, symbol: str) -> Quote:
        return self.data.get_quote(symbol)  # type: ignore[attr-defined]

    def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        return self.data.get_candles(symbol, interval, limit)  # type: ignore[attr-defined]

    def history(self, symbol: str, interval: str, bars: int) -> list[Candle]:
        return self.data.history(symbol, interval, bars)  # type: ignore[attr-defined]

    def rules(self, symbol: str) -> SymbolRules:
        return self.data.rules(symbol)  # type: ignore[attr-defined]

    def balances(self) -> dict[str, float]:
        return dict(self._balances)

    def place_order(self, request: OrderRequest) -> OrderResult:
        rules = self.rules(request.symbol)
        quote = self.get_quote(request.symbol)
        qty = rules.floor_qty(request.quantity)
        if qty < rules.min_qty or qty * quote.mid < rules.min_notional:
            return OrderResult(False, self.mode, "REJECTED", message="quantité sous le minimum de l'instrument")
        base = self._balances.get(rules.base_asset, 0.0)
        cash = self._balances.get(rules.quote_asset, 0.0)
        if request.side == Side.SELL and qty > base + 1e-12:
            raise BrokerError("vente supérieure à la position détenue (pas de vente à découvert au comptant)")
        maker = request.order_type == "LIMIT_MAKER"
        price = request.limit_price if maker and request.limit_price else (quote.ask if request.side == Side.BUY else quote.bid)
        cfg = self.execution.config
        # Le prix taker est déjà l'ask ou le bid (spread inclus) : restent commission et slippage.
        cost = cfg.maker_commission_bps / 1e4 if maker else (cfg.taker_commission_bps + cfg.slippage_bps) / 1e4
        notional = qty * price
        if request.side == Side.BUY:
            if notional * (1 + cost) > cash + 1e-9:
                return OrderResult(False, self.mode, "REJECTED", message="liquidités insuffisantes")
            self._balances[rules.quote_asset] = cash - notional * (1 + cost)
            self._balances[rules.base_asset] = base + qty
        else:
            self._balances[rules.quote_asset] = cash + notional * (1 - cost)
            self._balances[rules.base_asset] = base - qty
        self._save()
        return OrderResult(True, self.mode, "FILLED", order_id=None, filled_qty=qty, avg_price=price,
                           message="exécution simulée (papier)")

    def _save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"balances": self._balances}, indent=2))
            tmp.replace(self.state_path)
