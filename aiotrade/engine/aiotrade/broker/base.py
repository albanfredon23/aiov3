"""Interface courtier commune : le moteur ne dépend d'aucun courtier particulier.

L'API xAPI de XTB, citée dans la spécification, a été fermée par XTB le
14 mars 2025 : elle ne peut plus être utilisée. Le protocole ci-dessous permet
de brancher n'importe quel courtier (Binance Spot testnet fourni, Interactive
Brokers, MetaTrader…) sans toucher au pipeline de décision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class BrokerError(RuntimeError):
    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float  # en actif de base
    quote_volume: float  # en actif de cotation (notionnel)
    trades: int


@dataclass(frozen=True)
class SymbolRules:
    """Contraintes de l'instrument (pas de quantité, pas de prix, notionnel minimal)."""

    symbol: str
    base_asset: str
    quote_asset: str
    step_size: float
    min_qty: float
    max_qty: float
    tick_size: float
    min_notional: float

    def floor_qty(self, qty: float) -> float:
        steps = math.floor(qty / self.step_size + 1e-9)
        return round(steps * self.step_size, _decimals(self.step_size))

    def round_price(self, price: float, down: bool = True) -> float:
        f = math.floor if down else math.ceil
        return round(f(price / self.tick_size + (1e-9 if down else -1e-9)) * self.tick_size, _decimals(self.tick_size))


def _decimals(step: float) -> int:
    text = f"{step:.12f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: Side
    quantity: float
    order_type: str = "MARKET"  # MARKET | LIMIT_MAKER (maker, post-only)
    limit_price: float | None = None
    stop_loss: float | None = None  # stop borné par le SCG
    client_id: str | None = None


@dataclass
class OrderResult:
    accepted: bool
    mode: str  # paper | testnet-dry-run | testnet | live-dry-run | live
    status: str
    order_id: str | None = None
    filled_qty: float = 0.0
    avg_price: float | None = None
    stop_order_id: str | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mode": self.mode,
            "status": self.status,
            "order_id": self.order_id,
            "filled_qty": self.filled_qty,
            "avg_price": self.avg_price,
            "stop_order_id": self.stop_order_id,
            "message": self.message,
        }


class Broker(Protocol):
    name: str
    mode: str

    def get_quote(self, symbol: str) -> Quote: ...

    def get_candles(self, symbol: str, interval: str, limit: int) -> list[Candle]: ...

    def rules(self, symbol: str) -> SymbolRules: ...

    def balances(self) -> dict[str, float]: ...

    def place_order(self, request: OrderRequest) -> OrderResult: ...
