"""Connecteur Binance Spot : testnet par défaut, live verrouillé.

Sécurité :
- clés lues uniquement dans des variables d'environnement, jamais écrites ;
- environnement ``testnet`` (https://testnet.binance.vision) par défaut ;
- ``dry_run`` par défaut : les ordres passent par ``/api/v3/order/test``,
  validés par la plateforme mais jamais exécutés ;
- le live exige des clés distinctes, la variable
  ``AIOTRADE_LIVE_TRADING=JE_COMPRENDS_LES_RISQUES`` et un plafond de
  notionnel par ordre ``AIOTRADE_MAX_ORDER_NOTIONAL`` ;
- comptant uniquement : aucune vente à découvert, un signal SHORT ramène la
  position à zéro ;
- le stop loss borné par le SCG est posé côté plateforme (STOP_LOSS_LIMIT).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

from .base import BrokerError, Candle, OrderRequest, OrderResult, Quote, Side, SymbolRules

TESTNET_URL = "https://testnet.binance.vision"
LIVE_URL = "https://api.binance.com"
PUBLIC_DATA_URL = "https://data-api.binance.vision"  # données de marché publiques, sans clé
LIVE_CONFIRMATION = "JE_COMPRENDS_LES_RISQUES"
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_transport(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL contrôlée)
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise BrokerError(f"connexion impossible à {urllib.parse.urlsplit(url).netloc} : {exc.reason}") from exc


class BinanceSpot:
    name = "binance-spot"

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        environment: str = "testnet",
        dry_run: bool = True,
        max_order_notional: float = 1_000.0,
        base_url: str | None = None,
        live_confirmation: str | None = None,
        recv_window: int = 5_000,
        timeout: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        if environment not in {"testnet", "live", "public"}:
            raise ValueError("environment doit valoir testnet, live ou public")
        if environment == "live" and live_confirmation != LIVE_CONFIRMATION:
            raise BrokerError(f"live refusé : AIOTRADE_LIVE_TRADING doit valoir {LIVE_CONFIRMATION}")
        if max_order_notional <= 0:
            raise ValueError("max_order_notional doit être positif")
        default = {"testnet": TESTNET_URL, "live": LIVE_URL, "public": PUBLIC_DATA_URL}[environment]
        if base_url and base_url.rstrip("/") != default:
            host = urllib.parse.urlsplit(base_url).hostname or ""
            if host not in _LOCAL_HOSTS:
                raise ValueError("base_url personnalisée autorisée uniquement vers localhost (tests)")
        self.base_url = (base_url or default).rstrip("/")
        self.environment = environment
        self.dry_run = dry_run
        self.max_order_notional = max_order_notional
        self._key = api_key
        self._secret = api_secret.encode() if api_secret else None
        self.recv_window = recv_window
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self._offset_ms = 0
        self._rules: dict[str, SymbolRules] = {}

    # ----------------------------------------------------------- construction
    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, transport: Transport | None = None) -> "BinanceSpot":
        """Construit le connecteur depuis l'environnement (aucune clé en argument de ligne de commande)."""
        env = dict(os.environ if env is None else env)
        environment = env.get("AIOTRADE_BINANCE_ENV", "testnet")
        send = env.get("AIOTRADE_SEND_ORDERS", "0") == "1"
        if environment == "live":
            key, secret = env.get("AIOTRADE_BINANCE_LIVE_API_KEY"), env.get("AIOTRADE_BINANCE_LIVE_API_SECRET")
            cap = env.get("AIOTRADE_MAX_ORDER_NOTIONAL")
            if not cap:
                raise BrokerError("live refusé : définissez AIOTRADE_MAX_ORDER_NOTIONAL (plafond par ordre)")
            return cls(key, secret, "live", dry_run=not send, max_order_notional=float(cap),
                       live_confirmation=env.get("AIOTRADE_LIVE_TRADING"), base_url=env.get("AIOTRADE_BINANCE_BASE_URL"),
                       transport=transport)
        key, secret = env.get("AIOTRADE_BINANCE_API_KEY"), env.get("AIOTRADE_BINANCE_API_SECRET")
        cap = float(env.get("AIOTRADE_MAX_ORDER_NOTIONAL", "1000"))
        return cls(key, secret, environment, dry_run=not send, max_order_notional=cap,
                   base_url=env.get("AIOTRADE_BINANCE_BASE_URL"), transport=transport)

    @property
    def mode(self) -> str:
        return f"{self.environment}-dry-run" if self.dry_run else self.environment

    @property
    def has_credentials(self) -> bool:
        return bool(self._key and self._secret)

    # ----------------------------------------------------------- transport
    def _request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        headers = {"Accept": "application/json", "User-Agent": "aiotrade/0.2"}
        if signed:
            if not self.has_credentials:
                raise BrokerError("clés API absentes : définissez AIOTRADE_BINANCE_API_KEY et AIOTRADE_BINANCE_API_SECRET")
            params["timestamp"] = int(time.time() * 1000) + self._offset_ms
            params["recvWindow"] = self.recv_window
            query = urllib.parse.urlencode(params)
            signature = hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()  # type: ignore[arg-type]
            query += f"&signature={signature}"
            headers["X-MBX-APIKEY"] = self._key  # type: ignore[assignment]
        else:
            query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}"
        body = None
        if method == "GET" or method == "DELETE":
            url += f"?{query}" if query else ""
        else:
            body = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        status, raw = self._transport(method, url, headers, body, self.timeout)
        try:
            data = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError as exc:
            raise BrokerError(f"réponse non JSON (HTTP {status})") from exc
        if status >= 400 or (isinstance(data, dict) and isinstance(data.get("code"), int) and data["code"] < 0):
            code = data.get("code") if isinstance(data, dict) else None
            msg = data.get("msg", "erreur") if isinstance(data, dict) else "erreur"
            raise BrokerError(f"Binance HTTP {status} : {msg}", code)
        return data

    # ----------------------------------------------------------- marché
    def ping(self) -> bool:
        self._request("GET", "/api/v3/ping")
        return True

    def sync_time(self) -> int:
        server = int(self._request("GET", "/api/v3/time")["serverTime"])
        self._offset_ms = server - int(time.time() * 1000)
        return self._offset_ms

    def rules(self, symbol: str) -> SymbolRules:
        if symbol in self._rules:
            return self._rules[symbol]
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})
        sym = info["symbols"][0]
        filters = {f["filterType"]: f for f in sym["filters"]}
        lot, price = filters["LOT_SIZE"], filters["PRICE_FILTER"]
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {"minNotional": "0"}
        rules = SymbolRules(
            symbol=symbol,
            base_asset=sym["baseAsset"],
            quote_asset=sym["quoteAsset"],
            step_size=float(lot["stepSize"]),
            min_qty=float(lot["minQty"]),
            max_qty=float(lot["maxQty"]),
            tick_size=float(price["tickSize"]),
            min_notional=float(notional["minNotional"]),
        )
        self._rules[symbol] = rules
        return rules

    def get_quote(self, symbol: str) -> Quote:
        d = self._request("GET", "/api/v3/ticker/bookTicker", {"symbol": symbol})
        return Quote(symbol, float(d["bidPrice"]), float(d["askPrice"]))

    def get_candles(self, symbol: str, interval: str = "15m", limit: int = 1_000, end_time: int | None = None) -> list[Candle]:
        rows = self._request(
            "GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": min(limit, 1_000), "endTime": end_time}
        )
        return [
            Candle(
                open_time=datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc),
                open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
                volume=float(r[5]), quote_volume=float(r[7]), trades=int(r[8]),
            )
            for r in rows
        ]

    def history(self, symbol: str, interval: str, bars: int) -> list[Candle]:
        """Historique de ``bars`` chandeliers clôturés (pagination par 1 000)."""
        out: list[Candle] = []
        end: int | None = None
        while len(out) < bars:
            page = self.get_candles(symbol, interval, min(1_000, bars - len(out) + 1), end_time=end)
            if not page:
                break
            if out:
                page = [c for c in page if c.open_time < out[0].open_time]
            if not page:
                break
            out = page + out
            end = int(page[0].open_time.timestamp() * 1000) - 1
        now = datetime.now(timezone.utc)
        step = out[1].open_time - out[0].open_time if len(out) > 1 else None
        if step and out and out[-1].open_time + step > now:
            out = out[:-1]  # la dernière barre n'est pas encore clôturée
        return out[-bars:]

    # ----------------------------------------------------------- compte
    def balances(self) -> dict[str, float]:
        data = self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)
        return {b["asset"]: float(b["free"]) for b in data.get("balances", [])}

    def cancel_all(self, symbol: str) -> int:
        """Coupe-circuit : annule tous les ordres ouverts du symbole."""
        if self.dry_run:
            return 0
        try:
            data = self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        except BrokerError as exc:
            if exc.code == -2011:  # aucun ordre ouvert
                return 0
            raise
        return len(data) if isinstance(data, list) else 0

    # ----------------------------------------------------------- ordres
    def place_order(self, request: OrderRequest) -> OrderResult:
        rules = self.rules(request.symbol)
        qty = rules.floor_qty(request.quantity)
        quote = self.get_quote(request.symbol)
        ref_price = request.limit_price or (quote.ask if request.side == Side.BUY else quote.bid)
        if qty < rules.min_qty or qty * ref_price < rules.min_notional:
            return OrderResult(False, self.mode, "REJECTED", message="quantité sous le minimum de l'instrument")
        if qty * ref_price > self.max_order_notional:
            qty = rules.floor_qty(self.max_order_notional / ref_price)
            if qty < rules.min_qty or qty * ref_price < rules.min_notional:
                return OrderResult(False, self.mode, "REJECTED", message="plafond de notionnel trop bas pour l'instrument")
        if request.side == Side.SELL and self.has_credentials:
            held = self.balances().get(rules.base_asset, 0.0)
            if qty > held + 1e-12:
                qty = rules.floor_qty(held)
                if qty < rules.min_qty:
                    return OrderResult(False, self.mode, "REJECTED", message="aucune position à vendre (comptant)")
        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.value,
            "quantity": f"{qty:.{_decimals_of(rules.step_size)}f}",
            "newOrderRespType": "FULL",
        }
        if request.client_id:
            params["newClientOrderId"] = request.client_id[:36]
        if request.order_type == "LIMIT_MAKER":
            if request.limit_price is None:
                raise ValueError("limit_price requis pour un ordre maker")
            params["type"] = "LIMIT_MAKER"
            params["price"] = f"{rules.round_price(request.limit_price, down=request.side == Side.BUY):.{_decimals_of(rules.tick_size)}f}"
        else:
            params["type"] = "MARKET"
        path = "/api/v3/order/test" if self.dry_run else "/api/v3/order"
        data = self._request("POST", path, params, signed=True)
        if self.dry_run:
            result = OrderResult(True, self.mode, "VALIDATED", filled_qty=0.0, avg_price=None,
                                 message="ordre validé par la plateforme, non exécuté (dry run)")
        else:
            fills = data.get("fills", [])
            filled = float(data.get("executedQty", 0.0))
            notional = sum(float(f["price"]) * float(f["qty"]) for f in fills)
            avg = notional / filled if filled else None
            result = OrderResult(True, self.mode, data.get("status", "?"), order_id=str(data.get("orderId")),
                                 filled_qty=filled, avg_price=avg, raw=data)
        if request.side == Side.BUY and request.stop_loss is not None:
            stop_qty = result.filled_qty if not self.dry_run else qty
            if stop_qty >= rules.min_qty:
                result.stop_order_id = self._place_stop(request.symbol, stop_qty, request.stop_loss, rules)
        return result

    def _place_stop(self, symbol: str, qty: float, stop_price: float, rules: SymbolRules) -> str | None:
        stop = rules.round_price(stop_price, down=True)
        limit = rules.round_price(stop * (1 - 0.002), down=True)  # marge de 20 pb pour garantir la sortie
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "STOP_LOSS_LIMIT",
            "timeInForce": "GTC",
            "quantity": f"{rules.floor_qty(qty):.{_decimals_of(rules.step_size)}f}",
            "stopPrice": f"{stop:.{_decimals_of(rules.tick_size)}f}",
            "price": f"{limit:.{_decimals_of(rules.tick_size)}f}",
        }
        path = "/api/v3/order/test" if self.dry_run else "/api/v3/order"
        data = self._request("POST", path, params, signed=True)
        return None if self.dry_run else str(data.get("orderId"))


def _decimals_of(step: float) -> int:
    text = f"{step:.12f}".rstrip("0")
    return len(text.split(".")[1]) if "." in text else 0
