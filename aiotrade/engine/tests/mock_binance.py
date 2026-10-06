"""Faux serveur Binance Spot pour les tests (aucun accès réseau)."""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import urllib.parse
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aiotrade.market import MarketConfig, generate_market

KEY, SECRET = "cle-de-test", "secret-de-test"


class MockBinance:
    def __init__(self, bars: int = 2_200, balances: dict[str, float] | None = None) -> None:
        m = generate_market(MarketConfig(bars=bars, seed=31, price0=60_000.0, annual_vol=0.5))
        self.candles = []
        for i, t in enumerate(m.timestamps):
            ms = int(t.timestamp() * 1000)
            self.candles.append([ms, f"{m.open[i]:.2f}", f"{m.high[i]:.2f}", f"{m.low[i]:.2f}", f"{m.close[i]:.2f}",
                                 f"{m.volume[i] / m.close[i]:.5f}", ms + 899_999, f"{m.volume[i]:.2f}", 500 + i % 50,
                                 "0", "0", "0"])
        self.step = timedelta(minutes=15)
        self.balances = balances if balances is not None else {"USDT": 10_000.0, "BTC": 0.0}
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.next_id = 1

    # Transport injectable : (méthode, url, en-têtes, corps, délai) -> (statut, octets)
    def __call__(self, method, url, headers, body, timeout):
        parts = urllib.parse.urlsplit(url)
        raw = parts.query if method in ("GET", "DELETE") else (body or b"").decode()
        params = dict(urllib.parse.parse_qsl(raw))
        self.requests.append((method, parts.path, params))
        if "signature" in params:
            payload, _, sig = raw.rpartition("&signature=")
            expected = hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
            lower = {k.lower(): v for k, v in headers.items()}
            if lower.get("x-mbx-apikey") != KEY or sig != expected:
                return 401, json.dumps({"code": -1022, "msg": "Signature for this request is not valid."}).encode()
        status, data = self.route(method, parts.path, params)
        return status, json.dumps(data).encode()

    def route(self, method, path, p):
        if path == "/api/v3/ping":
            return 200, {}
        if path == "/api/v3/time":
            return 200, {"serverTime": 1_760_000_000_000}
        if path == "/api/v3/exchangeInfo":
            return 200, {"symbols": [{
                "symbol": p["symbol"], "baseAsset": "BTC", "quoteAsset": "USDT",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
                    {"filterType": "LOT_SIZE", "minQty": "0.00001000", "maxQty": "9000.00000000", "stepSize": "0.00001000"},
                    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
                ],
            }]}
        if path == "/api/v3/ticker/bookTicker":
            last = float(self.candles[-1][4])
            return 200, {"symbol": p["symbol"], "bidPrice": f"{last - 0.5:.2f}", "askPrice": f"{last + 0.5:.2f}"}
        if path == "/api/v3/klines":
            rows = self.candles
            if "endTime" in p:
                rows = [c for c in rows if c[0] <= int(p["endTime"])]
            return 200, rows[-int(p.get("limit", 500)):]
        if path == "/api/v3/account":
            return 200, {"balances": [{"asset": k, "free": str(v), "locked": "0"} for k, v in self.balances.items()]}
        if path == "/api/v3/order/test":
            return 200, {}
        if path == "/api/v3/order":
            oid = self.next_id
            self.next_id += 1
            if p["type"] == "MARKET":
                price = float(self.candles[-1][4])
                return 200, {"orderId": oid, "status": "FILLED", "executedQty": p["quantity"],
                             "fills": [{"price": f"{price:.2f}", "qty": p["quantity"]}]}
            if p["type"] == "STOP_LOSS_LIMIT":
                return 200, {"orderId": oid, "status": "NEW"}
            return 400, {"code": -2010, "msg": "Order would immediately match and take."}
        if path == "/api/v3/openOrders" and method == "DELETE":
            return 400, {"code": -2011, "msg": "Unknown order sent."}
        return 404, {"code": -1, "msg": "not found"}


def serve(mock: MockBinance):
    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            status, raw = mock(self.command, f"http://local{self.path}", dict(self.headers), body, 5)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = do_POST = do_DELETE = _handle

        def log_message(self, *args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
