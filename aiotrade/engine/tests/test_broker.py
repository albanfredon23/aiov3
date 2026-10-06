import pytest

from aiotrade.broker import BinanceSpot, BrokerError, OrderRequest, PaperBroker, Side
from aiotrade.broker.binance import LIVE_CONFIRMATION

from .mock_binance import KEY, SECRET, MockBinance, serve


def client(mock, **kw):
    kw.setdefault("dry_run", True)
    return BinanceSpot(KEY, SECRET, base_url="http://127.0.0.1:9", transport=mock, **kw)


def test_donnees_de_marche_et_regles():
    mock = MockBinance()
    b = client(mock)
    assert b.ping()
    rules = b.rules("BTCUSDT")
    assert rules.step_size == 0.00001 and rules.min_notional == 5.0
    assert rules.floor_qty(0.123456789) == pytest.approx(0.12345)
    q = b.get_quote("BTCUSDT")
    assert q.ask > q.bid
    hist = b.history("BTCUSDT", "15m", 1_500)
    assert len(hist) == 1_500
    assert all(hist[i].open_time < hist[i + 1].open_time for i in range(len(hist) - 1))


def test_signature_hmac_verifiee_par_le_serveur():
    mock = MockBinance()
    assert client(mock).balances()["USDT"] == 10_000.0
    bad = BinanceSpot(KEY, "mauvais-secret", base_url="http://127.0.0.1:9", transport=mock)
    with pytest.raises(BrokerError) as err:
        bad.balances()
    assert err.value.code == -1022
    assert SECRET not in str(err.value) and "mauvais-secret" not in str(err.value)


def test_dry_run_utilise_order_test_et_pose_le_stop():
    mock = MockBinance()
    res = client(mock, max_order_notional=1e7).place_order(OrderRequest("BTCUSDT", Side.BUY, 0.0123456, stop_loss=59_000.123))
    assert res.accepted and res.status == "VALIDATED" and res.mode == "testnet-dry-run"
    orders = [p for m, path, p in mock.requests if path == "/api/v3/order/test"]
    assert orders[0]["type"] == "MARKET" and orders[0]["quantity"] == "0.01234"
    assert orders[1]["type"] == "STOP_LOSS_LIMIT" and orders[1]["stopPrice"] == "59000.12"
    assert not any(path == "/api/v3/order" for _, path, _ in mock.requests)


def test_ordre_reel_testnet_et_stop_cote_plateforme():
    mock = MockBinance()
    res = client(mock, dry_run=False).place_order(OrderRequest("BTCUSDT", Side.BUY, 0.01, stop_loss=59_000))
    assert res.status == "FILLED" and res.filled_qty == pytest.approx(0.01)
    assert res.stop_order_id is not None


def test_plafond_de_notionnel_par_ordre():
    mock = MockBinance()
    client(mock, max_order_notional=100.0).place_order(OrderRequest("BTCUSDT", Side.BUY, 1.0))
    qty = float(next(p for _, path, p in mock.requests if path == "/api/v3/order/test")["quantity"])
    price = float(mock.candles[-1][4])
    assert qty * price <= 100.0


def test_pas_de_vente_a_decouvert():
    mock = MockBinance(balances={"USDT": 1_000.0, "BTC": 0.0})
    res = client(mock).place_order(OrderRequest("BTCUSDT", Side.SELL, 0.01))
    assert not res.accepted


def test_quantite_sous_le_minimum_refusee():
    res = client(MockBinance()).place_order(OrderRequest("BTCUSDT", Side.BUY, 0.00001))
    assert not res.accepted


def test_url_de_base_limitee_au_testnet_ou_localhost():
    with pytest.raises(ValueError):
        BinanceSpot(KEY, SECRET, base_url="https://exemple.com")
    assert BinanceSpot().base_url == "https://testnet.binance.vision"


def test_live_verrouille():
    with pytest.raises(BrokerError):
        BinanceSpot(KEY, SECRET, environment="live")
    with pytest.raises(BrokerError):
        BinanceSpot.from_env({"AIOTRADE_BINANCE_ENV": "live", "AIOTRADE_LIVE_TRADING": LIVE_CONFIRMATION})
    live = BinanceSpot.from_env({
        "AIOTRADE_BINANCE_ENV": "live", "AIOTRADE_LIVE_TRADING": LIVE_CONFIRMATION,
        "AIOTRADE_MAX_ORDER_NOTIONAL": "50", "AIOTRADE_BINANCE_LIVE_API_KEY": "k", "AIOTRADE_BINANCE_LIVE_API_SECRET": "s",
    })
    assert live.base_url == "https://api.binance.com" and live.dry_run and live.max_order_notional == 50.0


def test_cles_depuis_l_environnement():
    b = BinanceSpot.from_env({"AIOTRADE_BINANCE_API_KEY": KEY, "AIOTRADE_BINANCE_API_SECRET": SECRET})
    assert b.has_credentials and b.environment == "testnet" and b.dry_run
    assert not BinanceSpot.from_env({}).has_credentials


def test_vrai_serveur_http_local():
    mock = MockBinance()
    server = serve(mock)
    try:
        b = BinanceSpot(KEY, SECRET, base_url=f"http://127.0.0.1:{server.server_port}")
        assert b.balances()["USDT"] == 10_000.0
        assert b.place_order(OrderRequest("BTCUSDT", Side.BUY, 0.001)).status == "VALIDATED"
    finally:
        server.shutdown()


def test_courtier_papier():
    data = client(MockBinance())
    paper = PaperBroker(data, initial_cash=1_000.0)
    res = paper.place_order(OrderRequest("BTCUSDT", Side.BUY, 0.01))
    assert res.accepted and paper.balances()["BTC"] == pytest.approx(0.01)
    assert paper.balances()["USDT"] < 1_000.0 - 0.01 * res.avg_price + 1e-9
    with pytest.raises(BrokerError):
        paper.place_order(OrderRequest("BTCUSDT", Side.SELL, 0.02))
    assert paper.place_order(OrderRequest("BTCUSDT", Side.SELL, 0.01)).accepted
