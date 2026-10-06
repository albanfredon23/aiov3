from aiotrade.broker import BinanceSpot, PaperBroker
from aiotrade.ledger import XAILedger, verify_chain
from aiotrade.live import LiveRunner

from .mock_binance import KEY, SECRET, MockBinance


def test_pas_de_decision_live_sans_historique_suffisant():
    import pytest

    with pytest.raises(ValueError):
        LiveRunner(BinanceSpot(), history_bars=500)


def test_decision_testnet_journalisee(tmp_path):
    mock = MockBinance()
    broker = BinanceSpot(KEY, SECRET, base_url="http://127.0.0.1:9", transport=mock, dry_run=True)
    path = tmp_path / "decisions.jsonl"
    runner = LiveRunner(broker, history_bars=1_600, ledger_path=path, send_orders=True, log=lambda m: None)
    runner.run_forever(max_steps=1)
    rec = XAILedger.read_jsonl(path)[-1]
    assert rec["broker_mode"] == "testnet-dry-run"
    assert rec["decision"] in ("LONG", "CASH")  # comptant : jamais SHORT
    assert rec["equity"] > 0
    assert verify_chain(XAILedger.read_jsonl(path)).ok
    assert not any(p == "/api/v3/order" for _, p, _ in mock.requests)  # rien d'exécuté


def test_mode_papier(tmp_path):
    data = BinanceSpot(base_url="http://127.0.0.1:9", transport=MockBinance(), environment="public")
    paper = PaperBroker(data, initial_cash=5_000.0, state_path=tmp_path / "state.json")
    runner = LiveRunner(paper, history_bars=1_600, ledger_path=tmp_path / "d.jsonl", send_orders=True)
    step = runner.step()
    assert step.record["broker_mode"] == "paper"
    assert step.equity == 5_000.0


def test_panne_courtier_journalisee_sans_arret(tmp_path):
    def down(method, url, headers, body, timeout):
        return 403, b"<html>forbidden</html>"

    broker = BinanceSpot(base_url="http://127.0.0.1:9", transport=down, environment="public")
    paper = PaperBroker(broker, state_path=tmp_path / "state.json")
    logs: list[str] = []
    runner = LiveRunner(paper, history_bars=1_600, ledger_path=tmp_path / "d.jsonl", log=logs.append)
    runner.run_forever(max_steps=1)
    assert len(logs) == 1 and "ERREUR courtier" in logs[0]
    assert not (tmp_path / "d.jsonl").exists() or XAILedger.read_jsonl(tmp_path / "d.jsonl") == []


def test_cli_erreur_courtier_code_2(monkeypatch, capsys):
    from aiotrade import __main__ as cli
    from aiotrade.broker.base import BrokerError

    def boom(args):
        raise BrokerError("réponse non JSON (HTTP 403)")

    monkeypatch.setattr(cli, "_broker_check", boom)
    assert cli.main(["broker-check"]) == 2
    assert "erreur courtier" in capsys.readouterr().err
