import pytest
from fastapi.testclient import TestClient

from aiotrade.api import DEMO_TEST_BARS, REPORT_PATH, app
from aiotrade.ledger import XAILedger

client = TestClient(app)


def test_sante_et_configuration():
    assert client.get("/api/health").json() == {"status": "ok", "version": "0.2.0", "mode": "simulation"}
    cfg = client.get("/api/config").json()
    assert cfg["mandate"]["min_admissibility"] == 0.75
    assert cfg["mandate"]["max_leverage"] == 1.5
    assert cfg["sizing"]["kelly_fraction"] == 0.2
    assert cfg["macro_gate"]["blackout_minutes_before"] == 15
    assert set(cfg["arms"]) == {"A", "B", "C", "D"}


@pytest.fixture(scope="module")
def simulation():
    r = client.post("/api/simulate", json={"seed": 7, "shock": "flash_crash"})
    assert r.status_code == 200
    return r.json()


def test_simulation_complete(simulation):
    s = simulation["series"]
    assert len(s["time"]) == len(s["price"]) == len(s["d2"]) == DEMO_TEST_BARS
    assert set(s["equity"]) == {"A", "B", "C", "D"}
    assert all(len(v) == DEMO_TEST_BARS for v in s["equity"].values())
    a, b = simulation["shock_window"]
    assert "FREEZE" in s["status"][a : b + 1]
    assert simulation["ledger"]["verified"] is True
    assert simulation["ledger"]["records"] >= DEMO_TEST_BARS // 4
    assert simulation["thresholds"]["alert"] >= 13.28
    breakdown = simulation["ledger"]["breakdown"]
    assert sum(breakdown.values()) == simulation["ledger"]["records"]
    assert breakdown["freeze"] > 0 and breakdown["other"] == 0


def test_parametres_valides():
    assert client.post("/api/simulate", json={"seed": -1}).status_code == 422
    assert client.post("/api/simulate", json={"mandate": {"kelly_fraction": 0.5}}).status_code == 422
    assert client.post("/api/simulate", json={"shock": "inconnu"}).status_code == 422


def test_verification_scg():
    r = client.post("/api/scg/check", json={"direction": "LONG", "exposure": 3.0})
    body = r.json()
    assert r.status_code == 200
    assert body["rejections"]["leverage_violation"] == body["trajectories"]
    assert body["admitted"] is False
    small = client.post("/api/scg/check", json={"direction": "SHORT", "exposure": 0.1}).json()
    assert small["admissibility_ratio"] >= body["admissibility_ratio"]
    assert len(small["equity_quantiles"]["p50"]) == small["horizon_bars"] + 1


def test_verification_du_registre():
    ledger = XAILedger()
    for i in range(3):
        ledger.append({"status": "NOMINAL", "selected_allocation": i / 10})
    text = ledger.export_jsonl()
    assert client.post("/api/ledger/verify", json={"jsonl": text}).json()["ok"] is True
    tampered = text.replace('"selected_allocation":0.1', '"selected_allocation":0.9')
    res = client.post("/api/ledger/verify", json={"jsonl": tampered}).json()
    assert res["ok"] is False and res["first_invalid_seq"] == 1
    assert client.post("/api/ledger/verify", json={"jsonl": "{pas du json"}).status_code == 422


def test_rapport_de_benchmark():
    r = client.get("/api/benchmark")
    if not REPORT_PATH.exists():
        assert r.status_code == 404
        return
    body = r.json()
    assert set(body["aggregate"]) == {"A", "B", "C", "D"}
    assert body["protocol"]["calibration_months"] == 12 and body["protocol"]["test_months"] == 3
    assert all(entry["verified"] for entry in body["ledger"])


def test_contact_exige_le_consentement():
    base = {"name": "Alice", "email": "alice@example.com"}
    assert client.post("/api/contact", json={**base, "consent": False}).status_code == 422
    r = client.post("/api/contact", json={**base, "consent": True})
    assert r.status_code == 202 and r.json()["stored"] is False
