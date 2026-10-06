from fastapi.testclient import TestClient

from aiotrade.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["mode"] == "simulation"


def test_limits():
    data = client.get("/api/limits").json()
    assert data["max_drawdown"] == 0.10
    assert "flash_crash" in data["shocks"]


def test_simulate():
    r = client.post("/api/simulate", json={"seed": 3, "steps": 400, "shock": "liquidity_drop"})
    assert r.status_code == 200
    data = r.json()
    assert data["shock"] == "liquidity_drop"
    assert len(data["series"]["equity"]) == len(data["series"]["step"])


def test_simulate_validates_input():
    assert client.post("/api/simulate", json={"steps": 10}).status_code == 422
    assert client.post("/api/simulate", json={"shock": "inconnu"}).status_code == 422


def test_guard_check():
    ok = client.post("/api/guard/check", json={"weights": {"OBLIG_10A": 0.3}}).json()
    assert ok["approved"] is True
    ko = client.post("/api/guard/check", json={"weights": {"TECH_US": 1.5}}).json()
    assert ko["approved"] is False and ko["violations"]
    assert client.post("/api/guard/check", json={"weights": {"BTC": 1}}).status_code == 422


def test_contact_requires_consent():
    base = {"name": "Test", "email": "test@example.com"}
    assert client.post("/api/contact", json={**base, "consent": False}).status_code == 422
    assert client.post("/api/contact", json={**base, "email": "pas-un-email", "consent": True}).status_code == 422
    r = client.post("/api/contact", json={**base, "consent": True})
    assert r.status_code == 202 and r.json()["stored"] is False


def test_simulate_minimum_length_is_supported():
    r = client.post("/api/simulate", json={"steps": 300, "shock": "none"})
    assert r.status_code == 200
