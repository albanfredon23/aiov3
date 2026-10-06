import numpy as np
import pytest

from aiotrade.copilot import simulate
from aiotrade.market import MarketConfig, Shock


@pytest.fixture(scope="module")
def crash_result():
    return simulate(MarketConfig(steps=500, seed=7, shock=Shock.FLASH_CRASH, shock_at=420))


def test_simulation_produces_consistent_series(crash_result):
    r = crash_result
    n = r.equity.size
    assert r.benchmark.size == n and r.weights.shape == (n, 4) and r.safe_mode.size == n
    assert np.all(r.equity > 0)


def test_portfolio_is_flat_during_safe_mode(crash_result):
    r = crash_result
    assert r.safe_mode.any()
    assert np.allclose(r.weights[r.safe_mode], 0.0)
    assert any(e.kind == "safe_mode_on" for e in r.events)


def test_guarded_portfolio_beats_benchmark_drawdown_in_crashes():
    for seed in range(4):
        r = simulate(MarketConfig(steps=500, seed=seed, shock=Shock.FLASH_CRASH))
        s = r.summary()
        assert s["max_drawdown"] < s["benchmark_max_drawdown"]
        assert s["max_drawdown"] < 0.12  # limite 10 % + risque de gap d'un pas


def test_every_executed_allocation_was_approved(crash_result):
    for event in crash_result.events:
        if event.kind == "rebalance":
            assert "weights" in event.details
    rejected = [e for e in crash_result.events if e.kind == "rejected"]
    assert all(e.details["violations"] for e in rejected)


def test_serialisation_is_bounded(crash_result):
    data = crash_result.to_dict(max_points=100)
    assert len(data["series"]["equity"]) <= 100
    assert set(data["summary"]) >= {"final_return", "max_drawdown", "rejected_allocations"}
    assert data["last_scenarios"] and "guard" in data["last_scenarios"][0]
