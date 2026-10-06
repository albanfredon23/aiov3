import numpy as np
import pytest

from aiotrade.market import MarketConfig, generate_market
from aiotrade.tap import TAPConfig, TAPEngine


@pytest.fixture
def history():
    return generate_market(MarketConfig(steps=300, seed=4)).returns


def test_tap_proposes_four_scenarios_with_path_bundles(history):
    cfg = TAPConfig(n_paths=128, horizon=10)
    scenarios = TAPEngine(cfg, seed=1).propose(history)
    assert [s.name for s in scenarios] == ["trend", "mean_reversion", "hedge", "cash"]
    for sc in scenarios:
        assert sc.paths.shape == (128, 11)
        assert np.allclose(sc.paths[:, 0], 1.0)
        assert np.abs(sc.weights).sum() <= cfg.max_gross + 1e-9
        assert sc.p05 <= sc.p50 <= sc.p95


def test_trend_is_long_only_and_hedge_is_reduced(history):
    scenarios = {s.name: s for s in TAPEngine(seed=2).propose(history)}
    assert np.all(scenarios["trend"].weights >= 0)
    assert np.abs(scenarios["hedge"].weights).sum() == pytest.approx(TAPConfig().hedge_gross)
    assert np.all(scenarios["hedge"].weights >= 0)


def test_cash_scenario_is_riskless(history):
    cash = TAPEngine(seed=3).propose(history)[-1]
    assert np.allclose(cash.paths, 1.0)
    assert cash.score == 0.0 and cash.worst_path_drawdown == 0.0


def test_scenario_serialisation(history):
    sc = TAPEngine(seed=4).propose(history)[0]
    data = sc.to_dict(("A", "B", "C", "D"), max_paths=5)
    assert set(data["weights"]) == {"A", "B", "C", "D"}
    assert len(data["sample_paths"]) == 5


def test_rejects_short_history():
    with pytest.raises(ValueError):
        TAPEngine().propose(np.zeros((5, 4)))
