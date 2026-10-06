import numpy as np
import pytest

from aiotrade.market import MarketConfig, generate_market
from aiotrade.scg import RiskLimits, SCGGuard


@pytest.fixture
def history():
    return generate_market(MarketConfig(steps=300, seed=5)).returns[-120:]


def test_moderate_allocation_is_approved(history):
    guard = SCGGuard()
    decision = guard.check_allocation(np.array([0.1, 0.05, 0.3, 0.1]), history, 1e6, 1e6)
    assert decision.approved, decision.violations
    assert decision.metrics["var"] < guard.limits.var_budget


def test_var_budget_violation(history):
    decision = SCGGuard().check_allocation(np.array([0.0, 1.0, 0.0, 0.0]), history, 1e6, 1e6)
    assert not decision.approved
    assert "var_budget" in {v.rule for v in decision.violations}


def test_margin_violation(history):
    limits = RiskLimits(var_budget=0.5, max_drawdown=0.9, max_margin_usage=0.05)
    decision = SCGGuard(limits).check_allocation(np.array([0.0, 0.0, 1.5, 0.0]), history, 1e6, 1e6)
    assert [v.rule for v in decision.violations] == ["margin"]


def test_drawdown_violation_when_already_close_to_limit(history):
    weights = np.array([0.1, 0.05, 0.3, 0.1])
    decision = SCGGuard().check_allocation(weights, history, equity=905_000, peak_equity=1_000_000)
    assert not decision.approved
    assert "max_drawdown" in {v.rule for v in decision.violations}
    assert decision.metrics["projected_drawdown"] > 0.095


def test_cash_is_always_admissible(history):
    decision = SCGGuard().check_allocation(np.zeros(4), history, equity=850_000, peak_equity=1_000_000)
    assert decision.approved


def test_order_check_applies_delta(history):
    guard = SCGGuard()
    current = np.array([0.1, 0.0, 0.3, 0.1])
    assert guard.check_order(current, {2: 0.05}, history, 1e6, 1e6).approved
    assert not guard.check_order(current, {1: 1.2}, history, 1e6, 1e6).approved
    with pytest.raises(ValueError):
        guard.check_order(current, {9: 0.1}, history, 1e6, 1e6)


def test_invalid_limits():
    with pytest.raises(ValueError):
        SCGGuard(RiskLimits(max_drawdown=1.5))
    with pytest.raises(ValueError):
        SCGGuard(RiskLimits(margin_rates=(0.1,)))
