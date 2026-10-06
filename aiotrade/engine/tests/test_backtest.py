import numpy as np
import pytest

from aiotrade.backtest import ARMS, BacktestConfig, performance, run_fold, walk_forward
from aiotrade.market import MarketConfig, Shock, generate_market


@pytest.fixture(scope="module")
def fold():
    m = generate_market(MarketConfig(bars=3_960, seed=5, shock=Shock.FLASH_CRASH, shock_at=3_500))
    return m, run_fold(m, 0, 3_000, 3_960, BacktestConfig())


def test_quatre_bras_et_courbes_finies(fold):
    _, f = fold
    assert set(f.equity) == set(ARMS)
    for a in ARMS:
        assert f.equity[a].size == 961 and np.all(np.isfinite(f.equity[a]))
        assert f.equity[a][0] == 100_000.0


def test_levier_du_bras_d_borne(fold):
    _, f = fold
    assert np.max(np.abs(f.exposure_d)) <= 1.5 * 1.05


def test_bras_d_hors_marche_apres_blackout_et_freeze(fold):
    _, f = fold
    assert "BLACKOUT" in f.gate_states and "FREEZE" in f.integrity_status
    for k in range(1, len(f.exposure_d)):
        if f.gate_states[k - 1] == "BLACKOUT" or f.integrity_status[k - 1] == "FREEZE":
            assert f.exposure_d[k] == 0.0, k


def test_registre_complet_et_verifie(fold):
    _, f = fold
    assert len(f.ledger) >= f.scg_stats["decisions"]
    assert f.ledger.verify().ok


def test_performance_nette():
    eq = np.array([100.0, 110.0, 99.0, 120.0])
    perf = performance(eq, 252)
    assert perf["max_drawdown"] == pytest.approx(0.1)
    assert perf["total_return"] == pytest.approx(0.2)


def test_walk_forward_rapport():
    cfg = BacktestConfig(folds=1, calibration_months=2, test_months=1, trading_days_per_month=10)
    rep = walk_forward(cfg)
    assert set(rep["aggregate"]) == set(ARMS)
    assert set(rep["targets"]) == {"max_drawdown_reduction_vs_B", "sortino_net", "calmar_net", "scg_pruning_rate"}
    assert all(isinstance(t["met"], bool) for t in rep["targets"].values())
    assert rep["ledger"][0]["verified"] is True
    assert rep["protocol"]["calibration_months"] == 2
