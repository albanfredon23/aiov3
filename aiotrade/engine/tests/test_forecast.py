import numpy as np
import pytest

from aiotrade.backtest import _Series
from aiotrade.forecast import AGENTS, AgentSwarmForecaster, KronosForecaster
from aiotrade.market import MarketConfig, generate_market


@pytest.fixture(scope="module")
def setup():
    m = generate_market(MarketConfig(bars=4_000, seed=9))
    s = _Series(m)
    f = AgentSwarmForecaster()
    info = f.calibrate(m.close[:3_000], s.returns[:3_000], s.sigma_close[:3_000])
    return m, s, f, info


def test_calibrage_des_agents(setup):
    _, _, f, info = setup
    assert set(info["coefficients"]) == set(AGENTS)
    assert info["coefficients"]["risk"] == 0.0
    assert f.vol_edges[0] < f.vol_edges[1]
    assert 0.0 <= info["mean_reversion_speed"] <= 1.0


def test_trajectoires_et_poids(setup):
    _, s, f, _ = setup
    ctx = s.context(3_100)
    paths = f.sample(ctx, 256, 12, np.random.default_rng(0))
    assert paths.shape == (256, 12) and np.all(np.isfinite(paths))
    assert paths.std() == pytest.approx(ctx.sigma, rel=0.6)
    w = f.weights(ctx.sigma)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    assert np.isfinite(f.expected_return(ctx, 12))
    assert f.describe(ctx)["model"] == "agent_swarm"


def test_ponderation_en_ligne_par_regime(setup):
    _, s, f0, _ = setup
    f = AgentSwarmForecaster()
    f.coef, f.vol_edges, f.persistence = f0.coef.copy(), f0.vol_edges, f0.persistence.copy()
    ctx = s.context(3_200)
    before = f.weights(ctx.sigma).copy()
    for t in range(3_200, 3_500):
        f.observe(s.context(t), float(s.returns[t + 1]))
    after = f.weights(s.context(3_500).sigma)
    assert not np.allclose(before, after)
    assert after[AGENTS.index("risk")] < 0.5  # volatilité majorée pénalisée en régime normal


def test_calibrage_trop_court():
    with pytest.raises(ValueError):
        AgentSwarmForecaster().calibrate(np.ones(500), np.zeros(500), np.ones(500))


class FakeKronos:
    def __init__(self):
        self.calls = []

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count):
        import pandas as pd

        self.calls.append((T, top_p, sample_count))
        last = float(df["close"].iloc[-1])
        return pd.DataFrame({"close": last * (1 + 0.001 * np.arange(1, pred_len + 1))})


def test_adaptateur_kronos_echantillonnage_nucleus(setup):
    pytest.importorskip("pandas")
    _, s, _, _ = setup
    fake = FakeKronos()
    k = KronosForecaster(predictor=fake, temperature=0.8, top_p=0.95)
    paths = k.sample(s.context(3_100), 4, 5, np.random.default_rng(0))
    assert paths.shape == (4, 5)
    assert paths[0, 0] == pytest.approx(0.001)
    assert fake.calls == [(0.8, 0.95, 1)] * 4


def test_kronos_absent_message_explicite(setup):
    _, s, _, _ = setup
    with pytest.raises(RuntimeError, match="Kronos|pandas"):
        KronosForecaster().sample(s.context(3_100), 2, 3, np.random.default_rng(0))
