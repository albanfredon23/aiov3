import numpy as np
import pytest

from aiotrade.chi2 import Chi2Config, Chi2Filter, market_features
from aiotrade.market import MarketConfig, Shock, generate_market


def _run(shock, seed):
    m = generate_market(MarketConfig(steps=500, seed=seed, shock=shock, shock_at=400))
    features, names = market_features(m)
    f = Chi2Filter(series_names=names)
    readings = [f.update(features[: t + 1], t) for t in range(f.min_history - 1, m.steps)]
    return m, readings


@pytest.mark.parametrize("shock", [Shock.FLASH_CRASH, Shock.LIQUIDITY_DROP, Shock.MANIPULATION])
def test_shocks_trigger_safe_mode_within_three_steps(shock):
    m, readings = _run(shock, seed=11)
    start = m.shock_window[0]
    by_step = {r.step: r for r in readings}
    assert any(by_step[t].alert for t in range(start, start + 4)), "choc non détecté"
    assert by_step[start + 3].safe_mode


def test_calm_market_rarely_alerts():
    alerts = 0
    total = 0
    for seed in range(3):
        _, readings = _run(Shock.NONE, seed)
        alerts += sum(r.alert for r in readings)
        total += len(readings)
    assert alerts / total < 0.01


def test_safe_mode_is_released_after_cooldown():
    cfg = Chi2Config(cooldown=3)
    m = generate_market(MarketConfig(steps=500, seed=12, shock=Shock.FLASH_CRASH, shock_at=400))
    features, _ = market_features(m)
    f = Chi2Filter(cfg)
    states = [f.update(features[: t + 1], t).safe_mode for t in range(f.min_history - 1, m.steps)]
    assert any(states)
    assert states[-1] is False


def test_monte_carlo_p_value_is_calibrated_under_null():
    cfg = Chi2Config(mc_samples=20_000)
    f = Chi2Filter(cfg)
    rng = np.random.default_rng(0)
    data = rng.standard_normal((cfg.reference + cfg.window, 3))
    stat, df, contrib = f.statistic(data)
    assert df == 3 * (len(cfg.quantiles))
    assert contrib.shape == (3,)
    assert 0.0 < f.p_value(stat, 3) <= 1.0
    assert f.p_value(1e6, 3) == pytest.approx(1 / (cfg.mc_samples + 1))


def test_requires_enough_history():
    with pytest.raises(ValueError):
        Chi2Filter().statistic(np.zeros((10, 2)))
