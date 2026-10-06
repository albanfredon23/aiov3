import numpy as np
import pytest

from aiotrade.market import ASSETS, MarketConfig, Shock, generate_market


def test_market_shapes_and_determinism():
    a = generate_market(MarketConfig(steps=300, seed=1))
    b = generate_market(MarketConfig(steps=300, seed=1))
    assert a.returns.shape == (300, len(ASSETS))
    assert a.prices.shape == (301, len(ASSETS))
    assert np.array_equal(a.returns, b.returns)
    assert np.all(a.prices > 0) and np.all(a.volumes > 0) and np.all(a.spreads_bps > 0)


def test_flash_crash_is_injected():
    m = generate_market(MarketConfig(steps=400, seed=2, shock=Shock.FLASH_CRASH, shock_at=300))
    assert m.shock_window == (300, 308)
    crash = m.returns[300:303, 1].sum()  # TECH_US
    assert crash < -0.15
    assert m.spreads_bps[300].mean() > 5 * m.spreads_bps[:300].mean()


def test_liquidity_drop_and_manipulation():
    liq = generate_market(MarketConfig(steps=400, seed=3, shock=Shock.LIQUIDITY_DROP, shock_at=300))
    assert liq.volumes[300:325].mean() < 0.2 * liq.volumes[:300].mean()
    man = generate_market(MarketConfig(steps=400, seed=3, shock=Shock.MANIPULATION, shock_at=300))
    assert man.volumes[300:316, 1].mean() > 5 * man.volumes[:300, 1].mean()


def test_invalid_configuration():
    with pytest.raises(ValueError):
        generate_market(MarketConfig(steps=50))
    with pytest.raises(ValueError):
        generate_market(MarketConfig(steps=300, shock=Shock.FLASH_CRASH, shock_at=10))
