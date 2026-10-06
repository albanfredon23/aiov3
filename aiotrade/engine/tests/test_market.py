from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from aiotrade.market import MacroEvent, MarketConfig, Shock, generate_market, iso_utc, trading_timestamps


def test_generation_deterministe_et_coherente():
    a = generate_market(MarketConfig(bars=1_500, seed=3))
    b = generate_market(MarketConfig(bars=1_500, seed=3))
    assert np.array_equal(a.close, b.close)
    assert a.bars == 1_500
    assert np.all(a.high >= np.maximum(a.open, a.close))
    assert np.all(a.low <= np.minimum(a.open, a.close))
    assert np.all(a.spread > 0) and np.all(a.depth > 0) and np.all(a.volume > 0)
    assert a.returns[0] == 0.0


def test_horodatages_ouvres_et_utc():
    ts = trading_timestamps(datetime(2025, 1, 3, tzinfo=timezone.utc), 200, 15)  # un vendredi
    assert all(t.weekday() < 5 for t in ts)
    assert all(t.tzinfo is not None for t in ts)


def test_calendrier_macro_conscient_du_fuseau():
    m = generate_market(MarketConfig(bars=3_000, seed=5))
    assert m.events
    assert all(e.time_utc.utcoffset() == timedelta(0) for e in m.events)
    with pytest.raises(ValueError):
        MacroEvent("NFP", datetime(2026, 1, 9, 13, 30))  # naïf : refusé


@pytest.mark.parametrize("shock", [Shock.FLASH_CRASH, Shock.LIQUIDITY_DROP, Shock.SPOOFING])
def test_chocs_injectes(shock):
    base = generate_market(MarketConfig(bars=1_000, seed=2))
    m = generate_market(MarketConfig(bars=1_000, seed=2, shock=shock, shock_at=600))
    a, b = m.shock_window
    assert a == 600 and b > a
    assert m.spread[a] > base.spread[a]


def test_parametres_invalides():
    with pytest.raises(ValueError):
        generate_market(MarketConfig(bars=100))
    with pytest.raises(ValueError):
        generate_market(MarketConfig(bars=1_000, shock=Shock.FLASH_CRASH, shock_at=10))


def test_iso_utc_millisecondes():
    assert iso_utc(datetime(2026, 10, 6, 17, 15, 2, 104_000, tzinfo=timezone.utc)) == "2026-10-06T17:15:02.104Z"


def test_slice_conserve_la_fenetre_de_choc():
    m = generate_market(MarketConfig(bars=1_000, seed=2, shock=Shock.FLASH_CRASH, shock_at=600))
    part = m.slice(500, 800)
    assert part.bars == 300
    assert part.shock_window == (100, 100 + m.shock_window[1] - m.shock_window[0])
