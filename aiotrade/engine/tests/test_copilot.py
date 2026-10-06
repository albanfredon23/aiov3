from datetime import timedelta

import numpy as np
import pytest

from aiotrade.backtest import _Series
from aiotrade.copilot import Action, Copilot, CopilotConfig, MarketQuote
from aiotrade.forecast import AgentSwarmForecaster
from aiotrade.integrity import IntegrityReading, IntegrityStatus
from aiotrade.macro_gate import GateDecision, GateState, MacroCalendarGate, StaticCalendar
from aiotrade.market import MacroEvent, MarketConfig, generate_market
from aiotrade.sizing import SizingConfig
from aiotrade.tap import TAPConfig

NOMINAL = GateDecision(GateState.NOMINAL, True, 1.0, "CALENDRIER NOMINAL")
REQUIRED = {
    "timestamp_utc", "market_integrity_d2", "chi2_threshold", "status", "tap_trajectories_tested",
    "scg_rejections", "admissibility_ratio", "selected_allocation", "active_constraints",
}


@pytest.fixture(scope="module")
def env():
    m = generate_market(MarketConfig(bars=3_000, seed=21))
    s = _Series(m)
    f = AgentSwarmForecaster()
    f.calibrate(m.close[:2_500], s.returns[:2_500], s.sigma_close[:2_500])
    # Agent tendance volontairement confiant : garantit une décision exposée dans le test.
    f.coef[0] = 50.0
    return m, s, f


def reading(status=IntegrityStatus.NOMINAL, d2=3.0):
    return IntegrityReading(d2, 13.28, status, {})


def decide(env, gate=NOMINAL, status=IntegrityStatus.NOMINAL, config=None, t=2_900):
    m, s, f = env
    cfg = config or CopilotConfig(sizing=SizingConfig(confidence_z=0.0))
    cp = Copilot(f, cfg, m.symbol)
    quote = MarketQuote(float(m.close[t]), float(m.spread[t]), float(s.avg_volume[t]))
    return cp.decide(s.context(t), quote, reading(status), gate, 100_000.0, 100_000.0, np.random.default_rng(1))


def test_freeze_force_le_cash(env):
    d = decide(env, status=IntegrityStatus.FREEZE)
    assert d.action == Action.CASH and d.cash_reason == "freeze"
    assert "INTEGRITY_FREEZE" in d.record["active_constraints"]
    assert d.record["tap_trajectories_tested"] == 0


def test_blackout_macro_force_le_cash(env):
    m, _, _ = env
    event = MacroEvent("NFP", m.timestamps[2_900] + timedelta(minutes=5))
    gate = MacroCalendarGate(StaticCalendar([event])).evaluate(m.timestamps[2_900])
    d = decide(env, gate=gate)
    assert d.action == Action.CASH and d.cash_reason == "macro"
    assert d.record["macro_gate"] == "BLACKOUT"


def test_decision_exposee_admise_par_le_scg(env):
    d = decide(env)
    for t in range(2_900, 2_990, 4):
        d = decide(env, t=t)
        if d.action != Action.CASH:
            break
    assert d.action in (Action.LONG, Action.SHORT)
    assert REQUIRED <= set(d.record)
    assert d.record["admissibility_ratio"] >= 0.75
    assert d.report is not None and d.report.admitted
    assert abs(d.exposure) <= 1.5 + 1e-9
    assert any(c.startswith("STOP_LOSS_") for c in d.record["active_constraints"])
    assert d.size.risk_fraction <= 0.01 + 1e-9


def test_derisking_reduit_la_taille(env):
    full = halved = None
    for t in range(2_900, 2_990, 4):
        full = decide(env, t=t)
        if full.action != Action.CASH:
            halved = decide(env, gate=GateDecision(GateState.DERISK, True, 0.5, "DERISK"), t=t)
            break
    assert halved is not None
    assert abs(halved.exposure) <= abs(full.exposure) + 1e-12
    assert "MACRO_DERISK_0.5X" in halved.record["active_constraints"]
    assert halved.record["reasons"][0] == "DERISK"


def test_comptant_jamais_short(env):
    cfg = CopilotConfig(sizing=SizingConfig(confidence_z=0.0), allow_short=False, tap=TAPConfig(n_paths=256))
    actions = {decide(env, config=cfg, t=t).action for t in range(2_800, 2_990, 4)}
    assert Action.SHORT not in actions
