from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aiotrade.macro_gate import GateState, MacroCalendarGate, MacroGateConfig, StaticCalendar
from aiotrade.market import MacroEvent

UTC = timezone.utc
NFP = MacroEvent("NFP", datetime(2026, 11, 6, 13, 30, tzinfo=UTC))


def gate(*events, **cfg):
    return MacroCalendarGate(StaticCalendar(events), MacroGateConfig(**cfg))


@pytest.mark.parametrize(
    "minutes, state, multiplier, allowed",
    [
        (-60, GateState.NOMINAL, 1.0, True),
        (-45, GateState.DERISK, 0.5, True),
        (-16, GateState.DERISK, 0.5, True),
        (-15, GateState.BLACKOUT, 0.0, False),
        (0, GateState.BLACKOUT, 0.0, False),
        (10, GateState.BLACKOUT, 0.0, False),
        (11, GateState.NOMINAL, 1.0, True),
    ],
)
def test_fenetres_autour_d_une_annonce(minutes, state, multiplier, allowed):
    d = gate(NFP).evaluate(NFP.time_utc + timedelta(minutes=minutes))
    assert d.state == state
    assert d.exposure_multiplier == multiplier
    assert d.allow_new_positions is allowed


def test_l_etat_le_plus_restrictif_l_emporte():
    # Le code de référence s'arrêtait au premier événement trouvé (ici en DERISK) et ratait le BLACKOUT.
    first = MacroEvent("Fed", NFP.time_utc + timedelta(minutes=40))
    d = gate(first, NFP).evaluate(NFP.time_utc - timedelta(minutes=5))
    assert d.state == GateState.BLACKOUT
    assert d.event == NFP


def test_evenements_moyens_ignores():
    pmi = MacroEvent("PMI", NFP.time_utc, impact="MEDIUM")
    assert gate(pmi).evaluate(NFP.time_utc).state == GateState.NOMINAL


def test_horodatage_naif_refuse():
    with pytest.raises(ValueError):
        gate(NFP).evaluate(datetime(2026, 11, 6, 13, 30))


def test_configuration_validee():
    with pytest.raises(ValueError):
        MacroGateConfig(blackout_exposure=0.8)  # réduction de 20 % seulement : hors spécification
    with pytest.raises(ValueError):
        MacroGateConfig(derisk_minutes_before=10)


def test_calendrier_json_versionne():
    path = Path(__file__).resolve().parents[1] / "data" / "macro_calendar.example.json"
    cal = StaticCalendar.from_json(path)
    events = cal.events_between(datetime(2026, 1, 1, tzinfo=UTC), datetime(2027, 1, 1, tzinfo=UTC))
    assert events and all(e.time_utc.tzinfo is not None for e in events)
    assert MacroCalendarGate(cal).evaluate(events[0].time_utc).state in (GateState.BLACKOUT, GateState.NOMINAL)


def test_decision_serialisable():
    d = gate(NFP).evaluate(NFP.time_utc).to_dict()
    assert d["state"] == "BLACKOUT" and d["event"]["time_utc"] == "2026-11-06T13:30:00.000Z"
