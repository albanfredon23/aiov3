"""Macro Gate : coupe-circuit calendaire déterministe (Economic Calendar Hard Veto).

Le filtre d'intégrité ne voit que ce qui est déjà dans le carnet. Lors d'une
annonce à fort impact (NFP, IPC américain, décisions Fed / BCE), le carnet se
vide et le spread explose avant qu'un test statistique puisse réagir. Le gate
applique donc une règle fixée à l'avance, autour de chaque annonce « HIGH » :

- ``DERISK``   : de −45 à −15 min, ouverture autorisée mais exposition globale
  (positions ouvertes comprises) réduite à 50 % ;
- ``BLACKOUT`` : de −15 à +10 min, aucune nouvelle position et exposition
  ramenée à 0 % (réduction de 100 %).

Tous les événements sont évalués et l'état le plus restrictif l'emporte
(un événement en DERISK ne masque jamais un BLACKOUT plus loin dans la liste).
Les fenêtres et multiplicateurs sont des paramètres.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Protocol

from .market import UTC, MacroEvent, iso_utc


class GateState(str, Enum):
    NOMINAL = "NOMINAL"
    DERISK = "DERISK"
    BLACKOUT = "BLACKOUT"


_SEVERITY = {GateState.NOMINAL: 0, GateState.DERISK: 1, GateState.BLACKOUT: 2}


@dataclass(frozen=True)
class MacroGateConfig:
    blackout_minutes_before: int = 15
    blackout_minutes_after: int = 10
    derisk_minutes_before: int = 45
    derisk_exposure: float = 0.5  # réduction de 50 %
    blackout_exposure: float = 0.0  # réduction de 100 %

    def __post_init__(self) -> None:
        if not 0 <= self.blackout_exposure <= 0.5:
            raise ValueError("blackout_exposure doit être dans [0 ; 0,5] (réduction de 50 à 100 %)")
        if not 0 <= self.derisk_exposure <= 1:
            raise ValueError("derisk_exposure doit être dans [0 ; 1]")
        if self.derisk_minutes_before < self.blackout_minutes_before:
            raise ValueError("la fenêtre de derisking doit commencer avant le blackout")


@dataclass(frozen=True)
class GateDecision:
    state: GateState
    allow_new_positions: bool
    exposure_multiplier: float
    reason: str
    event: MacroEvent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "allow_new_positions": self.allow_new_positions,
            "exposure_multiplier": self.exposure_multiplier,
            "reason": self.reason,
            "event": self.event.to_dict() if self.event else None,
        }


class CalendarProvider(Protocol):
    def events_between(self, start: datetime, end: datetime) -> list[MacroEvent]: ...


class StaticCalendar:
    """Calendrier en mémoire ou chargé depuis un fichier JSON versionné."""

    def __init__(self, events: Iterable[MacroEvent] = ()) -> None:
        self._events = sorted(events, key=lambda e: e.time_utc)

    @classmethod
    def from_json(cls, path: str | Path) -> "StaticCalendar":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        events = [
            MacroEvent(e["name"], datetime.fromisoformat(e["time_utc"].replace("Z", "+00:00")).astimezone(UTC), e.get("impact", "HIGH"))
            for e in data["events"]
        ]
        return cls(events)

    def events_between(self, start: datetime, end: datetime) -> list[MacroEvent]:
        return [e for e in self._events if start <= e.time_utc <= end]


class MacroCalendarGate:
    def __init__(self, calendar: CalendarProvider, config: MacroGateConfig | None = None) -> None:
        self.calendar = calendar
        self.config = config or MacroGateConfig()

    def evaluate(self, now_utc: datetime) -> GateDecision:
        if now_utc.tzinfo is None:
            raise ValueError("now_utc doit être un datetime conscient du fuseau (UTC)")
        cfg = self.config
        before = timedelta(minutes=max(cfg.derisk_minutes_before, cfg.blackout_minutes_before))
        after = timedelta(minutes=cfg.blackout_minutes_after)
        best = GateDecision(GateState.NOMINAL, True, 1.0, "CALENDRIER NOMINAL")
        for event in self.calendar.events_between(now_utc - after, now_utc + before):
            if event.impact != "HIGH":
                continue
            delta = (now_utc - event.time_utc).total_seconds() / 60.0  # < 0 avant l'annonce
            if -cfg.blackout_minutes_before <= delta <= cfg.blackout_minutes_after:
                candidate = GateDecision(
                    GateState.BLACKOUT, False, cfg.blackout_exposure,
                    f"BLACKOUT MACRO : {event.name} à {iso_utc(event.time_utc)}", event,
                )
            elif -cfg.derisk_minutes_before <= delta < -cfg.blackout_minutes_before:
                candidate = GateDecision(
                    GateState.DERISK, True, cfg.derisk_exposure,
                    f"DERISKING MACRO : approche de {event.name} ({iso_utc(event.time_utc)})", event,
                )
            else:
                continue
            if _SEVERITY[candidate.state] > _SEVERITY[best.state] or (
                candidate.state == best.state and candidate.exposure_multiplier < best.exposure_multiplier
            ):
                best = candidate
        return best
