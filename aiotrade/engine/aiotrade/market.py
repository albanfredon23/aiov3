"""Marché synthétique intrajournalier (simulation uniquement, aucune donnée réelle).

Un instrument (par défaut un contrat de type EURUSD) en barres de 15 minutes,
24 h/24 du lundi au vendredi, avec :

- chandeliers OHLCV, fourchette bid/ask et profondeur du meilleur niveau (L1) ;
- volatilité en grappes (GARCH(1,1)) et régimes tendance / retour à la moyenne / bruit ;
- un calendrier synthétique d'annonces macroéconomiques à fort impact (NFP, IPC,
  Fed, BCE) : à l'heure de l'annonce, le carnet se vide, le spread explose et le
  prix saute, avant que tout test statistique puisse réagir ;
- des chocs injectables : ``flash_crash``, ``liquidity_drop``, ``spoofing``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np

UTC = timezone.utc
TRADING_DAYS_PER_YEAR = 252


class Shock(str, Enum):
    NONE = "none"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_DROP = "liquidity_drop"
    SPOOFING = "spoofing"


@dataclass(frozen=True)
class MacroEvent:
    """Annonce macroéconomique. ``time_utc`` doit être un datetime conscient du fuseau (UTC)."""

    name: str
    time_utc: datetime
    impact: str = "HIGH"  # HIGH | MEDIUM | LOW

    def __post_init__(self) -> None:
        if self.time_utc.tzinfo is None or self.time_utc.utcoffset() != timedelta(0):
            raise ValueError("time_utc doit être un datetime UTC conscient du fuseau")
        if self.impact not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("impact doit valoir HIGH, MEDIUM ou LOW")

    def to_dict(self) -> dict:
        return {"name": self.name, "time_utc": iso_utc(self.time_utc), "impact": self.impact}


def iso_utc(moment: datetime) -> str:
    """Horodatage ISO 8601 en UTC à la milliseconde, suffixe Z."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


@dataclass(frozen=True)
class MarketConfig:
    bars: int = 6_000
    seed: int = 7
    bar_minutes: int = 15
    start: datetime = datetime(2025, 1, 6, tzinfo=UTC)  # un lundi
    symbol: str = "EURUSD"
    price0: float = 1.0850
    annual_vol: float = 0.08
    spread_bps: float = 0.9
    bar_volume: float = 5.0e7  # notionnel moyen échangé par barre
    depth: float = 5.0e6  # notionnel moyen au meilleur niveau
    regime_bars: int = 400
    macro_events: bool = True
    shock: Shock = Shock.NONE
    shock_at: int | None = None  # par défaut : 80 % de la série

    @property
    def bars_per_day(self) -> int:
        return 24 * 60 // self.bar_minutes

    @property
    def bars_per_year(self) -> int:
        return self.bars_per_day * TRADING_DAYS_PER_YEAR


@dataclass
class MarketData:
    symbol: str
    bar_minutes: int
    timestamps: list[datetime]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray  # notionnel échangé
    spread: np.ndarray  # en prix
    depth: np.ndarray  # notionnel au meilleur niveau
    events: list[MacroEvent] = field(default_factory=list)
    shock: Shock = Shock.NONE
    shock_window: tuple[int, int] | None = None
    regimes: list[str] = field(default_factory=list)

    @property
    def bars(self) -> int:
        return int(self.close.size)

    @property
    def returns(self) -> np.ndarray:
        """Rendements simples close-to-close ; returns[0] = 0."""
        r = np.zeros_like(self.close)
        r[1:] = self.close[1:] / self.close[:-1] - 1.0
        return r

    @property
    def bars_per_year(self) -> int:
        return 24 * 60 // self.bar_minutes * TRADING_DAYS_PER_YEAR

    def slice(self, start: int, stop: int) -> "MarketData":
        lo, hi = self.timestamps[start], self.timestamps[stop - 1]
        window = None
        if self.shock_window and self.shock_window[1] > start and self.shock_window[0] < stop:
            window = (max(self.shock_window[0], start) - start, min(self.shock_window[1], stop) - start)
        return MarketData(
            symbol=self.symbol,
            bar_minutes=self.bar_minutes,
            timestamps=self.timestamps[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
            spread=self.spread[start:stop],
            depth=self.depth[start:stop],
            events=[e for e in self.events if lo - timedelta(hours=2) <= e.time_utc <= hi + timedelta(hours=2)],
            shock=self.shock if window else Shock.NONE,
            shock_window=window,
            regimes=self.regimes[start:stop],
        )


def macro_event_mask(market: "MarketData", bars_before: int = 1, bars_after: int = 3) -> np.ndarray:
    """Barres entourant chaque annonce macro (exclues du calibrage du filtre d'intégrité)."""
    index = {t: i for i, t in enumerate(market.timestamps)}
    mask = np.zeros(market.bars, dtype=bool)
    for e in market.events:
        start = e.time_utc.replace(minute=e.time_utc.minute - e.time_utc.minute % market.bar_minutes)
        i = index.get(start)
        if i is not None:
            mask[max(0, i - bars_before) : i + bars_after + 1] = True
    return mask


def trading_timestamps(start: datetime, bars: int, bar_minutes: int) -> list[datetime]:
    """Horodatages de barres 24 h/24, du lundi au vendredi (UTC)."""
    out: list[datetime] = []
    t = start
    step = timedelta(minutes=bar_minutes)
    while len(out) < bars:
        if t.weekday() < 5:
            out.append(t)
        t += step
    return out


_MACRO_TEMPLATES = (
    ("NFP (emploi américain)", 12, 30),
    ("IPC américain", 12, 30),
    ("Décision de taux Fed", 18, 0),
    ("Décision de taux BCE", 12, 15),
)


def synthetic_calendar(timestamps: list[datetime], rng: np.random.Generator) -> list[MacroEvent]:
    """Une annonce à fort impact tous les 3 à 5 jours ouvrés, à une heure réaliste."""
    events: list[MacroEvent] = []
    days = sorted({t.date() for t in timestamps})
    i = int(rng.integers(1, 4))
    while i < len(days):
        name, hour, minute = _MACRO_TEMPLATES[int(rng.integers(0, len(_MACRO_TEMPLATES)))]
        d = days[i]
        events.append(MacroEvent(name, datetime(d.year, d.month, d.day, hour, minute, tzinfo=UTC)))
        i += int(rng.integers(3, 6))
    return events


def generate_market(config: MarketConfig) -> MarketData:
    if config.bars < 500:
        raise ValueError("bars doit être >= 500")
    if 24 * 60 % config.bar_minutes:
        raise ValueError("bar_minutes doit diviser 1 440")
    rng = np.random.default_rng(config.seed)
    n = config.bars
    ts = trading_timestamps(config.start, n, config.bar_minutes)
    sigma_bar = config.annual_vol / np.sqrt(config.bars_per_year)

    # Volatilité en grappes : GARCH(1,1) normalisé sur la volatilité cible.
    a, b = 0.06, 0.92
    omega = sigma_bar**2 * (1 - a - b)
    shocks = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)  # queues épaisses, variance 1
    var = np.empty(n)
    eps = np.empty(n)
    var[0] = sigma_bar**2
    eps[0] = shocks[0] * sigma_bar
    for t in range(1, n):
        var[t] = omega + a * eps[t - 1] ** 2 + b * var[t - 1]
        eps[t] = shocks[t] * np.sqrt(var[t])

    # Régimes : tendance (dérive persistante), retour à la moyenne (OU), bruit.
    regimes: list[str] = []
    log_ret = np.zeros(n)
    logp = 0.0
    anchor = 0.0
    drift = 0.0
    for t in range(n):
        if t % config.regime_bars == 0:
            kind = str(rng.choice(["tendance", "retour_moyenne", "bruit"], p=[0.4, 0.35, 0.25]))
            drift = float(rng.choice([-1, 1]) * rng.uniform(0.04, 0.08) * sigma_bar)
            anchor = logp
        regimes.append(kind)
        if kind == "tendance":
            mu = drift
        elif kind == "retour_moyenne":
            mu = -0.04 * (logp - anchor)
        else:
            mu = 0.0
        log_ret[t] = mu + eps[t]
        logp += log_ret[t]

    volume = config.bar_volume * rng.lognormal(0.0, 0.3, n) * (1.0 + 0.5 * np.abs(eps) / sigma_bar)
    spread_frac = config.spread_bps / 1e4 * rng.lognormal(0.0, 0.12, n) * np.power(var / sigma_bar**2, 0.25)
    depth = config.depth * rng.lognormal(0.0, 0.25, n)

    events: list[MacroEvent] = []
    if config.macro_events:
        events = synthetic_calendar(ts, rng)
        index = {t: i for i, t in enumerate(ts)}
        for ev in events:
            i = index.get(ev.time_utc.replace(minute=ev.time_utc.minute - ev.time_utc.minute % config.bar_minutes))
            if i is None:
                continue
            log_ret[i] += rng.normal(0.0, 6.0 * sigma_bar)  # saut à la publication
            spread_frac[i : i + 2] *= 10.0
            depth[i : i + 2] /= 12.0
            volume[i : i + 3] *= 5.0

    shock_window: tuple[int, int] | None = None
    if config.shock != Shock.NONE:
        start = config.shock_at if config.shock_at is not None else int(n * 0.8)
        if not 300 <= start < n - 40:
            raise ValueError("shock_at doit laisser au moins 300 barres avant et 40 après le choc")
        shock_window = _inject_shock(config.shock, start, log_ret, spread_frac, depth, volume, sigma_bar, rng)

    close = config.price0 * np.exp(np.cumsum(log_ret))
    prev = np.concatenate([[config.price0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0.0, 0.1 * sigma_bar, n))
    wick = np.abs(rng.normal(0.0, 0.6, (2, n))) * np.sqrt(var) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    return MarketData(
        symbol=config.symbol,
        bar_minutes=config.bar_minutes,
        timestamps=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        spread=spread_frac * close,
        depth=depth,
        events=events,
        shock=config.shock,
        shock_window=shock_window,
        regimes=regimes,
    )


def _inject_shock(
    shock: Shock,
    start: int,
    log_ret: np.ndarray,
    spread: np.ndarray,
    depth: np.ndarray,
    volume: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if shock == Shock.FLASH_CRASH:
        crash = np.array([-6.0, -9.0, -5.0]) * sigma
        rebound = np.array([4.0, 3.0, 2.0, 1.0]) * sigma
        end = start + len(crash) + len(rebound)
        log_ret[start : start + 3] = crash
        log_ret[start + 3 : end] = rebound
        spread[start:end] *= np.linspace(15.0, 3.0, end - start)
        depth[start:end] /= np.linspace(20.0, 3.0, end - start)
        volume[start:end] *= 6.0
        return start, end
    if shock == Shock.LIQUIDITY_DROP:
        end = start + 30
        spread[start:end] *= 6.0
        depth[start:end] /= 15.0
        volume[start:end] *= 0.15
        log_ret[start:end] += rng.normal(0.0, 1.5 * sigma, end - start)
        return start, end
    if shock == Shock.SPOOFING:
        end = start + 20
        depth[start:end] *= np.tile([8.0, 0.2], 10)  # ordres fantômes posés puis retirés
        volume[start:end] *= 4.0
        log_ret[start:end] += np.tile([2.5, -2.5], 10) * sigma
        spread[start:end] *= 2.5
        return start, end
    raise ValueError(f"choc inconnu : {shock}")
