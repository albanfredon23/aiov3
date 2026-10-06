"""Générateur de marché synthétique (simulation uniquement, aucune donnée réelle).

Produit des prix, volumes et spreads corrélés pour quelques classes d'actifs,
avec des régimes de tendance et de range, et des chocs injectables :

- ``flash_crash``     : chute brutale en quelques pas puis rebond partiel ;
- ``liquidity_drop``  : volumes effondrés, spreads écartés, volatilité doublée ;
- ``manipulation``    : séquence pump & dump sur un actif, volumes anormaux.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

ASSETS: tuple[str, ...] = ("ACTIONS_EU", "TECH_US", "OBLIG_10A", "OR")
_ANNUAL_DRIFT = np.array([0.06, 0.10, 0.02, 0.04])
_ANNUAL_VOL = np.array([0.18, 0.28, 0.06, 0.15])
_CORRELATION = np.array(
    [
        [1.00, 0.75, -0.25, 0.05],
        [0.75, 1.00, -0.30, 0.00],
        [-0.25, -0.30, 1.00, 0.20],
        [0.05, 0.00, 0.20, 1.00],
    ]
)
_BASE_SPREAD_BPS = np.array([2.0, 4.0, 1.0, 3.0])
_BASE_VOLUME = np.array([2.0e6, 3.5e6, 1.0e6, 0.8e6])
# Sensibilité de chaque actif à un krach actions (valeur refuge pour l'obligataire et l'or).
_CRASH_BETA = np.array([1.0, 1.35, -0.25, -0.15])
_DT = 1.0 / 252.0


class Shock(str, Enum):
    NONE = "none"
    FLASH_CRASH = "flash_crash"
    LIQUIDITY_DROP = "liquidity_drop"
    MANIPULATION = "manipulation"


@dataclass(frozen=True)
class MarketConfig:
    steps: int = 600
    seed: int = 7
    shock: Shock = Shock.NONE
    shock_at: int | None = None  # par défaut : 75 % de la série
    regime_length: int = 80

    def resolved_shock_at(self) -> int:
        if self.shock_at is not None:
            return self.shock_at
        return int(self.steps * 0.75)


@dataclass
class MarketData:
    assets: tuple[str, ...]
    returns: np.ndarray  # (T, N) rendements simples par pas
    prices: np.ndarray  # (T + 1, N)
    volumes: np.ndarray  # (T, N)
    spreads_bps: np.ndarray  # (T, N)
    shock: Shock = Shock.NONE
    shock_window: tuple[int, int] | None = None
    regimes: list[str] = field(default_factory=list)

    @property
    def steps(self) -> int:
        return int(self.returns.shape[0])


def generate_market(config: MarketConfig) -> MarketData:
    if config.steps < 120:
        raise ValueError("steps doit être >= 120 pour disposer d'un historique de référence")
    rng = np.random.default_rng(config.seed)
    n = len(ASSETS)
    t_total = config.steps

    chol = np.linalg.cholesky(_CORRELATION)
    vol_step = _ANNUAL_VOL * np.sqrt(_DT)
    drift_step = _ANNUAL_DRIFT * _DT

    # Régimes : tendance haussière, baissière ou range (dérive additionnelle par régime).
    regimes: list[str] = []
    regime_drift = np.zeros((t_total, n))
    for start in range(0, t_total, config.regime_length):
        kind = rng.choice(["tendance_haussiere", "tendance_baissiere", "range"], p=[0.4, 0.25, 0.35])
        end = min(start + config.regime_length, t_total)
        regimes.extend([str(kind)] * (end - start))
        if kind == "tendance_haussiere":
            regime_drift[start:end] = 0.8 * vol_step * rng.uniform(0.05, 0.15)
        elif kind == "tendance_baissiere":
            regime_drift[start:end] = -0.8 * vol_step * rng.uniform(0.05, 0.15)

    shocks = rng.standard_normal((t_total, n)) @ chol.T
    returns = drift_step + regime_drift + shocks * vol_step

    # Retour à la moyenne léger en régime de range (autocorrélation négative).
    for t in range(1, t_total):
        if regimes[t] == "range":
            returns[t] -= 0.15 * returns[t - 1]

    volumes = _BASE_VOLUME * rng.lognormal(mean=0.0, sigma=0.2, size=(t_total, n))
    spreads = _BASE_SPREAD_BPS * rng.lognormal(mean=0.0, sigma=0.15, size=(t_total, n))

    shock_window: tuple[int, int] | None = None
    if config.shock != Shock.NONE:
        start = config.resolved_shock_at()
        if not 60 <= start < t_total - 10:
            raise ValueError("shock_at doit laisser au moins 60 pas avant et 10 pas après le choc")
        shock_window = _inject_shock(config.shock, start, returns, volumes, spreads, vol_step, rng)

    returns = np.clip(returns, -0.95, 3.0)
    prices = 100.0 * np.vstack([np.ones(n), np.cumprod(1.0 + returns, axis=0)])
    return MarketData(
        assets=ASSETS,
        returns=returns,
        prices=prices,
        volumes=volumes,
        spreads_bps=spreads,
        shock=config.shock,
        shock_window=shock_window,
        regimes=regimes,
    )


def _inject_shock(
    shock: Shock,
    start: int,
    returns: np.ndarray,
    volumes: np.ndarray,
    spreads: np.ndarray,
    vol_step: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    t_total = returns.shape[0]
    if shock == Shock.FLASH_CRASH:
        crash = np.array([-0.045, -0.06, -0.035])
        rebound = np.array([0.02, 0.015, 0.01, 0.008, 0.005])
        end = min(start + len(crash) + len(rebound), t_total)
        for i, r in enumerate(crash):
            returns[start + i] = r * _CRASH_BETA + rng.normal(0, 0.002, size=returns.shape[1])
        for i, r in enumerate(rebound):
            if start + len(crash) + i < t_total:
                returns[start + len(crash) + i] = r * _CRASH_BETA
        volumes[start:end] *= np.linspace(8.0, 3.0, end - start)[:, None]
        spreads[start:end] *= np.linspace(12.0, 3.0, end - start)[:, None]
        return start, end
    if shock == Shock.LIQUIDITY_DROP:
        end = min(start + 25, t_total)
        volumes[start:end] *= 0.08
        spreads[start:end] *= 9.0
        returns[start:end] += rng.standard_normal((end - start, returns.shape[1])) * vol_step * 1.5
        return start, end
    if shock == Shock.MANIPULATION:
        end = min(start + 16, t_total)
        target = 1  # TECH_US
        pattern = np.tile([0.045, -0.04], (end - start + 1) // 2)[: end - start]
        returns[start:end, target] = pattern
        volumes[start:end, target] *= 10.0
        spreads[start:end, target] *= 4.0
        return start, end
    raise ValueError(f"choc inconnu : {shock}")
