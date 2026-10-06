"""TAP — Trajectory Admissible Planning.

Le moteur ne devine pas un cours ponctuel P(t+1) : il demande au modèle de
prévision N trajectoires de rendements multi-pas sur l'horizon H, puis
prépare, pour chaque direction candidate (LONG, SHORT), le faisceau de
résultats que le garde-fou SCG va élaguer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .forecast import Forecaster, MarketContext


@dataclass(frozen=True)
class TAPConfig:
    n_paths: int = 1024
    horizon: int = 12  # barres (3 h en barres de 15 min)

    def __post_init__(self) -> None:
        if self.n_paths < 8 or self.horizon < 1:
            raise ValueError("n_paths >= 8 et horizon >= 1 requis")


@dataclass
class TrajectoryBundle:
    returns: np.ndarray  # (N, H) rendements simples par barre
    source: str

    @property
    def n_paths(self) -> int:
        return int(self.returns.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.returns.shape[1])

    @property
    def cumulative(self) -> np.ndarray:
        """(N, H) rendement cumulé composé depuis l'entrée."""
        return np.cumprod(1.0 + self.returns, axis=1) - 1.0

    def finals(self, direction: int) -> np.ndarray:
        """Rendement final par unité de notionnel dans la direction donnée (+1 / −1)."""
        return direction * self.cumulative[:, -1]

    def adverse_excursion(self, direction: int) -> np.ndarray:
        """Pire excursion défavorable par trajectoire (fraction positive)."""
        return np.maximum(0.0, -(direction * self.cumulative).min(axis=1))


class TAPPlanner:
    def __init__(self, config: TAPConfig | None = None) -> None:
        self.config = config or TAPConfig()

    def plan(self, forecaster: Forecaster, ctx: MarketContext, rng: np.random.Generator) -> TrajectoryBundle:
        paths = forecaster.sample(ctx, self.config.n_paths, self.config.horizon, rng)
        paths = np.asarray(paths, dtype=float)
        if paths.shape != (self.config.n_paths, self.config.horizon) or not np.all(np.isfinite(paths)):
            raise ValueError("le modèle de prévision doit renvoyer une matrice (N, H) finie")
        return TrajectoryBundle(np.clip(paths, -0.5, 0.5), getattr(forecaster, "name", "inconnu"))

    @staticmethod
    def stop_distance(bundle: TrajectoryBundle, direction: int, sigma: float, quantile: float = 0.8) -> float:
        """Distance de stop : quantile de l'excursion défavorable, plancher à 2 σ."""
        excursion = bundle.adverse_excursion(direction)
        return float(max(np.quantile(excursion, quantile), 2.0 * sigma, 1e-4))
