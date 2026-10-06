"""Filtre χ² de rupture de régime (liquidité, manipulation, flash-crash).

Pour chaque série surveillée (rendement, log-volume, log-spread de chaque
actif), on découpe une période de référence en classes de quantiles, avec des
classes de queue étroites (2 % de chaque côté). On compte ensuite où tombent
les observations de la fenêtre récente et on calcule la statistique de
Pearson :

    χ² = Σ (observé − attendu)² / attendu

agrégée sur toutes les séries, avec df = séries × (classes − 1).

Les effectifs attendus des classes de queue sont petits (0,2 observation), ce
qui rend l'approximation asymptotique du χ² trop permissive. La p-value de
décision est donc calibrée par Monte-Carlo sous l'hypothèse nulle (tirages
multinomiaux avec les probabilités de classe) ; la p-value asymptotique est
conservée à titre indicatif.

Une p-value inférieure à ``alpha`` signale que la distribution récente ne
ressemble plus au marché de référence : le portefeuille passe en mode
sécurité. Le mode sécurité est levé après ``cooldown`` pas consécutifs sans
alerte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .market import MarketData
from .stats import chi2_sf


@dataclass(frozen=True)
class Chi2Config:
    window: int = 10
    reference: int = 250
    quantiles: tuple[float, ...] = (0.02, 0.15, 0.50, 0.85, 0.98)
    alpha: float = 1e-4
    cooldown: int = 5
    mc_samples: int = 100_000

    def bin_probabilities(self) -> np.ndarray:
        edges = np.concatenate([[0.0], np.asarray(self.quantiles), [1.0]])
        return np.diff(edges)


@dataclass
class Chi2Reading:
    step: int
    statistic: float
    df: int
    p_value: float
    p_value_asymptotic: float
    alert: bool
    safe_mode: bool
    top_contributors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "statistic": round(self.statistic, 4),
            "df": self.df,
            "p_value": float(f"{self.p_value:.6g}"),
            "p_value_asymptotic": float(f"{self.p_value_asymptotic:.6g}"),
            "alert": self.alert,
            "safe_mode": self.safe_mode,
            "top_contributors": self.top_contributors,
        }


def market_features(market: MarketData) -> tuple[np.ndarray, list[str]]:
    """Empile rendements, log-volumes et log-spreads : (T, 3N) + noms des séries."""
    features = np.hstack([market.returns, np.log(market.volumes), np.log(market.spreads_bps)])
    names = (
        [f"rendement:{a}" for a in market.assets]
        + [f"volume:{a}" for a in market.assets]
        + [f"spread:{a}" for a in market.assets]
    )
    return features, names


@lru_cache(maxsize=16)
def _null_distribution(window: int, probs: tuple[float, ...], n_series: int, samples: int) -> np.ndarray:
    """Distribution de la statistique agrégée sous H0, triée (graine fixe : reproductible)."""
    rng = np.random.default_rng(20240601)
    p = np.asarray(probs)
    expected = p * window
    out = np.empty(samples)
    chunk = 20_000
    for start in range(0, samples, chunk):
        size = min(chunk, samples - start)
        counts = rng.multinomial(window, p, size=(size, n_series))
        out[start : start + size] = ((counts - expected) ** 2 / expected).sum(axis=(1, 2))
    out.sort()
    return out


class Chi2Filter:
    def __init__(self, config: Chi2Config | None = None, series_names: list[str] | None = None) -> None:
        self.config = config or Chi2Config()
        self.series_names = series_names
        self._probs = self.config.bin_probabilities()
        self._calm_steps = 0
        self.safe_mode = False

    @property
    def min_history(self) -> int:
        return self.config.reference + self.config.window

    def statistic(self, features: np.ndarray) -> tuple[float, int, np.ndarray]:
        """Statistique χ² sur la fin de ``features`` (T, S). Retourne (χ², df, contributions)."""
        cfg = self.config
        features = np.asarray(features, dtype=float)
        if features.shape[0] < self.min_history:
            raise ValueError(f"il faut au moins {self.min_history} observations")
        recent = features[-cfg.window :]
        reference = features[-(cfg.window + cfg.reference) : -cfg.window]
        expected = self._probs * cfg.window
        contributions = np.empty(features.shape[1])
        for s in range(features.shape[1]):
            edges = np.quantile(reference[:, s], cfg.quantiles)
            bins = np.searchsorted(edges, recent[:, s], side="right")
            observed = np.bincount(bins, minlength=len(self._probs))
            contributions[s] = float(np.sum((observed - expected) ** 2 / expected))
        df = features.shape[1] * (len(self._probs) - 1)
        return float(contributions.sum()), df, contributions

    def p_value(self, statistic: float, n_series: int) -> float:
        """p-value Monte-Carlo : part des tirages sous H0 au moins aussi extrêmes."""
        cfg = self.config
        null = _null_distribution(cfg.window, tuple(float(p) for p in self._probs), n_series, cfg.mc_samples)
        exceed = null.size - int(np.searchsorted(null, statistic, side="left"))
        return (1 + exceed) / (1 + null.size)

    def update(self, features: np.ndarray, step: int) -> Chi2Reading:
        """Évalue la fenêtre courante et met à jour l'état du mode sécurité."""
        stat, df, contributions = self.statistic(features)
        p_value = self.p_value(stat, contributions.size)
        alert = p_value < self.config.alpha
        if alert:
            self.safe_mode = True
            self._calm_steps = 0
        elif self.safe_mode:
            self._calm_steps += 1
            if self._calm_steps >= self.config.cooldown:
                self.safe_mode = False
                self._calm_steps = 0
        order = np.argsort(contributions)[::-1][:3]
        names = self.series_names or [f"serie_{i}" for i in range(len(contributions))]
        top = [{"series": names[i], "contribution": round(float(contributions[i]), 3)} for i in order]
        return Chi2Reading(step, stat, df, p_value, chi2_sf(stat, df), alert, self.safe_mode, top)
