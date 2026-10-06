"""Filtre d'intégrité microstructure (χ² / Mahalanobis).

À chaque barre, le vecteur de microstructure

    x_t = [rendement dévolatilisé, log spread, log profondeur L1, log volume]

est comparé à son état de référence suivi par moyenne et covariance
exponentielles (EWMA). L'innovation résiduelle donne la distance de Mahalanobis

    d² = (x_t − μ)ᵀ Σ⁻¹ (x_t − μ)

qui suit, sous l'hypothèse d'un marché intègre et gaussien, une loi du χ² à
k = 4 degrés de liberté. Le seuil critique au niveau α = 1 % vaut 13,28.

- ``NOMINAL`` : d² sous le seuil ;
- ``ALERT``   : un dépassement isolé du seuil à 1 % (pas de gel) ;
- ``FREEZE``  : d² au-dessus du seuil à 0,01 % (23,51) ou 3 alertes sur les 5
  dernières barres → le flux est gelé et le portefeuille reste en cash pendant
  ``cooldown`` barres sans nouvelle alerte.

L'état du filtre est de taille fixe (μ, Σ, Σ⁻¹ préalloués, mises à jour en
place) : aucun historique n'est conservé. Les observations anormales ne sont
pas absorbées telles quelles dans la référence (innovation écrêtée au seuil),
sinon un krach « normaliserait » sa propre signature.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .market import MarketData
from .stats import chi2_ppf

FEATURES = ("rendement", "log_spread", "log_profondeur", "log_volume")


class IntegrityStatus(str, Enum):
    NOMINAL = "NOMINAL"
    ALERT = "ALERT"
    FREEZE = "FREEZE"
    WARMUP = "WARMUP"


@dataclass(frozen=True)
class IntegrityConfig:
    alpha: float = 0.01  # seuil d'alerte (χ² à 99 % : 13,28 pour k = 4)
    alpha_freeze: float = 1e-4  # gel immédiat (χ² à 99,99 % : 23,51)
    persistence_alerts: int = 3  # … ou 3 alertes sur les 5 dernières barres
    persistence_window: int = 5
    ewma_lambda: float = 0.985
    warmup: int = 200
    cooldown: int = 4


@dataclass
class IntegrityReading:
    d2: float
    threshold: float
    status: IntegrityStatus
    contributions: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "d2": round(self.d2, 4),
            "threshold": round(self.threshold, 2),
            "status": self.status.value,
            "contributions": {k: round(v, 3) for k, v in self.contributions.items()},
        }


def ewma_volatility(returns: np.ndarray, lam: float = 0.94, floor: float = 1e-8) -> np.ndarray:
    """Volatilité EWMA causale (RiskMetrics) : vol[t] n'utilise que les rendements < t."""
    var = np.empty_like(returns)
    v = float(np.var(returns[: min(50, returns.size)])) or floor**2
    for t, r in enumerate(returns):
        var[t] = v
        v = lam * v + (1.0 - lam) * r * r
    return np.sqrt(np.maximum(var, floor**2))


def microstructure_features(market: MarketData) -> np.ndarray:
    """Matrice (T, 4) : rendement dévolatilisé, log spread relatif, log profondeur, log volume."""
    r = market.returns
    return np.column_stack(
        [r / ewma_volatility(r), np.log(market.spread / market.close), np.log(market.depth), np.log(market.volume)]
    )


class IntegrityFilter:
    def __init__(self, config: IntegrityConfig | None = None) -> None:
        self.config = config or IntegrityConfig()
        k = len(FEATURES)
        self.k = k
        self.threshold_theory = chi2_ppf(1.0 - self.config.alpha, k)  # 13,28 pour k = 4, α = 1 %
        self.threshold_freeze_theory = chi2_ppf(1.0 - self.config.alpha_freeze, k)
        self.threshold = self.threshold_theory
        self.threshold_freeze = self.threshold_freeze_theory
        self._mu = np.zeros(k)
        self._cov = np.eye(k)
        self._inv = np.eye(k)
        self._dev = np.zeros(k)
        self._tmp = np.zeros(k)
        self._n = 0
        self._recent = 0  # bits des dernières alertes
        self._freeze_left = 0

    @property
    def frozen(self) -> bool:
        return self._freeze_left > 0

    def update(self, x: np.ndarray) -> IntegrityReading:
        cfg = self.config
        x = np.asarray(x, dtype=float)
        np.subtract(x, self._mu, out=self._dev)
        if self._n < cfg.warmup:
            self._absorb(self._dev, 1.0 / (self._n + 1) if self._n < 20 else 1.0 - cfg.ewma_lambda)
            self._n += 1
            return IntegrityReading(0.0, self.threshold, IntegrityStatus.WARMUP, dict.fromkeys(FEATURES, 0.0))

        np.dot(self._inv, self._dev, out=self._tmp)
        d2 = float(np.dot(self._dev, self._tmp))
        contributions = dict(zip(FEATURES, (self._dev * self._tmp).tolist()))

        alert = d2 > self.threshold
        self._recent = (self._recent << 1 | int(alert)) & ((1 << cfg.persistence_window) - 1)
        persistent = bin(self._recent).count("1") >= cfg.persistence_alerts
        if d2 > self.threshold_freeze or persistent:
            self._freeze_left = cfg.cooldown
        elif self._freeze_left > 0:
            self._freeze_left = cfg.cooldown if alert else self._freeze_left - 1

        status = IntegrityStatus.FREEZE if self._freeze_left > 0 else (
            IntegrityStatus.ALERT if alert else IntegrityStatus.NOMINAL
        )
        # Innovation écrêtée : une anomalie n'entre dans la référence qu'au niveau du seuil.
        if alert:
            self._dev *= np.sqrt(self.threshold / d2)
        self._absorb(self._dev, 1.0 - cfg.ewma_lambda)
        self._n += 1
        return IntegrityReading(d2, self.threshold, status, contributions)

    def _absorb(self, dev: np.ndarray, w: float) -> None:
        self._mu += w * dev
        self._cov *= 1.0 - w
        self._cov += w * (1.0 - w) * np.outer(dev, dev)
        self._cov.flat[:: self.k + 1] += 1e-14
        self._inv[:] = np.linalg.inv(self._cov)

    def run(self, features: np.ndarray) -> list[IntegrityReading]:
        return [self.update(row) for row in features]

    def calibrate(self, features: np.ndarray, exclude: np.ndarray | None = None) -> dict[str, float]:
        """Calibre les seuils sur une période de référence (walk-forward : fenêtre d'entraînement).

        Les marchés réels ont des queues plus épaisses que la loi normale : le seuil
        théorique du χ²(4) déclencherait trop de gels. Le seuil retenu est le plus
        strict des deux : max(seuil théorique, quantile empirique au même niveau).
        Le seuil de gel extrapole le quantile empirique à 99,9 % avec le rapport
        théorique entre les deux niveaux quand l'échantillon est trop court.
        ``exclude`` masque les barres à ne pas prendre comme référence (fenêtres
        d'annonces macro, déjà couvertes par le Macro Gate).
        """
        cfg = self.config
        theory, theory_freeze = self.threshold_theory, self.threshold_freeze_theory
        readings = self.run(features)
        keep = np.ones(len(readings), dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
        d2 = np.array([r.d2 for r, k in zip(readings, keep) if k and r.status != IntegrityStatus.WARMUP])
        if d2.size < 500:
            raise ValueError("au moins 500 barres de calibration sont nécessaires")
        alert = max(theory, float(np.quantile(d2, 1.0 - cfg.alpha)))
        if d2.size * cfg.alpha_freeze >= 10:
            freeze = float(np.quantile(d2, 1.0 - cfg.alpha_freeze))
        else:
            freeze = float(np.quantile(d2, 0.999)) * theory_freeze / chi2_ppf(0.999, self.k)
        self.threshold = alert
        self.threshold_freeze = max(theory_freeze, freeze, alert)
        self._recent = 0
        self._freeze_left = 0
        return {
            "chi2_threshold_theoretical": round(theory, 2),
            "chi2_threshold": round(self.threshold, 2),
            "freeze_threshold": round(self.threshold_freeze, 2),
            "calibration_bars": int(d2.size),
        }
