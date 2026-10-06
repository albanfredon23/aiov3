"""Modèle d'exécution : impact de marché Almgren-Chriss et ordres maker / taker.

Coût d'impact (fraction du prix) d'un ordre de volume Q quand le volume moyen
par barre vaut V et la volatilité par barre σ :

    impact = γ · σ · (Q / V)^α

Un ordre taker (au marché) paie le demi-spread, le slippage, l'impact et la
commission taker. Un ordre maker (limite passive au meilleur prix) ne paie
ni spread ni impact, mais n'est exécuté que si le marché vient le chercher :
la probabilité de remplissage est simulée à partir de la barre suivante (le
prix traverse la limite) et de la position dans la file d'attente. Un ordre
maker non rempli est remplacé par un ordre taker (sorties de risque toujours
en taker).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionConfig:
    impact_gamma: float = 0.8
    impact_alpha: float = 0.6
    slippage_bps: float = 0.1
    taker_commission_bps: float = 0.25
    maker_commission_bps: float = 0.0
    queue_fill_probability: float = 0.35  # si le prix touche la limite sans la traverser

    def __post_init__(self) -> None:
        if self.impact_gamma < 0 or not 0 < self.impact_alpha <= 1:
            raise ValueError("impact_gamma >= 0 et 0 < impact_alpha <= 1 requis")
        if not 0 <= self.queue_fill_probability <= 1:
            raise ValueError("queue_fill_probability doit être dans [0 ; 1]")


@dataclass(frozen=True)
class CostBreakdown:
    spread: float
    slippage: float
    impact: float
    commission: float

    @property
    def total(self) -> float:
        return self.spread + self.slippage + self.impact + self.commission

    def to_dict(self) -> dict:
        return {
            "spread_bps": round(self.spread * 1e4, 4),
            "slippage_bps": round(self.slippage * 1e4, 4),
            "impact_bps": round(self.impact * 1e4, 4),
            "commission_bps": round(self.commission * 1e4, 4),
            "total_bps": round(self.total * 1e4, 4),
        }


class ExecutionModel:
    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def impact(self, order_notional: float, avg_volume: float, sigma: float) -> float:
        """Impact Almgren-Chriss en fraction du prix."""
        if order_notional <= 0 or avg_volume <= 0:
            return 0.0
        cfg = self.config
        return cfg.impact_gamma * sigma * (order_notional / avg_volume) ** cfg.impact_alpha

    def cost(
        self, order_notional: float, price: float, spread: float, sigma: float, avg_volume: float, maker: bool = False
    ) -> CostBreakdown:
        """Coût d'un aller simple, en fraction du notionnel."""
        cfg = self.config
        if maker:
            return CostBreakdown(0.0, 0.0, 0.0, cfg.maker_commission_bps / 1e4)
        return CostBreakdown(
            spread=0.5 * spread / price,
            slippage=cfg.slippage_bps / 1e4,
            impact=self.impact(order_notional, avg_volume, sigma),
            commission=cfg.taker_commission_bps / 1e4,
        )

    def round_trip(self, order_notional: float, price: float, spread: float, sigma: float, avg_volume: float) -> float:
        """Coût aller-retour prudent (deux exécutions taker), utilisé par le SCG."""
        return 2.0 * self.cost(order_notional, price, spread, sigma, avg_volume).total

    def maker_fill(
        self, side: int, limit_price: float, next_low: float, next_high: float, rng: np.random.Generator
    ) -> bool:
        """Remplissage simulé d'une limite passive sur la barre suivante.

        Achat (side = +1) : rempli si le plus bas passe sous la limite ; s'il ne
        fait que la toucher, rempli avec la probabilité de file d'attente.
        """
        if side > 0:
            through, touch = next_low < limit_price, next_low <= limit_price
        else:
            through, touch = next_high > limit_price, next_high >= limit_price
        if through:
            return True
        return bool(touch and rng.random() < self.config.queue_fill_probability)

    def fill_probability(self, side: int, limit_price: float, price: float, sigma: float) -> float:
        """Probabilité a priori qu'une limite à distance d du prix soit atteinte sur une barre."""
        distance = abs(price - limit_price) / price
        if sigma <= 0:
            return 0.0
        # Probabilité d'atteinte d'une barrière par un mouvement brownien sur une barre.
        from math import erfc, sqrt

        return float(min(1.0, erfc(distance / (sigma * sqrt(2.0)))))
