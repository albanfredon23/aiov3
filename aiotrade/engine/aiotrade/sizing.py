"""Dimensionnement : critère de Kelly fractionnaire plafonné par le SCG.

    f* = λ · (p · b − (1 − p)) / b,   0 < λ ≤ 0,25

p est la probabilité de gain et b le rapport gain moyen / perte moyenne, tous
deux mesurés sur le faisceau TAP (stop et frictions inclus). Kelly prudent :
p est remplacé par sa borne basse de confiance p − z·√(p(1 − p)/N), pour
qu'un avantage dû au seul bruit d'échantillonnage du faisceau ne soit jamais
misé. f* est la part
d'équité mise en risque ; la taille en notionnel s'obtient en divisant par la
perte au stop. Elle est ensuite plafonnée par le mandat (perte maximale par
trade, levier maximal L_max) et arrondie **vers le bas** au pas de lot, pour
que l'arrondi ne puisse jamais dépasser le plafond.

Corrections par rapport au code de référence : prise en compte du prix de
l'actif et de la taille de contrat, arrondi par défaut au pas de lot, faisceau
vide ou sans perte géré sans division par zéro, horodatages UTC conscients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.20  # λ
    max_risk_per_trade: float = 0.01  # 1 % de l'équité au stop
    max_leverage: float = 1.5
    contract_size: float = 100_000.0  # unités de l'actif par lot (FX standard)
    min_lot: float = 0.01
    max_lot: float = 10.0
    lot_step: float = 0.01
    confidence_z: float = 1.0  # borne basse de p à 1 écart-type

    def __post_init__(self) -> None:
        if not 0 < self.kelly_fraction <= 0.25:
            raise ValueError("kelly_fraction (λ) doit être dans ]0 ; 0,25]")
        if not 0 < self.max_risk_per_trade < 0.2:
            raise ValueError("max_risk_per_trade doit être dans ]0 ; 0,2[")
        if self.max_leverage <= 0 or self.contract_size <= 0:
            raise ValueError("max_leverage et contract_size doivent être positifs")
        if not 0 < self.lot_step <= self.min_lot <= self.max_lot:
            raise ValueError("il faut 0 < lot_step <= min_lot <= max_lot")


@dataclass(frozen=True)
class KellyEstimate:
    p: float
    b: float
    full_kelly: float
    fraction: float  # λ · f*, borné à [0, 1]
    p_lower: float = 0.0

    def to_dict(self) -> dict:
        return {
            "p_win": round(self.p, 4),
            "p_win_lower_bound": round(self.p_lower, 4),
            "payoff_ratio": round(self.b, 4) if math.isfinite(self.b) else None,
            "full_kelly": round(self.full_kelly, 4),
            "fractional_kelly": round(self.fraction, 4),
        }


@dataclass(frozen=True)
class PositionSize:
    lots: float
    units: float
    notional: float
    exposure: float  # notionnel / équité
    risk_fraction: float  # perte au stop / équité
    capped_by: str

    def to_dict(self) -> dict:
        return {
            "lots": self.lots,
            "units": round(self.units, 6),
            "notional": round(self.notional, 2),
            "exposure": round(self.exposure, 4),
            "risk_fraction": round(self.risk_fraction, 5),
            "capped_by": self.capped_by,
        }


ZERO_SIZE = PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "AUCUNE_POSITION")


class FractionalKellySizer:
    def __init__(self, config: SizingConfig | None = None) -> None:
        self.config = config or SizingConfig()

    def estimate(self, outcomes: np.ndarray) -> KellyEstimate:
        """Kelly à partir des résultats nets par unité de notionnel (une valeur par trajectoire)."""
        outcomes = np.asarray(outcomes, dtype=float)
        outcomes = outcomes[np.isfinite(outcomes)]
        if outcomes.size == 0:
            return KellyEstimate(0.0, 0.0, 0.0, 0.0)
        wins, losses = outcomes[outcomes > 0], -outcomes[outcomes < 0]
        n = outcomes.size
        p = wins.size / n
        p_low = max(0.0, p - self.config.confidence_z * math.sqrt(p * (1.0 - p) / n))
        if wins.size == 0:
            return KellyEstimate(p, 0.0, -1.0, 0.0, p_low)
        if losses.size == 0:
            # Aucune trajectoire perdante : on ne mise jamais plus que la fraction λ.
            return KellyEstimate(p, math.inf, 1.0, self.config.kelly_fraction, p_low)
        b = float(wins.mean() / losses.mean())
        full = (p_low * b - (1.0 - p_low)) / b
        return KellyEstimate(p, b, full, float(np.clip(self.config.kelly_fraction * full, 0.0, 1.0)), p_low)

    def size(
        self,
        kelly: KellyEstimate,
        equity: float,
        price: float,
        loss_at_stop: float,
        max_risk: float | None = None,
        max_leverage: float | None = None,
        exposure_multiplier: float = 1.0,
    ) -> PositionSize:
        """Taille de position en lots, toujours sous les plafonds du mandat.

        ``loss_at_stop`` : perte relative par unité de notionnel si le stop est
        touché (distance au stop + frictions aller-retour).
        """
        cfg = self.config
        if equity <= 0 or price <= 0 or loss_at_stop <= 0:
            raise ValueError("equity, price et loss_at_stop doivent être strictement positifs")
        if kelly.fraction <= 0 or exposure_multiplier <= 0:
            return ZERO_SIZE
        max_risk = min(cfg.max_risk_per_trade, max_risk if max_risk is not None else cfg.max_risk_per_trade)
        max_leverage = min(cfg.max_leverage, max_leverage if max_leverage is not None else cfg.max_leverage)

        risk = min(kelly.fraction, max_risk)
        capped_by = "KELLY_FRACTIONNAIRE" if kelly.fraction < max_risk else "PERTE_MAX_PAR_TRADE"
        exposure = risk / loss_at_stop
        if exposure > max_leverage:
            exposure, capped_by = max_leverage, "LEVIER_MAX"
        exposure *= exposure_multiplier
        if exposure_multiplier < 1.0:
            capped_by = "MACRO_GATE"

        lot_value = cfg.contract_size * price
        raw_lots = exposure * equity / lot_value
        lots = math.floor(raw_lots / cfg.lot_step + 1e-9) * cfg.lot_step
        if lots > cfg.max_lot:
            lots, capped_by = cfg.max_lot, "LOT_MAX"
        lots = round(lots, 8)
        if lots < cfg.min_lot:
            return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "SOUS_LOT_MIN")
        units = lots * cfg.contract_size
        notional = units * price
        return PositionSize(lots, units, notional, notional / equity, notional / equity * loss_at_stop, capped_by)

    def shrink(self, size: PositionSize, equity: float, price: float, loss_at_stop: float) -> PositionSize:
        """Divise la position par deux (au pas de lot inférieur)."""
        cfg = self.config
        lots = math.floor(size.lots / 2.0 / cfg.lot_step + 1e-9) * cfg.lot_step
        lots = round(lots, 8)
        if lots < cfg.min_lot:
            return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "SOUS_LOT_MIN")
        units = lots * cfg.contract_size
        notional = units * price
        return PositionSize(lots, units, notional, notional / equity, notional / equity * loss_at_stop, "SCG")
