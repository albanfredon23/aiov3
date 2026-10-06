"""AIOTrade — co-pilote de gestion des risques pour le trading algorithmique.

Principe : ne pas prédire le marché, mais générer un faisceau de trajectoires
d'allocation (moteur TAP) puis n'exécuter que celles qui respectent des
contraintes strictes (garde-fou SCG), sous la surveillance d'un filtre χ² qui
met le portefeuille en sécurité lors des ruptures de marché.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .chi2 import Chi2Config, Chi2Filter
from .copilot import AIOTradeCopilot, CopilotConfig, simulate
from .market import ASSETS, MarketConfig, Shock, generate_market
from .scg import GuardDecision, RiskLimits, SCGGuard
from .tap import Scenario, TAPConfig, TAPEngine

__all__ = [
    "ASSETS",
    "AIOTradeCopilot",
    "Chi2Config",
    "Chi2Filter",
    "CopilotConfig",
    "GuardDecision",
    "MarketConfig",
    "RiskLimits",
    "SCGGuard",
    "Scenario",
    "Shock",
    "TAPConfig",
    "TAPEngine",
    "generate_market",
    "simulate",
]
