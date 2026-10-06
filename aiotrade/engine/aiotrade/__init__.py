"""AIOTrade : co-pilote de gestion des risques pour le trading algorithmique.

« La rentabilité est une conséquence, la trajectoire admissible sous
contraintes de risque est l'objectif. »

Pipeline : filtre d'intégrité χ² / Mahalanobis → Macro Gate → prévision
(essaim d'agents ou Kronos) → TAP → SCG → Kelly fractionnaire → exécution
maker / taker → décision LONG / SHORT / CASH → XAI Ledger → courtier.
"""
from __future__ import annotations

__version__ = "0.2.0"
