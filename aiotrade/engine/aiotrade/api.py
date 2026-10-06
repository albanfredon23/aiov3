"""API HTTP d'AIOTrade (FastAPI). Simulation uniquement : aucun ordre réel."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .copilot import simulate
from .market import ASSETS, MarketConfig, Shock, generate_market
from .scg import RiskLimits, SCGGuard

app = FastAPI(
    title="AIOTrade API",
    version=__version__,
    description="Co-pilote de gestion des risques : moteur TAP, garde-fou SCG, filtre χ². Simulation uniquement.",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


class LimitsIn(BaseModel):
    max_drawdown: float = Field(0.10, gt=0.01, lt=0.5)
    var_budget: float = Field(0.02, gt=0.001, lt=0.2)
    max_margin_usage: float = Field(0.30, gt=0.01, le=1.0)

    def to_limits(self) -> RiskLimits:
        return RiskLimits(
            max_drawdown=self.max_drawdown, var_budget=self.var_budget, max_margin_usage=self.max_margin_usage
        )


class SimulationIn(BaseModel):
    seed: int = Field(7, ge=0, le=100_000)
    steps: int = Field(600, ge=300, le=2_000)
    shock: Shock = Shock.FLASH_CRASH
    limits: LimitsIn = Field(default_factory=LimitsIn)


class GuardCheckIn(BaseModel):
    weights: dict[str, float]
    equity: float = Field(1_000_000.0, gt=0)
    peak_equity: float = Field(1_000_000.0, gt=0)
    seed: int = Field(7, ge=0, le=100_000)
    limits: LimitsIn = Field(default_factory=LimitsIn)


class ContactIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    company: str = Field("", max_length=160)
    message: str = Field("", max_length=4_000)
    lead_magnet: bool = False
    consent: bool


@lru_cache(maxsize=64)
def _cached_simulation(seed: int, steps: int, shock: str, max_dd: float, var_budget: float, margin: float) -> dict:
    limits = RiskLimits(max_drawdown=max_dd, var_budget=var_budget, max_margin_usage=margin)
    result = simulate(MarketConfig(steps=steps, seed=seed, shock=Shock(shock)), limits=limits)
    return result.to_dict()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "mode": "simulation"}


@app.get("/api/limits")
def default_limits() -> dict:
    lim = RiskLimits()
    return {
        "assets": list(ASSETS),
        "max_drawdown": lim.max_drawdown,
        "var_budget": lim.var_budget,
        "var_confidence": lim.var_confidence,
        "max_margin_usage": lim.max_margin_usage,
        "margin_rates": dict(zip(ASSETS, lim.margin_rates)),
        "shocks": [s.value for s in Shock],
    }


@app.post("/api/simulate")
def run_simulation(body: SimulationIn) -> dict:
    lim = body.limits
    return _cached_simulation(
        body.seed, body.steps, body.shock.value, lim.max_drawdown, lim.var_budget, lim.max_margin_usage
    )


@app.post("/api/guard/check")
def guard_check(body: GuardCheckIn) -> dict:
    unknown = sorted(set(body.weights) - set(ASSETS))
    if unknown:
        raise HTTPException(status_code=422, detail=f"actifs inconnus : {', '.join(unknown)}")
    weights = np.array([body.weights.get(a, 0.0) for a in ASSETS])
    history = generate_market(MarketConfig(steps=400, seed=body.seed)).returns[-120:]
    guard = SCGGuard(body.limits.to_limits(), n_assets=len(ASSETS))
    try:
        decision = guard.check_allocation(weights, history, body.equity, body.peak_equity)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return decision.to_dict()


_contact_hits: dict[str, deque] = defaultdict(deque)
_CONTACT_LIMIT = 5  # requêtes par fenêtre
_CONTACT_WINDOW_S = 600


@app.post("/api/contact", status_code=202)
def contact(body: ContactIn, request: Request) -> dict:
    """Démonstration : la demande est validée puis ignorée (aucune donnée n'est conservée ni journalisée)."""
    if not body.consent:
        raise HTTPException(status_code=422, detail="Le consentement au traitement des données est requis.")
    client = request.client.host if request.client else "inconnu"
    now = time.monotonic()
    hits = _contact_hits[client]
    while hits and now - hits[0] > _CONTACT_WINDOW_S:
        hits.popleft()
    if len(hits) >= _CONTACT_LIMIT:
        raise HTTPException(status_code=429, detail="Trop de demandes, réessayez plus tard.")
    hits.append(now)
    return {"status": "accepted", "stored": False}
