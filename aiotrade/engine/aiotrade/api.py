"""API HTTP d'AIOTrade (FastAPI). Démonstration en simulation : aucun ordre n'est passé."""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from . import __version__
from .backtest import ARM_LABELS, ARMS, TARGETS, BacktestConfig, arm_summary, run_fold, scg_rates
from .copilot import CopilotConfig, MarketQuote
from .forecast import AgentSwarmForecaster
from .integrity import IntegrityConfig
from .ledger import verify_chain
from .macro_gate import MacroGateConfig
from .market import MarketConfig, Shock, generate_market, iso_utc
from .scg import RiskMandate
from .sizing import SizingConfig

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "benchmark.json"
DEMO_CALIBRATION_BARS = 12 * 21 * 96  # 12 mois de calibration, comme le protocole
DEMO_TEST_BARS = 480  # 5 jours ouvrés en barres de 15 min

app = FastAPI(
    title="AIOTrade API",
    version=__version__,
    description=(
        "Co-pilote de gestion des risques : filtre d'intégrité χ², Macro Gate, TAP, SCG, Kelly fractionnaire, "
        "exécution Almgren-Chriss, XAI Ledger. Démonstration en simulation, aucun ordre réel."
    ),
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)


# ------------------------------------------------------------------ limites
_hits: dict[tuple[str, str], deque] = defaultdict(deque)


def _rate_limit(request: Request, bucket: str, limit: int, window_s: float) -> None:
    client = request.client.host if request.client else "inconnu"
    now = time.monotonic()
    hits = _hits[(bucket, client)]
    while hits and now - hits[0] > window_s:
        hits.popleft()
    if len(hits) >= limit:
        raise HTTPException(status_code=429, detail="Trop de demandes, réessayez plus tard.")
    hits.append(now)


# ------------------------------------------------------------------ schémas
class MandateIn(BaseModel):
    max_drawdown: float = Field(0.10, gt=0.01, lt=0.5)
    max_loss_per_trade: float = Field(0.01, gt=0.001, le=0.05)
    max_leverage: float = Field(1.5, gt=0.1, le=5.0)
    min_admissibility: float = Field(0.75, ge=0.5, le=1.0)
    kelly_fraction: float = Field(0.20, gt=0.0, le=0.25)

    def copilot_config(self) -> CopilotConfig:
        mandate = RiskMandate(
            max_drawdown=self.max_drawdown,
            max_loss_per_trade=self.max_loss_per_trade,
            max_leverage=self.max_leverage,
            min_admissibility=self.min_admissibility,
        )
        sizing = SizingConfig(
            kelly_fraction=self.kelly_fraction,
            max_risk_per_trade=self.max_loss_per_trade,
            max_leverage=self.max_leverage,
        )
        return CopilotConfig(mandate=mandate, sizing=sizing)


class SimulationIn(BaseModel):
    seed: int = Field(7, ge=0, le=100_000)
    shock: Shock = Shock.FLASH_CRASH
    mandate: MandateIn = Field(default_factory=MandateIn)


class SCGCheckIn(BaseModel):
    direction: str = Field("LONG", pattern="^(LONG|SHORT)$")
    exposure: float = Field(1.0, gt=0.0, le=10.0)
    equity: float = Field(100_000.0, gt=0)
    peak_equity: float = Field(100_000.0, gt=0)
    seed: int = Field(7, ge=0, le=100_000)
    mandate: MandateIn = Field(default_factory=MandateIn)


class LedgerIn(BaseModel):
    jsonl: str = Field(..., max_length=2_000_000)


class ContactIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: str = Field(..., max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    company: str = Field("", max_length=160)
    message: str = Field("", max_length=4_000)
    lead_magnet: bool = False
    consent: bool


# ------------------------------------------------------------------ simulation
def _r(x: float, nd: int = 6) -> float | None:
    return round(float(x), nd) if np.isfinite(x) else None


@lru_cache(maxsize=64)
def _simulation(seed: int, shock: str, mandate_json: str) -> dict[str, Any]:
    mandate = MandateIn.model_validate_json(mandate_json)
    total = DEMO_CALIBRATION_BARS + DEMO_TEST_BARS
    shock_at = DEMO_CALIBRATION_BARS + DEMO_TEST_BARS // 2 if shock != Shock.NONE.value else None
    market = generate_market(MarketConfig(bars=total, seed=seed, shock=Shock(shock), shock_at=shock_at))
    cfg = BacktestConfig(copilot=mandate.copilot_config(), seed=seed)
    fold = run_fold(market, 0, DEMO_CALIBRATION_BARS, total, cfg)
    t0, t1 = DEMO_CALIBRATION_BARS, total
    bpy = market.bars_per_year
    records = fold.ledger.records()
    verify = fold.ledger.verify()
    window = market.shock_window
    events = [
        {**e.to_dict(), "index": i}
        for e in market.events
        for i in [next((k for k, t in enumerate(market.timestamps[t0:t1]) if t >= e.time_utc), None)]
        if i is not None and market.timestamps[t0] <= e.time_utc <= market.timestamps[t1 - 1]
    ]
    return {
        "protocol": {
            "data": "marché synthétique (simulation)",
            "symbol": market.symbol,
            "bar_minutes": market.bar_minutes,
            "calibration_bars": DEMO_CALIBRATION_BARS,
            "test_bars": DEMO_TEST_BARS,
            "seed": seed,
            "shock": shock,
        },
        "labels": ARM_LABELS,
        "summary": {a: {k: _r(v, 4) if isinstance(v, float) else v for k, v in arm_summary(fold, a, bpy, cfg.initial_equity).items()} for a in ARMS},
        "scg": {**fold.scg_stats, **{k: _r(v, 4) for k, v in scg_rates(fold.scg_stats).items()}},
        "calibration": fold.calibration,
        "series": {
            "time": [iso_utc(t) for t in market.timestamps[t0:t1]],
            "price": [_r(x, 6) for x in market.close[t0:t1]],
            "equity": {a: [_r(x / cfg.initial_equity, 6) for x in fold.equity[a][1:]] for a in ARMS},
            "exposure_d": [_r(x, 4) for x in fold.exposure_d],
            "d2": [_r(x, 3) for x in fold.integrity_d2],
            "status": fold.integrity_status,
            "gate": fold.gate_states,
        },
        "thresholds": {"alert": fold.calibration["chi2_threshold"], "freeze": fold.calibration["freeze_threshold"]},
        "shock_window": [window[0] - t0, window[1] - t0] if window else None,
        "events": events,
        "ledger": {
            "records": len(fold.ledger),
            "head_hash": fold.ledger.head_hash,
            "verified": verify.ok,
            "latest": records[-1] if records else None,
            "sample": [r for r in records if r["decision"] != "CASH"][-2:] + [r for r in records if r["status"] == "FREEZE"][:1],
            "breakdown": _decision_breakdown(records),
        },
    }


def _decision_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    """Répartit les décisions du bras D par motif (exposition ou raison du cash)."""
    out = dict.fromkeys(("exposed", "freeze", "macro", "edge", "scg", "other"), 0)
    for r in records:
        reasons = " ".join(r.get("reasons", []))
        if r["decision"] != "CASH":
            key = "exposed"
        elif r["status"] in ("FREEZE", "WARMUP"):
            key = "freeze"
        elif r.get("macro_gate") == "BLACKOUT":
            key = "macro"
        elif "AVANTAGE INSUFFISANT" in reasons:
            key = "edge"
        elif "SCG" in reasons or "TAILLE NULLE" in reasons:
            key = "scg"
        else:
            key = "other"
        out[key] += 1
    return out


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "mode": "simulation"}


@app.get("/api/config")
def config() -> dict:
    return {
        "mandate": asdict(RiskMandate()),
        "sizing": asdict(SizingConfig()),
        "macro_gate": asdict(MacroGateConfig()),
        "integrity": asdict(IntegrityConfig()),
        "tap": asdict(CopilotConfig().tap),
        "shocks": [s.value for s in Shock],
        "arms": ARM_LABELS,
        "targets": {k: list(v) if isinstance(v, tuple) else v for k, v in TARGETS.items()},
    }


@app.post("/api/simulate")
def simulate(body: SimulationIn, request: Request) -> dict:
    _rate_limit(request, "simulate", 30, 60)
    return _simulation(body.seed, body.shock.value, body.mandate.model_dump_json())


@app.post("/api/scg/check")
def scg_check(body: SCGCheckIn, request: Request) -> dict:
    """Soumet une exposition au garde-fou SCG sur le faisceau TAP de la dernière barre d'un marché simulé."""
    _rate_limit(request, "scg", 60, 60)
    from .backtest import _Series  # import local : utilitaire interne

    market = generate_market(MarketConfig(bars=2_500, seed=body.seed))
    s = _Series(market)
    t = market.bars - 1
    cfg = body.mandate.copilot_config()
    forecaster = AgentSwarmForecaster()
    forecaster.calibrate(market.close[:t], s.returns[:t], s.sigma_close[:t])
    from .copilot import Copilot

    copilot = Copilot(forecaster, cfg, market.symbol)
    ctx = s.context(t)
    rng = np.random.default_rng(body.seed)
    bundle = copilot.planner.plan(forecaster, ctx, rng)
    d = 1 if body.direction == "LONG" else -1
    stop = copilot.planner.stop_distance(bundle, d, ctx.sigma, cfg.stop_quantile)
    quote = MarketQuote(float(market.close[t]), float(market.spread[t]), float(s.avg_volume[t]))
    cost = copilot.execution.round_trip(body.equity * body.exposure, quote.price, quote.spread, ctx.sigma, quote.avg_volume)
    scale = RiskMandate.horizon_scale(bundle.horizon, market.bar_minutes)
    try:
        report = copilot.scg.evaluate(bundle, d * body.exposure, body.equity, body.peak_equity, stop, cost, scale)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    curves = copilot.scg.equity_paths(bundle, d * body.exposure, stop, cost)
    quantiles = np.quantile(curves, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    return {
        **report.to_dict(),
        "trajectories": bundle.n_paths,
        "horizon_bars": bundle.horizon,
        "round_trip_cost_bps": round(cost * 1e4, 3),
        "equity_quantiles": {q: [round(float(x), 6) for x in row] for q, row in zip(("p05", "p25", "p50", "p75", "p95"), quantiles)},
    }


@app.get("/api/benchmark")
def benchmark() -> dict:
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="rapport de benchmark absent : python -m aiotrade benchmark")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@app.post("/api/ledger/verify")
def ledger_verify(body: LedgerIn, request: Request) -> dict:
    """Vérifie la chaîne d'empreintes SHA-256 d'un registre XAI (JSON Lines)."""
    _rate_limit(request, "ledger", 30, 60)
    try:
        records = [json.loads(line) for line in body.jsonl.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"JSON invalide ligne {exc.lineno}") from exc
    if not all(isinstance(r, dict) for r in records):
        raise HTTPException(status_code=422, detail="chaque ligne doit être un objet JSON")
    return verify_chain(records).to_dict()


@app.post("/api/contact", status_code=202)
def contact(body: ContactIn, request: Request) -> dict:
    """Démonstration : la demande est validée puis ignorée (aucune donnée n'est conservée ni journalisée)."""
    if not body.consent:
        raise HTTPException(status_code=422, detail="Le consentement au traitement des données est requis.")
    _rate_limit(request, "contact", 5, 600)
    return {"status": "accepted", "stored": False}
