"""Modèles de prévision : génèrent N trajectoires de rendements sur l'horizon H.

- ``AgentSwarmForecaster`` (par défaut, léger et hors ligne) : essaim d'experts
  Trend, Mean-Reversion, Macro et Risque. Chaque trajectoire est tirée d'un
  agent selon sa pondération ; les innovations sont rééchantillonnées dans les
  résidus standardisés récents (queues épaisses réelles conservées).
- Pondération en ligne par régime de volatilité : poids exponentiels sur la
  vraisemblance prédictive de chaque agent, mis à jour à chaque barre sans
  réentraînement (« régression continue »).
- ``KronosForecaster`` (optionnel) : adaptateur vers le modèle de fondation de
  chandeliers Kronos, échantillonnage nucleus (T, top_p). Dépendances lourdes
  (PyTorch) installées à part : ``pip install -e .[kronos]``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Protocol

import numpy as np

from .market import MacroEvent

AGENTS = ("trend", "mean_reversion", "macro", "risk")
REGIMES = ("basse_vol", "vol_normale", "haute_vol")


@dataclass
class MarketContext:
    """Information disponible à l'instant t (aucune donnée future)."""

    timestamps: list[datetime]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    returns: np.ndarray
    sigma: float  # volatilité EWMA par barre
    bar_minutes: int
    upcoming_events: list[MacroEvent] = field(default_factory=list)

    @property
    def now(self) -> datetime:
        return self.timestamps[-1]


class Forecaster(Protocol):
    name: str

    def sample(self, ctx: MarketContext, n_paths: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        """Retourne une matrice (n_paths, horizon) de rendements simples par barre."""
        ...

    def observe(self, ctx: MarketContext, realized_return: float) -> None:
        """Rétroaction en ligne après réalisation du rendement suivant."""
        ...


@dataclass(frozen=True)
class SwarmConfig:
    trend_lookback: int = 192
    mr_lookback: int = 96
    macro_lookback: int = 960
    residual_window: int = 2_000
    learning_rate: float = 0.2
    discount: float = 0.995
    risk_vol_multiplier: float = 1.5
    ridge: float = 0.25  # rétrécissement des coefficients vers 0 (robustesse hors échantillon)
    max_drift_sigma: float = 0.5  # |μ| plafonné à 0,5 σ par barre


def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Moyenne glissante causale (fenêtre tronquée au début)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(1, x.size + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) / (idx - lo)


class AgentSwarmForecaster:
    """Essaim d'experts calibrés sur la fenêtre d'entraînement.

    - Trend : dérive proportionnelle au rendement moyen récent ;
    - Mean-Reversion : rappel proportionnel à l'écart du prix à sa moyenne ;
    - Macro : dérive lente de fond ;
    - Risque : aucune dérive, volatilité majorée (scénario défensif).

    Les coefficients sont estimés par moindres carrés rétrécis sur la fenêtre
    de calibration (walk-forward), puis gelés. Seuls les poids des agents
    évoluent en ligne, séparément pour chaque régime de volatilité.
    """

    name = "agent_swarm"

    def __init__(self, config: SwarmConfig | None = None) -> None:
        self.config = config or SwarmConfig()
        self.vol_edges = (0.0, np.inf)  # bornes basse/haute vol, calibrées
        self.coef = np.array([0.0, 0.0, 0.0, 0.0])
        self.persistence = np.ones(len(AGENTS))
        self._losses = np.zeros((len(REGIMES), len(AGENTS)))
        self._counts = np.zeros(len(REGIMES))

    # -------------------------------------------------------------- agents
    def signals(self, close: np.ndarray, returns: np.ndarray) -> np.ndarray:
        cfg = self.config
        logp = np.log(close[-cfg.mr_lookback :])
        return np.array(
            [
                returns[-cfg.trend_lookback :].mean(),
                logp[-1] - logp.mean(),
                returns[-cfg.macro_lookback :].mean(),
                0.0,
            ]
        )

    def agent_views(self, ctx: MarketContext) -> tuple[np.ndarray, np.ndarray]:
        """(μ, σ) par barre pour chaque agent."""
        cfg = self.config
        s = max(ctx.sigma, 1e-8)
        cap = cfg.max_drift_sigma * s
        mu = np.clip(self.coef * self.signals(ctx.close, ctx.returns), -cap, cap)
        horizon_end = ctx.now + timedelta(minutes=ctx.bar_minutes * 12)
        event_risk = any(ctx.now <= e.time_utc <= horizon_end and e.impact == "HIGH" for e in ctx.upcoming_events)
        sigma = np.array([s, s, s * (2.5 if event_risk else 1.0), s * cfg.risk_vol_multiplier])
        return mu, sigma

    def regime(self, sigma: float) -> int:
        lo, hi = self.vol_edges
        return 0 if sigma < lo else 2 if sigma > hi else 1

    def weights(self, sigma: float) -> np.ndarray:
        k = self.regime(sigma)
        loss = self._losses[k]
        w = np.exp(-self.config.learning_rate * (loss - loss.min()))
        return w / w.sum()

    # ------------------------------------------------------- apprentissage
    def calibrate(self, close: np.ndarray, returns: np.ndarray, sigmas: np.ndarray) -> dict[str, Any]:
        """Estime les coefficients des agents et les bornes de régime sur la fenêtre de calibration."""
        cfg = self.config
        close, returns, sigmas = (np.asarray(a, dtype=float) for a in (close, returns, sigmas))
        if close.size < cfg.macro_lookback + 200:
            raise ValueError("fenêtre de calibration trop courte pour l'essaim d'agents")
        self.vol_edges = (float(np.quantile(sigmas, 1 / 3)), float(np.quantile(sigmas, 2 / 3)))
        logp = np.log(close)
        raw = np.column_stack(
            [
                _rolling_mean(returns, cfg.trend_lookback),
                logp - _rolling_mean(logp, cfg.mr_lookback),
                _rolling_mean(returns, cfg.macro_lookback),
            ]
        )
        start = cfg.macro_lookback
        x, y = raw[start:-1], returns[start + 1 :]
        sxx = (x**2).sum(axis=0)
        beta = (x * y[:, None]).sum(axis=0) / (sxx * (1.0 + cfg.ridge) + 1e-300)
        self.coef = np.array([*beta, 0.0])
        kappa = float(np.clip(-beta[1], 0.0, 1.0))
        self.persistence = np.array([1.0, 1.0 - kappa, 1.0, 1.0])
        self._losses[:] = 0.0
        self._counts[:] = 0.0
        return {"coefficients": {a: float(c) for a, c in zip(AGENTS, self.coef)}, "mean_reversion_speed": kappa}

    def observe(self, ctx: MarketContext, realized_return: float) -> None:
        mu, sigma = self.agent_views(ctx)
        nll = 0.5 * ((realized_return - mu) / sigma) ** 2 + np.log(sigma / max(ctx.sigma, 1e-12))
        k = self.regime(ctx.sigma)
        self._losses[k] = self.config.discount * self._losses[k] + nll
        self._counts[k] += 1

    # --------------------------------------------------------- trajectoires
    def _drift_paths(self, mu: np.ndarray, horizon: int) -> np.ndarray:
        """(agents, H) dérive par pas : le rappel à la moyenne s'amortit le long de l'horizon."""
        return mu[:, None] * self.persistence[:, None] ** np.arange(horizon)[None, :]

    def sample(self, ctx: MarketContext, n_paths: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        mu, sigma = self.agent_views(ctx)
        w = self.weights(ctx.sigma)
        agents = rng.choice(len(AGENTS), size=n_paths, p=w)
        r = ctx.returns[-self.config.residual_window :]
        vol = np.sqrt(np.convolve(r**2, np.ones(48) / 48, mode="same")) + 1e-12
        resid = r / vol
        resid = resid[np.isfinite(resid)]
        resid = (resid - resid.mean()) / (resid.std() + 1e-12)
        eps = rng.choice(resid, size=(n_paths, horizon), replace=True)
        return self._drift_paths(mu, horizon)[agents] + sigma[agents, None] * eps

    def expected_return(self, ctx: MarketContext, horizon: int) -> float:
        mu, _ = self.agent_views(ctx)
        return float(self.weights(ctx.sigma) @ self._drift_paths(mu, horizon).sum(axis=1))

    def describe(self, ctx: MarketContext) -> dict[str, Any]:
        return {
            "model": self.name,
            "regime": REGIMES[self.regime(ctx.sigma)],
            "agent_weights": {a: round(float(w), 4) for a, w in zip(AGENTS, self.weights(ctx.sigma))},
        }


class KronosForecaster:
    """Adaptateur Kronos (https://github.com/shiyu-coder/Kronos).

    ``predictor`` doit exposer ``predict(df, x_timestamp, y_timestamp, pred_len,
    T, top_p, sample_count)`` comme ``KronosPredictor``. S'il n'est pas fourni,
    il est construit à la demande depuis Hugging Face (dépendances optionnelles).
    Chaque trajectoire correspond à un tirage nucleus indépendant.
    """

    name = "kronos"

    def __init__(
        self,
        predictor: Any | None = None,
        model_id: str = "NeoQuasar/Kronos-small",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        temperature: float = 1.0,
        top_p: float = 0.9,
        lookback: int = 400,
        device: str = "cpu",
        loader: Callable[[], Any] | None = None,
    ) -> None:
        self._predictor = predictor
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.temperature = temperature
        self.top_p = top_p
        self.lookback = lookback
        self.device = device
        self._loader = loader

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        if self._loader is not None:
            self._predictor = self._loader()
            return self._predictor
        try:
            from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dépend de l'installation
            raise RuntimeError(
                "Kronos n'est pas installé : clonez https://github.com/shiyu-coder/Kronos, ajoutez-le au "
                "PYTHONPATH et installez les dépendances optionnelles (pip install -e .[kronos])."
            ) from exc
        tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        model = Kronos.from_pretrained(self.model_id)
        self._predictor = KronosPredictor(model, tokenizer, device=self.device, max_context=512)
        return self._predictor

    def sample(self, ctx: MarketContext, n_paths: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas est requis pour Kronos (pip install -e .[kronos])") from exc
        predictor = self._get_predictor()
        n = min(self.lookback, ctx.close.size)
        df = pd.DataFrame(
            {
                "open": ctx.open[-n:],
                "high": ctx.high[-n:],
                "low": ctx.low[-n:],
                "close": ctx.close[-n:],
                "volume": ctx.volume[-n:],
            }
        )
        x_ts = pd.Series(pd.to_datetime(ctx.timestamps[-n:]))
        step = timedelta(minutes=ctx.bar_minutes)
        y_ts = pd.Series(pd.to_datetime([ctx.now + step * (i + 1) for i in range(horizon)]))
        last = float(ctx.close[-1])
        paths = np.empty((n_paths, horizon))
        for i in range(n_paths):
            pred = predictor.predict(
                df=df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=horizon,
                T=self.temperature, top_p=self.top_p, sample_count=1,
            )
            closes = np.concatenate([[last], np.asarray(pred["close"], dtype=float)[:horizon]])
            paths[i] = closes[1:] / closes[:-1] - 1.0
        return paths

    def observe(self, ctx: MarketContext, realized_return: float) -> None:
        return None

    def describe(self, ctx: MarketContext) -> dict[str, Any]:
        return {"model": self.name, "T": self.temperature, "top_p": self.top_p}
