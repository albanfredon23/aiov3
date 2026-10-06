"""Moteur TAP — Trajectoires d'Allocation Plurielles.

Le moteur ne cherche pas à prédire le prochain prix. À chaque décision il
propose plusieurs allocations cibles (tendance, retour à la moyenne,
couverture, et une option cash toujours disponible) et, pour chacune, un
faisceau de trajectoires simulées par bootstrap par blocs de l'historique
récent. Chaque scénario est résumé par sa distribution de résultats plutôt
que par une prévision ponctuelle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stats import historical_cvar


@dataclass(frozen=True)
class TAPConfig:
    lookback: int = 60
    horizon: int = 10
    n_paths: int = 256
    block: int = 5
    max_gross: float = 1.5  # exposition brute maximale proposée par un scénario
    risk_aversion: float = 4.0  # aversion au risque (utilité moyenne-variance)
    hedge_gross: float = 0.6


@dataclass
class Scenario:
    name: str
    label: str
    weights: np.ndarray
    rationale: str
    paths: np.ndarray = field(repr=False)  # (n_paths, horizon + 1) valeur relative, départ 1.0
    expected_return: float = 0.0
    p05: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    worst_path_drawdown: float = 0.0
    cvar: float = 0.0
    score: float = 0.0

    def to_dict(self, assets: tuple[str, ...], max_paths: int = 24) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "weights": {a: round(float(w), 4) for a, w in zip(assets, self.weights)},
            "gross_exposure": round(float(np.abs(self.weights).sum()), 4),
            "rationale": self.rationale,
            "expected_return": round(self.expected_return, 6),
            "p05": round(self.p05, 6),
            "p50": round(self.p50, 6),
            "p95": round(self.p95, 6),
            "worst_path_drawdown": round(self.worst_path_drawdown, 6),
            "cvar": round(self.cvar, 6),
            "score": round(self.score, 6),
            "sample_paths": np.round(self.paths[:max_paths], 5).tolist(),
        }


class TAPEngine:
    def __init__(self, config: TAPConfig | None = None, seed: int = 0) -> None:
        self.config = config or TAPConfig()
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ scénarios
    def propose(self, history: np.ndarray) -> list[Scenario]:
        """``history`` : rendements récents (T, N). Retourne les scénarios évalués."""
        cfg = self.config
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[0] < max(20, cfg.block * 2):
            raise ValueError("historique insuffisant pour le moteur TAP")
        window = history[-cfg.lookback :]
        n = window.shape[1]

        candidates = [
            ("trend", "Tendance", self._trend_weights(window), "Suit le momentum ajusté du risque."),
            (
                "mean_reversion",
                "Retour à la moyenne",
                self._mean_reversion_weights(window),
                "Joue l'écart à la moyenne mobile, long/short.",
            ),
            ("hedge", "Couverture", self._hedge_weights(window), "Minimise la variance, exposition réduite."),
            ("cash", "Cash", np.zeros(n), "Aucune exposition : option de repli toujours admissible."),
        ]
        paths_index = self._bootstrap_indices(window.shape[0])
        return [self._evaluate(name, label, w, why, window, paths_index) for name, label, w, why in candidates]

    def _trend_weights(self, window: np.ndarray) -> np.ndarray:
        vol = window.std(axis=0) + 1e-9
        momentum = window.mean(axis=0) / vol
        raw = np.clip(momentum, 0.0, None)
        if raw.sum() <= 0:
            return np.zeros(window.shape[1])
        return self._scale(raw / vol, self.config.max_gross)

    def _mean_reversion_weights(self, window: np.ndarray) -> np.ndarray:
        prices = np.cumprod(1.0 + window, axis=0)
        ma = prices[-20:].mean(axis=0)
        vol = window[-20:].std(axis=0) * np.sqrt(20) + 1e-9
        zscore = (prices[-1] / ma - 1.0) / vol
        raw = -np.tanh(zscore)
        if np.abs(raw).sum() < 1e-9:
            return np.zeros(window.shape[1])
        return self._scale(raw / (window.std(axis=0) + 1e-9), self.config.max_gross)

    def _hedge_weights(self, window: np.ndarray) -> np.ndarray:
        cov = np.cov(window, rowvar=False) + np.eye(window.shape[1]) * 1e-8
        inv = np.linalg.pinv(cov)
        raw = np.clip(inv @ np.ones(window.shape[1]), 0.0, None)
        if raw.sum() <= 0:
            raw = 1.0 / np.diag(cov)
        return raw / raw.sum() * self.config.hedge_gross

    @staticmethod
    def _scale(raw: np.ndarray, gross: float) -> np.ndarray:
        total = np.abs(raw).sum()
        return raw / total * gross if total > 0 else raw

    # ------------------------------------------------------------- trajectoires
    def _bootstrap_indices(self, length: int) -> np.ndarray:
        cfg = self.config
        n_blocks = int(np.ceil(cfg.horizon / cfg.block))
        starts = self._rng.integers(0, length - cfg.block + 1, size=(cfg.n_paths, n_blocks))
        idx = (starts[:, :, None] + np.arange(cfg.block)[None, None, :]).reshape(cfg.n_paths, -1)
        return idx[:, : cfg.horizon]

    def _evaluate(
        self,
        name: str,
        label: str,
        weights: np.ndarray,
        rationale: str,
        window: np.ndarray,
        paths_index: np.ndarray,
    ) -> Scenario:
        cfg = self.config
        pnl = window @ weights
        simulated = pnl[paths_index]  # (n_paths, horizon)
        paths = np.hstack([np.ones((cfg.n_paths, 1)), np.cumprod(1.0 + simulated, axis=1)])
        finals = paths[:, -1] - 1.0
        peaks = np.maximum.accumulate(paths, axis=1)
        worst_dd = float(np.max(1.0 - paths / peaks))
        cvar = historical_cvar(finals, 0.95)
        expected = float(finals.mean())
        return Scenario(
            name=name,
            label=label,
            weights=weights,
            rationale=rationale,
            paths=paths,
            expected_return=expected,
            p05=float(np.quantile(finals, 0.05)),
            p50=float(np.quantile(finals, 0.50)),
            p95=float(np.quantile(finals, 0.95)),
            worst_path_drawdown=worst_dd,
            cvar=cvar,
            score=expected - 0.5 * cfg.risk_aversion * float(finals.var()),
        )
