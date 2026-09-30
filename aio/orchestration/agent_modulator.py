"""
AIO v2 – AgentModulatorGate
Budget 60/20/20 – T-norme de Gödel – RLS récursive
"""
from __future__ import annotations
import time, asyncio, copy
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# RLS : Régression Linéaire Récursive par agent
# ─────────────────────────────────────────────────────────────────────────────
class OnlineLinearRegressor:
    """
    Estimateur RLS scalaire  y = beta * x  avec facteur d'oubli λ.
    P0 = 10·I  pour convergence rapide dès les premières observations.
    """

    def __init__(self, l2_reg: float = 10.0, forgetting_factor: float = 0.95):
        self.beta = torch.zeros(1, 1)                # poids estimé
        self.P = torch.eye(1) * l2_reg              # covariance inverse
        self.lam = forgetting_factor
        self.update_count = 0

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x.view(1, -1), self.beta)

    def update(self, x: torch.Tensor, y_true: float) -> None:
        x_vec = x.view(1, 1)
        y = torch.tensor([[y_true]], dtype=torch.float32)
        # Gain de Kalman
        Px = torch.mm(self.P, x_vec)
        denom = self.lam + torch.mm(x_vec.t(), Px).item()
        gain = Px / max(denom, 1e-9)
        # Innovation
        alpha = y - torch.mm(x_vec.t(), self.beta)
        # Mise à jour β et P
        self.beta = self.beta + gain * alpha
        self.P = (self.P - torch.mm(gain, torch.mm(x_vec.t(), self.P))) / self.lam
        self.update_count += 1

    @property
    def prior(self) -> float:
        """Prior clampé pour le softmax."""
        return float(torch.clamp(self.beta, 0.01, 1.0).item())

    def stats(self) -> Dict[str, Any]:
        return {"beta": float(self.beta.item()), "updates": self.update_count}


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper agent client
# ─────────────────────────────────────────────────────────────────────────────
class ClientAgentWrapper(nn.Module):
    def __init__(self, agent_id: str, emb_dim: int = 256):
        super().__init__()
        self.agent_id = agent_id
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# AgentModulatorGate – orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────
class AgentModulatorGate(nn.Module):
    """
    Orchestration 60/20/20 :
      60 % → routage + projection + synergie
      20 % → régression RLS (priors adaptatifs)
      20 % → vérification Gödel + télémétrie
    """

    def __init__(
        self,
        emb_dim: int = 256,
        activation_threshold: float = 0.05,
        godel_threshold: float = 0.50,
        telemetry_queue: Optional[asyncio.Queue] = None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.act_thr = activation_threshold
        self.godel_thr = godel_threshold
        self.tel_q = telemetry_queue

        self.agents = nn.ModuleDict()
        self.agent_ids: List[str] = []
        self.regressors: Dict[str, OnlineLinearRegressor] = {}

        # 60 % – routage
        self.router = nn.Linear(emb_dim, emb_dim)
        self.synergy = nn.Linear(emb_dim, emb_dim)

        self._stats: Dict[str, int] = {
            "forward_passes": 0, "admissible": 0, "pruned": 0, "feedback_steps": 0
        }

    # ── Gestion agents ────────────────────────────────────────────────────────
    def register_client_agent(self, agent_id: str, module: Optional[nn.Module] = None):
        if agent_id not in self.agents:
            self.agents[agent_id] = module or ClientAgentWrapper(agent_id, self.emb_dim)
            self.agent_ids.append(agent_id)
            self.regressors[agent_id] = OnlineLinearRegressor()

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        query_emb: torch.Tensor,
        constraints: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if not self.agent_ids:
            raise RuntimeError("Aucun agent enregistré.")

        t0 = time.perf_counter()
        self._stats["forward_passes"] += 1

        # ── 60 % ORCHESTRATION ────────────────────────────────────────────────
        t_orch = time.perf_counter()
        q_routed = F.relu(self.router(query_emb))              # (B, D)

        reprs = torch.stack(
            [self.agents[aid](query_emb) for aid in self.agent_ids], dim=1
        )                                                       # (B, N, D)

        # Priors RLS intégrés au logit
        priors = torch.tensor(
            [self.regressors[aid].prior for aid in self.agent_ids],
            dtype=torch.float32,
            device=query_emb.device,
        ).unsqueeze(0)                                          # (1, N)

        affinity = (reprs * q_routed.unsqueeze(1)).sum(-1)     # (B, N)
        logits = affinity + torch.log(priors + 1e-8)
        weights = F.softmax(logits, dim=-1)                    # (B, N)

        active = (weights > self.act_thr).float()
        contrib = reprs * (weights * active).unsqueeze(-1)
        synergy_ctx = self.synergy(contrib.sum(1))
        t_orch_ms = (time.perf_counter() - t_orch) * 1e3

        # ── 20 % VÉRIFICATION GÖDEL ──────────────────────────────────────────
        t_ver = time.perf_counter()
        conf = torch.sigmoid(affinity)                         # (B, N) ∈ (0,1)

        if constraints is not None:
            c = F.normalize(constraints, p=2, dim=-1).unsqueeze(1)
            r = F.normalize(reprs, p=2, dim=-1)
            cs = torch.clamp((( r * c).sum(-1) + 1.0) / 2.0, 0.0, 1.0)
        else:
            cs = torch.ones_like(conf)

        # T-norme de Gödel : Tg(a,b) = min(a,b)
        godel = torch.minimum(conf, cs)
        adm_mask = (godel >= self.godel_thr).float()

        n_adm = adm_mask.sum().item()
        n_pru = (1 - adm_mask).sum().item()
        self._stats["admissible"] += int(n_adm)
        self._stats["pruned"] += int(n_pru)

        final_ctx = (contrib * adm_mask.unsqueeze(-1)).sum(1)
        t_ver_ms = (time.perf_counter() - t_ver) * 1e3

        # ── Télémétrie non-bloquante ──────────────────────────────────────────
        payload = {
            "timestamp": time.time(),
            "budget_ms": {
                "orchestration_60pct": round(t_orch_ms, 3),
                "verification_20pct": round(t_ver_ms, 3),
                "total_ms": round((time.perf_counter() - t0) * 1e3, 3),
            },
            "agents": [
                {
                    "id": aid,
                    "weight": round(float(weights[0, i]), 4),
                    "active": bool(active[0, i]),
                    "godel_score": round(float(godel[0, i]), 4),
                    "admissible": bool(adm_mask[0, i]),
                    "rls_beta": round(self.regressors[aid].prior, 4),
                }
                for i, aid in enumerate(self.agent_ids)
            ],
            "stats": dict(self._stats),
        }
        if self.tel_q is not None:
            try:
                self.tel_q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

        return {
            "modulated_context": final_ctx,
            "synergy_context": synergy_ctx,
            "agent_weights": weights,
            "active_mask": active,
            "godel_admissibility": godel,
            "admissibility_mask": adm_mask,
            "n_active": int(active.sum().item()),
            "n_admissible": int(n_adm),
            "telemetry": payload,
        }

    # ── Feedback RLS (20 % adaptation en ligne) ──────────────────────────────
    def feedback_step(self, agent_id: str, success_score: float) -> None:
        """Met à jour l'estimateur RLS de l'agent après retour utilisateur."""
        if agent_id in self.regressors:
            self.regressors[agent_id].update(torch.tensor([[1.0]]), success_score)
            self._stats["feedback_steps"] += 1

    def global_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "agents": {aid: self.regressors[aid].stats() for aid in self.agent_ids},
        }

    # ── Calcul du budget max_tokens filtré (contrôle causal LLM) ─────────────
    def compute_filtered_max_tokens(
        self,
        base_max_tokens: int,
        agent_weights: torch.Tensor,
        admissibility_mask: torch.Tensor,
    ) -> int:
        """
        Réduit max_tokens proportionnellement au ratio d'agents admissibles.
        Garantit que le filtre Gödel module physiquement l'appel LLM.
        """
        n = len(self.agent_ids)
        if n == 0:
            return base_max_tokens
        ratio = admissibility_mask[0].sum().item() / n
        # ratio ∈ [0,1] → max_tokens ∈ [base//4, base]
        filtered = max(base_max_tokens // 4, int(base_max_tokens * (0.25 + 0.75 * ratio)))
        return filtered
