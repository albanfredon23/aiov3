"""
AIO v2 – GodelCompiler
4 mutations M1-M4 stables, sans crash ni fuite mémoire.
"""
from __future__ import annotations
import copy
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


class GodelCompiler:
    """Génère et filtre des candidats mutants pour un module nn."""

    def __init__(self, target_speedup: float = 1.05):
        self.target_speedup = target_speedup
        self.history: List[Dict[str, Any]] = []

    # ── Entrée principale ─────────────────────────────────────────────────────
    def generate_mutations(self, module: nn.Module) -> List[nn.Module]:
        mutations = []
        for name, fn in [
            ("M1_fuse", self._m1_fuse),
            ("M2_prune", self._m2_prune),
            ("M3_lowrank", self._m3_lowrank),
            ("M4_bn_fold", self._m4_bn_fold),
        ]:
            try:
                cand = fn(module)
                if cand is not None:
                    mutations.append(cand)
                    self.history.append({"mutation": name, "status": "generated"})
            except Exception as exc:
                self.history.append({"mutation": name, "status": "skipped", "reason": str(exc)})
        return mutations

    # ── M1 : Fusion couches linéaires consécutives ────────────────────────────
    def _m1_fuse(self, module: nn.Module) -> Optional[nn.Module]:
        if not isinstance(module, nn.Sequential):
            return None
        linears = [(i, m) for i, m in enumerate(module) if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            return None
        i0, l0 = linears[0]
        i1, l1 = linears[1]
        if l0.out_features != l1.in_features:
            return None
        fused_seq = copy.deepcopy(module)
        fused = nn.Linear(l0.in_features, l1.out_features, bias=(l0.bias is not None or l1.bias is not None))
        with torch.no_grad():
            fused.weight.copy_(torch.mm(l1.weight, l0.weight))
            b0 = l0.bias if l0.bias is not None else torch.zeros(l0.out_features)
            b1 = l1.bias if l1.bias is not None else torch.zeros(l1.out_features)
            fused.bias.copy_(torch.mv(l1.weight, b0) + b1)
        # Reconstruire la séquence sans les deux couches fusionnées
        layers = []
        skip = {i0, i1}
        for i, layer in enumerate(module):
            if i == i0:
                layers.append(fused)
            elif i not in skip:
                layers.append(copy.deepcopy(layer))
        return nn.Sequential(*layers)

    # ── M2 : Élagage L1 (deepcopy + PyTorch prune officiel) ──────────────────
    def _m2_prune(self, module: nn.Module, amount: float = 0.20) -> Optional[nn.Module]:
        """Élagage sans reconstruction par réflexion → aucun TypeError possible."""
        cand = copy.deepcopy(module)
        pruned_any = False
        for _, sub in cand.named_modules():
            if isinstance(sub, nn.Linear) and sub.weight is not None:
                prune.l1_unstructured(sub, name="weight", amount=amount)
                prune.remove(sub, "weight")   # fige le masque
                pruned_any = True
        return cand if pruned_any else None

    # ── M3 : Approximation low-rank SVD ──────────────────────────────────────
    def _m3_lowrank(self, module: nn.Module, rank_frac: float = 0.5) -> Optional[nn.Module]:
        cand = copy.deepcopy(module)
        applied = False
        for _, sub in cand.named_modules():
            if not isinstance(sub, nn.Linear):
                continue
            W = sub.weight.data
            if W.dim() != 2:
                continue
            rank = max(1, int(min(W.shape) * rank_frac))
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except RuntimeError:
                continue
            W_approx = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
            sub.weight.data.copy_(W_approx)
            applied = True
        return cand if applied else None

    # ── M4 : Folding BatchNorm dans Linear ───────────────────────────────────
    def _m4_bn_fold(self, module: nn.Module) -> Optional[nn.Module]:
        if not isinstance(module, nn.Sequential):
            return None
        layers_out = []
        i = 0
        applied = False
        mods = list(module.children())
        while i < len(mods):
            layer = mods[i]
            if (
                isinstance(layer, nn.Linear)
                and i + 1 < len(mods)
                and isinstance(mods[i + 1], nn.BatchNorm1d)
            ):
                bn = mods[i + 1]
                W = layer.weight.data.clone()
                b = layer.bias.data.clone() if layer.bias is not None else torch.zeros(W.shape[0])
                rv = bn.running_var if bn.running_var is not None else torch.ones(W.shape[0])
                rm = bn.running_mean if bn.running_mean is not None else torch.zeros(W.shape[0])
                std = (rv + bn.eps).sqrt()
                W_f = W / std.unsqueeze(1)
                b_f = (b - rm) / std
                if bn.affine:
                    W_f = W_f * bn.weight.unsqueeze(1)
                    b_f = b_f * bn.weight + bn.bias
                new_linear = copy.deepcopy(layer)
                new_linear.weight.data.copy_(W_f)
                new_linear.bias = nn.Parameter(b_f)
                layers_out.append(new_linear)
                i += 2
                applied = True
            else:
                layers_out.append(copy.deepcopy(layer))
                i += 1
        return nn.Sequential(*layers_out) if applied else None
