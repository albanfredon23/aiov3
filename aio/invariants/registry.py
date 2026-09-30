"""
AIO v2 – InvariantRegistry I1-I6
Validation structurelle des candidats mutants.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple, Dict


class InvariantRegistry:
    """Valide les 6 invariants exécutables."""

    @staticmethod
    def I1_shape_type(cand: nn.Module, ref: nn.Module) -> Tuple[bool, str]:
        for (n1, p1), (n2, p2) in zip(cand.named_parameters(), ref.named_parameters()):
            if n1 != n2:
                return False, f"Nom diverge: {n1} vs {n2}"
            if p1.shape != p2.shape:
                return False, f"Shape diverge [{n1}]: {p1.shape} vs {p2.shape}"
            if p1.dtype != p2.dtype:
                return False, f"Dtype diverge [{n1}]: {p1.dtype} vs {p2.dtype}"
        return True, "OK"

    @staticmethod
    def I2_numerical(y_c: torch.Tensor, y_r: torch.Tensor, eps: float = 1e-4) -> Tuple[bool, str]:
        ok = torch.allclose(y_c, y_r, atol=eps)
        diff = float((y_c - y_r).abs().max())
        return ok, f"max_diff={diff:.2e} tol={eps:.2e}"

    @staticmethod
    def I3a_grad_input(cand: nn.Module, ref: nn.Module, x: torch.Tensor, eps: float = 1e-3) -> Tuple[bool, str]:
        xc = x.clone().detach().requires_grad_(True)
        xr = x.clone().detach().requires_grad_(True)
        cand(xc).sum().backward()
        ref(xr).sum().backward()
        if xc.grad is None or xr.grad is None:
            return True, "grad=None (skip)"
        ok = torch.allclose(xc.grad, xr.grad, atol=eps)
        diff = float((xc.grad - xr.grad).abs().max())
        return ok, f"max_grad_diff={diff:.2e}"

    @staticmethod
    def I3b_grad_params(cand: nn.Module, ref: nn.Module, x: torch.Tensor, eps: float = 1e-3) -> Tuple[bool, str]:
        for m in [cand, ref]:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        cand(x).sum().backward()
        ref(x.detach().clone().requires_grad_(True)).sum().backward()
        for (nc, pc), (nr, pr) in zip(cand.named_parameters(), ref.named_parameters()):
            if pc.grad is not None and pr.grad is not None:
                if not torch.allclose(pc.grad, pr.grad, atol=eps):
                    d = float((pc.grad - pr.grad).abs().max())
                    return False, f"grad_param diverge [{nc}] diff={d:.2e}"
        return True, "OK"

    @staticmethod
    def I4_determinism(m: nn.Module, x: torch.Tensor, iters: int = 3) -> Tuple[bool, str]:
        m.eval()
        outs = []
        with torch.no_grad():
            for _ in range(iters):
                outs.append(m(x.clone()))
        for i in range(len(outs) - 1):
            if not torch.allclose(outs[i], outs[i + 1], atol=1e-5):
                return False, "Sorties non déterministes"
        return True, "OK"

    @staticmethod
    def I5_finiteness(m: nn.Module, x: torch.Tensor) -> Tuple[bool, str]:
        with torch.no_grad():
            y = m(x)
        if torch.isnan(y).any():
            return False, "NaN détecté"
        if torch.isinf(y).any():
            return False, "Inf détecté"
        return True, "OK"

    @staticmethod
    def I6_bounds(m: nn.Module, x: torch.Tensor, lo: float = -1e4, hi: float = 1e4) -> Tuple[bool, str]:
        with torch.no_grad():
            y = m(x)
        if (y < lo).any():
            return False, f"Valeur < {lo}"
        if (y > hi).any():
            return False, f"Valeur > {hi}"
        return True, "OK"

    @classmethod
    def validate_all(cls, cand: nn.Module, ref: nn.Module, x: torch.Tensor) -> Dict[str, Tuple[bool, str]]:
        """Valide I1-I6 et retourne un dict {Iname: (ok, msg)}."""
        x_req = x.clone().detach().requires_grad_(True)
        return {
            "I1": cls.I1_shape_type(cand, ref),
            "I2": cls.I2_numerical(cand(x.clone()), ref(x.clone())),
            "I3a": cls.I3a_grad_input(cand, ref, x_req),
            "I4": cls.I4_determinism(cand, x.clone()),
            "I5": cls.I5_finiteness(cand, x.clone()),
            "I6": cls.I6_bounds(cand, x.clone()),
        }

    @classmethod
    def all_pass(cls, cand: nn.Module, ref: nn.Module, x: torch.Tensor) -> bool:
        return all(ok for ok, _ in cls.validate_all(cand, ref, x).values())
