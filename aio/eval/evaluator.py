"""
AIO v2 – MultiTaskEvaluator
Évaluation normalisée : exact-match, tolérance arithmétique, invariants formels.
Remplace les heuristiques fragiles (in text / _extract_number).
"""
from __future__ import annotations
import re, math
from typing import Any, Dict, List, Optional, Tuple


def _normalize(text: str) -> str:
    """Normalisation canonique : minuscules, espaces, ponctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s.\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_number(text: str) -> Optional[float]:
    """Extrait le dernier nombre explicitement délimité dans text."""
    # Cherche les nombres décimaux avec signe optionnel
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)?(?:e[-+]?\d+)?", text.replace(",", "."))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


class MultiTaskEvaluator:
    """
    Évaluateur multi-tâches robuste :
      - exact_match : correspondance texte normalisée
      - numeric_match : tolérance arithmétique relative/absolue
      - contains_match : présence du token cible
    """

    def __init__(self, numeric_rtol: float = 0.01, numeric_atol: float = 0.5):
        self.rtol = numeric_rtol
        self.atol = numeric_atol

    # ── Évaluation unitaire ───────────────────────────────────────────────────
    def evaluate(
        self,
        prediction: str,
        reference: str,
        task_type: str = "exact_match",
    ) -> Dict[str, Any]:
        """
        Args:
            prediction: texte produit par le système
            reference:  réponse attendue
            task_type:  "exact_match" | "numeric" | "contains"
        Returns:
            {"correct": bool, "score": float, "detail": str}
        """
        if task_type == "numeric":
            return self._numeric(prediction, reference)
        elif task_type == "contains":
            return self._contains(prediction, reference)
        else:
            return self._exact(prediction, reference)

    def _exact(self, pred: str, ref: str) -> Dict[str, Any]:
        p, r = _normalize(pred), _normalize(ref)
        ok = p == r
        return {"correct": ok, "score": 1.0 if ok else 0.0,
                "detail": f"pred='{p[:40]}' ref='{r[:40]}'"}

    def _numeric(self, pred: str, ref: str) -> Dict[str, Any]:
        p_num = _extract_number(pred)
        r_num = _extract_number(ref)
        if p_num is None or r_num is None:
            # Fallback exact-match si pas de nombre
            return self._exact(pred, ref)
        abs_err = abs(p_num - r_num)
        rel_err = abs_err / max(abs(r_num), 1e-9)
        ok = abs_err <= self.atol or rel_err <= self.rtol
        return {"correct": ok, "score": 1.0 if ok else 0.0,
                "detail": f"pred={p_num} ref={r_num} abs_err={abs_err:.4f} rel_err={rel_err:.4f}"}

    def _contains(self, pred: str, ref: str) -> Dict[str, Any]:
        ok = _normalize(ref) in _normalize(pred)
        return {"correct": ok, "score": 1.0 if ok else 0.0,
                "detail": f"ref='{ref[:30]}' in pred: {ok}"}

    # ── Évaluation batch ──────────────────────────────────────────────────────
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = "exact_match",
    ) -> Dict[str, Any]:
        assert len(predictions) == len(references)
        results = [
            self.evaluate(p, r, task_type)
            for p, r in zip(predictions, references)
        ]
        n_correct = sum(r["correct"] for r in results)
        accuracy = n_correct / max(len(results), 1)
        return {
            "accuracy": round(accuracy, 4),
            "n_correct": n_correct,
            "n_total": len(results),
            "task_type": task_type,
            "per_sample": results,
        }
