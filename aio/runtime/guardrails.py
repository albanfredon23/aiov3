"""
AIO v3 – Feature 5: Prompt Shield (guardrails)
- Détection injection de prompt
- Scrubbing PII (email, téléphone, IP, carte bancaire)
- Validation de sortie JSON avec retry
- Modération basique
"""
from __future__ import annotations
import re, json
from typing import Any, Dict, List, Optional, Tuple


# ── Patterns PII ──────────────────────────────────────────────────────────────
_PII_PATTERNS: List[Tuple[str, str, str]] = [
    ("email",    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    ("phone",    r"(\+?\d[\d\s\-().]{7,}\d)", "[PHONE]"),
    ("ip",       r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
    ("cc",       r"\b(?:\d[ \-]?){13,16}\b", "[CARD]"),
    ("ssn",      r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b", "[SSN]"),
    ("api_key",  r"(sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9\-]{20,})", "[API_KEY]"),
]

# ── Patterns injection de prompt ──────────────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil|unrestricted)",
    r"(system\s*prompt|act\s+as\s+if|pretend\s+you)",
    r"disregard\s+(your\s+)?(rules?|guidelines?|restrictions?)",
    r"jailbreak|bypass.*filter|override.*safety",
]

# ── Patterns modération ───────────────────────────────────────────────────────
_MODERATION_PATTERNS = [
    r"\b(bomb|explosive|bioweapon|chemical\s+weapon)\b.*\b(make|build|create|synthesize)\b",
    r"\b(hack|exploit|malware|ransomware)\b.*\b(write|code|create|generate)\b",
]


class PromptShield:
    """Analyse et nettoie les prompts avant envoi au LLM."""

    def __init__(self, block_injection: bool = True, scrub_pii: bool = True, block_harmful: bool = True):
        self.block_injection = block_injection
        self.scrub_pii = scrub_pii
        self.block_harmful = block_harmful
        self._stats = {"injections_blocked": 0, "pii_scrubbed": 0, "harmful_blocked": 0, "clean": 0}

    def analyze(self, text: str) -> Dict[str, Any]:
        """Retourne {safe: bool, reason: str, cleaned: str, pii_found: list}."""
        # 1. Injection detection
        if self.block_injection:
            for pat in _INJECTION_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    self._stats["injections_blocked"] += 1
                    return {"safe": False, "reason": "prompt_injection", "cleaned": text, "pii_found": []}

        # 2. Harmful content
        if self.block_harmful:
            for pat in _MODERATION_PATTERNS:
                if re.search(pat, text, re.IGNORECASE):
                    self._stats["harmful_blocked"] += 1
                    return {"safe": False, "reason": "harmful_content", "cleaned": text, "pii_found": []}

        # 3. PII scrubbing
        cleaned = text
        pii_found = []
        if self.scrub_pii:
            for pii_type, pat, replacement in _PII_PATTERNS:
                matches = re.findall(pat, cleaned)
                if matches:
                    pii_found.append({"type": pii_type, "count": len(matches)})
                    cleaned = re.sub(pat, replacement, cleaned)
            if pii_found:
                self._stats["pii_scrubbed"] += len(pii_found)

        self._stats["clean"] += 1
        return {"safe": True, "reason": "ok", "cleaned": cleaned, "pii_found": pii_found}

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)


class OutputValidator:
    """
    Feature 6: Valide que la sortie LLM respecte un schéma JSON.
    Retry automatique jusqu'à max_retries si invalide.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._stats = {"validated": 0, "failed": 0, "retries": 0}

    def validate_json(self, text: str, required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Tente d'extraire et valider du JSON dans text."""
        # Cherche un bloc JSON dans la réponse
        json_match = re.search(r"\{[\s\S]*\}", text)
        if not json_match:
            return {"valid": False, "data": None, "error": "no_json_found"}
        try:
            data = json.loads(json_match.group())
            if required_keys:
                missing = [k for k in required_keys if k not in data]
                if missing:
                    return {"valid": False, "data": data, "error": f"missing_keys: {missing}"}
            self._stats["validated"] += 1
            return {"valid": True, "data": data, "error": None}
        except json.JSONDecodeError as e:
            self._stats["failed"] += 1
            return {"valid": False, "data": None, "error": str(e)}

    def build_retry_prompt(self, original_prompt: str, error: str, schema_hint: str = "") -> str:
        """Génère un prompt de retry avec instruction de correction."""
        self._stats["retries"] += 1
        hint = f" Schema attendu : {schema_hint}" if schema_hint else ""
        return (
            f"{original_prompt}\n\n"
            f"[CORRECTION REQUISE] Ta réponse précédente était invalide ({error}).{hint} "
            f"Réponds UNIQUEMENT avec du JSON valide, sans texte avant ou après."
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)
