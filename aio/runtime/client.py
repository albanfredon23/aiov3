"""
AIO v3 – Runtime Client
Features: 2 Streaming · 3 Circuit Breaker · 4 Model Selector
         + Cache (1) · Guardrails (5/6) · Tool Use (8)
"""
from __future__ import annotations
import asyncio, os, time, math, re
from typing import AsyncGenerator, Dict, Any, List, Optional, Callable

try:
    import litellm
    _LITELLM = True
except ImportError:
    _LITELLM = False

from aio.runtime.cache import SemanticCache
from aio.runtime.guardrails import PromptShield, OutputValidator


# ── Feature 3: Circuit Breaker ────────────────────────────────────────────────
class CircuitBreaker:
    """
    States: CLOSED → OPEN (trop d'erreurs) → HALF_OPEN (test) → CLOSED
    Fallback automatique vers providers alternatifs.
    """
    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 3, recovery_sec: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_sec = recovery_sec
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._state: Dict[str, str] = {}

    def state(self, provider: str) -> str:
        s = self._state.get(provider, self.CLOSED)
        if s == self.OPEN:
            if time.time() - self._opened_at.get(provider, 0) > self.recovery_sec:
                self._state[provider] = self.HALF_OPEN
                return self.HALF_OPEN
        return s

    def record_success(self, provider: str):
        self._failures[provider] = 0
        self._state[provider] = self.CLOSED

    def record_failure(self, provider: str):
        self._failures[provider] = self._failures.get(provider, 0) + 1
        if self._failures[provider] >= self.failure_threshold:
            self._state[provider] = self.OPEN
            self._opened_at[provider] = time.time()

    def is_available(self, provider: str) -> bool:
        return self.state(provider) != self.OPEN

    def all_states(self) -> Dict[str, str]:
        return {p: self.state(p) for p in self._state}


# ── Feature 4: Model Selector ─────────────────────────────────────────────────
class ModelSelector:
    """
    Route vers le modèle le plus rentable selon la complexité estimée.
    Simple → cheap model · Complexe → flagship model
    """
    _TECHNICAL_WORDS = {
        "algorithm", "architecture", "optimize", "differential", "tensor",
        "gradient", "eigenvalue", "polynomial", "theorem", "proof", "derive",
        "quantum", "neural", "transformer", "regression", "invariant",
    }

    def __init__(
        self,
        cheap_model: str = "gpt-3.5-turbo",
        flagship_model: str = "gpt-4o",
        complexity_threshold: float = 0.4,
    ):
        self.cheap = cheap_model
        self.flagship = flagship_model
        self.threshold = complexity_threshold

    def _complexity(self, text: str) -> float:
        words = re.findall(r"\w+", text.lower())
        if not words:
            return 0.0
        n = len(words)
        # Longueur (normalisée sur 500 mots)
        length_score = min(n / 500, 1.0) * 0.3
        # Vocabulaire technique
        tech_score = sum(1 for w in words if w in self._TECHNICAL_WORDS) / max(n, 1) * 2.0
        tech_score = min(tech_score, 0.4)
        # Entropie des caractères (diversité lexicale)
        unique_chars = len(set(text.lower()))
        entropy_score = min(unique_chars / 40, 1.0) * 0.3
        return length_score + tech_score + entropy_score

    def select(self, messages: List[Dict[str, str]], forced_model: Optional[str] = None) -> str:
        if forced_model:
            return forced_model
        text = " ".join(m.get("content", "") for m in messages)
        c = self._complexity(text)
        return self.flagship if c >= self.threshold else self.cheap

    def explain(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        text = " ".join(m.get("content", "") for m in messages)
        c = self._complexity(text)
        return {"complexity": round(c, 3), "threshold": self.threshold,
                "selected": self.flagship if c >= self.threshold else self.cheap}


# ── Feature 8: Tool Use ───────────────────────────────────────────────────────
BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a short Python snippet and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    """Exécute un outil builtin (stubs production-safe)."""
    if name == "web_search":
        return f"[STUB] Résultats pour: {args.get('query', '')}"
    if name == "read_file":
        path = args.get("path", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read(2000)
        except Exception as e:
            return f"[ERROR] {e}"
    if name == "run_python":
        import io, contextlib, ast
        code = args.get("code", "")
        buf = io.StringIO()
        try:
            ast.parse(code)  # validation syntaxique
            with contextlib.redirect_stdout(buf):
                exec(code, {"__builtins__": {"print": print, "range": range, "len": len}})
            return buf.getvalue()[:500]
        except Exception as e:
            return f"[ERROR] {e}"
    return f"[UNKNOWN TOOL] {name}"


# ── Client principal ──────────────────────────────────────────────────────────
MODEL_KEY = {
    "gpt-4o": "OPENAI_API_KEY",
    "gpt-4-turbo": "OPENAI_API_KEY",
    "gpt-3.5-turbo": "OPENAI_API_KEY",
    "claude-3-5-sonnet-20241022": "ANTHROPIC_API_KEY",
    "claude-3-haiku-20240307": "ANTHROPIC_API_KEY",
    "gemini/gemini-2.5-flash": "GOOGLE_API_KEY",
    "mistral/mistral-large-latest": "MISTRAL_API_KEY",
    "mistral/mistral-small-latest": "MISTRAL_API_KEY",
    "ollama/llama2": "OLLAMA_BASE_URL",
}

FALLBACK_CHAIN: List[str] = [
    "gpt-4o", "claude-3-5-sonnet-20241022", "gemini/gemini-2.5-flash", "mistral/mistral-large-latest"
]


class AIOv3Client:
    """
    Client complet AIO v3:
    Cache · Streaming · Circuit Breaker · Model Selector · Guardrails · Tool Use
    """

    def __init__(
        self,
        default_model: str = "gpt-4o",
        cache: Optional[SemanticCache] = None,
        shield: Optional[PromptShield] = None,
        validator: Optional[OutputValidator] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        model_selector: Optional[ModelSelector] = None,
        enable_tools: bool = False,
        timeout: int = 30,
    ):
        self.default_model = default_model
        self.cache = cache or SemanticCache()
        self.shield = shield or PromptShield()
        self.validator = validator or OutputValidator()
        self.cb = circuit_breaker or CircuitBreaker()
        self.selector = model_selector or ModelSelector()
        self.enable_tools = enable_tools
        self.timeout = timeout
        self._metrics: Dict[str, float] = {
            "requests": 0, "cache_hits": 0, "tokens_in": 0,
            "tokens_out": 0, "latency_ms": 0.0, "cb_redirects": 0,
        }

    def _api_key(self, model: str) -> str:
        return os.getenv(MODEL_KEY.get(model, "OPENAI_API_KEY"), "")

    def _provider(self, model: str) -> str:
        return model.split("/")[0] if "/" in model else "openai"

    # ── Feature 2: Streaming ──────────────────────────────────────────────────
    async def astream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Génère des tokens un par un (streaming SSE-ready)."""
        model = model or self.default_model
        if not _LITELLM:
            content = f"[STUB STREAM] {messages[-1]['content'][:40]}"
            for word in content.split():
                yield word + " "
                await asyncio.sleep(0.01)
            return
        try:
            response = await asyncio.to_thread(
                litellm.completion,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                api_key=self._api_key(model),
                timeout=self.timeout,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        except Exception as exc:
            yield f"[STREAM ERROR] {exc}"

    # ── Completion principale ─────────────────────────────────────────────────
    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        use_cache: bool = True,
        validate_json: bool = False,
        required_keys: Optional[List[str]] = None,
        schema_hint: str = "",
        **kw,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self._metrics["requests"] += 1

        # 1. Guardrails sur le dernier message user
        user_content = messages[-1].get("content", "") if messages else ""
        shield_result = self.shield.analyze(user_content)
        if not shield_result["safe"]:
            return {
                "content": f"[BLOCKED] {shield_result['reason']}",
                "model": "blocked",
                "tokens_in": 0, "tokens_out": 0,
                "latency_ms": (time.perf_counter() - t0) * 1e3,
                "success": False, "blocked": True,
                "reason": shield_result["reason"],
            }
        # Remplacer contenu par version nettoyée PII
        msgs = list(messages)
        if msgs:
            msgs[-1] = {**msgs[-1], "content": shield_result["cleaned"]}

        # 2. Cache lookup
        selected_model = self.selector.select(msgs, model)
        if use_cache:
            hit = self.cache.lookup(selected_model, user_content)
            if hit:
                self._metrics["cache_hits"] += 1
                hit["latency_ms"] = (time.perf_counter() - t0) * 1e3
                return hit

        # 3. Circuit breaker – sélection provider disponible
        provider = self._provider(selected_model)
        if not self.cb.is_available(provider):
            for fallback in FALLBACK_CHAIN:
                fp = self._provider(fallback)
                if self.cb.is_available(fp) and fallback != selected_model:
                    selected_model = fallback
                    self._metrics["cb_redirects"] += 1
                    break

        # 4. Appel LLM (avec tool use si activé)
        content, tok_in, tok_out = await self._call_llm(
            msgs, selected_model, max_tokens, temperature, **kw
        )

        # 5. Validation JSON + retry si demandé
        if validate_json and required_keys:
            for attempt in range(self.validator.max_retries):
                vr = self.validator.validate_json(content, required_keys)
                if vr["valid"]:
                    break
                if attempt < self.validator.max_retries - 1:
                    retry_msgs = msgs + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": self.validator.build_retry_prompt(
                            user_content, vr["error"], schema_hint
                        )},
                    ]
                    content, tok_in2, tok_out2 = await self._call_llm(
                        retry_msgs, selected_model, max_tokens, temperature
                    )
                    tok_in += tok_in2
                    tok_out += tok_out2

        ms = (time.perf_counter() - t0) * 1e3
        self._metrics["tokens_in"] += tok_in
        self._metrics["tokens_out"] += tok_out
        self._metrics["latency_ms"] += ms

        # Store in cache
        if use_cache and content:
            self.cache.store(selected_model, user_content, content, tok_in, tok_out)

        return {
            "content": content,
            "model": selected_model,
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "latency_ms": round(ms, 2),
            "success": bool(content),
            "cache_hit": False,
            "pii_scrubbed": shield_result["pii_found"],
        }

    async def _call_llm(
        self, msgs, model, max_tokens, temperature, **kw
    ) -> tuple[str, int, int]:
        if not _LITELLM:
            return (f"[STUB] {msgs[-1]['content'][:40]}...", 20, 30)
        provider = self._provider(model)
        tools = BUILTIN_TOOLS if self.enable_tools else None
        try:
            kw_extra = {"tools": tools, "tool_choice": "auto"} if tools else {}
            resp = await asyncio.to_thread(
                litellm.completion,
                model=model,
                messages=msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=self._api_key(model),
                timeout=self.timeout,
                **kw_extra, **kw,
            )
            self.cb.record_success(provider)
            msg = resp.choices[0].message
            # Handle tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_results = []
                for tc in msg.tool_calls:
                    import json as _json
                    args = _json.loads(tc.function.arguments or "{}")
                    result = _dispatch_tool(tc.function.name, args)
                    tool_results.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    })
                # Re-call with tool results
                msgs2 = list(msgs) + [{"role": "assistant", "content": None, "tool_calls": msg.tool_calls}] + tool_results
                resp2 = await asyncio.to_thread(
                    litellm.completion, model=model, messages=msgs2,
                    max_tokens=max_tokens, api_key=self._api_key(model),
                )
                content = resp2.choices[0].message.content or ""
                usage = resp2.usage
            else:
                content = msg.content or ""
                usage = resp.usage
            tok_in = usage.prompt_tokens if usage else 0
            tok_out = usage.completion_tokens if usage else 0
            return content, tok_in, tok_out
        except Exception as exc:
            self.cb.record_failure(provider)
            return (f"[ERROR] {exc}", 0, 0)

    def completion_sync(self, messages, model=None, **kw) -> Dict[str, Any]:
        return asyncio.run(self.acompletion(messages, model, **kw))

    @property
    def metrics(self) -> Dict[str, Any]:
        n = max(int(self._metrics["requests"]), 1)
        return {
            **{k: int(v) if k != "latency_ms" else round(v, 2) for k, v in self._metrics.items()},
            "avg_latency_ms": round(self._metrics["latency_ms"] / n, 2),
            "cache_stats": self.cache.stats,
            "circuit_breaker": self.cb.all_states(),
            "shield_stats": self.shield.stats,
        }
