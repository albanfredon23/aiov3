"""
AIO v3 – FastAPI complet
Toutes les 12 features exposées via REST + SSE + WebSocket + Dashboard
"""
from __future__ import annotations
import asyncio, json, time
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Header, Query, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from aio.runtime.client import AIOv3Client, CircuitBreaker, ModelSelector
from aio.runtime.cache import SemanticCache
from aio.runtime.guardrails import PromptShield, OutputValidator
from aio.runtime.memory import AgentMemory
from aio.tenancy.manager import TenantManager
from aio.telemetry.otel import METRICS
from aio.compiler.rollback_manager import RollbackManager
from aio.orchestration.agent_modulator import AgentModulatorGate

app = FastAPI(title="AIO v3 – Gödel Machine", version="3.0.0",
              description="12 features: Cache · Stream · CircuitBreaker · ModelSelect · Guardrails · OutputValid · Memory · Tools · MultiTenant · Dashboard · OTel · Rollback")

# ── Singletons ────────────────────────────────────────────────────────────────
telemetry_bus: asyncio.Queue = asyncio.Queue(maxsize=500)
cache         = SemanticCache()
shield        = PromptShield()
validator     = OutputValidator()
cb            = CircuitBreaker()
selector      = ModelSelector()
memory        = AgentMemory()
tenants       = TenantManager()
rollback_mgr  = RollbackManager(on_rollback=lambda mid, r: print(f"[ROLLBACK] {mid}: {r}"))

import torch
_orch_available = False
try:
    _orch = AgentModulatorGate(emb_dim=256, telemetry_queue=telemetry_bus)
    _orch_available = True
except Exception:
    _orch = None

_client = AIOv3Client(
    cache=cache, shield=shield, validator=validator,
    circuit_breaker=cb, model_selector=selector,
)


# ── Auth helper ───────────────────────────────────────────────────────────────
async def _auth(x_api_key: Optional[str]) -> Optional[str]:
    if not x_api_key:
        return None
    return tenants.authenticate(x_api_key)


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    max_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    system: Optional[str] = None
    session_id: Optional[str] = "default"
    agent_id: Optional[str] = None
    use_cache: bool = True
    validate_json: bool = False
    required_keys: Optional[list] = None

class FeedbackRequest(BaseModel):
    agent_id: str
    success_score: float = Field(..., ge=0.0, le=1.0)

class TenantCreate(BaseModel):
    name: str
    plan: str = "starter"

class FactSet(BaseModel):
    key: str
    value: object


# ── Routes principales ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    METRICS.gauge("aio_up", 1.0)
    return {
        "status": "ok",
        "version": "3.0.0",
        "features": [
            "semantic_cache", "streaming", "circuit_breaker", "model_selector",
            "guardrails", "output_validation", "agent_memory", "tool_use",
            "multi_tenancy", "dashboard", "opentelemetry", "auto_rollback"
        ],
        "circuit_breaker": cb.all_states(),
        "cache_hit_rate": cache.stats["hit_rate"],
    }


@app.post("/v1/query")
async def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    # Auth + quota
    tenant_id = await _auth(x_api_key)
    if x_api_key and not tenant_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if tenant_id:
        quota = tenants.check_quota(tenant_id)
        if not quota["allowed"]:
            raise HTTPException(status_code=429, detail=quota["reason"])

    # Historique mémoire agent
    msgs = []
    if req.system:
        msgs.append({"role": "system", "content": req.system})
    if req.agent_id:
        history = memory.get_history(req.agent_id, req.session_id or "default")
        msgs.extend(history)
    msgs.append({"role": "user", "content": req.query})

    # Appel client
    t0 = time.perf_counter()
    resp = await _client.acompletion(
        msgs, model=req.model, max_tokens=req.max_tokens,
        temperature=req.temperature, use_cache=req.use_cache,
        validate_json=req.validate_json, required_keys=req.required_keys,
    )
    ms = (time.perf_counter() - t0) * 1e3

    # Stocker en mémoire
    if req.agent_id and resp.get("success"):
        memory.add_message(req.agent_id, req.session_id or "default", "user", req.query)
        memory.add_message(req.agent_id, req.session_id or "default", "assistant", resp["content"])

    # Métriques
    METRICS.counter("aio_requests_total", labels={"model": resp.get("model", "unknown")})
    METRICS.histogram("aio_latency_ms", ms, labels={"model": resp.get("model", "unknown")})
    METRICS.counter("aio_tokens_total", resp.get("tokens_in", 0) + resp.get("tokens_out", 0))

    # Usage tenant
    if tenant_id:
        tenants.record_usage(tenant_id, 1, resp.get("tokens_in", 0), resp.get("tokens_out", 0))

    return {**resp, "tenant_id": tenant_id, "model_selection": selector.explain(msgs)}


# ── Feature 2: Streaming ──────────────────────────────────────────────────────
@app.post("/v1/stream")
async def stream_query(req: QueryRequest):
    msgs = [{"role": "user", "content": req.query}]
    if req.system:
        msgs.insert(0, {"role": "system", "content": req.system})

    async def gen():
        async for token in _client.astream(msgs, model=req.model, max_tokens=req.max_tokens):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


# ── Feature 7: Mémoire agent ──────────────────────────────────────────────────
@app.get("/agents/{agent_id}/memory")
async def get_memory(agent_id: str, session_id: str = "default", limit: int = 20):
    return {
        "history": memory.get_history(agent_id, session_id, limit),
        "facts": memory.get_all_facts(agent_id),
        "stats": memory.stats(agent_id),
    }

@app.post("/agents/{agent_id}/facts")
async def set_fact(agent_id: str, req: FactSet):
    memory.set_fact(agent_id, req.key, req.value)
    return {"agent_id": agent_id, "key": req.key, "value": req.value}


# ── Orchestration Gödel ───────────────────────────────────────────────────────
@app.post("/agents/register")
async def register_agent(agent_id: str = Query(...)):
    if _orch_available and _orch:
        _orch.register_client_agent(agent_id)
    return {"status": "registered", "agent_id": agent_id}

@app.post("/v1/feedback")
async def feedback(req: FeedbackRequest):
    if _orch_available and _orch and req.agent_id in _orch.agent_ids:
        _orch.feedback_step(req.agent_id, req.success_score)
    return {"status": "updated", **req.model_dump()}


# ── Feature 9: Multi-tenancy ──────────────────────────────────────────────────
@app.post("/admin/tenants")
async def create_tenant(req: TenantCreate):
    result = tenants.create_tenant(req.name, req.plan)
    return result

@app.get("/admin/tenants")
async def list_tenants():
    return {"tenants": tenants.list_tenants()}

@app.get("/admin/tenants/{tenant_id}/usage")
async def tenant_usage(tenant_id: str):
    return tenants.get_usage(tenant_id)

@app.delete("/admin/tenants/{tenant_id}")
async def deactivate(tenant_id: str):
    tenants.deactivate_tenant(tenant_id)
    return {"status": "deactivated", "tenant_id": tenant_id}


# ── Feature 11: Métriques Prometheus ─────────────────────────────────────────
@app.get("/metrics")
async def prometheus_metrics():
    return StreamingResponse(
        iter([METRICS.export_prometheus()]),
        media_type="text/plain; version=0.0.4"
    )

@app.get("/metrics/json")
async def metrics_json():
    return METRICS.snapshot()


# ── Feature 10: Dashboard temps réel ─────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("/home/claude/aio-v3/dashboard/index.html", "r") as f:
        return f.read()


# ── Télémétrie SSE ────────────────────────────────────────────────────────────
@app.get("/events/telemetry")
async def sse_telemetry():
    async def gen():
        while True:
            try:
                data = telemetry_bus.get_nowait()
                yield f"data: {json.dumps(data)}\n\n"
            except asyncio.QueueEmpty:
                # Heartbeat + métriques live
                snap = METRICS.snapshot()
                snap["ts"] = time.time()
                yield f"data: {json.dumps(snap)}\n\n"
                await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            snap = METRICS.snapshot()
            snap["ts"] = time.time()
            snap["cache"] = cache.stats
            snap["circuit_breaker"] = cb.all_states()
            snap["client"] = _client.metrics
            await ws.send_json(snap)
            await asyncio.sleep(1.0)
    except Exception:
        pass
    finally:
        await ws.close()


# ── Feature 12: Rollback ──────────────────────────────────────────────────────
@app.get("/mutations")
async def list_mutations():
    return {"mutations": rollback_mgr.all_statuses(), "history": rollback_mgr.history}

@app.post("/mutations/{mutation_id}/promote")
async def promote_mutation(mutation_id: str, name: str = "unnamed"):
    rollback_mgr.promote(mutation_id, name)
    return {"status": "promoted", "mutation_id": mutation_id}

@app.post("/mutations/{mutation_id}/record")
async def record_mutation(mutation_id: str, latency_ms: float, success: bool = True):
    reason = rollback_mgr.record(mutation_id, latency_ms, success)
    return {"rollback": reason is not None, "reason": reason}

@app.get("/stats")
async def stats():
    return {
        "client": _client.metrics,
        "cache": cache.stats,
        "shield": shield.stats,
        "validator": validator.stats,
        "rollback": rollback_mgr.all_statuses(),
        "tenants": len(tenants.list_tenants()),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
