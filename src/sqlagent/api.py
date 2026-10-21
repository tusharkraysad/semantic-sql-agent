"""
FastAPI service
===============
    GET  /health          liveness: process is up
    GET  /ready           readiness: artifacts loaded (used by Kubernetes probes)
    GET  /version         artifact version, data version, model summary
    POST /ask             {"question": "...", "thread_id": "optional"} -> answer
    POST /ask/stream      same, as server-sent events: one event per stage, then the answer
    GET  /schema          tables, columns, roles
    GET  /semantic        the full semantic model (as the service sees it)
    GET  /decisions       recent audit records
    POST /admin/reload    reload if a newer artifact version was promoted
    GET  /metrics         Prometheus metrics

A question never produces a 500: failures come back as a readable answer with
status "error", and every question writes an audit record.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import __version__, audit
from .config import get_settings
from .engine import Engine
from .graph import Agent
from .logging_utils import get_logger

log = get_logger("api")
STATE: dict = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    thread_id: str | None = Field(default=None, max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = Engine(settings)
    STATE.update(settings=settings, engine=engine, agent=Agent(engine, settings))
    log.info("API ready on artifacts %s (LLM %s)", engine.loaded.version, "on" if settings.llm_enabled else "off")
    yield
    STATE.clear()


app = FastAPI(title="Semantic SQL Agent", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    engine = STATE.get("engine")
    if engine is None or engine.loaded is None:
        raise HTTPException(503, "not ready")
    return {"status": "ready", "artifact_version": engine.loaded.version}


@app.get("/version")
def version() -> dict:
    engine: Engine = STATE["engine"]
    engine.maybe_reload()
    m = engine.loaded.manifest
    return {"code_version": __version__, "artifact_version": engine.loaded.version, "data_version": m["data_version"],
            "built_at": m["built_at"], "summary": m["summary"], "llm_enabled": STATE["settings"].llm_enabled}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return STATE["agent"].ask(req.question, req.thread_id)


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    labels = {"understand": "Understanding the question", "plan": "Planning the analysis", "compose": "Building the query",
              "execute": "Running the query", "diagnose": "Checking an empty result", "synthesize": "Writing the answer",
              "clarify": "Asking for clarification", "finish": "Finishing"}

    def events():
        for stage, response in STATE["agent"].stream(req.question, req.thread_id):
            if stage == "done":
                yield f"event: answer\ndata: {json.dumps(response, default=str)}\n\n"
            else:
                yield f"event: stage\ndata: {json.dumps({'stage': stage, 'label': labels.get(stage, stage)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/schema")
def schema() -> dict:
    L = STATE["engine"].loaded
    return {t: [{"column": c.column, "role": c.role, "label": c.label, "entity": c.entity,
                 "repeated_base": c.is_repeated_base} for c in L.model.columns_of(t)] for t in L.model.tables}


@app.get("/semantic")
def semantic() -> dict:
    return STATE["engine"].loaded.model.to_dict()


@app.get("/decisions")
def decisions(n: int = 50) -> list:
    return audit.tail(STATE["settings"].decisions_path, min(n, 500))


@app.post("/admin/reload")
def reload() -> dict:
    engine: Engine = STATE["engine"]
    changed = engine.maybe_reload()
    return {"reloaded": changed, "artifact_version": engine.loaded.version}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
