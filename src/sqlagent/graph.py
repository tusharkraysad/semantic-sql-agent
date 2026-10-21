"""
Agent graph (LangGraph)
=======================

    understand --clarify--> clarify ----------------------------> END
        |
        +--> plan --refused--> finish ---------------------------> END
               |
               +--> compose --cached / refused--> finish --------> END
                      |
                      +--> execute --empty--> diagnose --+
                              |                          |
                              +--ok------> synthesize <--+--> END

understand  resolver + parser: business facts, plan draft, confidence, and
            the route (simple, deep or clarify). Deterministic.
plan        optional model refinement of the draft, validated or discarded.
compose     the composer renders SQL in code; the validator checks it; the
            answer cache is consulted by plan + artifact version.
execute     Polars, read-only, concurrency-capped.
diagnose    explains an empty result instead of returning a confident nothing.
synthesize  deterministic answer, optional model rewrite with citation check,
            audit record, metrics, cache, and memory for follow-up questions.

Conversation memory is LangGraph's checkpointer keyed by thread_id: the last
plan survives between turns, so "now for Glint" re-uses the previous measure,
grouping and period and only swaps the brand.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import TypedDict

import polars as pl
from langgraph.graph import END, START, StateGraph

from . import audit, postprocess
from .cache import AnswerCache
from .composer import CannotCompose
from .config import Settings
from .engine import Engine
from .llm import LLMBudgetExceeded, LLMClient
from .logging_utils import get_logger
from .observability import (
    CACHE,
    CLARIFY,
    CONFIDENCE,
    EMPTY_RESULTS,
    LATENCY,
    LLM_CALLS,
    LLM_TOKENS,
    QUESTIONS,
    UNRESOLVED,
    UNVERIFIED,
)
from .parser import AnalysisPlan
from .planner_llm import refine_plan
from .synthesis import deterministic_answer, llm_answer

log = get_logger("graph")


class AgentState(TypedDict, total=False):
    question: str
    run_id: str
    previous: dict | None
    mode: str
    plan: dict | None
    sql: str | None
    status: str
    response: dict | None


class Agent:
    def __init__(self, engine: Engine, settings: Settings, checkpointer=None):
        self.engine = engine
        self.s = settings
        self.cache = AnswerCache(settings.cache_size)
        self._scratch: dict[str, dict] = {}
        if checkpointer is None:
            checkpointer = self._sqlite_checkpointer()
        self.graph = self._build().compile(checkpointer=checkpointer)

    def _sqlite_checkpointer(self):
        import os

        os.makedirs(self.s.state_dir, exist_ok=True)
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return SqliteSaver(sqlite3.connect(self.s.checkpoint_path, check_same_thread=False))
        except Exception as e:  # pragma: no cover - fall back to memory if sqlite saver is unavailable
            log.warning("SQLite checkpointer unavailable (%s); using in-memory conversation state", e)
            from langgraph.checkpoint.memory import InMemorySaver

            return InMemorySaver()

    # ------------------------------------------------------------------------
    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("understand", self.understand)
        g.add_node("clarify", self.clarify)
        g.add_node("plan", self.plan)
        g.add_node("compose", self.compose)
        g.add_node("execute", self.execute)
        g.add_node("diagnose", self.diagnose)
        g.add_node("synthesize", self.synthesize)
        g.add_node("finish", self.finish)
        g.add_edge(START, "understand")
        g.add_conditional_edges("understand", lambda s: "clarify" if s["mode"] == "clarify" else "plan",
                                {"clarify": "clarify", "plan": "plan"})
        g.add_conditional_edges("plan", lambda s: "finish" if s.get("status") == "refused" else "compose",
                                {"finish": "finish", "compose": "compose"})
        g.add_conditional_edges("compose", lambda s: "execute" if s.get("status") == "composed" else "finish",
                                {"execute": "execute", "finish": "finish"})
        g.add_conditional_edges("execute", lambda s: {"ok": "synthesize", "empty": "diagnose"}.get(s["status"], "finish"),
                                {"synthesize": "synthesize", "diagnose": "diagnose", "finish": "finish"})
        g.add_edge("diagnose", "synthesize")
        g.add_edge("clarify", END)
        g.add_edge("synthesize", END)
        g.add_edge("finish", END)
        return g

    # -- public ---------------------------------------------------------------
    def ask(self, question: str, thread_id: str | None = None) -> dict:
        final = None
        for _, update in self.stream(question, thread_id):
            if update is not None:
                final = update
        return final

    def stream(self, question: str, thread_id: str | None = None):
        """Yields (stage, None) per finished node, then ("done", response)."""
        self.engine.maybe_reload()
        run_id = uuid.uuid4().hex[:12]
        thread_id = thread_id or run_id
        llm = LLMClient(self.s)
        self._scratch[run_id] = {"t0": time.perf_counter(), "llm": llm, "notes": [], "thread_id": thread_id,
                                 "L": self.engine.loaded}
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        inputs = {"question": question, "run_id": run_id, "mode": "", "plan": None, "sql": None, "status": "",
                  "response": None}
        response = None
        try:
            for chunk in self.graph.stream(inputs, config=config, stream_mode="updates"):
                for node, update in chunk.items():
                    if update and update.get("response"):
                        response = update["response"]
                    yield node, None
        except Exception as e:  # never a 500 for a question; always a readable answer and an audit line
            log.exception("question failed")
            response = self._error_response(run_id, question, f"{type(e).__name__}: {e}")
        finally:
            self._scratch.pop(run_id, None)
        yield "done", response

    # -- nodes ------------------------------------------------------------------
    def understand(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        L = sc["L"]
        ctx = L.resolver.resolve(state["question"])
        parsed = L.parser.parse(ctx, state.get("previous"))
        sc.update(ctx=ctx, parsed=parsed)
        mode = "clarify" if parsed.clarify else ("deep" if parsed.needs_deep else "simple")
        return {"mode": mode, "plan": parsed.plan.model_dump() if parsed.plan else None}

    def clarify(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        CLARIFY.inc()
        resp = self._response(state, status="clarify", answer=sc["parsed"].clarify)
        return {"status": "clarify", "response": resp}

    def plan(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        L, llm = sc["L"], sc["llm"]
        plan = AnalysisPlan.model_validate(state["plan"])
        if llm.enabled and not sc["parsed"].followup:
            try:
                refs = L.retriever.retrieve(state["question"], sc["ctx"])
                block = _semantic_block(sc["ctx"])
                plan, refusal, notes = refine_plan(llm, L.model, state["question"], plan, block, L.retriever.schema_text(refs))
                sc["notes"] += notes
                if refusal:
                    return {"status": "refused", "response": self._response(state, status="refused", answer=refusal)}
            except LLMBudgetExceeded as e:
                sc["notes"].append(f"Model planning skipped: {e}.")
        return {"plan": plan.model_dump()}

    def compose(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        L = sc["L"]
        plan = AnalysisPlan.model_validate(state["plan"])
        key = AnswerCache.key(L.version, plan.model_dump())
        sc["cache_key"] = key
        hit = self.cache.get(key)
        if hit is not None:
            CACHE.labels(result="hit").inc()
            resp = dict(hit, cached=True, question=state["question"], thread_id=sc["thread_id"], run_id=state["run_id"],
                        latency_ms=round((time.perf_counter() - sc["t0"]) * 1000, 1))
            return {"status": "cached", "response": resp, "previous": {"plan": plan.model_dump()}}
        CACHE.labels(result="miss").inc()
        try:
            composed = L.composer.compose(plan)
        except CannotCompose as e:
            return {"status": "refused", "response": self._response(state, status="refused", answer=str(e))}
        verdict = L.validator.validate(composed.sql)
        sc.update(composed=composed, verdict=verdict)
        if not verdict.ok:
            return {"status": "error", "sql": composed.sql,
                    "response": self._response(state, status="error", sql=composed.sql,
                                               answer="The query failed validation: " + "; ".join(verdict.errors))}
        return {"status": "composed", "sql": composed.sql}

    def execute(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        try:
            result = sc["L"].executor.run(state["sql"])
        except Exception as e:
            return {"status": "error", "response": self._response(state, status="error", sql=state["sql"],
                                                                  answer=f"The query could not run: {e}")}
        sc["result"] = result
        return {"status": "ok" if result.frame.height else "empty"}

    def diagnose(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        EMPTY_RESULTS.inc()
        L, plan = sc["L"], AnalysisPlan.model_validate(state["plan"])
        reasons = []
        for f in plan.filters:
            t, c = f.column.split(".", 1)
            frame = L.registry.frames.get(t)
            if frame is not None and c in frame.columns:
                n = frame.filter(pl.col(c).cast(pl.Utf8).is_in(f.values)).height
                if n == 0:
                    reasons.append(f"no row has {c} in {f.values}")
        sc["notes"].append("No rows matched. " + ("Cause: " + "; ".join(reasons) + "." if reasons else
                           "Each filter matches rows on its own, so it is the combination that does not exist in the data."))
        return {"status": "empty"}

    def synthesize(self, state: AgentState) -> dict:
        sc = self._scratch[state["run_id"]]
        L, composed, frame = sc["L"], sc["composed"], sc["result"].frame
        plan = AnalysisPlan.model_validate(state["plan"])
        notes = list(composed.notes)
        if composed.post == "growth" and frame.height:
            rank = composed.derived[-1]
            frame, n2 = postprocess.growth(frame, composed.measures[1], rank, plan.top_n, self.s.small_base_ratio)
            notes += n2
        elif composed.post == "quadrant" and frame.height:
            frame, n2 = postprocess.quadrant(frame, composed.measures[0], composed.measures[1])
            notes += n2
        primary = composed.tables[0] if composed.tables else ""
        period = composed.windows.get(_primary(plan, L.model), "") if composed.windows else ""
        if composed.compare_windows:
            period = f"{period} vs {composed.compare_windows.get(_primary(plan, L.model), '')}"
        draft = deterministic_answer(frame, plan.archetype, composed.dims, composed.measures, composed.measure_info, period)
        answer, writer, unverified = draft, "deterministic", []
        if sc["llm"].enabled and frame.height:
            try:
                text, unverified = llm_answer(sc["llm"], state["question"], frame, period, notes + sc["ctx"].warnings, draft)
                if text:
                    answer, writer = text, "model"
                    if unverified:
                        UNVERIFIED.inc(len(unverified))
            except LLMBudgetExceeded:
                pass
        del primary
        resp = self._response(state, status="ok" if frame.height else "empty", answer=answer, frame=frame,
                              composed=composed, notes=notes, writer=writer, unverified=unverified, period=period)
        if frame.height:
            self.cache.put(sc["cache_key"], resp)
        return {"status": resp["status"], "response": resp, "previous": {"plan": plan.model_dump()}}

    def finish(self, state: AgentState) -> dict:
        resp = state.get("response") or self._response(state, status=state.get("status") or "error", answer="No answer.")
        if state.get("status") == "cached":
            sc = self._scratch[state["run_id"]]
            self._record(state, resp, sc)
        return {"response": resp}

    # -- response, audit, metrics --------------------------------------------------
    def _response(self, state: AgentState, status: str, answer: str = "", frame: pl.DataFrame | None = None,
                  composed=None, notes: list | None = None, writer: str = "deterministic", unverified=None,
                  sql: str | None = None, period: str = "") -> dict:
        sc = self._scratch[state["run_id"]]
        ctx, parsed, llm = sc.get("ctx"), sc.get("parsed"), sc["llm"]
        confidence = parsed.confidence if parsed else 0.0
        floor = self.s.confidence_floor
        if status == "ok" and confidence < floor:
            answer = f"Low confidence ({confidence:.2f}): check the assumptions below. " + answer
        rows = frame.to_dicts() if frame is not None else []
        resp = {
            "question": state["question"], "status": status, "answer": answer, "mode": state.get("mode", ""),
            "confidence": confidence, "penalties": parsed.penalties if parsed else {},
            "archetype": (state.get("plan") or {}).get("archetype"),
            "columns": frame.columns if frame is not None else [], "rows": rows,
            "dims": composed.dims if composed else [], "measures": composed.measures if composed else [],
            "chart": composed.chart if composed else {}, "sql": (composed.sql if composed else sql) or state.get("sql"),
            "sql_source": "semantic layer (composed in code)" if composed else None,
            "plan": state.get("plan"), "period": period,
            "assumptions": (ctx.assumptions if ctx else []) + (parsed.notes if parsed else []),
            "warnings": ctx.warnings if ctx else [],
            "notes": list(dict.fromkeys((notes or []) + sc["notes"] + (sc["verdict"].warnings if sc.get("verdict") else []))),
            "unresolved_terms": ctx.unresolved_terms if ctx else [],
            "unverified_figures": unverified or [], "writer": writer,
            "followup": bool(parsed and parsed.followup),
            "llm_calls": llm.calls, "tokens": {"prompt": llm.prompt_tokens, "completion": llm.completion_tokens},
            "latency_ms": round((time.perf_counter() - sc["t0"]) * 1000, 1),
            "artifact_version": sc["L"].version, "cached": False, "thread_id": sc["thread_id"], "run_id": state["run_id"],
        }
        self._record(state, resp, sc)
        return resp

    def _record(self, state, resp: dict, sc: dict) -> None:
        ctx = sc.get("ctx")
        mode = state.get("mode") or "unknown"
        QUESTIONS.labels(mode=mode, status=resp["status"]).inc()
        LATENCY.labels(mode=mode).observe(resp["latency_ms"] / 1000)
        if resp["status"] == "ok":
            CONFIDENCE.observe(resp["confidence"])
        if ctx is not None and ctx.unresolved_terms:
            UNRESOLVED.inc(len(ctx.unresolved_terms))
        llm = sc["llm"]
        if llm.calls:
            LLM_CALLS.labels(role="all").inc(llm.calls)
            LLM_TOKENS.labels(kind="prompt").inc(llm.prompt_tokens)
            LLM_TOKENS.labels(kind="completion").inc(llm.completion_tokens)
        audit.record(self.s.decisions_path, {
            "run_id": state["run_id"], "thread_id": sc["thread_id"], "question": state["question"],
            "status": resp["status"], "mode": mode, "confidence": resp["confidence"], "penalties": resp["penalties"],
            "resolved": ctx.summary() if ctx is not None else None, "plan": state.get("plan"), "sql": resp.get("sql"),
            "rows": len(resp.get("rows") or []), "notes": resp.get("notes"), "writer": resp.get("writer"),
            "unverified": resp.get("unverified_figures"), "llm_calls": llm.calls, "latency_ms": resp["latency_ms"],
            "artifact_version": resp["artifact_version"], "cached": resp.get("cached", False),
        })

    def _error_response(self, run_id: str, question: str, err: str) -> dict:
        sc = self._scratch.get(run_id, {})
        resp = {"question": question, "status": "error", "answer": f"Something went wrong answering that: {err}",
                "rows": [], "columns": [], "confidence": 0.0, "notes": [], "assumptions": [], "warnings": [],
                "run_id": run_id, "thread_id": sc.get("thread_id"), "artifact_version": self.engine.loaded.version,
                "latency_ms": round((time.perf_counter() - sc.get("t0", time.perf_counter())) * 1000, 1)}
        audit.record(self.s.decisions_path, {"run_id": run_id, "question": question, "status": "error", "error": err})
        QUESTIONS.labels(mode="unknown", status="error").inc()
        return resp


def _primary(plan: AnalysisPlan, model) -> str:
    m = plan.measures[0]
    if m.kind == "column":
        return m.ref.split(".", 1)[0]
    return model.metrics[m.ref].numerator.split(".", 1)[0]


def _semantic_block(ctx) -> str:
    import json

    return json.dumps(ctx.summary(), indent=1)
