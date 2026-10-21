"""
Evaluation harness
==================
Runs the golden set through the real agent, scores execution match against
independently computed reference answers, and logs the run to MLflow.

    python eval/run_eval.py                       # offline (no model), MLflow to ./mlflow.db
    python eval/run_eval.py --min-accuracy 0.9    # exit 1 below the bar (CI gate)
    python eval/run_eval.py --compare-baseline    # exit 1 on regression vs eval/baseline.json

Execution match means the numbers and labels in the answer table equal the
reference, within 1e-6 relative tolerance. SQL text is never compared.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import expected  # noqa: E402


def close(a, b) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    except (TypeError, ValueError):
        return False


def _label(v) -> str:
    return v.isoformat() if hasattr(v, "isoformat") else str(v)


def score(item: dict, resp: dict, ref) -> tuple[bool, str]:
    kind = item["kind"]
    if kind == "refuse":
        ok = resp["status"] in ("clarify", "refused")
        return ok, f"status={resp['status']}"
    if resp.get("status") != "ok":
        return False, f"status={resp.get('status')}: {resp.get('answer', '')[:120]}"
    rows, dims, measures = resp["rows"], resp.get("dims") or [], resp.get("measures") or []
    if kind == "scalar":
        got = rows[0][measures[0]] if rows and measures else None
        return close(got, ref), f"got {got} expected {ref}"
    if kind in ("ranked", "set"):
        got = [(_label(r[dims[0]]), r[measures[0]]) for r in rows] if dims else []
        exp = [(str(k), v) for k, v in ref]
        if kind == "set":
            got_d, exp_d = dict(got), dict(exp)
            ok = set(got_d) == set(exp_d) and all(close(got_d[k], exp_d[k]) for k in exp_d)
        else:
            got = got[: len(exp)]
            ok = len(got) == len(exp) and all(g[0] == e[0] and close(g[1], e[1]) for g, e in zip(got, exp))
        return ok, f"got {got[:4]} expected {exp[:4]}"
    if kind == "growth":
        got = [str(r[dims[0]]) for r in rows][: len(ref)]
        return got == ref, f"got {got} expected {ref}"
    if kind == "quadrant":
        got = {str(r[dims[0]]): r.get("Quadrant") for r in rows}
        return got == ref, f"mismatch {[k for k in ref if got.get(k) != ref[k]][:4]}"
    return False, f"unknown kind {kind}"


def run(data_dir: str) -> dict:
    os.environ.setdefault("AUTO_BUILD", "1")
    os.environ.setdefault("GENERATE_DEMO_DATA", "1")
    os.environ["DATA_DIR"] = data_dir
    from langgraph.checkpoint.memory import InMemorySaver

    from sqlagent.config import get_settings
    from sqlagent.engine import Engine
    from sqlagent.graph import Agent

    settings = get_settings()
    agent = Agent(Engine(settings), settings, checkpointer=InMemorySaver())
    with open(os.path.join(ROOT, "eval", "golden.yaml"), encoding="utf-8") as f:
        items = yaml.safe_load(f)["items"]
    refs = expected(data_dir)
    results = []
    for item in items:
        thread = item.get("thread") or item["id"]
        t0 = time.perf_counter()
        resp = agent.ask(item["question"], f"eval-{thread}")
        if item.get("score", True) is False:
            continue
        ok, detail = score(item, resp, refs.get(item["id"]))
        results.append({"id": item["id"], "question": item["question"], "kind": item["kind"], "pass": ok,
                        "detail": detail, "status": resp.get("status"), "confidence": resp.get("confidence"),
                        "llm_calls": resp.get("llm_calls", 0), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)})
    answerable = [r for r in results if r["kind"] != "refuse"]
    refusals = [r for r in results if r["kind"] == "refuse"]
    passed_conf = [r["confidence"] for r in answerable if r["pass"] and r["confidence"] is not None]
    metrics = {
        "execution_accuracy": sum(r["pass"] for r in answerable) / max(len(answerable), 1),
        "refusal_accuracy": sum(r["pass"] for r in refusals) / max(len(refusals), 1),
        "overall_accuracy": sum(r["pass"] for r in results) / max(len(results), 1),
        "mean_latency_ms": sum(r["latency_ms"] for r in results) / max(len(results), 1),
        "p95_latency_ms": sorted(r["latency_ms"] for r in results)[int(0.95 * (len(results) - 1))],
        "llm_calls_per_question": sum(r["llm_calls"] for r in results) / max(len(results), 1),
        "mean_confidence_correct": sum(passed_conf) / max(len(passed_conf), 1),
        "n_questions": len(results),
    }
    params = {"llm_provider": settings.llm_provider, "planner_model": settings.planner_model or "none",
              "writer_model": settings.writer_model or "none", "artifact_version": agent.engine.loaded.version,
              "confidence_floor": settings.confidence_floor, "value_fuzzy_threshold": settings.value_fuzzy_threshold,
              "dense_weight": settings.dense_weight, "lexical_weight": settings.lexical_weight}
    return {"metrics": metrics, "params": params, "results": results}


def log_mlflow(report: dict, out_dir: str) -> str | None:
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed; skipping tracking.")
        return None
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "semantic-sql-agent-eval"))
    with mlflow.start_run(run_name=f"eval-{report['params']['artifact_version']}") as run:
        mlflow.log_params(report["params"])
        mlflow.log_metrics(report["metrics"])
        mlflow.log_artifact(os.path.join(out_dir, "report.json"))
        mlflow.set_tags({"git_sha": os.environ.get("GITHUB_SHA", "local"), "trigger": os.environ.get("EVAL_TRIGGER", "manual")})
        return run.info.run_id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", os.path.join(ROOT, "data")))
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    ap.add_argument("--compare-baseline", action="store_true")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    report = run(args.data_dir)
    out_dir = os.path.join(ROOT, "eval", "reports")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    for r in report["results"]:
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:<5} {r['question'][:60]:<60} {'' if r['pass'] else r['detail'][:160]}")
    print(json.dumps(report["metrics"], indent=2))
    if not args.no_mlflow:
        run_id = log_mlflow(report, out_dir)
        if run_id:
            print(f"MLflow run: {run_id}")

    baseline_path = os.path.join(ROOT, "eval", "baseline.json")
    if args.write_baseline:
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump({k: report["metrics"][k] for k in ("execution_accuracy", "refusal_accuracy")}, f, indent=2)
    code = 0
    if report["metrics"]["execution_accuracy"] < args.min_accuracy:
        print(f"GATE FAILED: execution accuracy {report['metrics']['execution_accuracy']:.3f} < {args.min_accuracy}")
        code = 1
    if args.compare_baseline and os.path.exists(baseline_path):
        with open(baseline_path, encoding="utf-8") as f:
            base = json.load(f)
        for k, v in base.items():
            if report["metrics"][k] + 1e-9 < v:
                print(f"REGRESSION: {k} {report['metrics'][k]:.3f} < baseline {v:.3f}")
                code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
