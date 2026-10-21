"""
Model-assisted planning
=======================
Optional. The deterministic parser always produces a draft plan; when a model
is configured it may correct that draft (for example choose a better grouping)
by returning JSON that must pass the same Pydantic contract plus a check that
every column, metric and entity it names exists. An invalid plan is sent back
once with the exact error; after that the draft stands. A model can therefore
make a plan better, but never make it unsafe.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from .parser import AnalysisPlan
from .semantic.model import SemanticModel

PLANNER_SYSTEM = """You correct analysis plans for a governed analytics system. You never write SQL.
Return ONLY a JSON object matching the schema. Use only the columns, metrics and entities listed.
If the draft is already right, return it unchanged. If the question cannot be answered from the
listed schema, return {"cannot_answer": "<one sentence why>"}.
Archetypes: total, ranking, breakdown, trend, pareto, growth, quadrant, compare.
group_by items are 'entity:<name>', '<table>.<column>' or 'time'."""


def _check_refs(plan: AnalysisPlan, model: SemanticModel) -> list[str]:
    errors = []
    for m in plan.measures:
        if m.kind == "column" and (m.ref not in model.columns or not model.columns[m.ref].is_measure):
            errors.append(f"measures: {m.ref} is not a measure column")
        if m.kind == "metric" and (m.ref not in model.metrics or not model.metrics[m.ref].usable):
            errors.append(f"measures: {m.ref} is not an approved metric")
    for g in plan.group_by:
        if g == "time":
            continue
        if g.startswith("entity:") and g.split(":", 1)[1] not in model.entities:
            errors.append(f"group_by: unknown entity {g}")
        elif not g.startswith("entity:") and g not in model.columns:
            errors.append(f"group_by: unknown column {g}")
    for f in plan.filters:
        if f.column not in model.columns:
            errors.append(f"filters: unknown column {f.column}")
    return errors


def refine_plan(llm, model: SemanticModel, question: str, draft: AnalysisPlan, semantic_block: str,
                schema_text: str) -> tuple[AnalysisPlan, str | None, list[str]]:
    """Returns (plan, refusal or None, notes)."""
    notes: list[str] = []
    metrics = "\n".join(f"- {m.name}: {m.display} ({m.status})" for m in model.usable_metrics())
    entities = ", ".join(model.entities)
    user = (f"QUESTION: {question}\n\nSEMANTIC FACTS:\n{semantic_block}\n\nCOLUMNS:\n{schema_text}\n\n"
            f"METRICS:\n{metrics}\nENTITIES: {entities}\n\nDRAFT PLAN:\n{draft.model_dump_json(exclude_defaults=True)}\n\n"
            f"JSON SCHEMA:\n{json.dumps(AnalysisPlan.model_json_schema())}")
    from .llm import extract_json

    for _attempt in range(2):
        raw = llm.complete(PLANNER_SYSTEM, user, role="planner")
        payload = extract_json(raw)
        if payload is None:
            notes.append("Model returned no usable plan; kept the deterministic plan.")
            return draft, None, notes
        if "cannot_answer" in payload and len(payload) == 1:
            return draft, str(payload["cannot_answer"]), notes
        try:
            plan = AnalysisPlan.model_validate(payload)
            errors = _check_refs(plan, model)
        except ValidationError as e:
            errors = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
            plan = None
        if not errors:
            if plan.model_dump() != draft.model_dump():
                notes.append("A model adjusted the deterministic plan; the adjusted plan passed validation.")
            return plan, None, notes
        user += "\n\nYOUR PLAN WAS INVALID:\n" + "\n".join(errors) + "\nReturn a corrected JSON plan."
    notes.append("Model plan failed validation twice; kept the deterministic plan.")
    return draft, None, notes
