"""
Question parser
===============
Reads the SHAPE of the question (ranking, trend, pareto, growth, 2x2...) and
builds a structured, validated plan from the resolver's facts. Deterministic:
the same question always gives the same plan and the same confidence score.

The confidence score is computed from visible evidence, not asked of a model:
it starts at 1.0 and loses points for each thing that could not be pinned down.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .semantic.model import ROLE_TIME, SemanticModel
from .semantic.resolver import SemanticContext
from .text_utils import normalize

Archetype = Literal["total", "ranking", "breakdown", "trend", "pareto", "growth", "quadrant", "compare"]


class MeasureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["column", "metric"]
    ref: str = Field(description="'table.column' for a column, or the metric name")


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str = Field(description="'table.column' where the value was found")
    values: list[str]
    negated: bool = False


class AnalysisPlan(BaseModel):
    """The contract between understanding and SQL. Code renders it; nothing else does."""

    model_config = ConfigDict(extra="forbid")
    archetype: Archetype
    measures: list[MeasureSpec] = Field(min_length=1, max_length=2)
    group_by: list[str] = Field(default_factory=list, description="'entity:<name>', 'table.column' or 'time'")
    filters: list[FilterSpec] = Field(default_factory=list)
    top_n: int | None = Field(default=None, ge=1, le=100)
    direction: Literal["desc", "asc"] = "desc"
    time_phrase: str | None = None
    time_negated: bool = False
    compare_months: int | None = None


PENALTIES = {
    "no_measure": 0.45,
    "unresolved_period": 0.25,
    "weak_value": 0.20,
    "ambiguous_value": 0.15,
    "unresolved_term": 0.12,
    "unverified_synonym": 0.10,
    "warning": 0.05,
}

_TOP_N = re.compile(r"\b(top|bottom|best|worst|leading|largest|smallest|highest|lowest|biggest)\s+(\d{1,3})\b")
_PARETO = re.compile(r"\b(pareto|cumulative|80\s*/\s*20|80 20)\b")
_QUADRANT = re.compile(r"\b(quadrant|2x2|two by two|bcg|matrix)\b")
_GROWTH = re.compile(r"\b(growth|grew|grow|growing|grown|declin\w*|increase\w*|decrease\w*|change|yoy|year on year|fastest)\b")
_TREND = re.compile(r"\b(trend|over time|monthly|month on month|by month|per month|each month|month wise|month-wise)\b")
_RANK = re.compile(r"\b(top|bottom|highest|lowest|best|worst|leading|largest|smallest|biggest|rank|ranking|most|least|which)\b")
_ASC = re.compile(r"\b(bottom|lowest|worst|smallest|least)\b")
_BY = re.compile(r"\b(by|per|across|split|breakdown|each|wise)\b")
_FOLLOWUP = re.compile(r"^\s*(now|and|what about|how about|same for|same but|also|then)\b")


class ParsedQuestion(BaseModel):
    plan: AnalysisPlan | None = None
    confidence: float = 1.0
    penalties: dict[str, float] = Field(default_factory=dict)
    needs_deep: bool = False
    clarify: str | None = None
    followup: bool = False
    notes: list[str] = Field(default_factory=list)


def _archetype(q: str, n_measures: int, has_group: bool, wants_compare: bool) -> str:
    if _PARETO.search(q):
        return "pareto"
    if _QUADRANT.search(q) and n_measures >= 2:
        return "quadrant"
    if _GROWTH.search(q) or wants_compare:
        return "growth"
    if _TREND.search(q):
        return "trend"
    if n_measures >= 2 and has_group:
        return "compare"
    if _TOP_N.search(q) or (_RANK.search(q) and has_group):
        return "ranking"
    if has_group:
        return "breakdown"
    return "total"


class QuestionParser:
    def __init__(self, model: SemanticModel, confidence_floor: float = 0.55):
        self.model = model
        self.floor = confidence_floor

    def parse(self, ctx: SemanticContext, previous: dict | None = None) -> ParsedQuestion:
        q = " " + normalize(ctx.question) + " "
        out = ParsedQuestion()
        measures = self._measures(ctx)
        is_followup = bool(previous) and (bool(_FOLLOWUP.search(ctx.question.lower())) or not measures)

        if is_followup and not measures:
            prev = AnalysisPlan.model_validate(previous["plan"])
            out.followup = True
            measures = list(prev.measures)
            base_group, base_arch = list(prev.group_by), prev.archetype
            top_n, direction = prev.top_n, prev.direction
            time_phrase = prev.time_phrase if ctx.time_window is None else ctx.time_window.phrase
            filters = self._merge_filters(prev.filters, self._filters(ctx))
            out.notes.append("Follow-up: kept the previous question's measure, grouping and period, and changed only what this question names.")
            plan = AnalysisPlan(archetype=base_arch, measures=measures, group_by=base_group, filters=filters,
                                top_n=top_n, direction=direction, time_phrase=time_phrase,
                                time_negated=ctx.time_window.negated if ctx.time_window else prev.time_negated,
                                compare_months=prev.compare_months if ctx.time_window is None else ctx.time_window.compare_months)
            out.plan = plan
        else:
            if not measures:
                out.clarify = ("I could not find a measure in that question. Try naming what to measure, for example "
                               "turnover, gross margin, market share or marketing spend.")
                out.confidence = 1.0 - PENALTIES["no_measure"]
                out.penalties["no_measure"] = PENALTIES["no_measure"]
                if ctx.unresolved_terms:
                    out.clarify += f" Words I did not recognise: {', '.join(ctx.unresolved_terms)}."
                return out
            group = self._group_by(ctx, q, measures)
            wants_compare = bool(ctx.time_window and ctx.time_window.compare_months)
            arch = _archetype(q, len(measures), bool(group), wants_compare)
            if arch in ("total",) and len(measures) > 1:
                measures = measures[:1]
            if arch not in ("quadrant", "compare") and len(measures) > 1:
                measures = measures[:1]
            if arch == "trend":
                group = ["time"]
            if arch in ("ranking", "pareto", "growth", "quadrant", "compare") and not group:
                group = self._default_group(measures)
                out.notes.append(f"No breakdown was named, so the answer is broken down by {group[0].split(':')[-1]}.")
            top_n = None
            m = _TOP_N.search(q)
            if m:
                top_n = int(m.group(2))
            elif arch in ("ranking", "growth") and re.search(r"\b(top|bottom|best|worst|fastest|highest|lowest)\b", q):
                top_n = 5
            elif arch == "ranking" and re.search(r"\bwhich\b", q) and not re.search(r"\b(brands|regions|skus|categories)\b", q):
                top_n = 1
            plan = AnalysisPlan(
                archetype=arch, measures=measures, group_by=group, filters=self._filters(ctx), top_n=top_n,
                direction="asc" if _ASC.search(q) else "desc",
                time_phrase=ctx.time_window.phrase if ctx.time_window else None,
                time_negated=bool(ctx.time_window and ctx.time_window.negated),
                compare_months=ctx.time_window.compare_months if ctx.time_window else None,
            )
            out.plan = plan

        out.needs_deep = out.plan.archetype in ("pareto", "growth", "quadrant", "compare") or len(self._tables(out.plan)) > 1
        self._score(ctx, out)
        return out

    # ------------------------------------------------------------------------
    def _measures(self, ctx: SemanticContext) -> list[MeasureSpec]:
        metric_terms = {ctx.term_of.get(f"metric:{m.name}") for m in ctx.metrics}
        specs = [MeasureSpec(kind="metric", ref=m.name) for m in ctx.metrics]
        for col in ctx.measures:
            if ctx.term_of.get(f"column:{col.ref}") in metric_terms:
                continue  # 'gross margin' asked: the metric, not the profit column sharing the words
            if any(s.kind == "column" and s.ref.split(".")[0] == col.table and
                   ctx.term_of.get(f"column:{s.ref}") == ctx.term_of.get(f"column:{col.ref}") for s in specs):
                continue  # one phrase, one measure: the strongest match already took it
            specs.append(MeasureSpec(kind="column", ref=col.ref))
        return specs[:2]

    def _tables(self, plan: AnalysisPlan) -> set[str]:
        tables = set()
        for m in plan.measures:
            if m.kind == "column":
                tables.add(m.ref.split(".", 1)[0])
            else:
                tables.update(self.model.metrics[m.ref].tables)
        return tables

    def _primary_table(self, measures: list[MeasureSpec]) -> str:
        m = measures[0]
        if m.kind == "column":
            return m.ref.split(".", 1)[0]
        metric = self.model.metrics[m.ref]
        return metric.numerator.split(".", 1)[0]

    def _group_by(self, ctx: SemanticContext, q: str, measures: list[MeasureSpec]) -> list[str]:
        filtered_cols = {m.ref for m in ctx.value_matches}
        filtered_entities = {self.model.columns[m.ref].entity for m in ctx.value_matches if m.ref in self.model.columns}
        table = self._primary_table(measures)
        out: list[str] = []
        for col in ctx.dimensions:
            if col.ref in filtered_cols or col.role == ROLE_TIME:
                continue
            if col.entity and col.entity in filtered_entities and not _BY.search(q):
                continue
            key = f"entity:{col.entity}" if col.entity else col.ref
            # A named column in the measure's own file beats the entity in general.
            if col.table == table and not col.entity:
                key = col.ref
            if col.entity and col.table == table:
                key = col.ref
            if key not in out and not any(o.startswith("entity:") and key.startswith("entity:") for o in out):
                if col.entity and any(self.model.columns.get(o) and self.model.columns[o].entity == col.entity for o in out):
                    continue
                out.append(key)
        return out[:1]

    def _default_group(self, measures: list[MeasureSpec]) -> list[str]:
        table = self._primary_table(measures)
        brand = self.model.entities.get("brand")
        if brand and brand.column_in(table):
            return [f"{table}.{brand.column_in(table)}"]
        dims = [c for c in self.model.columns_of(table) if c.role == "dimension"]
        return [dims[0].ref] if dims else []

    @staticmethod
    def _filters(ctx: SemanticContext) -> list[FilterSpec]:
        by_col: dict[tuple[str, bool], list[str]] = {}
        for m in ctx.value_matches:
            by_col.setdefault((m.ref, m.negated), [])
            if m.value not in by_col[(m.ref, m.negated)]:
                by_col[(m.ref, m.negated)].append(m.value)
        return [FilterSpec(column=c, values=v, negated=n) for (c, n), v in by_col.items()]

    def _merge_filters(self, old: list[FilterSpec], new: list[FilterSpec]) -> list[FilterSpec]:
        def entity(ref: str) -> str:
            col = self.model.columns.get(ref)
            return (col.entity or ref) if col else ref

        replaced = {entity(f.column) for f in new}
        return [f for f in old if entity(f.column) not in replaced] + new

    def _score(self, ctx: SemanticContext, out: ParsedQuestion) -> None:
        pen: dict[str, float] = {}
        plan = out.plan
        if ctx.time_window is not None:
            tables = self._tables(plan)
            if tables and not any(ctx.windows.get(t) and ctx.windows[t].keys for t in tables):
                pen["unresolved_period"] = PENALTIES["unresolved_period"]
        if any(m.method not in ("exact", "case") and m.score < 0.95 for m in ctx.value_matches):
            pen["weak_value"] = PENALTIES["weak_value"]
        if any(m.alternatives for m in ctx.value_matches):
            pen["ambiguous_value"] = PENALTIES["ambiguous_value"]
        if ctx.unresolved_terms:
            pen["unresolved_term"] = PENALTIES["unresolved_term"]
        if ctx.unverified_matches:
            pen["unverified_synonym"] = PENALTIES["unverified_synonym"]
        if ctx.warnings:
            pen["warning"] = PENALTIES["warning"] * min(len(ctx.warnings), 3)
        out.penalties = pen
        out.confidence = round(max(0.0, 1.0 - sum(pen.values())), 2)
