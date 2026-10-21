"""
Resolver
========
Turns a question into checkable business facts before any model sees it:
which measures and metrics it names, which dimensions it groups by, which real
values it filters on (and which it excludes), which periods it means in each
file, and which words nothing understood. Pure lookup and string matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..text_utils import (
    ANALYSIS_WORDS,
    NEGATION_CUES,
    STOPWORDS,
    negated_terms,
    ngrams,
    normalize,
    similarity,
    singularize,
    tokens,
)
from .model import ROLE_DIMENSION, ROLE_TIME, SOURCE_ENRICHMENT, Metric, SemanticColumn, SemanticModel
from .time_layer import PeriodCatalog, TimeWindow
from .value_index import ValueIndex, ValueMatch

NEGATION_WORDS = {t for cue in NEGATION_CUES for t in cue.split()}


@dataclass
class SemanticContext:
    question: str
    measures: list = field(default_factory=list)          # SemanticColumn, strongest first
    metrics: list = field(default_factory=list)           # Metric
    dimensions: list = field(default_factory=list)        # SemanticColumn named in the question (group-by candidates)
    entities: list = field(default_factory=list)          # entity names named in the question
    value_matches: list = field(default_factory=list)     # ValueMatch used as filters
    shape_values: list = field(default_factory=list)      # 'matrix' read as the answer's shape
    table_scores: dict = field(default_factory=dict)
    time_window: TimeWindow | None = None
    windows: dict = field(default_factory=dict)           # table -> ResolvedWindow
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    unresolved_terms: list = field(default_factory=list)
    unverified_matches: list = field(default_factory=list)
    matched_terms: dict = field(default_factory=dict)     # term -> what it matched (for the audit record)
    term_of: dict = field(default_factory=dict)           # "kind:key" -> the question phrase that matched it

    @property
    def candidate_tables(self) -> list[str]:
        return [t for t, s in sorted(self.table_scores.items(), key=lambda kv: -kv[1]) if s > 0]

    def summary(self) -> dict:
        return {
            "measures": [c.ref for c in self.measures],
            "metrics": [m.name for m in self.metrics],
            "dimensions": [c.ref for c in self.dimensions],
            "entities": list(self.entities),
            "filters": [{"column": m.ref, "value": m.value, "term": m.term, "method": m.method,
                         "negated": m.negated, "alternatives": m.alternatives} for m in self.value_matches],
            "time": self.time_window.phrase if self.time_window else None,
            "windows": {t: w.label for t, w in self.windows.items()},
            "table_scores": {k: round(v, 2) for k, v in self.table_scores.items()},
            "unresolved_terms": self.unresolved_terms,
            "unverified_matches": self.unverified_matches,
        }


class Resolver:
    def __init__(self, model: SemanticModel, values: ValueIndex, periods: PeriodCatalog):
        self.model = model
        self.values = values
        self.periods = periods
        self.terms = model.term_index()

    # ------------------------------------------------------------------------
    def resolve(self, question: str) -> SemanticContext:
        ctx = SemanticContext(question=question)
        toks = tokens(question)
        covered = [False] * len(toks)

        # 1. Real values first, so 'Spark Ultra' is a filter and not two loose words.
        linked = [m for m in self.values.link(question) if not self._is_schema_word(m.term)]
        ctx.shape_values = [m for m in linked if self._names_the_shape(m, question)]
        shape_ids = {id(m) for m in ctx.shape_values}
        linked = self._prefer_coarsest_column([m for m in linked if id(m) not in shape_ids])
        ctx.value_matches = linked
        for m in linked + ctx.shape_values:
            self._cover(toks, covered, m.term)

        # 2. Business terms, longest phrase first, never reusing a consumed word.
        hits: dict[tuple[str, str], tuple[float, str]] = {}
        for start, n, phrase in ngrams(toks, 4):
            if any(covered[start : start + n]) or all(t in STOPWORDS for t in phrase.split()):
                continue
            found = list(self.terms.get(phrase, ()))
            sing = singularize(phrase)
            if sing != phrase:
                found += [h for h in self.terms.get(sing, ()) if h not in found]
            if not found:
                continue
            covered[start : start + n] = [True] * n
            weight = float(n) ** 1.5
            for kind, key in found:
                score = weight + (self._name_affinity(phrase, key) if kind == "column" else 0.0)
                if score > hits.get((kind, key), (0.0, ""))[0]:
                    hits[(kind, key)] = (score, phrase)

        term_best: dict[tuple[str, str], float] = {}

        def credit(table: str, term: str, score: float) -> None:
            term_best[(table, term)] = max(term_best.get((table, term), 0.0), score)

        measures, dims = [], []
        for (kind, key), (score, term) in hits.items():
            ctx.matched_terms[term] = f"{kind}:{key}"
            ctx.term_of[f"{kind}:{key}"] = term
            if kind == "column":
                col = self.model.columns[key]
                (measures if col.is_measure else dims).append((score, col))
                credit(col.table, term, score)
            elif kind == "metric":
                metric = self.model.metrics[key]
                ctx.metrics.append(metric)
                for t in metric.tables:
                    credit(t, f"metric:{term}", 2.0 + score)
            elif kind == "entity":
                ctx.entities.append(key)
                for ref in self.model.entities[key].columns:
                    col = self.model.columns[ref]
                    dims.append((score * 0.9, col))
                    credit(col.table, term, score * 0.5)
            elif kind == "table":
                credit(key, f"table:{term}", score)
        for m in ctx.value_matches:
            credit(m.table, f"value:{m.term}", 1.5 * m.score)

        ctx.measures = self._dedupe([c for _, c in sorted(measures, key=lambda p: -p[0])])
        ctx.dimensions = self._dedupe([c for _, c in sorted(dims, key=lambda p: -p[0])
                                       if c.role in (ROLE_DIMENSION, ROLE_TIME)])
        for (table, _), s in term_best.items():
            ctx.table_scores[table] = ctx.table_scores.get(table, 0.0) + s

        # 3. Negation, then time per file.
        window = self.periods.parse(question)
        cands = [m.term for m in ctx.value_matches] + ([window.phrase] if window else [])
        negated = negated_terms(question, cands)
        for m in ctx.value_matches:
            m.negated = m.term in negated
        if window is not None:
            window.negated = window.phrase in negated
            ctx.time_window = window
            for table in self.periods.axes:
                rw = self.periods.resolve(window, table)
                if rw is not None:
                    ctx.windows[table] = rw
            self._cover(toks, covered, window.phrase)

        ctx.unresolved_terms = self._leftovers(toks, covered)
        ctx.unverified_matches = self._unverified(hits)
        ctx.assumptions = self._assumptions(ctx)
        ctx.warnings = self._warnings(ctx)
        return ctx

    # ------------------------------------------------------------------------
    @staticmethod
    def _cover(toks: list[str], covered: list[bool], phrase: str) -> None:
        needle = tokens(phrase)
        for s in range(len(toks) - len(needle) + 1):
            if toks[s : s + len(needle)] == needle:
                covered[s : s + len(needle)] = [True] * len(needle)

    @staticmethod
    def _dedupe(cols: list[SemanticColumn]) -> list[SemanticColumn]:
        seen, out = set(), []
        for c in cols:
            if c.ref not in seen:
                seen.add(c.ref)
                out.append(c)
        return out

    def _is_schema_word(self, term: str) -> bool:
        """'brand' and 'category' name columns; used as words they describe the breakdown, not a row."""
        core = " ".join(t for t in tokens(term) if not t.isdigit())
        for cand in {normalize(term), singularize(term), core, singularize(core)}:
            if not cand:
                continue
            if any(kind in ("column", "entity", "table", "metric") for kind, _ in self.terms.get(cand, ())):
                return True
        return False

    def _names_the_shape(self, match: ValueMatch, question: str) -> bool:
        """'matrix' is a 2x2 AND a brand. The shape wins unless the question says 'matrix brand'."""
        if normalize(match.term) not in ANALYSIS_WORDS:
            return False
        q = normalize(question)
        col = self.model.columns.get(match.ref)
        names = {normalize(match.column)} | ({normalize(col.entity)} if col and col.entity else set())
        names |= {singularize(n) for n in names}
        term = normalize(match.term)
        return not any(f"{n} {term}" in q or f"{term} {n}" in q for n in names if n)

    def _prefer_coarsest_column(self, matches: list[ValueMatch]) -> list[ValueMatch]:
        """
        When one phrase matches values in two columns of the same file, keep the
        column with fewer distinct values: 'skin care' is a category, even though
        category-level rows also carry it in the BRAND column.
        """
        best: dict[tuple[str, str], ValueMatch] = {}
        for m in matches:
            key = (m.table, normalize(m.term))
            cur = best.get(key)
            if cur is None:
                best[key] = m
                continue
            n_new = len(self.values.values.get(m.ref, ())) or 10**9
            n_cur = len(self.values.values.get(cur.ref, ())) or 10**9
            if n_new < n_cur:
                best[key] = m
        return [m for m in matches if best.get((m.table, normalize(m.term))) is m]

    def _name_affinity(self, term: str, ref: str) -> float:
        """Tie-break: 'turnover' means 'Total Turnover', not 'Total Acquisition Turnover'."""
        col = self.model.columns.get(ref)
        if col is None:
            return 0.0
        return 0.9 * max((similarity(term, normalize(c)) for c in (col.column, col.label, *col.synonyms) if c), default=0.0)

    def _leftovers(self, toks: list[str], covered: list[bool]) -> list[str]:
        runs, cur = [], []
        for tok, cov in zip(toks, covered):
            if cov or tok in STOPWORDS or tok in ANALYSIS_WORDS or tok in NEGATION_WORDS or tok.isdigit() or len(tok) < 2:
                if cur:
                    runs.append(" ".join(cur))
                    cur = []
                continue
            cur.append(tok)
        if cur:
            runs.append(" ".join(cur))
        return list(dict.fromkeys(runs))

    def _unverified(self, hits) -> list[str]:
        out = []
        for (kind, key), (_, term) in hits.items():
            obj = {"column": self.model.columns, "metric": self.model.metrics, "entity": self.model.entities,
                   "table": self.model.tables}[kind].get(key)
            if obj is not None and getattr(obj, "synonym_sources", {}).get(normalize(term)) == SOURCE_ENRICHMENT:
                out.append(f"'{term}' was read as {key} through a model-suggested synonym nobody has approved")
        return out

    def _assumptions(self, ctx: SemanticContext) -> list[str]:
        out = []
        for m in ctx.value_matches:
            if m.method != "exact":
                out.append(f"Read '{m.term}' as {m.column} = '{m.value}' ({m.method} match).")
            if m.negated:
                out.append(f"Read '{m.term}' as an exclusion: every row except {m.column} = '{m.value}', keeping rows where it is empty.")
        for m in ctx.shape_values:
            out.append(f"Read '{m.term}' as the shape of the answer, not a filter on {m.column} = '{m.value}'. "
                       f"Say '{m.column.lower()} {m.term}' if the filter was meant.")
        if ctx.time_window and ctx.time_window.negated:
            out.append(f"Read '{ctx.time_window.phrase}' as an exclusion: those periods are left out of the totals.")
        return out

    def _warnings(self, ctx: SemanticContext) -> list[str]:
        out = []
        for m in ctx.value_matches:
            if m.alternatives:
                out.append(f"'{m.term}' also closely matches {m.alternatives[:3]}; answered for '{m.value}'.")
        for metric in ctx.metrics:
            if metric.status == "provisional":
                out.append(f"{metric.display} uses a provisional definition not yet confirmed by the data owner; treat it as indicative.")
        labels = {w.label for w in ctx.windows.values() if w.keys}
        if len(labels) > 1 and len(ctx.candidate_tables) > 1:
            out.append("The files end in different months, so the same time phrase covers different periods in each file.")
        return out


def metric_of(ctx: SemanticContext, name: str) -> Metric | None:
    for m in ctx.metrics:
        if m.name == name:
            return m
    return None
