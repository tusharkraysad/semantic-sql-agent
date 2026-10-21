"""
Semantic model
==============
Declarative business facts about the data, saved as YAML so people can review
and edit them:

  SemanticColumn  a physical column bound to a role (measure, dimension, time,
                  key), how it aggregates, whether it is a repeated base, and
                  the business words people use for it
  SemanticTable   what one file is, its default filters, its time column
  Entity          one business concept spread across files (brand, region)
  Metric          a governed calculation. Only `approved` and `provisional`
                  metrics are ever used; `draft` is invisible to the agent
  JoinPath        a permitted join with its measured value alignment

Each synonym records where it came from (human or enrichment), so a term a
model suggested can be told apart from one a person approved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ..text_utils import content_tokens, normalize, singularize

ROLE_MEASURE, ROLE_DIMENSION, ROLE_TIME, ROLE_KEY, ROLE_IGNORE = "measure", "dimension", "time", "key", "ignore"
ROLES = (ROLE_MEASURE, ROLE_DIMENSION, ROLE_TIME, ROLE_KEY, ROLE_IGNORE)
STATUS_APPROVED, STATUS_PROVISIONAL, STATUS_DRAFT = "approved", "provisional", "draft"
GRAIN_ADDITIVE, GRAIN_REPEATED = "additive", "repeated_base"
SOURCE_HUMAN, SOURCE_ENRICHMENT, SOURCE_BOOTSTRAP = "human", "enrichment", "bootstrap"


def qualify(table: str, column: str) -> str:
    return f"{table}.{column}"


def split_qualified(ref: str) -> tuple[str, str]:
    table, _, column = ref.partition(".")
    return table, column


class _Synonyms:
    synonyms: list
    synonym_sources: dict

    def canonical_terms(self) -> list[str]:
        return []

    def add_synonyms(self, terms, source: str = SOURCE_HUMAN) -> None:
        merged = set(self.synonyms)
        canonical = set(self.canonical_terms())
        for term in terms:
            clean = normalize(term)
            if len(clean) < 2:
                continue
            merged.add(clean)
            if clean in canonical:
                continue
            if source == SOURCE_HUMAN or self.synonym_sources.get(clean) != SOURCE_HUMAN:
                self.synonym_sources[clean] = source
        self.synonyms = sorted(merged)

    def remove_synonyms(self, terms) -> None:
        drop = {normalize(t) for t in terms}
        self.synonyms = [s for s in self.synonyms if s not in drop]
        for t in drop:
            self.synonym_sources.pop(t, None)

    def is_unverified(self, term: str) -> bool:
        return self.synonym_sources.get(normalize(term)) == SOURCE_ENRICHMENT


@dataclass
class SemanticColumn(_Synonyms):
    table: str
    column: str
    role: str = ROLE_DIMENSION
    label: str = ""
    description: str = ""
    entity: str | None = None
    synonyms: list = field(default_factory=list)
    synonym_sources: dict = field(default_factory=dict)
    dtype: str = ""
    n_unique: int = 0
    aggregation: str | None = None   # measures: sum | avg | max | min
    additive: bool = True            # False for ratios: never SUM
    grain_role: str = GRAIN_ADDITIVE
    grain: list = field(default_factory=list)  # for repeated bases: the key that owns one value
    value_case: str = "unknown"
    notes: str = ""

    @property
    def ref(self) -> str:
        return qualify(self.table, self.column)

    @property
    def is_measure(self) -> bool:
        return self.role == ROLE_MEASURE

    @property
    def is_repeated_base(self) -> bool:
        return self.is_measure and self.grain_role == GRAIN_REPEATED and bool(self.grain)

    @property
    def display(self) -> str:
        return self.label or self.column

    def canonical_terms(self) -> list[str]:
        return [normalize(t) for t in (self.column, self.label) if t]

    def search_terms(self) -> list[str]:
        return [normalize(t) for t in (self.column, self.label, *self.synonyms) if t]


@dataclass
class SemanticTable(_Synonyms):
    name: str
    label: str = ""
    description: str = ""
    kind: str = "fact"
    time_column: str = ""
    default_filters: list = field(default_factory=list)
    row_count: int = 0
    synonyms: list = field(default_factory=list)
    synonym_sources: dict = field(default_factory=dict)
    notes: str = ""

    def canonical_terms(self) -> list[str]:
        return [normalize(t) for t in (self.name, self.label) if t]


@dataclass
class Entity(_Synonyms):
    name: str
    label: str = ""
    columns: list = field(default_factory=list)
    synonyms: list = field(default_factory=list)
    synonym_sources: dict = field(default_factory=dict)
    notes: str = ""

    def canonical_terms(self) -> list[str]:
        return [normalize(t) for t in (self.name, self.label) if t]

    def column_in(self, table: str) -> str | None:
        for ref in self.columns:
            t, c = split_qualified(ref)
            if t == table:
                return c
        return None


@dataclass
class Metric(_Synonyms):
    name: str
    label: str = ""
    description: str = ""
    numerator: str = ""     # qualified column
    denominator: str = ""   # qualified column
    scale: float = 100.0
    required_filters: list = field(default_factory=list)  # [{table, sql}]
    synonyms: list = field(default_factory=list)
    synonym_sources: dict = field(default_factory=dict)
    status: str = STATUS_DRAFT
    unit: str = "%"
    notes: str = ""

    @property
    def usable(self) -> bool:
        return self.status in (STATUS_APPROVED, STATUS_PROVISIONAL)

    @property
    def tables(self) -> list[str]:
        return sorted({split_qualified(self.numerator)[0], split_qualified(self.denominator)[0]} - {""})

    @property
    def display(self) -> str:
        return self.label or self.name

    def canonical_terms(self) -> list[str]:
        return [normalize(t) for t in (self.name.replace("_", " "), self.label) if t]

    def search_terms(self) -> list[str]:
        return [normalize(t) for t in (self.name.replace("_", " "), self.label, *self.synonyms) if t]


@dataclass
class JoinPath:
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    entity: str = ""
    alignment: float = 0.0
    case_insensitive: bool = True
    enabled: bool = True
    notes: str = ""

    def involves(self, a: str, b: str) -> bool:
        return {self.left_table, self.right_table} == {a, b}

    def column_for(self, table: str) -> str | None:
        if table == self.left_table:
            return self.left_column
        if table == self.right_table:
            return self.right_column
        return None


@dataclass
class SemanticModel:
    version: int = 1
    tables: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    joins: list = field(default_factory=list)
    glossary: dict = field(default_factory=dict)

    # -- lookups -------------------------------------------------------------
    def column(self, table: str, column: str) -> SemanticColumn | None:
        return self.columns.get(qualify(table, column))

    def columns_of(self, table: str) -> list[SemanticColumn]:
        return [c for c in self.columns.values() if c.table == table]

    def measures_of(self, table: str) -> list[SemanticColumn]:
        return [c for c in self.columns_of(table) if c.is_measure]

    def usable_metrics(self) -> list[Metric]:
        return [m for m in self.metrics.values() if m.usable]

    def join_between(self, a: str, b: str, entity: str | None = None) -> JoinPath | None:
        for j in self.joins:
            if j.enabled and j.involves(a, b) and (entity is None or j.entity == entity):
                return j
        return None

    def term_index(self) -> dict[str, list[tuple[str, str]]]:
        """normalised term -> [(kind, key)]: full term, singular, and distinctive tokens."""
        index: dict[str, list[tuple[str, str]]] = {}

        def add(term: str, kind: str, key: str) -> None:
            term = normalize(term)
            if term:
                bucket = index.setdefault(term, [])
                if (kind, key) not in bucket:
                    bucket.append((kind, key))

        def add_all(term: str, kind: str, key: str, split_tokens: bool = True) -> None:
            add(term, kind, key)
            add(singularize(term), kind, key)
            parts = content_tokens(term)
            stripped = " ".join(parts)
            if parts and stripped != normalize(term):
                add(stripped, kind, key)
                add(singularize(stripped), kind, key)
            # Single words of a column name ('turnover' in 'Total Turnover') are safe
            # because the resolver tie-breaks on name affinity. Single words of a
            # metric synonym are not: 'turnover' must never mean 'bmi to turnover'.
            if split_tokens and len(parts) > 1:
                for p in parts:
                    add(p, kind, key)

        for key, col in self.columns.items():
            if col.role not in (ROLE_IGNORE, ROLE_KEY):
                for t in col.search_terms():
                    add_all(t, "column", key)
        for name, m in self.metrics.items():
            if m.usable:
                for t in m.search_terms():
                    add_all(t, "metric", name, split_tokens=False)
        for name, e in self.entities.items():
            for t in (e.name.replace("_", " "), e.label, *e.synonyms):
                if t:
                    add_all(t, "entity", name)
        for name, t in self.tables.items():
            for term in (name.replace("_", " "), t.label, *t.synonyms):
                if term:
                    add(term, "table", name)
        return index

    # -- persistence -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "tables": {k: asdict(v) for k, v in self.tables.items()},
            "entities": {k: asdict(v) for k, v in self.entities.items()},
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
            "joins": [asdict(j) for j in self.joins],
            "glossary": dict(self.glossary),
            "columns": {k: asdict(v) for k, v in self.columns.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticModel":
        def build(kind, payload):
            fields_ = kind.__dataclass_fields__
            return kind(**{k: v for k, v in payload.items() if k in fields_})

        return cls(
            version=d.get("version", 1),
            tables={k: build(SemanticTable, v) for k, v in (d.get("tables") or {}).items()},
            columns={k: build(SemanticColumn, v) for k, v in (d.get("columns") or {}).items()},
            entities={k: build(Entity, v) for k, v in (d.get("entities") or {}).items()},
            metrics={k: build(Metric, v) for k, v in (d.get("metrics") or {}).items()},
            joins=[build(JoinPath, j) for j in (d.get("joins") or [])],
            glossary=dict(d.get("glossary") or {}),
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True, width=120)

    @classmethod
    def load(cls, path: str) -> "SemanticModel":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def validate(self, registry) -> list[str]:
        """Anything the model claims that the loaded data does not back up."""
        issues = []
        for key, col in self.columns.items():
            if not registry.has_column(col.table, col.column):
                issues.append(f"column {key} does not exist in the data")
            if col.is_measure and not col.aggregation:
                issues.append(f"measure {key} has no aggregation")
        for name, m in self.metrics.items():
            if not m.usable:
                continue
            for ref in (m.numerator, m.denominator):
                t, c = split_qualified(ref)
                if not registry.has_column(t, c):
                    issues.append(f"metric {name} references missing column {ref}")
        for j in self.joins:
            for t, c in ((j.left_table, j.left_column), (j.right_table, j.right_column)):
                if not registry.has_column(t, c):
                    issues.append(f"join key {t}.{c} does not exist")
        return issues
