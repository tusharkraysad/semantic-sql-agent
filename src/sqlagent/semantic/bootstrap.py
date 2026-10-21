"""
Bootstrap
=========
Builds a first-draft semantic model from the profiled data, then merges the
human-approved overrides file on top (people always win).

Every decision here is a rule or a measurement, never a model call:

  roles          dtype and name tokens decide measure / dimension / time / key
  additivity     ratio-like names (pct, share, rate, margin) are never summed
  entities       columns grouped across files by name AND measured value overlap
  joins          proposed per entity pair with the measured alignment recorded;
                 below the threshold they are written disabled, with the reason
  repeated base  a measure that one key column fully determines while each key
                 still spans several rows is a total copied downwards: summing
                 it multiplies it, so it is marked and de-duplicated later
  hierarchy      a column whose values name other columns (brand, manufacturer,
                 category) mixes levels; the finest level becomes a default filter
  generic terms  'revenue' belongs to the plainest turnover column only, so a
                 sparse 'Total Acquisition Turnover' can never win a tie
"""

from __future__ import annotations

import polars as pl
import yaml

from ..config import Settings
from ..logging_utils import get_logger
from ..text_utils import detect_case, humanize, loose_key, normalize, tokens
from .model import (
    GRAIN_REPEATED,
    ROLE_DIMENSION,
    ROLE_KEY,
    ROLE_MEASURE,
    ROLE_TIME,
    SOURCE_BOOTSTRAP,
    SOURCE_HUMAN,
    Entity,
    JoinPath,
    Metric,
    SemanticColumn,
    SemanticModel,
    SemanticTable,
    split_qualified,
)

log = get_logger("semantic.bootstrap")

TIME_TOKENS = {"month", "period", "date", "year", "quarter", "week", "time", "fy"}
KEY_TOKENS = {"key", "id", "uuid", "code"}
RATIO_TOKENS = {"pct", "percent", "percentage", "share", "ratio", "rate", "avg", "average", "margin", "index", "price"}

# Business vocabulary the raw column names do not carry, keyed by a phrase in the name.
SYNONYM_SEEDS: dict[str, list[str]] = {
    "turnover": ["revenue", "sales", "net sales", "topline", "top line"],
    "gross profit": ["gp", "gross margin value", "profit"],
    "cop": ["cost of production", "cogs", "production cost", "cost"],
    "volume": ["tonnage", "volume", "quantity"],
    "main brand": ["brand", "brands"],
    "sub brand": ["variant", "sub brand"],
    "sku": ["product", "pack", "item"],
    "big c": ["division", "business unit"],
    "small c": ["category", "sub category"],
    "region": ["zone", "geography", "market"],
    "sales value": ["value sales", "retail sales", "offtake", "market sales"],
    "cat sales": ["category sales", "category size"],
    "manufacturer": ["company", "competitor", "player"],
    "media type": ["media", "channel"],
    "value": ["spend", "marketing spend", "bmi", "investment", "advertising spend", "a and p"],
}


def _name_tokens(name: str) -> set[str]:
    return set(tokens(name))


def classify_role(name: str, dtype: str, n_unique: int, n_rows: int) -> str:
    toks = _name_tokens(name)
    dt = dtype.lower()
    if "date" in dt:
        return ROLE_TIME
    if toks & TIME_TOKENS:
        return ROLE_TIME
    if toks & KEY_TOKENS:
        return ROLE_KEY
    if "float" in dt or "decimal" in dt:
        return ROLE_MEASURE
    if "int" in dt:
        return ROLE_MEASURE if n_unique > 50 else ROLE_DIMENSION
    if n_rows and n_unique > 0.5 * n_rows:
        return ROLE_KEY
    return ROLE_DIMENSION


# Seeds so generic that they may only apply to a column with exactly that name.
EXACT_ONLY_SEEDS = {"value"}


def seed_synonyms(column: str) -> list[str]:
    norm = normalize(column)
    found: list[str] = []
    for phrase, extra in SYNONYM_SEEDS.items():
        if phrase in EXACT_ONLY_SEEDS:
            if loose_key(phrase) == loose_key(column):
                found.extend(extra)
            continue
        if f" {phrase} " in f" {norm} " or loose_key(phrase) == loose_key(column):
            found.extend(extra)
    return sorted(set(found))


def _plainest(cols: list[SemanticColumn]) -> SemanticColumn:
    """The column without a qualifier is the ordinary case."""
    return min(cols, key=lambda c: (len(tokens(c.column)), len(c.column), c.column))


def resolve_generic_synonym_clashes(model: SemanticModel) -> None:
    for table in model.tables:
        for phrase, generic in SYNONYM_SEEDS.items():
            claimants = [c for c in model.measures_of(table) if phrase in normalize(c.column) and set(generic) & set(c.synonyms)]
            if len(claimants) < 2:
                continue
            keeper = _plainest(claimants)
            for col in claimants:
                if col is keeper:
                    continue
                col.remove_synonyms(generic)
                col.notes = f'Generic "{phrase}" vocabulary belongs to "{keeper.column}". Ask for this column by its full name.'
            log.info("%s: generic %r vocabulary kept on %r", table, phrase, keeper.column)


def _distinct(registry, table: str, column: str, cap: int) -> set[str] | None:
    frame = registry.frames.get(table)
    if frame is None or column not in frame.columns:
        return None
    s = frame[column].drop_nulls().cast(pl.Utf8).unique()
    if s.len() > cap:
        return None
    return set(s.to_list())


def measure_alignment(registry, t1: str, c1: str, t2: str, c2: str, cap: int) -> tuple[float, float]:
    """(raw overlap, normalised overlap) as a share of the smaller value set; (-1, -1) if unmeasurable."""
    a, b = _distinct(registry, t1, c1, cap), _distinct(registry, t2, c2, cap)
    if not a or not b:
        return -1.0, -1.0
    raw = len(a & b) / min(len(a), len(b))
    na, nb = {normalize(v) for v in a}, {normalize(v) for v in b}
    return raw, len(na & nb) / min(len(na), len(nb))


def build_entities(model: SemanticModel, registry, settings: Settings) -> None:
    dims = [c for c in model.columns.values() if c.role == ROLE_DIMENSION]
    groups: list[list[SemanticColumn]] = []
    for col in dims:
        placed = False
        for group in groups:
            head = group[0]
            if head.table == col.table or any(g.table == col.table for g in group):
                continue
            name_link = (
                loose_key(head.column) == loose_key(col.column)
                or bool(_name_tokens(head.column) & _name_tokens(col.column))
                or bool(set(seed_synonyms(head.column)) & ({normalize(col.column)} | set(seed_synonyms(col.column))))
            )
            if not name_link:
                continue
            _, norm = measure_alignment(registry, head.table, head.column, col.table, col.column,
                                        settings.max_values_indexed_per_column)
            if norm >= settings.min_join_value_alignment or loose_key(head.column) == loose_key(col.column):
                group.append(col)
                placed = True
                break
        if not placed:
            groups.append([col])

    for group in groups:
        if len({c.table for c in group}) < 2:
            continue
        base = min(group, key=lambda c: len(c.column))
        name = normalize(base.column).replace(" ", "_")
        if name in {"main_brand", "brand"}:
            name = "brand"
        entity = Entity(name=name, label=humanize(name), columns=sorted(c.ref for c in group),
                        notes="Same business concept across files; joins are measured before they are allowed.")
        entity.add_synonyms([normalize(c.column) for c in group] + [s for c in group for s in seed_synonyms(c.column)],
                            source=SOURCE_BOOTSTRAP)
        model.entities[name] = entity
        for c in group:
            c.entity = name


def build_joins(model: SemanticModel, registry, settings: Settings) -> None:
    joins = []
    for entity in model.entities.values():
        refs = [model.columns[r] for r in entity.columns if r in model.columns]
        for i, left in enumerate(refs):
            for right in refs[i + 1 :]:
                raw, norm = measure_alignment(registry, left.table, left.column, right.table, right.column,
                                              settings.max_values_indexed_per_column)
                enabled = norm >= settings.min_join_value_alignment
                notes = f"{norm:.0%} of values overlap after normalising"
                if not enabled:
                    notes = f"disabled: only {norm:.0%} of values overlap - the files use different encodings"
                elif raw < norm:
                    notes += "; casing differs, so the join compares LOWER(TRIM()) on both sides"
                joins.append(JoinPath(left.table, left.column, right.table, right.column, entity=entity.name,
                                      alignment=round(max(norm, 0.0), 4), case_insensitive=True,
                                      enabled=enabled, notes=notes))
    model.joins = joins


def detect_repeated_bases(model: SemanticModel, registry, settings: Settings) -> None:
    """
    A measure is a repeated base when some key column determines it (one value
    per key in >= 95% of keys) while each key still spans several rows, and the
    measure is not just mostly zero (enough distinct values per key).
    """
    for table, frame in registry.frames.items():
        n_rows = frame.height
        if n_rows < 50:
            continue
        keys = [c for c in model.columns_of(table) if c.role in (ROLE_DIMENSION, ROLE_TIME, ROLE_KEY)]
        key_uniques = {k.column: int(frame[k.column].n_unique()) for k in keys}
        for measure in model.measures_of(table):
            m_unique = int(frame[measure.column].n_unique())
            if m_unique < 2:
                continue
            candidates = sorted((k for k, n in key_uniques.items() if m_unique <= n <= 0.5 * n_rows),
                                key=lambda k: key_uniques[k])
            for key in candidates[:6]:
                grouped = frame.group_by(key).agg(pl.col(measure.column).n_unique().alias("_n"))
                ratio = float((grouped["_n"] <= 1).sum()) / max(grouped.height, 1)
                rows_per_group = n_rows / max(grouped.height, 1)
                distinct_per_group = m_unique / max(grouped.height, 1)
                if (ratio >= settings.repeated_base_constant_ratio
                        and rows_per_group >= settings.repeated_base_min_rows_per_group
                        and distinct_per_group >= settings.repeated_base_min_distinct_per_group):
                    measure.grain_role = GRAIN_REPEATED
                    measure.grain = [key]
                    measure.aggregation = "sum"
                    measure.notes = (f"Repeated base: one value per {key!r}, copied onto ~{rows_per_group:.0f} rows. "
                                     f"Summing it directly inflates it ~{rows_per_group:.0f}x; it is de-duplicated with MAX per {key!r} first.")
                    log.info("%s.%s is a repeated base (grain %s, ~%.0f rows per key)", table, measure.column, key, rows_per_group)
                    break


def detect_hierarchy_columns(model: SemanticModel, registry) -> None:
    """A column whose values name other columns of the same table mixes hierarchy levels."""
    for table, frame in registry.frames.items():
        names = {normalize(c) for c in frame.columns}
        for col in model.columns_of(table):
            if col.role != ROLE_DIMENSION or not (2 <= col.n_unique <= 6):
                continue
            values = frame[col.column].drop_nulls().cast(pl.Utf8)
            matches = {v for v in values.unique().to_list() if normalize(v) in names}
            if len(matches) < 2:
                continue
            finest = values.value_counts(sort=True)[0, 0]
            t = model.tables[table]
            filt = f'"{col.column}" = \'{finest}\''
            if filt not in t.default_filters:
                t.default_filters.append(filt)
            col.notes = (f"Hierarchy column: rows exist at levels {sorted(matches)}. Summing across levels counts the same "
                         f"sales more than once, so queries default to the finest level ({finest!r}).")
            log.info("%s.%s mixes hierarchy levels; default filter %s", table, col.column, filt)


def bootstrap(registry, settings: Settings) -> SemanticModel:
    model = SemanticModel()
    for table, prof in registry.tables.items():
        time_cols = []
        measures = 0
        for cp in prof.columns.values():
            role = classify_role(cp.name, cp.dtype, cp.n_unique, prof.n_rows)
            ratio_like = bool(_name_tokens(cp.name) & RATIO_TOKENS)
            col = SemanticColumn(
                table=table, column=cp.name, role=role, label=humanize(cp.name), dtype=cp.dtype, n_unique=cp.n_unique,
                aggregation=("avg" if ratio_like else "sum") if role == ROLE_MEASURE else None,
                additive=not ratio_like, value_case=detect_case(cp.sample_values),
            )
            col.add_synonyms(seed_synonyms(cp.name), source=SOURCE_BOOTSTRAP)
            model.columns[col.ref] = col
            measures += role == ROLE_MEASURE
            if role == ROLE_TIME:
                time_cols.append(cp.name)
        model.tables[table] = SemanticTable(
            name=table, label=humanize(table), description=prof.description,
            kind="fact" if measures else "dimension", time_column=time_cols[0] if time_cols else "",
            row_count=prof.n_rows,
        )
    build_entities(model, registry, settings)
    for entity in model.entities.values():
        for ref in entity.columns:
            model.columns[ref].add_synonyms(entity.synonyms, source=SOURCE_BOOTSTRAP)
    resolve_generic_synonym_clashes(model)
    build_joins(model, registry, settings)
    detect_repeated_bases(model, registry, settings)
    detect_hierarchy_columns(model, registry)
    return model


# --------------------------------------------------------------------------- #
def apply_overrides(model: SemanticModel, path: str) -> list[str]:
    """
    Merge the human-curated overrides file. People win on every field they set.
    Returns notes describing what was applied.
    """
    try:
        with open(path, encoding="utf-8") as f:
            ov = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return [f"no overrides file at {path}"]
    notes = []
    for ref, patch in (ov.get("columns") or {}).items():
        col = model.columns.get(ref)
        if col is None:
            notes.append(f"override for unknown column {ref} ignored")
            continue
        for key in ("role", "label", "description", "aggregation", "additive", "grain_role", "grain", "notes"):
            if key in patch:
                setattr(col, key, patch[key])
        if "synonyms" in patch:
            col.add_synonyms(patch["synonyms"], source=SOURCE_HUMAN)
        if "remove_synonyms" in patch:
            col.remove_synonyms(patch["remove_synonyms"])
    for name, patch in (ov.get("tables") or {}).items():
        t = model.tables.get(name)
        if t is None:
            continue
        for key in ("label", "description", "time_column", "default_filters", "notes"):
            if key in patch:
                setattr(t, key, patch[key])
        if "synonyms" in patch:
            t.add_synonyms(patch["synonyms"], source=SOURCE_HUMAN)
    for name, payload in (ov.get("metrics") or {}).items():
        m = Metric(name=name, **{k: v for k, v in payload.items() if k in Metric.__dataclass_fields__ and k != "synonyms"})
        m.add_synonyms(payload.get("synonyms", []), source=SOURCE_HUMAN)
        model.metrics[name] = m
    for patch in ov.get("joins") or []:
        for j in model.joins:
            if {j.left_table, j.right_table} == {patch["left_table"], patch["right_table"]} and j.entity == patch.get("entity", j.entity):
                j.enabled = patch.get("enabled", j.enabled)
                j.notes = patch.get("notes", j.notes)
    model.glossary.update(ov.get("glossary") or {})
    for name, payload in (ov.get("enrichment_synonyms") or {}).items():
        # Model-suggested vocabulary: kept, but labelled so it is never mistaken for approved.
        col = model.columns.get(name)
        if col is not None:
            col.add_synonyms(payload, source="enrichment")
    notes.append(f"applied overrides from {path}")
    return notes


def model_summary(model: SemanticModel) -> dict:
    return {
        "tables": len(model.tables),
        "columns": len(model.columns),
        "entities": sorted(model.entities),
        "joins_enabled": sum(j.enabled for j in model.joins),
        "joins_disabled": sum(not j.enabled for j in model.joins),
        "repeated_bases": sorted(c.ref for c in model.columns.values() if c.is_repeated_base),
        "metrics_usable": sorted(m.name for m in model.usable_metrics()),
    }


def metric_tables(metric: Metric) -> list[str]:
    return sorted({split_qualified(metric.numerator)[0], split_qualified(metric.denominator)[0]})
