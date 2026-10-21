"""
Composer
========
Renders an AnalysisPlan as SQL, in code. The model never writes this SQL.

The rules it enforces are the ones plain text-to-SQL kept breaking:

  * repeated bases are de-duplicated to their own grain with MAX() before any
    SUM, so a category total copied onto every brand row is counted once
  * ratio metrics are computed as SUM(numerator) / SUM(denominator) at the
    requested grain, never as an average of row-level ratios
  * a table's default filters (for example LEVEL = 'brand') always apply
    unless the question filters that column itself
  * each file gets its own time filter from the time layer
  * two files are only ever joined on an approved join path, after each side
    is aggregated to the shared key, compared as LOWER(TRIM())
  * exclusions keep NULL rows; text periods are ordered chronologically
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .parser import AnalysisPlan, FilterSpec, MeasureSpec
from .semantic.model import SemanticColumn, SemanticModel, split_qualified
from .semantic.time_layer import PeriodCatalog, ResolvedWindow, TimeWindow
from .semantic.value_index import ValueIndex
from .text_utils import quote_ident as qi
from .text_utils import quote_literal as ql


class CannotCompose(Exception):
    """The plan asks for something the semantic layer cannot answer safely."""


@dataclass
class MeasureInfo:
    label: str
    unit: str = ""
    is_ratio: bool = False
    tables: list = field(default_factory=list)


@dataclass
class Composed:
    sql: str
    dims: list = field(default_factory=list)
    measures: list = field(default_factory=list)
    derived: list = field(default_factory=list)
    measure_info: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    windows: dict = field(default_factory=dict)       # table -> label used
    compare_windows: dict = field(default_factory=dict)
    post: str | None = None                            # growth | quadrant
    chart: dict = field(default_factory=dict)
    tables: list = field(default_factory=list)


class Composer:
    def __init__(self, model: SemanticModel, periods: PeriodCatalog, values: ValueIndex, max_rows: int = 200):
        self.model = model
        self.periods = periods
        self.values = values
        self.max_rows = max_rows

    # ------------------------------------------------------------------------
    def compose(self, plan: AnalysisPlan) -> Composed:
        self._ctes: list[str] = []
        self._notes: list[str] = []
        self._windows: dict[str, str] = {}
        self._cmp_windows: dict[str, str] = {}
        self._n = 0
        window = self._window(plan)
        infos = [self._info(m) for m in plan.measures]
        tables = sorted({t for i in infos for t in i.tables})
        fold = len(tables) > 1
        group_cols = {t: self._group_columns(plan, t) for t in tables}
        if plan.group_by and plan.group_by != ["time"]:
            self._check_joinable(plan, tables, fold)
        if plan.group_by == ["time"] and len(tables) > 1:
            raise CannotCompose("A trend across two files is not supported: they store time in different formats "
                                "and end in different months. Ask for each measure's trend separately.")

        if plan.archetype == "growth":
            cur = self._measure_cte(plan.measures[0], plan, group_cols, window, "cur", fold)
            cmp_window = window
            prev = self._measure_cte(plan.measures[0], plan, group_cols, cmp_window, "cmp", fold)
            return self._final_growth(plan, infos[0], cur, prev, group_cols, tables, fold)

        names = [self._measure_cte(m, plan, group_cols, window, "cur", fold) for m in plan.measures]
        if len(names) == 1:
            return self._final_single(plan, infos[0], names[0], group_cols, tables, fold)
        return self._final_multi(plan, infos, names, group_cols, tables, fold)

    # -- measure metadata -------------------------------------------------------
    def _info(self, m: MeasureSpec) -> MeasureInfo:
        if m.kind == "metric":
            metric = self.model.metrics.get(m.ref)
            if metric is None or not metric.usable:
                raise CannotCompose(f"'{m.ref}' is not an approved metric.")
            if metric.status == "provisional":
                self._notes.append(f"{metric.display} uses a provisional definition; treat the figure as indicative.")
            return MeasureInfo(metric.display, metric.unit, True, metric.tables)
        col = self.model.columns.get(m.ref)
        if col is None or not col.is_measure:
            raise CannotCompose(f"'{m.ref}' is not a known measure.")
        return MeasureInfo(col.display, "", not col.additive, [col.table])

    # -- time ---------------------------------------------------------------------
    def _window(self, plan: AnalysisPlan) -> TimeWindow | None:
        w = self.periods.parse(plan.time_phrase) if plan.time_phrase else None
        if w is not None:
            w.negated = plan.time_negated
            if plan.compare_months:
                w.compare_months = plan.compare_months
        if plan.archetype == "growth":
            if w is None:
                table = self._primary_table(plan)
                w = self.periods.default_growth_window(table)
                if w is None:
                    raise CannotCompose("This file does not hold enough periods to measure growth.")
                self._notes.append(f"No period was named, so growth compares the {w.phrase}.")
            elif not w.compare_months:
                w.compare_months = 12
        return w

    def _resolved(self, window: TimeWindow | None, table: str) -> ResolvedWindow | None:
        if window is None:
            return None
        return self.periods.resolve(window, table)

    # -- grouping ---------------------------------------------------------------
    def _primary_table(self, plan: AnalysisPlan) -> str:
        m = plan.measures[0]
        if m.kind == "column":
            return m.ref.split(".", 1)[0]
        return split_qualified(self.model.metrics[m.ref].numerator)[0]

    def _group_columns(self, plan: AnalysisPlan, table: str) -> list[SemanticColumn]:
        cols = []
        for g in plan.group_by:
            if g == "time":
                axis = self.periods.axes.get(table)
                if axis is None:
                    raise CannotCompose(f"{table} has no time column to trend over.")
                cols.append(self.model.column(table, axis.column))
                continue
            if g.startswith("entity:"):
                entity = self.model.entities.get(g.split(":", 1)[1])
                col_name = entity.column_in(table) if entity else None
                if col_name is None:
                    raise CannotCompose(f"{table} has no {g.split(':', 1)[1]} column to break the answer down by.")
                cols.append(self.model.column(table, col_name))
                continue
            col = self.model.columns.get(g)
            if col is None:
                raise CannotCompose(f"Unknown grouping column {g}.")
            if col.table != table:
                entity = self.model.entities.get(col.entity) if col.entity else None
                mapped = entity.column_in(table) if entity else None
                if mapped is None:
                    raise CannotCompose(f"{table} has no column matching {col.column}.")
                col = self.model.column(table, mapped)
            cols.append(col)
        return cols

    def _check_joinable(self, plan: AnalysisPlan, tables: list[str], fold: bool) -> None:
        if not fold:
            return
        for i, a in enumerate(tables):
            for b in tables[i + 1 :]:
                ca, cb = self._group_columns(plan, a), self._group_columns(plan, b)
                for x, y in zip(ca, cb):
                    if x.entity is None or x.entity != y.entity or self.model.join_between(a, b, x.entity) is None:
                        disabled = [j for j in self.model.joins if j.involves(a, b) and not j.enabled]
                        reason = f" ({disabled[0].notes})" if disabled else ""
                        raise CannotCompose(f"There is no approved join between {a} and {b} on {x.column}{reason}. "
                                            f"Ask for each file separately.")
                    j = self.model.join_between(a, b, x.entity)
                    self._notes.append(f"{a} and {b} are joined on {x.entity} after each is aggregated separately "
                                       f"({j.alignment:.0%} of values align; unmatched names drop out).")

    # -- WHERE --------------------------------------------------------------------
    def _where(self, plan: AnalysisPlan, table: str, window: TimeWindow | None, which: str) -> list[str]:
        where: list[str] = []
        filtered_cols: set[str] = set()
        merged: dict[tuple[str, bool], list[str]] = {}
        for f in plan.filters:
            mapped = self._map_filter(f, table)
            if mapped is None:
                continue
            col, vals = mapped
            filtered_cols.add(col)
            bucket = merged.setdefault((col, f.negated), [])
            bucket.extend(v for v in vals if v not in bucket)
        for (col, negated), vals in merged.items():
            lits = ", ".join(ql(v) for v in vals)
            if negated:
                where.append(f"({qi(col)} IS NULL OR {qi(col)} NOT IN ({lits}))")
            elif len(vals) == 1:
                where.append(f"{qi(col)} = {lits}")
            else:
                where.append(f"{qi(col)} IN ({lits})")
        t = self.model.tables.get(table)
        for df in (t.default_filters if t else []):
            col_in_filter = df.split('"')[1] if '"' in df else ""
            if col_in_filter not in filtered_cols:
                where.append(df)
        rw = self._resolved(window, table)
        if rw is not None:
            sql = rw.filter_sql if which == "cur" else rw.compare_filter_sql
            keys = rw.keys if which == "cur" else rw.compare_keys
            label = rw.label if which == "cur" else rw.compare_label
            if not keys:
                raise CannotCompose(f"{table} has no data for that period. " + " ".join(rw.notes))
            where.append(sql)
            (self._windows if which == "cur" else self._cmp_windows)[table] = label
            for n in rw.notes:
                if n not in self._notes:
                    self._notes.append(n)
        return where

    def _map_filter(self, f: FilterSpec, table: str) -> tuple[str, list[str]] | None:
        src = self.model.columns.get(f.column)
        ftable, fcol = split_qualified(f.column)
        if ftable == table:
            return fcol, list(f.values)
        if src is None or not src.entity:
            return None
        entity = self.model.entities.get(src.entity)
        target = entity.column_in(table) if entity else None
        if target is None:
            return None
        out = []
        for v in f.values:
            canon, method, _ = self.values.canonicalize(f"{table}.{target}", v)
            if canon is None:
                note = f"'{v}' does not exist in {table}.{target}, so that filter could not be applied to {table}."
                if note not in self._notes:
                    self._notes.append(note)
                continue
            if method not in ("exact", "case"):
                self._notes.append(f"In {table}, '{v}' was matched to '{canon}' ({method}).")
            out.append(canon)
        return (target, out) if out else None

    # -- measure CTEs -------------------------------------------------------------
    def _name(self, stem: str) -> str:
        self._n += 1
        return f"{stem}{self._n}"

    def _group_select(self, cols: list[SemanticColumn], table: str, fold: bool) -> tuple[list[str], list[str]]:
        """(select expressions with aliases, group-by expressions) for the shared aliases g0, g1, ..."""
        sel, grp = [], []
        for i, c in enumerate(cols):
            expr = f"LOWER(TRIM(CAST({qi(c.column)} AS VARCHAR)))" if fold else qi(c.column)
            sel.append(f"{expr} AS g{i}")
            grp.append(expr)
            axis = self.periods.axes.get(table)
            if axis and axis.column == c.column and not fold:
                sel.append(f"{axis.sort_expr()} AS s{i}")
                grp.append(axis.sort_expr())
        return sel, grp

    def _aliases(self, cols: list[SemanticColumn], table: str, fold: bool) -> list[str]:
        out = []
        for i, c in enumerate(cols):
            out.append(f"g{i}")
            axis = self.periods.axes.get(table)
            if axis and axis.column == c.column and not fold:
                out.append(f"s{i}")
        return out

    def _column_agg(self, table: str, col: SemanticColumn, cols: list[SemanticColumn], where: list[str], fold: bool) -> str:
        sel, grp = self._group_select(cols, table, fold)
        aliases = self._aliases(cols, table, fold)
        wsql = (" WHERE " + " AND ".join(where)) if where else ""
        name = self._name("m")
        if col.is_repeated_base:
            base = self._name("base")
            grain = ", ".join(qi(g) for g in col.grain)
            self._ctes.append(
                f"{base} AS (SELECT {', '.join(sel + [f'{qi(g)} AS k{j}' for j, g in enumerate(col.grain)])}, "
                f"MAX({qi(col.column)}) AS _v FROM {table}{wsql} GROUP BY {', '.join(grp + [grain])})")
            gsel = ", ".join(aliases)
            self._ctes.append(f"{name} AS (SELECT {gsel + ', ' if gsel else ''}SUM(_v) AS v FROM {base}"
                              f"{' GROUP BY ' + gsel if gsel else ''})")
            note = (f"{col.display} is one figure per {', '.join(col.grain)} repeated on every row below it, so it is "
                    f"counted once per {', '.join(col.grain)} (MAX) before adding up.")
            if note not in self._notes:
                self._notes.append(note)
        else:
            agg = "SUM" if col.additive else "AVG"
            body = ", ".join(sel + [f"CAST({agg}({qi(col.column)}) AS DOUBLE) AS v"])
            self._ctes.append(f"{name} AS (SELECT {body} FROM {table}{wsql}{' GROUP BY ' + ', '.join(grp) if grp else ''})")
        return name

    def _measure_cte(self, m: MeasureSpec, plan: AnalysisPlan, group_cols: dict, window, which: str, fold: bool) -> str:
        if m.kind == "column":
            table, column = split_qualified(m.ref)
            col = self.model.column(table, column)
            return self._column_agg(table, col, group_cols[table], self._where(plan, table, window, which), fold)
        metric = self.model.metrics[m.ref]
        nt, nc = split_qualified(metric.numerator)
        dt, dc = split_qualified(metric.denominator)
        num = self._column_agg(nt, self.model.column(nt, nc), group_cols[nt], self._where(plan, nt, window, which), fold)
        den = self._column_agg(dt, self.model.column(dt, dc), group_cols[dt], self._where(plan, dt, window, which), fold)
        aliases = self._aliases(group_cols[nt], nt, fold)
        name = self._name("r")
        proj = ", ".join(f"n.{a} AS {a}" for a in aliases)
        if aliases:
            on = " AND ".join(f"n.{a} = d.{a}" for a in self._aliases(group_cols[nt], nt, True))
            frm = f"{num} n LEFT JOIN {den} d ON {on}"
        else:
            frm = f"{num} n CROSS JOIN {den} d"
        self._ctes.append(f"{name} AS (SELECT {proj + ', ' if proj else ''}{float(metric.scale)} * n.v / NULLIF(d.v, 0) AS v FROM {frm})")
        return name

    # -- final SELECTs ------------------------------------------------------------
    def _dim_labels(self, cols: list[SemanticColumn]) -> list[str]:
        return [c.display if c.label else c.column for c in cols]

    def _label_cte(self, table: str, cols: list[SemanticColumn]) -> str:
        """Display names for folded join keys, taken from the primary file."""
        name = self._name("lbl")
        sel = [f"LOWER(TRIM(CAST({qi(c.column)} AS VARCHAR))) AS g{i}" for i, c in enumerate(cols)]
        lab = [f"MAX({qi(c.column)}) AS l{i}" for i, c in enumerate(cols)]
        grp = [f"LOWER(TRIM(CAST({qi(c.column)} AS VARCHAR)))" for c in cols]
        self._ctes.append(f"{name} AS (SELECT {', '.join(sel + lab)} FROM {table} GROUP BY {', '.join(grp)})")
        return name

    def _dims_projection(self, plan, group_cols, tables, fold, src: str) -> tuple[str, str, list[str], str]:
        """(projection sql, extra join sql, dim labels, order-by for time)"""
        primary = self._primary_table(plan)
        cols = group_cols[primary]
        labels = self._dim_labels(cols)
        if not cols:
            return "", "", [], ""
        if fold:
            lbl = self._label_cte(primary, cols)
            proj = ", ".join(f"COALESCE(lb.l{i}, {src}.g{i}) AS {qi(labels[i])}" for i in range(len(cols)))
            join = " LEFT JOIN " + lbl + " lb ON " + " AND ".join(f"{src}.g{i} = lb.g{i}" for i in range(len(cols)))
            return proj, join, labels, ""
        proj = ", ".join(f"{src}.g{i} AS {qi(labels[i])}" for i in range(len(cols)))
        order = ""
        axis = self.periods.axes.get(primary)
        for i, c in enumerate(cols):
            if axis and axis.column == c.column:
                order = f"{src}.s{i}"
        return proj, "", labels, order

    def _wrap(self, select: str) -> str:
        return "WITH " + ",\n     ".join(self._ctes) + "\n" + select

    def _base(self, plan, tables) -> Composed:
        return Composed(sql="", notes=list(dict.fromkeys(self._notes)), windows=dict(self._windows),
                        compare_windows=dict(self._cmp_windows), tables=tables)

    def _final_single(self, plan, info: MeasureInfo, name: str, group_cols, tables, fold) -> Composed:
        mlabel = info.label
        proj, join, labels, time_order = self._dims_projection(plan, group_cols, tables, fold, "p")
        out = self._base(plan, tables)
        out.dims, out.measures, out.measure_info = labels, [mlabel], [info.__dict__]
        sep = ", " if proj else ""
        desc = "ASC" if plan.direction == "asc" else "DESC"
        limit = f" LIMIT {plan.top_n}" if plan.top_n else f" LIMIT {self.max_rows}"
        if plan.archetype == "total" or not labels:
            out.sql = self._wrap(f"SELECT p.v AS {qi(mlabel)} FROM {name} p")
            return out
        if plan.archetype == "trend":
            out.sql = self._wrap(f"SELECT {proj}{sep}p.v AS {qi(mlabel)} FROM {name} p{join} ORDER BY {time_order or 'p.g0'}")
            out.chart = {"type": "line", "x": labels[0], "y": [mlabel]}
            return out
        if plan.archetype == "pareto":
            r = self._name("pareto")
            gl = ", ".join(f"p.g{i} AS g{i}" for i in range(len(labels)))
            self._ctes.append(
                f"{r} AS (SELECT {gl}, p.v AS v, SUM(p.v) OVER (ORDER BY p.v DESC, p.g0) AS _cum, SUM(p.v) OVER () AS _tot "
                f"FROM {name} p)")
            proj2, join2, _, _ = self._dims_projection(plan, group_cols, tables, fold, "p")
            out.sql = self._wrap(
                f"SELECT {proj2}, p.v AS {qi(mlabel)}, 100.0 * p.v / NULLIF(p._tot, 0) AS {qi('Share %')}, "
                f"100.0 * p._cum / NULLIF(p._tot, 0) AS {qi('Cumulative %')} FROM {r} p{join2} "
                f"ORDER BY p.v DESC{limit}")
            out.derived = ["Share %", "Cumulative %"]
            out.chart = {"type": "bar", "x": labels[0], "y": [mlabel], "line": "Cumulative %"}
            if not plan.top_n:
                out.notes.append("Share and cumulative share are of the full total, not just the rows shown.")
            return out
        out.sql = self._wrap(f"SELECT {proj}{sep}p.v AS {qi(mlabel)} FROM {name} p{join} "
                             f"ORDER BY p.v {desc} NULLS LAST{limit}")
        out.chart = {"type": "bar", "x": labels[0], "y": [mlabel]}
        return out

    def _final_multi(self, plan, infos, names, group_cols, tables, fold) -> Composed:
        labels_m = [i.label for i in infos]
        if labels_m[0] == labels_m[1]:
            labels_m[1] += " (2)"
        proj, join, labels, _ = self._dims_projection(plan, group_cols, tables, fold, "a")
        if not labels:
            raise CannotCompose("Comparing two measures needs something to compare them across, such as brand.")
        out = self._base(plan, tables)
        out.dims, out.measures, out.measure_info = labels, labels_m, [i.__dict__ for i in infos]
        on = " AND ".join(f"a.g{i} = b.g{i}" for i in range(len(labels)))
        out.sql = self._wrap(
            f"SELECT {proj}, a.v AS {qi(labels_m[0])}, b.v AS {qi(labels_m[1])} FROM {names[0]} a "
            f"LEFT JOIN {names[1]} b ON {on}{join} ORDER BY a.v DESC NULLS LAST LIMIT {plan.top_n or self.max_rows}")
        if plan.archetype == "quadrant":
            out.post = "quadrant"
            out.chart = {"type": "scatter", "x": labels_m[0], "y": labels_m[1], "label": labels[0]}
        else:
            out.chart = {"type": "bar", "x": labels[0], "y": labels_m}
        return out

    def _final_growth(self, plan, info: MeasureInfo, cur: str, prev: str, group_cols, tables, fold) -> Composed:
        proj, join, labels, _ = self._dims_projection(plan, group_cols, tables, fold, "c")
        out = self._base(plan, tables)
        primary = self._primary_table(plan)
        cl, pl_ = self._windows.get(primary, "current"), self._cmp_windows.get(primary, "prior")
        cur_l, prev_l = f"{info.label} ({cl})", f"{info.label} ({pl_})"
        out.dims, out.measures, out.measure_info = labels, [cur_l, prev_l], [info.__dict__]
        if info.is_ratio:
            change = "Change (pp)"
            derived = f"c.v - pr.v AS {qi(change)}"
            order = qi(change)
            out.derived = [change]
        else:
            change = "Growth %"
            derived = f"c.v - pr.v AS {qi('Change')}, 100.0 * (c.v - pr.v) / NULLIF(ABS(pr.v), 0) AS {qi(change)}"
            order = qi(change)
            out.derived = ["Change", change]
        on = " AND ".join(f"c.g{i} = pr.g{i}" for i in range(len(labels))) if labels else "TRUE"
        sep = ", " if proj else ""
        out.sql = self._wrap(
            f"SELECT {proj}{sep}c.v AS {qi(cur_l)}, pr.v AS {qi(prev_l)}, {derived} FROM {cur} c "
            f"LEFT JOIN {prev} pr ON {on}{join} ORDER BY {order} {'ASC' if plan.direction == 'asc' else 'DESC'} NULLS LAST")
        out.post = "growth"
        out.chart = {"type": "bar", "x": labels[0] if labels else None, "y": [change]}
        return out
