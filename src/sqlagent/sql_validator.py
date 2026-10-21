"""
SQL validator
=============
Runs on every SQL string before execution, including composer output, as a
second line of defence. Read-only single SELECT, no star, only known tables
and columns, no SUM over a repeated base or a ratio column, and every string
literal compared to a dimension must exist in the data exactly as written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from .semantic.model import SemanticModel
from .semantic.value_index import ValueIndex

_FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command, exp.Merge, exp.TruncateTable)


class UnsafeSQL(Exception):
    pass


def assert_read_only(sql: str) -> None:
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"SQL does not parse: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQL("Exactly one statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.Subquery)) and not stmt.find(exp.Select):
        raise UnsafeSQL("Only SELECT queries are allowed.")
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN):
            raise UnsafeSQL(f"{type(node).__name__} statements are not allowed.")


@dataclass
class Verdict:
    ok: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class SQLValidator:
    def __init__(self, model: SemanticModel, values: ValueIndex, frames_columns: dict[str, list[str]]):
        self.model = model
        self.values = values
        self.columns = frames_columns
        self._all_columns = {c for cols in frames_columns.values() for c in cols}

    def validate(self, sql: str) -> Verdict:
        v = Verdict(ok=True)
        if "--" in sql or "/*" in sql:
            v.errors.append("Comments are not allowed in generated SQL.")
        try:
            assert_read_only(sql)
            stmt = sqlglot.parse_one(sql, read="postgres")
        except (UnsafeSQL, sqlglot.errors.ParseError) as e:
            return Verdict(False, [str(e)])

        if any(isinstance(s, exp.Star) for s in stmt.find_all(exp.Star)):
            v.errors.append("SELECT * is not allowed; name the columns.")

        cte_names = {c.alias_or_name for c in stmt.find_all(exp.CTE)}
        tables = set()
        for t in stmt.find_all(exp.Table):
            if t.name in cte_names:
                continue
            if t.name not in self.columns:
                v.errors.append(f"Unknown table {t.name!r}.")
            tables.add(t.name)

        aliases = {a.alias for a in stmt.find_all(exp.Alias)} | {a.alias_or_name for a in stmt.find_all(exp.TableAlias)}
        known = {c for t in tables for c in self.columns.get(t, [])}
        for col in stmt.find_all(exp.Column):
            name = col.name
            if not name or name in aliases or name in known:
                continue
            if name in self._all_columns:
                v.errors.append(f"Column {name!r} is not in the tables this query reads.")
            else:
                v.errors.append(f"Unknown column {name!r}.")

        for agg in stmt.find_all(exp.Sum):
            for col in agg.find_all(exp.Column):
                sem = self._semantic(col.name, tables)
                if sem is None:
                    continue
                if sem.is_repeated_base:
                    v.errors.append(f"SUM over {sem.column!r} counts a repeated total many times; de-duplicate it per "
                                    f"{', '.join(sem.grain)} with MAX() first.")
                if not sem.additive:
                    v.errors.append(f"{sem.column!r} is a ratio; it cannot be summed.")

        for node in list(stmt.find_all(exp.EQ)) + list(stmt.find_all(exp.NEQ)) + list(stmt.find_all(exp.In)):
            col = node.this if isinstance(node, exp.In) else (node.left if isinstance(node.left, exp.Column) else node.right)
            if not isinstance(col, exp.Column):
                continue
            lits = node.expressions if isinstance(node, exp.In) else [node.right if col is node.left else node.left]
            for lit in lits:
                if not isinstance(lit, exp.Literal) or not lit.is_string:
                    continue
                ref = self._indexed_ref(col.name, tables)
                if ref and not self.values.contains(ref, lit.this):
                    near = self.values.suggest(ref, lit.this, 3)
                    v.errors.append(f"{lit.this!r} is not a value of {ref}. Closest real values: {near}.")

        for t in tables:
            st = self.model.tables.get(t)
            for df in (st.default_filters if st else []):
                col_name = df.split('"')[1] if '"' in df else ""
                if col_name and not any(c.name == col_name for c in stmt.find_all(exp.Column)):
                    v.warnings.append(f"{t} is read without its default filter {df}; totals may mix levels.")
        v.ok = not v.errors
        return v

    def _semantic(self, column: str, tables: set[str]):
        for t in tables:
            sem = self.model.column(t, column)
            if sem is not None:
                return sem
        return None

    def _indexed_ref(self, column: str, tables: set[str]) -> str | None:
        refs = [f"{t}.{column}" for t in tables if self.values.has(f"{t}.{column}")]
        return refs[0] if len(refs) == 1 else None
