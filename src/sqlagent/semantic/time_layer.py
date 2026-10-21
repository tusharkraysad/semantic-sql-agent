"""
Time layer
==========
Turns 'last 3 months', 'YTD', 'dec-25', 'Q3 2025', 'vs last year' into exact
filters on each file's own time column, deterministically.

The three files speak three time languages (a real date, 'dec-25' text, and
'YYYYMMDD' text with forecast suffixes) and end in different months. So a
relative phrase is resolved against EACH file's newest period, never against
the wall clock, and per-file windows are reported when they differ.
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import date

import polars as pl

from ..text_utils import quote_ident, quote_literal
from .model import ROLE_TIME, SemanticModel

MONTHS = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}
MONTHS.update({calendar.month_name[i].lower(): i for i in range(1, 13)})
MONTHS["sept"] = 9

FMT_DATE, FMT_MON_YY, FMT_YYYYMMDD, FMT_ISO = "date", "mon_yy", "yyyymmdd", "iso"


def key_of(year: int, month: int) -> int:
    return year * 100 + month


def shift(key: int, months: int) -> int:
    y, m = divmod(key, 100)
    idx = y * 12 + (m - 1) + months
    return key_of(idx // 12, idx % 12 + 1)


def label_of(key: int) -> str:
    y, m = divmod(key, 100)
    return f"{calendar.month_abbr[m]}-{y % 100:02d}"


def parse_literal(raw) -> tuple[str, int] | None:
    if isinstance(raw, date):
        return FMT_DATE, key_of(raw.year, raw.month)
    text = str(raw).strip().lower()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return FMT_ISO, key_of(int(m[1]), int(m[2]))
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})", text)
    if m and 1 <= int(m[2]) <= 12:
        return FMT_YYYYMMDD, key_of(int(m[1]), int(m[2]))
    m = re.fullmatch(r"([a-z]{3,9})[-/ ]'?(\d{2}|\d{4})", text)
    if m and m[1] in MONTHS:
        year = int(m[2]) if len(m[2]) == 4 else 2000 + int(m[2])
        return FMT_MON_YY, key_of(year, MONTHS[m[1]])
    return None


@dataclass
class TimeAxis:
    table: str
    column: str
    fmt: str
    periods: dict = field(default_factory=dict)  # key -> literal as stored
    unparsed: list = field(default_factory=list)

    @property
    def keys(self) -> list[int]:
        return sorted(self.periods)

    @property
    def latest(self) -> int | None:
        return max(self.periods) if self.periods else None

    @property
    def earliest(self) -> int | None:
        return min(self.periods) if self.periods else None

    def literal(self, key: int) -> str:
        raw = self.periods[key]
        return f"CAST({quote_literal(raw)} AS DATE)" if self.fmt == FMT_DATE else quote_literal(raw)

    def filter_sql(self, keys: list[int], negated: bool = False) -> str:
        if not keys:
            return ""
        col = quote_ident(self.column)
        if self.fmt == FMT_DATE:
            body = f"{col} BETWEEN {self.literal(keys[0])} AND {self.literal(keys[-1])}"
            if negated:
                body = f"{col} NOT BETWEEN {self.literal(keys[0])} AND {self.literal(keys[-1])}"
        else:
            # Text periods do not sort ('dec-25' > 'apr-26'), so list them.
            values = ", ".join(self.literal(k) for k in keys)
            body = f"{col} {'NOT ' if negated else ''}IN ({values})"
        return f"({col} IS NULL OR {body})" if negated else body

    def sort_expr(self) -> str:
        """An expression that orders this axis chronologically, even for text periods."""
        col = quote_ident(self.column)
        if self.fmt == FMT_DATE:
            return col
        cases = " ".join(f"WHEN {quote_literal(v)} THEN {k}" for k, v in sorted(self.periods.items()))
        return f"CASE {col} {cases} END"


@dataclass
class TimeWindow:
    phrase: str
    kind: str                    # explicit | last_n | latest | ytd | year
    start: int | None = None
    end: int | None = None
    n: int = 1
    compare_months: int | None = None   # 12 = vs same period last year
    negated: bool = False


@dataclass
class ResolvedWindow:
    table: str
    column: str
    keys: list
    filter_sql: str
    label: str
    compare_keys: list = field(default_factory=list)
    compare_filter_sql: str = ""
    compare_label: str = ""
    negated: bool = False
    notes: list = field(default_factory=list)


_QUARTER = re.compile(r"\bq([1-4])\s*'?(\d{2}|\d{4})\b")
_MONTH_YEAR = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")[-/\s]'?(\d{4}|\d{2})\b")
_LAST_N = re.compile(r"\b(?:last|past|previous|trailing|recent)\s+(\d{1,2})\s+(months?|quarters?|years?)\b")
_LATEST = re.compile(r"\b(?:latest|last|this|current|most recent)\s+(month|quarter|year)\b")
_YTD = re.compile(r"\b(?:ytd|year to date)\b")
_YEAR = re.compile(r"\b(20\d{2})\b")
_YOY = re.compile(r"\b(?:yoy|year on year|year over year|vs last year|versus last year|compared to last year|same period last year)\b")
_VS_YEARS = re.compile(r"\b(20\d{2})\s+(?:vs|versus|compared to|against)\s+(20\d{2})\b")


class PeriodCatalog:
    def __init__(self, axes: dict[str, TimeAxis]):
        self.axes = axes

    @classmethod
    def build(cls, registry, model: SemanticModel) -> "PeriodCatalog":
        axes = {}
        for table, frame in registry.frames.items():
            t = model.tables.get(table)
            candidates = [t.time_column] if t and t.time_column else []
            candidates += [c.column for c in model.columns_of(table) if c.role == ROLE_TIME and c.column not in candidates]
            for column in candidates:
                distinct = frame[column].drop_nulls().unique().to_list()
                periods, unparsed, fmts = {}, [], {}
                for raw in distinct:
                    parsed = parse_literal(raw)
                    if parsed is None:
                        unparsed.append(str(raw))
                        continue
                    fmt, key = parsed
                    fmts[fmt] = fmts.get(fmt, 0) + 1
                    periods[key] = raw.isoformat() if isinstance(raw, date) else str(raw)
                if periods:
                    fmt = FMT_DATE if frame[column].dtype == pl.Date else max(fmts, key=fmts.get)
                    axes[table] = TimeAxis(table, column, fmt, periods, sorted(unparsed)[:10])
                    break
        return cls(axes)

    def save(self, path: str) -> None:
        payload = {t: {"column": a.column, "fmt": a.fmt, "periods": {str(k): v for k, v in a.periods.items()},
                       "unparsed": a.unparsed} for t, a in self.axes.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path: str) -> "PeriodCatalog":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return cls({t: TimeAxis(t, a["column"], a["fmt"], {int(k): v for k, v in a["periods"].items()}, a["unparsed"])
                    for t, a in payload.items()})

    def known_years(self) -> set[int]:
        return {k // 100 for a in self.axes.values() for k in a.periods}

    # -- parsing ---------------------------------------------------------------
    def parse(self, question: str) -> TimeWindow | None:
        q = " " + question.lower() + " "
        compare = 12 if _YOY.search(q) else None
        m = _VS_YEARS.search(q)
        if m:
            cur, prev = int(m[1]), int(m[2])
            return TimeWindow(m[0], "year", key_of(cur, 1), key_of(cur, 12), compare_months=12 * (cur - prev))
        m = _QUARTER.search(q)
        if m:
            y = int(m[2]) if len(m[2]) == 4 else 2000 + int(m[2])
            first = (int(m[1]) - 1) * 3 + 1
            return TimeWindow(m[0].strip(), "explicit", key_of(y, first), key_of(y, first + 2), compare_months=compare)
        m = _MONTH_YEAR.search(q)
        if m:
            y = int(m[2]) if len(m[2]) == 4 else 2000 + int(m[2])
            k = key_of(y, MONTHS[m[1]])
            return TimeWindow(m[0].strip(), "explicit", k, k, compare_months=compare)
        m = _LAST_N.search(q)
        if m:
            n = int(m[1]) * {"m": 1, "q": 3, "y": 12}[m[2][0]]
            return TimeWindow(m[0].strip(), "last_n", n=n, compare_months=compare)
        m = _LATEST.search(q)
        if m:
            unit = m[1]
            return TimeWindow(m[0].strip(), "ytd" if unit == "year" else "last_n", n={"month": 1, "quarter": 3}.get(unit, 12),
                              compare_months=compare)
        if _YTD.search(q):
            return TimeWindow("year to date", "ytd", compare_months=compare)
        for m in _YEAR.finditer(q):
            y = int(m[1])
            if y in self.known_years():
                return TimeWindow(m[1], "year", key_of(y, 1), key_of(y, 12), compare_months=compare)
        if compare:
            return TimeWindow("latest month vs last year", "last_n", n=1, compare_months=compare)
        return None

    # -- binding ---------------------------------------------------------------
    def resolve(self, window: TimeWindow, table: str) -> ResolvedWindow | None:
        axis = self.axes.get(table)
        if axis is None or axis.latest is None:
            return None
        latest = axis.latest
        if window.kind in ("explicit", "year"):
            start, end = window.start, window.end
        elif window.kind == "ytd":
            start, end = key_of(latest // 100, 1), latest
        else:
            start, end = shift(latest, -(window.n - 1)), latest
        keys = [k for k in axis.keys if start <= k <= end]
        label = (label_of(keys[0]) if len(keys) == 1 else f"{label_of(keys[0])}..{label_of(keys[-1])}") if keys else "(none)"
        if window.negated:
            label = f"all periods except {label}"
        rw = ResolvedWindow(table, axis.column, keys, axis.filter_sql(keys, window.negated), label, negated=window.negated)
        if not keys:
            rw.notes.append(f"{table} has no data for '{window.phrase}' (it covers {label_of(axis.earliest)}..{label_of(latest)}).")
        if window.compare_months and keys:
            ck = [k for k in axis.keys if shift(keys[0], -window.compare_months) <= k <= shift(keys[-1], -window.compare_months)]
            rw.compare_keys = ck
            rw.compare_filter_sql = axis.filter_sql(ck)
            rw.compare_label = (label_of(ck[0]) if len(ck) == 1 else f"{label_of(ck[0])}..{label_of(ck[-1])}") if ck else "(none)"
            if not ck:
                rw.notes.append(f"{table} has no data for the comparison period, so growth cannot be computed from it.")
        if axis.unparsed and not window.negated:
            rw.notes.append(f'{table}."{axis.column}" also holds non-period values such as {axis.unparsed[:2]} (e.g. forecasts); they fall outside this filter.')
        return rw

    def default_growth_window(self, table: str) -> TimeWindow | None:
        """Growth with no period named: latest 12 months vs the 12 before, if the file holds two years."""
        axis = self.axes.get(table)
        if axis is None or len(axis.keys) < 2:
            return None
        if len(axis.keys) >= 24:
            return TimeWindow("latest 12 months vs prior 12 months", "last_n", n=12, compare_months=12)
        half = len(axis.keys) // 2
        return TimeWindow(f"latest {half} months vs prior {half} months", "last_n", n=half, compare_months=half)
