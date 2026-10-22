"""
Table registry
==============
Loads the configured files into Polars frames and profiles every column.
CSV is read as text first and numeric columns are then coerced only if every
non-empty value parses, so a stray 'n/a' never silently becomes a null in a
measure.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import polars as pl
import yaml

from .logging_utils import get_logger

log = get_logger("registry")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    null_ratio: float
    sample_values: list = field(default_factory=list)


@dataclass
class TableProfile:
    name: str
    path: str
    description: str
    n_rows: int
    columns: dict[str, ColumnProfile] = field(default_factory=dict)


def _coerce(frame: pl.DataFrame) -> pl.DataFrame:
    out = []
    for name in frame.columns:
        col = frame[name]
        if col.dtype != pl.Utf8:
            out.append(col)
            continue
        stripped = col.str.strip_chars()
        non_empty = stripped.filter(stripped.is_not_null() & (stripped != ""))
        if non_empty.len() == 0:
            out.append(col)
            continue
        as_date = non_empty.str.to_date("%Y-%m-%d", strict=False)
        if as_date.null_count() == 0:
            out.append(stripped.str.to_date("%Y-%m-%d", strict=False).alias(name))
            continue
        as_num = non_empty.cast(pl.Float64, strict=False)
        # Zero tolerance: one unparseable value keeps the column as text.
        if as_num.null_count() == 0 and not non_empty.str.contains(r"^\d{8}").all():
            out.append(stripped.cast(pl.Float64, strict=False).alias(name))
        else:
            out.append(col)
    return pl.DataFrame(out)


def data_present(data_dir: str) -> bool:
    """True when data_sources.yaml exists AND every file it lists is on disk."""
    cfg_path = os.path.join(data_dir, "data_sources.yaml")
    if not os.path.exists(cfg_path):
        return False
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sources = cfg.get("sources", [])
    return bool(sources) and all(os.path.exists(os.path.join(data_dir, s["path"])) for s in sources)


class Registry:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.frames: dict[str, pl.DataFrame] = {}
        self.tables: dict[str, TableProfile] = {}

    @classmethod
    def load(cls, data_dir: str) -> Registry:
        reg = cls(data_dir)
        cfg_path = os.path.join(data_dir, "data_sources.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for src in cfg.get("sources", []):
            path = os.path.join(data_dir, src["path"])
            frame = reg._read(path)
            reg.frames[src["name"]] = frame
            reg.tables[src["name"]] = reg._profile(src["name"], path, src.get("description", ""), frame)
            log.info("Loaded %s: %d rows x %d columns", src["name"], frame.height, frame.width)
        return reg

    @staticmethod
    def _read(path: str) -> pl.DataFrame:
        lower = path.lower()
        if lower.endswith((".xlsx", ".xls")):
            return _coerce(pl.read_excel(path, infer_schema_length=0))
        return _coerce(pl.read_csv(path, infer_schema=False))

    @staticmethod
    def _profile(name: str, path: str, description: str, frame: pl.DataFrame) -> TableProfile:
        prof = TableProfile(name=name, path=path, description=description, n_rows=frame.height)
        for col in frame.columns:
            s = frame[col]
            sample = s.drop_nulls().unique(maintain_order=True).head(8).to_list()
            prof.columns[col] = ColumnProfile(
                name=col,
                dtype=str(s.dtype),
                n_unique=int(s.n_unique()),
                null_ratio=float(s.null_count() / max(frame.height, 1)),
                sample_values=[str(v) for v in sample],
            )
        return prof

    def data_version(self) -> str:
        """Content hash of the source files: the identity of a data release."""
        h = hashlib.sha256()
        for name in sorted(self.tables):
            with open(self.tables[name].path, "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:12]

    def has_column(self, table: str, column: str) -> bool:
        return table in self.frames and column in self.frames[table].columns

    def table_names(self) -> list[str]:
        return list(self.frames)
