"""
Data quality gates
==================
Run before every artifact build (preprocess --validate, and the Airflow DAG).
Compared against the manifest of the version currently being served, so
schema drift is caught at ingest rather than discovered by a user:

  * every table present and non-empty
  * no column that existed in the served version has disappeared
  * measures are not mostly null
  * row counts have not moved by more than the allowed fraction
"""

from __future__ import annotations

import json
import os

from .config import Settings
from .registry import Registry


def previous_manifest(settings: Settings) -> dict | None:
    from .artifacts import current_version

    version = current_version(settings)
    if not version:
        return None
    path = os.path.join(settings.artifact_root, version, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(registry: Registry, settings: Settings, max_row_change: float = 0.5, max_null: float = 0.2) -> list[str]:
    problems: list[str] = []
    prev = previous_manifest(settings) or {}
    prev_cols = prev.get("columns", {})
    prev_rows = prev.get("tables", {})
    for table in prev_rows:
        if table not in registry.frames:
            problems.append(f"table {table} is missing")
    for table, frame in registry.frames.items():
        if frame.height == 0:
            problems.append(f"{table} is empty")
            continue
        lost = set(prev_cols.get(table, [])) - set(frame.columns)
        if lost:
            problems.append(f"{table} lost columns {sorted(lost)} (schema drift)")
        old = prev_rows.get(table)
        if old and abs(frame.height - old) / old > max_row_change:
            problems.append(f"{table} row count moved from {old:,} to {frame.height:,}")
        for col, dtype in frame.schema.items():
            if dtype.is_numeric():
                ratio = frame[col].null_count() / frame.height
                if ratio > max_null:
                    problems.append(f"{table}.{col} is {ratio:.0%} null")
    return problems
