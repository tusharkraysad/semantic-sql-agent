"""
Executor
========
Runs validated SQL on the in-memory Polars frames. Lazy plans with the
streaming engine first (bounded memory on wide joins), in-memory as fallback.
Concurrency is capped so a burst of heavy questions cannot exhaust the pod's
memory. Every result is read-only: the frames are never modified.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import polars as pl

from .logging_utils import get_logger
from .sql_validator import assert_read_only

log = get_logger("executor")


@dataclass
class QueryResult:
    frame: pl.DataFrame
    elapsed_ms: float
    engine: str


class Executor:
    def __init__(self, frames: dict[str, pl.DataFrame], max_rows: int = 200, max_concurrent: int = 4):
        self.ctx = pl.SQLContext(frames={k: v.lazy() for k, v in frames.items()}, eager=False)
        self.max_rows = max_rows
        self._sem = threading.BoundedSemaphore(max_concurrent)

    def run(self, sql: str) -> QueryResult:
        assert_read_only(sql)
        with self._sem:
            start = time.perf_counter()
            lazy = self.ctx.execute(sql)
            try:
                frame, engine = lazy.collect(engine="streaming"), "streaming"
            except Exception as e:  # some window plans are not streamable yet
                log.debug("streaming engine fell back to in-memory: %s", e)
                frame, engine = lazy.collect(), "in-memory"
            elapsed = (time.perf_counter() - start) * 1000
        frame = frame.head(self.max_rows)
        # Decimal results (from literals like 100.0) become plain floats for JSON and charts.
        frame = frame.with_columns([pl.col(c).cast(pl.Float64) for c, t in frame.schema.items() if t.is_decimal()])
        return QueryResult(frame, elapsed, engine)
