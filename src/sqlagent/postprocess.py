"""
Post-processing
===============
Deterministic steps on the result table that SQL expresses badly:

  growth    rows whose prior-period base is tiny relative to the others (a
            brand launched last year) produce absurd percentages. They are
            flagged and ranked after the comparable rows, never dropped.
  quadrant  each row is placed in a 2x2 by the median of each measure.
"""

from __future__ import annotations

import polars as pl


def growth(frame: pl.DataFrame, prior_col: str, rank_col: str, top_n: int | None, ratio: float) -> tuple[pl.DataFrame, list[str]]:
    notes: list[str] = []
    if frame.height == 0 or prior_col not in frame.columns:
        return frame, notes
    prior = frame[prior_col].abs()
    median = prior.median() or 0.0
    small = (prior.fill_null(0.0) < ratio * median) if median else pl.Series([False] * frame.height)
    frame = frame.with_columns(small.alias("Small base"))
    flagged = frame.filter(pl.col("Small base"))
    if flagged.height:
        names = ", ".join(str(v) for v in flagged[frame.columns[0]].to_list()[:4])
        notes.append(f"{names} had a prior-period base under {ratio:.0%} of the median, so its growth rate is not comparable; "
                     f"it is listed after the others with its absolute change.")
    frame = frame.sort(["Small base", rank_col], descending=[False, True], nulls_last=True)
    if top_n:
        frame = frame.head(top_n)
    return frame, notes


def quadrant(frame: pl.DataFrame, x: str, y: str) -> tuple[pl.DataFrame, list[str]]:
    if frame.height == 0:
        return frame, []
    mx, my = frame[x].median(), frame[y].median()
    label = (
        pl.when((pl.col(x) >= mx) & (pl.col(y) >= my)).then(pl.lit(f"High {x}, high {y}"))
        .when(pl.col(x) >= mx).then(pl.lit(f"High {x}, low {y}"))
        .when(pl.col(y) >= my).then(pl.lit(f"Low {x}, high {y}"))
        .otherwise(pl.lit(f"Low {x}, low {y}"))
    )
    return frame.with_columns(label.alias("Quadrant")), [f"Quadrants split at the median {x} ({mx:,.2f}) and median {y} ({my:,.2f})."]
