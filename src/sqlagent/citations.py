"""
Citation check
==============
Every number written in the narrative must come from the result. Each figure
is compared with the result's numbers at the precision it was written; any
figure that cannot be traced is tagged [unverified: ...] in place, never
silently deleted, so the reader sees exactly which claim is unsupported.
"""

from __future__ import annotations

import re

import polars as pl

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def _candidates(frame: pl.DataFrame, extra: list[float]) -> list[float]:
    vals = list(extra)
    for c, t in frame.schema.items():
        if t.is_numeric():
            vals.extend(float(v) for v in frame[c].drop_nulls().to_list())
    return vals


def _supported(x: float, decimals: int, pool: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(abs(x) - abs(v)) <= tol for v in pool)


def check(text: str, frame: pl.DataFrame, question: str, extra: list[float] | None = None) -> tuple[str, list[str]]:
    pool = _candidates(frame, extra or [])
    pool += [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", question)]
    unsupported: list[str] = []

    def repl(m: re.Match) -> str:
        raw = m.group(0)
        num = raw.rstrip("%").replace(",", "")
        try:
            x = float(num)
        except ValueError:
            return raw
        decimals = len(num.split(".")[1]) if "." in num else 0
        if decimals == 0 and abs(x) <= 12:
            return raw  # small counts ("top 5", "3 brands")
        if 1900 <= x <= 2100 and decimals == 0:
            return raw  # years
        if _supported(x, decimals, pool):
            return raw
        unsupported.append(raw)
        return f"{raw} [unverified]"

    return _NUM.sub(repl, text), unsupported
