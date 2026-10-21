"""
Answer writing
==============
A deterministic writer always produces the answer from the result, so the app
works with no model at all. When a model is configured it may rewrite the
narrative for readability, but only from the facts handed to it, and its text
then goes through the citation check before anyone sees it.
"""

from __future__ import annotations

import json

import polars as pl

from .citations import check
from .formatting import fmt

WRITER_SYSTEM = """You write short, factual answers about business data for finance and commercial users.
Rules:
- Use ONLY the numbers in RESULT and FACTS. Never compute new numbers except simple differences already shown.
- 2 to 4 sentences. Lead with the direct answer. Mention the period.
- Keep any caveat in CAVEATS that changes how the numbers should be read.
- No headings, no bullet points, no emoji."""


def deterministic_answer(frame: pl.DataFrame, archetype: str, dims: list[str], measures: list[str],
                         info: list[dict], period: str) -> str:
    if frame.height == 0:
        return "No rows matched that question."
    unit = info[0].get("unit", "") if info else ""
    ratio = info[0].get("is_ratio", False) if info else False
    m0 = measures[0]
    when = f" for {period}" if period else ""
    if archetype == "total" or not dims:
        return f"{m0}{when} is {fmt(frame[m0][0], unit, ratio)}."
    d0 = dims[0]
    rows = frame.select([d0] + [c for c in frame.columns if c != d0]).to_dicts()
    if archetype == "trend":
        first, last = rows[0], rows[-1]
        hi = max(rows, key=lambda r: r[m0] if r[m0] is not None else float("-inf"))
        return (f"{m0} moved from {fmt(first[m0], unit, ratio)} in {first[d0]} to {fmt(last[m0], unit, ratio)} in {last[d0]}; "
                f"the peak was {fmt(hi[m0], unit, ratio)} in {hi[d0]}.")
    if archetype == "pareto":
        top = rows[0]
        n80 = next((i + 1 for i, r in enumerate(rows) if (r.get("Cumulative %") or 0) >= 80), None)
        text = (f"{top[d0]} is the largest {d0.lower()} by {m0}{when} at {fmt(top[m0], unit, ratio)} "
                f"({fmt(top['Share %'], '%')} of the total).")
        if n80:
            text += f" The top {n80} {d0.lower()} values make up 80% of the total."
        else:
            text += f" The {len(rows)} rows shown cover {fmt(rows[-1]['Cumulative %'], '%')} of the total."
        return text
    if archetype == "growth":
        rank_col = "Change (pp)" if "Change (pp)" in frame.columns else "Growth %"
        comparable = [r for r in rows if not r.get("Small base")]
        if not comparable:
            return "No comparable rows were found for that growth question."
        best = comparable[0]
        tail = f"{fmt(best[rank_col], '%') if rank_col == 'Growth %' else fmt(best[rank_col]) + ' pp'}"
        base_label = measures[0].split(" (")[0]
        text = (f"{best[d0]} grew the most: {base_label} went from {fmt(best[measures[1]], unit, ratio)} to "
                f"{fmt(best[measures[0]], unit, ratio)} ({tail}).")
        if len(comparable) > 1:
            text += " Next: " + ", ".join(f"{r[d0]} ({fmt(r[rank_col], '%') if rank_col == 'Growth %' else fmt(r[rank_col]) + ' pp'})"
                                          for r in comparable[1:3]) + "."
        return text
    if archetype == "quadrant":
        groups: dict[str, list[str]] = {}
        for r in rows:
            groups.setdefault(r.get("Quadrant", ""), []).append(str(r[d0]))
        parts = [f"{q}: {', '.join(v[:4])}" for q, v in groups.items()]
        return "Placed by median: " + "; ".join(parts) + "."
    if archetype == "compare":
        top = rows[0]
        return (f"{top[d0]} has the highest {m0}{when} at {fmt(top[m0], unit, ratio)}, with "
                f"{measures[1]} of {fmt(top[measures[1]], info[1].get('unit', '') if len(info) > 1 else '', info[1].get('is_ratio', False) if len(info) > 1 else False)}. "
                f"{len(rows)} {d0.lower()} values are compared.")
    top = rows[0]
    text = f"{top[d0]} {'is lowest' if len(rows) > 1 and rows[0][m0] is not None and rows[-1][m0] is not None and rows[0][m0] < rows[-1][m0] else 'leads'} on {m0}{when} at {fmt(top[m0], unit, ratio)}"
    if len(rows) > 1:
        text += ", followed by " + ", ".join(f"{r[d0]} ({fmt(r[m0], unit, ratio)})" for r in rows[1:3])
    return text + "."


def facts_for_writer(frame: pl.DataFrame, limit: int = 12) -> str:
    if frame.height <= limit:
        rows = frame.to_dicts()
    else:
        rows = frame.head(limit // 2).to_dicts() + frame.tail(limit // 2).to_dicts()
    return json.dumps(rows, default=str)


def llm_answer(llm, question: str, frame: pl.DataFrame, period: str, caveats: list[str], draft: str) -> tuple[str | None, list[str]]:
    user = (f"QUESTION: {question}\nPERIOD: {period or 'all available'}\n"
            f"RESULT (rows as JSON, {frame.height} rows total): {facts_for_writer(frame)}\n"
            f"FACTS (a correct draft you may improve): {draft}\nCAVEATS: {json.dumps(caveats)}")
    text = llm.complete(WRITER_SYSTEM, user, role="writer")
    if not text:
        return None, []
    checked, unsupported = check(text.strip(), frame, question)
    return checked, unsupported
