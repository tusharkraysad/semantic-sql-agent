"""Number formatting shared by the answer writer and the UI."""

from __future__ import annotations


def fmt(value, unit: str = "", is_ratio: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if is_ratio or unit == "%":
        return f"{v:,.1f}%"
    if abs(v) >= 100:
        return f"{v:,.1f}"
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:,.4f}"
