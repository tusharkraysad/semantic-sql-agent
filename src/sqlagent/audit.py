"""Append-only decision log: one JSON line per question, written even when the question fails."""

from __future__ import annotations

import json
import os
import threading
import time

_lock = threading.Lock()


def record(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload}
    line = json.dumps(payload, default=str)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tail(path: str, n: int = 50) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(x) for x in lines if x.strip()]
