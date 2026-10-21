"""
Answer cache
============
Keyed by the resolved plan plus the artifact version, not by the question
text, so 'top 5 brands by revenue 2025' and 'Top 5 brands by turnover in
2025' share an entry, and a data refresh invalidates everything at once.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict


class AnswerCache:
    def __init__(self, size: int = 256):
        self.size = size
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(version: str, plan: dict) -> str:
        return hashlib.sha256((version + json.dumps(plan, sort_keys=True)).encode()).hexdigest()[:24]

    def get(self, key: str) -> dict | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)
            return val

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)
