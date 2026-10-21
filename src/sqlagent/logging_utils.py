"""Structured logging to stdout, which is what Docker and Kubernetes collect."""

from __future__ import annotations

import json
import logging
import os
import sys


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        if os.environ.get("LOG_FORMAT", "text") == "json":
            handler.setFormatter(_JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("sqlagent")
        root.addHandler(handler)
        root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        root.propagate = False
        _configured = True
    return logging.getLogger(f"sqlagent.{name}")
