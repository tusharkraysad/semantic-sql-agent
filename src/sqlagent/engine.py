"""
Engine
======
Holds every loaded component for one artifact version and swaps the whole set
atomically when a newer version is promoted, so a data refresh never serves a
half-updated mix of old indexes and new data.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from . import artifacts, synthetic
from .composer import Composer
from .config import Settings
from .executor import Executor
from .logging_utils import get_logger
from .observability import DATA_VERSION
from .parser import QuestionParser
from .registry import Registry
from .retriever import HybridRetriever
from .semantic.resolver import Resolver
from .sql_validator import SQLValidator

log = get_logger("engine")


@dataclass
class Loaded:
    version: str
    manifest: dict
    registry: Registry
    model: object
    values: object
    periods: object
    resolver: Resolver
    parser: QuestionParser
    composer: Composer
    validator: SQLValidator
    executor: Executor
    retriever: HybridRetriever


class Engine:
    def __init__(self, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()
        self.loaded: Loaded | None = None
        self._current_mtime = 0.0
        self.load()

    def _ensure_data(self) -> None:
        if os.path.exists(os.path.join(self.s.data_dir, "data_sources.yaml")):
            return
        if not self.s.generate_if_missing:
            raise FileNotFoundError(f"No data_sources.yaml in {self.s.data_dir}. Set GENERATE_DEMO_DATA=1 for synthetic data.")
        log.info("No data found; generating synthetic demo data into %s", self.s.data_dir)
        synthetic.main(["--out", self.s.data_dir])

    def load(self) -> None:
        self._ensure_data()
        registry = Registry.load(self.s.data_dir)
        version = artifacts.current_version(self.s)
        try:
            if version is None:
                raise FileNotFoundError("no promoted artifacts")
            model, values, periods, manifest = artifacts.load(self.s, version, registry)
        except (FileNotFoundError, artifacts.ArtifactMismatch) as e:
            if not self.s.auto_build:
                raise
            log.warning("%s - building artifacts now (AUTO_BUILD=1)", e)
            version = artifacts.build(self.s, registry)
            artifacts.promote(self.s, version)
            model, values, periods, manifest = artifacts.load(self.s, version, registry)
        loaded = Loaded(
            version=version, manifest=manifest, registry=registry, model=model, values=values, periods=periods,
            resolver=Resolver(model, values, periods),
            parser=QuestionParser(model, self.s.confidence_floor),
            composer=Composer(model, periods, values, self.s.max_result_rows),
            validator=SQLValidator(model, values, {t: f.columns for t, f in registry.frames.items()}),
            executor=Executor(registry.frames, self.s.max_result_rows, self.s.max_concurrent_queries),
            retriever=HybridRetriever(model, self.s),
        )
        with self._lock:
            old = self.loaded.version if self.loaded else None
            self.loaded = loaded
            self._current_mtime = self._pointer_mtime()
        DATA_VERSION.clear()
        DATA_VERSION.labels(version=version).set(1)
        log.info("Serving artifacts %s (was %s)", version, old)

    def _pointer_mtime(self) -> float:
        path = os.path.join(self.s.artifact_root, "CURRENT")
        return os.path.getmtime(path) if os.path.exists(path) else 0.0

    def maybe_reload(self) -> bool:
        """Reload when a new version was promoted (the Airflow DAG's promote step)."""
        if self._pointer_mtime() <= self._current_mtime:
            return False
        if artifacts.current_version(self.s) == (self.loaded.version if self.loaded else None):
            self._current_mtime = self._pointer_mtime()
            return False
        self.load()
        return True
