"""
Artifacts
=========
Preprocessing writes everything the service needs into a versioned folder:

    artifacts/<version>/semantic_model.yaml
    artifacts/<version>/value_index.json
    artifacts/<version>/time_index.json
    artifacts/<version>/manifest.json
    artifacts/CURRENT            <- one line: the promoted version

The version is a hash of the data files plus the semantic model, so the same
inputs always give the same version. Promotion rewrites CURRENT atomically,
which is what makes rollback a one-line change. The service refuses to serve
artifacts built for different data unless AUTO_BUILD is on.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

from . import __version__
from .config import Settings
from .logging_utils import get_logger
from .registry import Registry
from .semantic.bootstrap import apply_overrides, bootstrap, model_summary
from .semantic.model import SemanticModel
from .semantic.time_layer import PeriodCatalog
from .semantic.value_index import ValueIndex

log = get_logger("artifacts")


class ArtifactMismatch(Exception):
    pass


def build(settings: Settings, registry: Registry | None = None) -> str:
    registry = registry or Registry.load(settings.data_dir)
    model = bootstrap(registry, settings)
    notes = apply_overrides(model, settings.overrides_path)
    issues = model.validate(registry)
    if issues:
        raise ValueError("Semantic model does not match the data: " + "; ".join(issues))
    values = ValueIndex.build(registry, model, settings)
    periods = PeriodCatalog.build(registry, model)

    data_version = registry.data_version()
    model_hash = hashlib.sha256(json.dumps(model.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:8]
    version = f"{data_version}-{model_hash}"
    out = os.path.join(settings.artifact_root, version)
    os.makedirs(out, exist_ok=True)
    model.save(os.path.join(out, "semantic_model.yaml"))
    values.save(os.path.join(out, "value_index.json"))
    periods.save(os.path.join(out, "time_index.json"))
    manifest = {
        "version": version, "data_version": data_version, "model_hash": model_hash,
        "code_version": __version__, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": model_summary(model), "notes": notes,
        "tables": {t: p.n_rows for t, p in registry.tables.items()},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Built artifacts %s -> %s", version, out)
    return version


def promote(settings: Settings, version: str) -> None:
    if not os.path.isdir(os.path.join(settings.artifact_root, version)):
        raise FileNotFoundError(f"no artifacts for version {version}")
    tmp = os.path.join(settings.artifact_root, "CURRENT.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(version + "\n")
    os.replace(tmp, os.path.join(settings.artifact_root, "CURRENT"))
    log.info("Promoted artifacts %s", version)


def current_version(settings: Settings) -> str | None:
    path = os.path.join(settings.artifact_root, "CURRENT")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip() or None


def load(settings: Settings, version: str, registry: Registry) -> tuple[SemanticModel, ValueIndex, PeriodCatalog, dict]:
    folder = os.path.join(settings.artifact_root, version)
    with open(os.path.join(folder, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest["data_version"] != registry.data_version():
        raise ArtifactMismatch(f"Artifacts {version} were built for data {manifest['data_version']}, "
                               f"but the loaded data is {registry.data_version()}. Rebuild with `python -m sqlagent.preprocess`.")
    model = SemanticModel.load(os.path.join(folder, "semantic_model.yaml"))
    values = ValueIndex.load(os.path.join(folder, "value_index.json"), settings)
    periods = PeriodCatalog.load(os.path.join(folder, "time_index.json"))
    return model, values, periods, manifest
