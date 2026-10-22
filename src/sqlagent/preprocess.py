"""
Preprocess: validate data, build artifacts, optionally promote them.

    python -m sqlagent.preprocess                       # build, print the version
    python -m sqlagent.preprocess --promote             # build and make it current
    python -m sqlagent.preprocess --generate --promote  # synthetic data first (demo / CI)
    python -m sqlagent.preprocess --validate-only       # quality gates only (exit 2 on failure)
    python -m sqlagent.preprocess --skip-if-unchanged   # no-op when the data is already served

In Kubernetes this runs as a Job; in Airflow as a task or a pod. When run in a
KubernetesPodOperator pod, the result is also written to /airflow/xcom/return.json.
"""

from __future__ import annotations

import argparse
import json
import os

from . import artifacts, quality, synthetic
from .config import get_settings
from .registry import Registry, data_present


def _emit(payload: dict) -> None:
    print(json.dumps(payload))
    if os.path.isdir("/airflow/xcom"):
        with open("/airflow/xcom/return.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--generate", action="store_true", help="generate synthetic data into DATA_DIR if it is empty")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--skip-if-unchanged", action="store_true")
    args = ap.parse_args(argv)
    s = get_settings()
    if args.generate and not data_present(s.data_dir):
        synthetic.main(["--out", s.data_dir, "--seed", str(args.seed)])
    registry = Registry.load(s.data_dir)
    current = artifacts.current_version(s)
    if args.skip_if_unchanged and current and current.startswith(registry.data_version()):
        _emit({"version": current, "skipped": True, "reason": "data unchanged"})
        return 0
    if not args.skip_validation:
        problems = quality.check(registry, s)
        if problems:
            _emit({"valid": False, "problems": problems})
            return 2
        if args.validate_only:
            _emit({"valid": True})
            return 0
    version = artifacts.build(s, registry)
    if args.promote:
        artifacts.promote(s, version)
    _emit({"version": version, "promoted": args.promote, "skipped": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
