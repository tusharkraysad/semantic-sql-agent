"""
Preprocess: build (and optionally promote) the artifacts the service loads.

    python -m sqlagent.preprocess                 # build only, print the version
    python -m sqlagent.preprocess --promote       # build and make it current
    python -m sqlagent.preprocess --generate      # also generate synthetic data first

In Kubernetes this runs as a Job; in Airflow it is the build task.
"""

from __future__ import annotations

import argparse
import json
import os

from . import synthetic
from .artifacts import build, promote
from .config import get_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--generate", action="store_true", help="generate synthetic data into DATA_DIR if it is empty")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    s = get_settings()
    if args.generate and not os.path.exists(os.path.join(s.data_dir, "data_sources.yaml")):
        synthetic.main(["--out", s.data_dir, "--seed", str(args.seed)])
    version = build(s)
    if args.promote:
        promote(s, version)
    print(json.dumps({"version": version, "promoted": args.promote}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
