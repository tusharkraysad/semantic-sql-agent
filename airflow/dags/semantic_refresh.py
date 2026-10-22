"""
Semantic layer refresh DAG
==========================

    detect_new_data -> validate_data -> build_artifacts -> evaluate -> promote -> verify_api

Runs daily and stops early when the data has not changed. A new build is
promoted only if it passes the golden-set evaluation (logged to MLflow), so a
bad data drop or a broken semantic change never reaches users. Rollback is
promoting the previous version; the API picks it up on reload.

RUN_MODE chooses where the heavy steps execute:
  local       (default) tasks import the sqlagent package in the Airflow worker
  kubernetes  validate, build, evaluate and promote run as pods from the same
              images the API uses (KubernetesPodOperator), against the shared
              state volume, so the code that builds artifacts is the code that serves them

Project imports live inside tasks, so parsing this file needs only Airflow.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

PROJECT = os.environ.get("SQLAGENT_PROJECT", "/opt/project")
RUN_MODE = os.environ.get("RUN_MODE", "local")
API_URL = os.environ.get("SQLAGENT_API_URL", "http://semantic-sql-api.semantic-sql:8000")
MIN_ACCURACY = os.environ.get("EVAL_MIN_ACCURACY", "0.9")
API_IMAGE = os.environ.get("SQLAGENT_IMAGE", "ghcr.io/tusharkraysad/semantic-sql-agent:main")
TOOLS_IMAGE = os.environ.get("SQLAGENT_TOOLS_IMAGE", "ghcr.io/tusharkraysad/semantic-sql-agent-tools:main")

default_args = {"owner": "data-platform", "retries": 1, "retry_delay": timedelta(minutes=2)}


def _project_path() -> None:
    src = os.path.join(PROJECT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


@dag(
    dag_id="semantic_layer_refresh",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["semantic-sql-agent", "mlops"],
    doc_md=__doc__,
)
def semantic_layer_refresh():
    @task(retries=3, retry_delay=timedelta(seconds=20))
    def verify_api(version: str) -> None:
        """Success in Airflow is not success for users: confirm the API now serves the new version."""
        import requests

        if isinstance(version, dict):
            version = version["version"]
        requests.post(f"{API_URL}/admin/reload", timeout=30).raise_for_status()
        served = requests.get(f"{API_URL}/version", timeout=10).json()["artifact_version"]
        if served != version:
            raise RuntimeError(f"API serves {served}, expected {version}")

    if RUN_MODE == "kubernetes":
        from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
        from kubernetes.client import models as k8s

        common = dict(
            namespace="semantic-sql",
            volumes=[k8s.V1Volume(name="state", persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
                claim_name="semantic-sql-state"))],
            volume_mounts=[k8s.V1VolumeMount(name="state", mount_path="/srv/state")],
            env_from=[k8s.V1EnvFromSource(config_map_ref=k8s.V1ConfigMapEnvSource(name="semantic-sql-config"))],
            get_logs=True,
            on_finish_action="delete_pod",
        )
        validate = KubernetesPodOperator(task_id="validate_data", name="sqlagent-validate", image=API_IMAGE,
                                         cmds=["python", "-m", "sqlagent.preprocess", "--validate-only"], **common)
        build = KubernetesPodOperator(task_id="build_artifacts", name="sqlagent-build", image=API_IMAGE,
                                      cmds=["python", "-m", "sqlagent.preprocess", "--skip-validation"],
                                      do_xcom_push=True, **common)
        version = "{{ ti.xcom_pull(task_ids='build_artifacts')['version'] }}"
        evaluate = KubernetesPodOperator(
            task_id="evaluate", name="sqlagent-eval", image=TOOLS_IMAGE,
            cmds=["python", "eval/run_eval.py", "--min-accuracy", MIN_ACCURACY, "--compare-baseline"],
            env_vars={"ARTIFACT_VERSION": version, "STATE_DIR": "/tmp/eval-state", "EVAL_TRIGGER": "airflow",
                      "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow.semantic-sql:5000")},
            **common)
        promote = KubernetesPodOperator(
            task_id="promote", name="sqlagent-promote", image=API_IMAGE,
            cmds=["python", "-c", "import sys; from sqlagent.config import get_settings; from sqlagent.artifacts import promote; "
                  "promote(get_settings(), sys.argv[1])", version], **common)
        validate >> build >> evaluate >> promote >> verify_api(version)
        return

    @task.short_circuit
    def detect_new_data() -> bool:
        _project_path()
        from sqlagent import artifacts
        from sqlagent.config import get_settings
        from sqlagent.registry import Registry

        s = get_settings()
        current = artifacts.current_version(s)
        data_version = Registry.load(s.data_dir).data_version()
        changed = not current or not current.startswith(data_version)
        print(json.dumps({"current": current, "data_version": data_version, "changed": changed}))
        return changed

    @task
    def validate_data() -> dict:
        _project_path()
        from sqlagent import quality
        from sqlagent.config import get_settings
        from sqlagent.registry import Registry

        s = get_settings()
        reg = Registry.load(s.data_dir)
        problems = quality.check(reg, s)
        if problems:
            raise ValueError("Data validation failed: " + "; ".join(problems))
        return {t: f.height for t, f in reg.frames.items()}

    @task
    def build_artifacts(row_counts: dict) -> str:
        _project_path()
        from sqlagent import artifacts
        from sqlagent.config import get_settings

        version = artifacts.build(get_settings())
        print(json.dumps({"built": version, "rows": row_counts}))
        return version

    @task
    def evaluate(version: str) -> dict:
        """Golden-set evaluation against the NEW build, not the one being served."""
        import subprocess

        env = dict(os.environ, ARTIFACT_VERSION=version, STATE_DIR="/tmp/eval-state", EVAL_TRIGGER="airflow")
        proc = subprocess.run([sys.executable, os.path.join(PROJECT, "eval", "run_eval.py"), "--min-accuracy", MIN_ACCURACY,
                               "--compare-baseline"], env=env, capture_output=True, text=True)
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            raise RuntimeError(f"Evaluation gate failed for {version}; not promoting.\n{proc.stderr[-2000:]}")
        with open(os.path.join(PROJECT, "eval", "reports", "report.json"), encoding="utf-8") as f:
            return {"version": version, "metrics": json.load(f)["metrics"]}

    @task
    def promote(result: dict) -> str:
        _project_path()
        from sqlagent import artifacts
        from sqlagent.config import get_settings

        artifacts.promote(get_settings(), result["version"])
        return result["version"]

    gate = detect_new_data()
    rows = validate_data()
    gate >> rows
    verify_api(promote(evaluate(build_artifacts(rows))))


semantic_layer_refresh()
