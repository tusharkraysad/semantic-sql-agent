# Runbook

## Deploy locally on Kubernetes (kind)

```bash
make kind-up        # kind create cluster --name sqlagent
make kind-deploy    # build images, load into kind, kubectl apply -k deploy/k8s/overlays/kind
kubectl -n semantic-sql port-forward svc/semantic-sql-ui 8501:8501
```

Order of events: the `semantic-sql-preprocess` Job generates demo data (if none), validates it, builds artifacts and promotes them. The API pod's init container waits for `artifacts/CURRENT`, then the API starts; the startup probe allows five minutes for loading.

To use a model, create the secret first (see `deploy/k8s/base/secret.example.yaml`). The Deployment reads it as optional, so without it the service runs deterministic.

## Refresh data

Automated: the `semantic_layer_refresh` Airflow DAG detects new data, validates it, builds a new version, evaluates it on the golden set, promotes it only if the gate passes, and confirms the API serves it.

Manual:

```bash
kubectl -n semantic-sql delete job semantic-sql-preprocess
kubectl apply -k deploy/k8s/overlays/kind          # re-creates and runs the Job
curl -X POST localhost:8000/admin/reload           # or wait for the next question
```

## Roll back

Promotion only rewrites `artifacts/CURRENT`. To roll back, write the previous version name into it and reload:

```bash
kubectl -n semantic-sql exec deploy/semantic-sql-api -- sh -c 'ls /srv/state/artifacts'
kubectl -n semantic-sql exec deploy/semantic-sql-api -- python -c \
  "from sqlagent.config import get_settings; from sqlagent.artifacts import promote; promote(get_settings(), '<version>')"
curl -X POST localhost:8000/admin/reload
```

## Scaling

- The API is memory-bound: each replica loads every file. Measured peak RSS on demo data is ~170 MB; re-measure on real data and set the limit to about 2 to 3 times the peak.
- Scale vertically first. To run several replicas: move artifacts to object storage (each pod copies its version at start), move the audit log and checkpoints to Postgres, switch the volume to ReadWriteMany or remove it, then add replicas and a HorizontalPodAutoscaler on request rate rather than CPU.
- For large data, replace in-process Polars with a warehouse query engine behind the same composer.

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| API pod stuck in `Init` | Preprocess Job failed or has not promoted | `kubectl -n semantic-sql logs job/semantic-sql-preprocess` |
| Job exits with code 2 | Data-quality gate failed (schema drift, empty table, row-count jump) | The JSON on the last log line lists the problems |
| `ArtifactMismatch` at startup | Data changed but artifacts were not rebuilt | Run the Job again, or set `AUTO_BUILD=1` for demos |
| Pod `OOMKilled` (exit 137) | Memory limit below the real peak | `kubectl describe pod`; measure, then raise the limit |
| New pod `ContainerCreating` forever after an update | RWO volume still attached elsewhere | The Deployment uses `Recreate`; check nobody changed it |
| Answers stale after a refresh | API has not reloaded | `GET /version`; `POST /admin/reload` |

## What to watch (Grafana dashboard)

- Share of answers below the confidence floor: rising means the vocabulary is drifting from how people ask.
- Unrecognised question words per hour: the worklist for new synonyms (also in `/decisions`).
- Empty results and clarifications: questions the layer cannot answer yet.
- Unverified figures: a model-written answer contained a number not in the result.
- p95 latency by path, cache hit rate, model tokens per minute.
