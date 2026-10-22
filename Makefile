.PHONY: setup data build test eval api ui lint compose compose-full kind-up kind-deploy kind-down

PY ?= python3
export PYTHONPATH := src

setup:            ## create a virtualenv with everything needed for development
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt

data:             ## generate the synthetic demo data
	$(PY) scripts/generate_data.py --out data

build: data       ## validate data, build and promote artifacts
	$(PY) -m sqlagent.preprocess --promote

test:             ## unit tests
	$(PY) -m pytest -q

eval:             ## golden-set evaluation, logged to MLflow (sqlite:///mlflow.db)
	MLFLOW_DISABLE_AGENT_HINT=1 $(PY) eval/run_eval.py --min-accuracy 0.95 --compare-baseline

lint:
	ruff check .

api: build        ## run the API on :8000
	uvicorn sqlagent.api:app --port 8000

ui:               ## run the chat UI on :8501 (embedded mode unless API_URL is set)
	streamlit run ui/app.py

compose:          ## API + UI in Docker
	docker compose up --build

compose-full:     ## + MLflow, evaluation job, Prometheus and Grafana
	docker compose --profile mlops --profile monitoring up --build

kind-up:          ## local Kubernetes cluster
	kind create cluster --name sqlagent

kind-deploy:      ## build images, load them into kind, deploy with kustomize
	docker build -f docker/Dockerfile --target api -t semantic-sql-agent:ci .
	docker build -f docker/Dockerfile --target ui -t semantic-sql-agent-ui:ci .
	kind load docker-image semantic-sql-agent:ci semantic-sql-agent-ui:ci --name sqlagent
	kubectl apply -k deploy/k8s/overlays/kind
	kubectl -n semantic-sql wait --for=condition=complete job/semantic-sql-preprocess --timeout=300s
	kubectl -n semantic-sql rollout status deploy/semantic-sql-api --timeout=300s
	@echo "kubectl -n semantic-sql port-forward svc/semantic-sql-ui 8501:8501"

kind-down:
	kind delete cluster --name sqlagent
