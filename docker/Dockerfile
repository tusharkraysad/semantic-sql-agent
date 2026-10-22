# syntax=docker/dockerfile:1.7
# One Dockerfile, three images:
#   --target api    FastAPI service (also runs the preprocess Job)
#   --target ui     Streamlit chat client
#   --target tools  api + MLflow + eval harness (evaluation Job, MLflow server)

ARG PYTHON_VERSION=3.12

# ---- build wheels once, so runtime layers never need a compiler -------------
FROM python:${PYTHON_VERSION}-slim AS wheels
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt requirements-ui.txt requirements-dev.txt ./
RUN pip wheel --wheel-dir /wheels/api -r requirements.txt \
 && pip wheel --wheel-dir /wheels/ui streamlit==1.66.1 pandas==3.0.6 \
 && pip wheel --wheel-dir /wheels/tools mlflow==3.17.2

# ---- api ----------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS api
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    DATA_DIR=/srv/state/data \
    ARTIFACT_DIR=/srv/state/artifacts \
    STATE_DIR=/srv/state/run \
    SEMANTIC_OVERRIDES=/app/config/semantic_overrides.yaml \
    LOG_FORMAT=json
RUN useradd --create-home --uid 10001 app && mkdir -p /srv/state && chown -R app /srv/state
COPY --from=wheels /wheels/api /wheels
RUN pip install --no-cache-dir --no-index /wheels/* && rm -rf /wheels
WORKDIR /app
COPY src/ src/
COPY config/ config/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "sqlagent.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- ui -----------------------------------------------------------------------
FROM api AS ui
USER root
COPY --from=wheels /wheels/ui /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels streamlit pandas && rm -rf /wheels
COPY ui/ ui/
USER 10001
ENV STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false HOME=/tmp
EXPOSE 8501
HEALTHCHECK NONE
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

# ---- tools (evaluation + MLflow) -----------------------------------------------
FROM api AS tools
USER root
COPY --from=wheels /wheels/tools /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels mlflow && rm -rf /wheels
COPY eval/ eval/
ENV MLFLOW_DISABLE_AGENT_HINT=1 HOME=/tmp
USER 10001
HEALTHCHECK NONE
CMD ["python", "eval/run_eval.py", "--min-accuracy", "0.9"]

# ---- demo (one container, no API, for free hosting) -----------------------------
# Streamlit in embedded mode: generates the synthetic data and builds artifacts on
# first start. Works on Hugging Face Spaces (Docker), Render, Fly.io, etc.
FROM ui AS demo
ENV AUTO_BUILD=1 GENERATE_DEMO_DATA=1 \
    DATA_DIR=/tmp/sqlagent/data ARTIFACT_DIR=/tmp/sqlagent/artifacts STATE_DIR=/tmp/sqlagent/state \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "streamlit run ui/app.py --server.port ${PORT} --server.address 0.0.0.0"]
