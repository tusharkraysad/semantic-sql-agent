# Hosting the demo for free

The `demo` image and the UI's embedded mode run everything in one process: they generate the synthetic data and build the artifacts on first start, so no API, database or key is needed. Free tiers change, so check each provider's current limits.

## Option 1: Streamlit Community Cloud (simplest)

1. Sign in at share.streamlit.io with GitHub and choose **Create app**.
2. Repository `tusharkraysad/semantic-sql-agent`, branch `main`, main file `ui/app.py`.
3. Advanced settings: Python 3.12. Dependencies come from `ui/requirements.txt`.
4. Optional, for model-assisted answers, add secrets in TOML form:
   ```toml
   LLM_PROVIDER = "openai-compatible"
   LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
   LLM_API_KEY = "..."
   PLANNER_MODEL = "..."
   WRITER_MODEL = "..."
   ```
   Streamlit exposes top-level secrets as environment variables.

Private repositories can be deployed, but sharing rules differ for private apps; a public repository is the simplest route for a portfolio link.

## Option 2: Hugging Face Spaces (Docker)

1. Create a new Space, SDK **Docker**, blank template.
2. Push this repository to the Space, and add a `README.md` header to the Space with `sdk: docker` and `app_port: 7860`.
3. Make the Space build the demo image: put a root `Dockerfile` in the Space containing the `wheels`, `api`, `ui` and `demo` stages from `docker/Dockerfile`, or copy `docker/Dockerfile` to the root and keep `demo` as the last stage.

## Option 3: Any Docker host (Render, Fly.io, a VM)

```bash
docker build -f docker/Dockerfile --target demo -t sqlagent-demo .
docker run -p 7860:7860 sqlagent-demo
```

The container listens on `$PORT` (default 7860).
