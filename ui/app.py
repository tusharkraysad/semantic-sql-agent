"""
Streamlit chat UI.

Two ways to run:
  * client mode   API_URL is set -> talks to the FastAPI service (Compose, Kubernetes)
  * embedded mode no API_URL     -> runs the agent in-process (free hosting, laptops)
"""

from __future__ import annotations

import os
import sys
import uuid

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

API_URL = os.environ.get("API_URL", "").rstrip("/")

st.set_page_config(page_title="Semantic SQL Agent", layout="wide")


@st.cache_resource(show_spinner="Loading data and the semantic layer...")
def _embedded_agent():
    os.environ.setdefault("AUTO_BUILD", "1")
    os.environ.setdefault("GENERATE_DEMO_DATA", "1")
    from sqlagent.config import get_settings
    from sqlagent.engine import Engine
    from sqlagent.graph import Agent

    settings = get_settings()
    return Agent(Engine(settings), settings)


def ask(question: str, thread_id: str) -> dict:
    if API_URL:
        r = requests.post(f"{API_URL}/ask", json={"question": question, "thread_id": thread_id}, timeout=120)
        r.raise_for_status()
        return r.json()
    return _embedded_agent().ask(question, thread_id)


def render(resp: dict) -> None:
    st.markdown(resp.get("answer", ""))
    rows = resp.get("rows") or []
    if rows:
        df = pd.DataFrame(rows, columns=resp.get("columns") or None)
        chart = resp.get("chart") or {}
        try:
            if chart.get("type") == "line":
                st.line_chart(df.set_index(chart["x"])[chart["y"]])
            elif chart.get("type") == "bar" and chart.get("x") and len(df) > 1:
                st.bar_chart(df.set_index(chart["x"])[chart["y"]])
            elif chart.get("type") == "scatter":
                st.scatter_chart(df, x=chart["x"], y=chart["y"], color="Quadrant" if "Quadrant" in df.columns else None)
        except Exception:
            pass
        st.dataframe(df, hide_index=True, use_container_width=True)
    meta = f"Confidence {resp.get('confidence', 0):.2f}"
    if resp.get("mode"):
        meta += f" | {resp['mode']} path"
    if resp.get("cached"):
        meta += " | cached"
    meta += f" | {resp.get('latency_ms', 0):.0f} ms"
    if resp.get("llm_calls"):
        meta += f" | {resp['llm_calls']} model call(s)"
    st.caption(meta)
    for label, key in (("Assumptions", "assumptions"), ("Cautions", "warnings"), ("Notes", "notes")):
        items = resp.get(key) or []
        if items:
            with st.expander(f"{label} ({len(items)})", expanded=(key == "assumptions")):
                for item in items:
                    st.markdown(f"- {item}")
    if resp.get("sql"):
        with st.expander("SQL (built from the semantic layer, not written by a model)"):
            st.code(resp["sql"], language="sql")


st.title("Semantic SQL Agent")
st.caption("Ask about turnover, gross margin, market share and marketing spend across three files. "
           "Synthetic demo data with invented brands.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex[:12]
    st.session_state.history = []

with st.sidebar:
    st.subheader("Try")
    examples = [
        "Top 5 brands by revenue in 2025",
        "Now for 2024",
        "Gross margin by brand in 2025",
        "Market share trend for Spark",
        "Which brands grew fastest in 2025 vs 2024",
        "Pareto of SKUs by turnover in 2025",
        "2x2 matrix of brands by turnover and gross margin in 2025",
        "Marketing intensity by brand in 2025",
        "Top 3 brands by turnover excluding Lumen in 2025",
        "Category sales for hair care in jan-26",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    if st.button("New conversation", type="secondary"):
        st.session_state.thread_id = uuid.uuid4().hex[:12]
        st.session_state.history = []
    st.caption(f"Mode: {'API ' + API_URL if API_URL else 'embedded'}")

for q, resp in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        render(resp)

question = st.chat_input("Ask a question about the data") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Working..."):
            try:
                resp = ask(question, st.session_state.thread_id)
            except Exception as e:
                where = f"the service at {API_URL}" if API_URL else "the embedded agent"
                resp = {"answer": f"Something went wrong in {where}: {e}"}
        render(resp)
    st.session_state.history.append((question, resp))
