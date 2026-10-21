"""Prometheus metrics. Scraped from /metrics by Prometheus in Compose or Kubernetes."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

QUESTIONS = Counter("sqlagent_questions_total", "Questions answered", ["mode", "status"])
LATENCY = Histogram("sqlagent_question_seconds", "End-to-end latency per question", ["mode"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
CONFIDENCE = Histogram("sqlagent_confidence", "Confidence score per answered question",
                       buckets=(0.2, 0.4, 0.55, 0.7, 0.8, 0.9, 0.95, 1.0))
LLM_CALLS = Counter("sqlagent_llm_calls_total", "Model calls made", ["role"])
LLM_TOKENS = Counter("sqlagent_llm_tokens_total", "Model tokens", ["kind"])
EMPTY_RESULTS = Counter("sqlagent_empty_results_total", "Queries that returned no rows")
UNVERIFIED = Counter("sqlagent_unverified_figures_total", "Figures in answers that could not be traced to the result")
CACHE = Counter("sqlagent_cache_total", "Answer cache lookups", ["result"])
CLARIFY = Counter("sqlagent_clarifications_total", "Questions answered with a clarification request")
UNRESOLVED = Counter("sqlagent_unresolved_terms_total", "Question words the semantic layer did not recognise")
DATA_VERSION = Gauge("sqlagent_artifact_info", "Loaded artifact version (value is always 1)", ["version"])
