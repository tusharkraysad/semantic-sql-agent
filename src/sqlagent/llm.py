"""
LLM client
==========
One small client for any OpenAI-compatible chat endpoint: Gemini's
OpenAI-compatible API, Groq, a local Ollama server, or OpenAI itself. Only the
base URL, key and model names change. With LLM_PROVIDER=none every call
returns None and the system runs fully deterministic, which is how CI and the
free demo run.

Two model roles: a planner (cheap, structured output) and a writer (prose).
"""

from __future__ import annotations

import json
import re
import time

import requests

from .config import Settings
from .logging_utils import get_logger

log = get_logger("llm")


class LLMBudgetExceeded(Exception):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def reset(self) -> None:
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def complete(self, system: str, user: str, role: str = "planner", max_tokens: int | None = None) -> str | None:
        if not self.enabled:
            return None
        if self.calls >= self.s.max_llm_calls_per_question:
            raise LLMBudgetExceeded(f"LLM call budget of {self.s.max_llm_calls_per_question} reached")
        model = self.s.planner_model if role == "planner" else (self.s.writer_model or self.s.planner_model)
        budget = max_tokens or (self.s.max_tokens_planner if role == "planner" else self.s.max_tokens_writer)
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"
        for attempt in range(3):
            payload = {"model": model, "temperature": 0, "max_tokens": budget,
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.s.llm_timeout_s)
            except requests.RequestException as e:
                log.warning("LLM request failed (%s), attempt %d", e, attempt + 1)
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(float(resp.headers.get("retry-after", 1.5 * (attempt + 1))))
                continue
            if resp.status_code >= 400:
                log.warning("LLM error %s: %s", resp.status_code, resp.text[:300])
                return None
            self.calls += 1
            data = resp.json()
            usage = data.get("usage") or {}
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            choice = (data.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content") or "").strip()
            # Reasoning models can spend the whole budget thinking and return nothing.
            if not content and choice.get("finish_reason") == "length" and budget < 8000:
                budget *= 2
                continue
            return content or None
        return None


def extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
