"""
Value index
===========
Every real value of every dimension column, plus the matching that turns a
phrase in a question ('dove', 'DOVE ', 'spark ultra', 'acg', 'brightwav') into
the exact string stored in the data. Deterministic, no model call.

Matching order: exact (normalised) -> containment -> initials -> fuzzy. Near
ties are reported as alternatives instead of being silently resolved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import polars as pl

from ..config import Settings
from ..text_utils import STOPWORDS, content_tokens, loose_key, ngrams, normalize, quote_ident, quote_literal, similarity, tokens
from .model import ROLE_DIMENSION, SemanticModel, split_qualified


def _trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def _initials(value: str) -> str:
    parts = content_tokens(value)
    return "".join(p[0] for p in parts) if len(parts) >= 2 else ""


@dataclass
class ValueMatch:
    table: str
    column: str
    value: str
    term: str
    score: float
    method: str
    alternatives: list = field(default_factory=list)
    negated: bool = False

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"

    def predicate(self, alias: str | None = None) -> str:
        col = f"{alias}.{quote_ident(self.column)}" if alias else quote_ident(self.column)
        lit = quote_literal(self.value)
        if self.negated:
            # `col <> x` drops NULL rows the question never asked to remove.
            return f"({col} IS NULL OR {col} <> {lit})"
        return f"{col} = {lit}"

    def to_dict(self) -> dict:
        return asdict(self)


class ValueIndex:
    def __init__(self, values: dict[str, list[str]], settings: Settings):
        self.values = values
        self.settings = settings
        self._exact: dict[str, list[tuple[str, str]]] = {}
        self._per_col_norm: dict[str, dict[str, str]] = {}
        self._tok: dict[str, list[tuple[str, str]]] = {}
        self._tri: dict[str, list[tuple[str, str]]] = {}
        self._abbr: dict[str, list[tuple[str, str]]] = {}
        for ref, vals in values.items():
            norm_map: dict[str, str] = {}
            for raw in vals:
                n = normalize(raw)
                if not n:
                    continue
                norm_map.setdefault(n, raw)
                self._exact.setdefault(loose_key(raw), []).append((ref, raw))
                for t in content_tokens(raw):
                    self._tok.setdefault(t, []).append((ref, raw))
                for g in _trigrams(n):
                    bucket = self._tri.setdefault(g, [])
                    if len(bucket) < 400:
                        bucket.append((ref, raw))
                ini = _initials(raw)
                if ini:
                    self._abbr.setdefault(ini, []).append((ref, raw))
            self._per_col_norm[ref] = norm_map

    # -- build / persist -----------------------------------------------------
    @classmethod
    def build(cls, registry, model: SemanticModel, settings: Settings) -> ValueIndex:
        values: dict[str, list[str]] = {}
        for ref, col in model.columns.items():
            if col.role != ROLE_DIMENSION:
                continue
            frame = registry.frames[col.table]
            distinct = frame[col.column].drop_nulls().cast(pl.Utf8).unique().to_list()
            if 0 < len(distinct) <= settings.max_values_indexed_per_column:
                values[ref] = sorted(str(v) for v in distinct if str(v).strip())
        return cls(values, settings)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"columns": self.values}, f)

    @classmethod
    def load(cls, path: str, settings: Settings) -> ValueIndex:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f)["columns"], settings)

    # -- lookups ---------------------------------------------------------------
    def has(self, ref: str) -> bool:
        return ref in self.values

    def contains(self, ref: str, value: str) -> bool:
        return value in set(self.values.get(ref, ()))

    def _score(self, query: str, candidate: str) -> tuple[float, str]:
        cn = normalize(candidate)
        best, method = similarity(query, cn), "fuzzy"
        q_toks, c_toks = content_tokens(query), content_tokens(cn)
        if q_toks and c_toks and len(query) >= 4 and set(q_toks) <= set(c_toks):
            coverage = len(q_toks) / len(c_toks)
            if coverage >= 0.5:
                contained = 0.86 + 0.14 * coverage
                if contained > best:
                    best, method = contained, "contains"
        ab = loose_key(query)
        if 2 <= len(ab) <= 5 and ab == _initials(candidate):
            if 0.9 > best:
                best, method = 0.9, "initials"
        return best, method

    def canonicalize(self, ref: str, value: str) -> tuple[str | None, str, float]:
        """Map a literal onto the stored value; strict (used for repairs, never for display)."""
        if ref not in self.values:
            return None, "unindexed", 0.0
        if self.contains(ref, value):
            return value, "exact", 1.0
        hit = self._per_col_norm[ref].get(normalize(value))
        if hit is not None:
            return hit, "case", 1.0
        q = normalize(value)
        scored = sorted(((self._score(q, c), c) for c in self.values[ref]), key=lambda x: -x[0][0])
        if scored and scored[0][0][0] >= self.settings.value_fuzzy_threshold:
            (s, m), c = scored[0]
            return c, m, s
        return None, "unknown", scored[0][0][0] if scored else 0.0

    def suggest(self, ref: str, value: str, n: int = 5) -> list[str]:
        q = normalize(value)
        scored = sorted(((self._score(q, c)[0], c) for c in self.values.get(ref, [])), reverse=True)
        return [c for _, c in scored[:n]]

    @staticmethod
    def _is_phrase(phrase: str) -> bool:
        parts = phrase.split()
        if not parts or parts[0] in STOPWORDS or parts[-1] in STOPWORDS:
            return False
        if parts[0].isdigit() or parts[-1].isdigit():
            return False  # '5 brands', 'top 3': a count, never part of a name
        return len(parts) > 1 or len(phrase) >= 3

    def link(self, question: str, limit: int = 8) -> list[ValueMatch]:
        """Phrases in the question that name real values. Longest phrases win."""
        toks = tokens(question)
        used: set[int] = set()
        out: list[ValueMatch] = []
        for start, n, phrase in ngrams(toks, 4):
            span = set(range(start, start + n))
            if span & used or not self._is_phrase(phrase):
                continue
            hits = self._exact.get(loose_key(phrase), [])
            if not hits:
                continue
            used |= span
            for ref, raw in hits[:6]:
                t, c = split_qualified(ref)
                out.append(ValueMatch(t, c, raw, phrase, 1.0 + 0.1 * n, "exact"))
        for start, n, phrase in ngrams(toks, 3):
            span = set(range(start, start + n))
            if span & used or not self._is_phrase(phrase) or phrase.isdigit():
                continue
            best = self._best_inexact(phrase)
            if best is None:
                continue
            used |= span
            score, ref, raw, method, alts = best
            t, c = split_qualified(ref)
            out.append(ValueMatch(t, c, raw, phrase, score, method, alts))
        out.sort(key=lambda m: -m.score)
        return out[: limit * 3]

    def _best_inexact(self, phrase: str):
        q = normalize(phrase)
        seen: set[tuple[str, str]] = set()
        scored = []
        pools = [self._tok.get(t, []) for t in content_tokens(q)] + [self._abbr.get(loose_key(q), [])]
        if sum(len(p) for p in pools) < 50:
            pools += [self._tri.get(g, []) for g in _trigrams(q)]
        for pool in pools:
            for ref, cand in pool:
                if (ref, cand) in seen:
                    continue
                seen.add((ref, cand))
                s, m = self._score(q, cand)
                if s >= self.settings.value_fuzzy_threshold:
                    scored.append((s, ref, cand, m))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[2]))
        top = scored[0]
        alts = sorted({c for s, r, c, _ in scored[1:] if normalize(c) != normalize(top[2]) and top[0] - s <= self.settings.value_ambiguity_margin})
        return top[0], top[1], top[2], top[3], alts[:5]
