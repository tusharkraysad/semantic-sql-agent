"""
Hybrid schema retriever
=======================
Chooses which columns a model gets to see when it is asked to refine a plan.
Four signals are fused: a local character n-gram vector similarity (a free
stand-in for an embedding model), BM25 word matching, a boost for columns the
resolver already matched, and a boost for columns holding a value named in
the question. Two guarantees sit on top: resolver columns are always kept and
at least one measure is always present.
"""

from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np

from .config import Settings
from .semantic.model import SemanticModel
from .semantic.resolver import SemanticContext
from .text_utils import content_tokens, normalize


def _ngram_vec(text: str, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    t = f"  {normalize(text)}  "
    for i in range(len(t) - 2):
        v[zlib.crc32(t[i : i + 3].encode()) % dim] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [content_tokens(d) for d in docs]
        self.avg = sum(len(t) for t in self.toks) / max(len(self.toks), 1)
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(w for t in self.toks for w in set(t))
        n = max(len(docs), 1)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = content_tokens(query)
        out = []
        for tf, toks in zip(self.tf, self.toks):
            s, length = 0.0, len(toks) or 1
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out


def _minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    return [0.0] * len(xs) if hi - lo < 1e-12 else [(x - lo) / (hi - lo) for x in xs]


class HybridRetriever:
    def __init__(self, model: SemanticModel, settings: Settings):
        self.model = model
        self.s = settings
        self.refs = [r for r, c in model.columns.items() if c.role != "ignore"]
        self.docs = []
        for r in self.refs:
            c = model.columns[r]
            self.docs.append(f"{c.table} {c.column} {c.label} {c.role} {c.description} {' '.join(c.synonyms)}")
        self.vecs = np.stack([_ngram_vec(d) for d in self.docs]) if self.docs else np.zeros((0, 512))
        self.bm25 = BM25(self.docs)

    def retrieve(self, question: str, ctx: SemanticContext | None = None, top_k: int | None = None) -> list[str]:
        if not self.refs:
            return []
        top_k = top_k or self.s.top_k_columns
        dense = _minmax(list(self.vecs @ _ngram_vec(question)))
        lex = _minmax(self.bm25.scores(question))
        must = set()
        values = set()
        prior = {}
        if ctx is not None:
            must = {c.ref for c in ctx.measures} | {c.ref for c in ctx.dimensions}
            values = {m.ref for m in ctx.value_matches}
            top = max(ctx.table_scores.values(), default=0) or 1.0
            prior = {t: s / top for t, s in ctx.table_scores.items()}
        scored = []
        for i, ref in enumerate(self.refs):
            c = self.model.columns[ref]
            s = self.s.dense_weight * dense[i] + self.s.lexical_weight * lex[i]
            s += self.s.semantic_boost if ref in must else 0.0
            s += self.s.value_boost if ref in values else 0.0
            s += self.s.table_prior_weight * prior.get(c.table, 0.0)
            s -= 0.3 if c.role == "key" else 0.0
            scored.append((s, ref))
        scored.sort(reverse=True)
        chosen = [r for r in must | values if r in self.model.columns]
        for _, ref in scored:
            if len(chosen) >= max(top_k, len(chosen)):
                break
            if ref not in chosen:
                chosen.append(ref)
        if not any(self.model.columns[r].is_measure for r in chosen):
            chosen.append(next(r for _, r in scored if self.model.columns[r].is_measure))
        return chosen

    def schema_text(self, refs: list[str]) -> str:
        lines = []
        for r in refs:
            c = self.model.columns[r]
            extra = []
            if c.is_measure:
                extra.append(f"aggregate={c.aggregation}")
            if c.is_repeated_base:
                extra.append(f"REPEATED per {c.grain} - never SUM directly")
            if not c.additive:
                extra.append("ratio - never SUM")
            if c.entity:
                extra.append(f"entity={c.entity}")
            lines.append(f"- {r} [{c.role}{', ' + ', '.join(extra) if extra else ''}]")
        return "\n".join(lines)
