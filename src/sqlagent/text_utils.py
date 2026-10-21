"""
Text helpers
============
Every string the system matches on goes through these functions, so 'Dove',
'dove' and ' DOVE ' always produce the same key wherever they came from.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz as _rf
except ImportError:  # pragma: no cover - optional speed-up
    _rf = None

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_SPACES = re.compile(r"\s+")

# Never a business term or a data value on their own.
STOPWORDS = frozenset(
    """
    a an the and or of for to in on by with at from as is are was were be been what which who how
    do does did show me give tell list please can could would should i we you it its this that these
    those there their our my all any each per over under between across during within into out up down
    top bottom best worst highest lowest most least more less than then
    total sum average avg count number amount value values data table column report
    vs versus compare comparison trend trends growth split breakdown share pct percent percentage
    month months year years quarter quarters latest last previous current
    """.split()
)

# Words that name the SHAPE of an answer. Kept apart from STOPWORDS because
# 'matrix' is also a real brand in the data.
ANALYSIS_WORDS = frozenset("matrix quadrant pareto cumulative grew grown growing declined decline increase decrease yoy growth trend 2x2 bcg fastest".split())

NEGATION_CUES = ("excluding", "exclude", "except", "other than", "apart from", "without", "not", "besides")


def normalize(text) -> str:
    return _SPACES.sub(" ", _NON_ALNUM.sub(" ", str(text).lower())).strip()


def loose_key(text) -> str:
    return normalize(text).replace(" ", "")


def tokens(text) -> list[str]:
    return [t for t in normalize(text).split() if t]


def content_tokens(text) -> list[str]:
    return [t for t in tokens(text) if len(t) > 1 and t not in STOPWORDS]


_IRREGULAR = {"skus": "sku", "categories": "category", "companies": "company"}


def singularize(text: str) -> str:
    out = []
    for tok in normalize(text).split():
        if tok in _IRREGULAR:
            out.append(_IRREGULAR[tok])
        elif tok.endswith("ies") and len(tok) > 4:
            out.append(tok[:-3] + "y")
        elif tok.endswith("ses") and len(tok) > 4:
            out.append(tok[:-2])
        elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
            out.append(tok[:-1])
        else:
            out.append(tok)
    return " ".join(out)


def similarity(a: str, b: str) -> float:
    if _rf is not None:
        return _rf.ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def ngrams(token_list: list[str], max_n: int = 4):
    """(start, length, phrase) for every n-gram up to max_n, longest first."""
    n_tok = len(token_list)
    for n in range(min(max_n, n_tok), 0, -1):
        for start in range(0, n_tok - n + 1):
            yield start, n, " ".join(token_list[start : start + n])


def humanize(name: str) -> str:
    words = normalize(name).split()
    return " ".join(w.upper() if len(w) <= 3 and name.isupper() else w.capitalize() for w in words) or name


def detect_case(values: list[str]) -> str:
    sample = [v for v in values[:300] if isinstance(v, str) and any(c.isalpha() for c in v)]
    if not sample:
        return "unknown"
    if all(v == v.lower() for v in sample):
        return "lower"
    if all(v == v.upper() for v in sample):
        return "upper"
    return "mixed"


def _cue_ends(toks: list[str]) -> list[int]:
    cues = sorted((tokens(c) for c in NEGATION_CUES), key=len, reverse=True)
    ends, i = [], 0
    while i < len(toks):
        for cue in cues:
            if toks[i : i + len(cue)] == cue:
                ends.append(i + len(cue))
                i += len(cue)
                break
        else:
            i += 1
    return ends


def negated_terms(text: str, terms, window: int = 4) -> set[str]:
    """
    Which candidate terms a negation cue governs: the first candidate starting
    within `window` tokens to the right of the cue, and nothing further. So
    'sales excluding January' negates January, while 'excluding returns, sales
    in January' does not.
    """
    hay = tokens(text)
    positions: list[tuple[int, str]] = []
    for term in terms:
        needle = tokens(term)
        if not needle:
            continue
        for s in range(len(hay) - len(needle) + 1):
            if hay[s : s + len(needle)] == needle:
                positions.append((s, term))
                break
    positions.sort()
    out: set[str] = set()
    for end in _cue_ends(hay):
        for start, term in positions:
            if end <= start < end + window:
                out.add(term)
                break
    return out


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"
