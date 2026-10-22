# Architecture

## The idea in one paragraph

The model never writes the final SQL. Business meaning lives in a semantic layer that is built from the data, corrected by people, versioned, and loaded read-only by the service. Each question is resolved against that layer by deterministic code, turned into a small structured plan, and rendered into SQL by a composer that knows the rules (repeated bases, hierarchy filters, approved joins, per-file time windows). A model, if configured, can only propose a better plan inside the same contract or rewrite the prose of an answer, and both are checked.

## Build time: the semantic layer

`python -m sqlagent.preprocess --promote` runs these steps, all deterministic:

| Step | Module | What it decides |
|---|---|---|
| Load and profile | `registry.py` | Reads CSV/Excel with Polars. Text columns become numbers or dates only if every value parses. |
| Roles | `semantic/bootstrap.py` | measure, dimension, time or key from dtype and name tokens; ratio-like names are never summed |
| Vocabulary | `bootstrap.py` + `config/semantic_overrides.yaml` | Seed synonyms ("revenue" -> turnover). Generic words stay on the plainest column only. |
| Entities | `bootstrap.py` | The same concept across files (brand, category, region), grouped by name *and* measured value overlap |
| Joins | `bootstrap.py` | One candidate per entity pair, with measured alignment; below 35% it is written disabled with the reason |
| Repeated bases | `bootstrap.py` | A measure that one key fully determines while each key spans several rows is a copied total |
| Hierarchy columns | `bootstrap.py` | A column whose values name other columns mixes levels; the finest level becomes a default filter |
| Overrides | `config/semantic_overrides.yaml` | Human-approved metrics, labels, synonyms, default filters. People always win. |
| Value index | `semantic/value_index.py` | Every real value of every dimension, with exact, containment, initials and fuzzy matching |
| Time index | `semantic/time_layer.py` | Each file's own time axis and format, so relative phrases resolve per file |
| Quality gates | `quality.py` | Empty tables, lost columns (schema drift), null-heavy measures, row-count jumps |
| Artifacts | `artifacts.py` | Versioned folder keyed by data hash + model hash; `CURRENT` pointer for promotion |

Metric status matters: `approved` metrics are used as-is, `provisional` ones are used with a caveat on every answer, and `draft` ones are invisible to the agent. Synonyms suggested by a model are stored with source `enrichment` and any answer that relied on one says so.

## Question time: the LangGraph agent

```
understand -> plan -> compose -> execute -> synthesize
     |          |        |          |
  clarify    refused  cached/    diagnose (empty result)
                      refused
```

- **understand**: `Resolver` finds measures, metrics, dimensions, real values, exclusions and the time window per file; `QuestionParser` reads the shape (ranking, trend, pareto, growth, 2x2, compare, total), builds an `AnalysisPlan` and computes a confidence score from visible evidence (penalties for no measure, weak value match, unresolved period, ambiguity, unknown words, unverified synonyms). Follow-up questions start from the previous plan held in the checkpointer.
- **plan**: optional. A model may return a corrected plan; it must validate against the Pydantic contract and every reference must exist. One retry with the exact error, then the deterministic draft stands.
- **compose**: `Composer` renders SQL. `SQLValidator` re-checks it as a second line of defence. The answer cache is keyed by plan + artifact version, not by wording.
- **execute**: Polars `SQLContext`, streaming engine first, concurrency-capped.
- **diagnose**: an empty result is explained (which filter matched nothing) instead of being reported as a confident nothing.
- **synthesize**: post-processing (small-base growth rule, 2x2 placement), a deterministic answer, an optional model rewrite that passes through the citation check, an audit record, metrics and the memory update.

## Why this shape

- **Determinism where the risk is**: aggregation rules, join safety and time windows are encoded once and tested, instead of being re-derived by a model on every question.
- **Models where language is**: refining an ambiguous plan and writing readable prose.
- **Every guess is visible**: assumptions, cautions and notes travel with the answer and into the audit log.
- **Measured, not asserted**: the golden set compares numbers with an independent implementation; CI gates on it.
