def test_simple_ranking(agent):
    r = agent.ask("Top 5 brands by revenue in 2025", "t1")
    assert r["status"] == "ok" and len(r["rows"]) == 5
    assert r["measures"] == ["Total Turnover"]
    assert r["mode"] == "simple"


def test_follow_up_keeps_context(agent):
    first = agent.ask("Turnover for Spark in 2025", "t2")
    second = agent.ask("Now for Glint", "t2")
    assert second["followup"] is True
    assert second["rows"][0]["Total Turnover"] != first["rows"][0]["Total Turnover"]


def test_cache_is_keyed_by_plan_not_wording(agent):
    agent.ask("Top 5 brands by revenue in 2025", "t3")
    again = agent.ask("top 5 brands by turnover in 2025", "t4")
    assert again["cached"] is True


def test_clarify_and_refuse(agent):
    assert agent.ask("What is the weather in Bengaluru", "t5")["status"] == "clarify"
    r = agent.ask("Turnover by big c and marketing spend", "t6")
    assert r["status"] == "refused" and "no approved join" in r["answer"].lower()


def test_small_base_is_not_ranked_first(agent):
    r = agent.ask("Which brands grew fastest in 2025 vs 2024", "t7")
    assert r["rows"][0]["Main Brand"] != "Nova"
    assert any("not comparable" in n for n in r["notes"])


def test_audit_record_written(agent, env):
    import json
    import os

    agent.ask("Total turnover in 2025", "t8")
    path = os.path.join(str(env / "state"), "decisions.jsonl")
    last = json.loads(open(path, encoding="utf-8").read().strip().splitlines()[-1])
    assert last["question"] == "Total turnover in 2025" and last["sql"]


def test_citation_check_tags_invented_numbers():
    import polars as pl

    from sqlagent.citations import check

    frame = pl.DataFrame({"Brand": ["A"], "Turnover": [314.13]})
    text, bad = check("A leads with 314.1, up from 999.9.", frame, "q")
    assert bad == ["999.9"] and "[unverified]" in text


def test_missing_csvs_are_regenerated(tmp_path, monkeypatch):
    """A checkout has data_sources.yaml but no CSVs (they are gitignored): the demo must still start."""
    import shutil

    from sqlagent.config import get_settings
    from sqlagent.engine import Engine
    from sqlagent.registry import data_present

    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(__import__("os").path.join(__import__("os").environ["DATA_DIR"], "data_sources.yaml"), data)
    assert not data_present(str(data))
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("GENERATE_DEMO_DATA", "1")
    engine = Engine(get_settings())
    assert data_present(str(data)) and engine.loaded.version
