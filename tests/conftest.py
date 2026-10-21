import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    data = base / "data"
    from sqlagent import synthetic

    synthetic.main(["--out", str(data), "--seed", "7"])
    os.environ.update({
        "DATA_DIR": str(data), "ARTIFACT_DIR": str(base / "artifacts"), "STATE_DIR": str(base / "state"),
        "SEMANTIC_OVERRIDES": os.path.join(ROOT, "config", "semantic_overrides.yaml"),
        "AUTO_BUILD": "1", "LLM_PROVIDER": "none",
    })
    return base


@pytest.fixture(scope="session")
def engine(env):
    from sqlagent.config import get_settings
    from sqlagent.engine import Engine

    return Engine(get_settings())


@pytest.fixture(scope="session")
def agent(engine):
    from langgraph.checkpoint.memory import InMemorySaver

    from sqlagent.config import get_settings
    from sqlagent.graph import Agent

    return Agent(engine, get_settings(), checkpointer=InMemorySaver())
