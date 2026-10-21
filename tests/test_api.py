from fastapi.testclient import TestClient


def test_api_end_to_end(env):
    from sqlagent.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/ready").status_code == 200
        r = client.post("/ask", json={"question": "Gross margin by brand in 2025", "thread_id": "api1"}).json()
        assert r["status"] == "ok" and r["rows"]
        stream = client.post("/ask/stream", json={"question": "Total turnover in 2025"}).text
        assert "event: stage" in stream and "event: answer" in stream
        assert "sqlagent_questions_total" in client.get("/metrics").text
        assert "finance_pnl" in client.get("/schema").json()
