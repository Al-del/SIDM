import copy
import json

from conftest import read, sse_events


def test_health(client):
    j = client.get("/api/health").get_json()
    assert j["ok"] is True and j["version"] == "0.2.0" and j["uptime_s"] >= 0


def test_status_reports_mock(client):
    j = client.get("/api/status").get_json()
    assert j["mock"] is True and j["decoder"] is True and j["llm"]["mock"] is True
    assert j["eeg"]["kind"] == "synthetic" and not j["loading"] and j["count"] == 0


def test_errors_are_json(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and "error" in r.get_json()
    r = client.get("/api/settings")
    assert r.status_code == 405 and "error" in r.get_json()
    r = client.options("/api/settings")
    assert r.status_code == 204 and r.headers["Access-Control-Allow-Origin"] == "*"
    assert client.post("/api/source", json={"kind": "bogus"}).status_code == 400
    assert client.post("/api/read/end").status_code == 400
    assert client.get("/api/generate").status_code == 400


def test_full_loop(client):
    j = client.post("/api/session", json={"question": "Why do we dream?"}).get_json()
    assert j["question"] == "Why do we dream?"
    ev = sse_events(client.get("/api/generate"))
    names = [e for e, _ in ev]
    assert names[0] == "plan" and names[-1] == "done" and "token" in names
    done = ev[-1][1]
    assert done["id"] == 0 and done["sentence"] and done["decode"] is None and not done["end"]
    assert done["sentence"] == "".join(t["text"] for t in done["tokens"]).strip()

    d = read(client)
    assert d["units"] and d["slots"] and d["neighbours"] and "slot_vecs" not in d
    assert len(d["z"]) == 256 and 0 <= d["alignment"] <= 1 and d["seconds"] > 0
    hist = client.get("/api/history").get_json()["history"]
    assert hist[0]["decode"]["alignment"] == d["alignment"]

    ev = sse_events(client.get("/api/generate"))
    plan = ev[0][1]
    assert plan["units"] and plan["mode"] == "mock" and plan["layer"] == 14 and plan["bias_tokens"] > 0
    assert any(t["steered"] for t in ev[-1][1]["tokens"])

    for _ in range(10):
        read(client)
        done = sse_events(client.get("/api/generate"))[-1][1]
        if done["end"]:
            break
    assert done["end"] and 4 <= len(client.get("/api/history").get_json()["history"]) <= 5

    s = client.get("/api/stats").get_json()
    assert s["sentences"] >= 4 and 0 <= s["mean_alignment"] <= 1 and s["mean_latency_ms"] >= 0
    assert 0 < s["steered_tokens"] < s["total_tokens"]


def test_unsteered_generation_has_no_plan_effect(client):
    client.post("/api/session", json={"question": "How do birds navigate?"})
    client.get("/api/generate").get_data()
    read(client)
    ev = sse_events(client.get("/api/generate?steer=0"))
    assert ev[0][1]["prefix"] is None and ev[0][1]["bias_tokens"] == 0
    assert not any(t["steered"] for t in ev[-1][1]["tokens"])


def test_compare_does_not_mutate_history(client, server):
    client.post("/api/session", json={"question": "Why is the sky blue?"})
    client.get("/api/generate").get_data()
    read(client)
    before = copy.deepcopy(server.session["history"])
    ev = sse_events(client.get("/api/compare"))
    assert server.session["history"] == before
    lanes = [d["lane"] for e, d in ev if e == "done"]
    assert lanes == ["steered", "baseline"]
    assert {d["lane"] for e, d in ev if e == "token"} == {"steered", "baseline"}
    assert all({"lane", "text", "steered", "bias"} <= set(d) for e, d in ev if e == "token")
    assert all({"lane", "sentence", "ms"} <= set(d) for e, d in ev if e == "done")
    e, summary = ev[-1]
    assert e == "summary" and 0 <= summary["overlap"] <= 1 and isinstance(summary["steered_words"], list)
    assert summary["steered_words"]


def test_generation_conflict(client, server):
    client.post("/api/session", json={"question": "Why do we dream?"})
    assert server.glock.acquire(blocking=False)
    try:
        assert client.get("/api/generate").status_code == 409
        assert client.get("/api/compare").status_code == 409
    finally:
        server.glock.release()
    r = client.get("/api/generate")
    assert r.status_code == 200 and sse_events(r)[-1][0] == "done"


def test_settings_are_clamped(client):
    j = client.post("/api/settings", json={"residual": 9, "prefix": -1, "prefix_tokens": "4.6", "bias": "abc",
                                           "hint": "false", "temperature": 0, "max_tokens": None,
                                           "max_sentences": float("nan"), "unknown": 1}).get_json()
    assert j["residual"] == 2.0 and j["prefix"] == 0.0 and j["prefix_tokens"] == 5 and j["bias"] == 1.0
    assert j["hint"] is False and j["temperature"] == 0.05 and j["max_tokens"] == 70 and j["max_sentences"] == 8
    assert "unknown" not in j
    assert client.post("/api/settings", data="not json").status_code == 200
    assert client.post("/api/settings", json=[1, 2]).status_code == 200
    j = client.post("/api/session", json={"question": 42, "settings": {"bias": 99}}).get_json()
    assert j["question"] == "42" and j["settings"]["bias"] == 5.0


def test_presets(client):
    p = client.get("/api/presets").get_json()
    assert set(p) == {"subtle", "balanced", "strong", "off"}
    assert all(v["residual"] <= 0.8 for v in p.values())
    assert p["off"]["prefix"] == p["off"]["bias"] == p["off"]["residual"] == 0
    j = client.post("/api/settings", json={"preset": "strong", "bias": 1.5}).get_json()
    assert j["residual"] == p["strong"]["residual"] and j["bias"] == 1.5
    r = client.post("/api/settings", json={"preset": "turbo"})
    assert r.status_code == 400 and "error" in r.get_json()


def test_export(client):
    client.post("/api/session", json={"question": "Why do we dream?"})
    client.get("/api/generate").get_data()
    read(client)
    r = client.get("/api/export")
    assert r.status_code == 200 and r.mimetype == "application/json"
    cd = r.headers["Content-Disposition"]
    assert cd.startswith("attachment;") and "neurosteer-session-" in cd and cd.endswith('.json"')
    j = json.loads(r.get_data())
    assert j["question"] == "Why do we dream?" and j["settings"] and j["model"]["mock"] is True
    assert j["source"]["kind"] == "synthetic" and j["exported_at"] and j["started_at"]
    h = j["history"][0]
    assert h["tokens"] and h["sentence"] and h["at"]
    assert "z" not in h["decode"] and "slot_vecs" not in h["decode"] and "alignment" in h["decode"]


def test_reset_keeps_settings(client):
    client.post("/api/session", json={"question": "Why do we dream?", "settings": {"bias": 2.5}})
    client.get("/api/generate").get_data()
    client.post("/api/read/start")
    assert client.post("/api/reset").get_json() == {"ok": True}
    s = client.get("/api/status").get_json()
    assert s["count"] == 0 and s["question"] == "" and s["settings"]["bias"] == 2.5
    assert client.post("/api/read/end").status_code == 400
    assert client.get("/api/stats").get_json()["sentences"] == 0


def test_stream_pings_and_cleans_up(server):
    import threading
    import time

    closed = threading.Event()

    def slow():
        try:
            yield server.event("a", 1)
            time.sleep(0.25)
            yield server.event("b", 2)
            yield server.event("c", 3)
        finally:
            closed.set()

    with server.app.test_request_context():
        body = "".join(server.streamed(slow(), every=0.05).response)
    assert ": ping" in body and body.index("event: a") < body.index(": ping") < body.index("event: b")
    assert closed.wait(1)

    closed.clear()
    with server.app.test_request_context():
        it = iter(server.streamed(slow(), every=0.05).response)
        assert next(it).startswith("event: a")
        it.close()
    assert closed.wait(1)


def test_short_epoch_is_rejected(client):
    client.post("/api/read/start")
    r = client.post("/api/read/end")
    assert r.status_code == 400 and "too short" in r.get_json()["error"]
