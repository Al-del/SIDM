import os
import sys
import time
from pathlib import Path

import pytest

os.environ["NEUROSTEER_MOCK"] = "1"
os.environ["NEUROSTEER_MOCK_DELAY"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as backend


@pytest.fixture(scope="session")
def server():
    backend.load_models()
    backend.acq.start("synthetic")
    time.sleep(0.2)
    yield backend
    backend.acq.stop()


@pytest.fixture
def client(server):
    with server.slock:
        server.clear_session()
        server.session["settings"] = dict(server.DEFAULTS)
    return server.app.test_client()


def sse_events(resp):
    import json

    out = []
    for block in resp.get_data(as_text=True).split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith(":")]
        name = next((l[7:] for l in lines if l.startswith("event: ")), None)
        data = next((l[6:] for l in lines if l.startswith("data: ")), None)
        if name:
            out.append((name, json.loads(data)))
    return out


def read(client, seconds=0.3):
    assert client.post("/api/read/start").status_code == 200
    time.sleep(seconds)
    r = client.post("/api/read/end")
    assert r.status_code == 200, r.get_json()
    return r.get_json()
