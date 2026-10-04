import argparse
import json
import logging
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context

import config
from eeg_source import Acquisition, electrode_positions

log = logging.getLogger("neurosteer")
app = Flask(__name__)
acq = Acquisition()
LABELS, POS = electrode_positions()
DISPLAY = [int(i) for i in np.argsort(np.arctan2(POS[:, 1], POS[:, 0]))[:: max(1, len(POS) // 24)][:24]]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
DEFAULTS = {"prefix": 0.8, "prefix_tokens": 6, "bias": 1.0, "residual": 0.3, "hint": False,
            "temperature": 0.7, "max_tokens": 70, "max_sentences": 8}

models = {"decoder": None, "llm": None, "errors": {}, "loading": True, "mock": config.MOCK}
session = {"question": "", "history": [], "settings": dict(DEFAULTS), "read": None, "last_decode": None, "ended": False}
slock = threading.Lock()


def load_models(mock=None):
    mock = config.MOCK if mock is None else mock
    specs = ((("decoder", "mock_models", "MockDecoder"), ("llm", "mock_models", "MockSteerer")) if mock else
             (("decoder", "decoder", "SemanticDecoder"), ("llm", "steering", "Steerer")))
    models.update({"decoder": None, "llm": None, "errors": {}, "loading": True, "mock": mock})
    for key, mod, cls in specs:
        t = time.time()
        try:
            models[key] = getattr(__import__(mod), cls)()
            log.info("%s ready in %.1f s", cls, time.time() - t)
        except Exception as e:
            models["errors"][key] = repr(e)
            log.exception("%s failed to load", cls)
    models["loading"] = False


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


def sse(gen):
    return Response(stream_with_context(gen), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/status")
def status():
    llm = models["llm"]
    return jsonify({
        "mock": models["mock"],
        "eeg": acq.status(),
        "decoder": models["decoder"] is not None,
        "llm": llm.info() if llm else None,
        "loading": models["loading"],
        "errors": models["errors"],
        "settings": session["settings"],
        "question": session["question"],
        "count": len(session["history"]),
    })


def display():
    src = acq.source
    if src is not None and src.display:
        return [i for i, _ in src.display], [l for _, l in src.display]
    return DISPLAY, [LABELS[i] for i in DISPLAY]


@app.get("/api/montage")
def montage():
    idx, names = display()
    sensors = acq.source.info.get("sensors", []) if acq.source is not None else []
    return jsonify({"labels": LABELS, "pos": POS.round(4).tolist(), "display": idx,
                    "display_labels": names, "sensors": sensors})


@app.post("/api/source")
def set_source():
    kind = (request.get_json(silent=True) or {}).get("kind", "synthetic")
    try:
        acq.start(kind)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(acq.status())


@app.get("/api/eeg")
def eeg_stream():
    def gen():
        last = acq.now()
        tick = 0
        while True:
            buf, src = acq.buffer, acq.source
            if buf is None:
                time.sleep(0.2)
                continue
            now = buf.total
            if now < last:
                last = now
            x = buf.read(last, now)
            last = now
            dec = max(1, src.fs // 125)
            frame = {"t": now, "fs": src.fs / dec,
                     "samples": np.round(x[display()[0], ::dec], 3).tolist(),
                     "reading": session["read"] is not None}
            yield event("eeg", frame)
            tick += 1
            if tick % 6 == 0:
                w = buf.latest(src.fs * 2)
                if w.shape[1] > 32:
                    w = w - w.mean(1, keepdims=True)
                    rms = np.sqrt((w[:, -src.fs:] ** 2).mean(1))
                    spec = np.abs(np.fft.rfft(w * np.hanning(w.shape[1]), axis=1)) ** 2
                    f = np.fft.rfftfreq(w.shape[1], 1 / src.fs)
                    bp = {k: float(spec[:, (f >= lo) & (f < hi)].mean()) for k, (lo, hi) in BANDS.items()}
                    tot = sum(bp.values()) or 1.0
                    yield event("metrics", {"rms": np.round(rms, 3).tolist(),
                                            "bands": {k: round(v / tot, 4) for k, v in bp.items()}})
            time.sleep(0.04)

    return sse(gen())


@app.post("/api/session")
def new_session():
    body = request.get_json(silent=True) or {}
    with slock:
        q = (body.get("question") or body.get("topic") or "").strip() or "Why do we dream?"
        session.update({"question": q, "history": [], "read": None, "last_decode": None, "ended": False})
        session["settings"].update({k: v for k, v in (body.get("settings") or {}).items() if k in DEFAULTS})
    return jsonify({"question": session["question"], "settings": session["settings"]})


@app.post("/api/settings")
def settings():
    body = request.get_json(silent=True) or {}
    session["settings"].update({k: type(DEFAULTS[k])(v) for k, v in body.items() if k in DEFAULTS})
    return jsonify(session["settings"])


@app.post("/api/read/start")
def read_start():
    if acq.buffer is None:
        return jsonify({"error": "no EEG source"}), 400
    session["read"] = {"start": acq.now(), "t0": time.time()}
    acq.source.reading = True
    return jsonify({"start": session["read"]["start"]})


@app.post("/api/read/end")
def read_end():
    r = session["read"]
    if r is None:
        return jsonify({"error": "not reading"}), 400
    session["read"] = None
    acq.source.reading = False
    end = acq.now()
    epoch = acq.buffer.read(r["start"], end)
    if models["decoder"] is None:
        return jsonify({"error": "decoder not loaded"}), 503
    t = time.time()
    d = models["decoder"].decode(epoch, acq.source.fs)
    d.update({"start": r["start"], "end": end, "latency_ms": round((time.time() - t) * 1000),
              "brain_derived": acq.source.brain_derived, "source": acq.source.label})
    session["last_decode"] = d
    log.info("decode %.1f s epoch -> %d units in %d ms: %s", d["seconds"], len(d["units"]), d["latency_ms"],
             " ".join(u["word"] for u in d["units"][:6]))
    public = {k: v for k, v in d.items() if k != "slot_vecs"}
    if session["history"]:
        session["history"][-1]["decode"] = public
    return jsonify(public)


@app.post("/api/readout")
def readout():
    d, llm = session["last_decode"], models["llm"]
    if d is None or llm is None:
        return jsonify({"text": None})
    t = time.time()
    return jsonify({"text": llm.readout(d), "ms": round((time.time() - t) * 1000)})


@app.get("/api/generate")
def generate():
    llm = models["llm"]
    if llm is None:
        return jsonify({"error": "LLM not loaded"}), 503
    s = dict(session["settings"])
    if request.args.get("steer") == "0":
        s.update({"prefix": 0, "bias": 0, "residual": 0, "hint": False})

    def gen():
        d = session["last_decode"]
        t0 = time.time()
        plan = llm.plan(d, s) if d else None
        summary = None
        if plan is not None:
            pre = plan["prefix"]
            strips = None
            if pre is not None:
                p = pre.float().cpu()
                p = p.reshape(p.shape[0], 32, -1).mean(-1)
                strips = (p / (p.abs().max() + 1e-6)).round(decimals=3).tolist()
            summary = {"units": [u["word"] for u in plan["units"]], "weights": [round(w, 3) for w in plan["weights"]],
                       "labels": plan["labels"], "mode": plan["mode"],
                       "prefix": strips, "bias_tokens": len(plan["bias"]),
                       "residual_norm": round(float(plan["residual"].norm()), 2) if plan["residual"] is not None else 0,
                       "layer": llm.layer}
        yield event("plan", summary)
        tokens = []
        history = [h["sentence"] for h in session["history"]]
        for item in llm.generate(session["question"], history, plan, s):
            if item.get("done"):
                end = item["end"] or len(session["history"]) + 1 >= s["max_sentences"]
                rec = {"id": len(session["history"]), "sentence": item["sentence"], "tokens": tokens,
                       "plan": summary, "decode": None, "ms": round((time.time() - t0) * 1000), "end": end,
                       "p_complete": item.get("p_complete", 0.0)}
                if item["end"]:
                    rec["tokens"], rec["sentence"] = [], ""
                else:
                    session["history"].append(rec)
                session["ended"] = end
                log.info("sentence %d in %d ms (%d tokens, %d steered)%s", rec["id"], rec["ms"], len(tokens),
                         sum(t["steered"] for t in tokens), " [end]" if end else "")
                yield event("done", rec)
            else:
                tokens.append(item)
                yield event("token", item)

    return sse(gen())


@app.get("/api/history")
def history():
    return jsonify({"question": session["question"], "history": session["history"]})


def main():
    ap = argparse.ArgumentParser(description="Neurosteer backend")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--mock", action="store_true", default=config.MOCK, help="fake decoder and LLM, no weights")
    args = ap.parse_args()
    config.MOCK = args.mock
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    acq.start("replay" if config.REPLAY_DIR.exists() else "synthetic")
    threading.Thread(target=load_models, daemon=True).start()
    log.info("Neurosteer %s on http://%s:%d%s", config.VERSION, args.host, args.port, " (mock)" if args.mock else "")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
