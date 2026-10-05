import argparse
import json
import logging
import queue
import threading
import time

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException

import config
from eeg_source import SOURCES, Acquisition, electrode_positions

log = logging.getLogger("neurosteer")
app = Flask(__name__)
acq = Acquisition()
LABELS, POS = electrode_positions()
DISPLAY = [int(i) for i in np.argsort(np.arctan2(POS[:, 1], POS[:, 0]))[:: max(1, len(POS) // 24)][:24]]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
DEFAULTS = {"prefix": 0.8, "prefix_tokens": 6, "bias": 1.0, "residual": 0.3, "hint": False,
            "temperature": 0.7, "max_tokens": 70, "max_sentences": 8}
LIMITS = {"prefix": (0.0, 3.0), "prefix_tokens": (1, 12), "bias": (0.0, 5.0), "residual": (0.0, 2.0),
          "temperature": (0.05, 2.0), "max_tokens": (8, 200), "max_sentences": (1, 20)}
MIN_EPOCH_S = 0.1
PING_S = 15.0
PRESETS = {
    "subtle": {"prefix": 0.5, "prefix_tokens": 4, "bias": 0.5, "residual": 0.15, "hint": False},
    "balanced": {"prefix": 0.8, "prefix_tokens": 6, "bias": 1.0, "residual": 0.3, "hint": False},
    "strong": {"prefix": 1.2, "prefix_tokens": 8, "bias": 2.0, "residual": 0.7, "hint": False},
    "off": {"prefix": 0.0, "bias": 0.0, "residual": 0.0, "hint": False},
}

models = {"decoder": None, "llm": None, "errors": {}, "loading": True, "mock": config.MOCK}
session = {"question": "", "history": [], "settings": dict(DEFAULTS), "read": None, "last_decode": None, "ended": False,
           "sid": 0}
slock = threading.Lock()
glock = threading.Lock()
dlock = threading.Lock()
STARTED = time.time()


def jbody():
    b = request.get_json(silent=True)
    return b if isinstance(b, dict) else {}


def clean_settings(body):
    out = {}
    for k, v in (body or {}).items() if isinstance(body, dict) else ():
        if k not in DEFAULTS:
            continue
        try:
            if isinstance(DEFAULTS[k], bool):
                v = v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
            else:
                v = type(DEFAULTS[k])(round(float(v)) if isinstance(DEFAULTS[k], int) else float(v))
                if v != v or v in (float("inf"), float("-inf")):
                    raise ValueError(v)
                lo, hi = LIMITS[k]
                v = min(max(v, lo), hi)
        except (TypeError, ValueError, OverflowError):
            log.warning("ignoring setting %s=%r", k, v)
            continue
        out[k] = v
    return out


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


@app.before_request
def preflight():
    if request.method == "OPTIONS":
        return "", 204


@app.errorhandler(HTTPException)
def http_error(e):
    return jsonify({"error": e.description, "status": e.code}), e.code


@app.errorhandler(Exception)
def server_error(e):
    log.exception("unhandled error on %s %s", request.method, request.path)
    return jsonify({"error": f"{type(e).__name__}: {e}", "status": 500}), 500


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


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "uptime_s": round(time.time() - STARTED, 1), "version": config.VERSION})


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


def display(src=None):
    src = src or acq.source
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
    kind = jbody().get("kind", "synthetic")
    if not isinstance(kind, str) or kind not in SOURCES:
        return jsonify({"error": f"unknown source {kind!r}", "sources": list(SOURCES)}), 400
    try:
        acq.start(kind)
    except Exception as e:
        log.warning("source %s failed: %s", kind, e)
        return jsonify({"error": str(e) or repr(e)}), 400
    return jsonify(acq.status())


@app.get("/api/eeg")
def eeg_stream():
    def gen():
        try:
            yield from frames()
        except GeneratorExit:
            log.info("EEG stream closed by client")

    def frames():
        last = acq.now()
        tick, pinged = 0, time.time()
        while True:
            buf, src = acq.buffer, acq.source
            if buf is None or src is None:
                if time.time() - pinged > PING_S:
                    pinged = time.time()
                    yield ": ping\n\n"
                time.sleep(0.2)
                continue
            now = buf.total
            if now < last:
                last = now
            x = buf.read(last, now)
            last = now
            dec = max(1, src.fs // 125)
            frame = {"t": now, "fs": src.fs / dec,
                     "samples": np.round(x[display(src)[0], ::dec], 3).tolist(),
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
    body = jbody()
    with slock:
        q = str(body.get("question") or body.get("topic") or "").strip()[:500] or "Why do we dream?"
        session.update({"question": q, "history": [], "read": None, "last_decode": None, "ended": False,
                        "sid": session["sid"] + 1})
        session["settings"].update(clean_settings(body.get("settings")))
    return jsonify({"question": session["question"], "settings": session["settings"]})


@app.post("/api/settings")
def settings():
    body = jbody()
    preset = body.pop("preset", None)
    if preset is not None and (not isinstance(preset, str) or preset not in PRESETS):
        return jsonify({"error": f"unknown preset {preset!r}", "presets": list(PRESETS)}), 400
    with slock:
        session["settings"].update({**(PRESETS[preset] if preset else {}), **clean_settings(body)})
    return jsonify(session["settings"])


@app.get("/api/presets")
def presets():
    return jsonify(PRESETS)


@app.post("/api/read/start")
def read_start():
    buf, src = acq.buffer, acq.source
    if buf is None or src is None:
        return jsonify({"error": "no EEG source"}), 400
    session["read"] = {"start": buf.total, "t0": time.time(), "buf": buf, "src": src,
                       "id": len(session["history"]) - 1}
    src.reading = True
    return jsonify({"start": session["read"]["start"]})


@app.post("/api/read/end")
def read_end():
    if not dlock.acquire(blocking=False):
        return jsonify({"error": "a decode is already running"}), 409
    try:
        return decode_read()
    finally:
        dlock.release()


def decode_read():
    r = session["read"]
    if r is None:
        return jsonify({"error": "not reading"}), 400
    session["read"] = None
    buf, src = r["buf"], r["src"]
    src.reading = False
    end = buf.total
    epoch = buf.read(r["start"], end)
    if models["decoder"] is None:
        return jsonify({"error": "decoder not loaded"}), 503
    if epoch.shape[1] < src.fs * MIN_EPOCH_S:
        return jsonify({"error": f"epoch too short ({epoch.shape[1] / src.fs:.2f} s)"}), 400
    t = time.time()
    d = models["decoder"].decode(epoch, src.fs)
    d.update({"start": r["start"], "end": end, "latency_ms": round((time.time() - t) * 1000),
              "brain_derived": src.brain_derived, "source": src.label})
    session["last_decode"] = d
    log.info("decode %.1f s epoch -> %d units in %d ms: %s", d["seconds"], len(d["units"]), d["latency_ms"],
             " ".join(u["word"] for u in d["units"][:6]))
    public = {k: v for k, v in d.items() if k != "slot_vecs"}
    if 0 <= r["id"] < len(session["history"]):
        session["history"][r["id"]]["decode"] = public
    return jsonify(public)


@app.post("/api/readout")
def readout():
    d, llm = session["last_decode"], models["llm"]
    if d is None or llm is None:
        return jsonify({"text": None})
    t = time.time()
    return jsonify({"text": llm.readout(d), "ms": round((time.time() - t) * 1000)})


def plan_summary(plan, llm):
    if plan is None:
        return None
    strips = None
    if plan["prefix"] is not None:
        p = plan["prefix"].float().cpu()
        p = p.reshape(p.shape[0], 32, -1).mean(-1)
        strips = [[round(v, 3) for v in row] for row in (p / (p.abs().max() + 1e-6)).tolist()]
    return {"units": [u["word"] for u in plan["units"]], "weights": [round(w, 3) for w in plan["weights"]],
            "labels": plan["labels"], "mode": plan["mode"], "prefix": strips, "bias_tokens": len(plan["bias"]),
            "residual_norm": round(float(plan["residual"].norm()), 2) if plan["residual"] is not None else 0,
            "layer": llm.layer}


def streamed(gen, every=PING_S):
    q, stop, done = queue.Queue(), threading.Event(), object()

    def pump():
        try:
            for item in gen:
                q.put(item)
                if stop.is_set():
                    break
        except Exception as e:
            log.exception("stream failed")
            q.put(event("error", {"error": f"{type(e).__name__}: {e}"}))
        finally:
            gen.close()
            q.put(done)

    threading.Thread(target=pump, daemon=True).start()

    def out():
        try:
            while True:
                try:
                    item = q.get(timeout=every)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                if item is done:
                    return
                yield item
        finally:
            stop.set()

    return sse(out())


def exclusive(gen):
    if not glock.acquire(blocking=False):
        return jsonify({"error": "a generation is already running"}), 409

    def run():
        try:
            yield from gen
        finally:
            glock.release()

    return streamed(run())


@app.get("/api/generate")
def generate():
    llm = models["llm"]
    if llm is None:
        return jsonify({"error": "LLM not loaded"}), 503
    if not session["question"]:
        return jsonify({"error": "no session: POST /api/session first"}), 400
    s = dict(session["settings"])
    if request.args.get("steer") == "0":
        s.update({"prefix": 0, "bias": 0, "residual": 0, "hint": False})
    sid, question, d = session["sid"], session["question"], session["last_decode"]
    history = [h["sentence"] for h in session["history"]]

    def gen():
        t0 = time.time()
        plan = llm.plan(d, s) if d else None
        summary = plan_summary(plan, llm)
        yield event("plan", summary)
        tokens = []
        for item in llm.generate(question, history, plan, s):
            if not item.get("done"):
                tokens.append(item)
                yield event("token", item)
                continue
            end = item["end"] or len(history) + 1 >= s["max_sentences"]
            rec = {"id": len(history), "sentence": item["sentence"], "tokens": tokens, "plan": summary,
                   "decode": None, "ms": round((time.time() - t0) * 1000), "end": end,
                   "p_complete": item.get("p_complete", 0.0), "at": time.time()}
            if item["end"]:
                rec["tokens"], rec["sentence"] = [], ""
            with slock:
                if session["sid"] == sid:
                    if not item["end"]:
                        session["history"].append(rec)
                    session["ended"] = end
            log.info("sentence %d in %d ms (%d tokens, %d steered)%s", rec["id"], rec["ms"], len(tokens),
                     sum(t["steered"] for t in tokens), " [end]" if end else "")
            yield event("done", rec)

    return exclusive(gen())


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
