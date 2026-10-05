import zlib

import numpy as np

import config

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
LEXICON = {
    "delta": ["sleep", "night", "rest", "deep", "slow", "dark", "body", "tired"],
    "theta": ["memory", "dream", "story", "past", "image", "wander", "brain", "feeling"],
    "alpha": ["calm", "light", "quiet", "open", "breath", "water", "space", "time"],
    "beta": ["focus", "task", "reason", "plan", "number", "signal", "question", "energy"],
    "gamma": ["insight", "pattern", "link", "sudden", "bright", "idea", "change", "learning"],
}
VOCAB = [w for ws in LEXICON.values() for w in ws]
NEIGHBOURS = [
    "The brain sorts the day's experiences while we sleep.",
    "She remembered the dream vividly the next morning.",
    "Quiet attention makes small patterns easier to notice.",
    "He solved the problem after a short walk outside.",
    "Memories become stable when they are replayed.",
    "The lake was calm and the light was fading.",
    "A sudden idea can feel like it comes from nowhere.",
]


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def unit_vec(rng, d):
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


class MockDecoder:

    mock = True
    dim = 256

    def band_power(self, epoch, fs):
        x = np.asarray(epoch, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] < 16:
            return {k: 1 / len(BANDS) for k in BANDS}
        x = x - x.mean(1, keepdims=True)
        spec = (np.abs(np.fft.rfft(x, axis=1)) ** 2).mean(0)
        f = np.fft.rfftfreq(x.shape[1], 1 / fs)
        bp = {k: float(spec[(f >= lo) & (f < hi)].mean() * (lo + hi) / 2) if ((f >= lo) & (f < hi)).any() else 0.0
              for k, (lo, hi) in BANDS.items()}
        tot = sum(bp.values()) or 1.0
        return {k: v / tot for k, v in bp.items()}

    def decode(self, epoch, fs, top_units=12):
        n = int(np.asarray(epoch).shape[-1]) if np.asarray(epoch).ndim == 2 else 0
        bp = self.band_power(epoch, fs)
        rng = np.random.default_rng(seed_of(*(round(v, 2) for v in bp.values()), n // max(1, int(fs))))
        score = {}
        for band, ws in LEXICON.items():
            for w in rng.choice(ws, 3, replace=False):
                score[str(w)] = max(score.get(str(w), 0.0), bp[band] * rng.uniform(0.6, 1.0))
        top = sorted(score.items(), key=lambda kv: -kv[1])[:min(top_units, 8)]
        hi = top[0][1] or 1.0
        units = [{"unit": VOCAB.index(w), "word": w, "words": [w], "weight": round(0.35 + 0.6 * s / hi, 4)}
                 for w, s in top]
        slots, vecs = [], []
        for i, u in enumerate(units):
            p = round(float(min(0.99, 0.5 + 0.5 * u["weight"] * rng.uniform(0.8, 1.0))), 4)
            band = next(b for b, ws in LEXICON.items() if u["word"] in ws)
            others = [w for w in LEXICON[band] if w != u["word"]]
            cands = [u["word"]] + [str(w) for w in rng.choice(others, 4, replace=False)]
            cos = np.sort(rng.uniform(0.35, 0.8, 5))[::-1]
            slots.append({"slot": i + 1, "p_active": p, "candidates": [
                {"unit": VOCAB.index(w), "words": [w], "cos": round(float(c), 4)} for w, c in zip(cands, cos)]})
            v = unit_vec(np.random.default_rng(seed_of(u["word"])), self.dim)
            vecs.append({"word": u["word"], "p_active": p, "vec": np.round(v, 4).tolist()})
        slots.sort(key=lambda d: -d["p_active"])
        nb = rng.choice(len(NEIGHBOURS), 5, replace=False)
        sims = np.sort(rng.uniform(0.55, 0.9, 5))[::-1]
        return {
            "units": units,
            "slots": slots,
            "neighbours": [{"sentence": NEIGHBOURS[j], "sim": round(float(s), 4)} for j, s in zip(nb, sims)],
            "z": np.round(unit_vec(rng, self.dim), 4).tolist(),
            "slot_vecs": vecs,
            "seconds": round(min(n / fs, config.EPOCH_SECONDS), 2) if fs else 0.0,
        }

    def alignment(self, units, sentence):
        from textutil import stem, stems

        s = stems(sentence)
        if not units or not s:
            return None
        score = 0.0
        for u in units:
            band = next((ws for ws in LEXICON.values() if u["word"] in ws), [])
            score += u["weight"] * (1.0 if stem(u["word"]) in s else 0.4 if any(stem(w) in s for w in band) else 0.0)
        return round(score / (sum(u["weight"] for u in units) or 1.0), 3)

    def info(self):
        return {"name": "mock-decoder (band power -> words)", "dim": self.dim, "mock": True}


TEMPLATES = [
    [("When it comes to {topic}, the key is how the brain organises {w} over time.", "information"),
     ("The short answer about {topic} is that the brain is quietly reworking {w}.", "experience")],
    [("Researchers think {w} plays a central role, because it links memory with emotion.", "attention"),
     ("One leading idea is that {w} acts as a bridge between what we notice and what we keep.", "context")],
    [("During this process, neurons replay recent patterns and strengthen the ones tied to {w}.", "learning"),
     ("Networks in the cortex rehearse the day, and traces linked to {w} tend to survive.", "practice")],
    [("This also explains why {w} can feel vivid even when nothing external is happening.", "imagery"),
     ("That is why {w} often shapes {topic} more than any single event does.", "mood")],
    [("In short, {topic} is the mind's way of turning {w} into something it can keep.", "noise"),
     ("So {topic} is less a mystery than a routine maintenance job on {w}.", "memory")],
]


class MockSteerer:

    mock = True
    layer = 14
    hidden = 2048

    def info(self):
        return {"name": "mock-qwen (no weights)", "layers": 28, "hidden": self.hidden, "steer_layer": self.layer,
                "device": "cpu", "translator": False, "mock": True}

    @staticmethod
    def token_id(word):
        return seed_of(word.lower()) % 151000

    def plan(self, decode, s):
        import torch

        units = decode["units"][: s["prefix_tokens"]]
        if not units:
            return None
        w = torch.tensor([u["weight"] for u in units])
        w = w / w.max()
        g = torch.Generator().manual_seed(seed_of(*(u["word"] for u in units)))
        plan = {"units": units, "weights": w.tolist(), "prefix": None, "labels": [], "bias": {}, "residual": None,
                "mode": "mock"}
        if s["prefix"] > 0:
            plan["prefix"] = torch.randn(len(units), self.hidden, generator=g) * w[:, None] * s["prefix"]
            plan["labels"] = [u["word"] for u in units]
        if s["residual"] > 0:
            plan["residual"] = torch.nn.functional.normalize(torch.randn(self.hidden, generator=g), dim=0) * 40 * s["residual"]
        if s["bias"] > 0:
            for u, wi in zip(units, w.tolist()):
                plan["bias"][self.token_id(u["word"])] = s["bias"] * wi
        return plan

    def readout(self, decode):
        ws = [u["word"] for u in decode.get("units", [])[:3]]
        return f"Something about {' and '.join(ws)}." if ws else None

    @staticmethod
    def topic(question):
        from textutil import content_words

        ws = content_words(question)
        return " ".join(ws[-2:]) if ws else "this question"

    def _sentence(self, question, history, plan, s):
        from textutil import stems

        i = min(len(history), len(TEMPLATES) - 1)
        rng = np.random.default_rng(seed_of(question, len(history), s["temperature"]))
        variants = TEMPLATES[i]
        text, default = variants[int(rng.integers(len(variants)))] if s["temperature"] > 0.3 else variants[0]
        steer, used = None, stems(" ".join(history))
        if plan is not None and (plan["prefix"] is not None or plan["bias"] or plan["residual"] is not None):
            fresh = [u["word"] for u in plan["units"] if u["word"] not in used]
            steer = (fresh or [u["word"] for u in plan["units"]])[0]
        return text.format(topic=self.topic(question), w=steer or default), steer

    def generate(self, question, history, plan, s):
        import time

        p_done = 0.0 if len(history) < 3 else round(0.35 + 0.15 * (len(history) - 3), 3)
        if p_done > 0.5:
            yield {"done": True, "sentence": "", "end": True, "p_complete": p_done}
            return
        sentence, steer = self._sentence(question, history, plan, s)
        bias = plan["bias"] if plan is not None else {}
        out = []
        for i, word in enumerate(sentence.split()[: s["max_tokens"]]):
            key = word.strip(".,;:!?'\"").lower()
            b = bias.get(self.token_id(key), 0.0) if key == (steer or "").lower() else 0.0
            text = (" " if i else "") + word
            out.append(text)
            time.sleep(config.MOCK_DELAY)
            yield {"text": text, "steered": b > 0, "bias": round(b, 3)}
        yield {"done": True, "sentence": "".join(out).strip(), "end": False, "p_complete": p_done}
