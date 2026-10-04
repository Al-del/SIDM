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

    def info(self):
        return {"name": "mock-decoder (band power -> words)", "dim": self.dim, "mock": True}
