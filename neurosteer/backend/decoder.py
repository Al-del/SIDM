import logging
import os
import sys
import time
from math import gcd

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

import config

os.environ.setdefault("BM_ROOT", str(config.ROSEF))
sys.path.insert(0, str(config.MOSAIC_CODE))
from rag_mosaic import CFG, RAGMosaic, zscore

log = logging.getLogger("neurosteer")


def preprocess(x, fs):
    if fs != config.FS:
        g = gcd(int(fs), config.FS)
        x = resample_poly(x, config.FS // g, int(fs) // g, axis=1)
    T = config.FS * config.EPOCH_SECONDS
    x = x[:, :T]
    n = x.shape[1]
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    x = np.clip(x, -10, 10)
    out = np.zeros((x.shape[0], T), dtype=np.float32)
    out[:, :n] = x
    return torch.from_numpy(out)[None], torch.tensor([n])


def standardize(x, dim=-1):
    return (x - x.mean(dim, keepdim=True)) / (x.std(dim, keepdim=True) + 1e-6)


class TopicMatcher:

    TASK = "Given a question, retrieve words that belong to its topic"

    def __init__(self, assets, n_units):
        from transformers import AutoModel, AutoTokenizer

        self.dev = torch.device(config.TOPIC_DEVICE)
        self.tok = AutoTokenizer.from_pretrained(config.TOPIC_MODEL, padding_side="left")
        self.model = AutoModel.from_pretrained(config.TOPIC_MODEL, dtype=torch.float32).to(self.dev).eval()
        self._last = (None, None)
        cache = config.TOPIC_CACHE
        if cache.exists():
            ck = torch.load(cache, weights_only=False)
            if ck["model"] == config.TOPIC_MODEL and ck["units"].shape[0] == n_units:
                self.units = ck["units"]
                return
        t = time.time()
        words = self._embed(list(assets["unit_texts"]))
        idx = {w: i for i, w in enumerate(assets["vocab"])}
        units = torch.zeros(n_units, words.shape[1])
        for w, c in assets["word2cid"].items():
            units[c] += words[idx[w]]
        self.units = F.normalize(units, dim=-1)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": config.TOPIC_MODEL, "units": self.units}, cache)
        log.info("embedded %d bank units for topic matching in %.1f s", n_units, time.time() - t)

    @torch.no_grad()
    def _embed(self, texts, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=64,
                         return_tensors="pt").to(self.dev)
            out.append(F.normalize(self.model(**b).last_hidden_state[:, -1].float(), dim=-1).cpu())
        return torch.cat(out)

    def scores(self, question):
        if self._last[0] != question:
            q = self._embed([f"Instruct: {self.TASK}\nQuery:{question}"])[0]
            self._last = (question, standardize(self.units @ q))
        return self._last[1]


class SemanticDecoder:
    def __init__(self):
        dev = torch.device(config.DECODER_DEVICE)
        a = torch.load(config.TEXT_ASSETS, weights_only=False)
        ck = torch.load(config.MOSAIC_WEIGHTS, weights_only=False)
        self.k = ck["args"]["k"]
        self.max_units = ck["args"]["max_units"]
        self.bank = F.normalize(a["bank_embeddings"].float(), dim=-1).to(dev)
        self.members = a["bank_members"]
        self.sentences = a["sentences"]
        self.semb = F.normalize(a["sentence_embeddings"].float(), dim=-1).to(dev)
        self.sent_units = a["sentence_unit_cids"]
        self.model = RAGMosaic(CFG, self.bank.shape[1], self.k).to(dev).eval()
        self.model.load_state_dict(ck["model"])
        self.dev = dev
        sims = self.bank @ self.bank.T
        hub = sims.topk(50, dim=1).values[:, 1:].mean(1)
        self.hub_mask = hub > torch.quantile(hub, 0.97)
        self.word2cid = a.get("word2cid", {})
        self.mu = float(sims.mean())
        self.topic = None
        try:
            self.topic = TopicMatcher(a, self.bank.shape[0])
        except Exception:
            log.exception("topic matcher unavailable; decoding without topic re-ranking")

    @torch.no_grad()
    def topic_scores(self, question):
        if self.topic is None or not question:
            return None
        return self.topic.scores(question).to(self.dev)

    def _ctx_empty(self):
        d = self.bank.shape[1]
        return {"ctx_emb": torch.zeros(1, 1, d, device=self.dev), "ctx_sim": torch.zeros(1, 1, device=self.dev),
                "ctx_rank": torch.zeros(1, 1, dtype=torch.long, device=self.dev),
                "ctx_pad": torch.zeros(1, 1, dtype=torch.bool, device=self.dev)}

    def _ctx(self, nbs, sims):
        U, d = self.max_units, self.bank.shape[1]
        emb = torch.zeros(1, self.k * U, d, device=self.dev)
        sim = torch.zeros(1, self.k * U, device=self.dev)
        rank = torch.zeros(1, self.k * U, dtype=torch.long, device=self.dev)
        pad = torch.ones(1, self.k * U, dtype=torch.bool, device=self.dev)
        for r, (j, s) in enumerate(zip(nbs, sims)):
            c = self.sent_units[j][:U]
            o = r * U
            emb[0, o:o + len(c)] = self.bank[torch.as_tensor(c, device=self.dev)]
            sim[0, o:o + len(c)] = float(s)
            rank[0, o:o + len(c)] = r
            pad[0, o:o + len(c)] = False
        return {"ctx_emb": emb, "ctx_sim": sim, "ctx_rank": rank, "ctx_pad": pad}

    def info(self):
        return {"name": config.MOSAIC_WEIGHTS.name, "units": int(self.bank.shape[0]), "dim": int(self.bank.shape[1]),
                "neighbours": self.k, "device": str(self.dev)}

    def _cid(self, word):
        from textutil import stem

        s = stem(word)
        for w in (word, s, s + "s", word + "s", s + "e"):
            if w in self.word2cid:
                return self.word2cid[w]
        return None

    @torch.no_grad()
    def alignment(self, units, sentence):
        from textutil import content_words

        cids = {c for c in map(self._cid, content_words(sentence)) if c is not None}
        if not units or not cids:
            return None
        u = self.bank[torch.tensor([x["unit"] for x in units], device=self.dev)]
        best = (u @ self.bank[torch.tensor(sorted(cids), device=self.dev)].T).max(1).values
        best = ((best - self.mu) / (1 - self.mu)).clamp(0, 1)
        w = torch.tensor([x["weight"] for x in units], device=self.dev)
        return round(float((best * w).sum() / w.sum().clamp_min(1e-6)), 3)

    @torch.no_grad()
    def decode(self, epoch, fs, top_units=12, question=None, topic_weight=0.0):
        x, L = preprocess(epoch, fs)
        x, L = x.to(self.dev), L.to(self.dev)
        z = F.normalize(self.model({"eeg": x, **self._ctx_empty()}, L)["sentence_embedding"], dim=-1)
        score = zscore(z @ self.semb.T)[0]
        top = torch.topk(score, self.k)
        nbs, sims = top.indices.tolist(), torch.sigmoid(top.values).tolist()
        out = self.model({"eeg": x, **self._ctx(nbs, sims)}, L)
        p_active = out["pred_logits"].softmax(-1)[0, :, 1]
        pe = F.normalize(out["pred_embeddings"][0], dim=-1)
        cos = pe @ self.bank.T
        cos[:, self.hub_mask] = -1
        topic = self.topic_scores(question) if topic_weight > 0 else None
        rank = cos if topic is None else standardize(cos) + topic_weight * topic[None]
        if topic is not None:
            rank[:, self.hub_mask] = -float("inf")
        slots = []
        unit_score = {}
        for s in range(1, pe.shape[0]):
            if p_active[s] < 0.5:
                continue
            ti = rank[s].topk(5).indices
            cands = [{"unit": int(c), "words": self.members[int(c)], "cos": round(float(cos[s, c]), 4)}
                     for c in ti]
            if topic is not None:
                for d, c in zip(cands, ti):
                    d["topic"] = round(float(topic[c]), 3)
            slots.append({"slot": s, "p_active": round(float(p_active[s]), 4), "candidates": cands})
            for r, c in enumerate(ti):
                w = float(p_active[s]) * max(float(cos[s, c]), 0) * (1 - 0.1 * r)
                unit_score[int(c)] = max(unit_score.get(int(c), 0), w)
        slots.sort(key=lambda d: -d["p_active"])
        vecs, taken = [], set()
        for s in slots[:12]:
            v, best = pe[s["slot"]], s["candidates"][0]
            if topic is not None:
                best = next((c for c in s["candidates"] if c["unit"] not in taken), best)
                taken.add(best["unit"])
                v = F.normalize(v + self.bank[best["unit"]], dim=-1)
            vecs.append({"word": best["words"][0], "p_active": s["p_active"],
                         "vec": [round(x, 4) for x in v.tolist()]})
        ranked = sorted(unit_score.items(), key=lambda kv: -kv[1])[:top_units]
        units = [{"unit": c, "word": self.members[c][0], "words": self.members[c], "weight": round(w, 4)}
                 for c, w in ranked]
        return {
            "units": units,
            "slots": slots,
            "neighbours": [{"sentence": self.sentences[j], "sim": round(s, 4)} for j, s in zip(nbs, sims)],
            "z": [round(float(v), 4) for v in z[0].tolist()],
            "slot_vecs": vecs,
            "seconds": round(int(L) / config.FS, 2),
            "topic": None if topic is None else {
                "weight": topic_weight,
                "nearest": [self.members[int(c)][0] for c in topic.masked_fill(self.hub_mask, -1e9).topk(8).indices]},
        }
