import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent

SENTENCE_MODES = ["statement", "question", "negative", "imperative"]
SUBJECTIVITY = ["objective", "subjective"]
SEMANTIC_FOCUS = ["I/we", "you", "others", "thing", "event"]

EXTRA_STOP = {"s", "t", "d", "ll", "ve", "re", "m", "also", "just", "like", "lrb", "rrb"}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]|\d{3,4}")


def content_words(sentence):
    out = []
    for tok in TOKEN_RE.findall(sentence):
        w = tok.lower().strip("'-")
        w = re.sub(r"'s$", "", w)
        if len(w) < 2 or w in ENGLISH_STOP_WORDS or w in EXTRA_STOP:
            continue
        if w not in out:
            out.append(w)
    return out


NEGATION = re.compile(r"\b(not|no|never|nothing|none|nobody|neither|nor|cannot)\b|n't\b", re.I)
SUBJECTIVE_HINT = re.compile(
    r"\b(good|bad|best|worst|great|boring|funny|beautiful|awful|terrible|brilliant|"
    r"dull|enjoy\w*|love\w*|hate\w*|should|must|i think|wonderful|mess|fails?|"
    r"charming|entertaining|predictable|pleasure|disappoint\w*)\b", re.I)


def heuristic_attributes(sentence, task):
    s = sentence.strip()
    first = s.split()[0].lower().strip("\"'(") if s else ""
    if s.endswith("?"):
        mode = 1
    elif NEGATION.search(s):
        mode = 2
    else:
        mode = 0
    if task.endswith("SR"):
        subj = 1
    else:
        subj = 1 if SUBJECTIVE_HINT.search(s) else 0
    if first in {"i", "we", "my", "our", "us"}:
        focus = 0
    elif first in {"you", "your"}:
        focus = 1
    elif first in {"he", "she", "they", "his", "her", "their"} or (s[:1].isupper() and first not in {"the", "a", "an", "this", "it", "in", "on", "at", "after", "during"}):
        focus = 2
    elif re.search(r"\b(was|were)\s+\w+ed\b|\b(founded|born|died|won|married|served|graduated)\b", s):
        focus = 4
    else:
        focus = 3
    return {"sentence_mode": mode, "subjectivity": subj, "semantic_focus": focus}


class Embedder:

    def __init__(self, name, device, max_length=128):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        self.model = AutoModel.from_pretrained(name, dtype=torch.bfloat16 if device != "cpu" else torch.float32)
        self.model.to(device).eval()
        self.device, self.max_length = device, max_length

    @torch.no_grad()
    def __call__(self, texts, dim, batch_size=64):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt").to(self.device)
            h = self.model(**batch).last_hidden_state
            emb = h[:, -1]
            if not torch.isfinite(emb).all():
                raise RuntimeError("non-finite embeddings (fp overflow?) - try --device cpu")
            out.append(F.normalize(emb.float()[:, :dim], dim=-1).cpu())
            print(f"\r  embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
        print()
        return torch.cat(out)


class ArkEmbedder:

    def __init__(self, model, workers=8):
        import os
        self.key = os.environ.get("ARK_API_KEY")
        if not self.key:
            raise SystemExit("set ARK_API_KEY to use Doubao embeddings")
        self.base = os.environ.get("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        self.model, self.workers = model, workers
        self.multimodal = "vision" in model

    def _post(self, path, payload):
        import time
        import urllib.error
        import urllib.request
        body = json.dumps({"model": self.model, "encoding_format": "float", **payload}).encode()
        for attempt in range(8):
            req = urllib.request.Request(f"{self.base}{path}", data=body, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(min(2 ** attempt, 60))
                    continue
                raise SystemExit(f"Ark API error {e.code}: {msg}")
            except (urllib.error.URLError, TimeoutError):
                if attempt < 7:
                    time.sleep(min(2 ** attempt, 60))
                    continue
                raise

    def _embed_one(self, text):
        d = self._post("/embeddings/multimodal", {"input": [{"type": "text", "text": text}]})["data"]
        d = d[0] if isinstance(d, list) else d
        return torch.tensor(d["embedding"], dtype=torch.float32)

    def __call__(self, texts, dim, batch_size=64):
        from concurrent.futures import ThreadPoolExecutor
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if self.multimodal:
                with ThreadPoolExecutor(self.workers) as ex:
                    emb = torch.stack(list(ex.map(self._embed_one, chunk)))
            else:
                data = sorted(self._post("/embeddings", {"input": chunk})["data"], key=lambda d: d["index"])
                emb = torch.tensor([d["embedding"] for d in data], dtype=torch.float32)
            out.append(F.normalize(emb[:, :dim], dim=-1))
            print(f"\r  embedded {min(i + batch_size, len(texts))}/{len(texts)}", end="", flush=True)
        print()
        return torch.cat(out)


def wordnet_definitions(words):
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    for pkg in ("wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    lem, out = WordNetLemmatizer(), {}
    for w in words:
        if w.isdigit():
            out[w] = f"the number or year {w}"
            continue
        for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
            synsets = wn.synsets(lem.lemmatize(w, pos), pos)
            if synsets:
                out[w] = synsets[0].definition()
                break
    return out


def cluster_units(emb, threshold):
    if len(emb) == 1:
        return np.zeros(1, dtype=int)
    sim = (emb @ emb.T).clamp(-1, 1).numpy().astype(np.float64)
    dist = np.clip(1.0 - sim, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    return fcluster(Z, t=1.0 - threshold, criterion="distance") - 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["zuco1_SR", "zuco1_NR", "zuco1_TSR"])
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--name", default=None, help="asset name (default: tasks joined)")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                    help="HF model id (Qwen3-Embedding-0.6B/4B/8B), or 'ark:<model>' for Doubao via the Ark API, "
                         "e.g. ark:doubao-embedding-large-text-250515 (Volcengine) or ark:skylark-embedding-vision-250615 "
                         "(BytePlus name of Seed1.6-embedding, the space the paper cites)")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--cluster_threshold", type=float, default=0.85,
                    help="paper used 0.8 in Doubao space; 0.85 merges only variants/synonyms in Qwen3-0.6B space")
    ap.add_argument("--definitions", default="wordnet",
                    help="'wordnet' (offline glosses), 'none' (word-based units), or a JSON {word: definition} "
                         "file, e.g. from llm_annotate.py (paper App. E: definition-based is best)")
    ap.add_argument("--attributes", default=None, help="JSON {sentence: {sentence_mode, subjectivity, semantic_focus}}")
    ap.add_argument("--extra_vocab", default=None, help="text file, one word per line: open-vocabulary expansion (Table 4A)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    work = Path(args.work)
    tag = args.model.split("/")[-1].replace("ark:", "")
    name = args.name or "+".join(args.tasks) + ("" if tag == "Qwen3-Embedding-0.6B" else f"__{tag}")
    sent_task = {}
    for task in args.tasks:
        files = sorted((work / "eeg" / task).glob("*.pt"))
        if not files:
            raise SystemExit(f"no EEG files for {task}; run prepare_zuco.py first")
        for f in files:
            for s in torch.load(f, weights_only=False)["sentences"]:
                sent_task.setdefault(s, task)
    sentences = sorted(sent_task)
    units = {s: content_words(s) for s in sentences}
    vocab = sorted({w for ws in units.values() for w in ws})
    if args.extra_vocab:
        extra = [l.strip().lower() for l in open(args.extra_vocab) if l.strip()]
        vocab = sorted(set(vocab) | set(extra))
    n_units = np.array([len(units[s]) for s in sentences])
    print(f"{len(sentences)} unique sentences, {len(vocab)} unit words, "
          f"units/sentence mean={n_units.mean():.1f} p95={np.percentile(n_units, 95):.0f} max={n_units.max()}")

    if args.definitions == "wordnet":
        defs = wordnet_definitions(vocab)
    elif args.definitions == "none":
        defs = {}
    else:
        defs = wordnet_definitions(vocab)
        defs.update(json.load(open(args.definitions)))
    unit_texts = [f"{w}: {defs[w]}" if w in defs else w for w in vocab]
    print(f"definition-based units: {sum(w in defs for w in vocab)}/{len(vocab)}")

    embedder = ArkEmbedder(args.model[4:]) if args.model.startswith("ark:") else Embedder(args.model, args.device)
    print("embedding units ...")
    word_emb = embedder(unit_texts, args.dim)
    print("embedding sentences ...")
    sent_emb = embedder(sentences, args.dim, batch_size=16)

    labels = cluster_units(word_emb, args.cluster_threshold)
    n_clusters = labels.max() + 1
    bank = torch.zeros(n_clusters, args.dim)
    bank.index_add_(0, torch.as_tensor(labels), word_emb)
    bank = F.normalize(bank, dim=-1)
    members = [[] for _ in range(n_clusters)]
    for w, c in zip(vocab, labels):
        members[c].append(w)
    word2cid = {w: int(c) for w, c in zip(vocab, labels)}
    print(f"unit bank: {len(vocab)} words -> {n_clusters} clusters (threshold {args.cluster_threshold})")
    merged = [m for m in members if len(m) > 1]
    print("  example merges:", merged[:8])

    g = torch.Generator().manual_seed(0)
    i, j = torch.randint(n_clusters, (2, 20000), generator=g)
    keep = i != j
    mus_exp = float((bank[i[keep]] * bank[j[keep]]).sum(-1).mean())
    print(f"MUS_exp (random unit pair similarity) = {mus_exp:.4f}")

    attr_file = json.load(open(args.attributes)) if args.attributes else {}
    attrs, n_from_file = [], 0
    for s in sentences:
        a = heuristic_attributes(s, sent_task[s])
        if s in attr_file:
            a.update({k: int(v) for k, v in attr_file[s].items() if k in a})
            n_from_file += 1
        attrs.append(a)
    print(f"attributes: {n_from_file} from file, {len(sentences) - n_from_file} heuristic")
    for k, names in [("sentence_mode", SENTENCE_MODES), ("subjectivity", SUBJECTIVITY), ("semantic_focus", SEMANTIC_FOCUS)]:
        cnt = np.bincount([a[k] for a in attrs], minlength=len(names))
        print(f"  {k}: " + ", ".join(f"{n}={c}" for n, c in zip(names, cnt)))

    out_dir = work / "text" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": args.model, "dim": args.dim, "cluster_threshold": args.cluster_threshold,
        "sentences": sentences, "sentence_embeddings": sent_emb,
        "sentence_units": [units[s] for s in sentences],
        "sentence_unit_cids": [sorted({word2cid[w] for w in units[s]}) for s in sentences],
        "attributes": attrs,
        "bank_embeddings": bank, "bank_members": members, "word2cid": word2cid,
        "vocab": vocab, "unit_texts": unit_texts, "word_embeddings": word_emb, "mus_exp": mus_exp,
        "attribute_names": {"sentence_mode": SENTENCE_MODES, "subjectivity": SUBJECTIVITY, "semantic_focus": SEMANTIC_FOCUS},
    }, out_dir / "text_assets.pt")
    with open(out_dir / "units.json", "w") as f:
        json.dump({s: units[s] for s in sentences}, f, indent=1, ensure_ascii=False)
    print(f"[OK] {out_dir / 'text_assets.pt'}")


if __name__ == "__main__":
    main()
