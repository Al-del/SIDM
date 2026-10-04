import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import config

OUT = Path(__file__).resolve().parent / "weights" / "translator.pt"
PLACEHOLDER = "<<NEURAL>>"
INSTRUCTION = f"Neural context: {PLACEHOLDER}\nRestate it in plain words."


class Translator(nn.Module):
    def __init__(self, d_in, d_llm, n_sent=8, hidden=1024, emb_norm=1.0):
        super().__init__()
        self.n_sent, self.d_llm = n_sent, d_llm
        self.sent = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                  nn.Linear(hidden, n_sent * d_llm))
        self.word = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, d_llm))
        self.sent_pos = nn.Parameter(torch.zeros(n_sent, d_llm))
        self.register_buffer("emb_norm", torch.tensor(float(emb_norm)))

    def _scale(self, x):
        return F.normalize(x, dim=-1) * self.emb_norm

    def sentence(self, z):
        return self._scale(self.sent(z).view(-1, self.n_sent, self.d_llm) + self.sent_pos)

    def words(self, u):
        return self._scale(self.word(u))


def perturb(x):
    c = torch.empty(len(x), 1, device=x.device).uniform_(0.6, 1.0)
    n = torch.randn_like(x)
    n = F.normalize(n - (n * x).sum(-1, keepdim=True) * x, dim=-1)
    return F.normalize(c * x + (1 - c ** 2).sqrt() * n, dim=-1)


class Frame:

    def __init__(self, tok, emb, dev):
        msgs = [{"role": "user", "content": INSTRUCTION}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        a, b = text.split(PLACEHOLDER)
        ids = lambda s: tok(s, add_special_tokens=False, return_tensors="pt").input_ids[0].to(dev)
        self.A, self.B = emb(ids(a)), emb(ids(b))
        self.tok, self.emb, self.dev = tok, emb, dev
        self.end = tok.convert_tokens_to_ids("<|im_end|>")

    def batch(self, prefix, texts, T):
        B, P, _ = prefix.shape
        tgt = [self.tok(t, add_special_tokens=False).input_ids[:T - 1] + [self.end] for t in texts]
        ids = torch.full((B, T), self.end, dtype=torch.long, device=self.dev)
        lab = torch.full((B, T), -100, dtype=torch.long, device=self.dev)
        for i, t in enumerate(tgt):
            ids[i, :len(t)] = torch.tensor(t, device=self.dev)
            lab[i, :len(t)] = ids[i, :len(t)]
        head = torch.cat([self.A.expand(B, -1, -1), prefix.to(self.A.dtype), self.B.expand(B, -1, -1)], 1)
        x = torch.cat([head, self.emb(ids)], 1)
        labels = torch.cat([torch.full((B, head.shape[1]), -100, device=self.dev, dtype=torch.long), lab], 1)
        mask = torch.ones(x.shape[:2], dtype=torch.long, device=self.dev)
        mask[:, head.shape[1]:] = (lab != -100).long()
        return x, mask, labels


def nll(model, x, mask, labels):
    h = model.model(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, :-1]
    sel = labels[:, 1:] != -100
    return F.cross_entropy(model.lm_head(h[sel]).float(), labels[:, 1:][sel])


@torch.no_grad()
def reconstruct(model, frame, prefix, max_new=40):
    out = []
    for p in prefix:
        x = torch.cat([frame.A, p.to(frame.A.dtype), frame.B])[None]
        r = model(inputs_embeds=x, use_cache=True)
        past, ids = r.past_key_values, []
        for _ in range(max_new):
            nxt = int(r.logits[0, -1].argmax())
            if nxt == frame.end:
                break
            ids.append(nxt)
            r = model(input_ids=torch.tensor([[nxt]], device=frame.dev), past_key_values=past, use_cache=True)
            past = r.past_key_values
        out.append(frame.tok.decode(ids, skip_special_tokens=True).strip())
    return out


def save(T, args, S, emb, emb_norm):
    OUT.parent.mkdir(exist_ok=True)
    torch.save({"model": T.state_dict(), "n_sent": args.n_sent, "d_in": S.shape[1],
                "d_llm": emb.weight.shape[1], "emb_norm": emb_norm, "llm": config.QWEN_MODEL}, OUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--bs_sent", type=int, default=8)
    ap.add_argument("--bs_word", type=int, default=16)
    ap.add_argument("--len_sent", type=int, default=48)
    ap.add_argument("--len_word", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_sent", type=int, default=8)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    random.seed(0)
    dev = torch.device(config.LLM_DEVICE)
    tok = AutoTokenizer.from_pretrained(config.QWEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)
    emb = model.get_input_embeddings()
    emb_norm = float(emb.weight[:20000].float().norm(dim=-1).mean())
    frame = Frame(tok, emb, dev)

    a = torch.load(config.TEXT_ASSETS, weights_only=False)
    S = F.normalize(a["sentence_embeddings"].float(), dim=-1).to(dev)
    W = F.normalize(a["word_embeddings"].float(), dim=-1).to(dev)
    sents, vocab = a["sentences"], a["vocab"]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(sents))
    n_val = len(sents) // 10
    val, tr = perm[:n_val], perm[n_val:]
    words_ok = [i for i, w in enumerate(vocab) if not w.isdigit()]

    T = Translator(S.shape[1], emb.weight.shape[1], args.n_sent, emb_norm=emb_norm).to(dev)
    if args.eval_only:
        T.load_state_dict(torch.load(OUT, weights_only=False)["model"])
    else:
        opt = torch.optim.AdamW(T.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        t0 = time.time()
        for step in range(args.steps):
            T.train()
            si = rng.choice(tr, args.bs_sent, replace=False)
            wi = rng.choice(words_ok, args.bs_word, replace=False)
            opt.zero_grad(set_to_none=True)
            x, m, l = frame.batch(T.sentence(perturb(S[si])), [sents[i] for i in si], args.len_sent)
            ls = nll(model, x, m, l)
            ls.backward()
            x, m, l = frame.batch(T.words(perturb(W[wi]))[:, None], [vocab[i] for i in wi], args.len_word)
            lw = nll(model, x, m, l)
            (0.5 * lw).backward()
            del x, m, l
            torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0)
            opt.step()
            sched.step()
            torch.mps.empty_cache()
            if step % 100 == 99:
                save(T, args, S, emb, emb_norm)
            if step % 25 == 0 or step == args.steps - 1:
                print(f"step {step:4d} | sentence nll {ls.item():.3f} | word nll {lw.item():.3f} | "
                      f"{time.time() - t0:.0f}s", flush=True)
        save(T, args, S, emb, emb_norm)
        print(f"saved -> {OUT}")

    T.eval()
    with torch.no_grad():
        res = {}
        for name, vec in [("true", S[val]), ("noisy(cos~0.8)", perturb(S[val])), ("shuffled", S[val][torch.randperm(len(val))])]:
            tot = 0.0
            for s in range(0, len(val), 16):
                idx = val[s:s + 16]
                x, m, l = frame.batch(T.sentence(vec[s:s + 16]), [sents[i] for i in idx], args.len_sent)
                tot += nll(model, x, m, l).item() * len(idx)
                torch.mps.empty_cache()
            res[name] = tot / len(val)
        print("held-out sentence NLL/token:", {k: round(v, 3) for k, v in res.items()})
        recon = reconstruct(model, frame, T.sentence(S[val[:6]]))
        for i, r in zip(val[:6], recon):
            print(f"  GOLD  {sents[i]}\n  READ  {r}\n")


if __name__ == "__main__":
    main()
