import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from model import EncoderLayer, ModernTCNEncoder, sinusoidal

ROOT = Path(os.environ.get("BM_ROOT", Path(__file__).resolve().parent.parent))
TEXT = ROOT / "brainmosaic/work/text/zuco1_SR+zuco1_NR+zuco1_TSR__Qwen3-Embedding-8B/text_assets.pt"


def load_trials(task, assets, scramble):
    s2i = {s: i for i, s in enumerate(assets["sentences"])}
    eeg, lengths, sent, subj, names = [], [], [], [], []
    for sid, f in enumerate(sorted((ROOT / "brainmosaic/work/eeg" / task).glob("*.pt"))):
        d = torch.load(f, weights_only=False)
        keep = [i for i, s in enumerate(d["sentences"]) if s in s2i and assets["sentence_unit_cids"][s2i[s]]]
        e, L = d["eeg"][keep], d["lengths"][keep]
        if scramble:
            p = torch.from_numpy(np.random.RandomState(1234 + sid).permutation(len(keep)))
            e, L = e[p], L[p]
        eeg.append(e)
        lengths.append(L)
        sent += [s2i[d["sentences"][i]] for i in keep]
        subj += [sid] * len(keep)
        names.append(f.stem)
    return torch.cat(eeg), torch.cat(lengths), np.array(sent), np.array(subj), names


class EEGEncoder(nn.Module):
    def __init__(self, text_dim, d=256, drop=0.2):
        super().__init__()
        self.tcn = ModernTCNEncoder(105, d=64, patch=25, blocks=2, large=25, small=5, dropout=0.1)
        self.proj = nn.Linear(64, d)
        self.layers = nn.ModuleList([EncoderLayer(d, 8, 1024, drop) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.text_head = nn.Linear(d, text_dim)

    def forward(self, x, lengths):
        h = self.proj(self.tcn(x))
        B, N, d = h.shape
        n_valid = torch.clamp((lengths + 24) // 25, 1, N)
        pad = torch.arange(N, device=x.device)[None] >= n_valid[:, None]
        pos = sinusoidal(N, d, x.device)[None]
        for layer in self.layers:
            h = layer(h, pos, pad)
        h = (h * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True)
        z = self.head(h)
        return F.normalize(z, dim=-1), F.normalize(self.text_head(z), dim=-1)


def supcon(z, labels, tau):
    sim = z @ z.T / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()


def vote_units(sim, mem_sent, assets, n_units, k, temp, K):
    out = []
    top = torch.topk(sim, min(k, sim.shape[1]), dim=1)
    for q in range(sim.shape[0]):
        w = torch.softmax(top.values[q] / temp, 0)
        score = torch.zeros(n_units)
        for wj, mj in zip(w.tolist(), top.indices[q].tolist()):
            cids = assets["sentence_unit_cids"][mem_sent[mj]]
            score[cids] += wj
        best = torch.topk(score, K)
        out.append((best.indices, best.values))
    return out


def uma_from_units(preds, gold_sents, assets, bank, taus, rng, n_random=5):
    res = {f"UMA@{t}": [] for t in taus}
    res.update({f"UMA@{t}_othersent": [] for t in taus})
    res.update({f"UMA@{t}_random": [] for t in taus})
    res["MUS"] = []
    all_sents = np.array(gold_sents)

    def match(pred_c, gold_c):
        P, G = bank[pred_c], bank[torch.as_tensor(gold_c)]
        s = (G @ P.T).numpy()
        r, c = linear_sum_assignment(-s)
        sims = np.zeros(len(gold_c))
        sims[r] = s[r, c]
        return sims

    for (pc, _), gs in zip(preds, gold_sents):
        gold = assets["sentence_unit_cids"][gs]
        s = match(pc, gold)
        res["MUS"] += list(s)
        o = assets["sentence_unit_cids"][int(rng.choice(all_sents[all_sents != gs]))]
        so = match(pc, o)
        sr = np.concatenate([match(pc, rng.choice(len(bank), size=len(gold), replace=False)) for _ in range(n_random)])
        for t in taus:
            res[f"UMA@{t}"] += list(s > t)
            res[f"UMA@{t}_othersent"] += list(so > t)
            res[f"UMA@{t}_random"] += list(sr > t)
    return {k: float(np.mean(v)) for k, v in res.items()}


def sentence_id(sim, mem_sent, q_sent):
    sents = np.unique(mem_sent)
    col = {s: i for i, s in enumerate(sents)}
    S = torch.full((sim.shape[0], len(sents)), -1e9)
    idx = torch.as_tensor([col[s] for s in mem_sent])
    S = S.scatter_reduce(1, idx[None].expand(sim.shape[0], -1), sim, reduce="amax", include_self=True)
    p1, p5, rk = [], [], []
    for q, s in enumerate(q_sent):
        if s not in col:
            continue
        t = S[q, col[s]]
        g = int((S[q] > t).sum())
        e = int((S[q] == t).sum()) - 1
        p1.append(min(max((1 - g) / (e + 1), 0), 1))
        p5.append(min(max((5 - g) / (e + 1), 0), 1))
        rk.append(g + e / 2)
    return {"sent_top1": float(np.mean(p1)), "sent_top5": float(np.mean(p5)),
            "sent_rank_pct": float(np.mean(rk) / len(sents)), "n_candidates": len(sents)}


@torch.no_grad()
def embed(model, eeg, lengths, idx, dev, bs=64):
    model.eval()
    zs = []
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        z, _ = model(eeg[b].to(dev).float(), lengths[b].to(dev))
        zs.append(z.cpu())
    return torch.cat(zs)


def evaluate_all(model, eeg, lengths, sent, mem_idx, q_idx, assets, bank, args, dev, with_baselines=False):
    zq = embed(model, eeg, lengths, q_idx, dev)
    zm = embed(model, eeg, lengths, mem_idx, dev)
    sim = zq @ zm.T
    mem_sent, q_sent = sent[mem_idx], sent[q_idx]
    rng = np.random.RandomState(0)
    K = args.slots
    out = {"eeg": {**sentence_id(sim, mem_sent, q_sent),
                   **uma_from_units(vote_units(sim, mem_sent, assets, len(bank), args.k, args.temp, K),
                                    q_sent, assets, bank, args.taus, rng)}}
    if with_baselines:
        lq, lm = lengths[q_idx].float().log(), lengths[mem_idx].float().log()
        simL = -(lq[:, None] - lm[None]).abs() * 20
        out["reading_time_only"] = {**sentence_id(simL, mem_sent, q_sent),
                                    **uma_from_units(vote_units(simL, mem_sent, assets, len(bank), args.k, args.temp, K),
                                                     q_sent, assets, bank, args.taus, rng)}
        freq = torch.zeros(len(bank))
        for s in mem_sent:
            freq[assets["sentence_unit_cids"][s]] += 1
        best = torch.topk(freq, K)
        out["frequency_prior"] = uma_from_units([(best.indices, best.values)] * len(q_idx), q_sent, assets, bank,
                                                args.taus, rng)
        orc = [(torch.as_tensor(assets["sentence_unit_cids"][s]), None) for s in q_sent]
        out["oracle"] = uma_from_units(orc, q_sent, assets, bank, args.taus, rng)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="zuco1_SR")
    ap.add_argument("--test_subjects", nargs="+", default=["ZAB", "ZJM", "ZKW"])
    ap.add_argument("--val_subjects", nargs="+", default=["ZKH"])
    ap.add_argument("--scramble_eeg", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sent_per_batch", type=int, default=16)
    ap.add_argument("--views", type=int, default=4, help="subjects per sentence in a batch (positives)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lambda_text", type=float, default=0.5)
    ap.add_argument("--k", type=int, default=10, help="neighbours that vote")
    ap.add_argument("--temp", type=float, default=0.05, help="softmax temperature of the vote")
    ap.add_argument("--slots", type=int, default=39)
    ap.add_argument("--taus", nargs="+", type=float, default=[0.7, 0.8, 0.85])
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    assets = torch.load(TEXT, weights_only=False)
    bank = F.normalize(assets["bank_embeddings"].float(), dim=-1)
    sent_emb = F.normalize(assets["sentence_embeddings"].float(), dim=-1)
    eeg, lengths, sent, subj, names = load_trials(args.task, assets, args.scramble_eeg)
    name_of = np.array([names[s] for s in subj])
    te = np.where(np.isin(name_of, args.test_subjects))[0]
    va = np.where(np.isin(name_of, args.val_subjects))[0]
    tr = np.where(~np.isin(name_of, args.test_subjects + args.val_subjects))[0]
    print(f"train users {sorted(set(name_of[tr]))} ({len(tr)} trials) | val {args.val_subjects} ({len(va)}) | "
          f"test {args.test_subjects} ({len(te)}) | scramble_eeg={args.scramble_eeg} | "
          f"{np.isin(sent[te], sent[tr]).mean():.0%} of test sentences exist in memory", flush=True)

    model = EEGEncoder(bank.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    by_sent = {}
    for i in tr:
        by_sent.setdefault(int(sent[i]), []).append(int(i))
    sents = [s for s, v in by_sent.items() if len(v) >= 2]
    steps = len(sents) // args.sent_per_batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    out_dir = Path(args.out) if args.out else ROOT / "brainmosaic/runs" / (
        f"retrieval_{args.task}{'_SCRAMBLED' if args.scramble_eeg else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_state, best_ep, hist = -1e9, None, -1, []
    rng = np.random.RandomState(0)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        rng.shuffle(sents)
        tot = 0.0
        for s in range(steps):
            batch_s = sents[s * args.sent_per_batch:(s + 1) * args.sent_per_batch]
            idx, lab = [], []
            for si in batch_s:
                pick = rng.choice(by_sent[si], size=min(args.views, len(by_sent[si])), replace=False)
                idx += list(pick)
                lab += [si] * len(pick)
            idx = torch.as_tensor(idx)
            x = eeg[idx].to(dev).float()
            x = x + 0.1 * torch.randn_like(x)
            z, zt = model(x, lengths[idx].to(dev))
            lab_t = torch.as_tensor(lab, device=dev)
            l_con = supcon(z, lab_t, args.tau)
            uniq = torch.as_tensor(batch_s, device=dev)
            T = sent_emb[uniq.cpu()].to(dev)
            tgt = (lab_t[:, None] == uniq[None]).float().argmax(1)
            l_txt = F.cross_entropy(zt @ T.T / args.tau, tgt)
            loss = l_con + args.lambda_text * l_txt
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        vm = evaluate_all(model, eeg, lengths, sent, tr, va, assets, bank, args, dev)["eeg"]
        score = vm["sent_top5"] + vm["UMA@0.7"]
        hist.append({"epoch": ep, "loss": tot / steps, **{f"val_{k}": v for k, v in vm.items()}})
        if score > best:
            best, best_ep, best_state = score, ep, copy.deepcopy(model.state_dict())
        print(f"  ep {ep:3d} loss {tot / steps:.3f} | val sentence top1 {vm['sent_top1']:.3f} top5 {vm['sent_top5']:.3f} "
              f"rank% {vm['sent_rank_pct']:.3f} | UMA@0.7 {vm['UMA@0.7']:.3f} (othersent {vm['UMA@0.7_othersent']:.3f}, "
              f"random {vm['UMA@0.7_random']:.3f}) MUS {vm['MUS']:.3f} | {time.time() - t0:.1f}s", flush=True)
        if ep - best_ep >= args.patience:
            print(f"  early stop at epoch {ep} (best {best_ep})")
            break

    model.load_state_dict(best_state)
    torch.save({"model": best_state, "args": vars(args)}, out_dir / "best.pth")
    res = evaluate_all(model, eeg, lengths, sent, tr, te, assets, bank, args, dev, with_baselines=True)
    json.dump({"test": res, "history": hist, "best_epoch": best_ep}, open(out_dir / "metrics.json", "w"), indent=1)
    print(f"\n=== TEST on new users {args.test_subjects} (best epoch {best_ep}) ===")
    for name, r in res.items():
        line = f"{name:18s} UMA@0.7 {r['UMA@0.7']:.4f} (othersent {r['UMA@0.7_othersent']:.4f}, random {r['UMA@0.7_random']:.4f})"
        line += f"  UMA@0.85 {r['UMA@0.85']:.4f}  MUS {r['MUS']:.4f}"
        if "sent_top1" in r:
            line += f"  | sentence top1 {r['sent_top1']:.3f} top5 {r['sent_top5']:.3f} (chance {1 / r['n_candidates']:.4f} / {5 / r['n_candidates']:.4f})"
        print(line, flush=True)
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
