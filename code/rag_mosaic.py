import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from criterion import SetCriterion
from model import DecoderLayer, EncoderLayer, ModernTCNEncoder, sinusoidal
from retrieval_mosaic import EEGEncoder, TEXT, load_trials
from train import evaluate

ROOT = Path(os.environ.get("BM_ROOT", Path(__file__).resolve().parent.parent))
CFG = {
    "model": {"hidden_dim": 256, "nheads": 8, "dim_feedforward": 1024, "enc_layers": 2, "dec_layers": 4,
              "dropout": 0.2, "num_queries": 40},
    "loss": {"cost_class": 1.0, "cost_emb": 2.0, "eos_coef": 0.3, "lambda_cls": 1.0, "lambda_cos": 1.0,
             "lambda_infonce": 0.2, "tau": 0.07, "lambda_rep": 0.0, "rep_margin": 0.6, "lambda_global": 0.2,
             "lambda_attr": 0.0},
    "eval": {"uma_taus": [0.7, 0.8, 0.85], "main_tau": 0.7, "exist_threshold": 0.5, "top_k": 5,
             "paper_random_uma": 0.2557},
}


def zscore(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


class RAGData:
    def __init__(self, eeg, lengths, sent, subj, assets, device, fingerprints, memory, k, max_units, alpha):
        self.eeg, self.lengths, self.sent, self.subj = eeg, lengths, sent, subj
        self.assets, self.device, self.k, self.max_units, self.alpha = assets, device, k, max_units, alpha
        self.Z = fingerprints
        self.memory = memory
        self.bank = F.normalize(assets["bank_embeddings"].float(), dim=-1).to(device)
        self.semb = F.normalize(assets["sentence_embeddings"].float(), dim=-1)
        self.mem_text = self.semb[sent[memory]]
        self.targets = []
        for s in sent:
            c = torch.tensor(assets["sentence_unit_cids"][s], device=device)
            self.targets.append({"unit_cids": c, "unit_embeddings": self.bank[c],
                                 "sentence_embedding": self.semb[s].to(device), "sentence_idx": int(s),
                                 "sentence_mode": 0, "subjectivity": 0, "semantic_focus": 0})
        self.search = "combined"
        self.model = None
        self.rng = np.random.RandomState(0)
        self.hits = []
        self.true_drop = 0.0

    def __len__(self):
        return len(self.sent)

    @torch.no_grad()
    def neighbours(self, idx, x, L):
        k = self.k
        if self.search == "random":
            nbs = np.stack([self.rng.choice(self.memory, size=k, replace=False) for _ in idx])
            return nbs, np.ones((len(idx), k), dtype=np.float32)
        parts = []
        if self.search in ("embedding", "combined"):
            was = self.model.training
            self.model.eval()
            z = F.normalize(self.model(dict(x, **self.empty_ctx(len(idx))), L)["sentence_embedding"], dim=-1).cpu()
            self.model.train(was)
            parts.append((self.alpha if self.search == "combined" else 1.0, zscore(z @ self.mem_text.T)))
        if self.search in ("fingerprint", "combined"):
            parts.append(((1 - self.alpha) if self.search == "combined" else 1.0,
                          zscore(self.Z[idx] @ self.Z[self.memory].T)))
        score = sum(w * p for w, p in parts)
        same = torch.as_tensor(self.subj[self.memory][None, :] == self.subj[idx][:, None])
        score = score.masked_fill(same, -1e9)
        top = torch.topk(score, k, dim=1)
        nbs = self.memory[top.indices.numpy()]
        sims = torch.sigmoid(top.values).numpy()
        if self.true_drop > 0:
            for b, i in enumerate(idx):
                for r in range(k):
                    if self.sent[nbs[b, r]] == self.sent[i] and self.rng.rand() < self.true_drop:
                        j = self.rng.choice(self.memory)
                        while self.subj[j] == self.subj[i]:
                            j = self.rng.choice(self.memory)
                        nbs[b, r] = j
        self.hits += [self.sent[i] in set(self.sent[n]) for i, n in zip(idx, nbs)]
        return nbs, sims

    def empty_ctx(self, B):
        d = self.bank.shape[1]
        pad = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        pad[:, 0] = False
        return {"ctx_emb": torch.zeros(B, 1, d, device=self.device), "ctx_sim": torch.zeros(B, 1, device=self.device),
                "ctx_rank": torch.zeros(B, 1, dtype=torch.long, device=self.device), "ctx_pad": pad}

    def batch(self, idx):
        idx = np.asarray([int(i) for i in idx])
        x = {"eeg": self.eeg[idx].to(self.device).float()}
        L = self.lengths[idx].to(self.device)
        if self.search == "none":
            x.update(self.empty_ctx(len(idx)))
        else:
            nbs, sims = self.neighbours(idx, x, L)
            U, k, B = self.max_units, self.k, len(idx)
            emb = torch.zeros(B, k * U, self.bank.shape[1], device=self.device)
            sim = torch.zeros(B, k * U, device=self.device)
            rank = torch.zeros(B, k * U, dtype=torch.long, device=self.device)
            pad = torch.ones(B, k * U, dtype=torch.bool, device=self.device)
            for b in range(B):
                for r, (j, sj) in enumerate(zip(nbs[b], sims[b])):
                    c = self.assets["sentence_unit_cids"][self.sent[j]][:U]
                    o = r * U
                    emb[b, o:o + len(c)] = self.bank[torch.as_tensor(c, device=self.device)]
                    sim[b, o:o + len(c)] = float(sj)
                    rank[b, o:o + len(c)] = r
                    pad[b, o:o + len(c)] = False
            x.update({"ctx_emb": emb, "ctx_sim": sim, "ctx_rank": rank, "ctx_pad": pad})
        return x, L, [self.targets[i] for i in idx]


class RAGMosaic(nn.Module):
    def __init__(self, cfg, text_dim, k):
        super().__init__()
        m = cfg["model"]
        d = m["hidden_dim"]
        self.tcn = ModernTCNEncoder(105, d=64, patch=25, blocks=2, large=25, small=5, dropout=0.1)
        self.q_proj = nn.Linear(64, d)
        self.ctx_proj = nn.Sequential(nn.Linear(text_dim, d), nn.GELU(), nn.Linear(d, d))
        self.sim_proj = nn.Linear(1, d)
        self.rank_emb = nn.Embedding(k, d)
        self.type_emb = nn.Embedding(2, d)
        self.enc = nn.ModuleList([EncoderLayer(d, m["nheads"], m["dim_feedforward"], m["dropout"])
                                  for _ in range(m["enc_layers"])])
        self.dec = nn.ModuleList([DecoderLayer(d, m["nheads"], m["dim_feedforward"], m["dropout"])
                                  for _ in range(m["dec_layers"])])
        self.dec_norm = nn.LayerNorm(d)
        K = m["num_queries"]
        self.qpos = nn.Embedding(K, d)
        self.qcontent = nn.Embedding(K, d)
        self.class_head = nn.Linear(d, 2)
        self.emb_head = nn.Linear(d, text_dim)
        self.attr = nn.ModuleList([nn.Linear(d, n) for n in (4, 2, 5)])
        nn.init.normal_(self.qpos.weight, std=1.0)
        nn.init.normal_(self.qcontent.weight, std=1.0)

    def forward(self, x, lengths=None):
        q = self.q_proj(self.tcn(x["eeg"]))
        B, N, d = q.shape
        n_valid = torch.clamp((lengths + 24) // 25, 1, N)
        q_pad = torch.arange(N, device=q.device)[None] >= n_valid[:, None]
        q = q + sinusoidal(N, d, q.device)[None] + self.type_emb.weight[0]
        for layer in self.enc:
            q = layer(q, torch.zeros_like(q), q_pad)
        c = self.ctx_proj(x["ctx_emb"]) + self.sim_proj(x["ctx_sim"].unsqueeze(-1)) + \
            self.rank_emb(x["ctx_rank"]) + self.type_emb.weight[1]
        mem = torch.cat([q, c], 1)
        pad = torch.cat([q_pad, x["ctx_pad"]], 1)
        qpos = self.qpos.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.qcontent.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.dec:
            hs = layer(hs, mem, qpos, torch.zeros_like(mem), pad)
        hs = self.dec_norm(hs)
        emb = self.emb_head(hs)
        out = {"pred_logits": self.class_head(hs), "pred_embeddings": emb, "sentence_embedding": emb[:, 0]}
        for name, head in zip(("sentence_mode", "subjectivity", "semantic_focus"), self.attr):
            out[name] = head(hs[:, 0])
        return out


def contrastive(out, targets, tau):
    z = F.normalize(out["sentence_embedding"], dim=-1)
    sids = [t["sentence_idx"] for t in targets]
    uniq = sorted(set(sids))
    T = torch.stack([targets[sids.index(s)]["sentence_embedding"] for s in uniq])
    lab = torch.tensor([uniq.index(s) for s in sids], device=z.device)
    return F.cross_entropy(z @ T.T / tau, lab)


@torch.no_grad()
def fingerprints(ret_ck, eeg, lengths, text_dim, dev):
    enc = EEGEncoder(text_dim).to(dev)
    enc.load_state_dict(torch.load(ret_ck, weights_only=False)["model"])
    enc.eval()
    Z = []
    for s in range(0, len(eeg), 64):
        Z.append(enc(eeg[s:s + 64].to(dev).float(), lengths[s:s + 64].to(dev))[0].cpu())
    return torch.cat(Z)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scrambled", action="store_true")
    ap.add_argument("--ret_ck", default=None, help="fingerprint encoder (default runs/retrieval_zuco1_SR[_SCRAMBLED]/best.pth)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test_subjects", nargs="+", default=["ZAB", "ZJM", "ZKW"])
    ap.add_argument("--val_subjects", nargs="+", default=["ZKH"])
    ap.add_argument("--init_encoder", default=None, help="self-supervised encoder from ssl_pretrain.py")
    ap.add_argument("--augment", action="store_true", help="amplitude scaling, channel dropout, time shift, noise")
    ap.add_argument("--init_from", default=None, help="full RAG-Mosaic checkpoint to continue from")
    ap.add_argument("--eval_only", action="store_true", help="skip training; test --init_from on the new users")
    ap.add_argument("--soft_uma", type=float, default=0.0, help="weight of the soft-UMA loss on matched slots")
    ap.add_argument("--soft_tau", type=float, default=0.7, help="UMA threshold targeted by the soft-UMA loss")
    ap.add_argument("--soft_T", type=float, default=0.05, help="temperature of the soft-UMA sigmoid")
    ap.add_argument("--swa_start", type=int, default=-1, help="epoch from which weights are averaged (SWA); -1 = off")
    ap.add_argument("--true_drop", type=float, default=0.0,
                    help="retrieval dropout in training: P(swap out a neighbour that holds the query's own sentence)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_units", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the embedding-space search in 'combined'")
    ap.add_argument("--lambda_con", type=float, default=0.5, help="contrastive loss weight")
    ap.add_argument("--con_tau", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    sfx = "_SCRAMBLED" if args.scrambled else ""
    out_dir = Path(args.out) if args.out else ROOT / f"brainmosaic/runs/rag_mosaic{sfx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = torch.load(TEXT, weights_only=False)
    eeg, lengths, sent, subj, names = load_trials("zuco1_SR", assets, args.scrambled)
    name_of = np.array([names[s] for s in subj])
    te = np.where(np.isin(name_of, args.test_subjects))[0]
    va = np.where(np.isin(name_of, args.val_subjects))[0]
    tr = np.where(~np.isin(name_of, args.test_subjects + args.val_subjects))[0]
    text_dim = assets["bank_embeddings"].shape[1]
    Z = fingerprints(args.ret_ck or ROOT / f"brainmosaic/runs/retrieval_zuco1_SR{sfx}/best.pth", eeg, lengths,
                     text_dim, dev)
    data = RAGData(eeg, lengths, sent, subj, assets, dev, Z, tr, args.k, args.max_units, args.alpha)
    model = RAGMosaic(CFG, text_dim, args.k).to(dev)
    if args.init_encoder:
        sd = torch.load(args.init_encoder, weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"init query encoder from {args.init_encoder}: loaded {len(sd)} tensors, unexpected {len(unexpected)}", flush=True)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, weights_only=False)["model"])
        print(f"continuing from {args.init_from}", flush=True)
    swa = torch.optim.swa_utils.AveragedModel(model) if args.swa_start >= 0 else None
    data.model = model
    bank = data.bank
    crit = SetCriterion(CFG, bank).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = len(tr) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    members = assets["bank_members"]
    print(f"train users {sorted(set(name_of[tr]))} | val {args.val_subjects} | test {args.test_subjects} | "
          f"k={args.k} alpha={args.alpha} lambda_con={args.lambda_con} | scrambled={args.scrambled}", flush=True)

    best, best_state, best_ep, best_uma = -1e9, None, -1, -1.0
    if args.eval_only:
        args.epochs, best_state, best_ep = 0, copy.deepcopy(model.state_dict()), "loaded"
    for ep in range(args.epochs):
        model.train()
        data.search, data.hits = "combined", []
        data.true_drop = args.true_drop
        t0 = time.time()
        perm = np.random.permutation(tr)
        tot = con = 0.0
        for s in range(steps):
            x, L, tg = data.batch(perm[s * args.batch_size:(s + 1) * args.batch_size])
            if args.augment:
                from ssl_pretrain import augment
                e = x["eeg"]
                shift = int(np.random.randint(-25, 26))
                e = torch.roll(e, shifts=shift, dims=-1)
                x["eeg"] = augment(e)
            else:
                x["eeg"] = x["eeg"] + 0.1 * torch.randn_like(x["eeg"])
            out = model(x, L)
            losses, indices = crit(out, tg)
            lc = contrastive(out, tg, args.con_tau)
            loss = sum(losses.values()) + args.lambda_con * lc
            if args.soft_uma > 0:
                pe = F.normalize(out["pred_embeddings"], dim=-1)
                cs = torch.cat([(pe[b, si.to(dev)] * tg[b]["unit_embeddings"][ti.to(dev)]).sum(-1)
                                for b, (si, ti) in enumerate(indices)])
                loss = loss + args.soft_uma * (1 - torch.sigmoid((cs - args.soft_tau) / args.soft_T)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            con += lc.item()
        train_hit = float(np.mean(data.hits))
        data.hits, data.true_drop = [], 0.0
        if swa is not None and ep >= args.swa_start:
            swa.update_parameters(model)
            if ep > args.swa_start:
                live = copy.deepcopy(model.state_dict())
                model.load_state_dict(swa.module.state_dict())
                sm, _ = evaluate(model, crit, data, va, CFG, bank, members)
                print(f"      SWA (epochs {args.swa_start}..{ep}) val UMA@0.7 {sm['UMA']:.3f} (othersent "
                      f"{sm['UMA@0.7_shuffled']:.3f}, random {sm['UMA@0.7_random']:.3f})", flush=True)
                torch.save({"model": swa.module.state_dict(), "args": vars(args), "epoch": ep, "val": sm},
                           out_dir / "swa.pth")
                model.load_state_dict(live)
                data.hits = []
        vm, _ = evaluate(model, crit, data, va, CFG, bank, members)
        val_hit = float(np.mean(data.hits))
        score = vm["UMA"] - vm["UMA@0.7_random"]
        if score > best:
            best, best_ep, best_state = score, ep, copy.deepcopy(model.state_dict())
        (out_dir / "epochs").mkdir(exist_ok=True)
        ck = {"model": model.state_dict(), "args": vars(args), "epoch": ep, "val": vm}
        torch.save(ck, out_dir / "epochs" / f"ep{ep:02d}.pth")
        if vm["UMA"] > best_uma:
            best_uma = vm["UMA"]
            torch.save(ck, out_dir / "best_uma.pth")
        print(f"  ep {ep:3d} loss {tot / steps:.3f} (contrastive {con / steps:.3f}) | true sentence retrieved: train "
              f"{train_hit:.1%} val {val_hit:.1%} | val UMA@0.7 {vm['UMA']:.3f} (othersent {vm['UMA@0.7_shuffled']:.3f}, "
              f"random {vm['UMA@0.7_random']:.3f}) MUS {vm['MUS']:.3f} | {time.time() - t0:.1f}s", flush=True)
        if ep - best_ep >= args.patience:
            print(f"  early stop at epoch {ep} (best {best_ep})")
            break

    model.load_state_dict(best_state)
    if not args.eval_only:
        torch.save({"model": best_state, "args": vars(args)}, out_dir / "best.pth")
    res = {}
    print(f"\n=== RAG-Mosaic{' (SCRAMBLED EEG)' if args.scrambled else ''} | TEST on new users "
          f"{args.test_subjects} | best epoch {best_ep} | chance of retrieving the true sentence ~{args.k / 400:.1%} ===")
    extra = [("best_uma", out_dir / "best_uma.pth"), ("swa", out_dir / "swa.pth")]
    for tag, path in extra:
        if path.exists():
            model.load_state_dict(torch.load(path, weights_only=False)["model"])
            data.search, data.hits, data.rng = "combined", [], np.random.RandomState(0)
            m, _ = evaluate(model, crit, data, te, CFG, bank, members)
            res[f"{tag}_combined"] = m
            print(f"[{tag:8s}] search=combined UMA@0.7 {m['UMA']:.4f} (othersent {m['UMA@0.7_shuffled']:.4f}, "
                  f"random {m['UMA@0.7_random']:.4f})  MUS {m['MUS']:.4f} (shuf {m['MUS_shuffled']:.4f})", flush=True)
    model.load_state_dict(best_state)
    for mode in ("combined", "embedding", "fingerprint", "random", "none"):
        data.search, data.hits, data.rng = mode, [], np.random.RandomState(0)
        m, preds = evaluate(model, crit, data, te, CFG, bank, members, collect=(mode == "combined"))
        m["true_sentence_retrieved"] = float(np.mean(data.hits)) if data.hits else None
        res[mode] = m
        if preds:
            json.dump(preds, open(out_dir / "test_predictions.json", "w"), indent=1)
        hit = f"{m['true_sentence_retrieved']:.1%}" if m["true_sentence_retrieved"] is not None else "  -  "
        print(f"search={mode:11s} retrieved-true {hit:>6s} | UMA@0.7 {m['UMA']:.4f} (othersent "
              f"{m['UMA@0.7_shuffled']:.4f}, random {m['UMA@0.7_random']:.4f})  UMA@0.85 {m['UMA@0.85']:.4f}  "
              f"MUS {m['MUS']:.4f} (shuf {m['MUS_shuffled']:.4f})", flush=True)
    json.dump({"test": res, "best_epoch": best_ep, "args": vars(args)}, open(out_dir / "metrics.json", "w"), indent=1)
    print(f"results -> {out_dir}")


if __name__ == "__main__":
    main()
