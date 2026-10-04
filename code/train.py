import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from criterion import SetCriterion, hungarian_match
from model import BrainMosaic

ROOT = Path(__file__).resolve().parent.parent
ATTRS = ("sentence_mode", "subjectivity", "semantic_focus")


class SubjectData:
    def __init__(self, eeg_file, assets, device, shuffle_eeg=False):
        files = eeg_file if isinstance(eeg_file, (list, tuple)) else [eeg_file]
        parts = [torch.load(f, weights_only=False) for f in files]
        d = {"eeg": torch.cat([p["eeg"] for p in parts]), "lengths": torch.cat([p["lengths"] for p in parts]),
             "sentences": [s for p in parts for s in p["sentences"]]}
        del parts
        s2i = {s: i for i, s in enumerate(assets["sentences"])}
        keep = [i for i, s in enumerate(d["sentences"]) if s in s2i and assets["sentence_unit_cids"][s2i[s]]]
        if len(keep) < len(d["sentences"]):
            print(f"  [warn] {len(d['sentences']) - len(keep)} trials without text assets/units dropped")
        self.eeg = d["eeg"][keep]
        self.lengths = d["lengths"][keep]
        if shuffle_eeg:
            perm = torch.from_numpy(np.random.RandomState(1234).permutation(len(keep)))
            self.eeg, self.lengths = self.eeg[perm], self.lengths[perm]
        self.sent_idx = torch.tensor([s2i[d["sentences"][i]] for i in keep])
        self.assets, self.device = assets, device
        bank = assets["bank_embeddings"].to(device)
        sent_emb = assets["sentence_embeddings"].to(device)
        self.targets = []
        for si in self.sent_idx.tolist():
            cids = torch.tensor(assets["sentence_unit_cids"][si], device=device)
            t = {"unit_cids": cids, "unit_embeddings": bank[cids], "sentence_embedding": sent_emb[si],
                 "sentence_idx": si}
            t.update(assets["attributes"][si])
            self.targets.append(t)

    def __len__(self):
        return len(self.sent_idx)

    def batch(self, idx):
        idx = torch.as_tensor(idx)
        x = self.eeg[idx].to(self.device, non_blocking=True).float()
        return x, self.lengths[idx].to(self.device), [self.targets[i] for i in idx.tolist()]


def split_indices(n, test_frac, val_frac, seed):
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(round(n * test_frac))
    test, rest = perm[:n_test], perm[n_test:]
    n_val = int(round(len(rest) * val_frac))
    return rest[n_val:], rest[:n_val], test


@torch.no_grad()
def evaluate(model, criterion, data, idx, cfg, bank, members, bs=64, collect=False):
    model.eval()
    E = cfg["eval"]
    taus, main_tau = E["uma_taus"], E["main_tau"]
    sums = {"loss": 0.0, "n_batches": 0}
    sims_all, sims_shuf, sims_unif = [], [], []
    n_gt = 0
    tp = n_pred = n_gold = 0
    n_active = 0
    sent_cos, attr_ok = [], {a: [] for a in ATTRS}
    s_hat_all, s_idx_all = [], []
    preds = []
    rng = np.random.RandomState(0)
    for start in range(0, len(idx), bs):
        bidx = idx[start:start + bs]
        x, lengths, targets = data.batch(bidx)
        out = model(x, lengths)
        losses, indices = criterion(out, targets)
        sums["loss"] += sum(v.item() for v in losses.values())
        sums["n_batches"] += 1

        pe = F.normalize(out["pred_embeddings"], dim=-1)
        for b, (si, ti) in enumerate(indices):
            gold = targets[b]["unit_embeddings"]
            sim = torch.zeros(len(gold), device=gold.device)
            sim[ti.to(gold.device)] = (pe[b, si.to(gold.device)] * gold[ti.to(gold.device)]).sum(-1)
            sims_all.append(sim.cpu())
            n_gt += len(gold)
        shuf_targets = [data.targets[j] for j in rng.choice(idx, size=len(bidx))]
        for b, (si, ti) in enumerate(hungarian_match(out, shuf_targets, cfg["loss"]["cost_class"], cfg["loss"]["cost_emb"])):
            gold = shuf_targets[b]["unit_embeddings"]
            sim = torch.zeros(len(gold), device=gold.device)
            sim[ti.to(gold.device)] = (pe[b, si.to(gold.device)] * gold[ti.to(gold.device)]).sum(-1)
            sims_shuf.append(sim.cpu())

        unif_targets = []
        for t in targets:
            cids = torch.as_tensor(rng.choice(len(bank), size=len(t["unit_cids"]), replace=False), device=bank.device)
            unif_targets.append({"unit_cids": cids, "unit_embeddings": bank[cids]})
        for b, (si, ti) in enumerate(hungarian_match(out, unif_targets, cfg["loss"]["cost_class"], cfg["loss"]["cost_emb"])):
            gold = unif_targets[b]["unit_embeddings"]
            sim = torch.zeros(len(gold), device=gold.device)
            sim[ti.to(gold.device)] = (pe[b, si.to(gold.device)] * gold[ti.to(gold.device)]).sum(-1)
            sims_unif.append(sim.cpu())

        prob = out["pred_logits"].softmax(-1)[..., 1]
        bank_sim = pe @ bank.T
        for b in range(len(bidx)):
            active = (prob[b] > E["exist_threshold"]).nonzero().flatten()
            active = active[active > 0]
            n_active += len(active)
            gold_c = set(targets[b]["unit_cids"].tolist())
            top1 = set(bank_sim[b, active].argmax(-1).tolist()) if len(active) else set()
            tp += len(top1 & gold_c)
            n_pred += len(top1)
            n_gold += len(gold_c)
            if collect:
                k = E["top_k"]
                slots = []
                for j in active.tolist():
                    sc, ci = bank_sim[b, j].topk(k)
                    slots.append({"slot": j, "p_active": round(prob[b, j].item(), 4),
                                  "candidates": [[members[c], round(s, 4)] for c, s in zip(ci.tolist(), sc.tolist())]})
                slots.sort(key=lambda s: -s["p_active"])
                si = targets[b]["sentence_idx"]
                preds.append({
                    "gold_sentence": data.assets["sentences"][si],
                    "gold_units": data.assets["sentence_units"][si],
                    "slots": slots,
                    **{f"{a}_probs": [round(v, 4) for v in out[a][b].softmax(-1).tolist()] for a in ATTRS},
                    **{f"gold_{a}": targets[b][a] for a in ATTRS},
                })

        s_hat = F.normalize(out["sentence_embedding"], dim=-1)
        s_gold = torch.stack([t["sentence_embedding"] for t in targets])
        sent_cos.append((s_hat * s_gold).sum(-1).cpu())
        s_hat_all.append(s_hat.cpu())
        s_idx_all += [t["sentence_idx"] for t in targets]
        for a in ATTRS:
            gold = torch.tensor([t[a] for t in targets], device=out[a].device)
            attr_ok[a].append((out[a].argmax(-1) == gold).cpu())

    sims = torch.cat(sims_all)
    shuf = torch.cat(sims_shuf)
    m = {"loss": sums["loss"] / max(sums["n_batches"], 1), "n_trials": len(idx)}
    unif_all = torch.cat(sims_unif)
    for t in taus:
        m[f"UMA@{t}"] = (sims > t).float().mean().item()
        m[f"UMA@{t}_shuffled"] = (shuf > t).float().mean().item()
        m[f"UMA@{t}_random"] = (unif_all > t).float().mean().item()
    m["UMA"] = m[f"UMA@{main_tau}"]
    m["MUS"] = sims.mean().item()
    unif = torch.cat(sims_unif)
    r = E.get("paper_random_uma", 0.2557)
    tau_star = torch.quantile(unif, 1 - r).item()
    m["tau_paper"] = tau_star
    m["UMA_paper"] = (sims > tau_star).float().mean().item()
    m["UMA_paper_othersent"] = (shuf > tau_star).float().mean().item()
    m["MUS_shuffled"] = shuf.mean().item()
    p = tp / max(n_pred, 1)
    r = tp / max(n_gold, 1)
    m.update({"ret_precision": p, "ret_recall": r, "ret_f1": 2 * p * r / max(p + r, 1e-9),
              "active_slots": n_active / len(idx)})
    m["sent_cos"] = torch.cat(sent_cos).mean().item()
    S = F.normalize(data.assets["sentence_embeddings"][s_idx_all], dim=-1)
    ranks = ((torch.cat(s_hat_all) @ S.T) > (torch.cat(s_hat_all) * S).sum(-1, keepdim=True)).sum(-1).float()
    m["sent_top1"] = (ranks == 0).float().mean().item()
    m["sent_mean_rank_pct"] = (ranks / len(ranks)).mean().item()
    for a in ATTRS:
        m[f"acc_{a}"] = torch.cat(attr_ok[a]).float().mean().item()
    return m, preds


def train_subject(cfg, data, split, out_dir, device, seed, log_every=0, train_limit=0, init=None):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tr_idx, va_idx, te_idx = split
    if train_limit:
        tr_idx = tr_idx[:train_limit]
    tr_probe = tr_idx[:64]
    assets = data.assets
    bank = assets["bank_embeddings"].to(device)
    members = assets["bank_members"]

    attr_counts = {a: np.bincount([data.targets[i][a] for i in tr_idx], minlength=n).tolist()
                   for a, n in zip(ATTRS, (4, 2, 5))}
    model = BrainMosaic(cfg, in_channels=data.eeg.shape[1], text_dim=bank.shape[1]).to(device)
    if init:
        missing, unexpected = model.load_state_dict(torch.load(init, weights_only=False)["model"], strict=False)
        print(f"  init from {init} (missing {len(missing)}, unexpected {len(unexpected)})")
    criterion = SetCriterion(cfg, bank, attr_counts).to(device)

    T = cfg["train"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone") and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if n.startswith("backbone") and p.requires_grad],
         "lr": T["lr_backbone"]},
    ]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, T["lr_drop"])
    bs = T["batch_size"]
    select_by = T["select_by"] if len(va_idx) else "last"

    best, best_state, best_epoch, history = -math.inf, None, -1, []
    for epoch in range(T["epochs"]):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(tr_idx)
        ep_loss, nb = {}, 0
        for start in range(0, len(perm) - bs + 1 if len(perm) >= bs else 1, bs):
            x, lengths, targets = data.batch(perm[start:start + bs])
            losses, _ = criterion(model(x, lengths), targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: { {k: v.item() for k, v in losses.items()} }")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if T["clip_max_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), T["clip_max_norm"])
            opt.step()
            for k, v in losses.items():
                ep_loss[k] = ep_loss.get(k, 0.0) + v.item()
            nb += 1
        sched.step()
        rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1), **{f"train_{k}": v / nb for k, v in ep_loss.items()}}
        if len(va_idx) and (epoch % T["eval_every"] == 0 or epoch == T["epochs"] - 1):
            vm, _ = evaluate(model, criterion, data, va_idx, cfg, bank, members)
            rec.update({f"val_{k}": v for k, v in vm.items()})
            if select_by == "val_gap":
                score = vm["UMA_paper"] - vm["UMA_paper_othersent"] + (vm["MUS"] - vm["MUS_shuffled"])
            else:
                score = vm["UMA"] if select_by == "val_uma" else vm["MUS"] if select_by == "val_mus" else -vm["loss"]
            if select_by != "last" and score > best:
                best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        history.append(rec)
        if log_every and (epoch % log_every == 0 or epoch == T["epochs"] - 1):
            msg = f"    ep {epoch:3d} loss {sum(v for k, v in rec.items() if k.startswith('train_')):.3f}"
            tm_, _ = evaluate(model, criterion, data, tr_probe, cfg, bank, members)
            msg += f" | train UMA {tm_['UMA']:.3f} MUS {tm_['MUS']:.3f} (shuf {tm_['MUS_shuffled']:.3f})"
            if "val_UMA" in rec:
                msg += f" | val UMA_paper {rec['val_UMA_paper']:.3f} (othersent {rec['val_UMA_paper_othersent']:.3f}) UMA {rec['val_UMA']:.3f} MUS {rec['val_MUS']:.3f} (shuf {rec['val_MUS_shuffled']:.3f}) F1 {rec['val_ret_f1']:.3f}"
            print(msg + f" | {rec['time_s']}s", flush=True)

    last_state = copy.deepcopy(model.state_dict())
    results = {}
    for tag, state in [("last", last_state)] + ([("best_val", best_state)] if best_state is not None else []):
        model.load_state_dict(state)
        tm, preds = evaluate(model, criterion, data, te_idx, cfg, bank, members, collect=True)
        results[tag] = tm
        torch.save({"model": state, "cfg": cfg, "epoch": best_epoch if tag == "best_val" else T["epochs"] - 1},
                   out_dir / f"{tag}.pth")
        with open(out_dir / f"test_predictions_{tag}.json", "w") as f:
            json.dump(preds, f, indent=1, ensure_ascii=False)
    results["best_epoch"] = best_epoch
    results["split_sizes"] = {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)}
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test": results, "history": history}, f, indent=1)
    return results


def summarize(all_results, keys):
    out = {}
    for tag in ("best_val", "last"):
        rows = [r[tag] for r in all_results if tag in r]
        if rows:
            out[tag] = {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
                        for k in keys if k in rows[0]}
            out[tag]["n_runs"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "brainmosaic/configs/zuco.json"))
    ap.add_argument("--task", default="zuco1_SR", help="zuco1_SR | zuco1_NR | zuco1_TSR | zuco2_NR | zuco2_TSR")
    ap.add_argument("--subjects", nargs="+", default=["all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="one random 8:2 split per seed")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else
                    "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--shuffle_eeg", action="store_true",
                    help="control: pair every sentence with another trial's EEG (measures the text-prior-only score)")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from (e.g. a --pretrain_tasks run)")
    ap.add_argument("--pretrain_tasks", nargs="+", default=None,
                    help="pool ALL subjects of these tasks into one model (95/5 train/val), e.g. zuco1_NR zuco1_TSR")
    ap.add_argument("--train_limit", type=int, default=0, help="debug: use only the first N training trials")
    ap.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                    help="override config entries, e.g. --set train.lr=3e-4 loss.lambda_infonce=0")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    for kv in args.set:
        key, val = kv.split("=", 1)
        sec, k = key.split(".", 1)
        cfg[sec][k] = json.loads(val)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    device = torch.device(args.device)

    assets = torch.load(ROOT / cfg["data"]["text_assets"], weights_only=False)
    cfg["eval"].setdefault("main_tau", assets["cluster_threshold"])
    if cfg["model"]["num_queries"] == "auto":
        cfg["model"]["num_queries"] = 1 + max(len(c) for c in assets["sentence_unit_cids"])
    print(f"text space: {assets['model']} dim={assets['dim']} | bank {len(assets['bank_members'])} units | "
          f"MUS_exp={assets['mus_exp']:.3f} | K={cfg['model']['num_queries']} | main tau={cfg['eval']['main_tau']}")

    if args.pretrain_tasks:
        files = [f for t in args.pretrain_tasks for f in sorted((ROOT / cfg["data"]["work_dir"] / cfg["data"].get("eeg_dir", "eeg") / t).glob("*.pt"))]
        run_dir = Path(args.out) if args.out else ROOT / "brainmosaic/runs" / f"pretrain_{'+'.join(args.pretrain_tasks)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        data = SubjectData(files, assets, device, shuffle_eeg=args.shuffle_eeg)
        perm = np.random.RandomState(args.seeds[0]).permutation(len(data))
        n_va = max(64, len(perm) // 20)
        va, tr = perm[:n_va], perm[n_va:]
        print(f"[pretrain] {len(files)} subject-task files, {len(data)} trials -> train {len(tr)} / val {len(va)}", flush=True)
        cfg["train"]["select_by"] = "last"
        res = train_subject(cfg, data, (tr, np.array([], dtype=int), va), run_dir, device, args.seeds[0],
                            args.log_every, 0, args.init)
        print(f"[pretrain] done -> {run_dir / 'last.pth'}  val UMA {res['last']['UMA']:.4f} MUS {res['last']['MUS']:.4f} "
              f"(shuf {res['last']['MUS_shuffled']:.4f})")
        return

    eeg_dir = ROOT / cfg["data"]["work_dir"] / cfg["data"].get("eeg_dir", "eeg") / args.task
    files = sorted(eeg_dir.glob("*.pt"))
    if args.subjects != ["all"]:
        files = [f for f in files if f.stem in args.subjects]
    if not files:
        raise SystemExit(f"no subject files in {eeg_dir}")
    run_dir = Path(args.out) if args.out else ROOT / "brainmosaic/runs" / f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump({**cfg, "task": args.task, "subjects": [f.stem for f in files], "seeds": args.seeds}, f, indent=1)

    all_results = []
    for f in files:
        data = SubjectData(f, assets, device, shuffle_eeg=args.shuffle_eeg)
        for seed in args.seeds:
            split = split_indices(len(data), cfg["data"]["test_frac"], cfg["data"]["val_frac"], seed)
            out_dir = run_dir / f.stem / f"seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{args.task}/{f.stem} seed {seed}] trials {len(data)} -> train {len(split[0])} / "
                  f"val {len(split[1])} / test {len(split[2])}", flush=True)
            res = train_subject(cfg, data, split, out_dir, device, seed, args.log_every, args.train_limit, args.init)
            res["subject"], res["seed"] = f.stem, seed
            all_results.append(res)
            for tag in ("best_val", "last"):
                if tag in res:
                    r = res[tag]
                    print(f"  TEST[{tag}] UMA_paper {r['UMA_paper']:.4f} (othersent {r['UMA_paper_othersent']:.4f}) | UMA {r['UMA']:.4f} (shuf {r[f'UMA@{cfg['eval']['main_tau']}_shuffled']:.4f}) "
                          f"MUS {r['MUS']:.4f} (shuf {r['MUS_shuffled']:.4f}) ret-F1 {r['ret_f1']:.4f} "
                          f"sent_cos {r['sent_cos']:.4f} sent_top1 {r['sent_top1']:.3f}", flush=True)
        del data
        if device.type == "mps":
            torch.mps.empty_cache()

    keys = sorted(all_results[0]["last"].keys())
    summary = summarize(all_results, keys)
    with open(run_dir / "summary.json", "w") as fh:
        json.dump({"summary": summary, "per_run": all_results}, fh, indent=1)
    print(f"\n=== {args.task}: mean +- std over {len(all_results)} subject-runs (test split) ===")
    for tag, s in summary.items():
        line = "  ".join(f"{k} {s[k]['mean']:.4f}±{s[k]['std']:.4f}"
                         for k in ("UMA_paper", "UMA_paper_othersent", "UMA", "MUS", "MUS_shuffled", "ret_f1", "sent_cos") if k in s)
        print(f"[{tag}] {line}")
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
