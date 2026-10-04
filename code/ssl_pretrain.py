import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from model import EncoderLayer, ModernTCNEncoder, sinusoidal

ROOT = Path(__file__).resolve().parent.parent
TASKS = ["zuco1_SR", "zuco1_NR", "zuco1_TSR", "zuco2_NR", "zuco2_TSR"]
PATCH, C = 25, 105


def augment(x, rng_scale=0.2, ch_drop=0.1, noise=0.1):
    B, Cc, T = x.shape
    scale = 1 + rng_scale * (2 * torch.rand(B, Cc, 1, device=x.device) - 1)
    keep = (torch.rand(B, Cc, 1, device=x.device) > ch_drop).float()
    return x * scale * keep + noise * torch.randn_like(x)


class SSLModel(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.tcn = ModernTCNEncoder(C, d=64, patch=PATCH, blocks=2, large=25, small=5, dropout=0.1)
        self.q_proj = nn.Linear(64, d)
        self.enc = nn.ModuleList([EncoderLayer(d, 8, 1024, 0.1) for _ in range(2)])
        self.recon = nn.Linear(d, C * PATCH)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 128))

    def forward(self, x):
        h = self.q_proj(self.tcn(x))
        B, N, d = h.shape
        pos = sinusoidal(N, d, x.device)[None]
        for layer in self.enc:
            h = layer(h, pos, None)
        return h


def load_windows(exclude, win, per_file, seed):
    rng = np.random.RandomState(seed)
    xs, n_subj = [], set()
    for task in TASKS:
        for f in sorted((ROOT / "brainmosaic/work/eeg" / task).glob("*.pt")):
            if f.stem in exclude:
                continue
            d = torch.load(f, weights_only=False)
            L = d["lengths"].numpy()
            ok = np.where(L >= win)[0]
            pick = rng.choice(ok, size=min(per_file, len(ok)), replace=False)
            for i in pick:
                o = rng.randint(0, L[i] - win + 1)
                xs.append(d["eeg"][i, :, o:o + win].clone())
            n_subj.add(f.stem)
            del d
    return torch.stack(xs), n_subj


def nt_xent(a, b, tau):
    z = F.normalize(torch.cat([a, b]), dim=-1)
    sim = z @ z.T / tau
    n = len(a)
    sim.fill_diagonal_(-1e9)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", nargs="+", default=["ZAB", "ZJM", "ZKW", "ZKH"], help="held-out users (val/test)")
    ap.add_argument("--win_sec", type=float, default=4.0)
    ap.add_argument("--per_file", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--lambda_con", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "brainmosaic/runs/ssl_encoder/ssl.pth"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    torch.manual_seed(0)
    win = int(args.win_sec * 250) // PATCH * PATCH
    t0 = time.time()
    X, subj = load_windows(set(args.exclude), win, args.per_file, 0)
    print(f"{len(X)} unlabelled {args.win_sec:.0f}s windows from {len(subj)} people "
          f"(excluded {args.exclude}) | loaded in {time.time() - t0:.0f}s", flush=True)
    perm = np.random.RandomState(1).permutation(len(X))
    n_va = max(256, len(X) // 50)
    va, tr = perm[:n_va], perm[n_va:]

    model = SSLModel().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = len(tr) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    N = win // PATCH

    def step(xb, train=True):
        x1, x2 = augment(xb), augment(xb)
        mask = torch.rand(len(xb), N, device=dev) < args.mask_ratio
        xm = x1 * (~mask).repeat_interleave(PATCH, 1).unsqueeze(1).float()
        h1 = model(xm)
        rec = model.recon(h1).view(len(xb), N, C, PATCH).permute(0, 2, 1, 3).reshape(len(xb), C, N * PATCH)
        err = ((rec - xb) ** 2).mean(1).view(len(xb), N, PATCH).mean(-1)
        l_rec = (err * mask).sum() / mask.sum().clamp(min=1)
        h2 = model(x2)
        l_con = nt_xent(model.proj(h1.mean(1)), model.proj(h2.mean(1)), args.tau)
        return l_rec, l_con

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        np.random.shuffle(tr)
        sr = sc = 0.0
        for s in range(steps):
            xb = X[tr[s * args.batch_size:(s + 1) * args.batch_size]].to(dev).float()
            l_rec, l_con = step(xb)
            loss = l_rec + args.lambda_con * l_con
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            sr += l_rec.item()
            sc += l_con.item()
        model.eval()
        with torch.no_grad():
            vr = vc = 0.0
            nb = 0
            for s in range(0, len(va), args.batch_size):
                a, b = step(X[va[s:s + args.batch_size]].to(dev).float(), train=False)
                vr += a.item()
                vc += b.item()
                nb += 1
        print(f"  ep {ep:3d} | train recon {sr / steps:.4f} contrast {sc / steps:.3f} | val recon {vr / nb:.4f} "
              f"contrast {vc / nb:.3f} (masked-patch MSE; z-scored EEG has variance ~1) | {time.time() - t0:.0f}s",
              flush=True)
        torch.save({"model": {k: v for k, v in model.state_dict().items() if k.split(".")[0] in ("tcn", "q_proj", "enc")},
                    "args": vars(args)}, args.out)
    print(f"[ssl] saved encoder -> {args.out}")


if __name__ == "__main__":
    main()
