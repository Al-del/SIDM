import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


@torch.no_grad()
def hungarian_match(outputs, targets, cost_class=1.0, cost_emb=2.0):
    prob = outputs["pred_logits"].softmax(-1)[..., 1]
    pred = F.normalize(outputs["pred_embeddings"], dim=-1)
    costs = []
    for b, t in enumerate(targets):
        tgt = t["unit_embeddings"]
        c = cost_class * -prob[b].unsqueeze(1) + cost_emb * (1 - pred[b] @ tgt.T)
        c[0] = 1e6
        costs.append(c)
    costs = [c.float().cpu() for c in costs]
    out = []
    for c in costs:
        r, k = linear_sum_assignment(c.numpy())
        out.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(k, dtype=torch.long)))
    return out


def class_weights(counts):
    counts = torch.as_tensor(counts, dtype=torch.float32).clamp(min=1)
    inv = 1.0 / counts
    return inv / inv.sum() * len(counts)


class SetCriterion(nn.Module):
    def __init__(self, cfg, bank, attr_counts=None):
        super().__init__()
        L = cfg["loss"]
        self.L = L
        self.register_buffer("bank", F.normalize(bank, dim=-1))
        self.register_buffer("empty_weight", torch.tensor([L["eos_coef"], 1.0]))
        self.attr_names = ("sentence_mode", "subjectivity", "semantic_focus")
        for name, n in zip(self.attr_names, (4, 2, 5)):
            w = class_weights(attr_counts[name]) if attr_counts else torch.ones(n)
            self.register_buffer(f"w_{name}", w)

    def forward(self, outputs, targets):
        L = self.L
        indices = hungarian_match(outputs, targets, L["cost_class"], L["cost_emb"])
        logits, pred = outputs["pred_logits"], outputs["pred_embeddings"]
        B, K, _ = logits.shape
        dev = logits.device
        losses = {}

        tcls = torch.zeros(B, K, dtype=torch.long, device=dev)
        for b, (si, _) in enumerate(indices):
            tcls[b, si.to(dev)] = 1
        losses["cls"] = L["lambda_cls"] * F.cross_entropy(logits.transpose(1, 2), tcls, weight=self.empty_weight)

        z = torch.cat([pred[b, si.to(dev)] for b, (si, _) in enumerate(indices)])
        y = torch.cat([targets[b]["unit_embeddings"][ti.to(dev)] for b, (_, ti) in enumerate(indices)])
        cid = torch.cat([targets[b]["unit_cids"][ti.to(dev)] for b, (_, ti) in enumerate(indices)])
        z = F.normalize(z, dim=-1)
        losses["cos"] = L["lambda_cos"] * (1 - (z * y).sum(-1)).mean()
        if L["lambda_infonce"] > 0 and len(z) > 1:
            sim = z @ y.T / L["tau"]
            same = (cid[:, None] == cid[None, :]) & ~torch.eye(len(cid), dtype=torch.bool, device=dev)
            sim = sim.masked_fill(same, float("-inf"))
            losses["infonce"] = L["lambda_infonce"] * F.cross_entropy(sim, torch.arange(len(z), device=dev))

        if L.get("lambda_rep", 0) > 0:
            free = tcls == 0
            free[:, 0] = False
            if free.any():
                zf = F.normalize(pred[free], dim=-1)
                losses["rep"] = L["lambda_rep"] * F.relu(zf @ self.bank.T - L["rep_margin"]).mean()

        s_hat = F.normalize(outputs["sentence_embedding"], dim=-1)
        s = torch.stack([t["sentence_embedding"] for t in targets])
        losses["sent"] = L["lambda_global"] * (1 - (s_hat * s).sum(-1)).mean()
        for name in self.attr_names:
            tgt = torch.tensor([t[name] for t in targets], device=dev)
            losses[name] = L["lambda_global"] * L["lambda_attr"] * F.cross_entropy(
                outputs[name], tgt, weight=getattr(self, f"w_{name}"))

        return losses, indices
