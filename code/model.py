import math

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(cin, cout, k, groups):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, groups=groups, bias=False), nn.BatchNorm1d(cout))


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, channels, large, small):
        super().__init__()
        self.large = conv_bn(channels, channels, large, channels)
        self.small = conv_bn(channels, channels, small, channels)

    def forward(self, x):
        return self.large(x) + self.small(x)


class ModernTCNBlock(nn.Module):

    def __init__(self, large, small, d, dff, m, drop):
        super().__init__()
        self.dw = ReparamLargeKernelConv(m * d, large, small)
        self.norm = nn.BatchNorm1d(d)
        self.ffn1 = nn.Sequential(nn.Conv1d(m * d, m * dff, 1, groups=m), nn.Dropout(drop), nn.GELU(),
                                  nn.Conv1d(m * dff, m * d, 1, groups=m), nn.Dropout(drop))
        self.ffn2 = nn.Sequential(nn.Conv1d(m * d, m * dff, 1, groups=d), nn.Dropout(drop), nn.GELU(),
                                  nn.Conv1d(m * dff, m * d, 1, groups=d), nn.Dropout(drop))

    def forward(self, x):
        B, M, D, N = x.shape
        h = self.dw(x.reshape(B, M * D, N))
        h = self.norm(h.reshape(B * M, D, N)).reshape(B, M * D, N)
        h = self.ffn1(h).reshape(B, M, D, N)
        h = h.permute(0, 2, 1, 3).reshape(B, D * M, N)
        h = self.ffn2(h).reshape(B, D, M, N).permute(0, 2, 1, 3)
        return x + h


class ModernTCNEncoder(nn.Module):

    def __init__(self, in_channels, d=64, patch=25, blocks=2, large=25, small=5, ffn_ratio=2.0,
                 stem_dim=None, dropout=0.0):
        super().__init__()
        m = stem_dim or in_channels
        self.patch = patch
        self.stem = nn.Sequential(nn.Conv1d(in_channels, m, 3, padding=1, bias=False), nn.BatchNorm1d(m), nn.ReLU())
        self.patch_embed = nn.Sequential(nn.Conv1d(1, d, patch, stride=patch), nn.BatchNorm1d(d))
        dff = int(d * ffn_ratio)
        self.blocks = nn.Sequential(*[ModernTCNBlock(large, small, d, dff, m, dropout) for _ in range(blocks)])
        self.mix_vars = nn.Conv2d(m, 1, 1, bias=False)
        self.out_dim = d

    def forward(self, x):
        B = x.shape[0]
        x = self.stem(x)
        M, T = x.shape[1], x.shape[2]
        x = self.patch_embed(x.reshape(B * M, 1, T))
        x = x.reshape(B, M, x.shape[1], x.shape[2])
        x = self.blocks(x)
        x = self.mix_vars(x).squeeze(1)
        return x.transpose(1, 2)


class LaBraMEncoder(nn.Module):

    def __init__(self, ckpt, channel_map, drop_path=0.1):
        super().__init__()
        import json
        import sys
        from functools import partial
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party/labram"))
        from modeling_finetune import NeuralTransformer
        cmap = json.load(open(channel_map))
        self.register_buffer("chan_idx", torch.tensor(cmap["egi_index"]), persistent=False)
        self.input_chans = cmap["labram_input_chans"]
        self.net = NeuralTransformer(patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4,
                                     qk_norm=partial(nn.LayerNorm, eps=1e-6), norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                     init_values=0.1, qkv_bias=False, drop_path_rate=drop_path, num_classes=0)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        sd = {k[len("student."):]: v for k, v in sd.items() if k.startswith("student.")}
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        print(f"  LaBraM loaded: {len(sd)} tensors, missing {len(missing)} {missing[:4]}, unexpected {len(unexpected)}")
        self.out_dim = 200
        self.patch = 200

    def forward(self, x):
        x = x[:, self.chan_idx]
        B, C, T = x.shape
        A = T // self.patch
        x = x[..., :A * self.patch].reshape(B, C, A, self.patch)
        return self.net.forward_features(x, input_chans=self.input_chans, return_patch_tokens=True)


class EncoderLayer(nn.Module):
    def __init__(self, d, nhead, dff, drop):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.ReLU(), nn.Dropout(drop), nn.Linear(dff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.d1, self.d2 = nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x, pos, pad_mask):
        q = k = x + pos
        x = self.n1(x + self.d1(self.attn(q, k, x, key_padding_mask=pad_mask)[0]))
        return self.n2(x + self.d2(self.ff(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, dff, drop):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, dff), nn.ReLU(), nn.Dropout(drop), nn.Linear(dff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.d1, self.d2, self.d3 = nn.Dropout(drop), nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, tgt, memory, qpos, pos, pad_mask):
        q = k = tgt + qpos
        tgt = self.n1(tgt + self.d1(self.self_attn(q, k, tgt)[0]))
        tgt = self.n2(tgt + self.d2(self.cross_attn(tgt + qpos, memory + pos, memory, key_padding_mask=pad_mask)[0]))
        return self.n3(tgt + self.d3(self.ff(tgt)))


def sinusoidal(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class BrainMosaic(nn.Module):
    def __init__(self, cfg, in_channels, text_dim, n_attr=(4, 2, 5)):
        super().__init__()
        m = cfg["model"]
        d = m["hidden_dim"]
        self.encoder_type = m.get("encoder", "moderntcn")
        if self.encoder_type == "labram":
            from pathlib import Path
            root = Path(__file__).resolve().parent.parent
            self.backbone = LaBraMEncoder(root / m["labram_ckpt"], root / m["labram_channel_map"], m.get("drop_path", 0.1))
        else:
            self.backbone = ModernTCNEncoder(in_channels, d=m["tcn_size"], patch=m["patch"], blocks=m["tcn_blocks"],
                                             large=m["tcn_large_kernel"], small=m["tcn_small_kernel"],
                                             ffn_ratio=m["tcn_ffn_ratio"], stem_dim=m.get("tcn_stem_dim"),
                                             dropout=m["tcn_dropout"])
        if m.get("freeze_backbone"):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.input_proj = nn.Linear(self.backbone.out_dim, d)
        self.use_pos = m.get("use_pos_embed", True)
        nhead, dff, drop = m["nheads"], m["dim_feedforward"], m["dropout"]
        self.encoder = nn.ModuleList([EncoderLayer(d, nhead, dff, drop) for _ in range(m["enc_layers"])])
        self.decoder = nn.ModuleList([DecoderLayer(d, nhead, dff, drop) for _ in range(m["dec_layers"])])
        self.dec_norm = nn.LayerNorm(d)
        self.num_queries = m["num_queries"]
        self.query_embed = nn.Embedding(self.num_queries, d)
        self.query_as_tgt = m.get("query_content", True)
        self.query_content = nn.Embedding(self.num_queries, d) if self.query_as_tgt else None
        self.slot_dropout_p = m.get("slot_dropout_p", 0.0)
        self.patch = m["patch"]

        self.class_head = nn.Linear(d, 2)
        self.emb_head = nn.Linear(d, text_dim)
        self.attr_heads = nn.ModuleList([nn.Linear(d, n) for n in n_attr])

        for name, p in self.named_parameters():
            if p.dim() > 1 and not name.startswith("backbone") and not name.startswith("query_"):
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.query_embed.weight, std=m.get("query_init_std", 0.2))
        if self.query_as_tgt:
            nn.init.normal_(self.query_content.weight, std=1.0)

    def forward(self, eeg, lengths=None):
        tokens = self.input_proj(self.backbone(eeg))
        B, N, d = tokens.shape
        pad_mask = None
        if self.encoder_type == "labram":
            A = eeg.shape[-1] // 200
            if lengths is not None:
                a_valid = torch.clamp((lengths + 199) // 200, min=1, max=A)
                pad_mask = (torch.arange(A, device=eeg.device)[None] >= a_valid[:, None]).repeat(1, N // A)
            pos = torch.zeros(1, N, d, device=eeg.device)
        else:
            if lengths is not None:
                n_valid = torch.clamp((lengths + self.patch - 1) // self.patch, min=1, max=N)
                pad_mask = torch.arange(N, device=eeg.device)[None] >= n_valid[:, None]
            pos = sinusoidal(N, d, eeg.device)[None] if self.use_pos else torch.zeros(1, N, d, device=eeg.device)

        mem = tokens
        for layer in self.encoder:
            mem = layer(mem, pos, pad_mask)

        qpos = self.query_embed.weight
        if self.training and self.slot_dropout_p > 0:
            keep = torch.rand(qpos.size(0), device=qpos.device) > self.slot_dropout_p
            keep[0] = True
            qpos = qpos * keep.float().unsqueeze(1)
        qpos = qpos.unsqueeze(0).expand(B, -1, -1)
        hs = self.query_content.weight.unsqueeze(0).expand(B, -1, -1) if self.query_as_tgt else torch.zeros_like(qpos)
        for layer in self.decoder:
            hs = layer(hs, mem, qpos, pos, pad_mask)
        hs = self.dec_norm(hs)

        emb = self.emb_head(hs)
        out = {
            "pred_logits": self.class_head(hs),
            "pred_embeddings": emb,
            "sentence_embedding": emb[:, 0],
        }
        for name, head in zip(("sentence_mode", "subjectivity", "semantic_focus"), self.attr_heads):
            out[name] = head(hs[:, 0])
        return out
