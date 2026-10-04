# EEG RAG-Mosaic: retrieval-augmented semantic decoding of EEG (ZuCo 1.0, new users)

This model decodes a set of **semantic units** (content words) from **sentence-level EEG** of a person it has
never seen in training. It extends BrainMosaic (Li et al., ICLR 2026, arXiv:2601.20447) with
**retrieval-augmented context**: annotated past EEG recordings of *other* people, retrieved by similarity,
are given to the decoder as extra input tokens.

```
model_EEG_RAG_and_est/
├── code/       training + architecture (snapshot of ROSEF_2026/brainmosaic)
├── weights/    trained weights + text-embedding assets
```

## Architecture

```
                      ┌────────────── frozen EEG fingerprint encoder (eeg_fingerprint_encoder.pth) ─┐
query EEG ───────────►│ ModernTCN → 2× Transformer → mean-pool → 256-d fingerprint                    │
[105 ch × 9 s, 250 Hz]└──────────────────────────────────────────────────────────────────┬──────────┘
      │                                                                                     │
      │   pass 1: RAG-Mosaic without context → predicted sentence embedding ẑ (text space)  │
      │                                                                                     ▼
      │   search over memory = past trials of OTHER users (each annotated with its unit set):
      │      score(m) = 0.5·z(cos(ẑ, text-emb of m's sentence)) + 0.5·z(cos(fingerprint_q, fingerprint_m))
      │      → top-5 neighbours  (same-subject trials are excluded)
      ▼                                                                                     │
┌────────────────────────── RAG-Mosaic (rag_mosaic_best_ep12.pth, 15.7 M params) ──────────┴──────────┐
│ query branch : ModernTCN (patch 25 = 100 ms, d=64, 2 blocks, kernels 25/5) → Linear 64→256          │
│                + sinusoidal position + type-emb → 2× Transformer encoder (d=256, 8 heads, FF 1024)   │
│ context branch: every unit of the 5 neighbours (≤20 each) → MLP(text-emb 256→256)                   │
│                + Linear(similarity) + rank-emb + type-emb                                            │
│ decoder      : 4× Transformer decoder, 40 slot queries (learned content + position),                │
│                cross-attending to [query tokens ; context tokens]                                    │
│ heads        : per slot → unit embedding (256-d) + active/empty logit; slot 0 = sentence embedding   │
└──────────────────────────────────────────────────────────────────────────────────────────────────────┘
      ▼
predicted set of unit embeddings → nearest units in the 4,175-unit bank (text_assets_qwen3emb8b.pt)
```

**Text space.** Units are content words (stop-words removed), each embedded as `"word: WordNet definition"`
with Qwen3-Embedding-8B (last-token pooling, MRL-truncated to 256-d, L2-normalised). Near-duplicates are
clustered (complete linkage, cosine ≥ 0.85), giving 4,175 units from 5,100 words over the 1,039 ZuCo-1 sentences.

## Training

| | |
|---|---|
| Data | ZuCo 1.0, task SR (movie reviews), sentence-level preprocessed EEG, 105 ch, 250 Hz, z-scored, padded/cropped to 9 s |
| Split | **new users**: train ZDM ZDN ZGW ZJN ZJS ZKB ZMG ZPH · validation ZKH · **test ZAB ZJM ZKW** (never seen) |
| Loss | BrainMosaic set loss: Hungarian matching (cost = 1·(−p_active) + 2·(1−cos)), cosine + 0.2·InfoNCE on matched slots, active/empty CE (no-object weight 0.3), 0.2·sentence-embedding cosine **+ 0.5 · contrastive InfoNCE** (predicted sentence embedding vs batch sentences, τ = 0.07) |
| Optimiser | AdamW, lr 2e-4 one-cycle, wd 0.05, batch 32, grad-clip 1.0, Gaussian input noise σ = 0.1 |
| Selection | epoch with the highest validation (UMA@0.7 − random-word UMA@0.7); early stop patience 10 → **epoch 12** |
| Fingerprint encoder | `retrieval_mosaic.py`: same EEG backbone, cross-subject supervised contrastive (same sentence ⇄ different people) + 0.5 · EEG→sentence-text InfoNCE |

## How to run

The code expects the preprocessed data of the original project (`ROSEF_2026/brainmosaic/work/`).
Set `BM_ROOT` to the folder that contains `brainmosaic/work`:

```bash
cd ~/ROSEF_2026/model_EEG_RAG_and_est/code
export BM_ROOT=~/ROSEF_2026
# evaluate the saved model on the 3 new users
../../.venv/bin/python rag_mosaic.py --eval_only --init_from ../weights/rag_mosaic_best_ep12.pth \
    --ret_ck ../weights/eeg_fingerprint_encoder.pth --out ./eval_out
# retrain from scratch (same settings as the saved model)
../../.venv/bin/python rag_mosaic.py --ret_ck ../weights/eeg_fingerprint_encoder.pth --out ./retrain_out
```
`retrieval_mosaic.py` and `rag_mosaic.py` load the unit bank from
`$BM_ROOT/brainmosaic/work/text/zuco1_SR+zuco1_NR+zuco1_TSR__Qwen3-Embedding-8B/text_assets.pt`,
which is the same file as `weights/text_assets_qwen3emb8b.pt`.

Data preparation from the raw ZuCo `.mat` files: `prepare_zuco.py` (EEG), `text_assets.py --model Qwen/Qwen3-Embedding-8B` (units).

## Neurosteer app

`neurosteer/` is a closed-loop demo built on this model: ask a question, Qwen3-1.7B answers one sentence at a time,
the EEG recorded while you read each sentence is decoded by RAG-Mosaic, translated into Qwen's embedding space and
used to steer the next sentence. Flask backend, Next.js frontend; see `neurosteer/README.md`.

## Files
| file | what |
|---|---|
| `weights/rag_mosaic_best_ep12.pth` | RAG-Mosaic weights (`model`) + training args |
| `weights/eeg_fingerprint_encoder.pth` | frozen EEG fingerprint encoder used for the search |
| `weights/text_assets_qwen3emb8b.pt` | unit bank, unit/sentence embeddings, sentence→units mapping |
