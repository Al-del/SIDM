# Neurosteer

Closed-loop EEG → LLM answering. You ask a question; Qwen answers one sentence at a time. You read
each sentence while EEG is recorded, the RAG-Mosaic decoder turns that epoch into vectors in its text
space (ẑ + slot embeddings, 256-d Qwen3-Embedding-8B), a trained translator maps those vectors into
Qwen's input-embedding space, and they steer the next sentence. After 3 sentences, a yes/no pass asks Qwen whether the
answer is complete; "Max sentences" is the hard cap. A 4-gram blocker and a reuse penalty stop it copying earlier sentences.

```
 question ─► Qwen3-1.7B ──sentence──► reader ──EEG epoch (≤9 s)──► RAG-Mosaic ──ẑ, slot vectors (256-d)
                 ▲                                                                   │
                 │                                              translator 256 → 2048 (trained)
                 └── neural prefix · residual Δh (layer 14) · logit bias ◄───────────┘
```

## Translator (decoder space → Qwen space)

`backend/translator.py` trains two small MLP heads with Qwen frozen:
* sentence head: ẑ → 8 virtual tokens; trained so Qwen reconstructs the sentence from them
* word head: slot vector → 1 virtual token; trained so Qwen reconstructs the word

Training pairs come from the text assets (1,039 sentence embeddings, 5,100 word embeddings); inputs are
perturbed to cos ≈ 0.6–1.0 with the clean vector, the accuracy range of the EEG decoder. 10 % of the
sentences are held out; on them the true vector gives a clearly lower loss than a shuffled one, so the translated
vectors carry sentence-specific meaning (gist-level: sentiment, topic, sentence type). Train once (about 25 min on an M-series Mac):

```bash
cd backend && ../../.venv/bin/python translator.py      # -> backend/weights/translator.pt
```
Without the file, the app falls back to looking the decoded words up in Qwen's token embeddings.
The UI's "Qwen reads from ẑ" panel shows what frozen Qwen decodes from the translated ẑ alone.

## Run

```bash
./start.sh            # Flask on :5050, Next.js on :3000
```
Open http://localhost:3000, type a question, press **Ask**, press **SPACE** when you've read
each sentence (an epoch is capped at 9 s, the decoder's window).

Requirements: the Python venv at `../../.venv` (torch, transformers; `pip install -r backend/requirements.txt`),
Node 20+, and the trained model in the repository root (paths in `backend/config.py`,
overridable with `MOSAIC_DIR`, `BM_WORK`, `QWEN_MODEL`, `LLM_DEVICE`).

## How the EEG steers Qwen

Decoded units (top content words, weighted by slot confidence × cosine) are injected three ways,
each with its own slider:

| channel | where | what |
|---|---|---|
| **Neural prefix** | input embeddings, before the transformer (Qwen uses RoPE, so this is the "after positional encoding" injection point: position is applied inside attention) | translated ẑ (8 tokens) + translated slot vectors (1 token each), spliced into the prompt as "Neural context" |
| **Residual Δh** | hidden state of layer 14 / 28, last position | layer-14 state of the translated tokens minus a neutral-token baseline |
| **Logit bias** | output logits | additive boost on the units' first tokens; tokens it picked are underlined amber in the UI |
| *Text hint* (off by default) | prompt text | lists the units explicitly; strongest but least "neural" |

Calibrated on Qwen3-1.7B: residual ≤ 0.8 keeps grammar intact, ≥ 1.0 breaks it; the defaults are
deliberately subtle.

## Signal sources

* **REPLAY**: held-out ZuCo EEG (ZAB/ZJM/ZKW) streamed in real time. Real brain data, but recorded on
  *other* sentences, so the decoded units do not reflect what you read in the app.
* **SYNTH**: generated signal for testing the pipeline. Not brain-derived.
* **LSL**: a live headset over Lab Streaming Layer (`pip install pylsl`). Channels are matched by
  label to the 105 EGI HydroCel channels the decoder was trained on; unmatched channels are zero.
  A consumer 8–32 channel cap is far outside the training distribution.

The decoder's EEG-specific signal on new users is small, so treat steering from real EEG as a weak prior. Compare against the steering toggle off and
against SYNTH before drawing conclusions.

## Notes

* On Apple Silicon, Qwen3 runs in float32: bfloat16 + SDPA on MPS produces garbage in torch 2.12.
* API: `GET /api/eeg` (SSE stream), `POST /api/read/start|end`, `GET /api/generate` (SSE tokens),
  `POST /api/session`, `POST /api/settings`, `POST /api/source`, `GET /api/status`, `GET /api/montage`.
