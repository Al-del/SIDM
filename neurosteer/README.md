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

### Demo mode (no weights)

```bash
./start.sh --mock                      # or: NEUROSTEER_MOCK=1 ./start.sh
cd backend && python app.py --mock --port 5077   # backend only
```
A fake decoder (words picked from the epoch's band power) and a fake LLM (a canned, question-aware
answer streamed word by word, 4–5 sentences, the decoded word woven in and marked steered) replace
the models, so the whole loop runs on any laptop. `/api/status` reports `"mock": true`.
`NEUROSTEER_MOCK_DELAY` sets the per-word delay (default 0.04 s).

Tests run in mock mode: `cd backend && python -m pytest tests -q`.

Requirements: the conda env from the repository root (`conda env create -f environment.yml`, which installs
Python 3.12, Node and every Python package for the model and the app; `start.sh` uses it automatically), and the trained model in the repository root (paths in `backend/config.py`,
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
* **BA DIRECT**: a BrainAccess headset through the BrainAccess Python API, inside the backend. Needs the
  backend on Linux (the SDK ships `libbacore.so` for Linux x86-64 and a DLL for Windows; there is no macOS build).
* **LSL**: any EEG stream on Lab Streaming Layer. Channel labels may be HCGSN names (`E22`) or 10-05 names
  (`Fp1`, `O1`, ...); they are placed with MNE montages and interpolated onto the 105 channels the decoder was
  trained on (inverse angular distance, 3 nearest), after a causal 1–40 Hz band-pass and 50 Hz notch.

### BrainAccess headset

```bash
# on the Linux machine paired with the headset (Bluetooth serial, e.g. /dev/rfcomm0)
export BRAINACCESS_SDK=~/BrainAccessSDK-linux-classic      # folder with libbacore.so and bacore.json
pip install numpy pylsl
python backend/brainaccess_lsl_bridge.py --port /dev/rfcomm0
#   custom cap: --cap '{"0":"Fp1","1":"Fp2","2":"O1","3":"O2"}'   gain: --gain 8
```
Then pick **LSL** in the app on the Mac (same network). If the backend itself runs on that Linux machine,
pick **BA DIRECT** instead and set `BRAINACCESS_SDK`, `BRAINACCESS_PORT`, `BRAINACCESS_CAP`, `BRAINACCESS_GAIN`
as needed. Default caps: MINI = F3 F4 C3 C4 P3 P4 O1 O2 (the SDK's default), HALO = Fp1 Fp2 O1 O2; any other
model needs `BRAINACCESS_CAP`.

A consumer headset with 4–32 channels is far outside the decoder's 105-channel training data; interpolation
keeps the input shape right, not the information.

The decoder's EEG-specific signal on new users is small, so treat steering from real EEG as a weak prior. Compare against the steering toggle off and
against SYNTH before drawing conclusions.

## Notes

* On Apple Silicon, Qwen3 runs in float32: bfloat16 + SDPA on MPS produces garbage in torch 2.12.
* Backend flags: `python app.py [--port 5050] [--host 127.0.0.1] [--mock]` (env: `PORT`, `HOST`, `NEUROSTEER_MOCK`).

## API

All endpoints return JSON; errors are `{"error": ..., "status": code}` with a matching HTTP status.
SSE streams send a `: ping` comment every 15 s while idle.

| endpoint | what |
|---|---|
| `GET /api/health` | `{"ok", "uptime_s", "version"}` |
| `GET /api/status` | `mock`, EEG source, model info, load errors, settings, question, sentence count |
| `GET /api/montage` | electrode labels/positions, displayed channels, headset sensors |
| `POST /api/source` | `{"kind": "synthetic"\|"replay"\|"lsl"\|"brainaccess"}` |
| `GET /api/eeg` | SSE: `eeg` frames and `metrics` (RMS, band power) |
| `POST /api/session` | `{"question", "settings"?}` starts a new answer |
| `POST /api/reset` | clears the session, keeps settings → `{"ok": true}` |
| `POST /api/settings` | any subset of the settings, and/or `{"preset": name}`; values are type-checked and clamped |
| `GET /api/presets` | `subtle`, `balanced`, `strong`, `off` (residual ≤ 0.8 in all) |
| `GET /api/generate[?steer=0]` | SSE: `plan`, `token` (`text`, `steered`, `bias`), `done` (history entry); 409 while another generation runs |
| `POST /api/read/start`, `POST /api/read/end` | record the epoch while reading; `end` decodes it → units, slots, neighbours, `alignment` |
| `POST /api/readout` | what frozen Qwen reads from the translated ẑ |
| `GET /api/compare` | SSE: next sentence steered then baseline (`token`/`done` with `lane`), then `summary` (`overlap`, `steered_words`); history is not changed |
| `GET /api/history` | question and every sentence with its tokens, plan and decode |
| `GET /api/stats` | `sentences`, `mean_alignment`, `mean_latency_ms` (generation), `mean_decode_ms`, `steered_tokens`, `total_tokens` |
| `GET /api/export` | the session as a JSON download (`neurosteer-session-<timestamp>.json`) |

**Alignment** (0–1) measures how well a decode's units match the sentence just read: with the real
decoder, each unit's best cosine to the sentence's words in the decoder's text-embedding bank, rescaled
so the bank's mean cosine is 0, weighted by unit weight; without embeddings, the weighted share of
units whose word appears in the sentence.
