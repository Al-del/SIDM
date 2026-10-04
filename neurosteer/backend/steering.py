import re
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

import config

PLACEHOLDER = "<<NEURAL>>"
NEUTRAL = ["the", "of", "and", "a", "to", "in", "is", "it", "that", "was"]
END = "[END]"
SYSTEM = ("You answer the user's question one sentence at a time. Each reply is exactly ONE new sentence that "
          "continues the answer so far and adds something new. Stay on the question. No preamble, no lists, no "
          "markdown, never repeat an earlier sentence, never talk about these instructions or the context. "
          f"When the answer is complete, reply with exactly {END}")
TRANSLATOR = Path(__file__).resolve().parent / "weights" / "translator.pt"


class Steerer:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.dev = torch.device(config.LLM_DEVICE)
        self.tok = AutoTokenizer.from_pretrained(config.QWEN_MODEL)
        dtype = torch.float32 if self.dev.type in ("mps", "cpu") else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL, dtype=dtype).to(self.dev).eval()
        self.model.requires_grad_(False)
        self.E = self.model.get_input_embeddings().weight
        self.emb_norm = float(self.E[:20000].float().norm(dim=-1).mean())
        self.layers = self.model.model.layers
        self.layer = len(self.layers) // 2
        self.lock = threading.Lock()
        self._neutral = None
        self._neutral_frame = None
        self.T = self.frame = None
        if TRANSLATOR.exists():
            from translator import Frame, Translator

            ck = torch.load(TRANSLATOR, weights_only=False)
            self.T = Translator(ck["d_in"], ck["d_llm"], ck["n_sent"], emb_norm=ck["emb_norm"]).to(self.dev).eval()
            self.T.load_state_dict(ck["model"])
            self.frame = Frame(self.tok, self.model.get_input_embeddings(), self.dev)

    def info(self):
        cfg = self.model.config
        return {"name": config.QWEN_MODEL, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                "steer_layer": self.layer, "device": str(self.dev), "translator": self.T is not None}

    def _ids(self, text):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(self.dev)

    @torch.no_grad()
    def _word_vec(self, word):
        return self.E[self._ids(" " + word)].float().mean(0)

    @torch.no_grad()
    def _hidden(self, words):
        ctx = self._ids("I keep thinking about")
        vecs = []
        for w in words:
            ids = torch.cat([ctx, self._ids(" " + w)])
            h = self.model(input_ids=ids[None], output_hidden_states=True).hidden_states[self.layer + 1]
            vecs.append(h[0, len(ctx):].float().mean(0))
        return torch.stack(vecs)

    @torch.no_grad()
    def _frame_hidden(self, prefix):
        f = self.frame
        x = torch.cat([f.A, prefix.to(f.A.dtype), f.B])[None]
        h = self.model(inputs_embeds=x, output_hidden_states=True).hidden_states[self.layer + 1]
        return h[0, len(f.A):len(f.A) + len(prefix)].float()

    @torch.no_grad()
    def translate(self, decode):
        z = torch.tensor(decode["z"], device=self.dev)[None]
        sent = self.T.sentence(F.normalize(z, dim=-1))[0]
        sv = decode.get("slot_vecs") or []
        if sv:
            u = F.normalize(torch.tensor([v["vec"] for v in sv], device=self.dev), dim=-1)
            words = self.T.words(u)
        else:
            words = sent[:0]
        return sent, words, [v["word"] for v in sv], [v["p_active"] for v in sv]

    @torch.no_grad()
    def plan(self, decode, s):
        units = decode["units"][: s["prefix_tokens"]]
        if not units:
            return None
        w = torch.tensor([u["weight"] for u in units], device=self.dev)
        w = w / w.max()
        plan = {"units": units, "weights": w.tolist(), "prefix": None, "labels": [], "bias": {}, "residual": None,
                "mode": "translator" if self.T is not None else "lookup"}
        if self.T is not None:
            sent, words, names, pa = self.translate(decode)
            n = s["prefix_tokens"]
            translated = torch.cat([sent, words[:n]])
            labels = [f"z{i + 1}" for i in range(len(sent))] + names[:n]
            if s["prefix"] > 0:
                plan["prefix"], plan["labels"] = translated * s["prefix"], labels
            if s["residual"] > 0:
                if self._neutral_frame is None:
                    neutral = torch.stack([self._word_vec(t) for t in NEUTRAL])
                    neutral = F.normalize(neutral, dim=-1) * self.T.emb_norm
                    self._neutral_frame = self._frame_hidden(neutral)
                v = self._frame_hidden(translated).mean(0) - self._neutral_frame.mean(0)
                scale = float(self._neutral_frame.norm(dim=-1).mean())
                plan["residual"] = F.normalize(v, dim=0) * scale * s["residual"]
        else:
            if s["prefix"] > 0:
                vecs = torch.stack([F.normalize(self._word_vec(u["word"]), dim=0) for u in units])
                plan["prefix"] = vecs * self.emb_norm * (0.5 + 0.5 * w[:, None]) * s["prefix"]
                plan["labels"] = [u["word"] for u in units]
            if s["residual"] > 0:
                if self._neutral is None:
                    self._neutral = self._hidden(NEUTRAL)
                h = self._hidden([u["word"] for u in units])
                v = (w[:, None] * h).sum(0) / w.sum() - self._neutral.mean(0)
                scale = float(self._neutral.norm(dim=-1).mean())
                plan["residual"] = F.normalize(v, dim=0) * scale * s["residual"]
        if s["bias"] > 0:
            for u, wi in zip(units, w.tolist()):
                for form in (" " + u["word"], " " + u["word"].capitalize(), u["word"]):
                    tid = self._ids(form)[0].item()
                    plan["bias"][tid] = max(plan["bias"].get(tid, 0.0), s["bias"] * wi)
        return plan

    @torch.no_grad()
    def readout(self, decode, max_new=40):
        if self.T is None:
            return None
        from translator import reconstruct

        with self.lock:
            sent, _, _, _ = self.translate(decode)
            return reconstruct(self.model, self.frame, sent[None], max_new)[0]

    def _prompt(self, question, history, plan, s):
        text = " ".join(history) if history else "(nothing yet: write the first sentence)"
        user = ""
        if plan is not None and plan["prefix"] is not None:
            user += f"Context: {PLACEHOLDER}\n\n"
        user += f"Question: {question}\n\nAnswer so far:\n{text}\n"
        if plan is not None and s["hint"]:
            words = ", ".join(u["word"] for u in plan["units"])
            user += f"\nWeave in, where it fits: {words}\n"
        user += f"\nWrite the next sentence of the answer, or {END} if the answer is complete."
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def complete(self, question, history):
        if len(history) < 3:
            return 0.0
        msgs = [{"role": "user", "content": f"Question: {question}\n\nAnswer: {' '.join(history)}\n\n"
                 "Does this answer already fully answer the question, so that another sentence would only repeat "
                 "it? Reply yes or no."}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        logits = self.model(input_ids=self._ids(p)[None]).logits[0, -1].float()
        yes = torch.logsumexp(logits[[int(self._ids(w)[0]) for w in ("yes", "Yes")]], 0)
        no = torch.logsumexp(logits[[int(self._ids(w)[0]) for w in ("no", "No")]], 0)
        return float(torch.sigmoid(yes - no))

    @torch.no_grad()
    def generate(self, question, history, plan, s):
        with self.lock:
            p_done = self.complete(question, history)
            if p_done > 0.5:
                yield {"done": True, "sentence": "", "end": True, "p_complete": round(p_done, 3)}
                return
            prompt = self._prompt(question, history, plan, s)
            emb_layer = self.model.get_input_embeddings()
            if PLACEHOLDER in prompt:
                a, b = prompt.split(PLACEHOLDER)
                emb = torch.cat([emb_layer(self._ids(a)), plan["prefix"].to(emb_layer.weight.dtype),
                                 emb_layer(self._ids(b))])[None]
            else:
                emb = emb_layer(self._ids(prompt))[None]
            hook = None
            if plan is not None and plan["residual"] is not None:
                vec = plan["residual"]

                def add(_m, _i, out):
                    h = (out[0] if isinstance(out, tuple) else out).clone()
                    h[:, -1:] += vec.to(h.dtype)
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

                hook = self.layers[self.layer].register_forward_hook(add)
            bias = plan["bias"] if plan is not None else {}
            bias_ids = torch.tensor(list(bias.keys()), device=self.dev, dtype=torch.long)
            bias_val = torch.tensor(list(bias.values()), device=self.dev)
            n = 4
            seen, openings, used = {}, {}, set()
            for h in history:
                hid = self._ids(h).tolist()
                for i in range(len(hid) - n + 1):
                    seen.setdefault(tuple(hid[i:i + n - 1]), set()).add(hid[i + n - 1])
                if len(hid) >= 2:
                    openings.setdefault(hid[0], set()).add(hid[1])
                used.update(t for t in hid if len(self.tok.decode([t]).strip()) > 3)
            used = torch.tensor(sorted(used), device=self.dev, dtype=torch.long)
            try:
                out = self.model(inputs_embeds=emb, use_cache=True)
                past, logits = out.past_key_values, out.logits[0, -1].float()
                ids, prev = [], ""
                for _ in range(s["max_tokens"]):
                    for t in set(ids[-24:]):
                        logits[t] = logits[t] / 1.15 if logits[t] > 0 else logits[t] * 1.15
                    if len(bias_ids):
                        logits[bias_ids] += bias_val
                    if len(used):
                        logits[used] -= 0.8
                    if len(ids) == 1 and ids[0] in openings:
                        logits[list(openings[ids[0]])] = -float("inf")
                    if len(ids) >= n - 1:
                        banned = seen.get(tuple(ids[-(n - 1):]))
                        if banned:
                            logits[list(banned)] = -float("inf")
                    probs = torch.softmax(logits / s["temperature"], -1)
                    sp, si = probs.sort(descending=True)
                    keep = sp.cumsum(0) - sp < 0.9
                    nxt = int(si[keep][torch.multinomial(sp[keep] / sp[keep].sum(), 1)])
                    if nxt in (self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")):
                        break
                    ids.append(nxt)
                    if len(ids) >= n:
                        seen.setdefault(tuple(ids[-n:-1]), set()).add(ids[-1])
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    delta, prev = text[len(prev):], text
                    if delta:
                        yield {"text": delta, "steered": nxt in bias, "bias": round(bias.get(nxt, 0.0), 3)}
                    if len(ids) > 3 and re.search(r"[.!?][\"')\]]?\s*$", text):
                        break
                    out = self.model(input_ids=torch.tensor([[nxt]], device=self.dev), past_key_values=past,
                                     use_cache=True)
                    past, logits = out.past_key_values, out.logits[0, -1].float()
            finally:
                if hook is not None:
                    hook.remove()
            sentence = self.tok.decode(ids, skip_special_tokens=True).strip()
            yield {"done": True, "sentence": sentence, "end": sentence.startswith(END[:4]) or not sentence,
                   "p_complete": round(p_done, 3)}
