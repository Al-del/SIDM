import json
import threading
import time

import numpy as np
import torch

import config


class RingBuffer:
    def __init__(self, channels, fs, seconds):
        self.fs = fs
        self.size = int(fs * seconds)
        self.data = np.zeros((channels, self.size), dtype=np.float32)
        self.total = 0
        self.lock = threading.Lock()

    def write(self, chunk):
        n = chunk.shape[1]
        with self.lock:
            idx = (self.total + np.arange(n)) % self.size
            self.data[:, idx] = chunk
            self.total += n

    def read(self, start, end):
        with self.lock:
            start = max(start, self.total - self.size)
            end = min(end, self.total)
            if end <= start:
                return np.zeros((self.data.shape[0], 0), dtype=np.float32)
            idx = np.arange(start, end) % self.size
            return self.data[:, idx].copy()

    def latest(self, n):
        return self.read(self.total - n, self.total)


def electrode_positions():
    locs = json.load(open(config.CHANLOCS))
    xyz = np.array([[c["X"], c["Y"], c["Z"]] for c in locs], dtype=np.float64)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    r = theta / (np.pi / 2)
    x, y = -r * np.sin(phi), r * np.cos(phi)
    return [c["label"] for c in locs], np.stack([x, y], 1)


class Source:
    kind = "base"
    label = ""
    brain_derived = False

    def __init__(self):
        self.fs = config.FS
        self.channels = config.N_CHANNELS
        self.reading = False

    def pull(self, n):
        raise NotImplementedError


class SyntheticSource(Source):
    kind = "synthetic"
    label = "SYNTHETIC GENERATOR"

    def __init__(self):
        super().__init__()
        _, pos = electrode_positions()
        rng = np.random.default_rng(7)
        centers = np.array([[0, -0.75], [0, 0.6], [-0.55, 0], [0.55, 0], [0, 0]])
        d = np.linalg.norm(pos[:, None] - centers[None], axis=-1)
        self.mix = np.exp(-(d ** 2) / 0.18)
        self.freqs = np.array([10.2, 6.1, 20.5, 18.0, 2.2])
        self.phase = rng.uniform(0, 2 * np.pi, len(self.freqs))
        self.t = 0
        self.pink = np.zeros(self.channels)
        self.rng = rng

    def pull(self, n):
        t = (self.t + np.arange(n)) / self.fs
        self.t += n
        alpha_gain = 0.45 if self.reading else 1.0
        amp = np.array([alpha_gain * 1.4, 0.6, 0.35, 0.35, 0.9])
        slow = 1 + 0.35 * np.sin(2 * np.pi * 0.11 * t)
        src = amp[:, None] * slow[None] * np.sin(2 * np.pi * self.freqs[:, None] * t[None] + self.phase[:, None])
        out = self.mix @ src
        white = self.rng.normal(0, 0.5, (self.channels, n))
        for i in range(n):
            self.pink = 0.97 * self.pink + white[:, i] * 0.25
            out[:, i] += self.pink + 0.15 * white[:, i]
        if self.rng.random() < 0.004:
            blink = np.exp(-((np.arange(n) - n / 2) ** 2) / 8.0) * 6
            out += self.mix[:, 1:2] * blink[None]
        return out.astype(np.float32)


class ReplaySource(Source):
    kind = "replay"
    brain_derived = True

    def __init__(self, subjects=None):
        super().__init__()
        subjects = subjects or config.REPLAY_SUBJECTS
        parts = []
        for s in subjects:
            d = torch.load(config.REPLAY_DIR / f"{s}.pt", weights_only=False)
            for e, L in zip(d["eeg"], d["lengths"]):
                parts.append(e[:, : int(L)].float().numpy())
        self.signal = np.concatenate(parts, 1)
        self.pos = 0
        self.label = f"ZUCO REPLAY · {'/'.join(subjects)}"

    def pull(self, n):
        idx = (self.pos + np.arange(n)) % self.signal.shape[1]
        self.pos += n
        return self.signal[:, idx]


class LSLSource(Source):
    kind = "lsl"
    brain_derived = True

    def __init__(self):
        super().__init__()
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("type", "EEG", timeout=5)
        if not streams:
            raise RuntimeError("no LSL stream of type EEG found")
        self.inlet = StreamInlet(streams[0], max_chunklen=64)
        info = self.inlet.info()
        self.fs = int(round(info.nominal_srate()))
        labels, _ = electrode_positions()
        names = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        where = {n: i for i, n in enumerate(names)}
        self.map = [(j, where[l]) for j, l in enumerate(labels) if l in where]
        self.coverage = len(self.map) / config.N_CHANNELS
        self.label = f"LSL · {info.name()} · {info.channel_count()}CH · {self.coverage:.0%} MAPPED"

    def pull(self, n):
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=n * 4)
        if not chunk:
            return np.zeros((self.channels, 0), dtype=np.float32)
        x = np.asarray(chunk, dtype=np.float32).T
        out = np.zeros((self.channels, x.shape[1]), dtype=np.float32)
        for j, i in self.map:
            out[j] = x[i]
        return out


SOURCES = {"synthetic": SyntheticSource, "replay": ReplaySource, "lsl": LSLSource}


class Acquisition:
    def __init__(self):
        self.source = None
        self.buffer = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self, kind):
        with self.lock:
            self.stop()
            src = SOURCES[kind]()
            self.source = src
            self.buffer = RingBuffer(src.channels, src.fs, config.BUFFER_SECONDS)
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
        self.thread = None

    def _loop(self):
        src, buf = self.source, self.buffer
        step = max(1, src.fs // 25)
        t_next = time.perf_counter()
        while self.running:
            chunk = src.pull(step)
            if chunk.shape[1]:
                buf.write(chunk)
            t_next += step / src.fs
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.perf_counter()

    def now(self):
        return self.buffer.total if self.buffer else 0

    def status(self):
        s = self.source
        if s is None:
            return {"kind": None}
        return {"kind": s.kind, "label": s.label, "fs": s.fs, "channels": s.channels,
                "brain_derived": s.brain_derived}
