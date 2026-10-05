import json
import warnings

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

import config


def _montage(name, fallback=None):
    import mne

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return mne.channels.make_standard_montage(name).get_positions()["ch_pos"]
        except ValueError:
            return mne.channels.make_standard_montage(fallback).get_positions()["ch_pos"]


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class ChannelMapper:
    def __init__(self, labels, k=3):
        self.target = [c["label"] for c in json.load(open(config.CHANLOCS))]
        hcgsn = _montage("GSN-HydroCel-128")
        ten = {k_.lower(): v for k_, v in _montage("colin27_1005", "standard_1005").items()}
        tpos = _unit([hcgsn[t] if t in hcgsn else ten[t.lower()] for t in self.target])
        self.labels, src, missing = [], [], []
        for i, lab in enumerate(labels):
            key = lab.strip()
            if key in hcgsn:
                pos = hcgsn[key]
            elif key.lower() in ten:
                pos = ten[key.lower()]
            else:
                missing.append(lab)
                continue
            self.labels.append((i, key))
            src.append(pos)
        if not src:
            raise RuntimeError(f"none of the channel labels are known 10-05 or HCGSN names: {labels}")
        self.missing = missing
        spos = _unit(src)
        ang = np.arccos(np.clip(tpos @ spos.T, -1, 1))
        W = np.zeros((len(self.target), len(labels)), dtype=np.float32)
        k = min(k, len(src))
        cols = np.array([i for i, _ in self.labels])
        for t in range(len(self.target)):
            near = np.argsort(ang[t])[:k]
            if ang[t, near[0]] < 1e-3:
                W[t, cols[near[0]]] = 1.0
                continue
            w = 1.0 / ang[t, near] ** 2
            W[t, cols[near]] = w / w.sum()
        self.W = W
        self.sensors = [int(np.argmin(ang[:, j])) for j in range(len(src))]

    def __call__(self, x):
        return self.W @ x

    def describe(self):
        return {"measured": [l for _, l in self.labels], "unknown": self.missing, "sensors": self.sensors}


class OnlineFilter:

    def __init__(self, channels, fs, band=(1.0, 40.0), notch=50.0):
        sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
        if notch and notch < fs / 2:
            b, a = iirnotch(notch, 30, fs)
            sos = np.vstack([sos, tf2sos(b, a)])
        self.sos = sos
        self.zi = np.repeat(sosfilt_zi(sos)[:, None, :], channels, axis=1)
        self.primed = False

    def __call__(self, x):
        if not self.primed:
            self.zi = self.zi * x[None, :, :1]
            self.primed = True
        y, self.zi = sosfilt(self.sos, x, axis=-1, zi=self.zi)
        return y.astype(np.float32)
