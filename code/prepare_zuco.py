import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
TASK_DIRS = {
    "zuco1_SR": ROOT / "data/zuco1/task1-SR/Matlab_files",
    "zuco1_NR": ROOT / "data/zuco1/task2-NR/Matlab_files",
    "zuco1_TSR": ROOT / "data/zuco1/task3-TSR/Matlab_files",
    "zuco2_NR": ROOT / "datasets/ZuCo2/osfstorage/task1 - NR/Matlab files",
    "zuco2_TSR": ROOT / "datasets/ZuCo2/osfstorage/task2 - TSR/Matlab files",
}
SRC_FS = 500


def clean_sentence(s):
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s.replace("emp11111ty", "empty")


def iter_zuco1(path):
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    for s in np.atleast_1d(m["sentenceData"]):
        raw = getattr(s, "rawData", None)
        if not isinstance(raw, np.ndarray) or raw.ndim != 2 or raw.size == 0:
            yield clean_sentence(s.content), None
            continue
        yield clean_sentence(s.content), raw.astype(np.float32)


def iter_zuco2(path):
    import h5py
    with h5py.File(path, "r") as f:
        sd = f["sentenceData"]
        for i in range(sd["content"].shape[0]):
            text = "".join(chr(int(c)) for c in f[sd["content"][i][0]][()].flatten())
            raw = f[sd["rawData"][i][0]][()]
            if raw.ndim != 2 or raw.size < 10:
                yield clean_sentence(text), None
                continue
            yield clean_sentence(text), raw.T.astype(np.float32)


def process_file(job):
    try:
        return _process_file(job)
    except Exception as e:
        return f"[error] {job[0]}/{job[1].name}: {type(e).__name__}: {e}"


def _process_file(job):
    task, path, out_path, fs, max_sec, clip, n_channels, scale = job
    reader = iter_zuco2 if task.startswith("zuco2") else iter_zuco1
    T = int(round(max_sec * fs))
    g = gcd(SRC_FS, fs)
    up, down = fs // g, SRC_FS // g
    eegs, lengths, sentences, skipped = [], [], [], 0
    for text, raw in reader(path):
        if raw is None or not text or raw.shape[0] != n_channels or raw.shape[1] < SRC_FS // 2:
            skipped += 1
            continue
        x = np.nan_to_num(raw, nan=0.0)
        if up != down:
            x = resample_poly(x, up, down, axis=-1).astype(np.float32)
        x = x[:, :T]
        mu = x.mean(axis=1, keepdims=True)
        if scale is None:
            sd = x.std(axis=1, keepdims=True) + 1e-6
            x = np.clip((x - mu) / sd, -clip, clip)
        else:
            x = np.clip((x - mu) * scale, -clip, clip)
        L = x.shape[1]
        buf = np.zeros((x.shape[0], T), dtype=np.float16)
        buf[:, :L] = x
        eegs.append(buf)
        lengths.append(L)
        sentences.append(text)
    if not eegs:
        return f"{task}/{path.name}: no usable sentences"
    torch.save(
        {
            "eeg": torch.from_numpy(np.stack(eegs)),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "sentences": sentences,
            "fs": fs,
        },
        out_path,
    )
    return f"{task}/{out_path.stem}: {len(eegs)} sentences, skipped {skipped}, C={eegs[0].shape[0]}, T={T}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["zuco1_SR", "zuco1_NR", "zuco1_TSR"], choices=list(TASK_DIRS))
    ap.add_argument("--out", default=str(ROOT / "brainmosaic/work"))
    ap.add_argument("--fs", type=int, default=250, help="target sampling rate (source is 500 Hz)")
    ap.add_argument("--max_sec", type=float, default=9.0, help="crop/pad length; ~95th pct of ZuCo1 SR reading time")
    ap.add_argument("--clip", type=float, default=10.0, help="clip z-scored values to +-clip")
    ap.add_argument("--scale", type=float, default=None,
                    help="skip z-scoring; demean and multiply by this (LaBraM: 0.01 for uV/100)")
    ap.add_argument("--dir_name", default="eeg", help="output sub-folder of --out (e.g. eeg_labram)")
    ap.add_argument("--channels", type=int, default=105, help="ZuCo EEG channels after preprocessing")
    ap.add_argument("--workers", type=int, default=4, help="each worker holds one ~1 GB .mat in RAM")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = []
    for task in args.tasks:
        out_dir = Path(args.out) / args.dir_name / task
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(TASK_DIRS[task].glob("results*.mat"))
        if not files:
            print(f"[warn] no .mat files in {TASK_DIRS[task]}")
        for p in files:
            subj = p.stem.replace("results", "").split("_")[0]
            out_path = out_dir / f"{subj}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"[skip] {out_path} exists")
                continue
            jobs.append((task, p, out_path, args.fs, args.max_sec, args.clip, args.channels, args.scale))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(process_file, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
