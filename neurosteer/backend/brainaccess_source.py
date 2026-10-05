import asyncio
import collections
import json
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SDK = Path(os.environ.get("BRAINACCESS_SDK", Path.home() / "Downloads" / "BrainAccessSDK-linux-classic")).expanduser()
DEFAULT_CAPS = {
    "MINI": {0: "F3", 1: "F4", 2: "C3", 3: "C4", 4: "P3", 5: "P4", 6: "O1", 7: "O2"},
    "HALO": {0: "Fp1", 1: "Fp2", 2: "O1", 3: "O2"},
}


@contextmanager
def _in_sdk():
    here = os.getcwd()
    os.chdir(SDK)
    try:
        yield
    finally:
        os.chdir(here)


def _import_sdk():
    if sys.platform == "darwin":
        raise RuntimeError("the BrainAccess SDK has no macOS library; run brainaccess_lsl_bridge.py on the Linux "
                           "or Windows machine paired with the headset and use the LSL source here")
    if not (SDK / "PythonAPI" / "brainaccess").exists():
        raise RuntimeError(f"BrainAccess SDK not found at {SDK} (set BRAINACCESS_SDK)")
    sys.path.insert(0, str(SDK / "PythonAPI"))
    with _in_sdk():
        import brainaccess.core as bacore
        import brainaccess.core.eeg_channel as eeg_channel
        from brainaccess.core.eeg_manager import EEGManager
        from brainaccess.core.gain_mode import multiplier_to_gain_mode
    return bacore, eeg_channel, EEGManager, multiplier_to_gain_mode


class BrainAccessDevice:

    def __init__(self, port=None, cap=None, gain=None):
        self.bacore, self.ch, EEGManager, to_gain = _import_sdk()
        with _in_sdk():
            self.bacore.init(self.bacore.Version(2, 0, 0))
        self.port = port or os.environ.get("BRAINACCESS_PORT") or ("COM4" if os.name == "nt" else "/dev/rfcomm0")
        self.gain = to_gain(int(gain or os.environ.get("BRAINACCESS_GAIN", 8)))
        self.mgr = EEGManager()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.queue = collections.deque()
        self.lock = threading.Lock()
        self.rows = []
        try:
            if not self._run(lambda: self.mgr.connect(self.port)):
                raise RuntimeError(f"could not connect to a BrainAccess device on {self.port}")
            info = self.mgr.get_device_info()[0]
            self.model = info.device_model.name
            self.serial = int(info.serial_number)
            self.fs = int(self.mgr.get_sample_frequency())
            cap = cap or (json.loads(os.environ["BRAINACCESS_CAP"]) if os.environ.get("BRAINACCESS_CAP") else None)
            cap = cap or DEFAULT_CAPS.get(self.model)
            if cap is None:
                raise RuntimeError(f"no default cap for BrainAccess {self.model}; set BRAINACCESS_CAP")
            self.cap = {int(k): v for k, v in cap.items()}
            self.labels = [self.cap[k] for k in sorted(self.cap)]
        except Exception:
            self.close()
            raise

    def _run(self, call, timeout=15):
        async def wrap():
            return await call()

        return asyncio.run_coroutine_threadsafe(wrap(), self.loop).result(timeout)

    def _on_chunk(self, chunk, size):
        x = np.asarray([chunk[r] for r in self.rows], dtype=np.float32)
        with self.lock:
            self.queue.append(x)

    def start(self):
        for e in sorted(self.cap):
            ch = self.ch.ELECTRODE_MEASUREMENT + e
            self.mgr.set_channel_enabled(ch, True)
            self.mgr.set_channel_gain(ch, self.gain)
        self.mgr.set_channel_enabled(self.ch.SAMPLE_NUMBER, True)
        self._run(self.mgr.start_stream)
        self.rows = [self.mgr.get_channel_index(self.ch.ELECTRODE_MEASUREMENT + e) for e in sorted(self.cap)]
        self.mgr.set_callback_chunk(self._on_chunk)

    def read(self):
        with self.lock:
            parts, self.queue = list(self.queue), collections.deque()
        if not parts:
            return np.zeros((len(self.labels), 0), dtype=np.float32)
        return np.concatenate(parts, 1)

    def battery(self):
        try:
            return int(self._run(self.mgr.get_full_battery_info, timeout=3).level)
        except Exception:
            return None

    def close(self):
        try:
            self.mgr.set_callback_chunk(None)
            if self.mgr.is_streaming():
                self._run(self.mgr.stop_stream)
            self.mgr.disconnect()
            self.mgr.destroy()
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        with _in_sdk():
            self.bacore.close()
