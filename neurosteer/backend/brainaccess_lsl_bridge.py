import argparse
import json
import time

from brainaccess_source import BrainAccessDevice


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="Bluetooth serial port (default /dev/rfcomm0 or COM4)")
    ap.add_argument("--cap", default=None, help='JSON {electrode: label}, e.g. {"0":"F3","1":"F4"}')
    ap.add_argument("--gain", type=int, default=None, help="amplifier gain 1/2/4/6/8/12/24")
    ap.add_argument("--name", default="BrainAccess")
    args = ap.parse_args()
    from pylsl import StreamInfo, StreamOutlet, local_clock

    dev = BrainAccessDevice(port=args.port, cap=json.loads(args.cap) if args.cap else None, gain=args.gain)
    info = StreamInfo(args.name, "EEG", len(dev.labels), dev.fs, "float32", f"brainaccess-{dev.serial}")
    chans = info.desc().append_child("channels")
    for label in dev.labels:
        c = chans.append_child("channel")
        c.append_child_value("label", label)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")
    info.desc().append_child_value("manufacturer", "Neurotechnology BrainAccess")
    outlet = StreamOutlet(info, chunk_size=25)
    dev.start()
    print(f"streaming BrainAccess {dev.model} #{dev.serial} on {dev.port}: {len(dev.labels)} ch "
          f"{dev.labels} @ {dev.fs} Hz, battery {dev.battery()}%  (Ctrl+C to stop)", flush=True)
    sent, t0 = 0, time.time()
    try:
        while True:
            x = dev.read()
            if x.shape[1]:
                outlet.push_chunk(x.T.tolist(), local_clock())
                sent += x.shape[1]
            if time.time() - t0 >= 5:
                print(f"  {sent / (time.time() - t0):.0f} samples/s", flush=True)
                sent, t0 = 0, time.time()
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()


if __name__ == "__main__":
    main()
