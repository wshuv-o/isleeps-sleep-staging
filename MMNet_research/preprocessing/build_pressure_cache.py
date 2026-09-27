"""Nasal-pressure airflow per epoch -> data/pressure_flow/SN<k>.npz, for the Huttunen port.

iSLEEPS records two airflow sensors: "Flow Th" (oronasal thermal), which the
multimodal cache carries as its Flow channel, and "Pressure Flow" (nasal pressure),
which it does not. Huttunen et al.'s Model 3 reads both, so this extracts the second
with build_multimodal's own reader (same 25 Hz resampling, same epoch alignment to
processed7), and stores it at the resolution the multimodal cache uses.

    pflow [n, 750] float16, per-recording z-scored later by the consumer
    valid bool      False where the channel is absent or flat (stored as zeros)

  D:/EEG-TransNet/testenv/python.exe MMNet_research/preprocessing/build_pressure_cache.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
sys.path.insert(0, HERE)
import build_multimodal as B          # noqa: E402

OUT = os.path.join(DATA, "pressure_flow")


def main():
    os.makedirs(OUT, exist_ok=True)
    B.CARD = [("PressureFlow", ["Pressure Flow"])]
    edf = {}
    for p in (glob.glob(os.path.join(DATA, "Dataset", "**", "*.edf"), recursive=True)
              + glob.glob(os.path.join(DATA, "zenodo", "*.edf"))):
        edf.setdefault(B.sid_of(p), p)
    done = missing = 0
    for p7 in sorted(glob.glob(os.path.join(DATA, "processed7", "SN*.npz")),
                     key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = os.path.basename(p7)[:-4]
        dst = os.path.join(OUT, sid + ".npz")
        if os.path.exists(dst) or sid not in edf:
            if sid not in edf:
                print("[no edf] %s" % sid)
            continue
        n = len(np.load(p7, allow_pickle=True)["y"])
        x, valid = B.read_card(edf[sid], n)
        np.savez_compressed(dst, pflow=x[:, 0].astype(np.float16), valid=bool(valid[0]))
        done += 1; missing += int(not valid[0])
        print("[ok] %s %d epochs, pressure flow %s" % (sid, n, "present" if valid[0] else "ABSENT"),
              flush=True)
    print("wrote %d recordings, %d without the channel -> %s" % (done, missing, OUT))


if __name__ == "__main__":
    main()
