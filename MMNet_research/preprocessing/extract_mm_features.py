"""
extract_mm_features.py -- one feature cache for the multimodal model.

Per subject -> data/mm_features/SN<k>.npz:
    Feeg  [n, 188]  engineered EEG/EOG/EMG features (spectral, Hjorth, spindle, SW, EOG, EMG)
    Fcard [n, 14]   engineered cardiorespiratory features (SpO2 desat, HR/var, airflow,
                    effort amplitude, thoraco-abdominal asynchrony)
    y     [n]       sleep stage 0..4
    apnea [n]       respiratory-event label
    cvalid[7]       which cardio channels are real

The EEG features are the SAME 188 that give the ensemble 0.746 (regenerated from the
signal, since featseq_cache was cleared). The cardio features are the new information.
"""
import os, sys, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "processing"))
sys.path.insert(0, os.path.join(ROOT, "extra"))
from features_v2 import extract_features_v2  # noqa
from cardio_features import cardio_feats      # noqa
MM = os.path.join(ROOT, "data", "multimodal")
OUT = os.path.join(ROOT, "data", "mm_features")


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(MM, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4]))
    print(f"{len(files)} subjects", flush=True)
    for i, f in enumerate(files):
        sid = os.path.basename(f)[:-4]
        dst = os.path.join(OUT, f"{sid}.npz")
        if os.path.exists(dst):
            continue
        d = np.load(f, allow_pickle=True)
        eeg = d["eeg"].astype(np.float32)                    # [n,7,3000] uV @100 Hz
        Feeg, _ = extract_features_v2(eeg, fs=100)
        Fcard = cardio_feats(d["card"].astype(np.float32))
        np.savez_compressed(dst, Feeg=np.nan_to_num(Feeg).astype(np.float32),
                            Fcard=Fcard[:len(d["y"])], y=d["y"].astype(np.int64),
                            apnea=d["apnea"].astype(np.int64), cvalid=d["cvalid"])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(files)}  (Feeg {Feeg.shape[1]}, Fcard {Fcard.shape[1]})", flush=True)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
