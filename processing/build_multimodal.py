"""
build_multimodal.py -- assemble the multimodal PSG cache.

EEG/EOG/EMG + labels come from the already-clean processed7 cache (100 subjects,
montage-normalised, epoch-aligned). Cardiorespiratory signals are extracted fresh
from the raw EDFs with pyedflib (which reads per-channel rates and physical units
correctly, unlike MNE here), resampled to 25 Hz, and aligned epoch-for-epoch.

Per subject -> data/multimodal/SN<k>.npz:
    eeg   [n, 7, 3000]  from processed7  (C4:M1 C3:M2 O2:M1 O1:M2 E1:M2 E2:M2 EMG)
    card  [n, 7,  750]  cardiorespiratory @25 Hz: ECG Flow Thorax Abdomen Effort SpO2 Pulse
    cvalid[7] bool      which cardio channels are real (non-zero) for this subject
    y     [n]           sleep stage 0=W 1=N1 2=N2 3=N3 4=REM
    apnea [n]           1 if an apnea/hypopnea event overlaps the epoch

  KMP_DUPLICATE_LIB_OK=TRUE python processing/build_multimodal.py --one SN5
  KMP_DUPLICATE_LIB_OK=TRUE python processing/build_multimodal.py
"""
import os, sys, glob, re, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "Dataset")
P7 = os.path.join(ROOT, "data", "processed7")
OUT = os.path.join(ROOT, "data", "multimodal")
EPOCH_S, FS_CARD = 30, 25

# canonical cardiorespiratory channels, each with accepted name variants
CARD = [("ECG",     ["ECG 2", "ECG"]),
        ("Flow",    ["Flow Th"]),
        ("Thorax",  ["Thorax", "RIP Thorax"]),
        ("Abdomen", ["Abdomen", "RIP Abdomen"]),
        ("Effort",  ["Sum Effort", "Sum RIPs"]),
        ("SpO2",    ["SPO2"]),
        ("Pulse",   ["Pulse"])]


def sid_of(p):
    m = re.search(r"SN\d+", os.path.basename(p)); return m.group() if m else None


def find():
    edf = {sid_of(p): p for p in glob.glob(os.path.join(DS, "**", "*.edf"), recursive=True)}
    xls = {sid_of(p): p for p in glob.glob(os.path.join(DS, "**", "*.xlsx"), recursive=True)
           if sid_of(p) and not os.path.basename(p).startswith("~$")}
    p7 = {sid_of(p): p for p in glob.glob(os.path.join(P7, "SN*.npz"))}
    return {s: (edf[s], xls.get(s), p7[s]) for s in edf if s in p7}


def read_card(edf_path, n_epochs):
    """extract the 7 cardiorespiratory channels, resample to 25 Hz, epoch to n_epochs."""
    import pyedflib
    from scipy.signal import resample_poly
    r = pyedflib.EdfReader(edf_path)
    L = r.getSignalLabels()
    spe = FS_CARD * EPOCH_S                                   # 750 samples/epoch
    out = np.zeros((n_epochs, len(CARD), spe), np.float32)
    valid = np.zeros(len(CARD), bool)
    for ci, (_, variants) in enumerate(CARD):
        idx = next((L.index(v) for v in variants if v in L), None)
        if idx is None:
            continue
        fs = r.getSampleFrequency(idx)
        x = r.readSignal(idx)                                 # physical units
        if x.std() < 1e-9:                                    # zero-filled channel
            continue
        if abs(fs - FS_CARD) > 1e-6:
            g = np.gcd(int(round(fs)), FS_CARD)
            x = resample_poly(x, FS_CARD // g, int(round(fs)) // g)
        need = n_epochs * spe
        if len(x) < need:
            x = np.concatenate([x, np.zeros(need - len(x), x.dtype)])
        out[:, ci, :] = x[:need].reshape(n_epochs, spe)
        valid[ci] = True
    r.close()
    return out, valid


def read_apnea(xlsx_path, n_epochs):
    import pandas as pd
    apnea = np.zeros(n_epochs, np.int64)
    if xlsx_path is None:
        return apnea
    try:
        d = pd.read_excel(xlsx_path, sheet_name="Flow Events", header=None)
    except Exception:
        return apnea
    id0 = d.iloc[:, 0].astype(str).tolist()
    t0 = None
    for a, v in zip(id0, d.iloc[:, 1].tolist()):
        if str(a).strip() == "Start Time":
            t0 = pd.to_datetime(v); break
    if t0 is None:
        return apnea
    for a in id0:
        ts = pd.to_datetime(a, errors="coerce")
        if pd.isna(ts):
            continue
        e = int((ts - t0).total_seconds() // EPOCH_S)
        if 0 <= e < n_epochs:
            apnea[e] = 1
    return apnea


def process(sid, edf, xlsx, p7, verbose=False):
    d = np.load(p7, allow_pickle=True)
    eeg = d["x"].astype(np.float16); y = d["y"].astype(np.int64); n = len(y)
    card, cvalid = read_card(edf, n)
    apnea = read_apnea(xlsx, n)
    if verbose:
        import collections
        names = [c for c, _ in CARD]
        real = [names[i] for i in range(len(names)) if cvalid[i]]
        print(f"  {sid}: n={n} | EEG {eeg.shape} | CARD {card.shape} real={real}")
        print(f"     stages {dict(collections.Counter(y.tolist()))} | apnea+ {int(apnea.sum())} "
              f"({100*apnea.mean():.0f}%)")
    return dict(eeg=eeg, card=card.astype(np.float16), cvalid=cvalid, y=y, apnea=apnea)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--one", default=None)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    F = find(); print(f"{len(F)} subjects with EDF + processed7")
    todo = [a.one] if a.one else sorted(F, key=lambda s: int(s[2:]))
    ok = 0; full = 0
    for sid in todo:
        if sid not in F:
            print(f"  {sid} not found"); continue
        edf, xlsx, p7 = F[sid]
        try:
            res = process(sid, edf, xlsx, p7, verbose=bool(a.one))
        except Exception as e:
            print(f"  {sid} FAILED: {type(e).__name__}: {e}"); continue
        np.savez_compressed(os.path.join(OUT, f"{sid}.npz"), **res)
        ok += 1; full += int(res["cvalid"].sum() >= 5)
        if not a.one and ok % 10 == 0:
            print(f"  ... {ok} done ({full} with full cardio)")
    print(f"\nwrote {ok} subjects -> {OUT}  ({full} with >=5 cardio channels)")


if __name__ == "__main__":
    main()
