"""ISRUC-Sleep-II -> MM-Net feature arrays, for external validation of BOTH heads.

Unlike Sleep-EDF, ISRUC supplies every modality MM-Net uses: the four iSLEEPS EEG
derivations verbatim (C4, C3, O2, O1 against the contralateral mastoid), two EOG
channels, chin EMG, ECG, airflow, thoracic and abdominal effort, and SpO2 --
plus scored respiratory events. So the respiratory head can be evaluated here,
which Sleep-EDF could not support.

Two practical wrinkles the file format forces:

* MNE refuses the `.rec` extension even though the contents are EDF, so this
  module reads EDF directly rather than copying 2.2 GB to rename it.
* Channel *labels* are inconsistent across recordings (`C4-A1` in some, `C4-M1`
  in others; `LOC-A2` vs `E1-M2`; `X1` vs `24`). The *transducer* field is
  consistent in all 16 files, so channel identity is resolved from that, with
  labels used only to pick which EEG derivation is which.

Missing relative to iSLEEPS: a dedicated pulse channel and the separate "Effort"
trace. Those two of the seven cardiorespiratory inputs are zero-filled, which the
notebook states; permutation importance put pulse/HRV at 0.013 AUC, so the loss
is small but real.

Usage:  python build_isruc_external.py
"""
import glob
import os
import re
import sys

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "utils"))

from features_v2 import extract_features_v2  # noqa: E402

RAW = os.path.join(REPO, "data", "isruc", "ISRUC-Sleep-II")
OUT = os.path.join(REPO, "data", "isruc_mm")
EEG_HZ, CARD_HZ, WIN = 100, 25, 30.0
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
RESP_EVENTS = {"OH", "CH", "MH", "OA", "CA", "MA"}
# the extractor's cardio channel order
CARD_ORDER = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]


# ---------------------------------------------------------------- EDF reading
def read_edf(path):
    """-> (labels, transducers, rates, signals) with signals in physical units."""
    with open(path, "rb") as f:
        head = f.read(256)
        n_rec = int(head[236:244])
        dur = float(head[244:252])
        ns = int(head[252:256])

        def fld(w):
            return [f.read(w).decode("ascii", "ignore").strip() for _ in range(ns)]

        lab, trd, _dim = fld(16), fld(80), fld(8)
        pmin = np.array([float(x) for x in fld(8)])
        pmax = np.array([float(x) for x in fld(8)])
        dmin = np.array([float(x) for x in fld(8)])
        dmax = np.array([float(x) for x in fld(8)])
        fld(80)                                     # prefiltering
        nsr = np.array([int(x) for x in fld(8)])
        f.read(32 * ns)                             # reserved

        raw = np.fromfile(f, dtype="<i2")

    per_rec = int(nsr.sum())
    n_rec = min(n_rec, len(raw) // per_rec)
    raw = raw[:n_rec * per_rec].reshape(n_rec, per_rec)

    scale = (pmax - pmin) / np.where(dmax - dmin == 0, 1, dmax - dmin)
    sigs, off = [], 0
    for i in range(ns):
        block = raw[:, off:off + nsr[i]].astype(np.float32)
        sigs.append(((block - dmin[i]) * scale[i] + pmin[i]).ravel())
        off += nsr[i]
    rates = nsr / dur
    return lab, trd, rates, sigs


def norm_label(s):
    return re.sub(r"\s+", "", s.upper()).replace("A1", "M1").replace("A2", "M2")


def pick_channels(lab, trd):
    """Resolve identity from the transducer tag; labels only disambiguate EEG."""
    idx = {}
    eeg = [i for i, t in enumerate(trd) if t.upper().startswith("EEG")]
    want = {"C4-M1": "c0", "C3-M2": "c1", "O2-M1": "c2", "O1-M2": "c3"}
    for i in eeg:
        key = want.get(norm_label(lab[i]))
        if key:
            idx[key] = i
    eog = [i for i, t in enumerate(trd) if t.upper().startswith("EOG")]
    for k, i in zip(("c4", "c5"), eog[:2]):
        idx[k] = i
    for key, tag in (("c6", "CHIN"), ("ECG", "EKG"), ("Flow", "FLOW_"),
                     ("Thorax", "EFFORT_"), ("Abdomen", "EFFORT2")):
        hit = [i for i, t in enumerate(trd) if t.upper().startswith(tag)]
        if hit:
            idx[key] = hit[0]
    spo2 = [i for i, t in enumerate(trd) if t.upper().startswith(("SPO2", "SAO2"))]
    if spo2:
        idx["SpO2"] = spo2[0]
    return idx


def to_rate(x, src, dst):
    if abs(src - dst) < 1e-6:
        return x
    from fractions import Fraction
    fr = Fraction(dst / src).limit_denominator(200)
    return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)


def epoch(x, hz, n_ep):
    per = int(hz * WIN)
    need = per * n_ep
    if len(x) < need:
        x = np.concatenate([x, np.zeros(need - len(x), np.float32)])
    return x[:need].reshape(n_ep, per)


# ---------------------------------------------------------------- annotations
def read_annotations(xlsx):
    import openpyxl
    ws = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)["Sheet1"]
    stages, apnea = [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        st = STAGE.get(str(r[1]).strip()) if r[1] is not None else None
        if st is None:
            continue
        stages.append(st)
        toks = {t.strip() for t in str(r[4] or "").split(",") if t.strip()}
        apnea.append(int(bool(toks & RESP_EVENTS)))
    return np.array(stages, np.int64), np.array(apnea, np.int64)


def process(rec_path, xlsx_path):
    lab, trd, rates, sigs = read_edf(rec_path)
    idx = pick_channels(lab, trd)
    missing = [k for k in ("c0", "c1", "c2", "c3", "c4", "c5", "c6") if k not in idx]
    if missing:
        raise RuntimeError("missing neural channels: %s" % missing)

    y, apn = read_annotations(xlsx_path)
    n_ep = min(len(y), int(len(sigs[idx["c0"]]) / rates[idx["c0"]] / WIN))
    y, apn = y[:n_ep], apn[:n_ep]

    neural = np.stack([
        epoch(to_rate(sigs[idx["c%d" % k]], rates[idx["c%d" % k]], EEG_HZ), EEG_HZ, n_ep)
        for k in range(7)], axis=1).astype(np.float32)          # [n,7,3000]

    card = np.zeros((n_ep, 7, int(CARD_HZ * WIN)), np.float32)  # [n,7,750]
    for j, name in enumerate(CARD_ORDER):
        if name in idx:
            i = idx[name]
            card[:, j] = epoch(to_rate(sigs[i], rates[i], CARD_HZ), CARD_HZ, n_ep)

    Feeg, _ = extract_features_v2(neural)
    from cardio_features import cardio_feats
    Fcard = cardio_feats(card)
    return (np.nan_to_num(Feeg).astype(np.float32),
            np.nan_to_num(Fcard).astype(np.float32), y, apn, sorted(idx))


def main():
    os.makedirs(OUT, exist_ok=True)
    recs = sorted(glob.glob(os.path.join(RAW, "*", "*", "*.rec")))
    print("found %d recordings\n" % len(recs))
    tot = ev = ok = 0
    for p in recs:
        sess = os.path.basename(os.path.dirname(p))
        subj = os.path.basename(os.path.dirname(os.path.dirname(p)))
        tag = "S%s_%s" % (subj, sess)
        dst = os.path.join(OUT, tag + ".npz")
        xlsx = os.path.join(os.path.dirname(p),
                            os.path.basename(p).replace(".rec", "_1.xlsx"))
        if os.path.exists(dst):
            print("[skip] %s" % tag); continue
        if not os.path.exists(xlsx):
            print("[FAIL] %s: no annotation" % tag); continue
        try:
            Feeg, Fcard, y, apn, chans = process(p, xlsx)
        except Exception as e:
            print("[FAIL] %s: %s: %s" % (tag, type(e).__name__, e)); continue
        np.savez_compressed(dst, Feeg=Feeg, Fcard=Fcard, y=y, apnea=apn,
                            session=int(sess), subject=int(subj))
        tot += len(y); ev += int(apn.sum()); ok += 1
        print("[ok]   %-7s %4d epochs  resp %3d (%4.1f%%)  Feeg%s Fcard%s"
              % (tag, len(y), apn.sum(), 100 * apn.mean(), Feeg.shape, Fcard.shape))
    if tot:
        print("\n=== %d recordings | %d epochs | %d respiratory-event epochs (%.2f%%) ==="
              % (ok, tot, ev, 100 * ev / tot))


if __name__ == "__main__":
    main()
