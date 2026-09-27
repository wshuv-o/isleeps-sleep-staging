"""Shared loaders for the round-3 CPU re-analyses.

Everything here reads files that are already on disk and never touches the GPU:

  * stage labels        data/processed7/SN*.npz            (100 recordings, 'y')
  * respiratory events  data/Dataset/Batch-*/SN*.xlsx       ('Flow Events' sheet)
  * clinical AHI        data/Dataset/subject_description.xlsx  (column AHI_1_B)
  * model outputs       results/revision/runs/final/per_subject_seed42.json
                        (seed-42 per-epoch respiratory probability, 99 patients)
  * montage status      data/mm_features/SN*.npz 'cvalid'   (96 of 99 on disk)

SN28 is excluded everywhere (byte-identical to SN15), giving N = 99.

The respiratory label the published model was trained and scored on is built by
preprocessing/build_multimodal.py:read_apnea, which marks the ONE epoch containing
each event's start time. `labels(..., rule="onset")` reproduces it exactly and is
checked against data/mm_features in labels_check.py; the other rules are the
sensitivity variants the referee asked for.
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MM = os.path.dirname(os.path.dirname(HERE))                      # MMNet_research
ROOT = os.path.dirname(MM)                                       # sleep-staging-psg
DATA = os.path.join(ROOT, "data")
FINAL = os.path.join(MM, "results", "revision", "runs", "final")
OUT = os.path.join(MM, "results", "revision", "runs", "round3")
EPOCH_S = 30.0
SUBJECTS = ["SN%d" % i for i in range(1, 101) if i != 28]


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    print("wrote", p)
    return p


def stages(sid):
    return np.load(os.path.join(DATA, "processed7", sid + ".npz"))["y"].astype(int)


def _xlsx(sid):
    for b in sorted(glob.glob(os.path.join(DATA, "Dataset", "Batch-*"))):
        p = os.path.join(b, sid + ".xlsx")
        if os.path.exists(p):
            return p
    return None


_EV_CACHE = {}


def events(sid):
    """[(start_s, end_s, name)] relative to the Flow Events 'Start Time'."""
    if sid in _EV_CACHE:
        return _EV_CACHE[sid]
    import pandas as pd
    p = _xlsx(sid)
    ev = []
    if p is not None:
        d = pd.read_excel(p, sheet_name="Flow Events", header=None)
        t0 = None
        for a, v in zip(d.iloc[:, 0].astype(str), d.iloc[:, 1]):
            if a.strip() == "Start Time":
                t0 = pd.to_datetime(v)
                break
        if t0 is not None:
            for _, r in d.iterrows():
                ts = pd.to_datetime(str(r.iloc[0]), errors="coerce")
                te = pd.to_datetime(str(r.iloc[1]), errors="coerce")
                if pd.isna(ts):
                    continue
                s = (ts - t0).total_seconds()
                e = (te - t0).total_seconds() if not pd.isna(te) else s
                ev.append((s, max(e, s), str(r.iloc[3]).strip()))
    _EV_CACHE[sid] = ev
    return ev


def labels(sid, n, rule="onset", min_s=0.0, frac=0.0, types=None):
    """Per-epoch binary respiratory label.

    onset   : the epoch holding the event start (the published rule)
    overlap : every epoch the event overlaps by more than `min_s` seconds and by at
              least `frac` of the event's own duration
    """
    y = np.zeros(n, np.int64)
    for s, e, name in events(sid):
        if types is not None and name not in types:
            continue
        if rule == "onset":
            k = int(s // EPOCH_S)
            if 0 <= k < n:
                y[k] = 1
            continue
        dur = max(e - s, 1e-6)
        for k in range(int(s // EPOCH_S), int(e // EPOCH_S) + 1):
            if not 0 <= k < n:
                continue
            ov = min(e, (k + 1) * EPOCH_S) - max(s, k * EPOCH_S)
            if ov > min_s and ov / dur >= frac:
                y[k] = 1
    return y


def probs():
    p = json.load(open(os.path.join(FINAL, "per_subject_seed42.json")))
    return {k: np.asarray(v["apnea"], float) for k, v in p.items()}, p


def ahi():
    import pandas as pd
    d = pd.read_excel(os.path.join(DATA, "Dataset", "subject_description.xlsx"))
    # rows 11-20 are labelled AN1..AN10 in the sheet; the row order is SN1..SN100,
    # which is also how foundation/regen_derived.py maps them (row index + 1)
    col = pd.to_numeric(d["AHI_1_B"], errors="coerce")
    return {"SN%d" % (i + 1): float(v) for i, v in enumerate(col) if not pd.isna(v)}


def montage_complete():
    """{sid: True/False/None}; None where the cached cvalid is not on disk."""
    out = {}
    for sid in SUBJECTS:
        f = os.path.join(DATA, "mm_features", sid + ".npz")
        out[sid] = bool(np.load(f)["cvalid"].sum() == 7) if os.path.exists(f) else None
    return out


def wilcoxon_p(a, b):
    from scipy.stats import wilcoxon
    return float(wilcoxon(a, b).pvalue)
