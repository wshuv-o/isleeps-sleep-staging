"""
extract_labels.py -- deterministic re-extraction of per-epoch respiratory-event TYPE
labels and per-subject AHI/severity, aligned to the existing mm_features epochs.

Event types (Flow Events sheet, column 3): Hypopnea, Obstructive Apnea, Central Apnea,
Mixed Apnea, RERA. Marked at the epoch containing each event's start time (identical rule
to processing/build_multimodal.read_apnea, so 'any' reconstructs the trained apnea label).

Outputs (results/revision/):
    event_labels.npz   dict SNk -> [n,6] int  columns: any, hypopnea, obstructive, central, mixed, rera
    ahi.json           {SNk: {ahi, severity}}   severity by AASM cut-offs (5/15/30)
"""
import os, sys, glob, re, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "data", "Dataset")
FE = os.path.join(ROOT, "data", "mm_features")
OUT = os.path.join(ROOT, "results", "revision")
os.makedirs(OUT, exist_ok=True)
EPOCH_S = 30
TYPES = ["hypopnea", "obstructive", "central", "mixed", "rera"]
TYPEMAP = {"hypopnea": "hypopnea", "obstructive apnea": "obstructive",
           "central apnea": "central", "mixed apnea": "mixed", "rera": "rera"}


def sid_of(p):
    m = re.search(r"SN\d+", os.path.basename(p)); return m.group() if m else None


def xlsx_map():
    return {sid_of(p): p for p in glob.glob(os.path.join(DS, "**", "*.xlsx"), recursive=True)
            if sid_of(p) and not os.path.basename(p).startswith("~$")}


def epoch_types(xlsx_path, n):
    """[n,6] int: columns any, hypopnea, obstructive, central, mixed, rera (start-epoch marking)."""
    lab = np.zeros((n, 6), np.int64)
    if xlsx_path is None or not os.path.exists(xlsx_path):
        return lab
    try:
        d = pd.read_excel(xlsx_path, sheet_name="Flow Events", header=None)
    except Exception:
        return lab
    col0 = d.iloc[:, 0].astype(str).tolist()
    t0 = None
    for a, v in zip(col0, d.iloc[:, 1].tolist()):
        if str(a).strip() == "Start Time":
            t0 = pd.to_datetime(v); break
    if t0 is None:
        return lab
    typ = d.iloc[:, 3].astype(str).str.strip().str.lower() if d.shape[1] > 3 else None
    for i, a in enumerate(col0):
        ts = pd.to_datetime(a, errors="coerce")
        if pd.isna(ts):
            continue
        e = int((ts - t0).total_seconds() // EPOCH_S)
        if not (0 <= e < n):
            continue
        lab[e, 0] = 1                                   # any respiratory event
        if typ is not None:
            key = TYPEMAP.get(typ.iloc[i], None)
            if key:
                lab[e, 1 + TYPES.index(key)] = 1
    return lab


def main():
    xm = xlsx_map()
    files = sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4]))
    out, mism = {}, 0
    for f in files:
        sid = os.path.basename(f)[:-4]
        d = np.load(f)
        n = len(d["y"]); apnea = d["apnea"].astype(np.int64)
        lab = epoch_types(xm.get(sid), n)
        out[sid] = lab.astype(np.int16)
        # sanity: reconstructed 'any' should match the trained apnea label
        if not np.array_equal(lab[:, 0], apnea):
            mism += 1
    np.savez_compressed(os.path.join(OUT, "event_labels.npz"), **out)
    tot = np.concatenate([out[s] for s in out], 0).sum(0)
    print(f"{len(out)} subjects | 'any' mismatch vs trained apnea: {mism}")
    print("epoch counts  any={} hypopnea={} obstructive={} central={} mixed={} rera={}".format(*tot))

    # AHI + severity
    desc = pd.read_excel(os.path.join(DS, "subject_description.xlsx"))
    name = desc["Annonymized_Name"].astype(str).str.replace(".edf", "", regex=False)
    ahi = pd.to_numeric(desc["AHI_1_B"], errors="coerce")
    cuts = [-1, 5, 15, 30, 1e9]; sev = ["Normal", "Mild", "Moderate", "Severe"]
    ahid = {}
    for nm, a in zip(name, ahi):
        if pd.isna(a):
            continue
        s = sev[int(np.digitize(a, cuts[1:-1]))]
        ahid[nm] = {"ahi": float(a), "severity": s}
    json.dump(ahid, open(os.path.join(OUT, "ahi.json"), "w"), indent=2)
    from collections import Counter
    print(f"AHI: {len(ahid)} subjects | severity {dict(Counter(v['severity'] for v in ahid.values()))}")
    print(f"saved -> {OUT}/event_labels.npz, ahi.json")


if __name__ == "__main__":
    main()
