"""Respiratory events at 1-second resolution -> data/resp_events/SN<k>.npz.

From each annotation workbook's "Flow Events" sheet: every apnea (obstructive, central,
mixed) and hypopnea with its start and end, relative to the sheet's "Start Time" -- the
same reference read_apnea in build_multimodal.py uses for the epoch-level label, and the
start of the epoch grid the multimodal cache is cut on. RERAs and body events are kept in
the event list but are not AHI events.

Per recording:
    events  [m, 3] float   start s, end s, type (1 hypopnea, 2 obstructive, 3 central, 4 mixed,
                           5 RERA, 0 other)
    mask    [n*30] uint8   1 where an AHI event (types 1-4) covers that second, on the epoch
                           grid of processed7 (n epochs)
    ahi_ann float          AHI from the annotations: AHI events starting in a sleep epoch per
                           hour of sleep (sleep = processed7 label != Wake)

  python MMNet_research/preprocessing/build_event_labels.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
OUT = os.path.join(DATA, "resp_events")
TYPES = {"hypopnea": 1, "obstructive apnea": 2, "central apnea": 3, "mixed apnea": 4, "rera": 5}


def _s(v):
    return str(v).strip()


def workbook(sid):
    for d in ("zenodo", "Dataset"):
        for p in glob.glob(os.path.join(DATA, d, "**", "SN%d.xlsx" % sid), recursive=True):
            if not os.path.basename(p).startswith("~$"):
                return p
    return None


def read_events(xlsx):
    d = pd.read_excel(xlsx, sheet_name="Flow Events", header=None)
    col0 = [_s(v) for v in d.iloc[:, 0]]
    t0 = None
    for a, v in zip(col0, d.iloc[:, 1].tolist()):
        if a == "Start Time":
            t0 = pd.to_datetime(v)
            break
    if t0 is None or "Time start" not in col0:
        return None
    h = col0.index("Time start")
    cols = [_s(v) for v in d.iloc[h]]
    t = d.iloc[h + 1:].copy()
    t.columns = cols
    t = t[t["Time start"].map(lambda v: _s(v) not in ("nan", ""))]
    ev = []
    for _, r in t.iterrows():
        a, b = pd.to_datetime(r["Time start"], errors="coerce"), pd.to_datetime(r["Time end"], errors="coerce")
        if pd.isna(a) or pd.isna(b):
            continue
        ev.append(((a - t0).total_seconds(), (b - t0).total_seconds(),
                   TYPES.get(_s(r.get("Event name", "")).lower(), 0)))
    return np.array(ev, np.float64).reshape(-1, 3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p7 in sorted(glob.glob(os.path.join(DATA, "processed7", "SN*.npz")),
                     key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])):
        sid = int(re.findall(r"\d+", os.path.basename(p7))[0])
        y = np.load(p7)["y"]
        n = len(y)
        xlsx = workbook(sid)
        ev = read_events(xlsx) if xlsx else None
        if ev is None:
            print("[no events] SN%d" % sid, flush=True)
            continue
        mask = np.zeros(n * 30, np.uint8)
        ahi_ev = ev[(ev[:, 2] >= 1) & (ev[:, 2] <= 4)]
        for a, b, _ in ahi_ev:
            i, j = max(0, int(np.floor(a))), min(n * 30, int(np.ceil(b)))
            if j > i:
                mask[i:j] = 1
        sleep = y != 0
        ep = (ahi_ev[:, 0] // 30).astype(int)
        inrange = (ep >= 0) & (ep < n)
        n_sleep_ev = int(sleep[ep[inrange]].sum())
        tst_h = sleep.sum() * 30 / 3600
        ahi_ann = n_sleep_ev / tst_h if tst_h > 0 else np.nan
        np.savez_compressed(os.path.join(OUT, "SN%d.npz" % sid), events=ev, mask=mask,
                            ahi_ann=ahi_ann, n_out_of_range=int((~inrange).sum()))
        rows.append((sid, len(ahi_ev), n_sleep_ev, tst_h, ahi_ann, int((~inrange).sum())))
        print("[ok] SN%-3d %4d AHI events (%4d in sleep), TST %.1f h, AHI_ann %.1f, out of range %d"
              % rows[-1], flush=True)
    print("wrote %d recordings -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
