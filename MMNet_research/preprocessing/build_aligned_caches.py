"""Cardiorespiratory caches aligned to the EEG epochs -> data/aligned/.

processed7 keeps only the Sleep-profile epochs scored W/N1/N2/N3/REM (StagingPreprocess.create_windows),
each cut from its true time, so processed7 epoch j sits at clock row rows[j] of the Sleep profile
(30 s per row from the recording start). build_multimodal.read_card and read_apnea instead cut the
cardiorespiratory signals and the event label on the CLOCK grid (epoch j = seconds 30j..30j+30), so
wherever an unscorable epoch is dropped before j, EEG/stage epoch j and cardio epoch j are different
times. This rebuilds every clock-based array on the processed7 grid:

    data/aligned/rows/SN<k>.npy            clock row of each processed7 epoch
    data/aligned/multimodal/SN<k>.npz      eeg (unchanged), card, cvalid, y, apnea  (as build_multimodal)
    data/aligned/pressure_flow/SN<k>.npz   pflow, valid                              (as build_pressure_cache)
    data/aligned/mm_features/SN<k>.npz     Feeg (unchanged), Fcard recomputed, y, apnea, cvalid
    data/aligned/resp_events/SN<k>.npz     mask on the processed7 grid (seconds of each kept epoch),
                                           events, ahi_ann (unchanged definition)

Needs pyedflib (prep interpreter):
    G:/codes/isleeps/prepenv/Scripts/python MMNet_research/preprocessing/build_aligned_caches.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")
AL = os.path.join(DATA, "aligned")
sys.path.insert(0, HERE)
import build_multimodal as B                   # noqa: E402
from cardio_features import cardio_feats       # noqa: E402

OK = ("Wake", "N1", "N2", "N3", "REM")
EPOCH, FS = 30, 25


def sources(sid):
    xl = [p for d in ("zenodo", "Dataset") for p in glob.glob(os.path.join(DATA, d, "**", "SN%d.xlsx" % sid), recursive=True)
          if not os.path.basename(p).startswith("~$")]
    ed = [p for d in ("zenodo", "Dataset") for p in glob.glob(os.path.join(DATA, d, "**", "SN%d.edf" % sid), recursive=True)]
    return (ed[0] if ed else None), (xl[0] if xl else None)


def kept_rows(xlsx, n):
    ann = pd.read_excel(xlsx, sheet_name="Sleep profile")[8:]          # as StagingPreprocess.read_annotations
    lab = [str(v) for v in ann.iloc[:, 1].tolist()]
    ok = np.array([l in OK for l in lab])
    rows = np.where(ok)[0][:n]
    if len(rows) != n:
        raise ValueError("only %d scorable rows for %d epochs" % (len(rows), n))
    sleep_row = np.array([l in ("N1", "N2", "N3", "REM") for l in lab])   # clock-grid sleep
    return rows, sleep_row


def read_signal(edf, variants_list, n_rows):
    """build_multimodal.read_card, but returning the CLOCK-grid signal for n_rows epochs."""
    saved = B.CARD
    try:
        B.CARD = variants_list
        x, valid = B.read_card(edf, n_rows)          # [n_rows, c, 750] on the clock grid
    finally:
        B.CARD = saved
    return x, valid


def main():
    for sub in ("rows", "multimodal", "pressure_flow", "mm_features", "resp_events"):
        os.makedirs(os.path.join(AL, sub), exist_ok=True)
    p7s = sorted(glob.glob(os.path.join(DATA, "processed7", "SN*.npz")), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
    for p7 in p7s:
        sid = int(re.findall(r"\d+", os.path.basename(p7))[0])
        dst = os.path.join(AL, "mm_features", "SN%d.npz" % sid)
        if os.path.exists(dst):
            continue
        edf, xlsx = sources(sid)
        old = np.load(os.path.join(DATA, "multimodal", "SN%d.npz" % sid))
        y = old["y"]; n = len(y)
        rows, sleep_row = kept_rows(xlsx, n)
        n_rows = int(rows[-1]) + 1
        np.save(os.path.join(AL, "rows", "SN%d.npy" % sid), rows)
        # seven cardiorespiratory channels + nasal pressure on the clock grid, then picked by row
        card_c, cvalid = read_signal(edf, B.CARD, n_rows)
        pf_c, pvalid = read_signal(edf, [("PressureFlow", ["Pressure Flow"])], n_rows)
        card = card_c[rows].astype(np.float16)
        pflow = pf_c[rows, 0].astype(np.float16)
        apnea_clock = B.read_apnea(xlsx, n_rows)       # event starts on the clock grid
        apnea = apnea_clock[rows].astype(np.int64)
        np.savez_compressed(os.path.join(AL, "multimodal", "SN%d.npz" % sid), card=card,   # eeg unchanged: data/multimodal
                            cvalid=cvalid, y=y, apnea=apnea)
        np.savez_compressed(os.path.join(AL, "pressure_flow", "SN%d.npz" % sid), pflow=pflow, valid=bool(pvalid[0]))
        f = np.load(os.path.join(DATA, "mm_features", "SN%d.npz" % sid))
        Fcard = cardio_feats(card.astype(np.float32))[:n]
        np.savez_compressed(dst, Feeg=f["Feeg"], Fcard=Fcard, y=y, apnea=apnea, cvalid=cvalid)
        ev = np.load(os.path.join(DATA, "resp_events", "SN%d.npz" % sid))
        mask_clock = np.zeros(n_rows * EPOCH, np.uint8)
        m0 = ev["mask"]
        mask_clock[:min(len(m0), len(mask_clock))] = m0[:len(mask_clock)]
        for a, b, t in ev["events"]:                    # events past the old grid end
            if 1 <= t <= 4:
                i, j = max(0, int(np.floor(a))), min(len(mask_clock), int(np.ceil(b)))
                if j > i:
                    mask_clock[i:j] = 1
        mask = mask_clock.reshape(n_rows, EPOCH)[rows].reshape(-1)
        # annotation AHI on the clock grid: AHI events starting in a scored-sleep row / hours of sleep rows
        e = ev["events"]; e = e[(e[:, 2] >= 1) & (e[:, 2] <= 4)]
        r_on = (e[:, 0] // EPOCH).astype(int)
        inr = (r_on >= 0) & (r_on < len(sleep_row))
        tst_h = sleep_row.sum() * EPOCH / 3600
        ahi_ann = float(sleep_row[r_on[inr]].sum() / tst_h) if tst_h > 0 else np.nan
        np.savez_compressed(os.path.join(AL, "resp_events", "SN%d.npz" % sid), mask=mask, events=ev["events"],
                            ahi_ann=ahi_ann, ahi_ann_old=ev["ahi_ann"], rows=rows, sleep_row=sleep_row)
        shift = int(rows[-1] - (n - 1))
        print("[ok] SN%-3d %4d epochs, clock rows %d, end shift %d epochs, apnea+ old %.3f new %.3f"
              % (sid, n, n_rows, shift, old["apnea"].mean(), apnea.mean()), flush=True)


if __name__ == "__main__":
    main()
