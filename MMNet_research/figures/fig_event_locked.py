"""
fig_event_locked.py -- event-locked average physiological traces (revision brief, Fig 5.2).

All scored respiratory events across the cohort are aligned at onset (t=0), a +-60 s window
is extracted from the raw cardiorespiratory signals, and the mean trace is plotted for SpO2,
respiratory effort, heart rate, and the MODEL's predicted apnea probability. If the predicted
probability rises with desaturation and falls on recovery, the model is grounded in the same
physiology a clinician reads -- a falsifiable interpretability claim (unlike a saliency map).

Uses the raw 25 Hz cardiorespiratory signals (data/multimodal) + exact event onsets (Flow
Events sheet) + the saved per-epoch apnea predictions (headline run). Saves source data.
"""
import os, sys, glob, re, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmnet_repro import REV, RUNS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM = os.path.join(ROOT, "data", "multimodal"); DS = os.path.join(ROOT, "data", "Dataset")
FIG = os.path.join(REV, "figures"); FD = os.path.join(REV, "figdata")
FS, WIN, EPOCH_S = 25, 60, 30           # 25 Hz, +-60 s window, 30 s epochs
CARD = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]


def onsets_seconds(xlsx):
    d = pd.read_excel(xlsx, sheet_name="Flow Events", header=None)
    col0 = d.iloc[:, 0].astype(str).tolist(); t0 = None
    for a, v in zip(col0, d.iloc[:, 1].tolist()):
        if str(a).strip() == "Start Time":
            t0 = pd.to_datetime(v); break
    if t0 is None:
        return []
    secs = []
    for a in col0:
        ts = pd.to_datetime(a, errors="coerce")
        if not pd.isna(ts):
            secs.append((ts - t0).total_seconds())
    return secs


def main():
    head = json.load(open(os.path.join(RUNS, "headline_concat.json")))
    apn_pred = {k: np.asarray(v["apnea_scores"]) for k, v in head["per_subject"].items()}
    xm = {re.search(r"SN\d+", os.path.basename(p)).group(): p
          for p in glob.glob(os.path.join(DS, "**", "*.xlsx"), recursive=True)
          if re.search(r"SN\d+", os.path.basename(p)) and not os.path.basename(p).startswith("~$")}
    w = WIN * FS
    acc = {k: [] for k in ["spo2", "effort", "hr", "pred"]}
    n_ev = 0
    for f in sorted(glob.glob(os.path.join(MM, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = os.path.basename(f)[:-4]
        if sid not in apn_pred or sid not in xm:
            continue
        d = np.load(f, allow_pickle=True)
        card = d["card"].astype(np.float32)                 # [n,7,750]
        n = card.shape[0]
        sig = card.transpose(1, 0, 2).reshape(7, n * 750)   # [7, n*25*30]
        spo2, effort, hr = sig[5], sig[4], sig[6]
        pred = apn_pred[sid]
        for sec in onsets_seconds(xm[sid]):
            c = int(sec * FS)
            if c - w < 0 or c + w >= sig.shape[1]:
                continue
            base = slice(c - w, c - w // 2)                 # pre-event baseline [-60,-30] s
            sl = slice(c - w, c + w)
            sp = spo2[sl]; ef = effort[sl]; h = hr[sl]
            if np.all(sp == 0):
                continue
            acc["spo2"].append(sp - np.nanmean(spo2[base]))
            acc["effort"].append((ef - np.nanmean(ef)) / (np.nanstd(ef) + 1e-6))
            acc["hr"].append(h - np.nanmean(hr[base]))
            # per-epoch prediction expanded to the sample grid
            ep = (np.arange(sl.start, sl.stop) // (FS * EPOCH_S)).clip(0, len(pred) - 1)
            acc["pred"].append(pred[ep])
            n_ev += 1
    t = np.linspace(-WIN, WIN, 2 * w)
    means = {k: np.nanmean(np.array(v), 0) for k, v in acc.items()}
    np.savez_compressed(os.path.join(FD, "event_locked.npz"), t=t, n_events=n_ev, **means)

    fig, ax = plt.subplots(4, 1, figsize=(5.2, 6.4), sharex=True)
    ax[0].plot(t, means["spo2"], color="#C44E52"); ax[0].set_ylabel("SpO2 change (%)")
    ax[1].plot(t, means["effort"], color="#55A868"); ax[1].set_ylabel("effort (z)")
    ax[2].plot(t, means["hr"], color="#8172B3"); ax[2].set_ylabel("heart rate change")
    ax[3].plot(t, means["pred"], color="#4C72B0"); ax[3].set_ylabel("predicted P(apnea)")
    for a in ax:
        a.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6); a.grid(alpha=0.25)
    ax[3].set_xlabel("time relative to event onset (s)")
    ax[0].set_title(f"Event-locked average over {n_ev:,} scored respiratory events", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_event_locked.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_event_locked.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote fig_event_locked ({n_ev} events); pred rises {means['pred'].max()-means['pred'].min():.3f} across window")


if __name__ == "__main__":
    main()
