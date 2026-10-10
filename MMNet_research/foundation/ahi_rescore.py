"""AHI on one consistent timeline (after the alignment fix), for every estimator so far.

Target: annotation AHI on the clock grid (data/aligned/resp_events: AHI events starting in a scored
sleep row / hours of sleep rows), and clinical AHI_1_B as a secondary reference.
Stages: predicted by the final MM-Net (3-seed, train-fold HMM) on the processed7 grid; epoch j sits
at clock row rows[j] (data/aligned/rows), so a clock-time event at second t counts if row t//30 is a
kept epoch predicted as sleep. Total sleep time = predicted-sleep epochs x 30 s.

Estimators (all choices on validation patients only):
    mmnet       expected event onsets per hour of predicted sleep from MM-Net's respiratory head,
                times one scale factor per fold (median ratio on that fold's validation patients)
    detector    event detector seed 42 (run_event_ahi.py), counting rule per fold from
                event_ahi_post_s42.json, events mapped to the EEG grid as above
    average     mean of the two

    python ahi_rescore.py [--mmnet base-nocw-cross-ftlae-p8-al] [--det-seed 42] [--name ahi_rescore]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

import mmnet_core as C
from event_postproc import count_events
from run_event_ahi import metrics, split

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
ENS = os.path.join(RUNS, "seed_ensemble")
S = (42, 1, 7)


def stages_and_resp(cfg, fi, subs):
    d = np.load(os.path.join(ENS, "%s_fold%d.npz" % (cfg, fi)))
    out = {}
    for s in subs:
        st = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in S], 0) + C.EPS))
        ap = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
        out[s] = (st, ap)
    return out


def ahi_from_events(events, stages, rows):
    """clock-time events -> count those whose row is a kept epoch predicted as sleep."""
    row_to_j = {int(r): j for j, r in enumerate(rows)}
    sleep = stages != 0
    tst = sleep.sum() * 30 / 3600
    n = 0
    for a, _ in events:
        j = row_to_j.get(int(a // 30))
        if j is not None and sleep[j]:
            n += 1
    return n / tst if tst > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmnet", default="base-nocw-cross-ftlae-p8-al")
    ap.add_argument("--det-seed", type=int, default=42)
    ap.add_argument("--name", default="ahi_rescore")
    a = ap.parse_args()
    ann = {s: float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    rows = {s: np.load(os.path.join(DATA, "aligned", "rows", "SN%d.npy" % s)) for s in C.SUBS}
    col = pd.to_numeric(pd.read_excel(os.path.join(DATA, "full100", "subject_description.xlsx"))["AHI_1_B"], errors="coerce")
    clin = {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}
    chosen = {c["fold"]: (c["rule"], c["thr"], c["par"]) for c in
              json.load(open(os.path.join(RUNS, "event_ahi_post_s%d.json" % a.det_seed)))["chosen"]}
    est = {"mmnet": {}, "detector": {}}
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        tr, va = split(fi, tr_all)
        sr = stages_and_resp(a.mmnet, fi, list(te) + list(va))
        raw = {}
        for s in list(te) + list(va):
            st, ap_ = sr[s]
            sl = st != 0
            raw[s] = ap_[sl].sum() / max(sl.sum() * 30 / 3600, 0.5)
        k = np.median([ann[s] / max(raw[s], 1e-3) for s in va])
        P = np.load(os.path.join(RUNS, "event_ahi", "fold%d_s%d_prob.npz" % (fi, a.det_seed)))
        for s in te:
            est["mmnet"][s] = raw[s] * k
            ev = count_events(P["SN%d" % s].astype(np.float32), *chosen[fi])
            est["detector"][s] = ahi_from_events(ev, sr[s][0], rows[s])
    subs = sorted(est["mmnet"])
    est["average"] = {s: (est["mmnet"][s] + est["detector"][s]) / 2 for s in subs}
    lines = ["AHI after the alignment fix (MM-Net %s, detector seed %d), %d test patients" % (a.mmnet, a.det_seed, len(subs)),
             "%-10s %-48s   %s" % ("", "vs annotation AHI (clock grid)", "vs clinical AHI_1_B")]
    for name, P_ in est.items():
        m1 = metrics([P_[s] for s in subs], [ann[s] for s in subs])
        cs = [s for s in subs if s in clin]
        m2 = metrics([P_[s] for s in cs], [clin[s] for s in cs])
        lines.append("%-10s rho %.3f MAE %5.1f sev acc %.2f kappa %.2f   rho %.3f MAE %5.1f sev acc %.2f kappa %.2f"
                     % (name, m1["rho"], m1["mae"], m1["sev_acc"], m1["sev_kappa"], m2["rho"], m2["mae"], m2["sev_acc"], m2["sev_kappa"]))
    text = "\n".join(lines)
    open(os.path.join(RUNS, a.name + ".txt"), "w").write(text + "\n")
    json.dump({k: {str(s): v for s, v in d.items()} for k, d in est.items()}, open(os.path.join(RUNS, a.name + ".json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
