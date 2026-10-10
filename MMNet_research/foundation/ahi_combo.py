"""AHI from MM-Net and both event detectors, combination chosen per fold on validation patients.

Per fold, on validation and test patients:
    mmnet   MM-Net respiratory head, expected onsets per hour of predicted sleep x validation scale
    v1      detector v1 (signals on the clock grid), seeds averaged, events mapped to EEG-grid stages
            through data/aligned/rows
    v2      detector v2 (EEG, aligned grid), seeds averaged
Each detector's counting rule (event_postproc.candidates) is chosen on the fold's validation patients
with true stages. The combination (any equal-weight mean of a non-empty subset) is chosen by median
absolute error against the annotation AHI on the same validation patients, with predicted stages.

    python ahi_combo.py [--v1 42 1 7] [--v2 42]
"""
import argparse
import itertools
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
from ahi_rescore import ahi_from_events
from event_postproc import candidates, count_events
from run_event_ahi import ahi_of, metrics, split

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
S = (42, 1, 7)


def probs(folder, fi, seeds, subs):
    Ps = [np.load(os.path.join(RUNS, folder, "fold%d_s%d_prob.npz" % (fi, sd))) for sd in seeds]
    return {s: np.mean([p["SN%d" % s].astype(np.float32) for p in Ps], 0) for s in subs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmnet", default="base-nocw-cross-ftlae-p8-al")
    ap.add_argument("--v1", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--v2", type=int, nargs="+", default=[42])
    ap.add_argument("--name", default="ahi_combo")
    a = ap.parse_args()
    ann = {s: float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    rows = {s: np.load(os.path.join(DATA, "aligned", "rows", "SN%d.npy" % s)) for s in C.SUBS}
    col = pd.to_numeric(pd.read_excel(os.path.join(DATA, "full100", "subject_description.xlsx"))["AHI_1_B"], errors="coerce")
    clin = {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}
    names = ("mmnet", "v1", "v2")
    combos = [c for r in (1, 2, 3) for c in itertools.combinations(names, r)]
    test = {"+".join(c): {} for c in combos}
    test["chosen on validation"] = {}
    log = []
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        tr, va = split(fi, tr_all)
        vt = list(va) + list(te)
        d = np.load(os.path.join(RUNS, "seed_ensemble", "%s_fold%d.npz" % (a.mmnet, fi)))
        stg, raw = {}, {}
        for s in vt:
            stg[s] = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in S], 0) + C.EPS))
            apr = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
            sl = stg[s] != 0
            raw[s] = apr[sl].sum() / max(sl.sum() * 30 / 3600, 0.5)
        k = np.median([ann[s] / max(raw[s], 1e-3) for s in va])
        P1, P2 = probs("event_ahi", fi, a.v1, vt), probs("event_ahi_v2", fi, a.v2, vt)
        e1 = {c: np.median([abs(ahi_from_events(count_events(P1[s], *c), C.DATA[s][2], rows[s]) - ann[s]) for s in va]) for c in candidates()}
        e2 = {c: np.median([abs(ahi_of(count_events(P2[s], *c), C.DATA[s][2]) - ann[s]) for s in va]) for c in candidates()}
        r1, r2 = min(e1, key=e1.get), min(e2, key=e2.get)
        est = {s: {"mmnet": raw[s] * k,
                   "v1": ahi_from_events(count_events(P1[s], *r1), stg[s], rows[s]),
                   "v2": ahi_of(count_events(P2[s], *r2), stg[s])} for s in vt}
        comb = lambda s, c: float(np.mean([est[s][n] for n in c]))
        verr = {c: np.median([abs(comb(s, c) - ann[s]) for s in va]) for c in combos}
        best = min(verr, key=verr.get)
        log.append("f%d %s (val %.1f)" % (fi, "+".join(best), verr[best]))
        for s in te:
            for c in combos:
                test["+".join(c)][s] = comb(s, c)
            test["chosen on validation"][s] = comb(s, best)
    subs = sorted(test["mmnet"])
    lines = ["AHI combinations (MM-Net %s, detector v1 seeds %s, v2 seeds %s), %d test patients"
             % (a.mmnet, a.v1, a.v2, len(subs)), "per-fold choice: " + "; ".join(log),
             "%-21s %-50s   %s" % ("", "vs annotation AHI", "vs clinical AHI_1_B")]
    for name, P_ in test.items():
        m1 = metrics([P_[s] for s in subs], [ann[s] for s in subs])
        cs = [s for s in subs if s in clin]
        m2 = metrics([P_[s] for s in cs], [clin[s] for s in cs])
        lines.append("%-21s rho %.3f MAE %5.1f sev acc %.2f kappa %.2f   rho %.3f MAE %5.1f sev acc %.2f kappa %.2f"
                     % (name, m1["rho"], m1["mae"], m1["sev_acc"], m1["sev_kappa"], m2["rho"], m2["mae"], m2["sev_acc"], m2["sev_kappa"]))
    text = "\n".join(lines)
    open(os.path.join(RUNS, a.name + ".txt"), "w").write(text + "\n")
    json.dump({k: {str(s): v for s, v in d.items()} for k, d in test.items()}, open(os.path.join(RUNS, a.name + ".json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
