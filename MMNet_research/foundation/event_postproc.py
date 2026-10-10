"""Post-processing of the event detector's per-second probabilities (run_event_ahi.py), chosen on
validation patients, scored on test patients. No retraining.

Counting rules (each with a threshold grid):
    runs    threshold, drop runs < 10 s, merge gaps < 3 s, one event per run (run_event_ahi.py)
    split   as runs, but a run longer than L s counts as ceil(length / L) events (L in 20..45 s),
            so back-to-back apneas merged into one long run are counted separately
    peaks   local maxima of the 5-s smoothed probability above the threshold, at least D s apart
            (D in 15..35 s)
AHI = events starting in predicted sleep / predicted sleep time (final MM-Net staging), as before.
For each fold the (rule, parameters) pair with the lowest median absolute AHI error on that fold's
validation patients is applied to its test patients.

    python event_postproc.py [--seed 42] [--stages base-nocw-cross-ftlae-p8]
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
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

import mmnet_core as C
from run_event_ahi import DATA, ENS, OUT, RUNS, ahi_of, events_from, metrics, split

THR = np.round(np.arange(0.1, 0.86, 0.05), 3)


def count_events(p, rule, thr, par):
    if rule == "runs":
        return events_from(p, thr)
    if rule == "split":
        out = []
        for a, b in events_from(p, thr):
            k = max(1, int(np.ceil((b - a) / par)))
            step = (b - a) / k
            out += [(a + i * step, a + (i + 1) * step) for i in range(k)]
        return out
    ps = uniform_filter1d(p, 5)
    pk, _ = find_peaks(ps, height=thr, distance=par)
    return [(int(i) - 5, int(i) + 5) for i in pk]


def candidates():
    for t in THR:
        yield ("runs", t, 0)
        for L in (20, 25, 30, 35, 45):
            yield ("split", t, L)
        for D in (15, 20, 25, 35):
            yield ("peaks", t, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", default="base-nocw-cross-ftlae-p8")
    a = ap.parse_args()
    ann = {s: float(np.load(os.path.join(DATA, "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    col = pd.to_numeric(pd.read_excel(os.path.join(DATA, "full100", "subject_description.xlsx"))["AHI_1_B"], errors="coerce")
    clin = {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}
    pred, pred_or, chosen = {}, {}, []
    for fi, (tr_all, te) in enumerate(C.FOLDS):
        tr, va = split(fi, tr_all)
        P = np.load(os.path.join(OUT, "fold%d_s%d_prob.npz" % (fi, a.seed)))
        st = np.load(os.path.join(ENS, "%s_fold%d.npz" % (a.stages, fi)))
        stg = {s: C.hmm(st["A_log"], st["pi_log"], np.log(np.mean([st["SN%d|%d" % (s, sd)] for sd in (42, 1, 7)], 0) + C.EPS))
               for s in list(te)}
        best, best_err = None, np.inf
        for cand in candidates():
            err = np.median([abs(ahi_of(count_events(P["SN%d" % s].astype(np.float32), *cand), C.DATA[s][2]) - ann[s]) for s in va])
            if err < best_err:
                best, best_err = cand, err
        chosen.append(dict(fold=fi, rule=best[0], thr=float(best[1]), par=best[2], val_median_err=float(best_err)))
        for s in te:
            ev = count_events(P["SN%d" % s].astype(np.float32), *best)
            pred[s] = ahi_of(ev, stg[s]); pred_or[s] = ahi_of(ev, C.DATA[s][2])
    subs = sorted(pred)
    lines = ["Event detector seed %d, post-processing chosen per fold on validation patients" % a.seed,
             "chosen: " + "; ".join("f%d %s t=%.2f p=%s" % (c["fold"], c["rule"], c["thr"], c["par"]) for c in chosen), ""]
    for lab, P_ in (("end to end (predicted stages)", pred), ("oracle stages", pred_or)):
        m1 = metrics([P_[s] for s in subs], [ann[s] for s in subs])
        cs = [s for s in subs if s in clin]
        m2 = metrics([P_[s] for s in cs], [clin[s] for s in cs])
        lines.append("%-30s vs annotation AHI: rho %.3f MAE %.1f severity acc %.2f kappa %.2f | vs clinical: rho %.3f MAE %.1f acc %.2f kappa %.2f"
                     % (lab, m1["rho"], m1["mae"], m1["sev_acc"], m1["sev_kappa"], m2["rho"], m2["mae"], m2["sev_acc"], m2["sev_kappa"]))
    text = "\n".join(lines)
    open(os.path.join(RUNS, "event_ahi_post_s%d.txt" % a.seed), "w").write(text + "\n")
    json.dump(dict(chosen=chosen, pred={str(s): pred[s] for s in subs}), open(os.path.join(RUNS, "event_ahi_post_s%d.json" % a.seed), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
