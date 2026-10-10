"""Final AHI comparison on the aligned (EEG-grid) timeline.

Detector v2 (run_event_ahi.py --aligned --eeg --tag v2): per-second probabilities of the seeds given
are averaged, then for each fold the counting rule of event_postproc.py (runs / split / peaks with
their thresholds) is chosen on that fold's validation patients (median absolute error against the
annotation AHI, true stages as denominator) and applied to its test patients with the final MM-Net's
predicted stages. All signals, events and stages are on the processed7 epoch grid.

MM-Net (respiratory head): expected event onsets per hour of predicted sleep, times one scale factor
per fold fitted on that fold's validation patients. Combination: mean of detector and MM-Net.

    python ahi_final.py [--seeds 42 1 7] [--mmnet base-nocw-cross-ftlae-p8-al] [--folds 10]
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
from event_postproc import candidates, count_events
from run_event_ahi import ahi_of, metrics, split

DATA = os.path.join(REPO, "data")
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "lesion")
S = (42, 1, 7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    ap.add_argument("--mmnet", default="base-nocw-cross-ftlae-p8-al")
    ap.add_argument("--name", default="ahi_final")
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    ann = {s: float(np.load(os.path.join(DATA, "aligned", "resp_events", "SN%d.npz" % s))["ahi_ann"]) for s in C.SUBS}
    col = pd.to_numeric(pd.read_excel(os.path.join(DATA, "full100", "subject_description.xlsx"))["AHI_1_B"], errors="coerce")
    clin = {i + 1: float(v) for i, v in enumerate(col) if not pd.isna(v)}
    est = {"detector v2": {}, "mmnet": {}}
    chosen = []
    for fi, (tr_all, te) in enumerate(C.FOLDS[:a.folds]):
        tr, va = split(fi, tr_all)
        Ps = [np.load(os.path.join(RUNS, "event_ahi_v2", "fold%d_s%d_prob.npz" % (fi, sd))) for sd in a.seeds]
        P = {s: np.mean([p["SN%d" % s].astype(np.float32) for p in Ps], 0) for s in list(va) + list(te)}
        d = np.load(os.path.join(RUNS, "seed_ensemble", "%s_fold%d.npz" % (a.mmnet, fi)))
        stg, raw = {}, {}
        for s in list(va) + list(te):
            stg[s] = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in S], 0) + C.EPS))
            apr = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
            sl = stg[s] != 0
            raw[s] = apr[sl].sum() / max(sl.sum() * 30 / 3600, 0.5)
        k = np.median([ann[s] / max(raw[s], 1e-3) for s in va])
        best, best_err = None, np.inf
        for cand in candidates():
            err = np.median([abs(ahi_of(count_events(P[s], *cand), C.DATA[s][2]) - ann[s]) for s in va])
            if err < best_err:
                best, best_err = cand, err
        chosen.append("f%d %s t=%.2f p=%s" % (fi, best[0], best[1], best[2]))
        for s in te:
            est["detector v2"][s] = ahi_of(count_events(P[s], *best), stg[s])
            est["mmnet"][s] = raw[s] * k
    subs = sorted(est["mmnet"])
    est["average"] = {s: (est["mmnet"][s] + est["detector v2"][s]) / 2 for s in subs}
    lines = ["Final AHI (aligned timeline), detector v2 seeds %s averaged, MM-Net %s, %d test patients"
             % (a.seeds, a.mmnet, len(subs)), "counting rules chosen on validation: " + "; ".join(chosen),
             "%-12s %-50s   %s" % ("", "vs annotation AHI", "vs clinical AHI_1_B")]
    for name, P_ in est.items():
        m1 = metrics([P_[s] for s in subs], [ann[s] for s in subs])
        cs = [s for s in subs if s in clin]
        m2 = metrics([P_[s] for s in cs], [clin[s] for s in cs])
        lines.append("%-12s rho %.3f MAE %5.1f sev acc %.2f kappa %.2f   rho %.3f MAE %5.1f sev acc %.2f kappa %.2f"
                     % (name, m1["rho"], m1["mae"], m1["sev_acc"], m1["sev_kappa"], m2["rho"], m2["mae"], m2["sev_acc"], m2["sev_kappa"]))
    text = "\n".join(lines)
    open(os.path.join(RUNS, a.name + ".txt"), "w").write(text + "\n")
    json.dump({k: {str(s): v for s, v in d.items()} for k, d in est.items()}, open(os.path.join(RUNS, a.name + ".json"), "w"), indent=1)
    print(text)


if __name__ == "__main__":
    main()
