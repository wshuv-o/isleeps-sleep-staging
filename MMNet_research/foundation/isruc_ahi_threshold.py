"""ISRUC AHI with a thresholded epoch count: threshold and scale chosen on iSLEEPS validation patients (final config folds), applied to the saved ISRUC probabilities of each configuration."""
import os, sys, json, numpy as np
sys.path.insert(0, "G:/codes/isleeps/isleeps-sleep-staging/MMNet_research/model"); sys.path.insert(0, os.getcwd())
import mmnet_core as C
from run_event_ahi import split, metrics
R = "G:/codes/isleeps/isleeps-sleep-staging/MMNet_research/results/revision/runs/lesion/"
S = (42, 1, 7)
def est(ap, st, t):
    sl = st != 0; h = max(sl.sum() * 30 / 3600, 0.5)
    return (ap[sl].sum() if t is None else (ap[sl] > t).sum()) / h
val = []   # (fold, ann, ap, st) for iSLEEPS validation patients
for fi, (tr_all, te) in enumerate(C.FOLDS):
    _, va = split(fi, tr_all)
    d = np.load(R + "seed_ensemble/base-nocw-cross-ftlae-p8-al_fold%d.npz" % fi)
    for s in va:
        st = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["SN%d|%d" % (s, sd)] for sd in S], 0) + C.EPS))
        ap = np.mean([d["A:SN%d|%d" % (s, sd)] for sd in S], 0)
        val.append((fi, float(np.load("G:/codes/isleeps/isleeps-sleep-staging/data/aligned/resp_events/SN%d.npz" % s)["ahi_ann"]), ap, st))
best = None
for t in [None] + list(np.arange(0.1, 0.81, 0.05)):
    ks = [np.median([a / max(est(ap, st, t), 1e-3) for f, a, ap, st in val if f == fi]) for fi in range(10)]
    k = float(np.median(ks))
    err = np.median([abs(k * est(ap, st, t) - a) for f, a, ap, st in val])
    print("t", t, "k %.3f val median err %.2f" % (k, err))
    if best is None or err < best[2]: best = (t, k, err)
print("chosen on iSLEEPS validation:", best)
t, k, _ = best
for cfg in ("final", "final7", "frozen"):
    d = np.load(R + "external/isruc_%s_probs.npz" % cfg); j = json.load(open(R + "external/isruc_%s.json" % cfg))
    for g in ("II", "III", "II+III"):
        ts = list(j[g]["ahi_true"]); tru = [j[g]["ahi_true"][x] for x in ts]
        pred = []
        for x in ts:
            st = C.hmm(d["A_log"], d["pi_log"], np.log(np.mean([d["%s|%d" % (x, sd)] for sd in S], 0) + C.EPS))
            pred.append(k * est(np.mean([d["A:%s|%d" % (x, sd)] for sd in S], 0), st, t))
        m = metrics(pred, tru)
        print(cfg, g, "rho %.3f MAE %.1f sev acc %.2f kappa %.2f" % (m["rho"], m["mae"], m["sev_acc"], m["sev_kappa"]))
