"""Ablate the direct cardiorespiratory-to-respiratory-head connection.

Section V asserts that this connection matters, on the reasoning that without it
the respiratory decision has to survive a fusion trained mostly for staging. An
external review pointed out that the claim is asserted and never tested: there is
no row in any table without the bypass. This runs it.

The flag is already plumbed through run_10fold, so nothing about the model or the
protocol changes except that the respiratory head stops receiving its direct copy
of the cardiorespiratory embedding. Same folds, same three seeds, same thirty
fold-values as every other headline figure, so the difference is testable with
the Wilcoxon procedure used throughout.

If the effect is null the design claim comes out of the paper. That is the point
of running it.

  KMP_DUPLICATE_LIB_OK=TRUE python run_bypass_ablation.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "MMNet_research", "model"))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

OUT = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final",
                   "bypass_ablation.json")
FINAL = dict(arm="A+labram", cardio="raw_cnn", cardio_mode="concat",
             temporal="lstm", hidden=256, drop=0.3, lr=3e-4, wd=1e-4)
SEEDS = [42, 1, 7]


def main():
    t0 = time.time()
    res = {}
    with sweep.config(C, FINAL["arm"], hidden=FINAL["hidden"], drop=FINAL["drop"],
                      lr=FINAL["lr"], wd=FINAL["wd"], cardio=FINAL["cardio"],
                      cardio_mode=FINAL["cardio_mode"]) as h:
        print("eeg %d-d, cardio %d-d" % (h.dim, h.n_card), flush=True)
        for bypass in (True, False):
            key = "with_bypass" if bypass else "no_bypass"
            res[key] = {}
            for seed in SEEDS:
                r = C.run_10fold(fusion="concat", temporal=FINAL["temporal"],
                                 seed=seed, bypass=bypass)
                # run_10fold returns aggregates as (mean, sd) TUPLES; the ten
                # fold-values live in per_fold, and those are what the paired
                # test needs.
                res[key]["seed|%d" % seed] = {
                    k: [float(f[k]) for f in r["per_fold"]]
                    for k in ("acc", "kappa", "auc", "ap")}
                print("%-12s seed %2d  acc %.4f  auc %.4f  [%.1f min]"
                      % (key, seed,
                         np.mean(res[key]["seed|%d" % seed]["acc"]),
                         np.mean(res[key]["seed|%d" % seed]["auc"]),
                         (time.time() - t0) / 60), flush=True)

    def pool(key, metric):
        return np.array([v for s in sorted(res[key]) for v in res[key][s][metric]])

    summary = {}
    print("\n%-10s %18s %18s %10s" % ("metric", "with bypass", "no bypass", "Wilcoxon p"))
    print("-" * 60)
    for m in ("auc", "ap", "acc", "kappa"):
        a, b = pool("with_bypass", m), pool("no_bypass", m)
        if len(a) != len(b) or not len(a):
            continue
        p = float(wilcoxon(a, b).pvalue)
        summary[m] = dict(with_bypass=[float(a.mean()), float(a.std(ddof=1))],
                          no_bypass=[float(b.mean()), float(b.std(ddof=1))],
                          delta=float(a.mean() - b.mean()), wilcoxon_p=p, n=len(a))
        print("%-10s %8.4f +- %.4f %8.4f +- %.4f %10.4g"
              % (m, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1), p))

    json.dump(dict(design="direct cardiorespiratory path to the respiratory head",
                   protocol="10 folds x 3 seeds = 30 fold-values, Wilcoxon paired",
                   summary=summary, per_seed=res,
                   minutes=(time.time() - t0) / 60), open(OUT, "w"), indent=1)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
