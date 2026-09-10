"""Significance tests for the permutation-importance table.

The table reported a mean drop and a standard deviation per channel and left the
reader to judge. Two things make that inadequate. The published `sd` is a
population standard deviation (`np.std`, ddof=0), which understates the spread on
ten folds; and a spread is not a test -- what a referee wants is whether the drop
is distinguishable from zero once multiplicity across eight channels is paid for.

The stored `per_fold` rows make a proper test straightforward. Each fold records
its own baseline AUC and the AUC after shuffling each channel, so the drop is a
*within-fold* quantity and the correct test is one-sample on the per-fold drops
(equivalently, paired between baseline and shuffled). Fold difficulty cancels,
which is the whole point -- between-fold variance on this cohort is larger than
every effect being measured.

Why permutation rather than ablation carries the attribution claim
-----------------------------------------------------------------
The two experiments answer different questions and the paper should say so.
Ablation retrains the model without a channel, so the network compensates by
re-learning the signal from correlated channels; what it measures is
*irreplaceability*. Permutation shuffles a channel with the trained model held
fixed; what it measures is what the model actually uses. A channel can be
heavily used and still be replaceable, which is exactly what SpO2 looks like
here: removing it and retraining costs 0.002 AUC, but shuffling it under the
fitted model costs 0.016.

Usage:  python permutation_significance.py [--out <json>]
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")
SRC = os.path.join(RUNS, "permutation_importance_final.json")


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "permutation_significance.json"))
    ap.add_argument("--metric", default="auc", choices=("auc", "acc"))
    a = ap.parse_args()

    src = json.load(open(SRC, encoding="utf-8"))
    rows = src["per_fold"]
    base_key = "base_" + a.metric
    channels = [k for k in rows[0] if k not in ("fold", "base_acc", "base_auc")]

    out = {}
    for ch in channels:
        drops = np.array([r[base_key] - r[ch][a.metric] for r in rows], float)
        drops = drops[~np.isnan(drops)]
        t, p = stats.ttest_1samp(drops, 0.0)
        try:
            _, pw = stats.wilcoxon(drops)
        except ValueError:
            pw = float("nan")
        se = drops.std(ddof=1) / np.sqrt(len(drops))
        lo, hi = stats.t.interval(0.95, len(drops) - 1, drops.mean(), se)
        out[ch] = dict(n_folds=len(drops), drop=float(drops.mean()),
                       sd=float(drops.std(ddof=1)), ci95=[float(lo), float(hi)],
                       folds_positive=int((drops > 0).sum()),
                       p_ttest=float(p), p_wilcoxon=float(pw),
                       cohen_dz=float(drops.mean() / drops.std(ddof=1)))

    names = list(out)
    for n, q in zip(names, holm([out[n]["p_ttest"] for n in names])):
        out[n]["p_holm"] = float(q)
        out[n]["significant_holm"] = bool(q < 0.05)

    print("Permutation importance on respiratory %s, one-sample on per-fold drops"
          % a.metric.upper())
    print("Holm across %d channels. Baseline %s = %.4f\n"
          % (len(names), a.metric.upper(), src["baseline"][a.metric]))
    print("%-12s %8s %8s %18s %7s %10s %s"
          % ("channel", "drop", "sd", "95% CI", "folds", "p (Holm)", ""))
    for ch, r in sorted(out.items(), key=lambda kv: -kv[1]["drop"]):
        print("%-12s %8.4f %8.4f [%+.4f,%+.4f] %4d/%-2d %10.5f %s"
              % (ch, r["drop"], r["sd"], r["ci95"][0], r["ci95"][1],
                 r["folds_positive"], r["n_folds"], r["p_holm"],
                 "**" if r["significant_holm"] else "ns"))

    sig = [c for c in out if out[c]["significant_holm"]]
    print("\nsurvive Holm: %s" % (", ".join(sig) or "none"))

    json.dump({"metric": a.metric,
               "baseline": src["baseline"][a.metric],
               "test": "one-sample t-test on per-fold drops; Holm across channels",
               "results": out}, open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
