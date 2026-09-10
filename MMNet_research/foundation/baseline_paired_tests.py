"""Paired significance tests for MM-Net against every retrained baseline.

Why paired, and why it matters here
-----------------------------------
Reporting `mean +- sd` per model and eyeballing the overlap is the wrong test on
this benchmark. Fold-to-fold variation is large -- AttnSleep spans 0.627 to 0.755
in accuracy across folds -- and it is *shared*: a fold that is hard for one model
is hard for all of them, because the difficulty lives in the patients, not the
architecture. An unpaired comparison charges that shared variance against the
difference and hides real effects.

Every model here is evaluated on `mmnet_core.FOLDS`, a single fixed partition
built once at import by `make_folds(SUBS)` with its own default seed. The `seed`
argument in `run_10fold` and in the baseline runners controls weight
initialisation, not the split. Fold *i* is therefore the same held-out patients
for every model and every seed, and pairing fold-to-fold is valid.

That distinction is not academic. Pairing MM-Net's seed-42 folds against
AttnSleep's *seed-averaged* folds -- 10 pairs, two thirds of MM-Net's runs
discarded, and one side smoothed -- gives p = 0.18 on accuracy. The correct
pairing, fold-to-fold within each seed, gives p = 0.004 on the same data.

Multiplicity is handled with Holm within each metric, across the family of
baselines compared on that metric.

Usage:  python baseline_paired_tests.py [--out <json>]
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs", "final")

# Baselines whose runner stored one entry per fold at seed 42 only.
SINGLE_SEED = {
    "AttnSleep (3-fold quote, superseded)": None,   # placeholder, not used
    "SleepTransformer": "sleeptransformer.json",
    "TinySleepNet": "tinysleepnet.json",
    "U-Time": "utime_stroke.json",
}
METRICS = ("acc", "mf1", "kappa")
LABEL = {"acc": "accuracy", "mf1": "macro-F1", "kappa": "kappa"}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def paired(mm, base):
    mm, base = np.asarray(mm, float), np.asarray(base, float)
    d = mm - base
    t, p = stats.ttest_rel(mm, base)
    try:
        _, pw = stats.wilcoxon(mm, base)
    except ValueError:
        pw = float("nan")
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = stats.t.interval(0.95, len(d) - 1, d.mean(), se)
    return dict(n_pairs=len(d), mmnet=float(mm.mean()), baseline=float(base.mean()),
                diff=float(d.mean()), ci95=[float(lo), float(hi)],
                wins=int((d > 0).sum()), p_ttest=float(p), p_wilcoxon=float(pw),
                cohen_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RUNS, "baseline_paired_tests.json"))
    a = ap.parse_args()

    mm = json.load(open(os.path.join(RUNS, "final_model.json"), encoding="utf-8"))
    attn = json.load(open(os.path.join(RUNS, "attnsleep_seeded.json"),
                          encoding="utf-8"))["per_seed"]
    seed_pairs = [("final|42", "seed|42"), ("final|1", "seed|1"), ("final|7", "seed|7")]

    results = {m: {} for m in METRICS}
    for metric in METRICS:
        # AttnSleep: three seeds, paired within seed -> 30 pairs
        results[metric]["AttnSleep"] = paired(
            np.concatenate([mm[a_][metric] for a_, _ in seed_pairs]),
            np.concatenate([attn[b][metric] for _, b in seed_pairs]))
        # the rest ran at seed 42 only -> 10 pairs
        for name, fn in SINGLE_SEED.items():
            if fn is None:
                continue
            d = json.load(open(os.path.join(RUNS, fn), encoding="utf-8"))
            vals = [f[metric] for f in d["per_fold"] if metric in f]
            if len(vals) != len(mm["final|42"][metric]):
                continue
            results[metric][name] = paired(mm["final|42"][metric], vals)

        names = list(results[metric])
        adj = holm([results[metric][n]["p_ttest"] for n in names])
        for n, q in zip(names, adj):
            results[metric][n]["p_holm"] = float(q)
            results[metric][n]["significant_holm"] = bool(q < 0.05)

    for metric in METRICS:
        print("=" * 78)
        print("%s  (MM-Net %.4f)" % (LABEL[metric].upper(),
                                     results[metric]["AttnSleep"]["mmnet"]))
        print("%-20s %8s %8s %7s %16s %10s %s"
              % ("baseline", "theirs", "diff", "wins", "95% CI", "p (Holm)", ""))
        for n, r in sorted(results[metric].items(), key=lambda kv: -kv[1]["baseline"]):
            print("%-20s %8.4f %+8.4f %4d/%-3d [%+.4f,%+.4f] %10.5f %s"
                  % (n, r["baseline"], r["diff"], r["wins"], r["n_pairs"],
                     r["ci95"][0], r["ci95"][1], r["p_holm"],
                     "**" if r["significant_holm"] else "ns"))
        print()

    payload = {
        "protocol": "10-fold patient-independent on mmnet_core.FOLDS; AttnSleep "
                    "paired within each of seeds 42/1/7 (30 pairs), the rest at "
                    "seed 42 (10 pairs)",
        "test": "paired t-test and Wilcoxon signed-rank; Holm within metric",
        "results": results,
    }
    json.dump(payload, open(a.out, "w", encoding="utf-8"), indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
