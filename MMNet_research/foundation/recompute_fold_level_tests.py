"""Recompute every paired test on fold-means instead of fold-by-seed values.

The paper's p-values are Wilcoxon tests over thirty values, ten folds by three
seeds. The three seeds within a fold train and test on the same patients, so they
are not independent observations, and the paper's own seed analysis shows seed
variance is four to thirty times smaller than fold variance. Treating them as
thirty samples triples the effective n and shrinks every p-value.

This averages over seeds first and tests the ten fold-means, which is the
defensible unit: one value per held-out patient group. Effects that survive here
survive; the marginal ones will not, and the paper should say which.

Both versions are reported side by side so the change is auditable.

  KMP_DUPLICATE_LIB_OK=TRUE python recompute_fold_level_tests.py
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "MMNet_research", "results", "revision", "runs")
FINAL = os.path.join(RUNS, "final")
OUT = os.path.join(FINAL, "fold_level_tests.json")
SEEDS = [42, 1, 7]


def holm(pvals):
    """Holm-Bonferroni, returning adjusted p in the original order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def both_tests(a_fold, b_fold, a_all, b_all):
    """Return (p on 10 fold-means, p on 30 fold-seed values)."""
    p10 = float(wilcoxon(a_fold, b_fold).pvalue) if not np.allclose(a_fold, b_fold) else 1.0
    p30 = float(wilcoxon(a_all, b_all).pvalue) if not np.allclose(a_all, b_all) else 1.0
    return p10, p30


# ---------------------------------------------------------------- baseline
base = json.load(open(os.path.join(FINAL, "final_model.json")))
BASE = {m: np.array([base["final|%d" % s][m] for s in SEEDS]) for m in
        ("acc", "mf1", "kappa", "auc", "ap")}          # [3 seeds, 10 folds]

report = {}

# ---------------------------------------------------------------- ablations
shards = {}
for f in sorted(glob.glob(os.path.join(FINAL, "ablation_shards", "*.json"))):
    for k, v in json.load(open(f)).items():
        cond, seed = k.split("|")
        shards.setdefault(cond, {})[int(seed)] = v

names, p10_acc, p30_acc, p10_auc, p30_auc, rows = [], [], [], [], [], []
for cond in sorted(shards):
    if len(shards[cond]) != 3:
        print("skip %s, %d seeds" % (cond, len(shards[cond])))
        continue
    A = {m: np.array([shards[cond][s][m] for s in SEEDS]) for m in ("acc", "auc")}
    r = dict(condition=cond)
    for metric, lo, hi in (("acc", p10_acc, p30_acc), ("auc", p10_auc, p30_auc)):
        a_fold, b_fold = A[metric].mean(0), BASE[metric].mean(0)
        p10, p30 = both_tests(a_fold, b_fold, A[metric].ravel(), BASE[metric].ravel())
        lo.append(p10); hi.append(p30)
        r["d_" + metric] = float(a_fold.mean() - b_fold.mean())
        r["p10_" + metric] = p10
        r["p30_" + metric] = p30
    names.append(cond); rows.append(r)

for tag, ps, key in (("acc", p10_acc, "holm10_acc"), ("acc30", p30_acc, "holm30_acc"),
                     ("auc", p10_auc, "holm10_auc"), ("auc30", p30_auc, "holm30_auc")):
    for r, adj in zip(rows, holm(np.array(ps))):
        r[key] = float(adj)

report["modality_ablation"] = rows
print("MODALITY ABLATION  (Holm within metric)")
print("%-14s %8s %10s %10s   %8s %10s %10s"
      % ("removed", "d_acc", "holm n=10", "holm n=30", "d_auc", "holm n=10", "holm n=30"))
for r in rows:
    print("%-14s %8.4f %10.4g %10.4g   %8.4f %10.4g %10.4g"
          % (r["condition"], r["d_acc"], r["holm10_acc"], r["holm30_acc"],
             r["d_auc"], r["holm10_auc"], r["holm30_auc"]))

# ---------------------------------------------------------------- bypass
bp = os.path.join(FINAL, "bypass_ablation.json")
if os.path.exists(bp):
    d = json.load(open(bp))["per_seed"]
    W = {m: np.array([d["with_bypass"]["seed|%d" % s][m] for s in SEEDS])
         for m in ("acc", "kappa", "auc", "ap")}
    N = {m: np.array([d["no_bypass"]["seed|%d" % s][m] for s in SEEDS])
         for m in ("acc", "kappa", "auc", "ap")}
    print("\nDIRECT CARDIORESPIRATORY PATH")
    print("%-8s %9s %9s %11s %11s" % ("metric", "with", "without", "p n=10", "p n=30"))
    bypass = {}
    for m in ("auc", "ap", "acc", "kappa"):
        p10, p30 = both_tests(W[m].mean(0), N[m].mean(0), W[m].ravel(), N[m].ravel())
        bypass[m] = dict(with_bypass=float(W[m].mean()), no_bypass=float(N[m].mean()),
                         p10=p10, p30=p30)
        print("%-8s %9.4f %9.4f %11.4g %11.4g"
              % (m, W[m].mean(), N[m].mean(), p10, p30))
    report["bypass"] = bypass



# ---------------------------------------------------------------- N1 gain
# neural-only against neural+cardio on the N1 column of the per-class F1
NO = os.path.join(FINAL, "neural_only_pcf.json")
if os.path.exists(NO):
    d = json.load(open(NO))
    # pcf is [10 folds][5 stages] per seed
    n_only = np.array([[f[1] for f in d["neural_only|%d" % s]["pcf"]] for s in SEEDS])
    full = np.array([[f[1] for f in base["final|%d" % s]["pcf"]] for s in SEEDS])
    p10, p30 = both_tests(full.mean(0), n_only.mean(0), full.ravel(), n_only.ravel())
    print("\nN1 PER-CLASS F1, cardiorespiratory stream added")
    print("  neural only %.4f   full %.4f   gain %+.4f"
          % (n_only.mean(), full.mean(), full.mean() - n_only.mean()))
    print("  p n=10 %.4g   p n=30 %.4g" % (p10, p30))
    report["n1_gain"] = dict(neural_only=float(n_only.mean()), full=float(full.mean()),
                             gain=float(full.mean() - n_only.mean()), p10=p10, p30=p30)

json.dump(report, open(OUT, "w"), indent=1)
print("\nrewrote", OUT)
sys.exit(0)
