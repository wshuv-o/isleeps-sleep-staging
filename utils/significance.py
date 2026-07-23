"""
significance.py -- statistical significance of the interpretable ensemble vs the deep
models, subject-paired (the "is it noise?" killer reviewers ask for). For each subject we
compute per-subject accuracy and macro-F1 from each model's out-of-fold probs, then run a
paired Wilcoxon signed-rank test (and paired t-test) across the 99 subjects, with
Bonferroni correction for the number of comparisons. Also Cohen's d effect size.
"""
import os, sys, json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from sklearn.metrics import accuracy_score, f1_score
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)
from datasets import DUPLICATE_DROP  # noqa

PROC7 = os.path.join(ROOT, "data", "processed7")
RESULTS = os.path.join(ROOT, "results")
FILES = {"ensemble": "ensemble7_v2_probs.npz", "featseq": "featseq_probs.npz",
         "kags": "kags_probs.npz", "asym": "asym_probs.npz"}


def labels(sid):
    return np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)["y"].astype(np.int64)


def cohend(a, b):
    d = a - b; return float(d.mean() / (d.std(ddof=1) + 1e-12))


def main():
    P = {}
    for name, fn in FILES.items():
        fp = os.path.join(RESULTS, fn)
        if os.path.exists(fp):
            dd = np.load(fp); P[name] = {k: dd[k] for k in dd.files}
    subs = sorted(set.intersection(*[set(int(k) for k in P[n]) for n in P]) - DUPLICATE_DROP)
    Y = {s: labels(s) for s in subs}
    print(f"significance over {len(subs)} subjects | models: {list(P)}")

    # per-subject metrics
    acc = {n: np.array([accuracy_score(Y[s], P[n][str(s)].argmax(1)) for s in subs]) for n in P}
    mf1 = {n: np.array([f1_score(Y[s], P[n][str(s)].argmax(1), average="macro", zero_division=0) for s in subs]) for n in P}
    for n in P:
        print(f"  {n:9s} per-subject: acc={acc[n].mean():.3f}+-{acc[n].std():.3f}  mF1={mf1[n].mean():.3f}")

    ref = "ensemble"; others = [n for n in P if n != ref]
    m = len(others) * 2 if others else 1                       # Bonferroni over comparisons
    out = {"reference": ref, "n_subjects": len(subs), "bonferroni_m": m, "comparisons": {}}
    print(f"\nPaired tests: {ref} vs each deep model (Bonferroni m={m}, alpha_adj={0.05/m:.4f})")
    for n in others:
        row = {}
        for metric, D in (("acc", acc), ("mF1", mf1)):
            a, b = D[ref], D[n]
            try: w_p = float(wilcoxon(a, b, zero_method="wilcox").pvalue)
            except Exception: w_p = float("nan")
            t_p = float(ttest_rel(a, b).pvalue)
            row[metric] = dict(mean_diff=float((a - b).mean()), wilcoxon_p=w_p, ttest_p=t_p,
                               cohens_d=cohend(a, b), sig_bonferroni=bool(w_p < 0.05 / m))
            star = "***" if w_p < 0.05 / m else ("*" if w_p < 0.05 else "ns")
            print(f"  {ref} vs {n:8s} [{metric}]: dmean={(a-b).mean():+.3f}  Wilcoxon p={w_p:.2e}  d={cohend(a,b):+.2f}  {star}")
        out["comparisons"][n] = row
    json.dump(out, open(os.path.join(RESULTS, "significance.json"), "w"), indent=2)
    print("saved -> results/significance.json")


if __name__ == "__main__":
    main()
