"""
conformal.py -- uncertainty quantification for the interpretable stager via split-conformal
prediction (LAC / threshold conformal) on the ensemble's out-of-fold probabilities.
Gives distribution-free prediction sets with a coverage guarantee -- the "trustworthy AI"
signal reviewers expect (cf. conformal prediction in KACQ-DCNN). Also reports calibration
(ECE). Uses the same 10-fold subject-independent split; calibrates each test fold on the
other folds' OOF probs (all OOF, so conformity scores are valid).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa

PROC7 = os.path.join(ROOT, "data", "processed7")
RESULTS = os.path.join(ROOT, "results")


def labels(sid):
    return np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)["y"].astype(np.int64)


def ece(P, y, bins=15):
    conf = P.max(1); pred = P.argmax(1); correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            e += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    d = np.load(os.path.join(RESULTS, "ensemble7_v2_probs.npz"))
    subs = sorted(int(k) for k in d.files if int(k) not in DUPLICATE_DROP)
    P = {s: d[str(s)].astype(np.float64) for s in subs}
    Y = {s: labels(s) for s in subs}
    folds = make_folds(subs, 10, seed=42)

    allP = np.concatenate([P[s] for s in subs]); allY = np.concatenate([Y[s] for s in subs])
    print(f"conformal on ensemble OOF probs | {len(subs)} subj | {len(allY)} epochs")
    print(f"raw calibration: ECE={ece(allP, allY):.4f}  (lower is better)")

    out = {"ece": ece(allP, allY), "levels": {}}
    for alpha in (0.10, 0.05):
        cov, size, single, empty = [], [], [], []
        for tr, te in folds:                       # calibrate on other folds (all OOF)
            calP = np.concatenate([P[s] for s in tr]); calY = np.concatenate([Y[s] for s in tr])
            scores = 1.0 - calP[np.arange(len(calY)), calY]           # nonconformity
            n = len(scores); q = np.quantile(scores, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
            for s in te:
                sets = P[s] >= (1.0 - q)                              # prediction sets
                yy = Y[s]; incl = sets[np.arange(len(yy)), yy]
                cov.append(incl.mean()); size.append(sets.sum(1).mean())
                single.append((sets.sum(1) == 1).mean()); empty.append((sets.sum(1) == 0).mean())
        r = dict(target_coverage=1 - alpha, empirical_coverage=float(np.mean(cov)),
                 avg_set_size=float(np.mean(size)), singleton_rate=float(np.mean(single)),
                 empty_rate=float(np.mean(empty)))
        out["levels"][f"alpha_{alpha}"] = r
        print(f"  alpha={alpha}: target cov={1-alpha:.2f}  empirical cov={r['empirical_coverage']:.3f}  "
              f"avg |set|={r['avg_set_size']:.2f}  singletons={r['singleton_rate']:.2f}  empty={r['empty_rate']:.3f}")
    json.dump(out, open(os.path.join(RESULTS, "conformal.json"), "w"), indent=2)
    print("saved -> results/conformal.json")


if __name__ == "__main__":
    main()
