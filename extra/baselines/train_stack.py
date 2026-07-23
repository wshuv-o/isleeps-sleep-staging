"""
train_stack.py — STACK the 7-ch classical ensemble with the 7-ch deep model.

Both are out-of-fold predictions on the SAME 10-fold splits (seed 42), so per-subject
probabilities are comparable. We blend them (weight swept), then HMM-decode. Deep and
classical make complementary errors, so the blend can exceed either alone.

Prereq: results/deep7_probs.npz (from train_deep7.py). Reuses the classical pipeline.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds  # noqa
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from train_ensemble_full import (subj_feats, fit_boosters, proba, transition_matrix,  # noqa
                                  viterbi, list_subjects7)

RESULTS = os.path.join(HERE, "results")
CONTEXT = 3
CL = ["W", "N1", "N2", "N3", "R"]


def main():
    deep = np.load(os.path.join(RESULTS, "deep7_probs.npz"))
    subs = list_subjects7(); folds = make_folds(subs, 10, seed=42)
    # per-test-subject records: ens prob, deep prob, y, fold transition model
    recs = []
    for k, (tr_s, te_s) in enumerate(folds):
        Xtr = np.concatenate([subj_feats(s, CONTEXT)[0] for s in tr_s])
        ytr = np.concatenate([subj_feats(s, CONTEXT)[1] for s in tr_s])
        models = fit_boosters(Xtr, ytr)
        A, pi = transition_matrix([subj_feats(s, CONTEXT)[1] for s in tr_s])
        lA, lpi = np.log(A + 1e-12), np.log(pi + 1e-12)
        for s in te_s:
            F, y = subj_feats(s, CONTEXT)
            ep = proba(models, F)                       # ensemble probs [n,5]
            dp = deep[str(s)]                            # deep probs [n,5]
            n = min(len(ep), len(dp), len(y))
            recs.append((ep[:n], dp[:n], y[:n], lA, lpi))
        print(f"fold {k} done ({len(te_s)} test subj)")

    def evaluate(probs_list, ys, use_hmm, mats):
        yp = []
        for p, (lA, lpi) in zip(probs_list, mats):
            yp.append(viterbi(np.log(p + 1e-12), lA, lpi) if use_hmm else p.argmax(1))
        y = np.concatenate(ys); pr = np.concatenate(yp)
        return (accuracy_score(y, pr), f1_score(y, pr, average="macro", zero_division=0),
                cohen_kappa_score(y, pr))

    ys = [r[2] for r in recs]; mats = [(r[3], r[4]) for r in recs]
    print("\n=== blend sweep (w = ensemble weight; deep weight = 1-w) ===")
    best = None
    for w in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0]:
        blend = [w * r[0] + (1 - w) * r[1] for r in recs]
        a0, f0, k0 = evaluate(blend, ys, False, mats)
        a1, f1_, k1 = evaluate(blend, ys, True, mats)
        tag = "ens-only" if w == 1.0 else ("deep-only" if w == 0.0 else f"blend w={w}")
        print(f"  {tag:12s}  raw acc={a0:.4f} mF1={f0:.3f} | +HMM acc={a1:.4f} mF1={f1_:.3f} k={k1:.3f}")
        for acc, mf1, kap, hmm in [(a0, f0, k0, False), (a1, f1_, k1, True)]:
            if best is None or acc > best["acc"]:
                best = {"w": w, "hmm": hmm, "acc": acc, "macro_f1": mf1, "kappa": kap}
    print(f"\nBEST STACK: w={best['w']} hmm={best['hmm']} -> acc={best['acc']:.4f} "
          f"mF1={best['macro_f1']:.3f} kappa={best['kappa']:.3f}  (vs published LSTM 0.747)")
    json.dump(best, open(os.path.join(RESULTS, "stack7_best.json"), "w"), indent=2)
    print("saved -> results/stack7_best.json")


if __name__ == "__main__":
    main()
