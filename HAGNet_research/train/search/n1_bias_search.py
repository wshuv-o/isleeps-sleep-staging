"""
n1_bias_search.py -- lever 8a: per-class decision bias, tuned to macro-F1.

The earlier prior-correction lever only tried a single uniform exponent (p * pi^alpha)
and returned exactly zero. A free per-class log-bias vector b in R^5 is a strictly
richer family: it can raise N1 (F1 = 0.315, our worst class and 1/5 of macro-F1)
without disturbing N2/Wake.

b is fit by coordinate ascent on HELD-OUT VALIDATION SUBJECTS inside each training
fold, then applied unchanged to that fold's test subjects. Nothing is tuned on test.
Light CPU only, so it can run beside a GPU job.
"""
import os, sys, glob, json
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa

RES = os.path.join(HERE, "results")
FC = os.path.join(HERE, "data", "featseq_cache")
NC, EPS = 5, 1e-12


def subjects():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FC, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def transitions(seqs):
    A = np.ones((NC, NC)); pi = np.ones(NC)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return np.log(A + EPS), np.log(pi + EPS)


def viterbi(le, lA, lpi):
    T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        p[t] = bp[t + 1, p[t + 1]]
    return p


def decode(LP, b, lA, lpi):
    return viterbi(LP + b[None, :], lA, lpi)


def mf1(Y, LP, subs, b, lA, lpi):
    yt = np.concatenate([Y[s] for s in subs])
    yp = np.concatenate([decode(LP[s], b, lA, lpi) for s in subs])
    return f1_score(yt, yp, average="macro", zero_division=0)


def main():
    subs = subjects()
    Y = {s: np.load(os.path.join(FC, f"SN{s}.npz"))["y"].astype(int) for s in subs}
    d = np.load(os.path.join(RES, "ensemble7_v2_probs.npz"))
    LP = {}
    for k in d.files:
        s = int(k)
        if s in Y:
            p = d[k].astype(np.float64)[:len(Y[s])]
            LP[s] = np.log(p / p.sum(1, keepdims=True).clip(EPS) + EPS)
    folds = make_folds(subs, 10, seed=42); rng = np.random.RandomState(0)
    GRID = np.array([-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6])

    base_r, tuned_r, biases = [], [], []
    for k, (tr_s, te_s) in enumerate(folds):
        val_s = sorted(rng.choice(tr_s, size=min(24, len(tr_s)), replace=False).tolist())
        fit_s = [s for s in tr_s if s not in val_s]
        lA, lpi = transitions([Y[s] for s in fit_s])

        b = np.zeros(NC)
        best = mf1(Y, LP, val_s, b, lA, lpi)
        for _ in range(2):                                  # 2 coordinate-ascent passes
            for c in range(NC):
                cur = b[c]; cand = cur; cbest = best
                for g in GRID:
                    b[c] = g; v = mf1(Y, LP, val_s, b, lA, lpi)
                    if v > cbest:
                        cbest, cand = v, g
                b[c] = cand; best = cbest
        biases.append(b.copy())

        yt = np.concatenate([Y[s] for s in te_s])
        yp0 = np.concatenate([decode(LP[s], np.zeros(NC), lA, lpi) for s in te_s])
        yp1 = np.concatenate([decode(LP[s], b, lA, lpi) for s in te_s])

        def m(y, p):
            return dict(acc=float(accuracy_score(y, p)),
                        mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                        kappa=float(cohen_kappa_score(y, p)),
                        pcf=f1_score(y, p, average=None, labels=range(NC), zero_division=0).tolist())
        base_r.append(m(yt, yp0)); tuned_r.append(m(yt, yp1))
        print(f"fold {k}: b={np.round(b,2).tolist()} | base mF1={base_r[-1]['mf1']:.4f} "
              f"-> tuned {tuned_r[-1]['mf1']:.4f}  (N1 {base_r[-1]['pcf'][1]:.3f} -> {tuned_r[-1]['pcf'][1]:.3f})",
              flush=True)

    def summ(rows):
        return dict(acc=float(np.mean([r["acc"] for r in rows])),
                    mf1=float(np.mean([r["mf1"] for r in rows])),
                    kappa=float(np.mean([r["kappa"] for r in rows])),
                    pcf=np.mean([r["pcf"] for r in rows], axis=0).tolist())
    B, T = summ(base_r), summ(tuned_r)
    print("\n============ PER-CLASS BIAS (10-fold, tuned on val subjects) ============")
    for nm, r in [("baseline +HMM", B), ("per-class bias", T)]:
        print(f"{nm:16s} acc={r['acc']:.4f} mF1={r['mf1']:.4f} kappa={r['kappa']:.4f}  "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, r["pcf"])))
    print(f"\ndelta: acc {T['acc']-B['acc']:+.4f}  mF1 {T['mf1']-B['mf1']:+.4f}  "
          f"kappa {T['kappa']-B['kappa']:+.4f}  N1 {T['pcf'][1]-B['pcf'][1]:+.4f}")
    json.dump({"baseline": B, "tuned": T, "mean_bias": np.mean(biases, 0).tolist()},
              open(os.path.join(RES, "n1_bias.json"), "w"), indent=2)
    print("saved -> results/n1_bias.json")


if __name__ == "__main__":
    main()
