"""
k_compare.py -- fair, full-scale k=3 vs k=5 head-to-head.

Same 10 subject-independent folds, same full training data, same single GPU model
(XGBoost/CUDA) on both sides, same HMM decoding. Only the context width differs, so
the delta is attributable to context alone. Uses the pre-built feat_cache_v2 caches
(already per-subject standardised and context-stacked), so no feature re-extraction.
"""
import os, sys, glob, json, time
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa

CACHE = os.path.join(HERE, "data", "feat_cache_v2")
RES = os.path.join(HERE, "results")
NC, EPS = 5, 1e-12


def subjects():
    s = [int(os.path.basename(p).split("_")[0][2:]) for p in glob.glob(os.path.join(CACHE, "SN*_c3.npz"))]
    return sorted(set(x for x in s if x not in DUPLICATE_DROP))


def get(sid, k):
    d = np.load(os.path.join(CACHE, f"SN{sid}_c{k}.npz"))
    return d["F"].astype(np.float32), d["y"].astype(int)


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


def m(y, p):
    return dict(acc=float(accuracy_score(y, p)),
                mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)),
                pcf=f1_score(y, p, average=None, labels=range(NC), zero_division=0).tolist())


def run_k(k, subs, folds):
    from xgboost import XGBClassifier
    raw, hmm = [], []
    for i, (tr_s, te_s) in enumerate(folds):
        t0 = time.time()
        Xtr = np.concatenate([get(s, k)[0] for s in tr_s])
        ytr = np.concatenate([get(s, k)[1] for s in tr_s])
        clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", device="cuda",
                            n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        del Xtr, ytr
        lA, lpi = transitions([get(s, k)[1] for s in tr_s])
        yt, pr, ph = [], [], []
        for s in te_s:
            X, y = get(s, k); p = clf.predict_proba(X)
            yt.append(y); pr.append(p.argmax(1)); ph.append(viterbi(np.log(p + EPS), lA, lpi))
        yt = np.concatenate(yt)
        raw.append(m(yt, np.concatenate(pr))); hmm.append(m(yt, np.concatenate(ph)))
        print(f"  k={k} fold{i}: +HMM acc={hmm[-1]['acc']:.4f} mF1={hmm[-1]['mf1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    def summ(rows):
        return dict(acc=float(np.mean([r["acc"] for r in rows])),
                    acc_std=float(np.std([r["acc"] for r in rows])),
                    mf1=float(np.mean([r["mf1"] for r in rows])),
                    kappa=float(np.mean([r["kappa"] for r in rows])),
                    pcf=np.mean([r["pcf"] for r in rows], axis=0).tolist())
    return summ(raw), summ(hmm)


def main():
    subs = subjects(); folds = make_folds(subs, 10, seed=42)
    print(f"subjects={len(subs)} | 10-fold | XGBoost-CUDA only (identical both sides)")
    out = {}
    for k in (3, 5):
        r, h = run_k(k, subs, folds)
        out[f"k{k}"] = {"raw": r, "hmm": h}
        print(f"== k={k}: raw {r['acc']:.4f}/{r['mf1']:.4f} | +HMM {h['acc']:.4f}/{h['mf1']:.4f}/{h['kappa']:.4f}", flush=True)
    a, b = out["k3"]["hmm"], out["k5"]["hmm"]
    print("\n================= k=3 vs k=5 (full 10-fold, matched) =================")
    for nm, r in [("k=3", a), ("k=5", b)]:
        print(f"{nm}: acc={r['acc']:.4f}+-{r['acc_std']:.4f} mF1={r['mf1']:.4f} kappa={r['kappa']:.4f}  "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, r["pcf"])))
    print(f"\nDELTA (k5-k3): acc {b['acc']-a['acc']:+.4f}  mF1 {b['mf1']-a['mf1']:+.4f}  "
          f"kappa {b['kappa']-a['kappa']:+.4f}")
    json.dump(out, open(os.path.join(RES, "k_compare.json"), "w"), indent=2)
    print("saved -> results/k_compare.json")


if __name__ == "__main__":
    main()
