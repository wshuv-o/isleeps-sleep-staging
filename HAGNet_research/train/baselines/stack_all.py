"""
stack_all.py -- out-of-fold stacked generalization: combine the per-epoch probabilities
of diverse models (boosting ensemble, feature-sequence BiLSTM, AsymGraphSSM) with a
meta-learner, evaluated with a proper subject-independent meta-CV + HMM decoding.
This is the single most reliable way to push past a strong single model, because the
base models' errors decorrelate. Uses whatever results/*_probs.npz files exist.
"""
import os, sys, glob, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP  # noqa

PROC7 = os.path.join(HERE, "data", "processed7")
RESULTS = os.path.join(HERE, "results")

CANDIDATES = {                                  # name -> prob file
    "ensemble": "ensemble7_v2_probs.npz",
    "featseq": "featseq_probs.npz",
    "asym": "asym_probs.npz",
}


def labels(sid):
    return np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)["y"].astype(np.int64)


def metrics(y, p):
    return dict(acc=float(accuracy_score(y, p)),
               mf1=float(f1_score(y, p, average="macro", zero_division=0)),
               kappa=float(cohen_kappa_score(y, p)))


def transition(subs, Y):
    A = np.ones((5, 5)); pi = np.ones(5)
    for s in subs:
        y = Y[s]; pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + 1e-12), np.log(pi + 1e-12)


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


def main():
    # load whatever base models are available
    P = {}
    for name, fn in CANDIDATES.items():
        fp = os.path.join(RESULTS, fn)
        if os.path.exists(fp):
            d = np.load(fp); P[name] = {k: d[k] for k in d.files}
            print(f"loaded {name}: {len(P[name])} subjects")
    if len(P) < 2:
        print("need >=2 base models to stack; found:", list(P.keys())); return
    names = list(P.keys())
    subs = sorted(set.intersection(*[set(int(k) for k in P[n]) for n in names]) - DUPLICATE_DROP)
    Y = {s: labels(s) for s in subs}
    print(f"stacking {names} over {len(subs)} common subjects")

    # standalone (sanity): each base model argmax
    for n in names:
        y = np.concatenate([Y[s] for s in subs]); p = np.concatenate([P[n][str(s)].argmax(1) for s in subs])
        m = metrics(y, p); print(f"  base {n:9s} acc={m['acc']:.4f} mF1={m['mf1']:.4f} k={m['kappa']:.4f}")

    def feat(s):                                # concat per-epoch probs of all base models
        return np.concatenate([P[n][str(s)] for n in names], axis=1)

    folds = make_folds(subs, 10, seed=42)
    yt, ps_meta, ps_avg, ph_meta = [], [], [], []
    for tr, te in folds:
        Xtr = np.concatenate([feat(s) for s in tr]); ytr = np.concatenate([Y[s] for s in tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", multi_class="multinomial")
        clf.fit(Xtr, ytr)
        lA, lpi = transition(tr, Y)
        for s in te:
            pm = clf.predict_proba(feat(s))     # meta-learner stacked probs
            pa = np.mean([P[n][str(s)] for n in names], axis=0)   # simple average
            yt.append(Y[s]); ps_meta.append(pm.argmax(1)); ps_avg.append(pa.argmax(1))
            ph_meta.append(viterbi(np.log(pm + 1e-12), lA, lpi))
    y = np.concatenate(yt)
    out = {"stack_meta": metrics(y, np.concatenate(ps_meta)),
           "stack_meta_hmm": metrics(y, np.concatenate(ph_meta)),
           "stack_avg": metrics(y, np.concatenate(ps_avg))}
    print("\n===== STACKED =====")
    for k, m in out.items():
        print(f"  {k:16s} acc={m['acc']:.4f} mF1={m['mf1']:.4f} kappa={m['kappa']:.4f}")
    print("  (published LSTM: 0.747 / 0.677 / 0.640  |  best single = ensemble+HMM 0.746 / 0.675 / 0.642)")
    json.dump({"models": names, **out}, open(os.path.join(RESULTS, "stack_all.json"), "w"), indent=2)
    print("saved -> results/stack_all.json")


if __name__ == "__main__":
    main()
