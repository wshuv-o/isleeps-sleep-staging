"""
benchmark_classical.py -- broaden the benchmark (Jahin DNA: "broadly superior, not a duel").
Runs a zoo of classical learners on the SAME 188 event/spectral features, SAME per-subject
standardization +/-3 context (1316-dim), SAME 10-fold subject-independent protocol, SAME HMM
Viterbi decoding as the HAG-Net classical prior -- so every row is apples-to-apples.
CPU-only (no GPU contention with a running deep job).
  d:/EEG-TransNet/testenv/python.exe benchmark_classical.py --folds 10
"""
import os, sys, glob, json, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "models"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa

FCACHE = os.path.join(HERE, "data", "featseq_cache")
RESULTS = os.path.join(HERE, "results")
CTX = 3
FEAT = {}


def list7():
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(FCACHE, "SN*.npz"))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def load(sid):
    d = np.load(os.path.join(FCACHE, f"SN{sid}.npz"))
    F = np.nan_to_num(d["F"]).astype(np.float32)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)                 # per-subject standardization
    Fp = np.pad(F, ((CTX, CTX), (0, 0)), mode="edge")        # +/-3 temporal context
    Fc = np.concatenate([Fp[i:i + len(F)] for i in range(2 * CTX + 1)], axis=1)
    FEAT[sid] = (Fc.astype(np.float32), d["y"].astype(np.int64))


def transition(seqs, n=5, eps=1.0):
    A = np.full((n, n), eps); pi = np.full(n, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return np.log(A + 1e-12), np.log(pi + 1e-12)


def viterbi(le, lA, lpi):
    T, S = le.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = lpi + le[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
    p = np.zeros(T, int); p[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): p[t] = bp[t + 1, p[t + 1]]
    return p


def met(y, p):
    return dict(acc=float(accuracy_score(y, p)), mf1=float(f1_score(y, p, average="macro", zero_division=0)),
               kappa=float(cohen_kappa_score(y, p)))


def make_models():
    m = {}
    m["Logistic Regression"] = lambda: LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", n_jobs=-1)
    m["Gaussian NB"] = lambda: GaussianNB()
    m["Decision Tree"] = lambda: DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42)
    m["Random Forest"] = lambda: RandomForestClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    m["Extra Trees"] = lambda: ExtraTreesClassifier(n_estimators=150, n_jobs=-1, class_weight="balanced_subsample", random_state=42)
    m["MLP (1-layer)"] = lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=120, early_stopping=True, random_state=42)
    try:
        from xgboost import XGBClassifier
        m["XGBoost (single)"] = lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                                      n_jobs=-1, random_state=42)
    except Exception: pass
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM (single)"] = lambda: LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05,
                                                        subsample=0.8, class_weight="balanced", n_jobs=-1,
                                                        random_state=42, verbose=-1)
    except Exception: pass
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
    subs = list7()
    for s in subs: load(s)
    folds = make_folds(subs, args.folds, seed=args.seed)
    models = make_models()
    print(f"Classical zoo | {len(subs)} subj | {args.folds}-fold | {len(models)} models | feat={FEAT[subs[0]][0].shape[1]}")
    out = {}
    for name, ctor in models.items():
        raw, hmm = [], []
        for k, (tr_s, te_s) in enumerate(folds):
            Xtr = np.concatenate([FEAT[s][0] for s in tr_s]); ytr = np.concatenate([FEAT[s][1] for s in tr_s])
            clf = ctor()
            try:
                if "XGBoost" in name:
                    clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
                else:
                    clf.fit(Xtr, ytr)
            except Exception as e:
                print(f"  [{name}] fit failed: {type(e).__name__}: {e}"); raw = None; break
            lA, lpi = transition([FEAT[s][1] for s in tr_s])
            ty, pr_, ph_ = [], [], []
            for s in te_s:
                Xte, yte = FEAT[s]
                pp = clf.predict_proba(Xte)
                ty.append(yte); pr_.append(pp.argmax(1)); ph_.append(viterbi(np.log(pp + 1e-12), lA, lpi))
            ty = np.concatenate(ty); raw.append(met(ty, np.concatenate(pr_))); hmm.append(met(ty, np.concatenate(ph_)))
        if raw is None: continue
        def summ(res):
            a = np.array([r["acc"] for r in res]); f = np.array([r["mf1"] for r in res]); kp = np.array([r["kappa"] for r in res])
            return dict(acc=float(a.mean()), acc_std=float(a.std()), mf1=float(f.mean()), kappa=float(kp.mean()))
        out[name] = {"raw": summ(raw), "hmm": summ(hmm)}
        h = out[name]["hmm"]
        print(f"  {name:22s} +HMM acc={h['acc']:.4f} mF1={h['mf1']:.4f} kappa={h['kappa']:.4f}")
    json.dump(out, open(os.path.join(RESULTS, "classical_zoo.json"), "w"), indent=2)
    print("saved -> results/classical_zoo.json")


if __name__ == "__main__":
    main()
