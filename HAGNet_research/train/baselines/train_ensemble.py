"""
train_ensemble.py — soft-vote ensemble of gradient boosters + HMM (Viterbi) temporal
smoothing. Sleep stages are highly sequential, so decoding the per-epoch class
probabilities through a transition model (estimated from training hypnograms) is a
standard, reliable accuracy boost. 5-fold subject-independent CV.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_ensemble.py
"""
import os
import sys
import json
import argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import load_subject, make_folds, list_subjects, CLASS_NAMES  # noqa
from features import extract_features  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE = {}


def subj_feats(sid, channels, k):
    key = (sid, k)
    if key in CACHE:
        return CACHE[key]
    x, y = load_subject(sid, channels=channels, normalize=False)
    F, _ = extract_features(x, fs=100)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    if k > 0:
        Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
        F = np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)
    CACHE[key] = (F.astype(np.float32), y)
    return CACHE[key]


def fit_boosters(Xtr, ytr):
    sw = compute_sample_weight("balanced", ytr)
    models = []
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42)
    xgb.fit(Xtr, ytr, sample_weight=sw); models.append(xgb)
    lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                         class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    lgb.fit(Xtr, ytr); models.append(lgb)
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         class_weight="balanced", random_state=42)
    hgb.fit(Xtr, ytr); models.append(hgb)
    return models


def proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def transition_matrix(seqs, n=5, eps=1.0):
    A = np.full((n, n), eps)
    pi = np.full(n, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]):
            A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum()
    return A, pi


def viterbi(log_e, log_A, log_pi):
    T, S = log_e.shape
    dp = np.zeros((T, S)); bp = np.zeros((T, S), int)
    dp[0] = log_pi + log_e[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + log_A
        bp[t] = sc.argmax(0); dp[t] = sc.max(0) + log_e[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="C4:M1,C3:M2,O2:M1,O1:M2")
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",")]
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects()
    folds = make_folds(subs, n_splits=5, seed=args.seed)

    raw_res, hmm_res = [], []
    for k, (tr_s, te_s) in enumerate(folds):
        Xtr = np.concatenate([subj_feats(s, channels, args.context)[0] for s in tr_s])
        ytr = np.concatenate([subj_feats(s, channels, args.context)[1] for s in tr_s])
        models = fit_boosters(Xtr, ytr)
        A, pi = transition_matrix([subj_feats(s, channels, args.context)[1] for s in tr_s])
        logA, logpi = np.log(A + 1e-12), np.log(pi + 1e-12)

        y_all, p_raw, p_hmm = [], [], []
        for s in te_s:
            Fte, yte = subj_feats(s, channels, args.context)
            pr = proba(models, Fte)                       # [n,5] ensemble probs (temporal order)
            y_all.append(yte)
            p_raw.append(pr.argmax(1))
            p_hmm.append(viterbi(np.log(pr + 1e-12), logA, logpi))
        y_all = np.concatenate(y_all)
        mr = metrics(y_all, np.concatenate(p_raw))
        mh = metrics(y_all, np.concatenate(p_hmm))
        raw_res.append(mr); hmm_res.append(mh)
        print(f"Fold {k}: ensemble acc={mr['acc']:.3f} mF1={mr['macro_f1']:.3f} "
              f"| +HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")

    def summarize(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res])
        kap = np.array([r["kappa"] for r in res]); pcf = np.array([r["per_class_f1"] for r in res]).mean(0)
        print(f"{name:18s} acc={acc.mean():.3f}+-{acc.std():.3f} mF1={mf1.mean():.3f}+-{mf1.std():.3f} "
              f"kappa={kap.mean():.3f}   " + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, pcf)))
        return {"acc": float(acc.mean()), "macro_f1": float(mf1.mean()), "kappa": float(kap.mean()),
                "per_class_f1": pcf.tolist(), "folds": res}

    print("\n===== 5-fold subject-independent =====")
    out = {"ensemble": summarize("ensemble(XGB+LGB+HGB)", raw_res),
           "ensemble_hmm": summarize("ensemble + HMM", hmm_res)}
    json.dump({"args": vars(args), "models": out},
              open(os.path.join(RESULTS, "ensemble_all.json"), "w"), indent=2)
    print(f"saved -> {os.path.relpath(os.path.join(RESULTS, 'ensemble_all.json'))}")


if __name__ == "__main__":
    main()
