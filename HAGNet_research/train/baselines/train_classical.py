"""
train_classical.py — feature-based classical ML staging (different paradigm from the
deep nets; usually stronger on small data). 5-fold subject-independent CV.

Per epoch: hand-crafted features (features.py) -> per-subject z-norm -> temporal
context stacking (+-k epochs) -> {RandomForest, ExtraTrees, HistGB, XGBoost, LightGBM}.
Class imbalance handled via balanced class/sample weights.

  KMP_DUPLICATE_LIB_OK=TRUE d:/EEG-TransNet/testenv/python.exe train_classical.py
"""
import os
import sys
import json
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
from datasets import load_subject, make_folds, list_subjects, CLASS_NAMES, CHANNELS  # noqa
from features import extract_features  # noqa

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def add_context(F, k):
    if k <= 0:
        return F
    n, D = F.shape
    Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([Fp[i:i + n] for i in range(2 * k + 1)], axis=1)


def subject_features(sid, channels, k, cache):
    if sid in cache:
        return cache[sid]
    x, y = load_subject(sid, channels=channels, normalize=False)
    F, _ = extract_features(x, fs=100)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)     # per-subject z-norm
    F = add_context(F, k)
    cache[sid] = (F.astype(np.float32), y)
    return cache[sid]


def stack(sids, channels, k, cache):
    Fs, ys = [], []
    for s in sids:
        F, y = subject_features(s, channels, k, cache)
        Fs.append(F); ys.append(y)
    return np.concatenate(Fs), np.concatenate(ys)


def make_models():
    m = {
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               min_samples_leaf=2, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                           min_samples_leaf=2, n_jobs=-1, random_state=42),
        "HistGB": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                 class_weight="balanced", random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                     n_jobs=-1, random_state=42)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        m["LightGBM"] = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                       subsample=0.8, class_weight="balanced", n_jobs=-1,
                                       random_state=42, verbose=-1)
    except Exception:
        pass
    return m


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="C4:M1,C3:M2,O2:M1,O1:M2")
    ap.add_argument("--context", type=int, default=3, help="+-k neighbour epochs stacked")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    channels = [c.strip() for c in args.channels.split(",")]
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects()
    folds = make_folds(subs, n_splits=5, seed=args.seed)
    cache = {}

    # warm the cache / report feature dim
    F0, _ = subject_features(subs[0], channels, args.context, cache)
    print(f"channels={channels} | context=+-{args.context} | feat_dim={F0.shape[1]} | subjects={len(subs)}")

    per_model = {name: [] for name in make_models()}
    for k, (tr_s, te_s) in enumerate(folds):
        Xtr, ytr = stack(tr_s, channels, args.context, cache)
        Xte, yte = stack(te_s, channels, args.context, cache)
        sw = compute_sample_weight("balanced", ytr)
        print(f"\n=== Fold {k} | train {Xtr.shape} test {Xte.shape} ===")
        for name, model in make_models().items():
            if name == "XGBoost":
                model.fit(Xtr, ytr, sample_weight=sw)
            else:
                model.fit(Xtr, ytr)
            mt = metrics(yte, model.predict(Xte))
            per_model[name].append(mt)
            print(f"  {name:13s} acc={mt['acc']:.3f} mF1={mt['macro_f1']:.3f} kappa={mt['kappa']:.3f}")

    print("\n===== 5-fold subject-independent summary =====")
    print(f"{'model':14s} {'acc':>13s} {'macroF1':>13s} {'kappa':>13s}   " + " ".join(f"{c:>5s}" for c in CLASS_NAMES))
    out = {}
    for name, res in per_model.items():
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res])
        kap = np.array([r["kappa"] for r in res]); pcf = np.array([r["per_class_f1"] for r in res]).mean(0)
        print(f"{name:14s} {acc.mean():.3f}+-{acc.std():.3f} {mf1.mean():.3f}+-{mf1.std():.3f} "
              f"{kap.mean():.3f}+-{kap.std():.3f}   " + " ".join(f"{v:5.3f}" for v in pcf))
        out[name] = {"acc": float(acc.mean()), "acc_std": float(acc.std()),
                     "macro_f1": float(mf1.mean()), "kappa": float(kap.mean()),
                     "per_class_f1": pcf.tolist(), "folds": res}
    best = max(out.items(), key=lambda kv: kv[1]["acc"])
    print(f"\nBEST: {best[0]} acc={best[1]['acc']:.3f} mF1={best[1]['macro_f1']:.3f}")
    json.dump({"args": vars(args), "channels": channels, "models": out},
              open(os.path.join(RESULTS, "classical_all.json"), "w"), indent=2)
    print(f"saved -> {os.path.relpath(os.path.join(RESULTS, 'classical_all.json'))}")


if __name__ == "__main__":
    main()
