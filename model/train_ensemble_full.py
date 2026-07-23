"""
train_ensemble_full.py — best model on the STANDARD montage (4 EEG + 2 EOG + chin EMG).

Same ensemble (XGB+LGBM+HGB) + HMM Viterbi as train_ensemble.py, but on the 7-channel
npz (data/processed7) so the model can see eye-movements (REM) and chin EMG (atonia).
5-fold subject-independent CV, same patient-exclusive protocol.
"""
import os, sys, glob, json, argparse
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # repo root
for _p in ("utils", "processing", "model"):
    sys.path.insert(0, os.path.join(ROOT, _p))
sys.path.insert(0, HERE)
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa
from features import extract_features  # noqa
from features_v2 import extract_features_v2  # noqa
try:                                    # v3 connectivity: null result, kept in extra/
    from features_conn import extract_features_v3  # noqa
except ImportError:
    extract_features_v3 = None

PROC7 = os.path.join(ROOT, "data", "processed7")
RESULTS = os.path.join(ROOT, "results")
FEAT_CACHE = os.path.join(ROOT, "data", "feat_cache")   # persist features across runs
CACHE = {}
EXTRACT = extract_features       # swapped to extract_features_v2 by --v2
CACHE_DIR = FEAT_CACHE


def list_subjects7():
    sids = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(PROC7, "SN*.npz"))]
    return sorted(s for s in sids if s not in DUPLICATE_DROP)


def subj_feats(sid, k):
    if (sid, k) in CACHE:
        return CACHE[(sid, k)]
    cf = os.path.join(CACHE_DIR, f"SN{sid}_c{k}.npz")
    if os.path.exists(cf):                                 # load precomputed features (fast)
        d = np.load(cf); res = (d["F"], d["y"]); CACHE[(sid, k)] = res; return res
    d = np.load(os.path.join(PROC7, f"SN{sid}.npz"), allow_pickle=True)
    x = d["x"].astype(np.float32); y = d["y"].astype(np.int64)     # [n,7,3000]
    F, _ = EXTRACT(x, fs=100)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)
    if k > 0:
        Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
        F = np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)
    F = F.astype(np.float32)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cf, F=F, y=y)                                 # persist for future runs
    CACHE[(sid, k)] = (F, y)
    return CACHE[(sid, k)]


def fit_boosters(Xtr, ytr):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    sw = compute_sample_weight("balanced", ytr)
    m = []
    # XGBoost on GPU (device='cuda') accelerates the booster fitting; falls back to CPU if unavailable
    try:
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", device="cuda",
                            n_jobs=-1, random_state=42)
        xgb.fit(Xtr, ytr, sample_weight=sw)
    except Exception:
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=42)
        xgb.fit(Xtr, ytr, sample_weight=sw)
    m.append(xgb)
    lgb = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                         class_weight="balanced", n_jobs=-1, random_state=42, verbose=-1)
    lgb.fit(Xtr, ytr); m.append(lgb)
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         class_weight="balanced", random_state=42)
    hgb.fit(Xtr, ytr); m.append(hgb)
    if os.environ.get("USE_CATBOOST") == "1":
        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(iterations=700, depth=6, learning_rate=0.05,
                                    loss_function="MultiClass", auto_class_weights="Balanced",
                                    task_type="GPU" if os.environ.get("CB_GPU") == "1" else "CPU",
                                    verbose=False, random_seed=42)
            cb.fit(Xtr, ytr); m.append(cb)
        except Exception as e:
            print(f"  [catboost skipped: {type(e).__name__}]")
    return m


def proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def transition_matrix(seqs, n=5, eps=1.0):
    A = np.full((n, n), eps); pi = np.full(n, eps)
    for y in seqs:
        pi[y[0]] += 1
        for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
    A /= A.sum(1, keepdims=True); pi /= pi.sum(); return A, pi


def viterbi(log_e, log_A, log_pi):
    T, S = log_e.shape; dp = np.zeros((T, S)); bp = np.zeros((T, S), int); dp[0] = log_pi + log_e[0]
    for t in range(1, T):
        sc = dp[t - 1][:, None] + log_A; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + log_e[t]
    path = np.zeros(T, int); path[-1] = dp[-1].argmax()
    for t in range(T - 2, -1, -1): path[t] = bp[t + 1, path[t + 1]]
    return path


def metrics(y, p):
    return {"acc": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y, p, average=None, labels=list(range(5)), zero_division=0).tolist(),
            "kappa": float(cohen_kappa_score(y, p)),
            "confusion": confusion_matrix(y, p, labels=list(range(5))).tolist()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="ensemble7")
    ap.add_argument("--v2", action="store_true", help="use enhanced event-based features")
    ap.add_argument("--v3", action="store_true", help="v2 + inter-hemispheric connectivity")
    ap.add_argument("--seed", type=int, default=42); args = ap.parse_args()
    global EXTRACT, CACHE_DIR
    if args.v3:
        EXTRACT = extract_features_v3; CACHE_DIR = os.path.join(ROOT, "data", "feat_cache_v3")
        print("[v3] v2 event features + inter-hemispheric connectivity (coherence/PLV)")
    elif args.v2:
        EXTRACT = extract_features_v2; CACHE_DIR = os.path.join(ROOT, "data", "feat_cache_v2")
        print("[v2] enhanced event-based features (spindle/slow-wave/EOG/EMG)")
    os.makedirs(RESULTS, exist_ok=True)
    subs = list_subjects7(); folds = make_folds(subs, args.folds, seed=args.seed)
    F0, _ = subj_feats(subs[0], args.context)
    print(f"7-channel montage | subjects={len(subs)} | {args.folds}-fold | context=+-{args.context} | feat_dim={F0.shape[1]}")
    raw_res, hmm_res, all_probs = [], [], {}
    for k, (tr_s, te_s) in enumerate(folds):
        Xtr = np.concatenate([subj_feats(s, args.context)[0] for s in tr_s])
        ytr = np.concatenate([subj_feats(s, args.context)[1] for s in tr_s])
        models = fit_boosters(Xtr, ytr)
        A, pi = transition_matrix([subj_feats(s, args.context)[1] for s in tr_s])
        logA, logpi = np.log(A + 1e-12), np.log(pi + 1e-12)
        ya, pr_, ph_ = [], [], []
        for s in te_s:
            Fte, yte = subj_feats(s, args.context); pp = proba(models, Fte)
            all_probs[str(s)] = pp.astype(np.float32)
            ya.append(yte); pr_.append(pp.argmax(1)); ph_.append(viterbi(np.log(pp + 1e-12), logA, logpi))
        ya = np.concatenate(ya)
        mr = metrics(ya, np.concatenate(pr_)); mh = metrics(ya, np.concatenate(ph_))
        raw_res.append(mr); hmm_res.append(mh)
        print(f"Fold {k}: ensemble acc={mr['acc']:.3f} mF1={mr['macro_f1']:.3f} | "
              f"+HMM acc={mh['acc']:.3f} mF1={mh['macro_f1']:.3f} k={mh['kappa']:.3f}")

    def summ(name, res):
        acc = np.array([r["acc"] for r in res]); mf1 = np.array([r["macro_f1"] for r in res])
        kap = np.array([r["kappa"] for r in res]); pcf = np.array([r["per_class_f1"] for r in res]).mean(0)
        print(f"{name:18s} acc={acc.mean():.4f}+-{acc.std():.4f} mF1={mf1.mean():.4f} kappa={kap.mean():.4f}  "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, pcf)))
        return {"acc": float(acc.mean()), "acc_std": float(acc.std()), "macro_f1": float(mf1.mean()),
                "kappa": float(kap.mean()), "per_class_f1": pcf.tolist()}
    print("\n===== 7-channel (EEG+EOG+EMG) full cohort =====")
    out = {"ensemble": summ("ensemble", raw_res), "ensemble_hmm": summ("ensemble + HMM", hmm_res)}
    json.dump(out, open(os.path.join(RESULTS, f"{args.tag}_all.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(RESULTS, f"{args.tag}_probs.npz"), **all_probs)
    print(f"saved -> results/{args.tag}_all.json, results/{args.tag}_probs.npz")


if __name__ == "__main__":
    main()
