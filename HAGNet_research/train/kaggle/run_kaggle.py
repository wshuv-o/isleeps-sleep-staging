"""
run_kaggle.py -- does the ONE thing that hasn't been tried: add new information.

Twelve levers (post-hoc decoding, calibration, class bias, context width) all
returned nothing because they re-sliced the same 188 features. The per-class-bias
experiment showed N1 gains are paid for one-for-one elsewhere, i.e. the features
genuinely do not separate N1. So this script extracts 61 NEW features (nonlinear
complexity, wavelet, transients, slow-vs-rapid EOG) and asks whether they add
anything on top of the existing 188.

Three arms, identical folds / model / decoding, so the delta is attributable only
to the feature set:
    BASE  = 188 existing            (reproduces our published operating point)
    NEW   = 61 new only
    BOTH  = 249 concatenated        <- the question

Baseline to beat (same protocol, XGBoost-CUDA, 10-fold, +HMM):
    acc 0.7442 +- 0.0160 | macro-F1 0.6752 | kappa 0.6396 | N1 F1 0.321

Kaggle: add both datasets, enable GPU, then
    !python run_kaggle.py --stage features     # ~25-30 min (4 cores, pywt)
    !python run_kaggle.py --stage evaluate     # ~25 min (GPU)
"""
import os, sys, glob, json, time, argparse
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_new as FN

NC, EPS = 5, 1e-12
CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]
DUPLICATE_DROP = {28}          # SN28 is a bit-identical duplicate of SN15
CTX = 3

# --- paths: override with env vars if your Kaggle dataset slugs differ ---------
RAW = os.environ.get("RAW_DIR", "/kaggle/input/isleeps-processed7/processed7")
BASE188 = os.environ.get("BASE_DIR", "/kaggle/input/isleeps-bundle/featseq_cache")
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
NEWDIR = os.path.join(OUT, "feat_new")


def subjects(d, pat="SN*.npz"):
    s = [int(os.path.basename(p)[2:-4]) for p in glob.glob(os.path.join(d, pat))]
    return sorted(x for x in s if x not in DUPLICATE_DROP)


def make_folds(subs, n_splits=10, seed=42):
    """IDENTICAL to models/datasets.py so Kaggle results are comparable."""
    rng = np.random.RandomState(seed)
    sids = list(subs); rng.shuffle(sids)
    folds = [sids[i::n_splits] for i in range(n_splits)]
    return [(sorted(s for s in sids if s not in folds[k]), sorted(folds[k]))
            for k in range(n_splits)]


# ============================================================ stage 1: features
def _one(sid):
    dst = os.path.join(NEWDIR, f"SN{sid}.npz")
    if os.path.exists(dst):
        return sid, "cached"
    d = np.load(os.path.join(RAW, f"SN{sid}.npz"))
    X = d["x"].astype(np.float32)
    F = FN.subject_features(X)
    np.savez_compressed(dst, F=F, y=d["y"].astype(np.int64))
    return sid, F.shape


def stage_features(n_jobs):
    os.makedirs(NEWDIR, exist_ok=True)
    subs = subjects(RAW)
    print(f"extracting {FN.n_features()} new features for {len(subs)} subjects "
          f"(pywt={FN.HAVE_PYWT}, n_jobs={n_jobs})", flush=True)
    t0 = time.time()
    try:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=10)(delayed(_one)(s) for s in subs)
    except Exception as e:
        print("joblib unavailable, serial:", e)
        res = [_one(s) for s in subs]
    print(f"done in {(time.time()-t0)/60:.1f} min -> {NEWDIR}")


# ============================================================ stage 2: evaluate
def load_arm(sid, arm):
    """returns (per-epoch feature matrix, labels) for the requested feature set"""
    yb = None; parts = []
    if arm in ("base", "both"):
        d = np.load(os.path.join(BASE188, f"SN{sid}.npz"))
        parts.append(np.nan_to_num(d["F"]).astype(np.float32)); yb = d["y"].astype(int)
    if arm in ("new", "both"):
        d = np.load(os.path.join(NEWDIR, f"SN{sid}.npz"))
        Fn = np.nan_to_num(d["F"]).astype(np.float32)
        if yb is None:
            yb = d["y"].astype(int)
        parts.append(Fn[:len(yb)])
    F = np.concatenate([p[:len(yb)] for p in parts], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-6)              # per-subject standardisation
    Fp = np.pad(F, ((CTX, CTX), (0, 0)), mode="edge")
    X = np.concatenate([Fp[i:i + len(F)] for i in range(2 * CTX + 1)], 1)
    return X.astype(np.float32), yb


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


def met(y, p):
    return dict(acc=float(accuracy_score(y, p)),
                mf1=float(f1_score(y, p, average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(y, p)),
                pcf=f1_score(y, p, average=None, labels=range(NC), zero_division=0).tolist())


def run_arm(arm, subs, folds, gpu=True):
    from xgboost import XGBClassifier
    hmm = []
    for i, (tr, te) in enumerate(folds):
        t0 = time.time()
        Xtr = np.concatenate([load_arm(s, arm)[0] for s in tr])
        ytr = np.concatenate([load_arm(s, arm)[1] for s in tr])
        clf = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            device=("cuda" if gpu else "cpu"), n_jobs=-1, random_state=42)
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        del Xtr, ytr
        lA, lpi = transitions([load_arm(s, arm)[1] for s in tr])
        yt, ph = [], []
        for s in te:
            X, y = load_arm(s, arm); p = clf.predict_proba(X)
            yt.append(y); ph.append(viterbi(np.log(p + EPS), lA, lpi))
        hmm.append(met(np.concatenate(yt), np.concatenate(ph)))
        print(f"  [{arm}] fold{i}: acc={hmm[-1]['acc']:.4f} mF1={hmm[-1]['mf1']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return dict(acc=float(np.mean([r["acc"] for r in hmm])),
                acc_std=float(np.std([r["acc"] for r in hmm])),
                mf1=float(np.mean([r["mf1"] for r in hmm])),
                kappa=float(np.mean([r["kappa"] for r in hmm])),
                pcf=np.mean([r["pcf"] for r in hmm], axis=0).tolist())


def stage_evaluate(arms, gpu):
    subs = sorted(set(subjects(BASE188)) & set(subjects(NEWDIR)))
    folds = make_folds(subs, 10, seed=42)
    print(f"subjects={len(subs)} | 10-fold | arms={arms} | gpu={gpu}", flush=True)
    out = {}
    for a in arms:
        out[a] = run_arm(a, subs, folds, gpu)
        r = out[a]
        print(f"== {a}: acc={r['acc']:.4f}+-{r['acc_std']:.4f} mF1={r['mf1']:.4f} "
              f"kappa={r['kappa']:.4f}", flush=True)
    print("\n=================== RESULT ===================")
    print(f"{'arm':6s} {'acc':>8s} {'macroF1':>9s} {'kappa':>8s}   per-class F1")
    for a, r in out.items():
        print(f"{a:6s} {r['acc']:8.4f} {r['mf1']:9.4f} {r['kappa']:8.4f}   "
              + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, r["pcf"])))
    if "base" in out and "both" in out:
        b, t = out["base"], out["both"]
        print(f"\nDELTA (both-base): acc {t['acc']-b['acc']:+.4f}  "
              f"mF1 {t['mf1']-b['mf1']:+.4f}  kappa {t['kappa']-b['kappa']:+.4f}  "
              f"N1 {t['pcf'][1]-b['pcf'][1]:+.4f}")
        print("\nVERDICT: the new features help only if these deltas are positive")
        print("         AND larger than the +-0.016 fold standard deviation.")
    json.dump(out, open(os.path.join(OUT, "kaggle_result.json"), "w"), indent=2)
    print("saved -> kaggle_result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "evaluate", "all"], default="all")
    ap.add_argument("--arms", default="base,new,both")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="disable GPU")
    a = ap.parse_args()
    if a.stage in ("features", "all"):
        stage_features(a.n_jobs)
    if a.stage in ("evaluate", "all"):
        stage_evaluate([x for x in a.arms.split(",") if x], gpu=not a.cpu)
