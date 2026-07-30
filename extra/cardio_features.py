"""
cardio_features.py -- the untested honest lever: add cardiorespiratory FEATURES to the
gradient-boosting ensemble that already scores 0.746 on EEG alone.

Rationale: the ensemble is information-limited, not subject-limited. It has only ever
seen EEG. The cardiorespiratory signals (SpO2, HR, airflow, effort) are orthogonal
information the disease (apnea) writes into this cohort. The MESA benchmark reports
+8.2 pp F1 for XGBoost when respiratory features are added -- exactly this setup.

Per epoch, from the 7 cardio channels @25 Hz (ECG Flow Thorax Abdomen Effort SpO2 Pulse),
we compute physiologically meaningful summaries (NOT the scored apnea label -> no
circularity):
    SpO2:  mean, min, std, desaturation depth (median - min)
    Pulse: mean, std                                   (heart rate + variability proxy)
    ECG:   std, line-length                            (cardiac activity)
    Flow:  std, line-length                            (airflow amplitude / variability)
    Thorax/Abdomen/Effort: std each                    (respiratory effort amplitude)
    thoraco-abdominal asynchrony: |corr(Thorax,Abdomen)| (paradoxical breathing in apnea)
Then +/-3 epoch context (same as the EEG features) and the ensemble is retrained.

  KMP_DUPLICATE_LIB_OK=TRUE python extra/cardio_features.py
"""
import os, sys, glob, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))
from datasets import make_folds, DUPLICATE_DROP, CLASS_NAMES  # noqa
MM = os.path.join(ROOT, "data", "multimodal")
FC = os.path.join(ROOT, "data", "featseq_cache")
FE = os.path.join(ROOT, "data", "mm_features")   # combined EEG+cardio feature cache
RES = os.path.join(ROOT, "results")
CARD = ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]
NC, EPS, CTX = 5, 1e-12, 3


def cardio_feats(card):
    """card [n,7,750] -> [n, F] physiological features. Zero for missing channels."""
    i = {c: k for k, c in enumerate(CARD)}
    def ll(x): return np.abs(np.diff(x, axis=1)).mean(1)          # line-length (amplitude*rate)
    sp = card[:, i["SpO2"]]
    f = [sp.mean(1), sp.min(1), sp.std(1), np.median(sp, 1) - sp.min(1),           # SpO2 + desat depth
         card[:, i["Pulse"]].mean(1), card[:, i["Pulse"]].std(1),                  # heart rate + var
         card[:, i["ECG"]].std(1), ll(card[:, i["ECG"]]),                          # cardiac
         card[:, i["Flow"]].std(1), ll(card[:, i["Flow"]]),                        # airflow
         card[:, i["Thorax"]].std(1), card[:, i["Abdomen"]].std(1), card[:, i["Effort"]].std(1)]  # effort
    # thoraco-abdominal asynchrony (paradoxical breathing marks obstructive events)
    th, ab = card[:, i["Thorax"]], card[:, i["Abdomen"]]
    asyn = np.array([np.corrcoef(th[k], ab[k])[0, 1] if th[k].std() > 0 and ab[k].std() > 0 else 0.0
                     for k in range(len(card))])
    f.append(np.nan_to_num(asyn))
    return np.nan_to_num(np.stack(f, 1)).astype(np.float32)        # [n, 14]


def context(F, k=CTX):
    Fp = np.pad(F, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([Fp[i:i + len(F)] for i in range(2 * k + 1)], axis=1)


def build():
    """returns per-subject (Xeeg, Xcard, y) with context, only subjects that have both.
    Reads the combined mm_features cache (Feeg 188 + Fcard 14 already computed)."""
    data = {}
    for f in sorted(glob.glob(os.path.join(FE, "SN*.npz")), key=lambda p: int(os.path.basename(p)[2:-4])):
        sid = int(os.path.basename(f)[2:-4])
        if sid in DUPLICATE_DROP:
            continue
        d = np.load(f)
        F = np.nan_to_num(d["Feeg"]).astype(np.float32); y = d["y"].astype(int)
        F = (F - F.mean(0)) / (F.std(0) + 1e-6)
        cf = np.nan_to_num(d["Fcard"]).astype(np.float32)[:len(y)]
        cf = (cf - cf.mean(0)) / (cf.std(0) + 1e-6)
        has_card = int(d["cvalid"].sum()) >= 5
        data[sid] = (context(F), context(cf), y, has_card)
    return data


def run(data, use_cardio, subs, folds):
    accs, mf1s, kaps, pcfs = [], [], [], []
    for tr, te in folds:
        def X(s):
            fe, fc, _, hc = data[s]
            return np.concatenate([fe, fc], 1) if (use_cardio and hc) else \
                   (np.concatenate([fe, np.zeros_like(fc)], 1) if use_cardio else fe)
        Xtr = np.concatenate([X(s) for s in tr]); ytr = np.concatenate([data[s][2] for s in tr])
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                             class_weight="balanced", random_state=42)
        clf.fit(Xtr, ytr)
        # HMM decode per fold (same as our protocol)
        A = np.ones((NC, NC)); pi = np.ones(NC)
        for s in tr:
            y = data[s][2]; pi[y[0]] += 1
            for a, b in zip(y[:-1], y[1:]): A[a, b] += 1
        lA = np.log(A / A.sum(1, keepdims=True) + EPS); lpi = np.log(pi / pi.sum() + EPS)
        def viterbi(le):
            T = le.shape[0]; dp = np.zeros((T, NC)); bp = np.zeros((T, NC), int); dp[0] = lpi + le[0]
            for t in range(1, T):
                sc = dp[t-1][:, None] + lA; bp[t] = sc.argmax(0); dp[t] = sc.max(0) + le[t]
            p = np.zeros(T, int); p[-1] = dp[-1].argmax()
            for t in range(T-2, -1, -1): p[t] = bp[t+1, p[t+1]]
            return p
        yt, yp = [], []
        for s in te:
            prob = clf.predict_proba(X(s)); yt.append(data[s][2]); yp.append(viterbi(np.log(prob + EPS)))
        yt = np.concatenate(yt); yp = np.concatenate(yp)
        accs.append(accuracy_score(yt, yp)); mf1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        kaps.append(cohen_kappa_score(yt, yp))
        pcfs.append(f1_score(yt, yp, average=None, labels=range(5), zero_division=0))
    return (np.mean(accs), np.std(accs), np.mean(mf1s), np.mean(kaps), np.mean(pcfs, 0))


def main():
    print("building EEG + cardiorespiratory feature sets...", flush=True)
    data = build()
    subs = sorted(data); full = sum(data[s][3] for s in subs)
    print(f"{len(subs)} subjects ({full} with cardio) | HistGB + HMM, 10-fold subject-independent\n", flush=True)
    folds = make_folds(subs, 10, seed=42)
    for tag, uc in [("EEG only            ", False), ("EEG + cardiorespiratory", True)]:
        a, asd, m, k, pc = run(data, uc, subs, folds)
        print(f"{tag}  acc={a:.4f}+-{asd:.4f}  mF1={m:.4f}  kappa={k:.4f}  "
              f"per-class={[round(float(x),3) for x in pc]}", flush=True)
    print("\n(EEG-only here uses HistGB alone; our full 4-booster ensemble is 0.7464.)")


if __name__ == "__main__":
    main()
